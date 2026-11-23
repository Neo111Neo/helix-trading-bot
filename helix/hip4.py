"""HIP-4 recorder: paper-only comparison against Hyperliquid's own outcome markets. It never trades.

HIP-4 markets are Yes/No contracts that pay $1 if the price is above a strike at expiry ("BTC above 85,429 at 19:00").
Their price IS a probability, so they are the market's own forecast of the hourly test's question. This records,
for every BTC/ETH/SOL/HYPE price-binary market, a snapshot when its time to expiry first falls inside a checkpoint
window (about 12 and about 55 minutes before expiry):

  market    best bid / ask / mid of the Yes book (the bar to beat: real money is behind it)
  model     zero-drift lognormal; model_emp the empirical fat-tailed baseline (the hourly test's formulas)
  jev       the hourly test's original yes/no wording; jev_v3 (default) a Jev Lab question set
  claude    one batched Claude call per snapshot round (optional)

Fairness rules (from an adversarial review of the first version):
  - Nobody but the market sees the market price.
  - Each market's Yes book and the perp book it is compared with are read back to back, before any Jev/Claude call;
    rows whose two exchange timestamps differ by more than MAX_BOOK_GAP_S are not scored.
  - TWAP settlement (60-300 s averages) narrows the true spread of outcomes; the baselines use the matching effective
    time, and Pyth-settled markets are compared with the perp price adjusted by the live oracle basis.
  - The PRIMARY comparison, fixed before any outcome was seen, is near-the-money rows (|ln(K/S)| <= 0.75 x the
    expected move to expiry) on markets that settle on Hyperliquid's own perp price with at most a 90 s average. Every
    question set was designed for strikes near the price; everything else is reported separately.
  - jev_v2 is not asked: its probability scale only fits the hourly test's three fixed strikes (spot, +-0.5 sigma).

Outcomes come from the market's own settlement (settledOutcome). Scoring: paired log loss against the market mid
(6-hour block bootstrap) and a paper trade check at the real bid/ask. Saved to data/hip4.json."""

from __future__ import annotations

import calendar
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

from .hourly import (build_questions, empirical_above, har_params, har_sigma, logloss, lognormal_above, past_z,
                     realized_sigma, seasonal_factors, sigma_per_hour)
from .hyperliquid import URLS, HLRest

CHECKPOINTS = (("12m", 9.0, 12.5), ("55m", 50.0, 56.0))   # (name, min, max) minutes to expiry. The 15-minute markets
                                                           # list 10-25 s after the previous expiry (~14.7 min left,
                                                           # HYPE sometimes a minute later) and quote 5-10c wide at once.
META_EVERY_S = 60
MAX_SPREAD = 0.10            # wider (or one-sided) books are recorded but not scored
MAX_BOOK_GAP_S = 2.0         # the Yes book and the perp book must be within 2 s of each other
NEAR_MONEY = 0.75            # primary comparison band, in units of the expected move to expiry
FEE = 0.0005                 # per $1 contract; HIP-4 charges ~0.035% on close/settle, so this is generous
GIVE_UP_S = 7 * 86400        # builders must settle within a week or get slashed
SETTLE_CALLS_PER_STEP = 4    # settledOutcome weighs 20; unsettled ids back off exponentially
MAX_SNAPS = 30000
SNAP_VERSION = 2             # snapshots from the first (reviewed and fixed) version are dropped on load


def parse_desc(desc: str) -> dict:
    return dict(p.split(":", 1) for p in (desc or "").split("|") if ":" in p)


def parse_market(o: dict, coins) -> dict | None:
    """A price-binary outcome on one of `coins` -> {id, coin, venue, strike, expiry, rule, source, twap_s}."""
    d = parse_desc(o.get("description", ""))
    if d.get("class") == "priceBinary":                               # Hyperliquid's own recurring markets
        coin, t, k, rule, venue = d.get("underlying"), d.get("expiry"), d.get("targetPrice"), ">=", "HL"
        source, twap = "HyperCore mark (interpolated)", 0
    elif o.get("name") == "template:binaryPrice":                     # builder markets (out, skew, txyz, ...)
        coin, t, k, rule, venue = d.get("perp"), d.get("time"), d.get("threshold"), ">", o.get("venue", "?")
        source, twap = d.get("priceDescription", ""), int(d.get("seconds") or 0)
    else:
        return None
    if coin not in coins or not t or not k:
        return None
    try:
        expiry = calendar.timegm(time.strptime(t, "%Y%m%d-%H%M"))     # UTC; never local time
        strike = float(k)
    except ValueError:
        return None
    return {"id": int(o["outcome"]), "coin": coin, "venue": venue, "strike": strike, "expiry": expiry, "rule": rule,
            "source": source, "twap_s": twap}


def primary(m: dict) -> bool:
    """Settles on Hyperliquid's own perp price (not Pyth) with at most a 90 s average."""
    return "pyth" not in (m.get("source") or "").lower() and int(m.get("twap_s") or 0) <= 90


def book_top(levels) -> dict:
    bids, asks = (levels[0] if levels else []), (levels[1] if levels and len(levels) > 1 else [])
    bid = float(bids[0]["px"]) if bids else None
    ask = float(asks[0]["px"]) if asks else None
    out = {"bid": bid, "ask": ask, "bid_sz": float(bids[0]["sz"]) if bids else 0.0,
           "ask_sz": float(asks[0]["sz"]) if asks else 0.0}
    out["mid"] = (bid + ask) / 2 if bid is not None and ask is not None else None
    out["usable"] = out["mid"] is not None and round(ask - bid, 9) <= MAX_SPREAD
    return out


def strike_state(coin: str, c5, c1, spot: float, now: float, expiry: float, strike: float, ctx: dict,
                 twap_s: int = 0) -> dict:
    """The hourly test's state and baselines (HourlyTest._coin_state), generalised to any expiry and ONE strike,
    which sits in the `mid` slot so the hourly question sets apply unchanged. Identical to the hourly test for a
    58-minute window; for other windows the empirical baseline uses past moves of the matching length."""
    closes = [c.close for c in c5]
    sig_h = sigma_per_hour(closes[-289:])
    tte_min = max(1.0, (expiry - now) / 60)
    eff_min = max(1.0, tte_min - 2 * twap_s / 180)                 # an average over w s ending at expiry ~ tau - 2w/3
    t_h = eff_min / 60
    factors = seasonal_factors(c5)
    f_now = factors[int((now + expiry) / 2 // 3600) % 24]           # the UTC hour in the middle of the window
    sigma, sigma_s = sig_h * math.sqrt(t_h), sig_h * f_now * math.sqrt(t_h)
    step = max(1, round(eff_min / 5))                                # past moves over the nearest 5-minute multiple,
    zs = [z / math.sqrt(step / 12)                                   # rescaled to that window's own width so the
          for z in past_z(c5, factors, step=step, every=step != 12)]  # kernel bandwidth means the same at any length
    model = lognormal_above(spot, strike, sigma)
    if len(zs) >= 100 and sigma_s > 0:
        model_emp = empirical_above(zs, math.log(strike / spot) / sigma_s)
    else:
        model_emp = lognormal_above(spot, strike, sigma_s)

    def ret_bps(minutes: int) -> float:
        bars = minutes // 5
        return (spot / closes[-(bars + 1)] - 1) * 1e4 if len(closes) > bars else 0.0

    r15 = (spot / c1[-16].close - 1) * 1e4 if len(c1) >= 16 else 0.0
    last_h = closes[-13:]
    rv1 = pstdev([math.log(b / a) for a, b in zip(last_h, last_h[1:])]) * math.sqrt(12) if len(last_h) == 13 else 0.0
    # the recent-volatility baseline (Jev Lab: beat model_emp on held-out 15-minute and 1-hour strike data)
    rv4 = realized_sigma(closes, 48)
    w_har, seasonal = har_params(eff_min)
    hs = har_sigma(rv1, rv4, sig_h, w_har) * (f_now if seasonal else 1.0) * math.sqrt(t_h)
    if len(zs) >= 100 and hs > 0:
        model_har = empirical_above(zs, math.log(strike / spot) / hs)
    else:
        model_har = lognormal_above(spot, strike, hs)
    day = c5[-288:]
    hi, lo = max(c.high for c in day), min(c.low for c in day)
    decimals = max(0, 6 - len(str(int(spot))))
    state = {
        "price": round(spot, decimals), "minutes_to_expiry": round(tte_min, 1),
        "return_15m_bps": round(r15, 1), "return_1h_bps": round(ret_bps(60), 1),
        "return_4h_bps": round(ret_bps(240), 1), "return_24h_bps": round(ret_bps(1440), 1),
        "expected_move_to_expiry_bps": round(sigma * 1e4, 1), "vol_per_hour_24h_bps": round(sig_h * 1e4, 1),
        "realized_vol_last_1h_bps": round(rv1 * 1e4, 1), "hour_of_day_vol_factor": round(f_now, 2),
        "funding_hr_bps": round((ctx.get("funding") or 0.0) * 1e4, 3),
        "premium_bps": round((ctx.get("premium") or 0.0) * 1e4, 2),
        "position_in_24h_range": round((spot - lo) / (hi - lo), 3) if hi > lo else 0.5,
        "strike_mid": strike, "strike_mid_distance_bps": round((strike / spot - 1) * 1e4, 1),
    }
    har = {"realized_vol_last_4h_bps": round(rv4 * 1e4, 1), "har_p_above_mid": round(model_har, 3),
           "strike_mid_distance_sigmas": round(math.log(strike / spot) / hs, 2) if hs > 0 else 0.0}
    return {"state": state, "model": model, "model_emp": model_emp, "model_har": model_har, "har": har,
            "n_z": len(zs), "sigma_h": sig_h, "moneyness": math.log(strike / spot) / sigma if sigma > 0 else 0.0}


CLAUDE_PROMPT = """You forecast the probability that crypto perpetual futures prices end ABOVE a strike price at a
given expiry, for a paper-trading research test. Nothing is traded. Forecasts are scored with log loss against what
actually happens, alongside an empirical volatility model.

Each item has an id, the coin, the expiry time (UTC) and the coin's state: price, minutes to expiry, returns over 15
minutes to 24 hours, the expected 1-sigma move to expiry (`expected_move_to_expiry_bps`), funding, premium, where the
price sits in its 24-hour range, and the strike (`strike_mid`) with its distance from the price in bps
(`strike_mid_distance_bps`). Return p_above for every id. Be calibrated; short-horizon moves are mostly noise."""


def _claude_schema():
    from pydantic import BaseModel

    class Item(BaseModel):
        id: str
        p_above: float

    class Batch(BaseModel):
        forecasts: list[Item]

    return Batch


def score(rows: list[dict], who: str, edge: float, n_boot: int = 2000) -> dict:
    """Log loss and Brier for one forecaster; for everyone but the market, the paired gain vs the market mid with a
    6-hour block bootstrap, and the paper trade check at the real bid/ask."""
    from .lab import block_ci
    r = [x for x in rows if x.get(who) is not None]
    if not r:
        return {"n": 0}
    out = {"n": len(r), "markets": len({x["id"] for x in r}),
           "logloss": round(fmean(logloss(x[who], x["y"]) for x in r), 4),
           "brier": round(fmean((x[who] - x["y"]) ** 2 for x in r), 4)}
    if who == "market":
        return out
    diffs = [logloss(x["market"], x["y"]) - logloss(x[who], x["y"]) for x in r]       # + = better than the market
    out["gain_vs_market"] = round(fmean(diffs), 4)
    lo, hi = block_ci(diffs, [x["expiry"] // (6 * 3600) for x in r], n_boot=n_boot)
    out["gain_ci"] = None if math.isnan(lo) else [round(lo, 4), round(hi, 4)]
    pnl = []
    for x in r:
        p, bid, ask, y = x[who], x["bid"], x["ask"], x["y"]
        if p >= ask + edge:
            pnl.append(y - ask - FEE)                    # buy Yes at the ask
        elif p <= bid - edge:
            pnl.append(bid - y - FEE)                    # sell Yes at the bid (= buy No at 1 - bid)
    out["trades"] = len(pnl)
    out["cents_per_trade"] = round(fmean(pnl) * 100, 2) if pnl else None
    out["win_rate"] = round(sum(1 for v in pnl if v > 0) / len(pnl), 3) if pnl else None
    return out


class HIP4Recorder:
    def __init__(self, cfg: dict, data_dir: Path, rest: HLRest | None = None, jev_client=None, variants: dict | None = None,
                 claude_client=None, claude_model: str | None = None, clock=time.time):
        hc = cfg.get("hip4", {})
        self.enabled = bool(hc.get("enabled", True))
        self.coins = list(hc.get("coins") or ["BTC", "ETH", "SOL", "HYPE"])
        self.edge = float(hc.get("edge", 0.03))
        self.rest = rest or HLRest(URLS[cfg.get("network", "mainnet")][0], timeout=10)
        self.jev = jev_client
        claude = claude_client if hc.get("claude", True) else None
        # a short timeout and no retries: one slow Claude call must not make the recorder miss checkpoint windows
        self.claude = claude.with_options(timeout=60.0, max_retries=0) if hasattr(claude, "with_options") else claude
        self.claude_model = claude_model
        self.clock = clock
        self.path = Path(data_dir) / "hip4.json"
        self.v1 = {"above_mid": build_questions()["above_mid"]}
        wanted = hc.get("variants", ["jev_v3"])
        self.variants = {n: (spec, {"above_mid": api["above_mid"]}, dec["above_mid"])
                         for n, (spec, api, dec) in (variants or {}).items() if n in wanted}
        self.forecasters = ["market", "model", "model_emp", "model_har", "jev", *self.variants, "claude"]
        self.snaps: list[dict] = []
        self.taken: dict[str, float] = {}
        self.outcomes: dict[str, dict] = {}
        self.retry: dict[str, list] = {}                   # id -> [next_try_ts, attempts] for unsettled markets
        self.markets: list[dict] = []
        self._meta_at = 0.0
        self._c5: dict[str, tuple[int, list]] = {}
        self._lock = threading.Lock()
        self._scores: dict = {}
        self._scores_all: dict = {}
        self._score_key = None
        self.save_ok = True
        self.status, self.error, self.error_ts = ("waiting" if self.enabled else "off"), None, None
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:   # never overwrite history we could not read: keep a copy, stop saving
            try:
                shutil.copy2(self.path, self.path.with_suffix(f".unreadable-{int(time.time())}.json"))
            except OSError:
                pass
            self.save_ok = False
            self._err(f"could not read {self.path.name} ({exc}); saving disabled until restart")
            return
        self.snaps = [s for s in d.get("snaps", []) if s.get("v") == SNAP_VERSION]
        keep = {s["key"] for s in self.snaps}
        self.taken = {k: v for k, v in d.get("taken", {}).items() if k in keep}
        self.outcomes, self.retry = d.get("outcomes", {}), d.get("retry", {})

    def _save(self) -> None:
        if not self.save_ok:
            return
        with self._lock:
            if len(self.snaps) > MAX_SNAPS:
                self.snaps = self.snaps[-MAX_SNAPS:]
            # bookkeeping only for snapshots we still hold and markets still listed
            ids = {str(s["id"]) for s in self.snaps} | {str(m["id"]) for m in self.markets}
            self.taken = {k: v for k, v in self.taken.items() if k.split(":")[0] in ids}
            self.outcomes = {k: v for k, v in self.outcomes.items() if k in ids}
            self.retry = {k: v for k, v in self.retry.items() if k in ids}
            data = json.dumps({"snaps": self.snaps, "taken": self.taken, "outcomes": self.outcomes, "retry": self.retry})
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self.path)

    def _err(self, msg: str) -> None:
        self.error, self.error_ts = msg[:240], self.clock()

    # ------------------------------------------------------------------ loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        while True:
            try:
                self.step()
            except Exception as exc:  # keep the thread alive no matter what
                self.status = "error"
                self._err(f"{type(exc).__name__}: {exc}")
            if stop.wait(20.0):
                return

    def step(self) -> None:
        now = self.clock()
        if self.error_ts is not None and now - self.error_ts > 900:   # show an error for 15 minutes, then clear it
            self.error = self.error_ts = None
        if now - self._meta_at >= META_EVERY_S:
            meta = self.rest.post({"type": "outcomeMeta"}) or {}
            self.markets = [m for m in (parse_market(o, self.coins) for o in meta.get("outcomes", [])) if m]
            self._meta_at = now
        due = []
        for m in self.markets:
            tte = (m["expiry"] - now) / 60
            for name, lo, hi in CHECKPOINTS:
                if lo <= tte <= hi and f"{m['id']}:{name}" not in self.taken:
                    due.append((m, name))
        if due:
            self.snapshot(due, now)
        self.resolve_due(now)
        self.refresh_scores()
        self.status = "ok"

    # ------------------------------------------------------------------ snapshots
    def _c5_for(self, coin: str, now: float) -> list:
        """15 days of closed 5-minute bars, kept fresh incrementally: one full download (~90 weight) every 6 hours,
        otherwise only the bars since the last one we have (~20 weight), so features never use stale bars."""
        got = self._c5.get(coin)
        if got is None or now - got[0] > 6 * 3600:
            self._c5[coin] = (now, self.rest.candles(coin, "5m", 15 * 1440 + 10, now))
        else:
            bars = got[1]
            if bars and now - (bars[-1].ts + 300) >= 300:   # at least one newer bar has closed
                fresh = self.rest.candles(coin, "5m", int((now - bars[-1].ts) / 60) + 10, now)
                bars = bars + [c for c in fresh if c.ts > bars[-1].ts]
                self._c5[coin] = (got[0], bars[-(15 * 288 + 4):])
        return self._c5[coin][1]

    def _ctxs(self) -> dict:
        for _ in range(2):
            try:
                return self.rest.asset_ctxs()
            except Exception as exc:
                self._err(f"asset contexts: {exc}")
        return {}

    def snapshot(self, due: list, now: float) -> None:
        ctxs = self._ctxs()
        hist: dict[str, tuple] = {}
        for coin in {m["coin"] for m, _ in due}:              # 1) candles, before any book is read
            try:
                hist[coin] = (self._c5_for(coin, now), self.rest.candles(coin, "1m", 20, now))
            except Exception as exc:
                self._err(f"{coin} candles: {exc}")
        new = []
        for m, ck in due:                                     # 2) each Yes book, then the perp book, back to back
            coin = m["coin"]
            if coin not in hist or len(hist[coin][0]) < 300:
                continue
            try:
                book = self.rest.post({"type": "l2Book", "coin": f"#{m['id'] * 10}"}) or {}
                bids, asks, perp_ts = self.rest.l2_book(coin)
            except Exception as exc:
                self._err(f"{coin} #{m['id']} books: {exc}")
                continue
            if not bids or not asks:
                continue
            spot = (bids[0][0] + asks[0][0]) / 2
            ctx = ctxs.get(coin, {})
            if "pyth" in m["source"].lower() and ctx.get("oracle") and ctx.get("mid"):
                spot *= ctx["oracle"] / ctx["mid"]            # Pyth settles on spot: apply the live oracle basis
            top = book_top(book.get("levels") or [])
            book_ts = (book.get("time") or 0) / 1000.0
            gap = abs(book_ts - perp_ts) if book_ts and perp_ts else None
            c5, c1 = hist[coin]
            feat = strike_state(coin, c5, c1, spot, now, m["expiry"], m["strike"], ctx, m["twap_s"])
            new.append({"v": SNAP_VERSION, "key": f"{m['id']}:{ck}", "checkpoint": ck, "ts": now, **m, "spot": spot,
                        **top, "usable": top["usable"] and gap is not None and gap <= MAX_BOOK_GAP_S, "book_gap_s": gap,
                        "ctx_ok": bool(ctx), "model": feat["model"], "model_emp": feat["model_emp"],
                        "model_har": feat["model_har"], "har": feat["har"], "n_z": feat["n_z"],
                        "moneyness": round(feat["moneyness"], 3), "state": feat["state"]})
        if not new:
            return
        with ThreadPoolExecutor(4) as ex:                     # 3) Jev v1 and variants without the claude block
            jobs = [(s, "jev", ex.submit(self._ask, s["state"], self.v1, None)) for s in new]
            jobs += [(s, n, ex.submit(self._ask_variant, s, *self.variants[n])) for s in new for n in self.variants
                     if "claude" not in self.variants[n][0].get("blocks_above", [])]
            if self.claude is not None:
                self._ask_claude(new)                         # runs alongside the Jev calls above
            for s, name, fut in jobs:
                s[name] = fut.result()
            later = [(s, n, ex.submit(self._ask_variant, s, *self.variants[n])) for s in new for n in self.variants
                     if "claude" in self.variants[n][0].get("blocks_above", [])]
            for s, name, fut in later:
                s[name] = fut.result()
        with self._lock:                                      # a failed snapshot is retried next step (20 s)
            for s in new:                                     # until its window closes
                self.snaps.append(s)
                self.taken[s["key"]] = now
        self._save()

    def _ask(self, state: dict, questions: dict, dec: dict | None) -> float | None:
        if self.jev is None:
            return None
        res = self.jev.ask(state, questions)
        if res.status not in ("ok", "mock"):
            return None
        a = res.answers.get("above_mid")
        if dec is None:
            return a.get("p") if a else None
        from .lab import decode
        return decode(dec, a, state)                   # nudge questions read their anchor from the state

    def _ask_variant(self, s: dict, spec: dict, api: dict, dec: dict) -> float | None:
        state = dict(s["state"])
        for block in spec.get("blocks_above", []):
            if block == "anchor":
                state["baseline_p_above_mid"] = round(s["model_emp"], 3)
            elif block == "claude":
                if s.get("claude") is None:
                    return None
                state["claude_p_above_mid"] = round(min(0.999, max(0.001, s["claude"])), 3)
            elif block == "har":
                state.update(s.get("har") or {})
        return self._ask(state, api, dec)

    def _ask_claude(self, snaps: list[dict]) -> None:
        payload = {s["key"]: {"coin": s["coin"], "expiry_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(s["expiry"])),
                              **s["state"]} for s in snaps}
        try:
            resp = self.claude.beta.messages.parse(
                model=self.claude_model, max_tokens=8000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low"}, system=CLAUDE_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                output_format=_claude_schema())
            got = {f.id: f.p_above for f in (resp.parsed_output.forecasts if resp.parsed_output else [])}
        except Exception as exc:  # API or parsing problems: record and keep going
            self._err(f"claude: {type(exc).__name__}: {exc}")
            got = {}
        for s in snaps:
            p = got.get(s["key"])
            s["claude"] = None if p is None else min(0.999, max(0.001, float(p)))

    # ------------------------------------------------------------------ outcomes
    def resolve_due(self, now: float) -> None:
        """settledOutcome for expired markets: at most SETTLE_CALLS_PER_STEP calls per step, newest expiry first, and
        unsettled ids back off (90 s, 3, 6, 12, 24 min ... up to hourly) so stuck markets can't eat the rate limit."""
        listed = {m["id"] for m in self.markets}             # still listed in outcomeMeta = not settled yet
        exp: dict[int, float] = {}
        for s in self.snaps:
            if "y" not in s and now >= s["expiry"] + 90 and s["id"] not in listed:
                exp[s["id"]] = s["expiry"]
        changed, calls = False, 0
        for oid in sorted(exp, key=lambda i: -exp[i]):
            key = str(oid)
            if key not in self.outcomes:
                nxt, tries = self.retry.get(key, [0.0, 0])
                if now < nxt:
                    continue
                if now > exp[oid] + GIVE_UP_S:
                    self.outcomes[key] = {"void": True, "details": "never settled"}
                elif calls >= SETTLE_CALLS_PER_STEP:
                    continue
                else:
                    calls += 1
                    try:
                        res = self.rest.post({"type": "settledOutcome", "outcome": oid})
                    except Exception as exc:
                        res = None
                        self._err(f"settle #{oid}: {exc}")
                    if not res:
                        self.retry[key] = [now + min(3600.0, 90.0 * 2 ** tries), tries + 1]
                        changed = True
                        continue
                    self.outcomes[key] = {"fraction": float(res.get("settleFraction", "nan")), "details": res.get("details")}
                self.retry.pop(key, None)
                changed = True
            out = self.outcomes[key]
            frac = out.get("fraction", float("nan"))
            for s in self.snaps:
                if s["id"] == oid and "y" not in s:
                    s["y"] = None if out.get("void") or not math.isfinite(frac) else frac
                    changed = True
        if changed:
            self._save()

    # ------------------------------------------------------------------ reporting
    def rows(self, primary_only: bool = True) -> list[dict]:
        out = []
        for s in self.snaps:
            if s.get("y") not in (0.0, 1.0) or not s.get("usable"):
                continue
            if primary_only and not (primary(s) and abs(s.get("moneyness", 9)) <= NEAR_MONEY):
                continue
            out.append({"id": s["id"], "expiry": s["expiry"], "y": int(s["y"]), "bid": s["bid"], "ask": s["ask"],
                        "market": s["mid"], **{w: s.get(w) for w in self.forecasters if w != "market"}})
        return out

    def refresh_scores(self) -> None:
        """Bootstraps run here, in the recorder's thread, only when new outcomes land - never in the web server."""
        rows, rows_all = self.rows(True), self.rows(False)
        key = (len(rows), len(rows_all))
        if key != self._score_key:
            self._scores = {w: score(rows, w, self.edge) for w in self.forecasters}
            self._scores_all = {w: score(rows_all, w, self.edge, n_boot=500) for w in self.forecasters}
            self._score_key = key

    def to_dict(self) -> dict:
        now = self.clock()
        with self._lock:
            recent = [{k: s.get(k) for k in ("key", "checkpoint", "ts", "id", "coin", "venue", "strike", "expiry", "rule",
                                             "source", "twap_s", "spot", "bid", "ask", "mid", "usable", "book_gap_s",
                                             "moneyness", "model", "model_emp", "model_har", *self.forecasters[4:], "y")}
                      for s in self.snaps[-40:][::-1]]
            for r in recent:
                r["primary"] = primary(r) and abs(r.get("moneyness") or 9) <= NEAR_MONEY
                r["void"] = bool(self.outcomes.get(str(r["id"]), {}).get("void"))
            n_snaps = len(self.snaps)
        upcoming = sorted(((m["expiry"] - now) / 60, m["coin"], m["venue"], m["strike"]) for m in self.markets
                          if 0 < m["expiry"] - now < 6 * 3600)[:12]
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "coins": self.coins,
                "edge": self.edge, "snapshots": n_snaps, "scored": self._scores.get("market", {}).get("n", 0),
                "scored_all": self._scores_all.get("market", {}).get("n", 0), "near_money": NEAR_MONEY,
                "markets_tracked": len(self.markets), "checkpoints": [c[0] for c in CHECKPOINTS],
                "forecasters": self.forecasters, "scores": self._scores, "scores_all": self._scores_all,
                "upcoming": [{"tte_min": round(t, 1), "coin": c, "venue": v, "strike": k} for t, c, v, k in upcoming],
                "recent": recent}
