"""Hourly probability test. Paper only: it never trades.

Every UTC hour (opened in the first 10 minutes, expiring at the next hour mark) and for each coin, it records
P(price is ABOVE strike K at expiry) for three strikes: spot, and spot x exp(+/-0.5 sigma) where sigma is the
expected move to expiry. Three forecasters answer the same question from the same numbers:

  model      zero-drift lognormal: P = Phi((ln(S/K) - sigma^2/2) / sigma), sigma from the last 24 h of 5-minute returns.
  model_emp  hour-of-day volatility + the empirical (fat-tailed) distribution of past hourly moves: the bar to beat.
  jev        one Jev call per coin, one independent yes/no ("noul") question per strike (the original wording).
  jev_v2/v3  the question sets chosen in the Jev Lab backtest (helix/prompts/*.json, config hourly_test.jev_variants):
             Jev classifies the present situation and code turns the class into a probability.
  claude     one Claude call per hour for all coins (when the Claude forecaster is configured).

Outcomes come from the 1-minute candle that closes exactly at expiry (Hyperliquid candleSnapshot), so a restart
or a late resolution never changes the answer. Windows never overlap (one round per hour), and every round is
saved to data/hourly_test.json, so restarts don't lose pending predictions.

Scoring: Brier score and log loss per forecaster, calibration bins, and a paired comparison against the model
with a bootstrap confidence interval that resamples whole hours (strikes and coins within an hour are correlated)."""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from pathlib import Path
from statistics import fmean, pstdev  # noqa: F401  (pstdev used in realized_sigma)

from .hyperliquid import URLS, HLRest
from .jev import JevClient

ROUND_OPEN_WINDOW_S = 600   # open a round only in the first 10 minutes of an hour
HIST_DAYS = 15              # 5-minute history for the empirical baseline (~4,300 bars, under the 5,000-candle limit)
TARGET_HOURS = 2000         # fixed in advance: no verdict before this (peeking at a running test inflates false positives)
ALPHA_ONE_SIDED = 0.025 / 5 # Bonferroni for five forecasters (jev, jev_v2, jev_v3, jev_v4, claude)
BLOCK_HOURS = 24            # moving-block bootstrap: volatility clusters make neighbouring hours correlated
GIVE_UP_S = 72 * 3600       # 1-minute candles stay available for ~83 h
STRIKE_SIGMAS = (-0.5, 0.0, 0.5)
STRIKE_NAMES = ("low", "mid", "high")
EPS = 1e-4


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lognormal_above(spot: float, strike: float, sigma: float) -> float:
    """P(S_T > K) for zero drift, where sigma is the st.dev. of ln(S_T/S_0) over the whole horizon."""
    if sigma <= 0:
        return 1.0 if spot > strike else 0.0
    return phi((math.log(spot / strike) - 0.5 * sigma * sigma) / sigma)


def sigma_per_hour(closes: list[float], bar_minutes: int = 5) -> float:
    """St.dev. of log returns per hour, from consecutive closes of `bar_minutes` bars."""
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 12:
        return 0.0
    return pstdev(rets) * math.sqrt(60 / bar_minutes)


def seasonal_factors(c5) -> list[float]:
    """Hour-of-day (UTC) volatility multipliers from squared 5-minute log returns. Clamped to [0.5, 2.5]."""
    sq: list[list[float]] = [[] for _ in range(24)]
    for a, b in zip(c5, c5[1:]):
        if a.close > 0 and b.close > 0 and b.ts - a.ts == 300:
            sq[(b.ts // 3600) % 24].append(math.log(b.close / a.close) ** 2)
    allv = [x for s in sq for x in s]
    base = fmean(allv) if allv else 0.0
    return [min(2.5, max(0.5, math.sqrt(fmean(s) / base))) if len(s) >= 24 and base > 0 else 1.0 for s in sq]


def past_z(c5, factors, lookback_bars: int = 288, step: int = 12, every: bool = False) -> list[float]:
    """Standardised moves over `step` 5-minute bars (12 = 1 hour): ln(P[t+step]/P[t]) / (24h HOURLY sigma at t x
    hour factor), sampled at past hour marks (default, as the hourly test does) or, with every=True, at every
    `step`-th bar (non-overlapping windows; far more samples for short windows). Each sigma uses only earlier bars."""
    out = []
    for i, c in enumerate(c5):
        start_ok = (c.ts // 300) % step == 0 if every else not (c.ts + 300) % 3600   # phase by time, not list index
        if not start_ok or i < lookback_bars or i + step >= len(c5) or c5[i + step].ts - c.ts != step * 300:
            continue
        s = sigma_per_hour([x.close for x in c5[i - lookback_bars:i + 1]]) * factors[((c.ts + 300) // 3600) % 24]
        if s > 0 and c.close > 0:
            out.append(math.log(c5[i + step].close / c.close) / s)
    return out


def empirical_above(zs: list[float], x: float, bandwidth: float = 0.3) -> float:
    """Kernel-smoothed P(Z > x) from past standardised moves (keeps fat tails, never exactly 0 or 1)."""
    return fmean(phi((z - x) / bandwidth) for z in zs)


# Recent-volatility blend (weights on the last 1 h, 4 h and 24 h variances), chosen on the Jev Lab's DESIGN period by
# grid search (lognormal log loss, 2026-10-05). With the empirical fat-tailed distribution ("model_har") it scored,
# held out: 15-minute windows 0.5880 vs model_emp 0.6006; 1-hour windows 0.5707 vs 0.5808 (lower is better).
# The shorter the window, the more the last hour matters (the HIP-4 market already prices volatility this way).
HAR_SHORT = ((0.8, 0.1, 0.1), False)     # (weights, use the hour-of-day factor)
HAR_LONG = ((0.2, 0.7, 0.1), True)


def har_params(window_min: float) -> tuple:
    return HAR_SHORT if window_min <= 30 else HAR_LONG


def har_sigma(h1: float, h4: float, h24: float, w=None) -> float:
    """Hourly sigma blending the last 1 h, 4 h and 24 h of 5-minute returns (a HAR-style volatility forecast)."""
    a, b, c = w or HAR_LONG[0]
    return math.sqrt(max(0.0, a * h1 * h1 + b * h4 * h4 + c * h24 * h24))


def realized_sigma(closes: list[float], bars: int) -> float:
    """Hourly sigma from the last `bars` 5-minute log returns (population st.dev., as the lab computes it)."""
    last = closes[-(bars + 1):]
    if len(last) < bars + 1:
        return 0.0
    return pstdev([math.log(b / a) for a, b in zip(last, last[1:])]) * math.sqrt(12)


def clip(p: float) -> float:
    return min(1 - EPS, max(EPS, float(p)))


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def logloss(p: float, y: int) -> float:
    p = clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def build_questions() -> dict:
    """One independent yes/no question per strike (Jev answers questions independently)."""
    q = {}
    for name in STRIKE_NAMES:
        q[f"above_{name}"] = {
            "type": "noul",
            "instructions": f"When `minutes_to_expiry` minutes have passed, will the price be above `strike_{name}`?",
            "criteria": {True: f"The price ends above `strike_{name}`", False: f"The price ends at or below `strike_{name}`"},
        }
    return q


CLAUDE_PROMPT = """You forecast the probability that crypto perpetual futures prices end ABOVE given strike prices at
the next full hour, for a paper-trading research test. Nothing is traded. Forecasts are scored with log loss
against what actually happens, alongside a zero-drift lognormal model.

For each coin you get the current price, minutes to expiry, returns over 15 minutes to 24 hours, the expected
1-sigma move to expiry (`expected_move_to_expiry_bps`), funding, premium, where the price sits in its 24-hour
range, and three strikes (`strike_low`, `strike_mid`, `strike_high`) with their distance from the price in bps.
Return p_above_low, p_above_mid and p_above_high for every coin: each is the probability that the price at expiry
is above that strike, so they must not increase from low to high. Be calibrated; short-horizon moves are mostly noise."""


def _claude_schema():
    from pydantic import BaseModel

    class CoinAbove(BaseModel):
        coin: str
        p_above_low: float
        p_above_mid: float
        p_above_high: float

    class AboveBatch(BaseModel):
        forecasts: list[CoinAbove]

    return AboveBatch


class HourlyTest:
    def __init__(self, cfg: dict, data_dir: Path, coins: list[str] | None = None, rest: HLRest | None = None,
                 jev_client=None, claude_client=None, claude_model: str | None = None, clock=time.time):
        hc = cfg.get("hourly_test", {})
        self.enabled = bool(hc.get("enabled", True))
        self.coins = list(coins or hc.get("coins") or ["BTC", "ETH", "SOL", "HYPE"])
        self.rest = rest or HLRest(URLS[cfg.get("network", "mainnet")][0], timeout=10)
        self.clock = clock
        self.path = Path(data_dir) / "hourly_test.json"
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self.jev = jev_client if jev_client is not None else (JevClient(key, cfg["jev"]["model"], 5000) if key else None)
        self.claude = claude_client
        self.claude_model = claude_model
        self.questions = build_questions()
        self.variants = self._load_variants(hc.get("jev_variants") or {})
        self.rounds: list[dict] = []
        self.status, self.error = ("waiting" if self.enabled else "off"), None
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _load_variants(paths: dict) -> dict:
        """name -> (spec, api questions, decoders). Paths are relative to the project root."""
        from . import lab
        root = Path(__file__).resolve().parent.parent
        out = {}
        for name, rel in paths.items():
            spec = json.loads((root / rel).read_text(encoding="utf-8"))
            api, dec, _ = lab.render(spec, "K")
            out[name] = (spec, api, dec)
        return out

    @property
    def forecasters(self) -> list[str]:
        return ["model", "model_emp", "model_har", "jev", *self.variants, "claude"]

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            self.rounds = json.loads(self.path.read_text(encoding="utf-8")).get("rounds", [])
        except (OSError, ValueError):
            self.rounds = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rounds": self.rounds[-5000:]}), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        while True:
            try:
                self.step()
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:300]
            if stop.wait(15.0):
                return

    def step(self) -> None:
        now = self.clock()
        hour = int(now // 3600) * 3600
        if now - hour < ROUND_OPEN_WINDOW_S:
            rnd = next((r for r in self.rounds if r["hour"] == hour), None)
            if rnd is None:
                self._retry_at = now
                self.open_round(hour, now)
            else:                                  # coins that failed (e.g. rate-limited): retry once a minute
                missing = [c for c in self.coins if c not in rnd["coins"]]
                if missing and now - getattr(self, "_retry_at", 0.0) >= 60:
                    self._retry_at = now
                    self.open_round(hour, now, only=missing)
        self.resolve_due(now)
        self.refresh_scores()
        self.status, self.error = "ok", None   # a step that completes clears an earlier transient error

    # ------------------------------------------------------------------ opening a round
    def _c5_for(self, coin: str, now: float) -> list:
        """The same ~15 days of closed 5-minute bars as a fresh download, kept up to date with small top-ups (a full
        download once a day) instead of ~90 request-weight per coin every hour."""
        cache = getattr(self, "_c5_cache", None)
        if cache is None:
            cache = self._c5_cache = {}
        got = cache.get(coin)
        want = HIST_DAYS * 1440 + 10
        if got is None or now - got[0] > 86400:
            got = (now, self.rest.candles(coin, "5m", want, now))
        else:
            bars = got[1]
            if bars and now - (bars[-1].ts + 300) >= 300:
                new = [c for c in self.rest.candles(coin, "5m", int((now - bars[-1].ts) / 60) + 10, now) if c.ts > bars[-1].ts]
                got = (got[0], [c for c in bars + new if c.ts >= now - want * 60])
        cache[coin] = got
        return [c for c in got[1] if c.ts >= now - want * 60 and c.ts + 300 <= now]

    def _coin_state(self, coin: str, now: float, expiry: float, ctxs: dict) -> dict | None:
        c5 = self._c5_for(coin, now)                                      # closed 5-minute bars, last ~15 days
        c1 = self.rest.candles(coin, "1m", 20, now)                         # closed 1-minute bars, last 20 min
        book = self.rest.l2_book(coin)
        bids, asks = book[0], book[1]
        if len(c5) < 300 or not bids or not asks:
            return None
        spot = (bids[0][0] + asks[0][0]) / 2
        closes = [c.close for c in c5]
        sig_h = sigma_per_hour(closes[-289:])                 # last 24 h
        t_h = max(1 / 60, (expiry - now) / 3600)
        sigma = sig_h * math.sqrt(t_h)
        factors = seasonal_factors(c5)
        f_now = factors[(expiry - 3600) // 3600 % 24]        # the UTC hour being forecast
        sigma_s = sig_h * f_now * math.sqrt(t_h)
        zs = past_z(c5, factors)

        def ret_bps(minutes: int) -> float:
            bars = minutes // 5                                # closes[-1] is the last closed bar, so n bars back is -(n+1)
            return (spot / closes[-(bars + 1)] - 1) * 1e4 if len(closes) > bars else 0.0

        r15 = (spot / c1[-16].close - 1) * 1e4 if len(c1) >= 16 else 0.0
        last_h = closes[-13:]
        rv1 = pstdev([math.log(b / a) for a, b in zip(last_h, last_h[1:])]) * math.sqrt(12) if len(last_h) == 13 else 0.0
        day = c5[-288:]
        hi, lo = max(c.high for c in day), min(c.low for c in day)
        ctx = ctxs.get(coin, {})
        strikes = [spot * math.exp(k * sigma) for k in STRIKE_SIGMAS]
        decimals = max(0, 6 - len(str(int(spot))))
        strikes = [round(k, decimals) for k in strikes]
        state = {
            "price": round(spot, decimals), "minutes_to_expiry": round((expiry - now) / 60, 1),
            "return_15m_bps": round(r15, 1), "return_1h_bps": round(ret_bps(60), 1),
            "return_4h_bps": round(ret_bps(240), 1), "return_24h_bps": round(ret_bps(1440), 1),
            "expected_move_to_expiry_bps": round(sigma * 1e4, 1), "vol_per_hour_24h_bps": round(sig_h * 1e4, 1),
            "realized_vol_last_1h_bps": round(rv1 * 1e4, 1), "hour_of_day_vol_factor": round(f_now, 2),
            "funding_hr_bps": round((ctx.get("funding") or 0.0) * 1e4, 3),
            "premium_bps": round((ctx.get("premium") or 0.0) * 1e4, 2),
            "position_in_24h_range": round((spot - lo) / (hi - lo), 3) if hi > lo else 0.5,
        }
        for name, k in zip(STRIKE_NAMES, strikes):
            state[f"strike_{name}"] = k
            state[f"strike_{name}_distance_bps"] = round((k / spot - 1) * 1e4, 1)
        model = [lognormal_above(spot, k, sigma) for k in strikes]
        # Stronger baseline: hour-of-day sigma + the empirical (fat-tailed) distribution of past standardised hourly moves
        if len(zs) >= 100 and sigma_s > 0:
            model_emp = [empirical_above(zs, math.log(k / spot) / sigma_s) for k in strikes]
        else:
            model_emp = [lognormal_above(spot, k, sigma_s) for k in strikes]
        # Added 2026-10-05 (not part of the pre-registered comparison): the recent-volatility blend
        w_har, seasonal = har_params((expiry - now) / 60)
        rv4 = realized_sigma(closes, 48)
        hs = har_sigma(rv1, rv4, sig_h, w_har) * (f_now if seasonal else 1.0) * math.sqrt(t_h)
        if len(zs) >= 100 and hs > 0:
            model_har = [empirical_above(zs, math.log(k / spot) / hs) for k in strikes]
        else:
            model_har = [lognormal_above(spot, k, hs) for k in strikes]
        har = {"realized_vol_last_4h_bps": round(rv4 * 1e4, 1)}
        for name, k, p in zip(STRIKE_NAMES, strikes, model_har):
            har[f"har_p_above_{name}"] = round(p, 3)
            har[f"strike_{name}_distance_sigmas"] = round(math.log(k / spot) / hs, 2) if hs > 0 else 0.0
        return {"spot": spot, "sigma": sigma, "sigma_h": sig_h, "sigma_seasonal": sigma_s, "t_h": t_h, "n_z": len(zs),
                "strikes": strikes, "state": state, "model": model, "model_emp": model_emp, "model_har": model_har,
                "har": har}

    def open_round(self, hour: int, now: float, only: list | None = None) -> None:
        """Open this hour's round, or (with `only`) add coins that failed at first - e.g. a rate-limited request -
        while the 10-minute opening window is still open, instead of silently leaving them out of the test."""
        expiry = hour + 3600
        try:
            ctxs = self.rest.asset_ctxs()
        except Exception:
            ctxs = {}
        coins = {}
        for coin in (only or self.coins):
            try:
                cs = self._coin_state(coin, now, expiry, ctxs)
            except Exception as exc:
                self.error = f"{coin}: {exc}"[:200]
                cs = None
            if cs is None:
                continue
            cs["jev"] = self._ask_jev(cs["state"])
            coins[coin] = cs
        if self.claude is not None and coins:
            self._ask_claude(coins)
        for cs in coins.values():                     # after Claude: a variant may read Claude's numbers
            for name in self.variants:
                cs[name] = self._ask_variant(name, cs)
        with self._lock:
            rnd = next((r for r in self.rounds if r["hour"] == hour), None)
            if rnd is None:
                self.rounds.append({"hour": hour, "created": now, "expiry": expiry, "coins": coins, "resolved": False})
            else:
                rnd["coins"].update(coins)
            self._save()

    def _ask_jev(self, state: dict) -> dict:
        if self.jev is None:
            return {"status": "offline"}
        res = self.jev.ask(state, self.questions)
        if res.status not in ("ok", "mock"):
            return {"status": res.status, "error": res.error}
        probs = [res.answers.get(f"above_{n}", {}).get("p") for n in STRIKE_NAMES]
        return {"status": "ok", "probs": probs, "latency_ms": res.latency_ms, "model": res.model}

    def _ask_variant(self, name: str, cs: dict) -> dict:
        """Same state as the lab built it, plus the blocks the variant asked for."""
        from . import lab
        spec, api, dec = self.variants[name]
        if self.jev is None:
            return {"status": "offline"}
        state = dict(cs["state"])
        for block in spec.get("blocks_above", []):
            if block == "anchor":
                state.update({f"baseline_p_above_{s}": round(p, 3) for s, p in zip(STRIKE_NAMES, cs["model_emp"])})
            elif block == "claude":
                cp = (cs.get("claude") or {}).get("probs")
                if not cp or any(p is None for p in cp):
                    return {"status": "no claude"}
                state.update({f"claude_p_above_{s}": round(min(0.999, max(0.001, p)), 3) for s, p in zip(STRIKE_NAMES, cp)})
            elif block == "har":
                state.update(cs.get("har") or {})
        res = self.jev.ask(state, api)
        if res.status not in ("ok", "mock"):
            return {"status": res.status, "error": res.error}
        probs = [lab.decode(dec[f"above_{s}"], res.answers.get(f"above_{s}"), state) for s in STRIKE_NAMES]
        return {"status": "ok", "probs": probs, "latency_ms": res.latency_ms}

    def _ask_claude(self, coins: dict) -> None:
        payload = {c: v["state"] for c, v in coins.items()}
        try:
            resp = self.claude.beta.messages.parse(
                model=self.claude_model, max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low"},
                system=CLAUDE_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                output_format=_claude_schema(),
            )
            out = resp.parsed_output
            got = {f.coin: [f.p_above_low, f.p_above_mid, f.p_above_high] for f in (out.forecasts if out else [])}
            for c, v in coins.items():
                v["claude"] = {"status": "ok", "probs": got[c]} if c in got else {"status": "missing"}
        except Exception as exc:  # API, network or parsing problems: record and keep going
            for v in coins.values():
                v["claude"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    # ------------------------------------------------------------------ resolving
    def resolve_due(self, now: float) -> None:
        changed = False
        for rnd in self.rounds:
            if rnd["resolved"] or now < rnd["expiry"] + 75:   # give the expiry candle time to close and publish
                continue
            for coin, v in rnd["coins"].items():
                if "outcome" in v:
                    continue
                try:
                    px = self._price_at(coin, rnd["expiry"], now)
                except Exception as exc:  # one coin's network error must not block the others or the save
                    self.error = f"resolve {coin}: {exc}"[:200]
                    continue
                if px is None:
                    continue
                v["outcome"] = {"price": px, "above": [int(px > k) for k in v["strikes"]]}
                changed = True
            done = all("outcome" in v for v in rnd["coins"].values()) or now > rnd["expiry"] + GIVE_UP_S
            if done and not rnd["resolved"]:
                rnd["resolved"], changed = True, True
        if changed:
            with self._lock:
                self._save()

    def _price_at(self, coin: str, expiry: float, now: float) -> float | None:
        """Close of the 1-minute candle that ends exactly at expiry."""
        minutes = int((now - expiry) / 60) + 5
        for c in self.rest.candles(coin, "1m", minutes, now):
            if c.ts == int(expiry) - 60:
                return c.close
        return None

    # ------------------------------------------------------------------ scoring
    def rows(self) -> list[dict]:
        out = []
        for rnd in self.rounds:
            for coin, v in rnd["coins"].items():
                if "outcome" not in v:
                    continue
                for j, name in enumerate(STRIKE_NAMES):
                    row = {"hour": rnd["hour"], "coin": coin, "strike": name, "y": v["outcome"]["above"][j],
                           "model": v["model"][j]}
                    if v.get("model_emp"):
                        row["model_emp"] = v["model_emp"][j]
                    if v.get("model_har"):
                        row["model_har"] = v["model_har"][j]
                    for who in ("jev", *self.variants, "claude"):
                        p = (v.get(who) or {}).get("probs")
                        if p and p[j] is not None:
                            row[who] = float(p[j])
                    out.append(row)
        return out

    @staticmethod
    def score(rows: list[dict], who: str, n_boot: int = 5000, seed: int = 7, target_hours: int = TARGET_HOURS) -> dict:
        """Scores for one forecaster. For Jev/Claude also a paired comparison against the stronger baseline
        ("model_emp" when present), with the at-the-money (mid) strike as the pre-registered primary test:
        moving-block bootstrap over 24-hour blocks, Bonferroni-corrected, and no verdict before target_hours."""
        r = [x for x in rows if who in x]
        if not r:
            return {"n": 0}
        out = {"n": len(r), "hours": len({x["hour"] for x in r}),
               "brier": round(fmean(brier(x[who], x["y"]) for x in r), 4),
               "logloss": round(fmean(logloss(x[who], x["y"]) for x in r), 4)}
        out["calibration"] = [
            {"lo": lo, "hi": lo + 0.2, "n": len(b), "avg_p": round(fmean(x[who] for x in b), 3), "hit": round(fmean(x["y"] for x in b), 3)}
            for lo in (0.0, 0.2, 0.4, 0.6, 0.8)
            for b in [[x for x in r if lo <= x[who] < lo + 0.2 or (lo == 0.8 and x[who] == 1.0)]] if b]
        if who in ("model", "model_emp"):
            return out
        bar = "model_emp" if all("model_emp" in x for x in r) else "model"
        gain = lambda x, base: logloss(x[base], x["y"]) - logloss(x[who], x["y"])  # noqa: E731  (positive = better than base)
        out["bar"] = bar
        out["gain_vs_model"] = round(fmean(gain(x, "model") for x in r), 4)
        out["gain_vs_bar"] = round(fmean(gain(x, bar) for x in r), 4)
        out["per_strike"] = {s: round(fmean(gain(x, bar) for x in r if x["strike"] == s), 4)
                             for s in STRIKE_NAMES if any(x["strike"] == s for x in r)}
        per_hour: dict[int, list[float]] = {}
        for x in r:
            if x["strike"] == "mid":
                per_hour.setdefault(x["hour"], []).append(gain(x, bar))
        hours = sorted(per_hour)
        H = len(hours)
        out["primary"] = "mid strike vs " + bar
        out["primary_gain"] = round(fmean(d for h in hours for d in per_hour[h]), 4) if hours else None
        out["hours_done"], out["hours_needed"] = H, target_hours
        if H >= 2 * BLOCK_HOURS:
            sums, cnts = [sum(per_hour[h]) for h in hours], [len(per_hour[h]) for h in hours]
            ps, pc = [0.0], [0]                       # prefix sums over the circular series: each block is O(1)
            for k in range(H + BLOCK_HOURS):
                ps.append(ps[-1] + sums[k % H])
                pc.append(pc[-1] + cnts[k % H])
            rng, boots, nb = random.Random(seed), [], math.ceil(H / BLOCK_HOURS)
            for _ in range(n_boot):
                tot = cnt = 0.0
                for _ in range(nb):
                    s = rng.randrange(H)
                    tot += ps[s + BLOCK_HOURS] - ps[s]
                    cnt += pc[s + BLOCK_HOURS] - pc[s]
                boots.append(tot / cnt)
            boots.sort()
            lo, hi = boots[int(ALPHA_ONE_SIDED * n_boot)], boots[int((1 - ALPHA_ONE_SIDED) * n_boot) - 1]
            out["primary_ci"] = [round(lo, 4), round(hi, 4)]
        else:
            lo = hi = None
        if H < target_hours:
            out["beats_bar"] = None
            out["verdict"] = f"collecting: {H}/{target_hours} hours, no verdict before the fixed horizon"
        else:
            out["beats_bar"] = lo is not None and lo > 0
            out["verdict"] = ("beats the baseline" if out["beats_bar"] else
                              "worse than the baseline" if hi is not None and hi < 0 else "no evidence of skill")
        return out

    def refresh_scores(self) -> None:
        """Recompute scores only when new outcomes land (called from the test's own thread, not the web server)."""
        rows = self.rows()
        if getattr(self, "_score_key", None) != len(rows):
            self._scores = {who: self.score(rows, who) for who in self.forecasters}
            self._score_key = len(rows)

    def to_dict(self) -> dict:
        now = self.clock()
        if not hasattr(self, "_scores"):
            self.refresh_scores()
        recent = []
        for rnd in self.rounds[-6:][::-1]:
            recent.append({"hour": rnd["hour"], "expiry": rnd["expiry"], "resolved": rnd["resolved"],
                           "coins": {c: {"spot": v["spot"], "strikes": v["strikes"], "model": v["model"], "model_emp": v.get("model_emp"),
                                         "model_har": v.get("model_har"),
                                         **{w: (v.get(w) or {}).get("probs") for w in ("jev", *self.variants, "claude")},
                                         "outcome": v.get("outcome")} for c, v in rnd["coins"].items()}})
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "coins": self.coins,
                "forecasters": self.forecasters, "variants": {n: s.get("theory", "")[:300] for n, (s, _, _) in self.variants.items()},
                "rounds": len(self.rounds), "resolved_rounds": sum(1 for r in self.rounds if r["resolved"]),
                "next_round": int(now // 3600) * 3600 + 3600, "target_hours": TARGET_HOURS,
                "scores": self._scores, "recent": recent}
