"""Jev v2 signal: the Jev Lab's chosen question set, asked once a minute per coin on exactly the features the
backtest used (lab.direction_snapshot is shared code), so a live answer means what the backtest measured.

How Jev is spoken to here (the Jev Lab's finding): Jev is never asked to forecast. Asked "where will price be
in 5 minutes", it restated the trailing move at ~90% confidence while short moves actually reversed, which put
the old bot on the wrong side of the market. Instead it classifies the PRESENT ("how stretched is price against
its own volatility?") against criteria written in terms of state fields, and code turns the class it picks into
a probability through the prompt's level_values, calibrated on the backtest's design period.

Per coin, after each 1-minute candle closes: 1m candles (~4 h) and 5m candles (~8 days, refreshed every 30 min)
from Hyperliquid's public REST API -> direction_snapshot -> one Jev call -> P(price higher in H minutes)."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import lab
from .hyperliquid import URLS, HLRest
from .market import Candle

C5_MINUTES = (lab.C5_NEED + 300) * 5     # enough 5-minute bars for the 7-day statistics and the 24 h range


def aggregate_5m(c1: list, after: int, now: float) -> list:
    """5-minute bars after bar `after`, built from contiguous closed 1-minute candles (open of the first minute,
    high/low over the five, close of the last, summed volume). Only complete, contiguous bars; [] if the 1-minute
    history doesn't start early enough."""
    by_ts = {c.ts: c for c in c1}
    out, b = [], after + 300
    while b + 300 <= now:
        five = [by_ts.get(b + 60 * k) for k in range(5)]
        if any(c is None for c in five):
            break
        out.append(Candle(b, five[0].open, max(c.high for c in five), min(c.low for c in five), five[-1].close,
                          sum(c.volume for c in five)))
        b += 300
    return out


class JevSignal:
    def __init__(self, cfg: dict, jev_client, rest: HLRest | None = None, clock=time.time, candle_source=None):
        """candle_source(coin, now) -> closed 1-minute candles from the live feed's WebSocket stream (no REST call);
        REST is used only when the stream hasn't delivered the newest candle yet."""
        sc = cfg["signal"]
        self.candle_source = candle_source
        self.max_continuation = float(sc.get("max_continuation", 0.5))
        root = Path(__file__).resolve().parent.parent
        self.spec = json.loads((root / sc["prompt"]).read_text(encoding="utf-8"))
        if "claude" in self.spec.get("blocks_direction", []):
            raise ValueError("signal.prompt needs Claude's per-minute forecast (claude block); pick a prompt without it")
        self.name = self.spec["name"]
        self.api, self.dec, _ = lab.render(self.spec, "D")
        self.h = int(sc["horizon_min"])
        self.entry_prob = float(sc["entry_prob"])
        self.max_age_s = float(sc["max_age_s"])
        self.coins = list(cfg["market"]["coins"])
        self.jev = jev_client
        self.enabled = bool(sc["enabled"]) and jev_client is not None
        self.rest = rest or HLRest(URLS[cfg.get("network", "mainnet")][0], timeout=10)
        self.clock = clock
        self.latest: dict[str, dict] = {}
        self._c5: dict[str, tuple[float, list]] = {}
        self._fund: dict[str, tuple[float, list]] = {}
        self._ctxs: dict = {}
        self._ctx_minute = -1
        self._momentum: dict[str, bool] = {}
        self._lock = threading.Lock()
        self.status = "waiting" if self.enabled else ("off" if not sc["enabled"] else "no Jev key")
        self.error: str | None = None
        self.calls = self.tokens = 0

    # ------------------------------------------------------------------ loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        while not stop.wait(2.0):
            try:
                self.step()
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:300]

    def step(self) -> None:
        now = self.clock()
        minute = int(now // 60) * 60
        if now - minute < 3:                       # let the exchange publish the candle that just closed
            return
        due = [c for c in self.coins if self.latest.get(c, {}).get("ts", 0) < minute]
        if not due:
            return
        if self._ctx_minute != minute:             # live context is only a funding fallback: once a minute at most
            self._ctx_minute = minute
            try:
                self._ctxs = self.rest.asset_ctxs()
            except Exception:
                pass
        with ThreadPoolExecutor(min(8, len(due))) as ex:
            list(ex.map(lambda c: self._one(c, now, minute, self._ctxs.get(c, {})), due))
        if any(self.latest.get(c, {}).get("status") == "ok" for c in self.coins):
            self.status, self.error = "ok", None

    def _offset(self, coin: str) -> int:
        return self.coins.index(coin) if coin in self.coins else 0

    def _funding(self, coin: str, now: float, minute: int, ctx: dict) -> tuple[float, float]:
        """(funding, premium) as the backtest saw them: the last SETTLED hourly record at or before the state's time
        (fundingHistory; the lab uses the same cut-off). Fetched once per hour per coin, spread over the first
        ~2.5 minutes of the hour so 20 coins don't hit the API at once; the live context is only a fallback."""
        got = self._fund.get(coin)
        hour_start = minute // 3600 * 3600
        stale = got is None or (max((r[0] for r in got[1]), default=0.0) < hour_start and now - got[0] > 120)
        if stale and (got is None or now - hour_start >= 30 + 6 * self._offset(coin)):
            try:
                self._fund[coin] = got = (now, lab.fetch_funding(self.rest, coin, now - 3 * 3600, now))
            except Exception:
                pass
        rec = max((r for r in (got[1] if got else []) if r[0] <= minute), key=lambda r: r[0], default=None)
        if rec is None:
            return (ctx.get("funding") or 0.0), (ctx.get("premium") or 0.0)
        return rec[1], rec[2]

    def _c5_for(self, coin: str, now: float, c1: list | None) -> list:
        """~8 days of closed 5-minute bars. One REST download per coin (again after 24 h, staggered), then new bars
        are built from the 1-minute stream: the same trades, so they match the exchange's 5-minute bars exactly
        (checked live: 900 bars, 0.0 bps difference). Falls back to a small REST top-up when that isn't possible."""
        got = self._c5.get(coin)
        if got is None or now - got[0] > 24 * 3600 + 60 * self._offset(coin):
            got = (now, self.rest.candles(coin, "5m", C5_MINUTES, now))
            self._c5[coin] = got
        bars = got[1]
        if not bars or now - (bars[-1].ts + 300) < 300:
            return bars
        built = aggregate_5m(c1 or [], bars[-1].ts, now)
        if not built:                              # stream history doesn't reach back far enough: REST top-up
            built = [c for c in self.rest.candles(coin, "5m", int((now - bars[-1].ts) / 60) + 10, now)
                     if c.ts > bars[-1].ts]
        bars = (bars + built)[-(C5_MINUTES // 5 + 10):]
        self._c5[coin] = (got[0], bars)
        return bars

    def _c1_for(self, coin: str, now: float, minute: int) -> tuple[list | None, bool]:
        """-> (closed 1-minute candles ending at the minute that just finished, or None; True if REST was tried).
        The stream is used only when its last 241 candles are contiguous (no gap from a missed minute)."""
        if self.candle_source is not None:
            c1 = self.candle_source(coin, now)
            if (c1 and c1[-1].ts + 60 == minute and len(c1) >= lab.C1_NEED
                    and c1[-1].ts - c1[-lab.C1_NEED].ts == (lab.C1_NEED - 1) * 60):
                return c1, False
            if now - minute < 10:                  # give the stream a few seconds before falling back to REST
                return None, False
        c1 = self.rest.candles(coin, "1m", 250, now)
        if c1 and c1[-1].ts + 60 < minute:
            if now - minute < 10:                  # maybe just not published yet: ask again in a few seconds
                return None, False
            if minute - (c1[-1].ts + 60) <= 3 * 60:   # REST leaves out minutes with no trades: flat candles
                px = c1[-1].close
                c1 = c1 + [Candle(t, px, px, px, px, 0.0) for t in range(c1[-1].ts + 60, minute, 60)]
        return (c1 if c1 and c1[-1].ts + 60 == minute else None), True

    def _fail(self, coin: str, minute: int, status: str, error: str) -> None:
        """A transient failure keeps a still-fresh good signal (with a warning) instead of wiping it."""
        prev = self.latest.get(coin)
        if prev and prev.get("status") == "ok" and self.clock() - prev.get("made", 0) <= self.max_age_s:
            self._set(coin, dict(prev, ts=minute, warn=f"{status}: {error}"[:200]))
        else:
            self._set(coin, {"ts": minute, "status": status, "error": error[:200]})

    def _one(self, coin: str, now: float, minute: int, ctx: dict) -> None:
        try:
            c1, tried_rest = self._c1_for(coin, now, minute)
            if c1 is None:
                if tried_rest:                     # no candle even over REST: wait for the next minute, no retry storm
                    self._fail(coin, minute, "no candle", "the newest 1-minute candle isn't available yet")
                return                             # otherwise give the stream a few more seconds
            fr, prem = self._funding(coin, now, minute, ctx)
            snap = lab.direction_snapshot(coin, c1, self._c5_for(coin, now, c1), fr * 1e4, prem * 1e4)
        except Exception as exc:
            self._fail(coin, minute, "data error", str(exc))
            return
        if snap is None:
            self._fail(coin, minute, "warming up", "not enough candle history yet")
            return
        state = dict(snap["state"])
        for block in self.spec.get("blocks_direction", []):
            state.update(snap[block])
        res = self.jev.ask(state, self.api)
        self.calls += 1
        self.tokens += int((res.usage or {}).get("input_tokens") or 0)
        if res.status not in ("ok", "mock"):
            self._set(coin, {"ts": minute, "status": res.status, "error": res.error})
            return
        q = f"p_up_{self.h}m"
        p = lab.decode(self.dec[q], res.answers.get(q))
        if p is None:
            self._set(coin, {"ts": minute, "status": "error", "error": f"no answer for {q}"})
            return
        probs = (res.answers.get(q) or {}).get("probabilities") or {}
        level = max(probs, key=lambda k: probs[k]) if probs else None
        crit = self.dec[q].get("criteria") or []
        label = crit[int(level)].split(":")[0] if level is not None and isinstance(crit, list) else None
        other = 15 if self.h == 5 else 5
        self._set(coin, {
            # signal_ts: the minute this forecast describes; _fail re-stamps "ts" when it keeps an older signal, but
            # never this, so the holding clock always starts at the forecast's own minute
            "ts": minute, "signal_ts": minute, "made": self.clock(), "status": "ok", "prompt": self.name, "horizon_min": self.h,
            "p_up": round(p, 4), f"p_up_{other}m": lab.decode(self.dec[f"p_up_{other}m"], res.answers.get(f"p_up_{other}m")),
            "class": label, "class_p": round(float(probs.get(level, 0.0)), 3) if level is not None else None,
            "extras": {k: v.get("p") for k, v in res.answers.items() if k not in self.dec and v.get("type") == "noul"},
            "move_sigmas": state.get(f"return_{self.h}m_bps", 0.0) / state[f"expected_move_{self.h}m_bps"]
            if state.get(f"expected_move_{self.h}m_bps") else None,
            "sigma_bps": state.get(f"expected_move_{self.h}m_bps"),   # candle-based 1-sigma move over the horizon
            "latency_ms": None if res.latency_ms is None else round(res.latency_ms),
            # regime guard input: Jev v2's mapping assumes short moves reverse (continuation < 0.5)
            "continuation": snap["stats"].get(f"continuation_rate_{self.h}m_7d"),
            "momentum_regime": self._regime(coin, snap["stats"].get(f"continuation_rate_{self.h}m_7d") or 0.0),
        })

    def _regime(self, coin: str, cont: float) -> bool:
        """Momentum regime with hysteresis: in above max_continuation, back out only below it minus 1 point, so a coin
        hovering around 50% doesn't flip in and out every few minutes."""
        was = self._momentum.get(coin, False)
        now = cont >= self.max_continuation - 0.01 if was else cont > self.max_continuation
        self._momentum[coin] = now
        return now

    def _set(self, coin: str, rec: dict) -> None:
        with self._lock:
            self.latest[coin] = rec

    # ------------------------------------------------------------------ reading
    def fresh(self, coin: str, now: float) -> dict | None:
        """The coin's latest usable signal, or None when it is missing, failed or older than max_age_s."""
        with self._lock:
            s = self.latest.get(coin)
        if not s or s.get("status") != "ok" or now - s.get("made", 0) > self.max_age_s:
            return None
        return s

    def to_dict(self) -> dict:
        with self._lock:
            latest = {c: dict(v) for c, v in self.latest.items()}
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "prompt": self.name,
                "theory": self.spec.get("theory", "")[:400], "horizon_min": self.h, "entry_prob": self.entry_prob,
                "calls": self.calls, "tokens": self.tokens, "latest": latest}
