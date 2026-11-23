"""Simulated perp market for `--demo` and tests.

A hidden "pressure" process drives order flow, book imbalance and future returns, with regime
switches (calm, trend, chop, stress) and thin-book episodes. It exists so every panel can be
exercised offline. It is never mixed with real data, and the dashboard labels it SIMULATED.
Nothing it produces says anything about how the strategy would do on the real market."""

from __future__ import annotations

import math
import random
import threading
from collections import deque

from .market import Candle, PerpView, Trade

# price, vol per sqrt(second), half-spread bps, level step bps, base level size (coins), open interest (coins)
_SPEC = {
    "BTC": (84000.0, 0.00011, 0.06, 0.25, 2.5, 25000.0),
    "ETH": (2900.0, 0.00015, 0.17, 0.4, 35.0, 600000.0),
    "SOL": (150.0, 0.00018, 0.33, 0.6, 450.0, 4000000.0),
}
_REGIMES = {"calm": (0.7, 1.25), "trend": (1.0, 1.0), "chop": (1.25, 0.9), "stress": (2.4, 0.45)}


class _SimCoin:
    def __init__(self, coin: str, seed: str, t0: float):
        spec = _SPEC.get(coin, (100.0, 0.0002, 0.4, 0.5, 100.0, 1e6))
        self.coin = coin
        self.mid, self.vol, self.half_spread, self.step_bps, self.size, self.oi = spec
        self.rng = random.Random(seed)
        self.t = t0
        self.pressure = 0.0
        self.bias = 0.0
        self.regime, self.regime_left = "calm", 0.0
        self.vol_mult, self.liq_mult = 1.0, 1.0
        self.mids: deque = deque()
        self.trades: deque = deque()
        self.candles: deque = deque(maxlen=240)
        self.bar: list | None = None  # [minute_ts, o, h, l, c, v]
        self.prev_day_px = self.mid * (1 + self.rng.uniform(-0.04, 0.04))
        self.funding_base = self.rng.uniform(-0.000005, 0.000018)
        self.day_volume = self.mid * self.oi * self.rng.uniform(0.6, 1.4)

    def _poisson(self, lam: float) -> int:
        limit, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def step(self, dt: float) -> None:
        r = self.rng
        if self.regime_left <= 0:
            self.regime = r.choices(list(_REGIMES), weights=[4, 3, 3, 1])[0]
            self.regime_left = r.uniform(90, 720)
            self.vol_mult, self.liq_mult = _REGIMES[self.regime]
            self.bias = r.choice((-1, 1)) * 0.35 if self.regime == "trend" else 0.0
        self.regime_left -= dt
        theta = dt / 40.0
        self.pressure += -theta * (self.pressure - self.bias) + 0.22 * math.sqrt(dt) * r.gauss(0, 1)
        drift = self.pressure * self.vol * 0.08
        ret = drift * dt + self.vol * self.vol_mult * math.sqrt(dt) * r.gauss(0, 1)
        if r.random() < 0.00005 * dt:
            ret += r.choice((-1, 1)) * r.uniform(3, 8) * self.vol * 10
        self.mid *= math.exp(ret)
        self.t += dt

        rate = 5.0 * self.vol_mult * (1 + 0.6 * abs(self.pressure))
        p_buy = 1 / (1 + math.exp(-1.7 * self.pressure))
        for _ in range(self._poisson(rate * dt)):
            side = "b" if r.random() < p_buy else "s"
            px = self.mid * (1 + (1 if side == "b" else -1) * self.half_spread / 1e4)
            sz = self.size * 0.08 * r.lognormvariate(0, 1.0)
            self.trades.append(Trade(self.t - r.uniform(0, dt), px, sz, side))
            if self.bar:
                self.bar[5] += sz
        while self.trades and self.t - self.trades[0].ts > 300:
            self.trades.popleft()

        if not self.mids or self.t - self.mids[-1][0] >= 0.25:
            self.mids.append((self.t, self.mid))
        while self.mids and self.t - self.mids[0][0] > 900:
            self.mids.popleft()

        minute = int(self.t // 60 * 60)
        if self.bar is None or self.bar[0] != minute:
            if self.bar is not None:
                self.candles.append(Candle(*self.bar))
            self.bar = [minute, self.mid, self.mid, self.mid, self.mid, 0.0]
        else:
            self.bar[2] = max(self.bar[2], self.mid)
            self.bar[3] = min(self.bar[3], self.mid)
            self.bar[4] = self.mid

    def advance_to(self, t: float, coarse_until: float) -> None:
        while self.t < t:
            dt = 1.0 if self.t < coarse_until else 0.1
            self.step(min(dt, max(0.001, t - self.t)) if t - self.t < dt else dt)

    def view(self, light: bool = False) -> PerpView:
        r = random.Random(f"{self.coin}-{int(self.t * 4)}")
        tilt = math.tanh(0.75 * self.pressure)
        half = self.half_spread * (0.8 + 0.4 * self.vol_mult)
        bids, asks = [], []
        for i in range(20):
            off = (half + i * self.step_bps * (1 + 0.15 * i)) / 1e4
            base = self.size * self.liq_mult * (0.5 + 0.18 * i) * r.lognormvariate(0, 0.4)
            bids.append((self.mid * (1 - off), base * (1 + 0.55 * tilt)))
            asks.append((self.mid * (1 + off), base * (1 - 0.55 * tilt)))
        premium = 0.00025 * math.tanh(self.pressure)
        ctx = {
            "mark": self.mid, "mid": self.mid, "oracle": self.mid * (1 - premium), "premium": premium,
            "funding": self.funding_base + 0.000012 * math.tanh(self.pressure),
            "open_interest": self.oi, "day_volume": self.day_volume, "prev_day_px": self.prev_day_px,
        }
        if light:
            return PerpView(self.coin, bids, asks, self.t, self.t, ctx=ctx, source="sim")
        return PerpView(self.coin, bids, asks, self.t, self.t, list(self.trades), list(self.mids),
                        list(self.candles), ctx, "sim")


class SimPerpFeed:
    name = "simulated"

    def __init__(self, coins: list[str], clock, seed: int = 7, warmup_s: float = 3 * 3600):
        self.coins = list(coins)
        self.clock = clock
        now = clock()
        self._lock = threading.Lock()
        self.sims = {c: _SimCoin(c, f"{seed}-{c}", now - warmup_s) for c in self.coins}
        for s in self.sims.values():
            s.advance_to(now, coarse_until=now - 900)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def view(self, coin: str) -> PerpView:
        with self._lock:
            s = self.sims[coin]
            s.advance_to(self.clock(), coarse_until=0)
            return s.view()

    def quote(self, coin: str) -> PerpView:
        with self._lock:
            s = self.sims[coin]
            s.advance_to(self.clock(), coarse_until=0)
            return s.view(light=True)

    def health(self) -> dict:
        return {"venue": "simulated", "network": "sim", "connected": True, "msgs_per_s": None, "reconnects": 0,
                "last_msg_age_s": 0.0, "ws_error": None, "rest_error": None, "rest_polls": 0,
                "sources": {c: "sim" for c in self.coins}}
