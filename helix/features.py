"""Feature engineering for the fast loop. Pure math on the current book, recent trades,
sub-second mids and 1m candles. Uses only data that existed at decision time, never calls AI."""

from __future__ import annotations

import math
from statistics import fmean, median, pstdev

from .market import PerpView

MIN_SECONDS = 120  # need two minutes of mids before deciding


def ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    seed_len = min(period, len(values))
    e = fmean(values[:seed_len])
    out = [e] * seed_len
    for v in values[seed_len:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def resample(mids: list[tuple[float, float]], now: float, seconds: int, step: float = 1.0) -> list[float]:
    """Mid at each `step` for the last `seconds`, carrying the last known value forward.
    Only samples at or before `now` are used."""
    if not mids:
        return []
    out, j, last = [], 0, None
    start = now - seconds
    grid = [start + i * step for i in range(int(seconds / step) + 1)]
    for t in grid:
        while j < len(mids) and mids[j][0] <= t:
            last = mids[j][1]
            j += 1
        if last is not None:
            out.append(last)
    return out


def _bps(a: float, b: float) -> float:
    return (a / b - 1.0) * 1e4 if b else 0.0


def depth_within(levels: list[tuple[float, float]], mid: float, bps: float) -> float:
    """USD notional resting within `bps` of mid on one side."""
    lim = mid * bps / 1e4
    return sum(px * sz for px, sz in levels if abs(px - mid) <= lim)


def compute_features(v: PerpView, now: float, depth_history: list[float] | None = None) -> dict | None:
    mid = v.mid
    if not mid:
        return None
    grid = resample(v.mids, now, 300)
    if len(grid) < MIN_SECONDS:
        return None
    lr = [math.log(b / a) * 1e4 for a, b in zip(grid, grid[1:]) if a > 0 and b > 0]  # 1s log returns, bps
    vol_1s = pstdev(lr) if len(lr) > 2 else 0.0
    vol_60 = pstdev(lr[-60:]) if len(lr) >= 60 else vol_1s

    def ret(sec: int) -> float:
        return _bps(grid[-1], grid[-1 - sec]) if len(grid) > sec else 0.0

    b_sz, a_sz = v.bids[0][1], v.asks[0][1]
    micro = (v.bids[0][0] * a_sz + v.asks[0][0] * b_sz) / (a_sz + b_sz) if a_sz + b_sz > 0 else mid

    def imbalance(n: int) -> float:
        b = sum(s for _, s in v.bids[:n])
        a = sum(s for _, s in v.asks[:n])
        return (b - a) / (b + a) if b + a > 0 else 0.0

    depth_bid = depth_within(v.bids, mid, 10)
    depth_ask = depth_within(v.asks, mid, 10)
    depth = depth_bid + depth_ask
    hist = [d for d in (depth_history or []) if d > 0]
    depth_ratio = depth / median(hist) if len(hist) >= 10 and median(hist) > 0 else 1.0

    ref = v.book_ts or now  # trade times are exchange times: measure windows on the exchange clock

    def flow(window: int) -> tuple[float, float, int]:
        recent = [t for t in v.trades if ref - window <= t.ts <= ref + 1]
        buy = sum(t.px * t.sz for t in recent if t.side == "b")
        sell = sum(t.px * t.sz for t in recent if t.side == "s")
        return (buy / (buy + sell) if buy + sell > 0 else 0.5), buy + sell, len(recent)

    flow15, notional15, n15 = flow(15)
    flow60, notional60, n60 = flow(60)
    _, _, n300 = flow(300)
    rate15 = n15 / 15
    rate300 = n300 / 300
    recent60 = [t.px * t.sz for t in v.trades if ref - 60 <= t.ts <= ref + 1]
    big_share = 0.0
    if len(recent60) >= 10:
        cut = sorted(recent60)[int(len(recent60) * 0.9)]
        big_share = sum(x for x in recent60 if x >= cut) / sum(recent60)

    closes = [c.close for c in v.candles[-120:]]
    trend_gap = rsi14 = None
    if len(closes) >= 30:
        e20, e60 = ema_series(closes, 20)[-1], ema_series(closes, 60)[-1]
        trend_gap = _bps(e20, e60)
        rsi14 = rsi(closes, 14)

    def candle_ret(minutes: int) -> float:
        """Longer context for multi-minute horizons: mid now vs the 1m close `minutes` ago."""
        return _bps(mid, closes[-1 - minutes]) if len(closes) > minutes else 0.0

    ctx = v.ctx or {}
    funding = ctx.get("funding")
    oracle = ctx.get("oracle")
    return {
        "mid": mid,
        "micro_vs_mid_bps": _bps(micro, mid),
        "spread_bps": v.spread_bps or 0.0,
        "ret_5s": ret(5), "ret_15s": ret(15), "ret_30s": ret(30), "ret_60s": ret(60), "ret_300s": ret(min(300, len(grid) - 1)),
        "ret_15m": candle_ret(15), "ret_60m": candle_ret(60),
        "vol_1s_bps": vol_1s,
        "vol_ratio": vol_60 / vol_1s if vol_1s > 0 else 1.0,
        "imbalance_5": imbalance(5),
        "imbalance_20": imbalance(20),
        "depth_10bps_usd": depth,
        "depth_ratio": depth_ratio,
        "flow_15s": flow15,
        "flow_60s": flow60,
        "notional_15s": notional15,
        "trade_rate_15s": rate15,
        "trade_rate_ratio": rate15 / rate300 if rate300 > 0 else 1.0,
        "big_trade_share_60s": big_share,
        "funding_hr_bps": funding * 1e4 if funding is not None else 0.0,
        "premium_bps": _bps(mid, oracle) if oracle else 0.0,
        "trend_1m_bps": trend_gap if trend_gap is not None else 0.0,
        "rsi14_1m": rsi14 if rsi14 is not None else 50.0,
        "data_age_s": max(0.0, now - v.received),
    }


def sigma_bps(feats: dict, horizon_s: float) -> float:
    """Expected 1-sigma move over the horizon, from 1-second realized volatility."""
    return feats["vol_1s_bps"] * math.sqrt(max(1.0, horizon_s))
