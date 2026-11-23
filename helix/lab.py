"""Jev Lab: an offline backtest of HOW Jev is asked, on real Hyperliquid history. Paper research only.

Every step caches under data/lab/, so reruns are cheap and resumable:
  fetch   1-minute candles (~3.5 days: the API keeps the latest 5,000), 5-minute candles (~17 days), funding history
  build   point-in-time states. Every feature uses only candles that had closed at that moment; outcomes come
          strictly after. The direction task carries no price level and no date (nothing to recall).
  claude  Claude's forecasts for the same states (optional, capped in dollars)
  jev     every question-design variant asked against every state (~$0.00006 per call)
  score   chronological split: the first 60% of the time is the DESIGN period (stacking weights, trade thresholds
          and the variant ranking are all chosen there); the last 40% is HELD OUT and only ever reported.

Task D  P(price is higher H minutes from now), H = 5 and 15; six coins, one state every 10 minutes.
        Bars to clear: 0.5, and a logistic "quant" model on the same numbers. Jev is useful only if its answers
        add to that quant model out of sample (stacked), or beat it directly.
Task K  The hourly test's question: P(price above strike at the next hour mark), three strikes, four coins.
        Bar to clear: the empirical fat-tailed baseline (model_emp), at-the-money strike primary."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

from .hourly import (HourlyTest, STRIKE_NAMES, STRIKE_SIGMAS, empirical_above, lognormal_above, logloss,
                     phi)
from .market import Candle

DIRECTION_COINS = ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE"]
HOURLY_COINS = ["BTC", "ETH", "SOL", "HYPE"]
HORIZONS = (5, 15)
STEP_D_S = 600            # one direction state every 10 minutes per coin
DESIGN_SHARE = 0.6        # first 60% of the timeline = design period
BLOCK_S = 6 * 3600        # bootstrap blocks for paired comparisons (volatility clusters)
COST_TAKER_BPS = 10.0     # round trip: 2 x 4.5 bps taker + ~1 bp slippage
COST_MAKER_BPS = 3.0      # round trip if both legs rest as maker orders (2 x 1.5 bps)
L2_D = 1000.0             # ridge penalty for direction stacks, chosen on design-period out-of-fold log loss only, grid
                          # 2/30/100/300/1000/3000 (2 -> -0.034/-0.061 vs a coin flip at 5/15 min; 300 -> -0.001/-0.004;
                          # 1000 -> +0.0016/+0.0029, best; 3000 -> +0.0011/+0.0019). An independent audit over 2..100000
                          # found the same optimum. Targets are near-noise, so heavy shrinkage wins.
L2_K = 50.0               # same rule for the hourly stacks: best overall design log loss (4 -> 0.6202, 50 -> 0.6171)
D_STATE_FIELDS = ["coin", "hour_of_day_utc", "return_1m_bps", "return_5m_bps", "return_15m_bps", "return_60m_bps",
                  "return_4h_bps", "return_24h_bps", "vol_1m_bps", "vol_ratio_15m_vs_4h", "expected_move_5m_bps",
                  "expected_move_15m_bps", "neutral_band_5m_bps", "neutral_band_15m_bps", "move_15m_in_sigmas",
                  "range_15m_bps", "up_bars_last_15", "volume_15m_vs_4h", "rsi14_1m", "ema20_gap_1m_bps",
                  "position_in_4h_range", "position_in_24h_range", "funding_hr_bps", "premium_bps"]
D_BLOCKS = {"stats": ["continuation_rate_5m_7d", "continuation_rate_15m_7d", "up_rate_15m_7d"],
            "anchor": ["baseline_p_up_5m", "baseline_p_up_15m"],
            "claude": ["claude_p_up_5m", "claude_p_up_15m"]}
K_STATE_FIELDS = ["price", "minutes_to_expiry", "return_15m_bps", "return_1h_bps", "return_4h_bps", "return_24h_bps",
                  "expected_move_to_expiry_bps", "vol_per_hour_24h_bps", "realized_vol_last_1h_bps",
                  "hour_of_day_vol_factor", "funding_hr_bps", "premium_bps", "position_in_24h_range"] + \
                 [f"strike_{s}{suf}" for s in STRIKE_NAMES for suf in ("", "_distance_bps")]
K_BLOCKS = {"anchor": [f"baseline_p_above_{s}" for s in STRIKE_NAMES],
            "claude": [f"claude_p_above_{s}" for s in STRIKE_NAMES],
            "har": ["realized_vol_last_4h_bps"] + [f"{p}{s}{q}" for s in STRIKE_NAMES
                                                   for p, q in (("har_p_above_", ""), ("strike_", "_distance_sigmas"))]}


# ====================================================================================== data
def _c(row) -> Candle:
    return Candle(int(row[0]), *map(float, row[1:6]))


def fetch_funding(rest, coin: str, start: float, end: float) -> list[list[float]]:
    out, t = [], int(start * 1000)
    while t < end * 1000:
        page = rest.post({"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": int(end * 1000)}) or []
        if not page:
            break
        out += [[x["time"] / 1000.0, float(x["fundingRate"]), float(x.get("premium") or 0.0)] for x in page]
        t = int(page[-1]["time"]) + 1
        if len(page) < 500:
            break
    return out


def _patient(fn, *args, tries: int = 6):
    """Retry a Hyperliquid call that hit the per-IP rate limit (shared with the live bot), backing off."""
    for k in range(tries):
        try:
            return fn(*args)
        except Exception as exc:
            if "429" not in str(exc) or k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def fetch_all(rest, coins: list[str], lab_dir: Path, now: float | None = None, skip_existing: bool = False,
              weight_per_min: float = 300.0) -> None:
    """Download history for each coin, paced to ~`weight_per_min` of Hyperliquid's 1,200/min per-IP budget: the live
    bot shares the IP, and a separate lab process has its own in-process budget that cannot see the bot's."""
    now = now or time.time()
    lab_dir.mkdir(parents=True, exist_ok=True)
    # per coin: two candle snapshots of 4,990 bars (20 + 4990/60 each) and ~1 funding page (20 + 432/20)
    coin_weight = 2 * (20 + 4990 / 60) + (20 + 18 * 24 / 20)
    for coin in coins:
        if skip_existing and (lab_dir / f"raw_{coin}.json").exists():
            continue
        t0 = time.monotonic()
        c1 = _patient(rest.candles, coin, "1m", 4990, now)
        c5 = _patient(rest.candles, coin, "5m", 4990 * 5, now)
        fund = _patient(fetch_funding, rest, coin, now - 18 * 86400, now)
        time.sleep(max(3.0, coin_weight * 60 / weight_per_min - (time.monotonic() - t0)))
        raw = {"coin": coin, "fetched": now, "c1": [[c.ts, c.open, c.high, c.low, c.close, c.volume] for c in c1],
               "c5": [[c.ts, c.open, c.high, c.low, c.close, c.volume] for c in c5], "funding": fund}
        (lab_dir / f"raw_{coin}.json").write_text(json.dumps(raw), encoding="utf-8")
        print(f"  {coin}: {len(c1)} x 1m, {len(c5)} x 5m, {len(fund)} funding records")


def load_raw(lab_dir: Path, coin: str) -> dict:
    raw = json.loads((lab_dir / f"raw_{coin}.json").read_text(encoding="utf-8"))
    raw["c1"] = [_c(r) for r in raw["c1"]]
    raw["c5"] = [_c(r) for r in raw["c5"]]
    return raw


def _funding_at(funding: list, t: float) -> tuple[float, float]:
    i = bisect.bisect_right([f[0] for f in funding], t) - 1
    return (funding[i][1], funding[i][2]) if i >= 0 else (0.0, 0.0)


def _prefix(xs) -> list[float]:
    out = [0.0]
    for x in xs:
        out.append(out[-1] + x)
    return out


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


# ====================================================================================== Task D states
STATS_BARS = 2016          # 7 days of 5-minute bars
C1_NEED, C5_NEED = 241, STATS_BARS + 6


def momentum_stats(cl5: list[float]) -> dict | None:
    """7-day continuation statistics from 5-minute closes ending at the last CLOSED bar. Index k counts only if
    its returns are fully inside the list, so passing the full history or just its last C5_NEED closes agrees."""
    n = len(cl5)
    s5 = v5 = s15 = v15 = u15 = vu = 0
    for k in range(max(0, n - STATS_BARS), n):
        if k >= 2:
            a, b = cl5[k - 1] / cl5[k - 2] - 1, cl5[k] / cl5[k - 1] - 1
            if a and b:
                v5 += 1
                s5 += _sign(a) == _sign(b)
        if k >= 3:
            b = cl5[k] / cl5[k - 3] - 1
            if b:
                vu += 1
                u15 += b > 0
            if k >= 6:
                a = cl5[k - 3] / cl5[k - 6] - 1
                if a and b:
                    v15 += 1
                    s15 += _sign(a) == _sign(b)
    if min(v5, v15, vu) < 200:
        return None
    return {"continuation_rate_5m_7d": s5 / v5, "continuation_rate_15m_7d": s15 / v15, "up_rate_15m_7d": u15 / vu}


def direction_snapshot(coin: str, c1: list[Candle], c5: list[Candle], funding_hr: float, premium: float) -> dict | None:
    """The direction-task state at the moment the LAST candle in c1 closed. c1: closed 1-minute bars (the last
    C1_NEED are used), c5: closed 5-minute bars (the last C5_NEED). The live bot and the backtest both call this,
    so Jev sees identical numbers in both. -> {state, stats, anchor} or None if history is short or gappy."""
    if len(c1) < C1_NEED or len(c5) < 289 or c1[-1].ts - c1[-C1_NEED].ts != 14400:
        return None
    c1 = c1[-C1_NEED:]
    cl = [c.close for c in c1]
    px, t = cl[-1], c1[-1].ts + 60
    ret = lambda m: (px / cl[-1 - m] - 1) * 1e4           # noqa: E731
    lr = [math.log(b / a) for a, b in zip(cl, cl[1:])]
    vol1 = pstdev(lr[-60:])
    v15, v240 = pstdev(lr[-15:]), pstdev(lr)
    em5, em15 = vol1 * math.sqrt(5) * 1e4, vol1 * math.sqrt(15) * 1e4
    ema, a = cl[1], 2 / 21
    for x in cl[2:]:
        ema += a * (x - ema)
    last15, h4 = c1[-15:], c1[-240:]
    d14 = [cl[k] - cl[k - 1] for k in range(len(cl) - 14, len(cl))]
    gains, losses = sum(x for x in d14 if x > 0), -sum(x for x in d14 if x < 0)
    hi4, lo4 = max(c.high for c in h4), min(c.low for c in h4)
    day = c5[-288:]
    hi24, lo24 = max(c.high for c in day), min(c.low for c in day)
    vol15m, vol4h = fmean(c.volume for c in last15), fmean(c.volume for c in h4)
    stats = momentum_stats([c.close for c in c5[-C5_NEED:]])
    if stats is None:
        return None
    state = {
        "coin": coin, "hour_of_day_utc": (t // 3600) % 24,
        "return_1m_bps": round(ret(1), 2), "return_5m_bps": round(ret(5), 2), "return_15m_bps": round(ret(15), 2),
        "return_60m_bps": round(ret(60), 1), "return_4h_bps": round(ret(240), 1),
        "return_24h_bps": round((px / c5[-289].close - 1) * 1e4, 1),
        "vol_1m_bps": round(vol1 * 1e4, 2), "vol_ratio_15m_vs_4h": round(v15 / v240, 2) if v240 > 0 else 1.0,
        "expected_move_5m_bps": round(em5, 1), "expected_move_15m_bps": round(em15, 1),
        "neutral_band_5m_bps": round(em5 / 2, 1), "neutral_band_15m_bps": round(em15 / 2, 1),
        "move_15m_in_sigmas": round(ret(15) / em15, 2) if em15 > 0 else 0.0,
        "range_15m_bps": round((max(c.high for c in last15) - min(c.low for c in last15)) / px * 1e4, 1),
        "up_bars_last_15": sum(1 for c in last15 if c.close > c.open),
        "volume_15m_vs_4h": round(vol15m / vol4h, 2) if vol4h > 0 else 1.0,
        "rsi14_1m": round(100 - 100 / (1 + gains / losses), 1) if losses > 0 else 100.0,
        "ema20_gap_1m_bps": round((px / ema - 1) * 1e4, 1),
        "position_in_4h_range": round((px - lo4) / (hi4 - lo4), 3) if hi4 > lo4 else 0.5,
        "position_in_24h_range": round((px - lo24) / (hi24 - lo24), 3) if hi24 > lo24 else 0.5,
        "funding_hr_bps": round(funding_hr, 3), "premium_bps": round(premium, 2),
    }
    anchor = {"baseline_p_up_5m": round(0.5 + (stats["continuation_rate_5m_7d"] - 0.5) * _sign(ret(5)), 3),
              "baseline_p_up_15m": round(0.5 + (stats["continuation_rate_15m_7d"] - 0.5) * _sign(ret(15)), 3)}
    stats = {k: round(v, 3) for k, v in stats.items()}
    return {"ts": t, "price": px, "state": state, "stats": stats, "anchor": anchor}


def build_direction(raw: dict, step_s: int = STEP_D_S) -> list[dict]:
    coin, c1, c5 = raw["coin"], raw["c1"], raw["c5"]
    funding = sorted(raw["funding"])
    c5_idx = {c.ts: i for i, c in enumerate(c5)}
    out = []
    for i in range(240, len(c1) - 15):
        t = c1[i].ts + 60                                 # the moment this state describes: bar i has just closed
        if t % step_s or c1[i + 15].ts - c1[i].ts != 900:
            continue
        j = c5_idx.get(t - 300)                           # the 5-minute bar closing at t
        if j is None or j < 288 + STATS_BARS // 4:
            continue
        fr, prem = _funding_at(funding, t)
        snap = direction_snapshot(coin, c1[i - 240:i + 1], c5[max(0, j - C5_NEED - 288):j + 1], fr * 1e4, prem * 1e4)
        if snap is None:
            continue
        px = snap["price"]
        out.append({"id": f"D:{coin}:{t}", "task": "D", "coin": coin, "ts": t, "state": snap["state"],
                    "stats": snap["stats"], "anchor": snap["anchor"],
                    "fwd": {str(h): (c1[i + h].close / px - 1) * 1e4 for h in HORIZONS},
                    "trail": {str(h): (px / c1[i - h].close - 1) * 1e4 for h in HORIZONS}})
    return out


# ====================================================================================== Task K / K15 states
from .hourly import HAR_LONG, HAR_SHORT, har_params, har_sigma  # noqa: E402,F401  (shared with the live tests)


def build_hourly(raw: dict, min_hist_days: float = 5.0) -> list[dict]:
    """Same state, strikes and baselines as HourlyTest._coin_state, rebuilt point-in-time from 5-minute bars."""
    return build_strikes(raw, 12, 3600, min_hist_days, "K")


def build_strikes(raw: dict, h: int = 12, step_s: int = 3600, min_hist_days: float = 5.0, task: str = "K") -> list[dict]:
    """P(price above strike after h 5-minute bars) for strikes at spot and spot x exp(+-0.5 sigma). h=12 is exactly
    the hourly test (hour marks, its baselines); other h use every non-overlapping h-bar window for the empirical
    baseline, rescaled to that window's width (as the HIP-4 recorder does). Records also carry the volatility
    pieces (last 1 h / 4 h / 24 h) for the recent-volatility baseline `model_har`."""
    coin, c5 = raw["coin"], raw["c5"]
    funding = sorted(raw["funding"])
    n = len(c5)
    cl = [c.close for c in c5]
    lr = [0.0] + [math.log(cl[k] / cl[k - 1]) for k in range(1, n)]
    S1, S2 = _prefix(lr), _prefix(x * x for x in lr)

    def sig24(i: int) -> float:                         # == sigma_per_hour(closes[i-288 : i+1])
        if i < 288:
            return 0.0
        s, s2 = S1[i + 1] - S1[i - 287], S2[i + 1] - S2[i - 287]
        var = max(0.0, s2 / 288 - (s / 288) ** 2)
        return math.sqrt(var) * math.sqrt(12)

    hours_sq = [[0.0] * (n + 1) for _ in range(24)]
    hours_n = [[0.0] * (n + 1) for _ in range(24)]
    for k in range(n):
        for hr in range(24):
            hours_sq[hr][k + 1], hours_n[hr][k + 1] = hours_sq[hr][k], hours_n[hr][k]
        if k >= 1 and c5[k].ts - c5[k - 1].ts == 300:
            hr = (c5[k].ts // 3600) % 24
            hours_sq[hr][k + 1] += lr[k] ** 2
            hours_n[hr][k + 1] += 1

    def factors(w0: int, j: int) -> list[float]:        # == seasonal_factors(c5[w0 : j+1])
        sq = [hours_sq[hr][j + 1] - hours_sq[hr][w0 + 1] for hr in range(24)]
        cnt = [hours_n[hr][j + 1] - hours_n[hr][w0 + 1] for hr in range(24)]
        tot_n = sum(cnt)
        base = sum(sq) / tot_n if tot_n else 0.0
        return [min(2.5, max(0.5, math.sqrt((sq[hr] / cnt[hr]) / base))) if cnt[hr] >= 24 and base > 0 else 1.0
                for hr in range(24)]

    hourly = h == 12
    scale = math.sqrt(h / 12)
    marks = [(i, ((c5[i].ts + 300) // 3600) % 24, math.log(cl[i + h] / cl[i]) / sig24(i) / scale)
             for i in range(288, n - h)
             if ((c5[i].ts + 300) % 3600 == 0 if hourly else (c5[i].ts // 300) % h == 0)   # phase by time, as live
             and c5[i + h].ts - c5[i].ts == h * 300 and sig24(i) > 0]
    out = []
    for j in range(n - h):
        t = c5[j].ts + 300
        if t % step_s or c5[j + h].ts - c5[j].ts != h * 300 or j < min_hist_days * 288:
            continue
        w0 = max(0, j - 4321)
        f = factors(w0, j)
        zs = [z / f[hh] for i, hh, z in marks if w0 + 288 <= i <= j - h]
        spot = cl[j]
        sig_h = sig24(j)
        sigma, f_now = sig_h * scale, f[((t + h * 150) // 3600) % 24]   # the UTC hour in the middle of the window
        sigma_s = sigma * f_now
        strikes = [spot * math.exp(k * sigma) for k in STRIKE_SIGMAS]
        decimals = max(0, 6 - len(str(int(spot))))
        strikes = [round(k, decimals) for k in strikes]
        model = [lognormal_above(spot, k, sigma) for k in strikes]
        if len(zs) >= 100 and sigma_s > 0:
            model_emp = [empirical_above(zs, math.log(k / spot) / sigma_s) for k in strikes]
        else:
            model_emp = [lognormal_above(spot, k, sigma_s) for k in strikes]
        last_h = cl[j - 12:j + 1]
        rv1 = pstdev([math.log(b / a) for a, b in zip(last_h, last_h[1:])]) * math.sqrt(12)
        last_4 = cl[j - 48:j + 1]
        rv4 = pstdev([math.log(b / a) for a, b in zip(last_4, last_4[1:])]) * math.sqrt(12)
        w_har, seasonal = har_params(h * 5)
        hs = har_sigma(rv1, rv4, sig_h, w_har) * (f_now if seasonal else 1.0) * scale
        if len(zs) >= 100 and hs > 0:
            model_har = [empirical_above(zs, math.log(k / spot) / hs) for k in strikes]
        else:
            model_har = [lognormal_above(spot, k, hs) for k in strikes]
        day = c5[j - 287:j + 1]
        hi, lo = max(c.high for c in day), min(c.low for c in day)
        fr, prem = _funding_at(funding, t)
        state = {
            "price": round(spot, decimals), "minutes_to_expiry": float(h * 5),
            "return_15m_bps": round((spot / cl[j - 3] - 1) * 1e4, 1), "return_1h_bps": round((spot / cl[j - 12] - 1) * 1e4, 1),
            "return_4h_bps": round((spot / cl[j - 48] - 1) * 1e4, 1), "return_24h_bps": round((spot / cl[j - 288] - 1) * 1e4, 1),
            "expected_move_to_expiry_bps": round(sigma * 1e4, 1), "vol_per_hour_24h_bps": round(sig_h * 1e4, 1),
            "realized_vol_last_1h_bps": round(rv1 * 1e4, 1), "hour_of_day_vol_factor": round(f_now, 2),
            "funding_hr_bps": round(fr * 1e4, 3), "premium_bps": round(prem * 1e4, 2),
            "position_in_24h_range": round((spot - lo) / (hi - lo), 3) if hi > lo else 0.5,
        }
        for name, k in zip(STRIKE_NAMES, strikes):
            state[f"strike_{name}"] = k
            state[f"strike_{name}_distance_bps"] = round((k / spot - 1) * 1e4, 1)
        px_T = cl[j + h]
        har = {"realized_vol_last_4h_bps": round(rv4 * 1e4, 1)}
        for name, k, p in zip(STRIKE_NAMES, strikes, model_har):
            har[f"har_p_above_{name}"] = round(p, 3)
            har[f"strike_{name}_distance_sigmas"] = round(math.log(k / spot) / hs, 2) if hs > 0 else 0.0
        out.append({"id": f"{task}:{coin}:{t}", "task": task, "coin": coin, "ts": t, "state": state, "model": model,
                    "model_emp": model_emp, "model_har": model_har, "y": [int(px_T > k) for k in strikes],
                    "n_z": len(zs), "anchor": {f"baseline_p_above_{s}": round(p, 3) for s, p in zip(STRIKE_NAMES, model_emp)},
                    "har": har, "vols": {"h1": rv1, "h4": rv4, "h24": sig_h, "f": f_now, "scale": scale,
                                         "spot": spot, "strikes": strikes}})
    return out


def build_all(lab_dir: Path, coins=None, tasks=("D", "K")) -> dict:
    """D for every coin; K (next hour) and K15 (15 minutes, like HIP-4's 15-minute markets) for the HIP-4 coins."""
    out = {t: [] for t in tasks}
    for coin in coins or DIRECTION_COINS:
        if not (lab_dir / f"raw_{coin}.json").exists():
            continue
        raw = load_raw(lab_dir, coin)
        if "D" in out:
            out["D"] += build_direction(raw)
        if "K" in out and coin in HOURLY_COINS:
            out["K"] += build_strikes(raw, 12, 3600, 5.0, "K")
        if "K15" in out and coin in HOURLY_COINS:
            out["K15"] += build_strikes(raw, 3, 900, 5.0, "K15")
    for t, rows in out.items():
        rows.sort(key=lambda r: (r["ts"], r["coin"]))
        (lab_dir / f"states_{t}.json").write_text(json.dumps(rows), encoding="utf-8")
    return {t: len(rows) for t, rows in out.items()}


def supports(spec: dict, task: str) -> bool:
    return bool(spec.get("direction")) if task == "D" else bool(spec.get("above"))


def load_states(lab_dir: Path, task: str) -> list[dict]:
    return json.loads((lab_dir / f"states_{task}.json").read_text(encoding="utf-8"))


def split_ts(states: list[dict]) -> float:
    ts = sorted({s["ts"] for s in states})
    return ts[int(len(ts) * DESIGN_SHARE)]


# ====================================================================================== variants
_FIELD = re.compile(r"`([A-Za-z0-9_]+)`")


def _sub(x, mapping: dict):
    if isinstance(x, str):
        for k, v in mapping.items():
            x = x.replace(k, v)
        return x
    if isinstance(x, list):
        return [_sub(v, mapping) for v in x]
    if isinstance(x, dict):
        return {_sub(k, mapping): _sub(v, mapping) for k, v in x.items()}
    return x


def _clean_q(q: dict) -> dict:
    """The question exactly as the API gets it (decoding hints stripped)."""
    out = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        out["criteria"] = q["criteria"]
    return out


def render(spec: dict, task: str) -> tuple[dict, dict, dict]:
    """-> (api_questions, decode_specs for the forecast questions, extra question ids). Raises on unknown fields."""
    api, dec = {}, {}
    if task == "D":
        for h in HORIZONS:
            q = _sub(spec["direction"], {"{H}": str(h)})
            api[f"p_up_{h}m"], dec[f"p_up_{h}m"] = _clean_q(q), q
        extras = spec.get("extras_direction") or {}
        allowed = set(D_STATE_FIELDS) | {f for b in spec.get("blocks_direction", []) for f in D_BLOCKS[b]}
    else:
        for s in STRIKE_NAMES:
            q = _sub(spec["above"], {"{S}": s})
            api[f"above_{s}"], dec[f"above_{s}"] = _clean_q(q), q
        extras = spec.get("extras_above") or {}
        allowed = set(K_STATE_FIELDS) | {f for b in spec.get("blocks_above", []) for f in K_BLOCKS[b]}
    for qid, q in extras.items():
        api[qid] = _clean_q(q)
    unknown = sorted({f for f in _FIELD.findall(json.dumps(api)) if f not in allowed})
    if unknown:
        raise ValueError(f"variant {spec['name']} ({task}) references unknown fields: {unknown}")
    for qid, q in dec.items():
        if q["type"] == "choice" and not (q.get("up_key") in q["criteria"] and q.get("down_key") in q["criteria"]):
            raise ValueError(f"variant {spec['name']}: choice question {qid} needs up_key/down_key among its options")
        if q["type"] == "score" and "level_shifts" in q:
            if task == "D":
                raise ValueError(f"variant {spec['name']}: nudge questions (level_shifts) are for strike tasks only")
            anchor = q.get("anchor") or ""
            if (len(q["level_shifts"]) != len(q["criteria"]) or anchor not in allowed
                    or not anchor.startswith(("har_p_above_", "baseline_p_above_"))):
                raise ValueError(f"variant {spec['name']}: nudge question {qid} needs one level_shift per level and "
                                 f"an anchor probability (har_p_above_* or baseline_p_above_*) from its state")
        elif q["type"] == "score" and len(q.get("level_values") or []) != len(q["criteria"]):
            raise ValueError(f"variant {spec['name']}: score question {qid} needs one level_value per level")
    return api, dec, {qid: q for qid, q in extras.items()}


def jev_state(rec: dict, spec: dict, claude: dict | None) -> dict | None:
    st = dict(rec["state"])
    blocks = spec.get("blocks_direction" if rec["task"] == "D" else "blocks_above", [])
    for b in blocks:
        if b == "claude":
            c = (claude or {}).get(rec["id"])
            if c is None:
                return None
            st.update(c)
        else:
            st.update(rec[b])
    return st


def decode(q: dict, ans: dict | None, state: dict | None = None) -> float | None:
    """Answer -> probability. noul: p(yes); choice: P(up)/(P(up)+P(down)); score: sum P(level) x level_value; or,
    for a NUDGE question (level_shifts + anchor), the anchor's probability moved in log-odds by
    sum P(level) x level_shift - Jev only says whether the evidence leans above or below a baseline it is shown."""
    if not ans:
        return None
    if q["type"] == "noul":
        return ans.get("p")
    probs = ans.get("probabilities") or {}
    if q["type"] == "choice":
        pu, pd = probs.get(q["up_key"], 0.0), probs.get(q["down_key"], 0.0)
        return pu / (pu + pd) if pu + pd > 0 else 0.5
    tot = sum(probs.values())
    if "level_shifts" in q:
        anchor = (state or {}).get(q["anchor"])
        if anchor is None or tot <= 0:
            return None
        shift = sum(probs.get(i, probs.get(str(i), 0.0)) * v for i, v in enumerate(q["level_shifts"])) / tot
        return _sigmoid(_logit(float(anchor)) + shift)
    # level keys are ints from the client but strings after a JSON round-trip through the cache
    return sum(probs.get(i, probs.get(str(i), 0.0)) * v for i, v in enumerate(q["level_values"])) / tot if tot > 0 else None


def extra_features(extras: dict, answers: dict) -> list[float]:
    out = []
    for qid, q in sorted(extras.items()):
        a = answers.get(qid) or {}
        if q["type"] == "noul":
            out.append(float(a.get("p", 0.5)))
        elif q["type"] == "choice":
            probs = a.get("probabilities") or {}
            out += [float(probs.get(k, 0.0)) for k in list(q["criteria"])[:-1]]   # last option is implied
        else:
            levels = len(q["criteria"])
            out.append(float(a.get("score", (levels - 1) / 2)) / max(1, levels - 1))
    return out


def _sim_str(t: dict) -> str:
    return "      -    " if t.get("bps_per_trade") is None else f"{t['bps_per_trade']:>+6.2f}/{t['n']:<4}"


def variant_hash(spec: dict, task: str) -> str:
    api = render(spec, task)[0]
    blocks = spec.get("blocks_direction" if task == "D" else "blocks_above", [])
    return hashlib.sha1(json.dumps([api, sorted(blocks)], sort_keys=True).encode()).hexdigest()[:8]


def load_variants(lab_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((lab_dir / "variants").glob("*.json")):
        spec = json.loads(p.read_text(encoding="utf-8"))
        out[spec["name"]] = spec
    return out


# ====================================================================================== asking
def _read_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:      # blank, or a line still being written by a running step
                continue
            out[r["id"]] = r
    return out


def run_jev(client, lab_dir: Path, task: str, spec: dict, states: list[dict], claude: dict | None,
            workers: int = 4, log=print, max_fail_streak: int | None = None) -> dict:
    """Ask Jev every state not already answered (answers are appended to a jsonl, so a rerun resumes). With
    `max_fail_streak`, give up (RuntimeError) after that many states in a row fail - e.g. the API key ran out -
    instead of retrying thousands of states for hours."""
    api, _, _ = render(spec, task)
    path = lab_dir / f"jev_{task}_{spec['name']}_{variant_hash(spec, task)}.jsonl"
    done = _read_jsonl(path)
    todo = [(r, s) for r in states if r["id"] not in done for s in [jev_state(r, spec, claude)] if s is not None]
    lock, stats = threading.Lock(), {"ok": 0, "fail": 0, "tokens": 0, "streak": 0, "last_error": None}
    abort = threading.Event()
    if not todo:
        return done

    def one(item):
        if abort.is_set():
            return
        rec, st = item
        for attempt in range(5):
            res = client.ask(st, api)
            if res.status == "ok":
                row = {"id": rec["id"], "answers": res.answers, "model": res.model}
                with lock:
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    done[rec["id"]] = row
                    stats["ok"] += 1
                    stats["streak"] = 0
                    stats["tokens"] += int((res.usage or {}).get("input_tokens", 0))
                    if stats["ok"] % 200 == 0:
                        log(f"    {spec['name']}/{task}: {stats['ok']}/{len(todo)}")
                return
            if abort.is_set():
                return
            time.sleep(1.5 * (attempt + 1) * (3 if "429" in str(res.error) else 1))
        with lock:
            stats["fail"] += 1
            stats["streak"] += 1
            stats["last_error"] = str(res.error)[:200]
            if max_fail_streak and stats["streak"] >= max_fail_streak:
                abort.set()

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    if abort.is_set():
        raise RuntimeError(f"Jev failed {stats['streak']} states in a row (last error: {stats['last_error']}); "
                           f"{stats['ok']} answered before stopping")
    log(f"    {spec['name']}/{task}: {stats['ok']} new answers, {stats['fail']} failed, "
        f"~${stats['tokens'] * 0.042 / 1e6:.3f}")
    return done


CLAUDE_D = """You forecast whether crypto perpetual futures prices will be HIGHER 5 and 15 minutes from now, for a
paper-trading research backtest. Nothing is traded; forecasts are scored with log loss on what actually happened.

Each item has an id and a numeric state: returns from 1 minute to 24 hours (bps), 1-minute volatility, the expected
1-sigma move over 5 and 15 minutes, how many of the last 15 one-minute bars closed up, volume versus its 4-hour
average, RSI, the gap to a 20-bar EMA, where price sits in its 4 h and 24 h ranges, funding and premium.
Return p_up_5m and p_up_15m for every id: the probability that the price then is above the price now.

Short-horizon moves are close to a coin flip. Be calibrated, not confident: probabilities far from 0.5 must be
earned by specific evidence. A strong recent move is information, not a guarantee that it continues."""

CLAUDE_K = """You forecast the probability that crypto perpetual futures prices end ABOVE given strike prices at
the next full hour, for a paper-trading research backtest. Nothing is traded. Forecasts are scored with log loss
against what actually happened, alongside an empirical volatility model.

Each item has an id and the coin's state: price, minutes to expiry, returns over 15 minutes to 24 hours, the
expected 1-sigma move to expiry (`expected_move_to_expiry_bps`), funding, premium, where the price sits in its
24-hour range, and three strikes with their distance from the price in bps. Return p_above_low, p_above_mid and
p_above_high for every id; they must not increase from low to high. Be calibrated; short-horizon moves are mostly noise."""


def _claude_models():
    from pydantic import BaseModel

    class DItem(BaseModel):
        id: str
        p_up_5m: float
        p_up_15m: float

    class DBatch(BaseModel):
        forecasts: list[DItem]

    class KItem(BaseModel):
        id: str
        p_above_low: float
        p_above_mid: float
        p_above_high: float

    class KBatch(BaseModel):
        forecasts: list[KItem]

    return DBatch, KBatch


def run_claude(client, model: str, lab_dir: Path, task: str, states: list[dict], max_usd: float,
               price_in: float = 4.0, price_out: float = 20.0, batch: int = 24, workers: int = 3,
               subsample=None, log=print) -> dict:
    path = lab_dir / f"claude_{task}.jsonl"
    done = _read_jsonl(path)
    todo = [r for r in states if r["id"] not in done and (subsample is None or subsample(r))]
    DBatch, KBatch = _claude_models()
    lock, spent = threading.Lock(), {"usd": 0.0, "calls": 0}
    cost_path = lab_dir / "claude_cost.json"
    try:
        spent["usd"] = json.loads(cost_path.read_text(encoding="utf-8")).get("usd", 0.0)
    except (OSError, ValueError):
        pass
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def one(group):
        with lock:
            if spent["usd"] >= max_usd:
                return
        payload = {r["id"]: r["state"] for r in group}
        try:
            resp = client.beta.messages.parse(
                model=model, max_tokens=12000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low"}, system=CLAUDE_D if task == "D" else CLAUDE_K,
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                output_format=DBatch if task == "D" else KBatch)
        except Exception as exc:  # network/API problems: skip this batch, it is retried on the next run
            log(f"    claude {task}: batch failed ({type(exc).__name__}: {str(exc)[:120]})")
            return
        u = resp.usage
        with lock:
            spent["usd"] += (u.input_tokens * price_in + u.output_tokens * price_out) / 1e6
            spent["calls"] += 1
            out = resp.parsed_output
            with path.open("a", encoding="utf-8") as fh:
                for f in (out.forecasts if out else []):
                    if f.id not in payload:
                        continue
                    p = ({"claude_p_up_5m": f.p_up_5m, "claude_p_up_15m": f.p_up_15m} if task == "D" else
                         {"claude_p_above_low": f.p_above_low, "claude_p_above_mid": f.p_above_mid,
                          "claude_p_above_high": f.p_above_high})
                    p = {k: round(min(0.999, max(0.001, v)), 3) for k, v in p.items()}
                    row = {"id": f.id, "p": p}
                    fh.write(json.dumps(row) + "\n")
                    done[f.id] = row
            cost_path.write_text(json.dumps({"usd": spent["usd"]}), encoding="utf-8")
            if spent["calls"] % 10 == 0:
                log(f"    claude {task}: {spent['calls']}/{len(groups)} calls, ${spent['usd']:.2f} so far")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, groups))
    log(f"    claude {task}: {len(done)} forecasts total, ${spent['usd']:.2f} spent (cap ${max_usd:g})")
    return done


def claude_fields(lab_dir: Path, task: str) -> dict:
    return {k: v["p"] for k, v in _read_jsonl(lab_dir / f"claude_{task}.jsonl").items()}


# ====================================================================================== statistics
def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


class Logit:
    """L2-regularised logistic regression (Newton steps) on standardised features. Pure Python."""

    def __init__(self, l2: float = 2.0):
        self.l2 = l2

    def fit(self, X: list[list[float]], y: list[float]) -> "Logit":
        cols = list(zip(*X))
        self.mu = [fmean(c) for c in cols]
        self.sd = [pstdev(c) or 1.0 for c in cols]
        Z = [self._z(x) for x in X]
        d = len(Z[0])
        w = [0.0] * d
        for _ in range(30):
            g = [self.l2 * w[k] if k < d - 1 else 0.0 for k in range(d)]
            H = [[(self.l2 if (a == b and a < d - 1) else 0.0) for b in range(d)] for a in range(d)]
            for z, yi in zip(Z, y):
                p = _sigmoid(sum(wk * zk for wk, zk in zip(w, z)))
                r, wt = p - yi, p * (1 - p)
                for a in range(d):
                    za = z[a]
                    g[a] += r * za
                    wa, Ha = wt * za, H[a]
                    for b in range(a, d):
                        Ha[b] += wa * z[b]
            for a in range(d):
                for b in range(a):
                    H[a][b] = H[b][a]
            step = _solve(H, g)
            w = [wk - sk for wk, sk in zip(w, step)]
            if max(abs(s) for s in step) < 1e-7:
                break
        self.w = w
        return self

    def _z(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)] + [1.0]

    def predict(self, x: list[float]) -> float:
        return _sigmoid(sum(wk * zk for wk, zk in zip(self.w, self._z(x))))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    ma, mb = fmean(a), fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def auc(p: list[float], y: list[int]) -> float | None:
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return None
    r = _ranks(p)
    return (sum(ri for ri, yi in zip(r, y) if yi) - pos * (pos - 1) / 2) / (pos * neg)


def _residual(target: list[float], X: list[list[float]]) -> list[float]:
    """OLS residual of target on X (+ intercept)."""
    Z = [x + [1.0] for x in X]
    d = len(Z[0])
    A = [[sum(z[a] * z[b] for z in Z) + (1e-9 if a == b else 0) for b in range(d)] for a in range(d)]
    beta = _solve(A, [sum(z[a] * t for z, t in zip(Z, target)) for a in range(d)])
    return [t - sum(bk * zk for bk, zk in zip(beta, z)) for t, z in zip(target, Z)]


def block_ci(diffs: list[float], blocks: list[int], n_boot: int = 2000, seed: int = 11) -> tuple[float, float]:
    """95% CI of the mean paired difference, resampling whole time blocks."""
    by: dict[int, list[float]] = {}
    for d, b in zip(diffs, blocks):
        by.setdefault(b, []).append(d)
    keys = list(by)
    if len(keys) < 4:
        return (float("nan"), float("nan"))
    sums, cnts = [sum(by[k]) for k in keys], [len(by[k]) for k in keys]
    rng, boots = random.Random(seed), []
    for _ in range(n_boot):
        idx = [rng.randrange(len(keys)) for _ in keys]
        boots.append(sum(sums[i] for i in idx) / sum(cnts[i] for i in idx))
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


# ====================================================================================== scoring
def _numeric_core(rec: dict) -> list[float]:
    st = rec["state"]
    return [float(st[k]) for k in D_STATE_FIELDS if k not in ("coin", "hour_of_day_utc")] + \
           [math.sin(st["hour_of_day_utc"] * math.pi / 12), math.cos(st["hour_of_day_utc"] * math.pi / 12)]


def stacked(des: list, others: list, feat, label, l2: float = 2.0, folds: int = 4) -> dict:
    """Predictions from a model fit on the design period: OUT-OF-FOLD for design items (contiguous time folds, so a
    model never scores the data it was fit on) and from the full design fit for everything else."""
    out = {}
    order = sorted(des, key=lambda r: r["ts"])
    size = math.ceil(len(order) / folds)
    for f in range(folds):
        test = order[f * size:(f + 1) * size]
        train = order[:f * size] + order[(f + 1) * size:]
        if not test or len(train) < 100:
            continue
        m = Logit(l2).fit([feat(r) for r in train], [label(r) for r in train])
        for r in test:
            out[r["id"]] = m.predict(feat(r))
    m = Logit(l2).fit([feat(r) for r in des], [label(r) for r in des])
    for r in others:
        out[r["id"]] = m.predict(feat(r))
    return out


def _trade_sim(p_design, fwd_design, p_hold, fwd_hold, cost: float) -> dict:
    """Pick the entry threshold on the design period, report the held-out result."""
    def run(ps, fw, th):
        pnl = [(f if p >= th else -f) - cost for p, f in zip(ps, fw) if p >= th or p <= 1 - th]
        return pnl
    best, best_th = None, None
    for th in (0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.66, 0.70):
        pnl = run(p_design, fwd_design, th)
        if len(pnl) >= 30 and (best is None or fmean(pnl) > best):
            best, best_th = fmean(pnl), th
    if best_th is None:
        return {"threshold": None, "design_bps": None, "n": 0, "bps_per_trade": None}
    hold = run(p_hold, fwd_hold, best_th)
    return {"threshold": best_th, "design_bps": round(best, 2), "n": len(hold),
            "bps_per_trade": round(fmean(hold), 2) if hold else None}


def score_direction(lab_dir: Path, states: list[dict], variants: dict, claude: dict) -> dict:
    cut = split_ts(states)
    report = {"split_ts": cut, "n_states": len(states), "horizons": {}}
    answers = {}
    for name, spec in variants.items():
        p = lab_dir / f"jev_D_{name}_{variant_hash(spec, 'D')}.jsonl"
        answers[name] = {k: v["answers"] for k, v in _read_jsonl(p).items()}
    for h in HORIZONS:
        hk = str(h)
        recs = [r for r in states if r["fwd"][hk] != 0]
        des = [r for r in recs if r["ts"] < cut]
        hold = [r for r in recs if r["ts"] >= cut]
        y = lambda rs: [1 if r["fwd"][hk] > 0 else 0 for r in rs]   # noqa: E731
        lab1 = lambda r: 1 if r["fwd"][hk] > 0 else 0               # noqa: E731
        qp = stacked(des, hold, _numeric_core, lab1, l2=L2_D)
        forecasters: dict[str, dict] = {"base_0.5": {r["id"]: 0.5 for r in recs}, "quant": qp}
        forecasters["anchor"] = {r["id"]: r["anchor"][f"baseline_p_up_{h}m"] for r in recs}
        cl = {r["id"]: claude[r["id"]][f"claude_p_up_{h}m"] for r in recs if r["id"] in claude}
        if cl:
            forecasters["claude"] = cl
        stack_of = {}
        for name, spec in variants.items():
            api, dec, extras = render(spec, "D")
            qid = f"p_up_{h}m"
            got = {rid: a for rid, a in answers[name].items()}
            direct = {r["id"]: decode(dec[qid], got.get(r["id"], {}).get(qid)) for r in recs if r["id"] in got}
            direct = {k: v for k, v in direct.items() if v is not None}
            forecasters[f"jev:{name}"] = direct
            feats = {rid: [_logit(direct[rid])] + extra_features(extras, got[rid]) for rid in direct}
            d_ids = [r for r in des if r["id"] in feats]
            h_ids = [r for r in hold if r["id"] in feats]
            if len(d_ids) >= 200:
                forecasters[f"jevstack:{name}"] = stacked(d_ids, h_ids, lambda r: feats[r["id"]], lab1, l2=L2_D)
                forecasters[f"quant+jev:{name}"] = stacked(d_ids, h_ids, lambda r: _numeric_core(r) + feats[r["id"]], lab1,
                                                           l2=L2_D)
                stack_of[name] = f"quant+jev:{name}"
        if cl:
            d_ids = [r for r in des if r["id"] in cl]
            if len(d_ids) >= 200:
                forecasters["quant+claude"] = stacked(d_ids, [r for r in hold if r["id"] in cl],
                                                      lambda r: _numeric_core(r) + [_logit(cl[r["id"]])], lab1, l2=L2_D)
        rows = {}
        for who, fp in forecasters.items():
            out = {}
            for period, rs in (("design", des), ("holdout", hold)):
                rs = [r for r in rs if r["id"] in fp]
                if len(rs) < 30:
                    continue
                ps, ys = [fp[r["id"]] for r in rs], y(rs)
                fw, tr = [r["fwd"][hk] for r in rs], [r["trail"][hk] for r in rs]
                qq = [qp[r["id"]] for r in rs]
                ll = [logloss(p, yy) for p, yy in zip(ps, ys)]
                llq = [logloss(p, yy) for p, yy in zip(qq, ys)]
                diffs_q = [b - a for a, b in zip(ll, llq)]
                diffs_h = [math.log(2) - a for a in ll]
                blocks = [r["ts"] // BLOCK_S for r in rs]
                trail_X = [[r["state"]["return_1m_bps"], r["state"]["return_5m_bps"], r["state"]["return_15m_bps"],
                            r["state"]["return_60m_bps"]] for r in rs]
                resid = _residual([_logit(p) for p in ps], trail_X) if len(set(ps)) > 1 else [0.0] * len(ps)
                m = {"n": len(rs), "logloss": round(fmean(ll), 5), "gain_vs_half": round(fmean(diffs_h), 5),
                     "gain_vs_quant": round(fmean(diffs_q), 5), "auc": auc(ps, ys),
                     "ic": round(spearman(ps, fw), 4) if len(set(ps)) > 1 else 0.0,
                     "echo": round(spearman(ps, tr), 4) if len(set(ps)) > 1 else 0.0,
                     "partial_ic": round(spearman(resid, fw), 4) if len(set(resid)) > 1 else 0.0,
                     "mean_p": round(fmean(ps), 3), "sd_p": round(pstdev(ps), 3)}
                if m["auc"] is not None:
                    m["auc"] = round(m["auc"], 4)
                if period == "holdout":
                    m["gain_vs_half_ci"] = [round(x, 5) for x in block_ci(diffs_h, blocks)]
                    m["gain_vs_quant_ci"] = [round(x, 5) for x in block_ci(diffs_q, blocks)]
                out[period] = m
            if "design" in out and "holdout" in out:
                dsg = [r for r in des if r["id"] in fp]
                hld = [r for r in hold if r["id"] in fp]
                for label, cost in (("taker", COST_TAKER_BPS), ("maker", COST_MAKER_BPS)):
                    out[f"trade_{label}"] = _trade_sim([fp[r["id"]] for r in dsg], [r["fwd"][hk] for r in dsg],
                                                       [fp[r["id"]] for r in hld], [r["fwd"][hk] for r in hld], cost)
            rows[who] = out
        # paired, out-of-fold: how much each variant's answers add to the quant model on the SAME design states
        ranking = sorted((n for n in stack_of if "design" in rows.get(stack_of[n], {})),
                         key=lambda n: -rows[stack_of[n]]["design"]["gain_vs_quant"])
        report["horizons"][hk] = {"forecasters": rows, "design_ranking": ranking,
                                  "winner": ranking[0] if ranking else None}
    return report


def score_hourly(lab_dir: Path, states: list[dict], variants: dict, claude: dict, task: str = "K") -> dict:
    cut = split_ts(states)
    by_id = {r["id"]: r for r in states}
    rows_by_id: dict[str, list[dict]] = {}
    for r in states:
        rows_by_id[r["id"]] = [{"hour": r["ts"], "coin": r["coin"], "strike": s, "y": r["y"][j],
                                "model": r["model"][j], "model_emp": r["model_emp"][j]}
                               for j, s in enumerate(STRIKE_NAMES)]
        if "model_har" in r:
            for j in range(len(STRIKE_NAMES)):
                rows_by_id[r["id"]][j]["model_har"] = r["model_har"][j]
        if r["id"] in claude:
            for j, s in enumerate(STRIKE_NAMES):
                rows_by_id[r["id"]][j]["claude"] = claude[r["id"]][f"claude_p_above_{s}"]
    names = []
    for name, spec in variants.items():
        api, dec, extras = render(spec, task)
        got = {k: v["answers"] for k, v in
               _read_jsonl(lab_dir / f"jev_{task}_{name}_{variant_hash(spec, task)}.jsonl").items()}
        if not got:
            continue
        names.append(name)
        feats = {}
        for rid, a in got.items():
            if rid not in rows_by_id:
                continue
            st = jev_state(by_id[rid], spec, claude)
            ps = [decode(dec[f"above_{s}"], a.get(f"above_{s}"), st) for s in STRIKE_NAMES]
            if any(p is None for p in ps):
                continue
            ex = extra_features(extras, a)
            for j, row in enumerate(rows_by_id[rid]):
                row[f"jev:{name}"] = ps[j]
                feats[(rid, j)] = [_logit(row["model_emp"]), _logit(ps[j])] + ex
        items = [{"id": k, "ts": int(k[0].rsplit(":", 1)[1])} for k in feats]
        des = [x for x in items if x["ts"] < cut]
        if len(des) >= 200:
            preds = stacked(des, [x for x in items if x["ts"] >= cut], lambda x: feats[x["id"]],
                            lambda x: rows_by_id[x["id"][0]][x["id"][1]]["y"], l2=L2_K)
            for (rid, j), p in preds.items():
                rows_by_id[rid][j][f"jevstack:{name}"] = p
    allrows = [x for rs in rows_by_id.values() for x in rs]
    design = [x for x in allrows if x["hour"] < cut]
    hold = [x for x in allrows if x["hour"] >= cut]
    hold_hours = len({x["hour"] for x in hold})
    whos = ["model", "model_emp"] + (["model_har"] if any("model_har" in x for x in allrows) else []) + \
           (["claude"] if claude else []) + [f"{kind}:{n}" for n in names for kind in ("jev", "jevstack")]
    out = {"split_ts": cut, "n_states": len(states), "hold_hours": hold_hours, "forecasters": {}}
    for who in whos:
        out["forecasters"][who] = {
            "design": HourlyTest.score(design, who, n_boot=1000, target_hours=10 ** 9),
            "holdout": HourlyTest.score(hold, who, n_boot=2000, target_hours=hold_hours)}
    ranked = sorted(((n, k) for n in names for k in ("jev", "jevstack")
                     if out["forecasters"][f"{k}:{n}"]["design"].get("primary_gain") is not None),
                    key=lambda nk: -out["forecasters"][f"{nk[1]}:{nk[0]}"]["design"]["primary_gain"])
    out["design_ranking"] = [f"{k}:{n}" for n, k in ranked]
    out["winner"] = out["design_ranking"][0] if ranked else None
    return out


def summarize(report: dict) -> str:
    lines = []
    for hk, h in report.get("D", {}).get("horizons", {}).items():
        lines.append(f"\nTask D - P(up in {hk} min), held-out period (gain = log-loss improvement; + is better)")
        lines.append(f"  {'forecaster':34} {'n':>5} {'vs 0.5':>8} {'vs quant':>9} {'95% CI vs quant':>19} "
                     f"{'AUC':>6} {'IC':>6} {'echo':>6} {'pIC':>6} {'sd_p':>5}  taker bps/n  maker bps/n")
        for who, m in h["forecasters"].items():
            ho = m.get("holdout")
            if not ho:
                continue
            ci = ho.get("gain_vs_quant_ci", [float("nan")] * 2)
            lines.append(f"  {who:34} {ho['n']:>5} {ho['gain_vs_half']:>8.4f} {ho['gain_vs_quant']:>9.4f} "
                         f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] {ho['auc'] or 0:>6.3f} {ho['ic']:>6.3f} {ho['echo']:>6.3f} "
                         f"{ho['partial_ic']:>6.3f} {ho['sd_p']:>5.3f}  {_sim_str(m.get('trade_taker', {}))}  "
                         f"{_sim_str(m.get('trade_maker', {}))}")
        lines.append(f"  design-period ranking (quant+jev stacks): {h['design_ranking']} -> winner {h['winner']}")
    pc = report.get("D", {}).get("per_coin")
    if pc:
        lines.append("\nPer coin, held-out, Jev v2 (present_stretch_reversion) at 15 min; reversal = Spearman(last 15 min, next 15 min)")
        lines.append(f"  {'coin':9} {'n':>4} {'reversal':>9} {'7d cont.':>8} {'Jev IC':>7} {'vs 0.5':>8} {'trades':>6} {'gross bps':>9} "
                     f"{'net@3bps':>8} {'net@7bps':>8}")
        for c, m in sorted(pc.items(), key=lambda kv: -(kv[1].get("gross_bps") or -99)):
            gb = m.get("gross_bps")
            lines.append(f"  {c:9} {m['n']:>4} {m['autocorr']:>+9.3f} {m['continuation']:>8.3f} {m['ic']:>+7.3f} "
                         f"{m['gain_vs_half']:>+8.4f} {m['trades']:>6} {(gb if gb is not None else float('nan')):>+9.2f} "
                         f"{(gb - 3 if gb is not None else float('nan')):>+8.2f} {(gb - 7 if gb is not None else float('nan')):>+8.2f}")
    for key, label in (("K", "next hour"), ("K15", "15 minutes")):
        k = report.get(key)
        if not k:
            continue
        lines.append(f"\nTask {key} - P(above strike in {label}), held-out {k['hold_hours']} rounds, mid strike vs model_emp")
        lines.append(f"  {'forecaster':34} {'n':>5} {'logloss':>8} {'ATM gain':>9} {'97.5% one-sided CI':>21} verdict")
        for who, m in k["forecasters"].items():
            ho = m["holdout"]
            if not ho.get("n"):
                continue
            ci = ho.get("primary_ci") or [float("nan")] * 2
            g = ho.get("primary_gain")
            lines.append(f"  {who:34} {ho['n']:>5} {ho['logloss']:>8.4f} {(g if g is not None else float('nan')):>9.4f} "
                         f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] {ho.get('verdict', 'baseline')}")
        lines.append(f"  design-period ranking: {k['design_ranking']} -> winner {k['winner']}")
    return "\n".join(lines)


def per_coin_direction(lab_dir: Path, states: list[dict], variants: dict, name: str = "present_stretch_reversion",
                       h: str = "15", th: float = 0.54, holdout_only: bool = True) -> dict:
    """Per coin (held-out period, or all states when nothing was fitted): does the reversal hold there, does Jev v2
    read it, and what would trading it gross?"""
    spec = variants.get(name)
    if spec is None:
        return {}
    _, dec, _ = render(spec, "D")
    ans = {k: v["answers"] for k, v in _read_jsonl(lab_dir / f"jev_D_{name}_{variant_hash(spec, 'D')}.jsonl").items()}
    cut = split_ts(states) if holdout_only else float("-inf")
    out = {}
    for coin in sorted({r["coin"] for r in states}):
        rs = [r for r in states if r["coin"] == coin and r["ts"] >= cut and r["fwd"][h] != 0]
        ps = {r["id"]: decode(dec[f"p_up_{h}m"], ans.get(r["id"], {}).get(f"p_up_{h}m")) for r in rs}
        rs_p = [r for r in rs if ps.get(r["id"]) is not None]
        if len(rs_p) < 30:
            continue
        p = [ps[r["id"]] for r in rs_p]
        fw = [r["fwd"][h] for r in rs_p]
        y = [1 if f > 0 else 0 for f in fw]
        trades = [(f if q >= th else -f) for q, f in zip(p, fw) if q >= th or q <= 1 - th]
        out[coin] = {"n": len(rs_p), "autocorr": round(spearman([r["trail"][h] for r in rs_p], fw), 3),
                     "continuation": round(fmean(r["stats"][f"continuation_rate_{h}m_7d"] for r in rs_p), 3),
                     "ic": round(spearman(p, fw), 3) if len(set(p)) > 1 else 0.0,
                     "gain_vs_half": round(fmean(math.log(2) - logloss(q, yy) for q, yy in zip(p, y)), 4),
                     "trades": len(trades), "gross_bps": round(fmean(trades), 2) if trades else None}
    return out
