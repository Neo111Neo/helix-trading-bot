"""The compact market state Jev sees: numbers only, about 250 tokens, nothing from the future."""

from __future__ import annotations

import json


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def build_state(f: dict, position: dict, account: dict, system: dict, outlook: dict | None = None) -> dict:
    """position: side (1/0/-1), upnl_bps, hold_s. account: day_pnl_pct, drawdown_pct.
    system: jev_latency_ms (recent average), decision_interval_ms.
    outlook: sigma_bps (expected 1-sigma move over the horizon) and band_bps (the neutral band the
    direction answer is scored against), so Jev answers the same question calibration grades."""
    state = {
        "spread_bps": _r(f["spread_bps"], 3),
        "micro_vs_mid_bps": _r(f["micro_vs_mid_bps"], 3),
        "return_5s_bps": _r(f["ret_5s"]),
        "return_15s_bps": _r(f["ret_15s"]),
        "return_60s_bps": _r(f["ret_60s"]),
        "return_300s_bps": _r(f["ret_300s"]),
        "return_15m_bps": _r(f.get("ret_15m", 0.0)),
        "return_60m_bps": _r(f.get("ret_60m", 0.0)),
        "vol_1s_bps": _r(f["vol_1s_bps"], 3),
        "vol_60s_vs_300s": _r(f["vol_ratio"]),
        "book_imbalance_top5": _r(f["imbalance_5"], 3),
        "book_imbalance_top20": _r(f["imbalance_20"], 3),
        "depth_10bps_usd_k": _r(f["depth_10bps_usd"] / 1000, 1),
        "depth_vs_5m_median": _r(f["depth_ratio"]),
        "aggr_buy_share_15s": _r(f["flow_15s"], 3),
        "aggr_buy_share_60s": _r(f["flow_60s"], 3),
        "trades_per_s_15s": _r(f["trade_rate_15s"]),
        "trade_rate_vs_5m": _r(f["trade_rate_ratio"]),
        "big_trade_share_60s": _r(f["big_trade_share_60s"], 3),
        "funding_hr_bps": _r(f["funding_hr_bps"], 4),
        "premium_bps": _r(f["premium_bps"]),
        "trend_1m_ema_gap_bps": _r(f["trend_1m_bps"]),
        "rsi14_1m": _r(f["rsi14_1m"], 1),
        "position_side": int(position.get("side", 0)),
        "position_upnl_bps": _r(position.get("upnl_bps", 0.0), 1),
        "position_hold_s": int(position.get("hold_s", 0)),
        "account_day_pnl_pct": _r(account.get("day_pnl_pct", 0.0), 3),
        "account_drawdown_pct": _r(account.get("drawdown_pct", 0.0), 3),
        "jev_latency_ms": int(system.get("jev_latency_ms", 0)),
    }
    if outlook:
        state["typical_move_bps"] = _r(outlook["sigma_bps"], 1)
        state["neutral_band_bps"] = _r(outlook["band_bps"], 1)
    return state


def estimate_tokens(state: dict) -> int:
    return int(len(json.dumps(state, separators=(",", ":"))) / 3.5) + 1
