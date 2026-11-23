"""Risk engine: deterministic hard limits checked before every order. No model can override
them, and every check uses numbers the engine measures itself (book, clock, P&L)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .features import sigma_bps
from .market import PerpView, walk_book


@dataclass
class Check:
    name: str
    ok: bool
    value: str
    limit: str

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "value": self.value, "limit": self.limit}


@dataclass
class RiskDecision:
    approved: bool
    side: int
    qty: float = 0.0
    notional: float = 0.0
    stop: float | None = None
    target: float | None = None
    stop_bps: float = 0.0
    target_bps: float = 0.0
    slip_bps: float = 0.0
    cost_bps: float = 0.0
    checks: list[Check] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"approved": self.approved, "side": self.side, "qty": self.qty, "notional": round(self.notional, 2),
                "stop": self.stop, "target": self.target, "stop_bps": round(self.stop_bps, 2),
                "target_bps": round(self.target_bps, 2), "slip_bps": round(self.slip_bps, 3),
                "cost_bps": round(self.cost_bps, 2), "reason": self.reason,
                "checks": [c.to_dict() for c in self.checks]}


class RiskEngine:
    def __init__(self, risk: dict, taker_bps: float, maker_bps: float, extra_slip_bps: float):
        self.r = risk
        self.taker_bps = taker_bps
        self.maker_bps = maker_bps
        self.extra_slip_bps = extra_slip_bps

    def round_trip_bps(self, slip_bps: float, maker_entry: bool = False) -> float:
        """Worst case: taker in (or a resting maker entry: no slippage in), taker out (a stop) with slippage."""
        if maker_entry:
            return self.maker_bps + self.taker_bps + max(0.0, slip_bps) + self.extra_slip_bps
        return 2 * self.taker_bps + 2 * (max(0.0, slip_bps) + self.extra_slip_bps)

    def evaluate_entry(self, *, book, view: PerpView, feats: dict, side: int, size_mult: float, marks: dict,
                       now: float, horizon_s: float, decision_latency_ms: float | None, errors_in_row: int,
                       engine_paused: bool, maker_entry: bool = False, bet_mult: float = 1.0,
                       sigma_floor_bps: float | None = None) -> RiskDecision:
        r, coin = self.r, view.coin
        checks: list[Check] = []
        equity = book.equity(marks)
        mid = view.mid or 0.0

        killed = engine_paused or book.killed
        checks.append(Check("Kill switch", not killed, "ENGAGED" if killed else "off", "off"))
        day = book.day_pnl_pct(marks)
        if r["max_daily_loss_pct"] > 0:
            checks.append(Check("Daily loss", day > -r["max_daily_loss_pct"] * 100 and not book.daily_stop,
                                f"{day:+.2f}%", f"> -{r['max_daily_loss_pct'] * 100:.1f}%"))
        else:  # 0 = no daily stop (paper mode); the drawdown kill switch still applies
            checks.append(Check("Daily loss", True, f"{day:+.2f}%", "off"))
        dd = book.drawdown_pct(marks)
        checks.append(Check("Drawdown", dd < r["max_drawdown_pct"] * 100, f"{dd:.2f}%", f"< {r['max_drawdown_pct'] * 100:.0f}%"))
        age_ms = feats.get("data_age_s", 0.0) * 1000
        checks.append(Check("Data age", age_ms <= r["max_data_age_ms"], f"{age_ms:.0f}ms", f"≤ {r['max_data_age_ms']}ms"))
        if decision_latency_ms is not None:
            checks.append(Check("Decision latency", decision_latency_ms <= r["max_decision_latency_ms"],
                                f"{decision_latency_ms:.0f}ms", f"≤ {r['max_decision_latency_ms']}ms"))
        checks.append(Check("Feed errors in a row", errors_in_row < r["max_consecutive_errors"], str(errors_in_row),
                            f"< {r['max_consecutive_errors']}"))
        pending = getattr(book, "pending", {})              # resting maker entries count as positions
        dup = coin in book.positions or coin in pending
        checks.append(Check("Duplicate", not dup, ("open" if coin in book.positions else "resting order") if dup else "none",
                            "one per market"))
        n_open = len(book.positions) + len(pending)
        checks.append(Check("Open positions", n_open < r["max_open_positions"], str(n_open),
                            f"< {r['max_open_positions']}"))
        cool = book.cooldown_until.get(coin, 0) - now
        checks.append(Check("Cooldown", cool <= 0, f"{cool:.0f}s left" if cool > 0 else "clear",
                            f"{r['cooldown_s']}s after a stop"))
        n_entries = book.entries_last(now)
        checks.append(Check("Entry rate", n_entries < r["max_entries_per_min"], f"{n_entries}/min",
                            f"< {r['max_entries_per_min']}/min"))
        checks.append(Check("Direction allowed", side > 0 or r["allow_short"], "short" if side < 0 else "long",
                            "long/short" if r["allow_short"] else "long only"))
        spread = view.spread_bps or 0.0
        checks.append(Check("Spread", spread <= r["max_spread_bps"], f"{spread:.2f}bps", f"≤ {r['max_spread_bps']}bps"))

        sig = sigma_bps(feats, horizon_s)
        # the 1-s book estimate (last 5 min) ran well below realised 15-min moves on volatile alts (NEAR 48 vs 84 bps,
        # PUMP 41 vs 78): floor it with the signal's candle-based estimate so stops and sizes match the real risk
        if sigma_floor_bps and math.isfinite(sigma_floor_bps) and sigma_floor_bps > sig:
            sig = float(sigma_floor_bps)
        stop_bps = max(r["min_stop_bps"], r["stop_sigma"] * sig)
        target_bps = max(r["min_target_bps"], r["target_sigma"] * sig)
        stop = mid * (1 - side * stop_bps / 1e4) if mid else None
        target = mid * (1 + side * target_bps / 1e4) if mid else None
        checks.append(Check("Stop & target set", stop is not None and stop > 0, f"SL {stop_bps:.1f} / TP {target_bps:.1f}bps",
                            "required"))

        # Bet multiplier: scales the $ risked per trade, and the leverage caps with it (never below the base caps,
        # never above the ceilings). The drawdown kill switch is NOT scaled: bigger bets reach it sooner.
        bet_mult = max(0.1, float(bet_mult)) if math.isfinite(float(bet_mult)) else 1.0
        # below 1x the caps shrink too (so 0.5x always halves the bet); above 1x they grow up to the ceilings
        pos_cap = min(r["max_position_leverage"] * bet_mult, r.get("leverage_ceiling_position", r["max_position_leverage"]))
        tot_cap = min(r["max_total_leverage"] * bet_mult, r.get("leverage_ceiling_total", r["max_total_leverage"]))
        if bet_mult != 1.0:
            checks.append(Check("Bet size", True, f"{bet_mult:.2f}x", f"caps {pos_cap:.1f}x / {tot_cap:.1f}x"))
        qty = notional = slip = cost = 0.0
        if mid > 0 and equity > 0:
            notional = equity * r["risk_per_trade"] * max(0.0, min(1.0, size_mult)) * bet_mult / (stop_bps / 1e4)
            notional = min(notional, equity * pos_cap)
            committed = book.exposure(marks) + (book.pending_notional() if hasattr(book, "pending_notional") else 0.0)
            room = equity * tot_cap - committed
            if room <= 0:   # say "leverage" when there is no room left, not "book too thin"
                checks.append(Check("Leverage", False, f"{committed / equity:.2f}x already committed", f"≤ {tot_cap:g}x"))
            notional = max(0.0, min(notional, room))
            qty = notional / mid
            levels = view.asks if side > 0 else view.bids
            avg, filled = walk_book(levels, qty)
            depth_ok = filled >= qty * 0.999 and qty > 0
            slip = abs(avg / mid - 1) * 1e4 if depth_ok else float("inf")
            checks.append(Check("Book depth / slippage", depth_ok and slip <= r["max_slippage_bps"],
                                f"{slip:.2f}bps" if depth_ok else "book too thin", f"≤ {r['max_slippage_bps']}bps"))
            cost = self.round_trip_bps(slip if depth_ok else 0.0, maker_entry)
            need = cost * r["min_edge_multiple"]
            checks.append(Check("Fees vs target", target_bps >= need, f"TP {target_bps:.1f}bps",
                                f"≥ {need:.1f}bps ({r['min_edge_multiple']:g}× {cost:.1f} cost)"))
            lev_after = (committed + notional) / equity
            checks.append(Check("Leverage", lev_after <= tot_cap + 1e-9, f"{lev_after:.2f}x", f"≤ {tot_cap:g}x"))
            checks.append(Check("Min order", notional >= r["min_order_usd"], f"${notional:,.0f}",
                                f"≥ ${r['min_order_usd']:,.0f}"))

        failed = [c for c in checks if not c.ok]
        approved = not failed
        reason = "All risk checks passed" if approved else f"{failed[0].name}: {failed[0].value} (limit {failed[0].limit})"
        dec = 6 if mid < 10 else 4 if mid < 1000 else 2
        return RiskDecision(approved, side, round(qty, 8), notional,
                            None if stop is None else round(stop, dec), None if target is None else round(target, dec),
                            stop_bps, target_bps, slip, cost, checks, reason)

    def guard(self, book, marks: dict) -> tuple[str | None, str | None]:
        dd = book.drawdown_pct(marks)
        if dd >= self.r["max_drawdown_pct"] * 100:
            return "kill", f"Max drawdown {dd:.2f}% ≥ {self.r['max_drawdown_pct'] * 100:.0f}%"
        day = book.day_pnl_pct(marks)
        if self.r["max_daily_loss_pct"] > 0 and day <= -self.r["max_daily_loss_pct"] * 100:
            return "daily", f"Daily loss {day:.2f}% hit the {self.r['max_daily_loss_pct'] * 100:.1f}% limit"
        return None, None
