"""Paper perp execution on live order books.

Market orders walk the real L2 book (so size moves price like it would live), take-profits
rest as maker limits, stops and time exits go out as market orders, fees are charged on
every fill, and hourly funding is applied to open positions. No live broker exists here.

Entries can also rest as post-only maker limits at the touch (place_entry). Such an order fills
only when the market trades THROUGH its price (the best ask drops below a resting bid, or the
best bid rises above a resting offer). Fills at exactly our price are ignored because queue
position is unknown, so maker fills here are, if anything, rarer than live. That also keeps
adverse selection in the simulation: a resting bid fills when price is falling."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .market import PerpView, walk_book


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


EXIT_KINDS = (("time_maker", "15-min exit, maker"), ("time_market", "15-min exit, market"),
              ("target", "Take-profit"), ("stop", "Stop loss"), ("other", "Signal / other"))


def exit_kind(reason: str) -> str:
    if reason.startswith("Maker exit"):
        return "time_maker"
    if reason.startswith("Max hold"):
        return "time_market"
    if reason.startswith("Take profit"):
        return "target"
    if reason.startswith("Stop loss"):
        return "stop"
    return "other"


def exit_breakdown(trades: list) -> list[dict]:
    """How bets end, with the price move (gross) separated from fees, per exit type. 'Max hold' trades are what is
    LEFT after the take-profit removes the winners and the maker exit fills the trades drifting our way, so their
    average alone overstates what the 15-minute timer costs; the time exits together are the fair comparison."""
    groups: dict[str, list] = {k: [] for k, _ in EXIT_KINDS}
    for t in trades:
        groups[exit_kind(t.get("reason", ""))].append(t)
    out = []
    for kind, label in EXIT_KINDS:
        ts = groups[kind]
        if not ts:
            continue
        notional = [t["qty"] * t["entry"] or 1.0 for t in ts]
        n = len(ts)
        out.append({
            "kind": kind, "label": label, "n": n, "wins": sum(1 for t in ts if t["pnl"] > 0),
            "gross_bps": round(sum(t["gross"] / x * 1e4 for t, x in zip(ts, notional)) / n, 2),
            "fees_bps": round(sum(t["fees"] / x * 1e4 for t, x in zip(ts, notional)) / n, 2),
            "net_bps": round(sum(t["pnl_bps"] for t in ts) / n, 2),
            "pnl": round(sum(t["pnl"] for t in ts), 2), "fees": round(sum(t["fees"] for t in ts), 2),
        })
    return out


@dataclass
class Position:
    coin: str
    side: int  # 1 long, -1 short
    qty: float
    entry: float  # average fill price
    stop: float
    target: float
    opened: float
    entry_fee: float
    entry_slip_bps: float = 0.0
    funding_paid: float = 0.0
    last_funding_hour: int = 0
    reason: str = ""
    exit_px: float = 0.0          # resting maker exit near the horizon (0 = none)
    exit_placed: float = 0.0
    clock_start: float = 0.0      # when the holding clock starts: the signal's minute (0 = the fill time)

    def age(self, ts: float) -> float:
        """Time since the clock start: the Jev Lab measured the 15-minute move from the SIGNAL minute, and a maker
        entry fills ~45 s (up to ~150 s) later, so a clock started at the fill held each trade past the tested
        window."""
        return ts - (self.clock_start or self.opened)

    def upnl(self, px: float) -> float:
        return self.side * self.qty * (px - self.entry)


@dataclass
class PerpPaperBook:
    name: str
    starting_cash: float
    taker_bps: float
    maker_bps: float
    extra_slip_bps: float = 0.0
    cash: float = 0.0
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # [ts, equity]
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    day_key: str = ""
    day_start_equity: float = 0.0
    killed: bool = False
    kill_reason: str | None = None
    daily_stop: bool = False
    cooldown_until: dict = field(default_factory=dict)
    last_marks: dict = field(default_factory=dict)
    funnel: dict = field(default_factory=lambda: {"decisions": 0, "signals": 0, "risk_pass": 0, "fills": 0, "exits": 0})
    entry_times: deque = field(default_factory=lambda: deque(maxlen=500))
    pending: dict = field(default_factory=dict)  # coin -> resting maker entry (plain dict, persisted)
    maker_stats: dict = field(default_factory=lambda: {"placed": 0, "filled": 0, "expired": 0})

    def __post_init__(self):
        if not self.cash and not self.positions and not self.trades:
            self.cash = self.starting_cash
        self.peak_equity = self.peak_equity or self.starting_cash
        self.day_start_equity = self.day_start_equity or self.starting_cash
        if not isinstance(self.entry_times, deque):
            self.entry_times = deque(self.entry_times or [], maxlen=500)

    # ---- valuation -------------------------------------------------------------------
    def _px(self, coin: str, marks: dict) -> float:
        return marks.get(coin) or self.last_marks.get(coin) or self.positions[coin].entry

    def equity(self, marks: dict) -> float:
        return self.cash + sum(p.upnl(self._px(c, marks)) for c, p in self.positions.items())

    def exposure(self, marks: dict) -> float:
        return sum(p.qty * self._px(c, marks) for c, p in self.positions.items())

    def leverage(self, marks: dict) -> float:
        eq = self.equity(marks)
        return self.exposure(marks) / eq if eq > 0 else 0.0

    def drawdown_pct(self, marks: dict) -> float:
        eq = self.equity(marks)
        peak = max(self.peak_equity, eq)
        return (peak - eq) / peak * 100 if peak else 0.0

    def day_pnl_pct(self, marks: dict) -> float:
        base = self.day_start_equity or self.starting_cash
        return (self.equity(marks) / base - 1) * 100

    def entries_last(self, now: float, window: float = 60.0) -> int:
        return sum(1 for t in self.entry_times if now - t <= window)

    # ---- fills -----------------------------------------------------------------------
    def market_fill(self, view: PerpView, buying: bool, qty: float) -> tuple[float, float, float]:
        """(price, filled_qty, slippage_bps vs mid) for a taker order walking the book."""
        levels = view.asks if buying else view.bids
        avg, filled = walk_book(levels, qty)
        if filled <= 0:
            return 0.0, 0.0, 0.0
        sign = 1 if buying else -1
        avg *= 1 + sign * self.extra_slip_bps / 1e4  # latency between decision and fill
        mid = view.mid or avg
        return avg, filled, sign * (avg / mid - 1) * 1e4

    def open(self, view: PerpView, side: int, qty: float, stop: float, target: float, ts: float,
             reason: str = "", clock_start: float = 0.0) -> Position | None:
        coin = view.coin
        if coin in self.positions:
            raise ValueError(f"{self.name}: position already open in {coin}")
        px, filled, slip = self.market_fill(view, buying=side > 0, qty=qty)
        if filled < qty * 0.999:
            return None  # book too thin for the full size: no partial positions
        fee = filled * px * self.taker_bps / 1e4
        self.cash -= fee
        self.fees_paid += fee
        pos = Position(coin, side, filled, px, stop, target, ts, fee, slip, 0.0, int(ts // 3600), reason,
                       clock_start=min(clock_start, ts) if clock_start else 0.0)
        self.positions[coin] = pos
        self.entry_times.append(ts)
        return pos

    # ---- resting maker entries ---------------------------------------------------------
    def place_entry(self, view: PerpView, side: int, qty: float, stop_bps: float, target_bps: float, ts: float,
                    ttl_s: float, reason: str = "", clock_start: float = 0.0) -> dict | None:
        """Post-only limit at the touch: our bid = best bid (long) or our offer = best ask (short). clock_start (the
        signal's minute) is carried to the position so its holding clock matches the forecast window."""
        coin = view.coin
        if coin in self.positions or coin in self.pending or not view.bids or not view.asks:
            return None
        price = view.best_bid if side > 0 else view.best_ask
        order = {"coin": coin, "side": side, "qty": qty, "price": price, "stop_bps": stop_bps,
                 "target_bps": target_bps, "placed": ts, "expires": ts + ttl_s, "reason": reason,
                 "clock_start": min(clock_start, ts) if clock_start else 0.0}
        self.pending[coin] = order
        self.maker_stats["placed"] += 1
        return order

    def pending_notional(self) -> float:
        return sum(o["qty"] * o["price"] for o in self.pending.values())

    def cancel_entry(self, coin: str) -> dict | None:
        return self.pending.pop(coin, None)

    def check_entry(self, view: PerpView, ts: float) -> tuple[str, Position | dict | None]:
        """-> ("filled", Position) | ("expired", order) | ("resting", order) | ("none", None)."""
        o = self.pending.get(view.coin)
        if o is None:
            return "none", None
        if view.coin in self.positions:
            self.pending.pop(view.coin)
            return "expired", o
        if ts >= o["expires"]:          # expiry first: after a restart or feed gap, a dead order must not fill late
            self.pending.pop(view.coin)
            self.maker_stats["expired"] += 1
            return "expired", o
        through = bool(view.bids and view.asks) and (
            (o["side"] > 0 and view.best_ask < o["price"]) or (o["side"] < 0 and view.best_bid > o["price"]))
        if through:
            self.pending.pop(view.coin)
            px, side = o["price"], o["side"]
            fee = o["qty"] * px * self.maker_bps / 1e4
            self.cash -= fee
            self.fees_paid += fee
            pos = Position(view.coin, side, o["qty"], px, px * (1 - side * o["stop_bps"] / 1e4),
                           px * (1 + side * o["target_bps"] / 1e4), ts, fee, 0.0, 0.0, int(ts // 3600),
                           "Maker entry · " + o["reason"], clock_start=o.get("clock_start") or 0.0)
            self.positions[view.coin] = pos
            self.entry_times.append(ts)
            self.maker_stats["filled"] += 1
            return "filled", pos
        return "resting", o

    def _close(self, coin: str, px: float, ts: float, reason: str, maker: bool, slip_bps: float) -> dict:
        pos = self.positions.pop(coin)
        fee = pos.qty * px * (self.maker_bps if maker else self.taker_bps) / 1e4
        gross = pos.side * pos.qty * (px - pos.entry)
        self.cash += gross - fee
        self.fees_paid += fee
        pnl = gross - pos.entry_fee - fee - pos.funding_paid
        notional = pos.qty * pos.entry
        trade = {
            "book": self.name, "coin": coin, "side": pos.side, "qty": pos.qty, "entry": pos.entry, "exit": px,
            "opened": pos.opened, "closed": ts, "hold_s": ts - pos.opened, "pnl": pnl,
            "pnl_bps": pnl / notional * 1e4 if notional else 0.0, "gross": gross, "fees": pos.entry_fee + fee,
            "funding": pos.funding_paid, "slip_bps": pos.entry_slip_bps + slip_bps, "reason": reason,
            "entry_reason": pos.reason, "maker_exit": maker,
        }
        self.trades.append(trade)
        if len(self.trades) > 2000:
            self.trades = self.trades[-2000:]
        return trade

    def close_market(self, view: PerpView, ts: float, reason: str) -> dict | None:
        pos = self.positions.get(view.coin)
        if pos is None:
            return None
        px, filled, slip = self.market_fill(view, buying=pos.side < 0, qty=pos.qty)
        if filled < pos.qty * 0.999:  # book too thin: close at the worst visible level instead
            levels = view.bids if pos.side > 0 else view.asks
            px = levels[-1][0] if levels else (view.mid or pos.entry)
            slip = abs(px / (view.mid or px) - 1) * 1e4
        return self._close(view.coin, px, ts, reason, False, slip)

    def manage(self, view: PerpView, ts: float, max_hold_s: float, maker_exit: bool = False,
               exit_window_s: float = 120.0, reprice_s: float = 15.0) -> dict | None:
        """Stop (market), take-profit (resting maker limit) and max holding time. With maker_exit, the time exit
        first rests at the touch (our offer at the best ask for a long, our bid at the best bid for a short) from
        exit_window_s before max hold, re-priced to the touch every reprice_s, and fills only when price trades
        THROUGH it; whatever is still open at max hold goes out as a market order. The holding clock runs from
        the position's clock_start (the signal minute) when it has one, else from the fill."""
        pos = self.positions.get(view.coin)
        if pos is None or not view.bids or not view.asks:
            return None
        mid = view.mid
        if (pos.side > 0 and mid <= pos.stop) or (pos.side < 0 and mid >= pos.stop):
            return self.close_market(view, ts, "Stop loss")
        # the take-profit is a resting limit like the maker entries and exits, so it too needs price to trade
        # THROUGH it (a touch at our price says nothing about our queue position)
        if (pos.side > 0 and view.best_bid > pos.target) or (pos.side < 0 and view.best_ask < pos.target):
            return self._close(view.coin, pos.target, ts, "Take profit", True, 0.0)
        held, age = ts - pos.opened, pos.age(ts)
        if pos.exit_px and ts - pos.exit_placed > 2 * reprice_s:   # not re-priced (e.g. engine paused or restarted):
            pos.exit_px = 0.0                                    # the old offer is void, never filled at a stale price
        if maker_exit and pos.exit_px:
            if (pos.side > 0 and view.best_bid > pos.exit_px) or (pos.side < 0 and view.best_ask < pos.exit_px):
                return self._close(view.coin, pos.exit_px, ts, f"Maker exit at horizon ({held:.0f}s)", True, 0.0)
        if age >= max_hold_s:
            return self.close_market(view, ts, f"Max hold {max_hold_s:.0f}s")
        if maker_exit and age >= max_hold_s - exit_window_s and (not pos.exit_px or ts - pos.exit_placed >= reprice_s):
            pos.exit_px, pos.exit_placed = (view.best_ask if pos.side > 0 else view.best_bid), ts
        return None

    def apply_funding(self, coin: str, oracle: float | None, rate: float | None, ts: float) -> float:
        """Hourly funding: payment = size * oracle * rate. Positive rate: longs pay shorts."""
        pos = self.positions.get(coin)
        if pos is None or oracle is None or rate is None:
            return 0.0
        hour = int(ts // 3600)
        hours = hour - pos.last_funding_hour
        if hours <= 0:
            return 0.0
        pay = pos.side * pos.qty * oracle * rate * hours
        pos.last_funding_hour = hour
        pos.funding_paid += pay
        self.cash -= pay
        self.funding_paid += pay
        return pay

    def mark(self, marks: dict, ts: float, sample_every: float = 5.0) -> float:
        self.last_marks.update({k: v for k, v in marks.items() if v})
        today = utc_day(ts)
        eq = self.equity(marks)
        if today != self.day_key:
            self.day_key, self.day_start_equity, self.daily_stop = today, eq, False
        self.peak_equity = max(self.peak_equity, eq)
        dd = (self.peak_equity - eq) / self.peak_equity if self.peak_equity else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)
        if not self.equity_curve or ts - self.equity_curve[-1][0] >= sample_every:
            self.equity_curve.append([ts, round(eq, 2)])
            if len(self.equity_curve) > 17280:
                half = len(self.equity_curve) // 2
                self.equity_curve = self.equity_curve[:half:2] + self.equity_curve[half:]
        return eq

    def flatten(self, views: dict, ts: float, reason: str) -> list[dict]:
        self.pending.clear()          # resting entries are cancelled first
        out = []
        for coin in list(self.positions):
            v = views.get(coin)
            if v is not None and v.bids and v.asks:
                t = self.close_market(v, ts, reason)
            else:
                t = self._close(coin, self._px(coin, {}), ts, reason, False, 0.0)
            if t:
                out.append(t)
        return out

    # ---- stats -----------------------------------------------------------------------
    def stats(self, marks: dict, now: float) -> dict:
        eq = self.equity(marks)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
        n = len(self.trades)
        recent = [t for t in self.trades if now - t["closed"] <= 3600]
        return {
            "equity": round(eq, 2), "cash": round(self.cash, 2), "start": self.starting_cash,
            "pnl": round(eq - self.starting_cash, 2), "pnl_pct": round((eq / self.starting_cash - 1) * 100, 3),
            "day_pnl_pct": round(self.day_pnl_pct(marks), 3), "drawdown_pct": round(self.drawdown_pct(marks), 3),
            "max_drawdown_pct": round(max(self.max_drawdown * 100, self.drawdown_pct(marks)), 3),
            "fees": round(self.fees_paid, 2), "funding": round(self.funding_paid, 2),
            "gross": round(sum(t["gross"] for t in self.trades), 2),
            "trades": n, "wins": len(wins), "win_rate": round(len(wins) / n * 100, 1) if n else None,
            "avg_pnl_bps": round(sum(t["pnl_bps"] for t in self.trades) / n, 2) if n else None,
            "avg_hold_s": round(sum(t["hold_s"] for t in self.trades) / n, 1) if n else None,
            "profit_factor": round(gw / gl, 2) if gl > 0 else (None if not wins else 99.0),
            "trades_last_hour": len(recent), "entries_last_min": self.entries_last(now),
            "exposure": round(self.exposure(marks), 2), "leverage": round(self.leverage(marks), 3),
            "killed": self.killed, "kill_reason": self.kill_reason, "daily_stop": self.daily_stop,
            "funnel": dict(self.funnel), "exits": exit_breakdown(self.trades),
        }

    # ---- persistence -----------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = {c: asdict(p) for c, p in self.positions.items()}
        d["entry_times"] = list(self.entry_times)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PerpPaperBook":
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        pos_keys = set(Position.__dataclass_fields__)   # ignore fields from a newer version instead of failing to load
        d["positions"] = {c: Position(**{k: v for k, v in p.items() if k in pos_keys})
                          for c, p in (d.get("positions") or {}).items()}
        return cls(**d)
