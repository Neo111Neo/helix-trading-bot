"""HIP-4 replay backtester: plays the recorded tape (helix/hip4_tape.py) back event by event, in arrival order, and runs
a strategy against what really happened - the real books, the real trades, the real settlements.

  mm      the LIVE market maker (helix/hip4_mm.py), driven by the replay clock, so the backtest tests the code that runs
  sniper  stale-quote taker: when a Yes book's ask is below fair - edge (or its bid above fair + edge), send an order
          that arrives `latency_ms` later and fills only against the book AT THAT TIME (the quote may be gone)
  arb     locks that need no forecast: Yes ask + No ask < $1 (buy both), Yes bid + No bid > $1 (sell both), and
          cross-strike: P(above K1) >= P(above K2) when K1 < K2, so ask(K1) < bid(K2) is buy K1 / sell K2. Both legs
          must still be there after the latency, or neither is done.

Honesty rules: orders never see the future (latency is applied against later book states), takers pay the fee on
every contract, positions are paid from the tape's settlement records (or settledOutcome for ones the tape missed),
and anything still unsettled at the end is marked at fair and reported separately. Fair = hip4_mm.FairModel on 5-minute
bars that had CLOSED by the replay time.

CLI:  python -m helix.hip4_backtest --strategy sniper --hours 24 --param latency_ms=300 --param edge=0.03"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import sys
import time
from pathlib import Path

from .hip4 import primary
from .hip4_mm import FairModel, HIP4MarketMaker, trend_return, trend_sides
from .hip4_tape import read_tape, tape_files
from .hyperliquid import URLS, HLRest
from .market import Candle

STRATEGIES = {
    "mm": {"edge": 0.02, "skew": 0.02, "quote_size": 20, "max_position": 100, "max_risk_usd": 1000, "min_p": 0.05,
           "stop_min": 3, "max_tte_min": 1440, "fee": 0.0005, "market_weight": 0.5, "blend_max_spread": 0.10,
           "quote_every_s": 2.0, "latency_s": 0.25, "levels": 2, "level2_edge_mult": 2.0,
           "exits": 1, "take_profit": 0.15, "stop_loss": 0.30},
    # the sniper's PRE-REGISTERED rules (fixed 2026-10-06 18:50 UTC, before any held-out tape): the same as the live
    # paper sniper (config hip4_sniper), including the exchange's $10 minimum order
    "sniper": {"edge": 0.05, "latency_ms": 300, "size": 20, "max_position": 100, "fee": 0.0005, "min_p": 0.03,
               "stop_min": 3, "max_tte_min": 1440, "cooldown_s": 5, "check_every_ms": 200, "min_notional": 10,
               "band_lo": 0.0, "band_hi": 1.0, "sides": -1,            # = the live config: sell Yes only, 24 h
               # sides: 0 both, 1 buy Yes only, -1 sell Yes only, 2 auto (each underlying's trend, hip4_mm.trend_sides)
               "trend_lookback_min": 360, "trend_threshold": 0.003, "trend_neutral": 0},   # neutral: 0/1/-1, 9 = none
    "arb": {"min_edge": 0.005, "latency_ms": 300, "size": 50, "fee": 0.0005, "max_position": 500},
}


class Ledger:
    """Cash, per-market positions and settlement for the taker strategies (the market maker keeps its own)."""

    def __init__(self, cash: float = 10_000.0):
        self.start = self.cash = cash
        self.pos: dict[int, dict] = {}
        self.fills: list[dict] = []
        self.settled: list[dict] = []

    def fill(self, mid: int, meta: dict, side: int, px: float, qty: float, fee: float, t: float, **extra) -> None:
        p = self.pos.setdefault(mid, {"coin": meta.get("coin"), "strike": meta.get("strike"), "expiry": meta.get("expiry"),
                                      "venue": meta.get("venue"), "qty": 0.0, "cash": 0.0, "fees": 0.0, "fills": 0})
        flow = -side * px * qty - fee * qty
        p["qty"] += side * qty
        p["cash"] += flow
        p["fees"] += fee * qty
        p["fills"] += 1
        self.cash += flow
        self.fills.append({"t": t, "id": mid, "coin": p["coin"], "strike": p["strike"], "side": side, "px": px,
                           "qty": qty, **extra})

    def settle(self, mid: int, y: float | None, t: float) -> None:
        p = self.pos.pop(mid, None)
        if p is None:
            return
        payout = p["qty"] * y if y is not None else -(p["cash"] + p["fees"])
        self.cash += payout
        self.settled.append({"id": mid, "coin": p["coin"], "strike": p["strike"], "expiry": p["expiry"], "y": y,
                             "qty": p["qty"], "pnl": p["cash"] + payout, "fills": p["fills"], "t": t})


class World:
    """Market state as of the replay clock."""

    def __init__(self, bar_loader):
        self.t = 0.0
        self.markets: dict[int, dict] = {}
        self.meta_all: dict[int, dict] = {}
        self.books: dict[str, tuple] = {}         # "#<id*10+side>" -> (t, bids [[px, sz]], asks)
        self.spot: dict[str, float] = {}
        self.settled: dict[int, float | None] = {}
        self.model = FairModel(zs_refresh_s=3600, grid=True)
        self._loader = bar_loader                 # coin -> 5-minute bars, called once per underlying on the tape
        self._bars: dict[str, list] = {}
        self._bar_ts: dict[str, list] = {}
        self._bar_n: dict[str, int] = {}

    def ensure(self, coin: str) -> None:
        if coin in self._bars:
            return
        try:
            bars = self._loader(coin) or []
        except Exception:
            bars = []
        self._bars[coin], self._bar_ts[coin] = bars, [b.ts for b in bars]

    def advance_bars(self, t: float) -> None:
        for coin, ts in self._bar_ts.items():
            n = bisect.bisect_right(ts, t - 300)          # bars whose close (ts + 300) is at or before t
            if n != self._bar_n.get(coin) and n > 300:
                self._bar_n[coin] = n
                self.model.set_bars(coin, self._bars[coin][max(0, n - (15 * 288 + 4)):n])

    def top(self, coin_str: str):
        b = self.books.get(coin_str)
        if b is None:
            return None, 0.0, None, 0.0
        _, bids, asks = b
        return (bids[0][0] if bids else None, bids[0][1] if bids else 0.0,
                asks[0][0] if asks else None, asks[0][1] if asks else 0.0)

    def fair(self, mid: int) -> float | None:
        m = self.markets.get(mid) or self.meta_all.get(mid)
        spot = self.spot.get(m["coin"]) if m else None
        if not m or not spot or self.t >= m["expiry"]:
            return None
        return self.model.fair(m["coin"], spot, m["strike"], self.t, m["expiry"], m["twap_s"])


class Strategy:
    name = "base"

    def __init__(self, params: dict, world: World):
        self.p, self.w = params, world
        self.ledger = Ledger()

    def on_time(self, t): pass
    def on_tick(self, t): pass
    def on_book(self, coin_str, t): pass
    def on_spot(self, coin, t): pass
    def on_trade(self, rec): pass

    def on_settle(self, mid, y, t):
        self.ledger.settle(mid, y, t)

    def open_positions(self) -> dict[int, dict]:
        return self.ledger.pos

    def summary(self) -> dict:
        lg = self.ledger
        return {"fills": len(lg.fills), "contracts": sum(f["qty"] for f in lg.fills), "settled": lg.settled,
                "cash": lg.cash, "start": lg.start, "sample": lg.fills[-40:]}


class MMStrategy(Strategy):
    name = "mm"

    def __init__(self, params, world):
        super().__init__(params, world)
        p = dict(params)
        exits = {"enabled": bool(p.pop("exits", 0)), "take_profit": p.pop("take_profit", 0.15),
                 "stop_loss": p.pop("stop_loss", 0.30), "min_hold_s": 15, "cooldown_s": 60, "taker_fee": 0.001}
        p["levels"] = int(p.get("levels", 1))
        cfg = {"network": "mainnet", "hip4_mm": {"enabled": True, "coins": ["all"], "book_stale_s": 30, "start_cash": 10_000,
                                                 "kill_drawdown": 1.0, "exits": exits, **p}}
        self.mm = HIP4MarketMaker(cfg, None, spot_fn=lambda c: world.spot.get(c), rest=object(), clock=lambda: world.t)
        self.mm.model = world.model
        self.next_tick = 0.0

    def on_tick(self, t):
        mm, w = self.mm, self.w
        mm.markets = {i: m for i, m in w.markets.items() if primary(m) and m["expiry"] - t <= mm.max_tte_s + 60}
        mm._update_fairs(t)
        mm._markouts(t)
        mm._risk(t)
        mm._exits(t)
        mm._requote(t)

    def on_book(self, coin_str, t):
        _, bids, asks = self.w.books[coin_str]
        lv = [[{"px": p, "sz": s} for p, s in bids], [{"px": p, "sz": s} for p, s in asks]]
        self.mm.handle({"channel": "l2Book", "data": {"coin": coin_str, "levels": lv}}, now=t)

    def on_trade(self, rec):
        self.mm.handle({"channel": "trades", "data": [{"coin": rec["c"], "side": rec["sd"], "px": rec["p"],
                                                       "sz": rec["s"], "time": rec["et"], "tid": rec.get("tid")}]},
                       now=self.w.t)

    def on_settle(self, mid, y, t):
        self.mm.settle_one(str(mid), y, "tape", t)

    def open_positions(self):
        return {int(k): v for k, v in self.mm.pos.items()}

    def summary(self):
        mm = self.mm
        mo = [f["markout"] for f in mm.fills if f.get("markout") is not None]
        ed = [f["edge"] for f in mm.fills if f.get("edge") is not None]
        settled = [{"id": s["id"], "coin": s["coin"], "strike": s["strike"], "expiry": s["expiry"], "y": s["y"],
                    "qty": s["qty"], "pnl": s["pnl"], "fills": s["fills"], "t": s["ts"]} for s in mm.settled]
        return {"fills": len(mm.fills), "contracts": sum(f["qty"] for f in mm.fills), "settled": settled,
                "cash": mm.cash, "start": mm.start_cash,
                "avg_edge": sum(ed) / len(ed) if ed else None, "avg_markout": sum(mo) / len(mo) if mo else None,
                "markouts": len(mo), "killed": mm.kill_reason,
                "sample": [{"t": f["ts"], "id": f["id"], "coin": f["coin"], "strike": f["strike"], "side": f["side"],
                            "px": f["px"], "qty": f["qty"], "edge": f["edge"], "markout": f["markout"]}
                           for f in mm.fills[-40:]]}


class SniperStrategy(Strategy):
    name = "sniper"

    def __init__(self, params, world):
        super().__init__(params, world)
        self.pending: dict[int, dict] = {}
        self.last_check: dict[int, float] = {}
        self.cool: dict[int, float] = {}
        self.signals = 0
        self.missed = 0

    def _eligible(self, mid, t):
        # the same universe as the live sniper: only markets that settle on Hyperliquid's own perp price with <= 90 s
        # averaging (hip4.primary) - before 2026-10-07 the backtest also traded the others, which the live bot skips
        m = self.w.markets.get(mid)
        return m is not None and primary(m) and m["coin"] in self.w.spot and \
            self.p["stop_min"] * 60 < m["expiry"] - t <= self.p["max_tte_min"] * 60

    def _check(self, mid, t):
        if mid in self.pending or t < self.cool.get(mid, 0) or not self._eligible(mid, t):
            return
        if (t - self.last_check.get(mid, -1e9)) * 1000 < self.p["check_every_ms"]:
            return
        self.last_check[mid] = t
        fair = self.w.fair(mid)
        if fair is None or not (self.p["min_p"] <= fair <= 1 - self.p["min_p"]):
            return
        bid, _, ask, _ = self.w.top(f"#{mid * 10}")
        qty = self.ledger.pos.get(mid, {}).get("qty", 0.0)
        lo, hi, sides = self.p.get("band_lo", 0.0), self.p.get("band_hi", 1.0), self.p.get("sides", 0)
        if sides == 2:
            sides = self._auto_sides(self.w.markets[mid]["coin"], t)
            if sides is None:
                return
        inb = lambda px: px is not None and lo - 1e-9 <= px <= hi + 1e-9   # noqa: E731
        if sides >= 0 and inb(ask) and ask <= fair - self.p["edge"] and qty < self.p["max_position"]:
            self.pending[mid] = {"side": 1, "limit": ask, "due": t + self.p["latency_ms"] / 1000, "fair": fair, "t0": t}
        elif sides <= 0 and inb(bid) and bid >= fair + self.p["edge"] and qty > -self.p["max_position"]:
            self.pending[mid] = {"side": -1, "limit": bid, "due": t + self.p["latency_ms"] / 1000, "fair": fair, "t0": t}
        else:
            return
        self.signals += 1

    NEUTRAL = {0: "both", 1: "buy", -1: "sell", 9: "none"}
    SIDE_CODE = {"both": 0, "buy": 1, "sell": -1, "none": None}

    def _auto_sides(self, coin, t):
        """The live rule (hip4_mm.trend_sides) on the replay's closed bars and perp mid; None = take nothing."""
        ret = trend_return(self.w._bars.get(coin), self.w.spot.get(coin), t, self.p["trend_lookback_min"] * 60)
        name = trend_sides(ret, self.p["trend_threshold"], self.NEUTRAL[int(self.p["trend_neutral"])])
        return self.SIDE_CODE[name]

    def on_time(self, t):
        for mid, o in list(self.pending.items()):
            if t < o["due"]:
                continue
            del self.pending[mid]
            bid, bsz, ask, asz = self.w.top(f"#{mid * 10}")
            qty = self.ledger.pos.get(mid, {}).get("qty", 0.0)
            if o["side"] > 0 and ask is not None and ask <= o["limit"] + 1e-9:
                cap = min(asz, self.p["max_position"] - qty)
                px, pay = ask, ask
            elif o["side"] < 0 and bid is not None and bid >= o["limit"] - 1e-9:
                cap = min(bsz, self.p["max_position"] + qty)
                px, pay = bid, 1 - bid                                     # a Yes sell is paid as a No buy
            else:
                self.missed += 1                 # the stale quote was gone by the time we arrived
                continue
            n = math.floor(min(self.p["size"], cap) + 1e-9)
            mn = self.p.get("min_notional", 0)
            if n > 0 and mn and pay * n < mn - 1e-9:                      # the exchange's minimum order
                need = math.ceil(mn / max(pay, 1e-6) - 1e-9)
                n = need if need <= math.floor(cap + 1e-9) else 0
            if n <= 0:
                continue
            self.ledger.fill(mid, self.w.markets.get(mid) or self.w.meta_all.get(mid, {}), o["side"], px, n,
                             self.p["fee"], t, fair=round(o["fair"], 4), edge=round((o["fair"] - px) * o["side"], 4),
                             wait_ms=round((t - o["t0"]) * 1000))
            self.cool[mid] = t + self.p["cooldown_s"]

    def on_book(self, coin_str, t):
        n = int(coin_str[1:])
        if n % 10 == 0:
            self._check(n // 10, t)

    def on_spot(self, coin, t):
        for mid, m in self.w.markets.items():
            if m["coin"] == coin:
                self._check(mid, t)

    def summary(self):
        s = super().summary()
        s.update(signals=self.signals, missed=self.missed,
                 avg_edge=(sum(f["edge"] for f in self.ledger.fills) / len(self.ledger.fills)) if self.ledger.fills else None)
        return s


class ArbStrategy(Strategy):
    name = "arb"

    def __init__(self, params, world):
        super().__init__(params, world)
        self.pending: list[dict] = []
        self.signals = {"yes_no_buy": 0, "yes_no_sell": 0, "strike": 0}
        self.missed = 0
        self.busy: set = set()

    def _group(self, m):
        return (m["coin"], m["expiry"], m["venue"], m["source"], m["twap_s"])

    def on_book(self, coin_str, t):
        mid = int(coin_str[1:]) // 10
        m = self.w.markets.get(mid)
        if m is None or m["expiry"] - t < 60 or mid in self.busy:
            return
        f, e = self.p["fee"], self.p["min_edge"]
        yb, ybs, ya, yas = self.w.top(f"#{mid * 10}")
        nb, nbs, na, nas = self.w.top(f"#{mid * 10 + 1}")
        legs = None
        if ya is not None and na is not None and ya + na < 1 - 2 * f - e:
            legs, kind = [(mid, 0, +1, ya), (mid, 1, +1, na)], "yes_no_buy"
        elif yb is not None and nb is not None and yb + nb > 1 + 2 * f + e:
            legs, kind = [(mid, 0, -1, yb), (mid, 1, -1, nb)], "yes_no_sell"
        else:
            g = self._group(m)
            for oid, o in self.w.markets.items():
                if oid == mid or self._group(o) != g or oid in self.busy:
                    continue
                lo, hi = (mid, oid) if m["strike"] < o["strike"] else (oid, mid)
                _, _, a_lo, _ = self.w.top(f"#{lo * 10}")
                b_hi, _, _, _ = self.w.top(f"#{hi * 10}")
                if a_lo is not None and b_hi is not None and a_lo + 2 * f + e < b_hi:
                    legs, kind = [(lo, 0, +1, a_lo), (hi, 0, -1, b_hi)], "strike"
                    break
        if legs:
            self.signals[kind] += 1
            self.busy |= {leg[0] for leg in legs}
            self.pending.append({"legs": legs, "due": t + self.p["latency_ms"] / 1000, "kind": kind})

    def on_time(self, t):
        keep = []
        for o in self.pending:
            if t < o["due"]:
                keep.append(o)
                continue
            fills, ok = [], True
            for mid, side_tok, side, limit in o["legs"]:
                b, bs, a, as_ = self.w.top(f"#{mid * 10 + side_tok}")
                px, sz = (a, as_) if side > 0 else (b, bs)
                if px is None or (side > 0 and px > limit + 1e-9) or (side < 0 and px < limit - 1e-9):
                    ok = False
                    break
                fills.append((mid, side_tok, side, px, sz))
            self.busy -= {leg[0] for leg in o["legs"]}
            if not ok:
                self.missed += 1
                continue
            n = min([self.p["size"]] + [sz for *_, sz in fills])
            if n <= 0:
                continue
            for mid, side_tok, side, px, _ in fills:
                # a No contract is the mirror of Yes: buying No at q = selling Yes at 1 - q (same payoff, same cash)
                yes_side, yes_px = (side, px) if side_tok == 0 else (-side, 1 - px)
                self.ledger.fill(mid, self.w.markets.get(mid) or self.w.meta_all.get(mid, {}), yes_side, yes_px, n,
                                 self.p["fee"], t, kind=o["kind"])
        self.pending = keep

    def summary(self):
        s = super().summary()
        s.update(signals=self.signals, missed=self.missed)
        return s


CLASSES = {"mm": MMStrategy, "sniper": SniperStrategy, "arb": ArbStrategy}


# ---------------------------------------------------------------------- data
def load_bars(rest: HLRest, coin: str, start: float, end: float, cache_dir: Path) -> list:
    """5-minute candles from 15 days before `start` to `end`, cached per coin."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{coin.replace(':', '_')}.json"
    bars = []
    if path.exists():
        try:
            bars = [Candle(*row) for row in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            bars = []
    need_from = start - 15 * 86400 - 3600
    if not bars or bars[0].ts > need_from + 600 or bars[-1].ts + 600 < min(end, time.time() - 300):
        fresh = rest.candles(coin, "5m", int((min(end, time.time()) - need_from) / 60) + 10, min(end, time.time()))
        merged = {b.ts: b for b in bars}
        merged.update({b.ts: b for b in fresh})
        bars = [merged[k] for k in sorted(merged)]
        path.write_text(json.dumps([[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in bars]), encoding="utf-8")
    return bars


def tape_span(tape_dir: Path) -> tuple[float, float] | None:
    files = tape_files(tape_dir)
    if not files:
        return None
    first = next(read_tape(files[0]), None)
    last = None
    for r in read_tape(files[-1]):
        last = r
    return (first or {}).get("t"), (last or {}).get("t")


def run_backtest(data_dir: Path, strategy: str, params: dict | None = None, hours: float | None = None,
                 start: float | None = None, end: float | None = None, rest: HLRest | None = None,
                 progress=None, network: str = "mainnet") -> dict:
    if strategy not in CLASSES:
        raise ValueError(f"strategy must be one of {sorted(CLASSES)}")
    p = {**STRATEGIES[strategy], **(params or {})}
    tape_dir = Path(data_dir) / "tape"
    span = tape_span(tape_dir)
    if span is None or span[0] is None:
        raise ValueError("no tape recorded yet (data/tape/ is empty)")
    end = min(end or span[1], span[1])
    start = max(start or (end - hours * 3600 if hours else span[0]), span[0])
    rest = rest or HLRest(URLS[network][0], timeout=15)
    w = World(lambda c: load_bars(rest, c, start, end, tape_dir / "bars"))
    strat = CLASSES[strategy](p, w)
    tick_s = float(p.get("quote_every_s", 1.0))
    next_tick, next_eq = start, start
    equity, events, t0 = [], 0, time.time()
    files = [f for f in tape_files(tape_dir) if f.name[:11] >= time.strftime("%Y%m%d-%H", time.gmtime(start - 3600))
             and f.name[:11] <= time.strftime("%Y%m%d-%H", time.gmtime(end))]

    def equity_now():
        """Open positions at the MARKET mid (a tight book), never at our own fair: a taker that bought because the
        model said "cheap" would otherwise be graded by the same model. Model fair only when there is no real book."""
        marks = 0.0
        for mid, pos in strat.open_positions().items():
            b, _, a, _ = w.top(f"#{mid * 10}")
            f = (b + a) / 2 if b is not None and a is not None and a - b <= 0.10 else w.fair(mid)
            marks += pos["qty"] * (f if f is not None else pos.get("last_mark", 0.5))
        s = strat.summary()
        return s["cash"] + marks

    for i, path in enumerate(files):
        for rec in read_tape(path):
            t = rec.get("t", 0)
            k = rec.get("k")
            if t > end:
                break
            w.t = t
            if k == "m":
                w.markets = {m["id"]: m for m in rec["mk"]}
                w.meta_all.update(w.markets)
                for c in {m["coin"] for m in rec["mk"]}:
                    w.ensure(c)
                continue
            if k == "x":
                w.settled[rec["id"]] = rec.get("y")
                if t >= start:
                    strat.on_settle(rec["id"], rec.get("y"), t)
                continue
            if k == "pb":
                if rec.get("b") is not None and rec.get("a") is not None:
                    w.spot[rec["c"]] = (rec["b"] + rec["a"]) / 2
            elif k == "ob":
                w.books[rec["c"]] = (t, rec["b"], rec["a"])
            if t < start:
                continue
            w.advance_bars(t)
            while next_tick <= t:
                strat.on_tick(next_tick)
                next_tick += tick_s
            strat.on_time(t)
            if k == "pb":
                strat.on_spot(rec["c"], t)
            elif k == "ob":
                strat.on_book(rec["c"], t)
            elif k == "ot":
                strat.on_trade(rec)
            events += 1
            if t >= next_eq:
                equity.append([round(t), round(equity_now(), 2)])
                next_eq = t + 300
        if progress:
            progress((i + 1) / len(files))
    # positions the tape never saw settle: ask the exchange (they may have settled after the tape ends)
    late = 0
    for mid, pos in list(strat.open_positions().items()):
        if pos["expiry"] and pos["expiry"] + 90 < time.time():
            try:
                res = rest.post({"type": "settledOutcome", "outcome": int(mid)})
            except Exception:
                res = None
            if res:
                y = float(res.get("settleFraction", "nan"))
                strat.on_settle(int(mid), y if math.isfinite(y) else None, end)
                late += 1
    final = equity_now()
    s = strat.summary()
    settled = s.pop("settled")
    pnl_settled = sum(x["pnl"] for x in settled)
    by_coin: dict[str, dict] = {}
    for x in settled:
        c = by_coin.setdefault(x["coin"], {"markets": 0, "pnl": 0.0, "won": 0})
        c["markets"] += 1
        c["pnl"] += x["pnl"]
        c["won"] += x["pnl"] > 0
    equity.append([round(end), round(final, 2)])
    contracts = s.get("contracts") or 0
    return {
        "strategy": strategy, "params": p, "from": start, "to": end, "hours": round((end - start) / 3600, 2),
        "events": events, "runtime_s": round(time.time() - t0, 1), "markets_seen": len(w.meta_all),
        "pnl": round(final - s["start"], 4), "pnl_settled": round(pnl_settled, 4),
        "pnl_open_marked": round(final - s["start"] - pnl_settled, 4),
        "open_markets": len(strat.open_positions()), "settled_markets": len(settled), "late_settled": late,
        "won_markets": sum(1 for x in settled if x["pnl"] > 0),
        "cents_per_contract": round((final - s["start"]) / contracts * 100, 3) if contracts else None,
        "by_coin": {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in by_coin.items()},
        "equity": equity, "settled": settled[-60:], **s,
    }


# the automatic comparison: is there an edge, and at what speed?
AUTO_SUITE = [("sniper", {"latency_ms": 10}), ("sniper", {"latency_ms": 100}), ("sniper", {"latency_ms": 300}),
              ("sniper", {"latency_ms": 1000}), ("arb", {"latency_ms": 300}), ("mm", {})]
HOLDOUT_SUITE = [("sniper", {})]          # the pre-registered rules on tape recorded after they were fixed


class BacktestRunner:
    """Backtests for the web page, one at a time from a queue, each in its OWN PROCESS so a long replay never takes
    CPU (the GIL) from the live bots. Every `auto_every_h` hours the AUTO_SUITE is queued over the whole tape.
    The last 60 results are kept in data/backtests.json."""

    def __init__(self, data_dir: Path, network: str = "mainnet", auto_every_h: float = 0.0, clock=time.time,
                 holdout_from: float = 0.0):
        import threading
        self.data_dir, self.network, self.clock = Path(data_dir), network, clock
        self.holdout_from = holdout_from
        self.path = self.data_dir / "backtests.json"
        self.lock = threading.Lock()
        self.queue: list[dict] = []
        self.running, self.progress, self.error, self.current = False, 0.0, None, None
        self.auto_every_s = auto_every_h * 3600
        try:
            self.results = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.results = []
        last_auto = max((r.get("ran_at", 0) for r in self.results if r.get("label") == "auto"), default=0)
        # on schedule from the last automatic suite, or an hour into the tape; NOT from process start, which pushed the
        # first suite back on every restart (2026-10-06: restarted more often than hourly, so it never ran)
        span = tape_span(self.data_dir / "tape") if self.auto_every_s else None
        first = (span[0] + 3600) if span and span[0] else clock() + 3600
        self.next_auto = (last_auto + self.auto_every_s if last_auto else first) if self.auto_every_s else None
        self._wake = threading.Event()
        threading.Thread(target=self._worker, name="hip4-backtests", daemon=True).start()

    def start(self, strategy: str, params: dict, hours: float | None, label: str = "manual",
              start: float | None = None) -> tuple[bool, str]:
        if strategy not in CLASSES:
            return False, f"strategy must be one of {sorted(CLASSES)}"
        bad = [k for k in params if k not in STRATEGIES[strategy]]
        if bad:
            return False, f"unknown parameter(s) for {strategy}: {', '.join(bad)}"
        with self.lock:
            if len(self.queue) >= 20:
                return False, "the queue is full"
            self.queue.append({"strategy": strategy, "params": params, "hours": hours, "label": label, "start": start})
        self._wake.set()
        return True, "queued" if self.running else "started"

    def queue_auto(self) -> None:
        for s, p in AUTO_SUITE:
            self.start(s, p, None, label="auto")
        if self.holdout_from:                    # and the held-out test (only once there is tape after the cut)
            span = tape_span(self.data_dir / "tape")
            if span and span[1] and span[1] - self.holdout_from > 1800:
                for s, p in HOLDOUT_SUITE:
                    self.start(s, p, None, label="auto", start=self.holdout_from)

    def _worker(self) -> None:
        while True:
            if self.next_auto is not None and self.clock() >= self.next_auto:
                self.next_auto = self.clock() + self.auto_every_s
                self.queue_auto()
            with self.lock:
                job = self.queue.pop(0) if self.queue else None
            if job is None:
                self._wake.wait(30)
                self._wake.clear()
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        import os
        import subprocess
        import tempfile
        self.running, self.progress, self.error = True, 0.0, None
        self.current = {**job, "started": self.clock()}
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            out, prog = Path(d) / "result.json", Path(d) / "progress"
            cmd = [sys.executable, "-m", "helix.hip4_backtest", "--strategy", job["strategy"], "--data",
                   str(self.data_dir), "--json-out", str(out), "--progress-file", str(prog), "--network", self.network]
            if job["hours"]:
                cmd += ["--hours", str(job["hours"])]
            if job.get("start"):
                cmd += ["--start", str(job["start"])]
            for k, v in job["params"].items():
                cmd += ["--param", f"{k}={v}"]
            try:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                        creationflags=flags, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                while proc.poll() is None:
                    try:
                        self.progress = float(prog.read_text() or 0)
                    except (OSError, ValueError):
                        pass
                    time.sleep(1.0)
                err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                if proc.returncode != 0 or not out.exists():
                    self.error = (err.strip().splitlines() or ["backtest failed"])[-1][:300]
                    return
                res = json.loads(out.read_text(encoding="utf-8"))
                res.update(id=int(self.clock() * 1000), ran_at=self.clock(), label=job["label"],
                           holdout=bool(job.get("start")))
                with self.lock:
                    self.results = (self.results + [res])[-60:]
                    data = json.dumps(self.results, default=str)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(data, encoding="utf-8")
                tmp.replace(self.path)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                self.running, self.progress = False, 1.0

    SPAN_TTL_S = 300.0

    def _span(self):
        """The tape's time span, re-read at most every SPAN_TTL_S: reading it scans the whole current hour's tape file
        (4+ s on the Tokyo box), and the dashboard asks for it every few seconds (2026-10-13: that alone used ~35% of a
        core and queued requests behind it)."""
        now = time.time()
        cached = getattr(self, "_span_cache", None)
        if cached is None or now - cached[0] >= self.SPAN_TTL_S:
            cached = (now, tape_span(self.data_dir / "tape"))
            self._span_cache = cached
        return cached[1]

    def to_dict(self) -> dict:
        span = self._span()
        with self.lock:
            queue = [{k: j[k] for k in ("strategy", "params", "label")} for j in self.queue]
            results = self.results[::-1]
        return {"running": self.running, "progress": round(self.progress, 3), "error": self.error,
                "current": self.current, "queue": queue, "defaults": STRATEGIES, "next_auto": self.next_auto,
                "auto_every_h": self.auto_every_s / 3600, "auto_suite": [[s, p] for s, p in AUTO_SUITE],
                "tape_from": span[0] if span else None, "tape_to": span[1] if span else None,
                "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay the HIP-4 tape through a strategy")
    ap.add_argument("--strategy", choices=sorted(CLASSES), default="sniper")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--start", type=float, default=None, help="unix time to start from (e.g. the holdout cut)")
    ap.add_argument("--end", type=float, default=None, help="unix time to stop at (e.g. one UTC day per run)")
    ap.add_argument("--param", action="append", default=[], help="name=value, e.g. latency_ms=300")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--network", default="mainnet")
    ap.add_argument("--json-out", default=None, help="write the full result here (used by the web runner)")
    ap.add_argument("--progress-file", default=None)
    args = ap.parse_args(argv)
    params = {}
    for kv in args.param:
        k, v = kv.split("=", 1)
        params[k] = float(v)

    def progress(f):
        if args.progress_file:
            Path(args.progress_file).write_text(f"{f:.4f}")
        else:
            print(f"\r  {f:.0%}", end="", file=sys.stderr)

    res = run_backtest(Path(args.data), args.strategy, params, hours=args.hours, start=args.start, end=args.end,
                       progress=progress, network=args.network)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, default=str), encoding="utf-8")
        return 0
    print(file=sys.stderr)
    out = copy.deepcopy(res)
    out.pop("equity")
    out.pop("sample", None)
    out["settled"] = len(res["settled"])
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
