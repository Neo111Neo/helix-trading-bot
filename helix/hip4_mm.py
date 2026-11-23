"""HIP-4 market maker (PAPER): rests a bid and an ask on Hyperliquid's "price above K at T" outcome markets around a
fair probability, and earns the spread instead of predicting direction. Nothing here can place a real order.

Why this and not a forecast: on 2026-10-06 every forecaster we have (Jev v1-v4, Claude, the volatility models) scored
no better than the HIP-4 market's own mid, and trading against the book at the bid/ask lost the spread. A market maker
needs a fair price that is merely as good as the market's, which the HAR volatility model ("model_har") is.

  fair      model_har from hip4.strike_state, the Jev Lab's best held-out baseline, on the live perp mid. The heavy
            parts (15 days of standardised past moves) are cached per coin and 5-minute bar, so a requote costs ~1 ms.
            When the market has a real two-sided book (spread <= blend_max_spread), fair is blended with its mid
            (market_weight): the mid forecast as well as the model, and the model alone would be a directional bet.
  quotes    bid = fair - edge - skew, ask = fair + edge - skew, rounded away from fair to 0.001 and never crossing the
            market (a maker never takes). skew leans both quotes against the inventory we hold.
  fills     only from REAL trades on the market's Yes book (WebSocket), and only when the trade went THROUGH our price:
            a seller printing below our bid, a buyer printing above our ask. At exactly our price we fill only if our
            quote was strictly better than the rest of the book (we would have been first in the queue).
  risk      contracts per market, total worst-case loss across markets, no quotes in the last `stop_min` minutes
            (settlement averages and gamma), none on near-certain markets, and a drawdown kill switch.
  settle    positions are held to expiry and paid from the market's own settledOutcome.
  markout   every fill is re-marked against fair 60 s later: the honest measure of whether we are being picked off.

State: data/hip4_mm.json."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path
from statistics import pstdev

from .hip4 import GIVE_UP_S, parse_market, primary
from .hip4_tape import _Every
from .hourly import empirical_above, har_params, har_sigma, lognormal_above, past_z, phi, realized_sigma, \
    seasonal_factors, sigma_per_hour
from .hyperliquid import URLS, HLRest
from .wsclient import WebSocket, WebSocketError

TICK = 0.001
META_EVERY_S = 60
MARKOUT_S = 60
MAX_FILLS = 2000
MAX_SETTLED = 1000
REGIME_ALARM_C = -5.0      # cents per contract (see HIP4MarketMaker.regime)


def _floor(p: float) -> float:
    return math.floor(p / TICK + 1e-9) * TICK


def _ceil(p: float) -> float:
    return math.ceil(p / TICK - 1e-9) * TICK


def trend_return(bars: list | None, spot: float | None, now: float, lookback_s: float) -> float | None:
    """log(spot now / the close of the last 5-minute bar that had closed `lookback_s` ago); None without the data."""
    if not bars or not spot or spot <= 0 or lookback_s <= 0:
        return None
    cutoff = now - lookback_s - 300               # a bar opened at ts closes at ts + 300
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid].ts <= cutoff:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0 or bars[lo - 1].close <= 0:
        return None
    return math.log(spot / bars[lo - 1].close)


def trend_sides(ret: float | None, threshold: float, neutral: str = "both") -> str:
    """Auto direction (sides: auto): in a downtrend "above K" settles No more often than the book prices it, so sell
    Yes only; in an uptrend buy Yes only; in between (or without data) the `neutral` setting."""
    if ret is None:
        return neutral
    if ret <= -threshold:
        return "sell"
    if ret >= threshold:
        return "buy"
    return neutral


class FairModel:
    """hip4.strike_state's model_har, split so the parts that only change with a new 5-minute bar are computed once.
    tests/test_hip4_mm.py checks it against strike_state to the last digit."""

    def __init__(self, zs_refresh_s: float = 0.0, grid: bool = False):
        """Live: exact (zs_refresh_s=0, grid=False). Replay: the 15-day table of past moves is rebuilt once per
        `zs_refresh_s` of simulated time (an hour of new bars barely moves a 15-day sample) and P(Z > x) is read off
        a precomputed grid (|error| < 1e-4), which makes a day of replay minutes instead of hours."""
        self.zs_refresh_s, self.grid = zs_refresh_s, grid
        self.bars: dict[str, list] = {}
        self._base: dict[str, tuple] = {}      # coin -> (last bar ts, closes, sig_h, factors, rv1, rv4)
        self._zs: dict[tuple, list] = {}       # (coin, last bar ts, step) -> standardised past moves
        self._grids: dict[tuple, list] = {}

    def set_bars(self, coin: str, c5: list) -> None:
        if not c5:
            return
        self.bars[coin] = c5
        if coin in self._base and self._base[coin][0] == c5[-1].ts:
            return
        closes = [c.close for c in c5]
        last_h = closes[-13:]
        rv1 = pstdev([math.log(b / a) for a, b in zip(last_h, last_h[1:])]) * math.sqrt(12) if len(last_h) == 13 else 0.0
        bucket = c5[-1].ts // self.zs_refresh_s if self.zs_refresh_s else c5[-1].ts
        self._base[coin] = (bucket, closes, sigma_per_hour(closes[-289:]), seasonal_factors(c5), rv1,
                            realized_sigma(closes, 48))
        self._zs = {k: v for k, v in self._zs.items() if k[0] != coin or k[1] == bucket}
        self._grids = {k: v for k, v in self._grids.items() if k[0] != coin or k[1] == bucket}

    def fair(self, coin: str, spot: float, strike: float, now: float, expiry: float, twap_s: int = 0) -> float | None:
        base = self._base.get(coin)
        if base is None or not spot or spot <= 0 or strike <= 0:
            return None
        bucket, _closes, sig_h, factors, rv1, rv4 = base
        tte_min = max(1.0, (expiry - now) / 60)
        eff_min = max(1.0, tte_min - 2 * twap_s / 180)
        t_h = eff_min / 60
        f_now = factors[int((now + expiry) / 2 // 3600) % 24]
        step = max(1, round(eff_min / 5))
        key = (coin, bucket, step)
        zs = self._zs.get(key)
        if zs is None:
            zs = [z / math.sqrt(step / 12) for z in past_z(self.bars[coin], factors, step=step, every=step != 12)]
            self._zs[key] = zs
        w_har, seasonal = har_params(eff_min)
        hs = har_sigma(rv1, rv4, sig_h, w_har) * (f_now if seasonal else 1.0) * math.sqrt(t_h)
        if len(zs) >= 100 and hs > 0:
            x = math.log(strike / spot) / hs
            return self._grid_above(key, zs, x) if self.grid else empirical_above(zs, x)
        return lognormal_above(spot, strike, hs)

    GRID_LO, GRID_STEP, GRID_N = -8.0, 0.01, 1601

    def _grid_above(self, key: tuple, zs: list, x: float) -> float:
        g = self._grids.get(key)
        if g is None:
            # histogram the past moves into 0.01-wide bins, then the kernel sum per grid point runs over bins
            bins: dict[int, int] = {}
            for z in zs:
                b = round(z / 0.01)
                bins[b] = bins.get(b, 0) + 1
            items, n = [(b * 0.01, c) for b, c in bins.items()], len(zs)
            g = [sum(c * phi((zc - (self.GRID_LO + i * self.GRID_STEP)) / 0.3) for zc, c in items) / n
                 for i in range(self.GRID_N)]
            self._grids[key] = g
        pos = (x - self.GRID_LO) / self.GRID_STEP
        if pos <= 0:
            return g[0]
        if pos >= self.GRID_N - 1:
            return g[-1]
        i = int(pos)
        return g[i] + (g[i + 1] - g[i]) * (pos - i)


class HIP4MarketMaker:
    def __init__(self, cfg: dict, data_dir: Path | None, spot_fn, rest: HLRest | None = None, ws_factory=WebSocket,
                 clock=time.time, notify=None, section: str = "hip4_mm", name: str = "hip4_mm"):
        mc = cfg[section]
        self.name = name
        self.enabled = bool(mc["enabled"])
        # strategy "maker" rests quotes (quoting: false pauses them, positions still settle); "sniper" takes the book
        # when it is >= snipe_edge past our MODEL fair, near expiry, and holds to settlement (see _snipe)
        self.strategy = mc.get("strategy", "maker")
        self.quoting = bool(mc.get("quoting", True))
        self.snipe_edge = float(mc.get("snipe_edge", 0.05))
        self.snipe_latency_s = float(mc.get("snipe_latency_ms", 300)) / 1000
        self.snipe_cooldown_s = float(mc.get("snipe_cooldown_s", 5))
        # only take contracts whose Yes price is inside this band (the model is weakest on long shots), and only on
        # these sides ("both", "buy" = buy Yes only, "sell" = sell Yes only)
        band = mc.get("price_band") or [0.0, 1.0]
        self.band_lo, self.band_hi = float(band[0]), float(band[1])
        self.snipe_sides = str(mc.get("sides", "both"))
        # sides "auto": per underlying, follow its trend over lookback_min (see trend_sides)
        tr = mc.get("trend") or {}
        self.trend_lookback_s = float(tr.get("lookback_min", 360)) * 60
        self.trend_threshold = float(tr.get("threshold", 0.003))
        self.trend_neutral = str(tr.get("neutral", "sell"))
        self.trend_now: dict[str, dict] = {}      # coin -> {"ret", "sides"} as of the last sniper step (for the page)
        self._pending: dict[int, dict] = {}
        self.snipe_stats = {"signals": 0, "filled": 0, "missed": 0, "skipped": 0}
        self.on_signal = None                     # the live pilot (helix/hip4_exec.py) hooks in here; None = paper only
        self.on_step = None
        self.live_fire_on = "fill"                # the pilot sets it: call on_signal on the paper "fill" or the "signal"
        self._watch: dict[int, list] = {}         # dry run: quotes the pilot would have hit, watched until they go
        self.book_hooks: list = []                # called (mid, now) on every Yes-book update (the shadow lab)
        self.all_coins = [str(c).lower() for c in mc["coins"]] == ["all"]   # every underlying with HIP-4 markets
        self.coins = [] if self.all_coins else list(mc["coins"])
        self.edge = float(mc["edge"])
        self.skew_max = float(mc["skew"])
        self.quote_size = float(mc["quote_size"])
        self.max_pos = float(mc["max_position"])
        self.max_risk = float(mc["max_risk_usd"])
        self.min_p = float(mc["min_p"])
        self.stop_s = float(mc["stop_min"]) * 60
        self.max_tte_s = float(mc["max_tte_min"]) * 60
        self.book_stale_s = float(mc["book_stale_s"])
        self.fee = float(mc["fee"])
        self.start_cash = float(mc["start_cash"])
        self.kill_dd = float(mc["kill_drawdown"])
        self.every_s = float(mc["quote_every_s"])
        self.market_weight = float(mc["market_weight"])
        self.blend_spread = float(mc["blend_max_spread"])
        self.latency_s = float(mc.get("latency_s", 0.25))
        # bet size: one multiplier scales contracts per quote, the per-market position limit and the risk cap together
        self.base = {"quote_size": self.quote_size, "max_position": self.max_pos, "max_risk_usd": self.max_risk}
        self.max_mult = float(mc.get("max_size_multiplier", 5.0))
        self.size_mult = 1.0
        # a second, deeper quote on each side catches the bigger sweeps
        self.levels = int(mc.get("levels", 1))
        self.level2_mult = float(mc.get("level2_edge_mult", 2.0))
        # early exits against the real book (taker): take profit on a big favourable jump, stop out on a big adverse one
        ex = mc.get("exits") or {}
        self.exits_on = bool(ex.get("enabled", False))
        self.take_profit = float(ex.get("take_profit", 0.15))
        self.stop_loss = float(ex.get("stop_loss", 0.30))
        self.exit_min_hold_s = float(ex.get("min_hold_s", 15))
        self.exit_cooldown_s = float(ex.get("cooldown_s", 60))
        self.taker_fee = float(ex.get("taker_fee", 0.001))
        self.cool: dict[int, float] = {}         # market id -> no quotes until (after an early exit)
        self.min_notional = float(mc.get("min_notional", 0))      # the exchange's $10 minimum order value (0 = off)
        self.requote_thr = float(mc.get("requote_threshold", 0))  # sticky quotes (0 = re-price every step)
        self.max_markets = int(mc.get("max_markets", 0))           # quote only the N busiest markets (0 = all)
        self.since: float | None = None                            # first paper fill (the go-live gate counts days)
        self.kill_log: list[float] = []
        self.guard_move = float(mc.get("guard_move", 0.05))
        self.guard_s = float(mc.get("guard_s", 60))
        self._fair_hist: dict[int, deque] = {}
        self._trades: dict[int, deque] = {}       # market id -> receive times of real trades (for max_markets)
        self.spot_fn = spot_fn
        net = cfg.get("network", "mainnet")
        self.rest = rest or HLRest(URLS[net][0], timeout=10)
        self.ws_url, self.ws_factory = URLS[net][1], ws_factory
        self.clock = clock
        self.notify = notify or (lambda level, text: None)
        self.path = Path(data_dir) / f"{name}.json" if data_dir is not None else None
        # many markets on many coins: the fast fair model (15-day table refreshed every 15 min, grid lookup)
        self.model = FairModel(zs_refresh_s=float(mc.get("zs_refresh_s", 0)), grid=bool(mc.get("fast_model", False)))
        self.lock = threading.RLock()
        self.markets: dict[int, dict] = {}      # listed markets on our coins that settle on the perp (hip4.primary)
        self.books: dict[int, dict] = {}        # id -> {bid, ask, bid_sz, ask_sz, ts} of the Yes book
        self.fairs: dict[int, tuple] = {}       # id -> (fair, ts): the model blended with a tight market mid
        self.models: dict[int, float] = {}      # id -> the model alone, for display
        self.quotes: dict[int, dict] = {}       # id -> {bid, ask, bid_sz, ask_sz, bid_best, ask_best, placed_bid, ...}
        self.pos: dict[str, dict] = {}          # str(id) -> {coin, strike, expiry, venue, qty, cash, fills, ...}
        self.fills: list[dict] = []
        self.settled: list[dict] = []
        self.retry: dict[str, list] = {}
        self.cash = self.start_cash
        self.peak = self.start_cash
        self.killed, self.kill_reason = False, None
        self.status, self.error = ("waiting" if self.enabled else "off"), None
        self.ws_state = {"connected": False, "subs": 0, "trades": 0, "books": 0, "error": None}
        self._meta_at = 0.0
        self._bars_at: dict[str, float] = {}
        self._saved_at = 0.0
        self._dirty = False
        self._snapshot_due: set = set()          # coins whose next trades message is the on-subscribe history
        self._tids: deque = deque(maxlen=5000)
        self._tid_set: set = set()
        self.hist: dict[int, deque] = {}
        self._hist_at = 0.0
        self._bars_thread = False
        self._regime_status = None
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.error = f"could not read {self.path.name}: {exc}; starting a fresh book"
            return
        self.cash = float(d.get("cash", self.start_cash))
        self.start_cash = float(d.get("start_cash", self.start_cash))
        self.peak = float(d.get("peak", self.cash))
        self.killed, self.kill_reason = bool(d.get("killed")), d.get("kill_reason")
        self.pos, self.fills, self.settled = d.get("pos", {}), d.get("fills", []), d.get("settled", [])
        self.retry = d.get("retry", {})
        for s in self.settled:                     # markets settled before the regime alarm existed
            if "surprise" not in s and s.get("y") is not None:
                fs = [f for f in self.fills if f.get("id") == s["id"] and f.get("fair") is not None]
                q = sum(f["qty"] for f in fs)
                s["surprise"] = round(sum(f["side"] * (s["y"] - f["fair"]) * f["qty"] for f in fs) / q, 4) if q else None
        self.kill_log = [float(x) for x in d.get("kill_log", [])]
        firsts = [x["ts"] for x in self.fills[:1] + self.settled[:1] if x.get("ts")]
        self.since = d.get("since") or (min(firsts) if firsts else None)
        try:
            self._apply_size(float(d.get("size_mult", 1.0)))
        except (TypeError, ValueError):
            pass

    def _save(self, force: bool = False) -> None:
        if self.path is None:
            return
        now = self.clock()
        if not force and not (self._dirty and now - self._saved_at >= 10):
            return
        with self.lock:
            data = json.dumps({"cash": self.cash, "start_cash": self.start_cash, "peak": self.peak,
                               "killed": self.killed, "kill_reason": self.kill_reason, "pos": self.pos,
                               "fills": self.fills[-MAX_FILLS:], "settled": self.settled[-MAX_SETTLED:],
                               "retry": self.retry, "size_mult": self.size_mult, "since": self.since,
                               "kill_log": self.kill_log[-200:]})
            self._dirty = False
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self.path)
        self._saved_at = now

    # ------------------------------------------------------------------ operator controls
    def kill(self, reason: str) -> None:
        with self.lock:
            if not self.killed:
                self.killed, self.kill_reason = True, reason
                self.kill_log.append(self.clock())
                self.quotes.clear()
                self.notify("risk", f"HIP-4 market maker stopped: {reason}. Open positions are held to settlement.")
            self._dirty = True
        self._save(force=True)

    def resume(self) -> None:
        with self.lock:
            self.killed, self.kill_reason = False, None
            self.peak = self.equity()
            self._dirty = True
        self._save(force=True)

    def _apply_size(self, mult: float) -> None:
        if not (math.isfinite(mult) and 0.25 <= mult <= self.max_mult):
            raise ValueError(f"size multiplier must be between 0.25 and {self.max_mult:g}")
        self.size_mult = mult
        self.quote_size = self.base["quote_size"] * mult
        self.max_pos = self.base["max_position"] * mult
        self.max_risk = self.base["max_risk_usd"] * mult

    def set_size(self, mult: float) -> dict:
        """Operator bet size. Takes effect at the next requote (within a second); open positions are untouched, and a
        smaller size simply stops adding to positions that are now above the new limit."""
        with self.lock:
            before = self.size_mult
            self._apply_size(float(mult))
            self._dirty = True
        if mult != before:
            self.notify("risk", f"HIP-4 maker size {before:g}x -> {mult:g}x: {self.quote_size:g} contracts/quote, "
                                f"max {self.max_pos:g}/market, risk cap ${self.max_risk:,.0f}")
        self._save(force=True)
        return self.size_info()

    def size_info(self) -> dict:
        return {"multiplier": self.size_mult, "max_multiplier": self.max_mult, "quote_size": self.quote_size,
                "max_position": self.max_pos, "max_risk_usd": self.max_risk, "base": dict(self.base)}

    def reset(self) -> None:
        with self.lock:
            self.cash = self.peak = self.start_cash
            self.pos, self.fills, self.settled, self.retry, self.quotes = {}, [], [], {}, {}
            self.killed, self.kill_reason = False, None
            self.since, self.kill_log = None, []                    # a reset restarts the go-live clock
            self._dirty = True
        self._save(force=True)

    # ------------------------------------------------------------------ accounting
    def _mark(self, key: str, p: dict) -> float:
        f = self.fairs.get(int(key))
        if f is not None:
            return f[0]
        b = self.books.get(int(key))
        if b and b.get("bid") is not None and b.get("ask") is not None:
            return (b["bid"] + b["ask"]) / 2
        return p.get("last_mark", 0.5)

    def equity(self) -> float:
        with self.lock:
            return self.cash + sum(p["qty"] * self._mark(k, p) for k, p in self.pos.items())

    def risk(self) -> float:
        """Worst-case loss across open markets: each pays qty x (0 or 1) on top of the cash it has already moved."""
        with self.lock:
            return sum(max(0.0, -(p["cash"] + min(0.0, p["qty"]))) for p in self.pos.values())

    def _fill(self, mid: int, side: int, px: float, qty: float, ts: float, how: str, fee_rate: float | None = None) -> None:
        m = self.markets.get(mid) or {}
        key = str(mid)
        p = self.pos.setdefault(key, {"coin": m.get("coin"), "strike": m.get("strike"), "expiry": m.get("expiry"),
                                      "venue": m.get("venue"), "qty": 0.0, "cash": 0.0, "fees": 0.0, "fills": 0,
                                      "volume": 0.0})
        p["last_fill"] = ts
        if self.since is None:
            self.since = ts
        fee = (self.fee if fee_rate is None else fee_rate) * qty
        flow = -side * px * qty - fee
        p["qty"] += side * qty
        p["cash"] += flow
        p["fees"] += fee
        p["fills"] += 1
        p["volume"] += qty
        self.cash += flow
        fair = self.fairs.get(mid, (None, 0))[0]
        self.fills.append({"ts": ts, "id": mid, "coin": p["coin"], "strike": p["strike"], "expiry": p["expiry"],
                           "side": side, "px": round(px, 6), "qty": qty, "fair": fair, "how": how,
                           "edge": round((fair - px) * side, 4) if fair is not None else None,
                           "mo_at": ts + MARKOUT_S, "markout": None, "inv_after": p["qty"]})
        if len(self.fills) > MAX_FILLS:
            self.fills = self.fills[-MAX_FILLS:]
        self._dirty = True

    def on_trade(self, mid: int, px: float, sz: float, aggressor_buy: bool, ts: float) -> None:
        """One real trade on market `mid`'s Yes book, `ts` = when WE received it (local clock). A seller printing
        through our bid fills it; a buyer printing through our ask fills it. Price = our quote, size = min(trade, what
        is left on our quote). The quote only counts once it could have reached the exchange: a trade must arrive at
        least one network round trip (latency_s) after we placed the quote. Local times on both sides, so a PC clock
        that is off from the exchange's (this one ran 3.4 s slow) cannot make early trades look like fills."""
        with self.lock:
            q = self.quotes.get(mid)
            if q is None or self.killed:
                return
            lat, left = self.latency_s, sz
            # a sweep fills our best quote first, then (if it went that far) the deeper level-2 quote
            for lvl in ("", "2"):
                if left <= 0:
                    break
                if not aggressor_buy:
                    k, kb = "bid" + lvl, "bid" + lvl + "_sz"
                    if q.get(k) is None or q.get(kb, 0) <= 0 or ts < q["placed_" + k] + lat:
                        continue
                    if px < q[k] - 1e-9 or (lvl == "" and abs(px - q[k]) <= 1e-9 and q["bid_best"]):
                        take = min(left, q[kb])
                        q[kb] -= take
                        left -= take
                        self._fill(mid, +1, q[k], take, ts, ("level 2 " if lvl else "") +
                                   ("through" if px < q[k] - 1e-9 else "at, alone at top"))
                else:
                    k, ka = "ask" + lvl, "ask" + lvl + "_sz"
                    if q.get(k) is None or q.get(ka, 0) <= 0 or ts < q["placed_" + k] + lat:
                        continue
                    if px > q[k] + 1e-9 or (lvl == "" and abs(px - q[k]) <= 1e-9 and q["ask_best"]):
                        take = min(left, q[ka])
                        q[ka] -= take
                        left -= take
                        self._fill(mid, -1, q[k], take, ts, ("level 2 " if lvl else "") +
                                   ("through" if px > q[k] + 1e-9 else "at, alone at top"))

    def _snipe(self, now: float) -> None:
        """The sniper (rules fixed 2026-10-06 BEFORE any held-out data, from a 2.4 h backtest): when the Yes book's
        ask is >= snipe_edge below our MODEL fair (or its bid that far above), with stop_min..max_tte left, send a taker
        order that lands snipe_latency_s later and fills only against the book AT THAT TIME (the quote may be gone).
        Whole contracts, the $10 minimum, per-market and total risk caps, taker fee; held to settlement."""
        if self.killed:
            self._pending.clear()
            return
        with self.lock:
            risk = self.risk()
            for mid, o in list(self._pending.items()):
                if now < o["due"]:
                    continue
                del self._pending[mid]
                b, qty = self.books.get(mid), self.pos.get(str(mid), {}).get("qty", 0.0)
                if b is None or now - b["ts"] > self.book_stale_s:
                    self.snipe_stats["missed"] += 1
                    self.cool[mid] = now + self.snipe_cooldown_s
                    continue
                if o["side"] > 0:
                    px, avail, room = b.get("ask"), b.get("ask_sz", 0), self.max_pos - qty
                    ok = px is not None and px <= o["limit"] + 1e-9
                else:
                    px, avail, room = b.get("bid"), b.get("bid_sz", 0), self.max_pos + qty
                    ok = px is not None and px >= o["limit"] - 1e-9
                if not ok:
                    self.snipe_stats["missed"] += 1          # the mispricing was gone by the time we arrived
                    self.cool[mid] = now + self.snipe_cooldown_s
                    continue
                pay = px if o["side"] > 0 else 1 - px        # a sell of Yes is paid as a No buy at 1 - bid
                cap = min(avail, room, max(0.0, (self.max_risk - risk) / max(pay, 1e-6)))
                n = self._live_size(pay, min(self.quote_size, cap), cap)
                if n <= 0:
                    # e.g. a 12-contract dust order at 92c: the $10 minimum can't be met at that price, so it is
                    # not a real opportunity (live trading would fail on it too)
                    self.snipe_stats["skipped"] += 1
                    self.cool[mid] = now + self.snipe_cooldown_s
                    continue
                self._fill(mid, o["side"], px, n, now, "snipe", fee_rate=self.taker_fee)
                f = self.fills[-1]
                f["fair"], f["edge"] = round(o["fair"], 4), round((o["fair"] - px) * o["side"], 4)
                self.snipe_stats["filled"] += 1
                risk += n * pay
                self.cool[mid] = now + self.snipe_cooldown_s
                # the live pilot follows FILLS (fire_on "fill"): only when the paper book found real depth at this price
                if self.on_signal is not None and self.live_fire_on == "fill":
                    try:
                        self.on_signal(mid, o["side"], px, o["fair"], dict(self.markets.get(mid) or {}))
                    except Exception as exc:  # the pilot must never break the paper book
                        self.error = f"live pilot: {type(exc).__name__}: {exc}"[:240]
            sides_by_coin: dict[str, str] = {}
            for mid, m in self.markets.items():
                if mid in self._pending or now < self.cool.get(mid, 0):
                    continue
                if not (self.stop_s < m["expiry"] - now <= self.max_tte_s):
                    continue
                p, b = self.models.get(mid), self.books.get(mid)
                if p is None or b is None or now - b["ts"] > self.book_stale_s or not (self.min_p <= p <= 1 - self.min_p):
                    continue
                sides = self.snipe_sides
                if sides == "auto":
                    sides = sides_by_coin.get(m["coin"])
                    if sides is None:
                        sides = sides_by_coin[m["coin"]] = self._auto_sides(m["coin"], now)
                qty = self.pos.get(str(mid), {}).get("qty", 0.0)
                inb = lambda px: px is not None and self.band_lo - 1e-9 <= px <= self.band_hi + 1e-9   # noqa: E731
                if (sides in ("both", "buy") and inb(b.get("ask")) and b["ask"] <= p - self.snipe_edge
                        and qty < self.max_pos):
                    self._pending[mid] = {"side": 1, "limit": b["ask"], "due": now + self.snipe_latency_s, "fair": p}
                elif (sides in ("both", "sell") and inb(b.get("bid")) and b["bid"] >= p + self.snipe_edge
                        and qty > -self.max_pos):
                    self._pending[mid] = {"side": -1, "limit": b["bid"], "due": now + self.snipe_latency_s, "fair": p}
                else:
                    continue
                self.snipe_stats["signals"] += 1
                # fire_on "signal": the pilot sends now (a real order takes its own latency to land), but only when
                # the top of the book alone is a real order (>= the $10 minimum), never on dust
                if self.on_signal is not None and self.live_fire_on == "signal":
                    o = self._pending[mid]
                    sz = b.get("ask_sz", 0) if o["side"] > 0 else b.get("bid_sz", 0)
                    pay = o["limit"] if o["side"] > 0 else 1 - o["limit"]
                    if sz * pay >= max(self.min_notional, 10.0) - 1e-9:
                        try:
                            self.on_signal(mid, o["side"], o["limit"], p,
                                           dict(m, detect=now, book_et=b.get("et"), path="step"))
                        except Exception as exc:  # the pilot must never break the paper book
                            self.error = f"live pilot: {type(exc).__name__}: {exc}"[:240]

    # ------------------------------------------------------------------ dry run: would the live IOC have filled?
    WATCH_MAX_S = 5.0

    def watch_quote(self, mid: int, side: int, yes_limit: float, cb) -> None:
        """Watch the Yes book of `mid` from now on: cb(gone_et, gone_local) the first time the quote our IOC needed is
        no longer there (a Yes bid >= yes_limit for a sell, a Yes ask <= yes_limit for a buy), with the exchange time of
        that book; cb(None, None) if it is still there WATCH_MAX_S later (expire_watches)."""
        with self.lock:
            self._watch.setdefault(mid, []).append({"side": side, "limit": yes_limit, "cb": cb, "t": self.clock()})

    def _check_watch(self, mid: int, now: float, et) -> None:
        b = self.books.get(mid) or {}
        with self.lock:
            ws, keep = self._watch.get(mid, []), []
            done = []
            for w in ws:
                there = (b.get("bid") is not None and b["bid"] >= w["limit"] - 1e-9) if w["side"] < 0 else \
                    (b.get("ask") is not None and b["ask"] <= w["limit"] + 1e-9)
                (keep if there else done).append(w)
            self._watch[mid] = keep
        for w in done:
            try:
                w["cb"](float(et) if et is not None else None, now)
            except Exception as exc:
                self.error = f"dry run: {type(exc).__name__}: {exc}"[:240]

    def expire_watches(self, now: float) -> None:
        """Quotes still there WATCH_MAX_S after the signal: the IOC would have found them (cb(None, None))."""
        done = []
        with self.lock:
            for mid, ws in list(self._watch.items()):
                keep = [w for w in ws if now - w["t"] < self.WATCH_MAX_S]
                done += [w for w in ws if now - w["t"] >= self.WATCH_MAX_S]
                self._watch[mid] = keep
        for w in done:
            try:
                w["cb"](None, None)
            except Exception as exc:
                self.error = f"dry run: {type(exc).__name__}: {exc}"[:240]

    def _fast_live(self, mid: int, now: float) -> None:
        """Event-driven LIVE check, run the moment a market's book update arrives (WebSocket thread). The paper sniper
        only looks every quote_every_s (0.5 s), so a stale quote waited ~250 ms on average before live even saw it;
        the truly stale ones are taken by faster bots in that time (2026-10-08: live filled 2W/6L while missing
        paper's winners). Same entry rules as _snipe (window, fair band, sides, price band, edge, paper position
        gate, top of book >= $10); the paper book is untouched; the pilot's own cooldown dedups the later paper signal."""
        try:
            with self.lock:
                if self.killed:
                    return
                m, p, b = self.markets.get(mid), self.models.get(mid), self.books.get(mid)
                if m is None or p is None or b is None or not (self.stop_s < m["expiry"] - now <= self.max_tte_s):
                    return
                try:                                  # the fair on the perp price NOW, not as of the last 0.5 s step
                    spot = self.spot_fn(m["coin"])
                    fresh = self.model.fair(m["coin"], spot, m["strike"], now, m["expiry"], m["twap_s"]) if spot else None
                    if fresh is not None:
                        p = fresh
                except Exception:
                    pass
                if not (self.min_p <= p <= 1 - self.min_p):
                    return
                sides = self.snipe_sides
                if sides == "auto":
                    sides = self._auto_sides(m["coin"], now)
                qty = self.pos.get(str(mid), {}).get("qty", 0.0)
                inb = lambda px: px is not None and self.band_lo - 1e-9 <= px <= self.band_hi + 1e-9   # noqa: E731
                if (sides in ("both", "buy") and inb(b.get("ask")) and b["ask"] <= p - self.snipe_edge
                        and qty < self.max_pos and b.get("ask_sz", 0) * b["ask"] >= max(self.min_notional, 10.0) - 1e-9):
                    side, limit = 1, b["ask"]
                elif (sides in ("both", "sell") and inb(b.get("bid")) and b["bid"] >= p + self.snipe_edge
                        and qty > -self.max_pos and b.get("bid_sz", 0) * (1 - b["bid"]) >= max(self.min_notional, 10.0) - 1e-9):
                    side, limit = -1, b["bid"]
                else:
                    return
                market = dict(m, detect=now, book_et=b.get("et"), path="fast")
            self.on_signal(mid, side, limit, p, market)
        except Exception as exc:  # the pilot must never break the feed
            self.error = f"live fast path: {type(exc).__name__}: {exc}"[:240]

    def _auto_sides(self, coin: str, now: float) -> str:
        try:
            spot = self.spot_fn(coin)
        except Exception:
            spot = None
        ret = trend_return(self.model.bars.get(coin), spot, now, self.trend_lookback_s)
        sides = trend_sides(ret, self.trend_threshold, self.trend_neutral)
        self.trend_now[coin] = {"ret": None if ret is None else round(ret, 5), "sides": sides, "ts": now}
        return sides

    def _exits(self, now: float) -> None:
        """Early exits, as a taker against the REAL book top (size-limited, taker fee). Per contract, against the
        position's break-even price: take profit when closing now would make >= take_profit (a big favourable jump),
        stop out when it would lose >= stop_loss. After an exit the market sits out `cooldown_s` (no new quotes)."""
        if not self.exits_on or self.killed:
            return
        with self.lock:
            for key, p in list(self.pos.items()):
                qty, mid = p["qty"], int(key)
                b = self.books.get(mid)
                if abs(qty) < 1e-9 or mid not in self.markets or b is None or now - b["ts"] > self.book_stale_s:
                    continue
                if now - p.get("last_fill", 0) < self.exit_min_hold_s or now >= (p["expiry"] or 0) - 20:
                    continue
                if qty > 0:
                    px, avail = b.get("bid"), b.get("bid_sz", 0)
                    per = (px - (-p["cash"] / qty)) if px is not None else None       # sell at the bid vs break-even
                else:
                    px, avail = b.get("ask"), b.get("ask_sz", 0)
                    per = (p["cash"] / -qty - px) if px is not None else None         # buy back at the ask
                if per is None or b.get("bid") is None or b.get("ask") is None or b["ask"] - b["bid"] > 0.25:
                    continue
                how = "take-profit" if per >= self.take_profit else "stop-loss" if per <= -self.stop_loss else None
                n = min(abs(qty), avail)
                if how is None or n <= 0:
                    continue
                self._fill(mid, -1 if qty > 0 else 1, px, n, now, how, fee_rate=self.taker_fee)
                self.fills[-1]["exit_pc"] = round(per, 4)
                self.cool[mid] = now + self.exit_cooldown_s
                self.quotes.pop(mid, None)
                self.notify("trade", f"HIP-4 {how}: {p['coin']} >{p['strike']:g} {'sold' if qty > 0 else 'bought back'} "
                                     f"{n:g} @ {px * 100:.1f}c ({per * 100:+.1f}c/contract)")

    # ------------------------------------------------------------------ WebSocket (Yes books and trades)
    def handle(self, msg: dict, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        ch, data = msg.get("channel"), msg.get("data")
        if ch == "l2Book" and isinstance(data, dict):
            mid, yes = self._coin_id(data.get("coin"))
            if mid is None or not yes:
                return
            lv = data.get("levels") or [[], []]
            bids, asks = (lv[0] if lv else []), (lv[1] if len(lv) > 1 else [])
            self.books[mid] = {"bid": float(bids[0]["px"]) if bids else None, "ask": float(asks[0]["px"]) if asks else None,
                               "bid_sz": float(bids[0]["sz"]) if bids else 0.0,
                               "ask_sz": float(asks[0]["sz"]) if asks else 0.0, "ts": now, "et": data.get("time"),
                               # the levels behind the top (the shadow lab's fill check walks them, as an IOC would)
                               "bid_lv": [(float(x["px"]), float(x["sz"])) for x in bids[:5]],
                               "ask_lv": [(float(x["px"]), float(x["sz"])) for x in asks[:5]]}
            self.ws_state["books"] += 1
            if self._watch.get(mid):
                self._check_watch(mid, now, data.get("time"))
            if self.strategy == "sniper" and self.on_signal is not None and self.live_fire_on == "signal":
                self._fast_live(mid, now)
            for hook in self.book_hooks:
                try:
                    hook(mid, now)
                except Exception as exc:  # a lab must never break the feed
                    self.error = f"book hook: {type(exc).__name__}: {exc}"[:240]
        elif ch == "trades" and isinstance(data, list):
            for t in data:
                mid, yes = self._coin_id(t.get("coin"))
                tid = t.get("tid")
                if mid is None or not yes or tid in self._tid_set:
                    continue
                if len(self._tids) == self._tids.maxlen:
                    self._tid_set.discard(self._tids[0])
                self._tids.append(tid)
                self._tid_set.add(tid)
                if t["coin"] in self._snapshot_due:          # the first trades message after subscribing is history
                    continue
                self.ws_state["trades"] += 1
                self._trades.setdefault(mid, deque(maxlen=500)).append(now)
                self.on_trade(mid, float(t["px"]), float(t["sz"]), t.get("side") == "B", now)
            for coin in {t.get("coin") for t in data}:
                self._snapshot_due.discard(coin)

    @staticmethod
    def _coin_id(coin) -> tuple[int | None, bool]:
        if not isinstance(coin, str) or not coin.startswith("#") or not coin[1:].isdigit():
            return None, False
        n = int(coin[1:])
        return n // 10, n % 10 == 0

    def _ws_loop(self, stop: threading.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            ws, subbed = None, set()
            try:
                ws = self.ws_factory(self.ws_url, timeout=10)
                ws.settimeout(1.0)
                self.ws_state.update(connected=True, error=None)
                backoff, last_ping = 1.0, time.monotonic()
                while not stop.is_set():
                    want = {f"#{mid * 10}" for mid in list(self.markets)}
                    for coin in want - subbed:
                        for typ in ("l2Book", "trades"):
                            ws.send(json.dumps({"method": "subscribe", "subscription": {"type": typ, "coin": coin}}))
                        self._snapshot_due.add(coin)
                    for coin in subbed - want:
                        for typ in ("l2Book", "trades"):
                            ws.send(json.dumps({"method": "unsubscribe", "subscription": {"type": typ, "coin": coin}}))
                        self._snapshot_due.discard(coin)
                    subbed = want
                    self.ws_state["subs"] = len(subbed)
                    if time.monotonic() - last_ping > 30:
                        ws.send(json.dumps({"method": "ping"}))
                        last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except socket.timeout:
                        continue
                    try:
                        self.handle(json.loads(raw))
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        self.ws_state["error"] = f"bad message: {exc}"[:200]
            except (OSError, WebSocketError, ValueError) as exc:
                self.ws_state["error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                self.ws_state["connected"] = False
                self.books.clear()                   # no stale books across a reconnect: quoting waits for fresh ones
                if ws is not None:
                    ws.close()
            stop.wait(backoff)
            backoff = min(30.0, backoff * 2)

    # ------------------------------------------------------------------ main loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self._ws_loop, args=(stop,), name="hip4-mm-ws", daemon=True).start()
        self._bars_thread = True
        threading.Thread(target=self._bars_loop, args=(stop,), name="hip4-mm-bars", daemon=True).start()
        while not stop.is_set():
            try:
                self.step()
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:240]
            stop.wait(self.every_s)
        self._save(force=True)

    def step(self) -> None:
        now = self.clock()
        if now - self._meta_at >= META_EVERY_S:
            meta = self.rest.post({"type": "outcomeMeta"}) or {}
            wanted = _Every() if self.all_coins else self.coins
            listed = [m for m in (parse_market(o, wanted) for o in meta.get("outcomes", [])) if m and primary(m)]
            with self.lock:
                self.markets = {m["id"]: m for m in listed if m["expiry"] - now <= self.max_tte_s + META_EVERY_S}
                for mid in [k for k in self.books if k not in self.markets]:
                    self.books.pop(mid, None)
            self._meta_at = now
        if not self._bars_thread:                     # tests and backtests call step() without run(): fetch inline
            for coin in self.wanted_coins():
                self._refresh_bars(coin, now)
        self._update_fairs(now)
        self._markouts(now)
        self._settle(now)
        self._risk(now)
        if self.strategy == "sniper":
            self._snipe(now)
            self.quotes = {}
            if self.on_step is not None:
                try:
                    self.on_step(now)
                except Exception as exc:
                    self.error = f"live pilot: {type(exc).__name__}: {exc}"[:240]
        else:
            self._exits(now)
            if self.quoting:
                self._requote(now)
            else:
                self.quotes = {}
        self._record_history(now)
        self.status = ("killed" if self.killed else "sniping" if self.strategy == "sniper"
                       else "paused" if not self.quoting else "quoting" if self.quotes else "watching")
        self._save()

    def wanted_coins(self) -> set:
        return {m["coin"] for m in list(self.markets.values())} | {p["coin"] for p in list(self.pos.values())}

    def _bars_loop(self, stop: threading.Event) -> None:
        """5-minute bars for every coin we quote, in their own thread: with 20 underlyings the downloads (rate-limited
        REST) would otherwise stall the half-second quote loop."""
        while not stop.is_set():
            for coin in sorted(self.wanted_coins()):
                if stop.is_set():
                    return
                self._refresh_bars(coin, self.clock())
            stop.wait(5.0)

    def _record_history(self, now: float) -> None:
        """Every 2 s per market, for the visualizer's market chart (90 minutes kept, in memory only)."""
        if now - self._hist_at < 2:
            return
        self._hist_at = now
        with self.lock:
            for mid, m in self.markets.items():
                b, f, q = self.books.get(mid, {}), self.fairs.get(mid), self.quotes.get(mid)
                spot = None
                try:
                    spot = self.spot_fn(m["coin"])
                except Exception:
                    pass
                h = self.hist.setdefault(mid, deque(maxlen=2700))
                h.append([round(now, 1), b.get("bid"), b.get("ask"), round(f[0], 4) if f else None,
                          round(self.models[mid], 4) if mid in self.models else None,
                          q["bid"] if q and q["bid_sz"] > 0 else None, q["ask"] if q and q["ask_sz"] > 0 else None,
                          spot])
            for mid in [k for k in self.hist if k not in self.markets and not str(k) in self.pos]:
                if self.hist[mid] and now - self.hist[mid][-1][0] > 5400:
                    del self.hist[mid]

    def _refresh_bars(self, coin: str, now: float) -> None:
        """15 days of 5-minute bars: one full download every 6 hours, otherwise just the bars since the last one."""
        bars = self.model.bars.get(coin)
        try:
            if bars is None and coin not in self._bars_at:            # a restart: start from the disk cache if recent
                cached = self._load_bars_cache(coin, now)
                if cached:
                    self.model.set_bars(coin, cached)
                    self._bars_at[coin] = now
                    bars = cached
            if bars is None or now - self._bars_at.get(coin, 0) > 6 * 3600:
                full = self.rest.candles(coin, "5m", 15 * 1440 + 10, now)
                self.model.set_bars(coin, full)
                self._bars_at[coin] = now
                self._save_bars_cache(coin, full)
            elif now - (bars[-1].ts + 300) >= 300:
                fresh = self.rest.candles(coin, "5m", int((now - bars[-1].ts) / 60) + 10, now)
                self.model.set_bars(coin, (bars + [c for c in fresh if c.ts > bars[-1].ts])[-(15 * 288 + 4):])
        except Exception as exc:
            self.error = f"{coin} candles: {exc}"[:240]

    def _bars_cache(self, coin: str) -> Path | None:
        return self.path.parent / "tape" / "bars" / f"{coin.replace(':', '_')}.json" if self.path is not None else None

    def _load_bars_cache(self, coin: str, now: float):
        """The same cache the backtester keeps: 15 days of bars, usable when its newest bar is under 6 hours old
        (the incremental fetch then adds the rest), so a restart quotes in seconds instead of minutes."""
        p = self._bars_cache(coin)
        try:
            rows = json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else None
        except (OSError, ValueError):
            return None
        if not rows or now - rows[-1][0] > 6 * 3600:
            return None
        from .market import Candle
        return [Candle(*r) for r in rows if r[0] >= now - 15 * 86400 - 3600]

    def _save_bars_cache(self, coin: str, bars: list) -> None:
        p = self._bars_cache(coin)
        if p is None or not bars:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps([[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in bars]), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass

    def _update_fairs(self, now: float) -> None:
        spots = {}
        for coin in {m["coin"] for m in self.markets.values()}:
            try:
                spots[coin] = self.spot_fn(coin)
            except Exception:
                spots[coin] = None
        fairs, models = {}, {}
        for mid, m in self.markets.items():
            spot = spots.get(m["coin"])
            if spot and now < m["expiry"]:
                p = self.model.fair(m["coin"], spot, m["strike"], now, m["expiry"], m["twap_s"])
                if p is None:
                    continue
                models[mid] = p
                # the market's own mid forecast as well as our model did (2026-10-06 scoring), so a real two-sided
                # book gets a say: quoting on the model alone would be a directional bet against the market
                b = self.books.get(mid)
                if b and b["bid"] is not None and b["ask"] is not None and b["ask"] - b["bid"] <= self.blend_spread \
                        and now - b["ts"] <= self.book_stale_s:
                    p = self.market_weight * (b["bid"] + b["ask"]) / 2 + (1 - self.market_weight) * p
                fairs[mid] = (p, now)
        with self.lock:
            self.fairs, self.models = fairs, models
            for k, p in self.pos.items():
                if int(k) in fairs:
                    p["last_mark"] = fairs[int(k)][0]

    def _markouts(self, now: float) -> None:
        with self.lock:
            for f in self.fills:
                if f["markout"] is None and now >= f["mo_at"] and f.get("fair") is not None:
                    cur = self.fairs.get(f["id"])
                    if cur is not None:
                        f["markout"] = round((cur[0] - f["px"]) * f["side"], 4)
                    elif now >= f["expiry"]:
                        f["mo_at"] = float("inf")       # expired inside the window: settlement tells the story
                        f["markout"] = None

    def _settle(self, now: float) -> None:
        due = [k for k, p in self.pos.items() if now >= (p["expiry"] or 0) + 90 and int(k) not in self.markets]
        calls = 0
        for key in due:
            nxt, tries = self.retry.get(key, [0.0, 0])
            if now < nxt:
                continue
            p = self.pos[key]
            if now > p["expiry"] + GIVE_UP_S:
                y, details = None, "never settled"
            else:
                if calls >= 4:
                    continue
                calls += 1
                try:
                    res = self.rest.post({"type": "settledOutcome", "outcome": int(key)})
                except Exception as exc:
                    res = None
                    self.error = f"settle #{key}: {exc}"[:240]
                if not res:
                    self.retry[key] = [now + min(3600.0, 90.0 * 2 ** tries), tries + 1]
                    continue
                y, details = float(res.get("settleFraction", "nan")), res.get("details")
                if not math.isfinite(y):
                    y = None
            self.settle_one(key, y, details, now)

    def settle_one(self, key: str, y: float | None, details, now: float) -> None:
        """Pay out one market. y = settle fraction (1 = Yes paid); None = void: premiums refunded, fees stay paid."""
        with self.lock:
            p = self.pos.get(key)
            if p is None:
                return
            payout = p["qty"] * y if y is not None else -(p["cash"] + p["fees"])
            self.cash += payout
            pnl = p["cash"] + payout
            # how the outcome compared with our model at the fills: side * (y - fair), contract-weighted. The regime
            # alarm (regime()) watches it: the sell edge depends on the underlying not drifting up
            fs = [f for f in self.fills if f["id"] == int(key) and f.get("fair") is not None]
            q = sum(f["qty"] for f in fs)
            surprise = sum(f["side"] * (y - f["fair"]) * f["qty"] for f in fs) / q if q and y is not None else None
            self.settled.append({"id": int(key), "coin": p["coin"], "strike": p["strike"], "expiry": p["expiry"],
                                 "venue": p.get("venue"), "y": y, "qty": p["qty"], "fills": p["fills"],
                                 "volume": p["volume"], "fees": round(p["fees"], 4), "pnl": round(pnl, 4),
                                 "ts": now, "details": details,
                                 "surprise": None if surprise is None else round(surprise, 4)})
            self.settled = self.settled[-MAX_SETTLED:]
            del self.pos[key]
            self.retry.pop(key, None)
            self._dirty = True
        reg = self.regime()
        if reg["status"] == "turned" and self._regime_status != "turned":
            self.notify("warn", f"{self.name}: the market regime may have turned - over the last {reg['markets']} "
                                f"settled sell markets the outcomes ran {reg['surprise_c']:+.1f}c/contract against "
                                f"our model (alarm at {REGIME_ALARM_C:g}c). Sell-only depends on prices not drifting up.")
        self._regime_status = reg["status"]

    def regime(self, need: int = 30, window: int = 60) -> dict:
        """The smart alarm for sides=sell: over the last `window` settled markets where we were short Yes (at least
        `need`), the average of side * (outcome - model fair at our fills), each market weighted equally. About 0 or
        above: the sell edge is holding. "turned" needs the shortfall to be both big (<= REGIME_ALARM_C) and real
        (t <= -2): binary outcomes make 30 markets swing +-8c by luck alone (2026-10-08: -9.9c over 30 had t = -1.2,
        and the 72-market mean was -1.5c). A big dip that is not yet significant reads "watch"."""
        rows = [s for s in self.settled if s.get("surprise") is not None and (s.get("qty") or 0) < 0][-window:]
        n = len(rows)
        if not n:
            return {"status": "collecting", "markets": 0, "need": need, "surprise_c": None, "se_c": None, "t": None,
                    "pnl": 0.0, "won": 0, "edge_c": None, "edge_markets": 0, "size_up_ok": False}
        xs = [s["surprise"] * 100 for s in rows]
        avg = sum(xs) / n
        se = (sum((x - avg) ** 2 for x in xs) / (n - 1)) ** 0.5 / math.sqrt(n) if n > 1 else None
        t = avg / se if se else None
        if n < need:
            status = "collecting"
        elif avg <= REGIME_ALARM_C and t is not None and t <= -2:
            status = "turned"
        elif avg <= REGIME_ALARM_C / 2 or (t is not None and t <= -1):
            status = "watch"
        else:
            status = "ok"
        # the size gauge (research 2026-10-08): net $ per contract of the settled sell markets, contract-weighted. On
        # the tape, every 6 h stretch where this paid >= +14c made the 1x live pilot money, ~0c made it flat; go back
        # above 1x only when it is >= +10c over >= 30 markets
        last = [s for s in self.settled if (s.get("qty") or 0) < 0 and s.get("y") is not None][-window:]
        ct = sum(abs(s["qty"]) for s in last)
        edge_c = sum(s["pnl"] for s in last) / ct * 100 if ct else None
        return {"status": status, "markets": n, "need": need, "surprise_c": round(avg, 2),
                "se_c": None if se is None else round(se, 2), "t": None if t is None else round(t, 2),
                "pnl": round(sum(s["pnl"] for s in rows), 2), "won": sum(1 for s in rows if s["pnl"] > 0),
                "edge_c": None if edge_c is None else round(edge_c, 2), "edge_markets": len(last),
                "size_up_ok": bool(edge_c is not None and len(last) >= 30 and edge_c >= 10)}

    def _risk(self, now: float) -> None:
        eq = self.equity()
        with self.lock:
            self.peak = max(self.peak, eq)
            dd = 1 - eq / self.peak if self.peak > 0 else 0.0
        if not self.killed and self.kill_dd > 0 and dd >= self.kill_dd:      # kill_drawdown 0 = no kill switch
            self.kill(f"Max drawdown {dd:.2%} ≥ {self.kill_dd:.0%}")

    def _sticky(self, old: float | None, new: float, fair: float, thr: float, sign: int) -> float:
        """Keep the resting price `old` (sign +1 = a bid, -1 = an ask) while it is within `thr` of the new target and
        still at least half the edge away from fair on its own side; otherwise move to the new target."""
        if old is None:
            return new
        safe = (fair - old) * sign >= self.edge / 2 - 1e-9
        return old if safe and abs(old - new) < thr - 1e-9 else new

    def _live_size(self, pay_px: float, sz: float, room: float) -> float:
        """Whole contracts, and at least min_notional of value per order: a quote below it is raised to the minimum
        when the position limit leaves room for that, otherwise it is not quoted (it could not exist live)."""
        sz = math.floor(sz + 1e-9)
        if sz <= 0 or not self.min_notional:
            return float(max(0, sz))
        if pay_px <= 0:
            return 0.0
        if pay_px * sz >= self.min_notional - 1e-9:
            return float(sz)
        need = math.ceil(self.min_notional / pay_px - 1e-9)
        return float(need) if need <= math.floor(room + 1e-9) else 0.0

    def _fair_drift(self, mid: int, now: float) -> float | None:
        """How far fair has moved over the last guard_s (from the fair history kept every quote step)."""
        h = self._fair_hist.get(mid)
        if not h:
            return None
        old = next((v for t, v in h if now - t <= self.guard_s), None)
        return None if old is None else self.fairs.get(mid, (old,))[0] - old

    def _requote(self, now: float) -> None:
        for mid, (f, _) in self.fairs.items():                       # fair history for the adverse-move guard
            h = self._fair_hist.setdefault(mid, deque(maxlen=400))
            if not h or now - h[-1][0] >= 1.0:
                h.append((now, f))
        for mid in [k for k in self._fair_hist if k not in self.markets]:
            del self._fair_hist[mid]
        # focus: only the busiest markets (real trades in the last 2 h) - fewer resting orders, less order budget
        focus = None
        if self.max_markets:
            busy = {m: sum(1 for t in self._trades.get(m, ()) if now - t <= 7200) for m in self.markets}
            held = {int(k) for k, p in self.pos.items() if abs(p["qty"]) > 1e-9}
            focus = set(sorted(busy, key=lambda m: -busy[m])[:self.max_markets]) | held
        total_risk = self.risk()
        new = {}
        with self.lock:
            if self.killed:
                self.quotes = {}
                return
            for mid, m in self.markets.items():
                tte = m["expiry"] - now
                f, b = self.fairs.get(mid), self.books.get(mid)
                if not (self.stop_s < tte <= self.max_tte_s) or f is None or b is None or now - b["ts"] > self.book_stale_s:
                    continue
                if now < self.cool.get(mid, 0):                       # sitting out after an early exit
                    continue
                if focus is not None and mid not in focus:
                    continue
                fair = f[0]
                if not (self.min_p <= fair <= 1 - self.min_p):
                    continue
                qty = self.pos.get(str(mid), {}).get("qty", 0.0)
                skew = self.skew_max * qty / self.max_pos
                bid, ask = _floor(fair - self.edge - skew), _ceil(fair + self.edge - skew)
                if b["ask"] is not None:
                    bid = min(bid, round(b["ask"] - TICK, 6))          # a maker never crosses the market
                if b["bid"] is not None:
                    ask = max(ask, round(b["bid"] + TICK, 6))
                bid_sz = max(0.0, min(self.quote_size, self.max_pos - qty))
                ask_sz = max(0.0, min(self.quote_size, self.max_pos + qty))
                if total_risk >= self.max_risk:                       # at the risk cap: only quotes that reduce it
                    bid_sz = min(bid_sz, max(0.0, -qty))
                    ask_sz = min(ask_sz, max(0.0, qty))
                # adverse-move guard: fair ran >= guard_move against what we hold in the last guard_s -> stop adding
                # (2026-10-06: the maker bought a 29c -> 9c slide all the way to its 100-contract limit and lost $23)
                drift = self._fair_drift(mid, now)
                if drift is not None:
                    if qty > 0 and drift <= -self.guard_move:
                        bid_sz = 0.0
                    if qty < 0 and drift >= self.guard_move:
                        ask_sz = 0.0
                if bid <= 0:
                    bid_sz = 0.0
                if ask >= 1:
                    ask_sz = 0.0
                # live rules: whole contracts and the exchange's minimum order value (an ask is paid as 1 - ask when
                # it has to be placed as a No bid because we hold no Yes to sell)
                bid_sz = self._live_size(bid, bid_sz, self.max_pos - qty)
                ask_sz = self._live_size(ask if qty >= ask_sz > 0 else 1 - ask, ask_sz, self.max_pos + qty)
                old = self.quotes.get(mid, {})
                # sticky quotes: every live re-price spends the exchange's order budget (shadow mode measured ~200
                # re-prices a minute without this), so a resting quote keeps its price - and its place in the queue -
                # unless fair moved requote_threshold away, or it became unsafe (closer than half the edge to fair)
                if self.requote_thr:
                    bid = self._sticky(old.get("bid"), bid, fair, self.requote_thr, +1)
                    ask = self._sticky(old.get("ask"), ask, fair, self.requote_thr, -1)
                    if b["ask"] is not None:                          # still never crossing the market
                        bid = min(bid, round(b["ask"] - TICK, 6))
                    if b["bid"] is not None:
                        ask = max(ask, round(b["bid"] + TICK, 6))
                # an unchanged price and size keeps its place in time; a new price, or topping a partly filled quote
                # back up, is a new order (placed now, so only trades after it can fill it)
                keep_bid = old.get("bid") == round(bid, 6) and old.get("bid_sz", -1) >= bid_sz
                keep_ask = old.get("ask") == round(ask, 6) and old.get("ask_sz", -1) >= ask_sz
                q = {"bid": round(bid, 6), "ask": round(ask, 6), "fair": fair,
                     "bid_sz": bid_sz, "ask_sz": ask_sz,
                     "placed_bid": old["placed_bid"] if keep_bid else now,
                     "placed_ask": old["placed_ask"] if keep_ask else now,
                     "bid_best": b["bid"] is None or bid > b["bid"] + 1e-9,
                     "ask_best": b["ask"] is None or ask < b["ask"] - 1e-9}
                if self.levels >= 2:                                  # deeper second level, inside the same limits
                    e2 = self.edge * self.level2_mult
                    bid2, ask2 = min(_floor(fair - e2 - skew), round(bid - TICK, 6)), max(_ceil(fair + e2 - skew), round(ask + TICK, 6))
                    if self.requote_thr:
                        bid2 = min(self._sticky(old.get("bid2"), bid2, fair, self.requote_thr * 2, +1), round(bid - TICK, 6))
                        ask2 = max(self._sticky(old.get("ask2"), ask2, fair, self.requote_thr * 2, -1), round(ask + TICK, 6))
                    bid2_sz = max(0.0, min(self.quote_size, self.max_pos - qty - bid_sz)) if bid2 > 0 else 0.0
                    ask2_sz = max(0.0, min(self.quote_size, self.max_pos + qty - ask_sz)) if ask2 < 1 else 0.0
                    if total_risk >= self.max_risk:
                        bid2_sz = min(bid2_sz, max(0.0, -qty - bid_sz))
                        ask2_sz = min(ask2_sz, max(0.0, qty - ask_sz))
                    bid2_sz = self._live_size(bid2, bid2_sz, self.max_pos - qty - bid_sz)
                    ask2_sz = self._live_size(1 - ask2, ask2_sz, self.max_pos + qty - ask_sz)
                    keep_b2 = old.get("bid2") == round(bid2, 6) and old.get("bid2_sz", -1) >= bid2_sz
                    keep_a2 = old.get("ask2") == round(ask2, 6) and old.get("ask2_sz", -1) >= ask2_sz
                    q.update(bid2=round(bid2, 6), ask2=round(ask2, 6), bid2_sz=bid2_sz, ask2_sz=ask2_sz,
                             placed_bid2=old["placed_bid2"] if keep_b2 else now, placed_ask2=old["placed_ask2"] if keep_a2 else now)
                new[mid] = q
            self.quotes = new

    # ------------------------------------------------------------------ reporting
    def to_dict(self) -> dict:
        now = self.clock()
        with self.lock:
            eq = self.equity()
            mo = [f["markout"] for f in self.fills if f.get("markout") is not None]
            ed = [f["edge"] for f in self.fills if f.get("edge") is not None]
            vol = sum(f["qty"] for f in self.fills)
            rows = []
            for mid, m in sorted(self.markets.items(), key=lambda kv: (kv[1]["expiry"], kv[1]["coin"], kv[1]["strike"])):
                b, q, f = self.books.get(mid, {}), self.quotes.get(mid), self.fairs.get(mid)
                p = self.pos.get(str(mid))
                rows.append({"id": mid, "coin": m["coin"], "venue": m["venue"], "strike": m["strike"],
                             "expiry": m["expiry"], "tte_min": round((m["expiry"] - now) / 60, 1),
                             "fair": round(f[0], 4) if f else None,
                             "model": round(self.models[mid], 4) if mid in self.models else None, "mkt_bid": b.get("bid"), "mkt_ask": b.get("ask"),
                             "bid": q["bid"] if q and q["bid_sz"] > 0 else None,
                             "ask": q["ask"] if q and q["ask_sz"] > 0 else None,
                             "bid_sz": q["bid_sz"] if q else 0, "ask_sz": q["ask_sz"] if q else 0,
                             "qty": p["qty"] if p else 0.0,
                             "upnl": round(p["cash"] + p["qty"] * self._mark(str(mid), p), 2) if p else 0.0})
            open_pos = [{"id": int(k), **{x: p[x] for x in ("coin", "strike", "expiry", "venue", "qty", "fills")},
                         "upnl": round(p["cash"] + p["qty"] * self._mark(k, p), 2)} for k, p in self.pos.items()]
            settled_pnl = sum(s["pnl"] for s in self.settled)
            return {
                "enabled": self.enabled, "status": self.status, "error": self.error, "ws": dict(self.ws_state),
                "name": self.name, "strategy": self.strategy, "quoting": self.quoting,
                "snipe": {**self.snipe_stats, "pending": len(self._pending), "edge": self.snipe_edge,
                          "band": [self.band_lo, self.band_hi], "sides": self.snipe_sides,
                          "latency_ms": round(self.snipe_latency_s * 1000), "max_tte_min": self.max_tte_s / 60,
                          "trend": {"lookback_min": self.trend_lookback_s / 60, "threshold": self.trend_threshold,
                                    "neutral": self.trend_neutral,
                                    "coins": {c: v for c, v in self.trend_now.items() if now - v["ts"] < 600}}
                          if self.snipe_sides == "auto" else None, "regime": self.regime()}
                if self.strategy == "sniper" else None,
                "killed": self.killed, "kill_reason": self.kill_reason,
                "equity": round(eq, 2), "start_cash": self.start_cash, "pnl": round(eq - self.start_cash, 2),
                "cash": round(self.cash, 2), "peak": round(self.peak, 2), "risk": round(self.risk(), 2),
                "drawdown": round(1 - eq / self.peak, 4) if self.peak > 0 else 0.0,
                "config": {"edge": self.edge, "quote_size": self.quote_size, "max_position": self.max_pos,
                           "max_risk_usd": self.max_risk, "stop_min": self.stop_s / 60, "max_tte_min": self.max_tte_s / 60,
                           "coins": self.coins or "all", "fee": self.fee, "kill_drawdown": self.kill_dd,
                           "levels": self.levels, "level2_edge": round(self.edge * self.level2_mult, 4),
                           "exits": {"enabled": self.exits_on, "take_profit": self.take_profit, "stop_loss": self.stop_loss,
                                     "taker_fee": self.taker_fee, "cooldown_s": self.exit_cooldown_s},
                           "markets_quoted": len(self.quotes), "markets_tracked": len(self.markets)},
                "size": self.size_info(),
                "stats": {"fills": len(self.fills), "contracts": round(vol, 1),
                          "avg_edge": round(sum(ed) / len(ed), 4) if ed else None,
                          "avg_markout": round(sum(mo) / len(mo), 4) if mo else None, "markouts": len(mo),
                          "settled": len(self.settled), "settled_pnl": round(settled_pnl, 2),
                          "settled_wins": sum(1 for s in self.settled if s["pnl"] > 0)},
                "markets": rows, "open": open_pos,
                "fills": [{k: f.get(k) for k in ("ts", "id", "coin", "strike", "expiry", "side", "px", "qty", "fair",
                                                 "edge", "markout", "how", "inv_after")} for f in self.fills[-25:][::-1]],
                "settled": self.settled[-15:][::-1],
            }
