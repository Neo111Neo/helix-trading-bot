"""Hyperliquid market data, read-only and keyless.

WebSocket (wss://api.hyperliquid.xyz/ws) streams l2Book, bbo, trades, activeAssetCtx and 1-minute candles for each
coin. A REST thread (POST https://api.hyperliquid.xyz/info) backfills candle history, refreshes asset contexts and
takes over the order book by polling whenever a stream goes silent ("WebSocket first, polling as backup"). Every REST
request in the process goes through one weight budget (Hyperliquid allows ~1,200 per minute per IP). Nothing here can
place an order."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections import deque

import requests

from .market import Candle, DataError, PerpView, Trade
from .wsclient import WebSocket, WebSocketError

URLS = {
    "mainnet": ("https://api.hyperliquid.xyz/info", "wss://api.hyperliquid.xyz/ws"),
    "testnet": ("https://api.hyperliquid-testnet.xyz/info", "wss://api.hyperliquid-testnet.xyz/ws"),
}
MID_SPACING_S = 0.25
MID_KEEP_S = 900
TRADE_KEEP_S = 300


# ---- parsing (pure functions, unit-tested against documented shapes) ---------------------
def parse_levels(levels) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """levels = [bids, asks], each a list of {"px": str, "sz": str, "n": int}, best first."""
    bids = [(float(lv["px"]), float(lv["sz"])) for lv in (levels[0] if levels else [])]
    asks = [(float(lv["px"]), float(lv["sz"])) for lv in (levels[1] if len(levels) > 1 else [])]
    return bids, asks


def parse_trade(t: dict) -> Trade:
    # side "B" = the aggressor bought (lifted the ask), "A" = the aggressor sold (hit the bid)
    return Trade(float(t["time"]) / 1000.0, float(t["px"]), float(t["sz"]), "b" if t.get("side") == "B" else "s")


def parse_ctx(ctx: dict) -> dict:
    def f(key):
        try:
            return float(ctx[key])
        except (KeyError, TypeError, ValueError):
            return None
    return {
        "mark": f("markPx"), "oracle": f("oraclePx"), "mid": f("midPx"),
        "funding": f("funding"),  # applied hourly: payment = size * oracle * funding
        "open_interest": f("openInterest"), "day_volume": f("dayNtlVlm"), "prev_day_px": f("prevDayPx"),
        "premium": f("premium"),
    }


def parse_candle(c: dict) -> Candle:
    return Candle(int(c["t"]) // 1000, float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"]))


_INTERVAL_S = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def request_weight(body: dict) -> float:
    """Hyperliquid's documented info-request weights: 2 for l2Book/allMids/clearinghouseState/orderStatus, 20 for the
    rest, plus 1 per 60 candles returned by candleSnapshot and 1 per 20 records for fundingHistory."""
    kind = body.get("type")
    if kind in ("l2Book", "allMids", "clearinghouseState", "orderStatus", "spotClearinghouseState", "exchangeStatus"):
        return 2.0
    if kind == "candleSnapshot":
        req = body.get("req") or {}
        span = max(0.0, (req.get("endTime", 0) - req.get("startTime", 0)) / 1000.0)
        return 20.0 + span / _INTERVAL_S.get(req.get("interval"), 60) / 60.0
    if kind == "fundingHistory":
        span = max(0.0, ((body.get("endTime") or time.time() * 1000) - body.get("startTime", 0)) / 1000.0)
        return 20.0 + span / 3600 / 20.0
    return 20.0


class WeightBudget:
    """One token bucket for every Hyperliquid REST request this process makes (feed, Jev signal, hourly test, HIP-4
    recorder, weekly lab). Hyperliquid allows ~1,200 weight per minute per IP; this keeps the whole process under
    `per_min` by making callers wait, and pauses everyone for a while after a 429."""

    def __init__(self, per_min: float):
        self.per_min = per_min
        self.tokens = per_min
        self.updated = time.monotonic()
        self.blocked_until = 0.0
        self.lock = threading.Lock()
        self.waited_s = 0.0

    def acquire(self, weight: float) -> None:
        weight = min(weight, self.per_min)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.per_min, self.tokens + (now - self.updated) * self.per_min / 60.0)
                self.updated = now
                if now >= self.blocked_until and self.tokens >= weight:
                    self.tokens -= weight
                    return
                wait = max(self.blocked_until - now, (weight - self.tokens) * 60.0 / self.per_min)
            wait = min(2.0, max(0.05, wait))
            self.waited_s += wait
            time.sleep(wait)

    def penalize(self, seconds: float = 10.0) -> None:
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)
            self.tokens = 0.0


BUDGET = WeightBudget(float(os.getenv("HELIX_REST_WEIGHT_PER_MIN", "900")))


class HLRest:
    def __init__(self, url: str, session: requests.Session | None = None, timeout: float = 5.0,
                 budget: WeightBudget | None = None):
        self.url = url
        self.session = session or requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "helix-paper-engine/2.0"})
        self.timeout = timeout
        self.budget = budget or BUDGET

    def post(self, body: dict):
        self.budget.acquire(request_weight(body))
        try:
            r = self.session.post(self.url, data=json.dumps(body), timeout=self.timeout)
            if r.status_code == 429:
                self.budget.penalize(10.0)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"Hyperliquid {body.get('type')} failed: {exc}") from exc

    def l2_book(self, coin: str):
        data = self.post({"type": "l2Book", "coin": coin})
        bids, asks = parse_levels(data.get("levels") or [])
        return bids, asks, float(data.get("time", time.time() * 1000)) / 1000.0

    def candles(self, coin: str, interval: str = "1m", minutes: int = 180, now: float | None = None) -> list[Candle]:
        now = now or time.time()
        data = self.post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                           "startTime": int((now - minutes * 60) * 1000),
                                                           "endTime": int(now * 1000)}})
        out = [parse_candle(c) for c in data or [] if int(c.get("T", 0)) / 1000.0 <= now]  # closed candles only
        return sorted(out, key=lambda c: c.ts)

    def asset_ctxs(self) -> dict[str, dict]:
        data = self.post({"type": "metaAndAssetCtxs"})
        meta, ctxs = data[0], data[1]
        return {u["name"]: parse_ctx(c) for u, c in zip(meta.get("universe", []), ctxs)}


class _Buffer:
    """Latest state for one coin. Written by feed threads, read by the engine."""

    def __init__(self, coin: str, clock=time.time):
        self.coin = coin
        self.clock = clock
        self.needs_backfill = True               # set again on every WebSocket (re)connect
        self.lock = threading.Lock()
        self.bids: list = []
        self.asks: list = []
        self.book_ts = 0.0
        self.received = 0.0
        self.source = "ws"
        self.trades: deque = deque(maxlen=20000)
        self.mids: deque = deque(maxlen=int(MID_KEEP_S / MID_SPACING_S) + 10)
        self.candles: list[Candle] = []
        self.forming: Candle | None = None      # the 1-minute candle still trading (WebSocket "candle" channel)
        self.forming_end = 0.0
        self.candle_msg = 0.0
        self.ctx: dict = {}

    def _mid(self, now: float) -> None:
        mid = (self.bids[0][0] + self.asks[0][0]) / 2
        if not self.mids or now - self.mids[-1][0] >= MID_SPACING_S:
            self.mids.append((now, mid))
        while self.mids and now - self.mids[0][0] > MID_KEEP_S:
            self.mids.popleft()

    def on_book(self, bids, asks, book_ts: float, source: str, now: float) -> None:
        if not bids or not asks:
            return
        with self.lock:
            self.bids, self.asks, self.book_ts, self.received, self.source = bids, asks, book_ts, now, source
            self._mid(now)

    def on_bbo(self, bid: tuple | None, ask: tuple | None, book_ts: float, now: float) -> None:
        """Best bid/offer pushed on every change: keeps the top of the book fresh between full l2Book snapshots
        (Hyperliquid pushes those only every ~5 s). Deeper levels that the new top crossed are dropped. Ignored
        until a first full snapshot exists (a lone top level would read as an extremely thin book)."""
        with self.lock:
            if not self.bids or not self.asks:
                return
            if bid is not None and bid[0] < (ask or self.asks[0])[0]:
                self.bids = [bid] + [lv for lv in self.bids if lv[0] < bid[0]]
            if ask is not None and ask[0] > self.bids[0][0]:
                self.asks = [ask] + [lv for lv in self.asks if lv[0] > ask[0]]
            self.book_ts, self.received, self.source = max(self.book_ts, book_ts), now, "ws"
            self._mid(now)

    def on_candle(self, c: Candle, end_ts: float, now: float) -> None:
        """1-minute candle pushes. Hyperliquid pushes only while a minute trades, so a minute with no trades never
        arrives; REST history (and the lab) record such minutes as flat candles at the previous close with zero
        volume, so the same flat candles are filled in here. After a reconnect `forming` is None and nothing is
        filled - the REST backfill repairs that stretch."""
        with self.lock:
            self.candle_msg = now
            f = self.forming
            if self.candles and c.ts == self.candles[-1].ts:      # a late update for a minute already closed
                self.candles[-1] = c
                return
            if self.candles and c.ts < self.candles[-1].ts:
                return
            if f is not None and c.ts > f.ts:              # a new minute started: the previous candle has closed
                if not self.candles or f.ts > self.candles[-1].ts:
                    if not self.needs_backfill:            # quiet minutes before f (known only while connected)
                        self._flat_until(self.candles, f.ts)
                    self.candles.append(f)
                    self._flat_until(self.candles, c.ts)   # quiet minutes between f and this one
                    del self.candles[:-300]
            if f is None or c.ts >= f.ts:
                self.forming, self.forming_end = c, end_ts

    @staticmethod
    def _flat_until(out: list, ts: int) -> None:
        """Append flat zero-volume candles after out[-1] for every minute before `ts` (minutes with no trades)."""
        if out:
            px = out[-1].close
            for t in range(out[-1].ts + 60, ts, 60):
                out.append(Candle(t, px, px, px, px, 0.0))

    def on_reconnect(self) -> None:
        with self.lock:
            self.forming, self.forming_end, self.needs_backfill = None, 0.0, True

    def closed_candles(self, now: float) -> list[Candle]:
        """Closed 1-minute candles up to the last finished minute: the latest pushed candle once its minute has ended,
        then flat zero-volume candles for finished minutes with no trades - but only while the stream is alive (a
        silent stream means an outage, and guessing flat candles there would be wrong)."""
        with self.lock:
            out = list(self.candles)
            f = self.forming
            trusted = not self.needs_backfill and now - self.candle_msg < 90
            if f is not None and now >= self.forming_end and (not out or f.ts > out[-1].ts):
                if trusted:
                    self._flat_until(out, f.ts)
                out.append(f)
            if trusted and f is not None:
                self._flat_until(out, int(now // 60) * 60)          # finished minutes with no trades yet pushed
            return out

    def on_trades(self, trades: list[Trade], now: float) -> None:
        with self.lock:
            self.trades.extend(trades)
            newest = max(t.ts for t in self.trades) if self.trades else now
            while self.trades and newest - self.trades[0].ts > TRADE_KEEP_S:  # exchange clock, not ours
                self.trades.popleft()

    def view(self, light: bool = False) -> PerpView:
        with self.lock:
            if not self.bids or not self.asks:
                raise DataError(f"{self.coin}: no order book yet")
            if light:
                return PerpView(self.coin, list(self.bids), list(self.asks), self.book_ts, self.received,
                                ctx=dict(self.ctx), source=self.source)
            trades, mids, ctx = list(self.trades), list(self.mids), dict(self.ctx)
            bids, asks, book_ts, received, source = list(self.bids), list(self.asks), self.book_ts, self.received, self.source
        candles = self.closed_candles(self.clock())
        return PerpView(self.coin, bids, asks, book_ts, received, trades, mids, candles[-180:], ctx, source)


class HyperliquidFeed:
    name = "hyperliquid"

    def __init__(self, coins: list[str], network: str = "mainnet", rest: HLRest | None = None,
                 ws_factory=WebSocket, stale_after_s: float = 3.0, clock=time.time, ws_url: str | None = None):
        if network not in URLS:
            raise ValueError("network must be mainnet or testnet")
        self.coins = list(coins)
        self.network = network
        rest_url, default_ws = URLS[network]
        self.ws_url = ws_url or default_ws
        self.rest = rest or HLRest(rest_url)
        self.ws_factory = ws_factory
        self.stale_after_s = stale_after_s
        self.clock = clock
        self.buffers = {c: _Buffer(c, clock) for c in self.coins}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ws: WebSocket | None = None
        self.stats = {"connected": False, "reconnects": 0, "messages": 0, "last_msg": 0.0, "ws_error": None,
                      "rest_error": None, "rest_polls": 0, "msg_times": deque(maxlen=3000)}

    # ---- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        for target, name in ((self._ws_loop, "hl-ws"), (self._rest_loop, "hl-rest")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            self._ws.close()

    # ---- engine interface ------------------------------------------------------------
    def view(self, coin: str) -> PerpView:
        return self.buffers[coin].view()

    def quote(self, coin: str) -> PerpView:
        """Book and context only (no trade/mid history): cheap enough for every monitor step."""
        return self.buffers[coin].view(light=True)

    def health(self) -> dict:
        now = self.clock()
        # tuple() copies in one C call; iterating the live deque raced the WebSocket thread's appends
        # ("deque mutated during iteration" failed two engine ticks)
        times = [t for t in tuple(self.stats["msg_times"]) if now - t <= 10]
        return {
            "venue": "hyperliquid", "network": self.network, "connected": self.stats["connected"],
            "msgs_per_s": round(len(times) / 10, 1), "reconnects": self.stats["reconnects"],
            "last_msg_age_s": round(now - self.stats["last_msg"], 2) if self.stats["last_msg"] else None,
            "ws_error": self.stats["ws_error"], "rest_error": self.stats["rest_error"],
            "rest_polls": self.stats["rest_polls"],
            "sources": {c: b.source for c, b in self.buffers.items()},
        }

    # ---- WebSocket -------------------------------------------------------------------
    def handle(self, msg: dict) -> None:
        """Route one pushed message into the buffers. Public so tests can feed fixtures."""
        now = self.clock()
        channel = msg.get("channel")
        data = msg.get("data")
        if channel == "l2Book" and isinstance(data, dict) and data.get("coin") in self.buffers:
            bids, asks = parse_levels(data.get("levels") or [])
            self.buffers[data["coin"]].on_book(bids, asks, float(data.get("time", now * 1000)) / 1000.0, "ws", now)
        elif channel == "bbo" and isinstance(data, dict) and data.get("coin") in self.buffers:
            b, a = ((data.get("bbo") or []) + [None, None])[:2]
            if b or a:
                self.buffers[data["coin"]].on_bbo((float(b["px"]), float(b["sz"])) if b else None,
                                                  (float(a["px"]), float(a["sz"])) if a else None,
                                                  float(data.get("time", now * 1000)) / 1000.0, now)
        elif channel == "candle" and isinstance(data, dict) and data.get("s") in self.buffers and data.get("i") == "1m":
            self.buffers[data["s"]].on_candle(parse_candle(data), (int(data["T"]) + 1) / 1000.0, now)
        elif channel == "trades" and isinstance(data, list):
            by_coin: dict[str, list[Trade]] = {}
            for t in data:
                if t.get("coin") in self.buffers:
                    by_coin.setdefault(t["coin"], []).append(parse_trade(t))
            for coin, trades in by_coin.items():
                self.buffers[coin].on_trades(trades, now)
        elif channel == "activeAssetCtx" and isinstance(data, dict):
            coin = data.get("coin")
            if coin in self.buffers:
                ctx = parse_ctx(data.get("ctx") or {})
                with self.buffers[coin].lock:
                    self.buffers[coin].ctx.update({k: v for k, v in ctx.items() if v is not None})
        elif channel == "error":
            self.stats["ws_error"] = str(data)[:200]

    def _ws_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = self.ws_factory(self.ws_url, timeout=10)
                self._ws = ws
                for buf in self.buffers.values():     # candles missed while disconnected: refill them over REST
                    buf.on_reconnect()
                for coin in self.coins:
                    for sub in ({"type": "l2Book", "coin": coin}, {"type": "trades", "coin": coin},
                                {"type": "activeAssetCtx", "coin": coin}, {"type": "bbo", "coin": coin},
                                {"type": "candle", "coin": coin, "interval": "1m"}):
                        ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                ws.settimeout(1.0)
                self.stats["connected"], self.stats["ws_error"] = True, None
                backoff = 1.0
                last_ping = time.monotonic()
                while not self._stop.is_set():
                    if time.monotonic() - last_ping > 30:
                        ws.send(json.dumps({"method": "ping"}))
                        last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except socket.timeout:
                        continue
                    now = self.clock()
                    self.stats["messages"] += 1
                    self.stats["last_msg"] = now
                    self.stats["msg_times"].append(now)
                    try:
                        self.handle(json.loads(raw))
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        self.stats["ws_error"] = f"Bad message: {exc}"
            except (OSError, WebSocketError, ValueError) as exc:
                self.stats["ws_error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                if self.stats["connected"]:
                    self.stats["reconnects"] += 1
                self.stats["connected"] = False
                if self._ws:
                    self._ws.close()
                    self._ws = None
            self._stop.wait(backoff)
            backoff = min(30.0, backoff * 2)

    # ---- REST (context, candles, and book backup) ------------------------------------
    def _rest_loop(self) -> None:
        """REST is now only the backup: the WebSocket carries books (l2Book every ~5 s + bbo on every change),
        trades, context and 1-minute candles. REST backfills candle history at start (and when the candle stream
        goes quiet), refreshes context every 60 s, and polls a book only when its stream is silent. That keeps the
        request weight low enough for 20+ coins (Hyperliquid allows ~1,200 per minute per IP)."""
        next_ctx = 0.0
        tried: dict[str, float] = {}
        while not self._stop.is_set():
            now = self.clock()
            errors = []
            if now >= next_ctx:                         # each section on its own: one failure can't starve the others,
                next_ctx = now + 60                     # and a failed call is not retried every second
                try:
                    ctxs = self.rest.asset_ctxs()
                    for coin, buf in self.buffers.items():
                        if coin in ctxs:
                            with buf.lock:
                                buf.ctx.update({k: v for k, v in ctxs[coin].items() if v is not None})
                except Exception as exc:
                    errors.append(f"ctx: {exc}")
            try:                                        # history at start / after a reconnect / if the stream stops
                for coin, buf in self.buffers.items():
                    quiet = now - buf.candle_msg > 120
                    if (buf.needs_backfill or quiet) and now - tried.get(coin, 0.0) > 120:
                        tried[coin] = now
                        candles = self.rest.candles(coin, "1m", 250, now)
                        if candles:                     # an empty answer must not wipe the stream's history
                            with buf.lock:
                                newer = [c for c in buf.candles if c.ts > candles[-1].ts]
                                # REST leaves out trailing minutes with no trades; the stream was live by then, so
                                # anything missing between the two had no trades: flat candles, as REST records them
                                if newer:
                                    last = candles[-1]
                                    candles = candles + [Candle(t, last.close, last.close, last.close, last.close, 0.0)
                                                         for t in range(last.ts + 60, newer[0].ts, 60)]
                                buf.candles = (candles + newer)[-300:]
                                buf.needs_backfill = False
                        self.stats["rest_polls"] += 1
                        break                           # one per pass: spread the burst out
            except Exception as exc:
                errors.append(f"candles: {exc}")
            for coin, buf in self.buffers.items():
                if now - buf.received > self.stale_after_s and now - tried.get("book:" + coin, 0.0) > self.stale_after_s:
                    tried["book:" + coin] = now         # stream silent: poll instead (at most once per stale period)
                    try:
                        bids, asks, ts = self.rest.l2_book(coin)
                        buf.on_book(bids, asks, ts, "rest", self.clock())
                        self.stats["rest_polls"] += 1
                    except Exception as exc:
                        errors.append(f"book {coin}: {exc}")
            self.stats["rest_error"] = "; ".join(errors)[:200] or None
            self._stop.wait(1.0)
