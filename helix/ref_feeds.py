"""Outside reference prices for the shadow lab (2026-10-12), RECORD + TEST ONLY - nothing here sends an order.

KalshiFeed: Kalshi's public 15-minute "BTC / ETH up or down" markets (KXBTC15M / KXETH15M) ask nearly the same question
as HIP-4's 15-minute BTC/ETH markets, with ~100x the trading. On a week of tape, Kalshi's price at the same instant
forecast HIP-4's outcomes better than HIP-4's own book, and taking HIP-4 quotes >= 5c through Kalshi's made ~+13c per
contract on paper (n=108, unproven: quotes vanish in ~2 s, half are < $10 deep). Public REST, no account, no key;
polled every POLL_S. Every change is recorded to data/kalshi/<UTC day>.jsonl for later study.

CexFeed: Coinbase's public BTC-USD / ETH-USD ticker WebSocket (no account; Coinbase is fine for US users). Outside
venues lead Hyperliquid by ~1 s, and live fills used to come right after the price had moved against the bet; the lab
tests a "don't bet right after a jump against us" veto with it. Recorded at most once a second per coin to
data/cex/<UTC day>.jsonl.

Both: a failure only marks the feed stale (the lab then skips what needs it); never raises into the caller."""
from __future__ import annotations

import collections
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M"}
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
COINBASE_PRODUCTS = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
KEEP_DAYS = 30


def _parse_utc(s: str) -> int | None:
    try:
        import calendar
        return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class _Recorder:
    """Appends compact JSON lines to <dir>/<UTC day>.jsonl and deletes days older than KEEP_DAYS."""

    def __init__(self, folder: Path | None):
        self.dir = Path(folder) if folder is not None else None
        self._day = None

    def write(self, rec: dict, now: float) -> None:
        if self.dir is None:
            return
        day = time.strftime("%Y%m%d", time.gmtime(now))
        try:
            if day != self._day:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._day = day
                cutoff = time.strftime("%Y%m%d", time.gmtime(now - KEEP_DAYS * 86400))
                for p in self.dir.glob("*.jsonl"):
                    if p.stem < cutoff:
                        p.unlink(missing_ok=True)
            with open(self.dir / f"{day}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError:
            pass


class KalshiFeed:
    """The market LIST is served from a CDN cache (max-age 15 s), so it is only used to find each coin's open market
    and its strike. The market's own page (/markets/<ticker>) is a ~10 s server-side snapshot (measured from Tokyo
    2026-10-13: its yes bid/ask changed every 9-12 s while the order book changed every 1-2 s), so prices come from the
    ORDER BOOK (/markets/<ticker>/orderbook): Yes bid = best Yes bid, Yes ask = 1 - best No bid. Each coin polls on its
    own thread; a quote is stamped with the time its request was SENT minus any cache Age (never fresher than it is),
    together with Hyperliquid's price for the coin at that moment (spot_fn), so the lab can adjust for moves since."""
    POLL_S, MAX_AGE_S, BACKOFF_S, DISCOVER_S, TIMEOUT_S = 1.0, 2.0, 15.0, 20.0, 4.0

    def __init__(self, data_dir: Path | None, fetch=None, clock=time.time, spot_fn=None):
        self.clock = clock
        self.fetch = fetch or self._fetch                  # url -> (json body, cache age in seconds)
        self.spot_fn = spot_fn
        self.rec = _Recorder(Path(data_dir) / "kalshi" if data_dir is not None else None)
        self.lock = threading.Lock()
        self.quotes: dict[tuple, dict] = {}                # (coin, expiry) -> bid, ask, sizes, strike, ts, spot
        self.open: dict[str, list] = {}                    # coin -> [(ticker, expiry)] from the market list
        self._disc_at: dict[str, float] = {}
        self.ok_at: dict[str, float] = {}
        self.error, self.polls, self.fails = None, 0, 0

    @classmethod
    def _fetch(cls, url: str) -> tuple[dict, float]:
        req = urllib.request.Request(url, headers={"User-Agent": "helix-research/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=cls.TIMEOUT_S) as r:
            age = _f(r.headers.get("Age")) or 0.0
            return json.loads(r.read()), age

    def _spot(self, coin: str) -> float | None:
        try:
            return _f(self.spot_fn(coin)) if self.spot_fn is not None else None
        except Exception:                                  # noqa: BLE001
            return None

    def _discover(self, coin: str, now: float) -> None:
        d, _age = self.fetch(f"{KALSHI_API}/markets?series_ticker={KALSHI_SERIES[coin]}&status=open&limit=10")
        found = []
        for m in d.get("markets") or []:
            exp = _parse_utc(m.get("close_time") or "")
            if exp is not None and m.get("ticker"):
                found.append((m["ticker"], exp, _f(m.get("floor_strike"))))
        self.open[coin] = found
        self._disc_at[coin] = now

    @staticmethod
    def _top(levels) -> tuple[float | None, float | None]:
        """Best (highest) price and its size from a Kalshi book side [[price, size], ...]."""
        best = None
        for lv in levels or []:
            px, sz = (_f(lv[0]), _f(lv[1])) if isinstance(lv, (list, tuple)) and len(lv) >= 2 else (None, None)
            if px is not None and sz and (best is None or px > best[0]):
                best = (px, sz)
        return best or (None, None)

    def poll_coin(self, coin: str) -> None:
        now = self.clock()
        if now - self._disc_at.get(coin, 0.0) >= self.DISCOVER_S or not any(e > now for _t, e, _k in self.open.get(coin, [])):
            self._discover(coin, now)
        for ticker, exp, strike in self.open.get(coin, []):
            if exp <= self.clock():
                continue
            sent, spot = self.clock(), self._spot(coin)
            d, age = self.fetch(f"{KALSHI_API}/markets/{ticker}/orderbook")
            if age > self.MAX_AGE_S:
                raise RuntimeError(f"Kalshi answered from a {age:.0f} s old cache")
            ob = d.get("orderbook_fp") or d.get("orderbook") or {}
            bid, bid_sz = self._top(ob.get("yes_dollars"))
            no_bid, no_sz = self._top(ob.get("no_dollars"))
            if bid is None or no_bid is None:
                continue
            ask = round(1.0 - no_bid, 4)
            q = {"bid": bid, "ask": ask, "bid_sz": bid_sz, "ask_sz": no_sz, "strike": strike, "ts": sent - age,
                 "spot": spot}
            with self.lock:
                old = self.quotes.get((coin, exp))
                self.quotes[(coin, exp)] = q
                self.quotes = {k: v for k, v in self.quotes.items() if k[1] > sent - 600}
            if old is None or (old["bid"], old["ask"]) != (bid, ask):
                self.rec.write({"t": round(q["ts"], 3), "c": coin, "e": exp, "b": bid, "a": ask, "bs": q["bid_sz"],
                                "as": q["ask_sz"], "k": q["strike"], "s": spot}, sent)
        self.ok_at[coin] = self.clock()

    def quote(self, coin: str, expiry: float, now: float) -> dict | None:
        """The live Kalshi quote for this coin and 15-minute expiry, or None when missing or older than MAX_AGE_S."""
        with self.lock:
            q = self.quotes.get((coin, int(expiry)))
        return q if q is not None and now - q["ts"] <= self.MAX_AGE_S else None

    def status(self) -> dict:
        now = self.clock()
        ok = all(now - self.ok_at.get(c, 0.0) <= 30 for c in KALSHI_SERIES)
        last = min((self.ok_at.get(c, 0.0) for c in KALSHI_SERIES), default=0.0)
        return {"ok": ok, "last_ok_s": round(now - last, 1) if last else None, "polls": self.polls, "fails": self.fails,
                "error": self.error, "markets": len(self.quotes)}

    def _loop(self, coin: str, stop: threading.Event) -> None:
        while not stop.is_set():
            self.polls += 1
            try:
                self.poll_coin(coin)
                wait = self.POLL_S
            except Exception as exc:                               # noqa: BLE001 - stale, never fatal
                self.fails += 1
                self.error = f"{coin}: {type(exc).__name__}: {exc}"[:200]
                wait = self.BACKOFF_S
            stop.wait(wait)

    def run(self, stop: threading.Event) -> None:
        """One polling thread per coin, so a slow or hung request for one never delays or re-stamps the other."""
        for coin in KALSHI_SERIES:
            threading.Thread(target=self._loop, args=(coin, stop), name=f"kalshi-{coin}", daemon=True).start()
        stop.wait()


class CexFeed:
    """Coinbase ticker mids, kept for the last HISTORY_S seconds per coin."""
    HISTORY_S, STALE_S, RECORD_EVERY_S, STALL_S = 15.0, 3.0, 1.0, 30.0

    def __init__(self, data_dir: Path | None, clock=time.time, connect=None):
        self.clock = clock
        self.connect = connect
        self.rec = _Recorder(Path(data_dir) / "cex" if data_dir is not None else None)
        self.lock = threading.Lock()
        self.hist: dict[str, collections.deque] = {c: collections.deque() for c in COINBASE_PRODUCTS.values()}
        self._rec_at: dict[str, float] = {}
        self.ok_at, self.error, self.msgs, self.reconnects = 0.0, None, 0, 0

    def on_message(self, raw: str) -> None:
        try:
            d = json.loads(raw)
        except ValueError:
            return
        if d.get("type") == "error":
            self.error = f"Coinbase: {d.get('message')} {d.get('reason') or ''}".strip()[:200]
            return
        if d.get("type") != "ticker":
            return
        coin = COINBASE_PRODUCTS.get(d.get("product_id"))
        bid, ask = _f(d.get("best_bid")), _f(d.get("best_ask"))
        if coin is None or not bid or not ask or ask < bid:
            return
        now, mid = self.clock(), (bid + ask) / 2
        with self.lock:
            h = self.hist[coin]
            h.append((now, mid))
            while h and h[0][0] < now - self.HISTORY_S:
                h.popleft()
        self.msgs += 1
        self.ok_at = now
        if now - self._rec_at.get(coin, 0.0) >= self.RECORD_EVERY_S:
            self._rec_at[coin] = now
            self.rec.write({"t": round(now, 3), "c": coin, "b": bid, "a": ask}, now)

    def move(self, coin: str, now: float, window_s: float) -> float | None:
        """log(latest mid / the mid `window_s` ago); None without fresh data on both ends."""
        with self.lock:
            h = list(self.hist.get(coin) or ())
        if not h or now - h[-1][0] > self.STALE_S:
            return None
        then = [m for t, m in h if t <= now - window_s]
        if not then:
            return None
        return math.log(h[-1][1] / then[-1])

    def status(self) -> dict:
        now = self.clock()
        return {"ok": bool(self.ok_at) and now - self.ok_at <= 10, "last_msg_s": round(now - self.ok_at, 1) if self.ok_at else None,
                "msgs": self.msgs, "reconnects": self.reconnects, "error": self.error}

    def _open(self):
        if self.connect is not None:
            return self.connect()
        from .wsclient import WebSocket
        ws = WebSocket(COINBASE_WS, timeout=10)
        ws.send(json.dumps({"type": "subscribe", "product_ids": list(COINBASE_PRODUCTS), "channels": ["ticker"]}))
        return ws

    def run(self, stop: threading.Event) -> None:
        backoff = 2.0
        while not stop.is_set():
            ws = None
            try:
                ws = self._open()
                ws.settimeout(5.0)
                opened, msgs0 = self.clock(), self.msgs
                while not stop.is_set():
                    try:
                        self.on_message(ws.recv())
                    except TimeoutError:
                        pass
                    if self.msgs > msgs0:
                        backoff = 2.0                      # reset only once this connection delivered prices
                    if self.clock() - max(self.ok_at, opened) > self.STALL_S:
                        raise ConnectionError(f"no Coinbase ticker for {self.STALL_S:g} s")   # half-open: reconnect
            except Exception as exc:                               # noqa: BLE001 - stale, then reconnect
                self.error = f"{type(exc).__name__}: {exc}"[:200]
                self.reconnects += 1
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:                              # noqa: BLE001
                        pass
            stop.wait(backoff)
            backoff = min(60.0, backoff * 2)
