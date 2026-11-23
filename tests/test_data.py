"""Data-layer tests: the stdlib WebSocket client against a local server, Hyperliquid message
parsing against the documented shapes, the live feed end to end over a local socket, and the
simulator."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.hyperliquid import HyperliquidFeed, parse_candle, parse_ctx, parse_levels, parse_trade  # noqa: E402
from helix.market import DataError, walk_book  # noqa: E402
from helix.simfeed import SimPerpFeed  # noqa: E402
from helix.wsclient import WebSocket, WebSocketClosed  # noqa: E402

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def server_frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    return head + payload


class MiniServer:
    """A tiny WebSocket server: completes the handshake, runs `script(conn, server)`, records
    the text frames the client sends."""

    def __init__(self, script):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.script = script
        self.received: list[str] = []
        self.pongs: list[bytes] = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def _serve(self):
        try:
            while True:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        except OSError:
            pass

    def _handle(self, conn):
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += conn.recv(4096)
        key = [line.split(":", 1)[1].strip() for line in buf.decode().split("\r\n") if line.lower().startswith("sec-websocket-key")][0]
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        conn.settimeout(5)
        try:
            self.script(conn, self)
        except OSError:
            pass
        finally:
            conn.close()

    def read_client_frame(self, conn):
        def exact(n):
            out = b""
            while len(out) < n:
                chunk = conn.recv(n - len(out))
                if not chunk:
                    raise OSError("closed")
                out += chunk
            return out
        b1, b2 = exact(2)
        assert b2 & 0x80, "client frames must be masked"
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack("!H", exact(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", exact(8))[0]
        key = exact(4)
        data = bytes(b ^ key[i % 4] for i, b in enumerate(exact(n)))
        return b1 & 0x0F, data

    def close(self):
        self.sock.close()


class WebSocketClientTests(unittest.TestCase):
    def test_round_trip_ping_fragments_and_close(self):
        big = json.dumps({"x": "y" * 70000})

        def script(conn, srv):
            op, data = srv.read_client_frame(conn)
            srv.received.append(data.decode())
            conn.sendall(server_frame(0x1, b'{"channel":"hello"}'))
            conn.sendall(server_frame(0x9, b"pp"))  # ping: the client must answer with a pong
            op, data = srv.read_client_frame(conn)
            if op == 0xA:
                srv.pongs.append(data)
            conn.sendall(server_frame(0x1, b'{"part":', fin=False) + server_frame(0x0, b'"two"}'))
            conn.sendall(server_frame(0x1, big.encode()))
            conn.sendall(server_frame(0x8, struct.pack("!H", 1000)))
            time.sleep(0.2)

        srv = MiniServer(script)
        try:
            ws = WebSocket(srv.url, timeout=5)
            ws.send(json.dumps({"method": "ping"}))
            self.assertEqual(json.loads(ws.recv()), {"channel": "hello"})
            self.assertEqual(json.loads(ws.recv()), {"part": "two"})
            self.assertEqual(len(ws.recv()), len(big))
            with self.assertRaises(WebSocketClosed):
                ws.recv()
            self.assertEqual(srv.received, ['{"method": "ping"}'])
            self.assertEqual(srv.pongs, [b"pp"])
        finally:
            srv.close()

    def test_timeout_keeps_partial_frame(self):
        def script(conn, srv):
            frame = server_frame(0x1, b'{"ok":true}')
            conn.sendall(frame[:3])
            time.sleep(0.4)
            conn.sendall(frame[3:])
            time.sleep(0.3)

        srv = MiniServer(script)
        try:
            ws = WebSocket(srv.url, timeout=5)
            ws.settimeout(0.1)
            got = None
            for _ in range(30):
                try:
                    got = ws.recv()
                    break
                except socket.timeout:
                    continue
            self.assertEqual(json.loads(got), {"ok": True})
            ws.close()
        finally:
            srv.close()

    def test_bad_handshake(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]

        def serve():
            conn, _ = s.accept()
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            conn.close()
        threading.Thread(target=serve, daemon=True).start()
        from helix.wsclient import WebSocketError
        with self.assertRaises(WebSocketError):
            WebSocket(f"ws://127.0.0.1:{port}/ws", timeout=5)
        s.close()


BOOK_MSG = {"channel": "l2Book", "data": {"coin": "BTC", "time": 1754450974231, "levels": [
    [{"px": "113377.0", "sz": "7.6699", "n": 17}, {"px": "113376.0", "sz": "1.5", "n": 3}],
    [{"px": "113378.0", "sz": "0.4", "n": 2}, {"px": "113379.0", "sz": "2.0", "n": 5}]]}}
TRADES_MSG = {"channel": "trades", "data": [
    {"coin": "BTC", "side": "B", "px": "113378.0", "sz": "0.01", "hash": "0x1", "time": 1754450974300, "tid": 1,
     "users": ["0xa", "0xb"]},
    {"coin": "BTC", "side": "A", "px": "113377.0", "sz": "0.02", "hash": "0x2", "time": 1754450974400, "tid": 2,
     "users": ["0xc", "0xd"]}]}
CTX_MSG = {"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {
    "dayNtlVlm": "1169046.29406", "funding": "0.0000125", "impactPxs": ["113376.5", "113378.5"], "markPx": "113377.5",
    "midPx": "113377.5", "openInterest": "688.11", "oraclePx": "113390.0", "premium": "0.00031774",
    "prevDayPx": "111000.0"}}}


class ParsingTests(unittest.TestCase):
    def test_levels(self):
        bids, asks = parse_levels(BOOK_MSG["data"]["levels"])
        self.assertEqual(bids[0], (113377.0, 7.6699))
        self.assertEqual(asks[0], (113378.0, 0.4))

    def test_trade_sides(self):
        t1, t2 = (parse_trade(t) for t in TRADES_MSG["data"])
        self.assertEqual((t1.side, t2.side), ("b", "s"))
        self.assertAlmostEqual(t1.ts, 1754450974.3)

    def test_ctx(self):
        c = parse_ctx(CTX_MSG["data"]["ctx"])
        self.assertAlmostEqual(c["funding"], 0.0000125)
        self.assertAlmostEqual(c["oracle"], 113390.0)
        self.assertAlmostEqual(c["prev_day_px"], 111000.0)

    def test_candle(self):
        c = parse_candle({"t": 1681923600000, "T": 1681924499999, "s": "BTC", "i": "15m", "o": "29295.0",
                          "c": "29258.0", "h": "29309.0", "l": "29250.0", "v": "0.98639", "n": 189})
        self.assertEqual((c.ts, c.open, c.close, c.volume), (1681923600, 29295.0, 29258.0, 0.98639))

    def test_walk_book(self):
        avg, filled = walk_book([(100.0, 1.0), (101.0, 1.0)], 1.5)
        self.assertAlmostEqual(avg, (100 + 50.5) / 1.5)
        self.assertAlmostEqual(filled, 1.5)
        self.assertAlmostEqual(walk_book([(100.0, 1.0)], 3.0)[1], 1.0)

    def test_handle_routes_messages(self):
        feed = HyperliquidFeed(["BTC"], rest=FakeRest())
        with self.assertRaises(DataError):
            feed.view("BTC")
        for msg in (BOOK_MSG, TRADES_MSG, CTX_MSG, {"channel": "pong"},
                    {"channel": "subscriptionResponse", "data": {"method": "subscribe"}}):
            feed.handle(msg)
        v = feed.view("BTC")
        self.assertAlmostEqual(v.mid, 113377.5)
        self.assertEqual(len(v.trades), 2)
        self.assertAlmostEqual(v.ctx["mark"], 113377.5)
        self.assertEqual(v.source, "ws")


    def test_bbo_refreshes_the_top_of_book_between_snapshots(self):
        now = [1754450975.0]
        feed = HyperliquidFeed(["BTC"], rest=FakeRest(), clock=lambda: now[0])
        feed.handle(BOOK_MSG)
        now[0] += 4.0                                       # 4 s later, only the best bid/ask moved
        feed.handle({"channel": "bbo", "data": {"coin": "BTC", "time": 1754450979000,
                                                "bbo": [{"px": "113378.0", "sz": "1.0", "n": 1}, {"px": "113380.0", "sz": "3.0", "n": 2}]}})
        v = feed.view("BTC")
        self.assertEqual(v.bids[0], (113378.0, 1.0))
        self.assertEqual(v.asks[0], (113380.0, 3.0))
        self.assertTrue(all(p < 113378.0 for p, _ in v.bids[1:]))   # levels the new top crossed are gone
        self.assertTrue(all(p > 113380.0 for p, _ in v.asks[1:]))
        self.assertEqual(v.received, now[0])                       # fresh, so the data-age check passes
        feed.handle({"channel": "bbo", "data": {"coin": "BTC", "time": 1, "bbo": [None, {"px": "1", "sz": "1"}]}})
        self.assertEqual(feed.view("BTC").bids[0], (113378.0, 1.0))  # one-sided bbo ignored

    def test_candle_stream_closes_candles(self):
        feed = HyperliquidFeed(["BTC"], rest=FakeRest(), clock=lambda: 0.0)
        buf = feed.buffers["BTC"]

        def push(t_min, close):
            feed.handle({"channel": "candle", "data": {"t": t_min * 60000, "T": t_min * 60000 + 59999, "s": "BTC", "i": "1m",
                                                       "o": "1", "c": str(close), "h": "2", "l": "0.5", "v": "3", "n": 4}})
        push(100, 10.0)
        push(100, 11.0)                                     # same minute, updated
        self.assertEqual(buf.closed_candles(100 * 60 + 30), [])     # still trading
        self.assertEqual([c.close for c in buf.closed_candles(101 * 60 + 1)], [11.0])   # its minute ended
        push(101, 12.0)                                     # next minute starts: 100 is closed for good
        self.assertEqual([(c.ts, c.close) for c in buf.candles], [(6000, 11.0)])
        self.assertEqual([c.ts for c in buf.closed_candles(101 * 60 + 30)], [6000])
        feed.handle({"channel": "candle", "data": {"t": 1, "T": 2, "s": "BTC", "i": "5m", "o": "1", "c": "1", "h": "1",
                                                   "l": "1", "v": "1", "n": 1}})              # other intervals ignored
        self.assertEqual(len(buf.candles), 1)


class CandleBufferTests(unittest.TestCase):
    """Minutes with no trades are never pushed; REST and the lab record them as flat candles, so the stream must too -
    but only while it is known to be connected (after a reconnect, the REST backfill repairs the gap instead)."""

    @staticmethod
    def c(t_min, close):
        from helix.market import Candle
        return Candle(t_min * 60, close, close + 1, close - 1, close, 5.0)

    def buf(self):
        from helix.hyperliquid import _Buffer
        b = _Buffer("BTC", clock=lambda: 0.0)
        b.needs_backfill = False                            # as after the first REST backfill
        return b

    def test_quiet_minutes_become_flat_candles(self):
        b = self.buf()
        b.on_candle(self.c(100, 10.0), 100 * 60 + 59.999, now=6001)
        b.on_candle(self.c(103, 13.0), 103 * 60 + 59.999, now=6181)   # 101 and 102 had no trades
        self.assertEqual([(c.ts // 60, c.close, c.volume) for c in b.candles],
                         [(100, 10.0, 5.0), (101, 10.0, 0.0), (102, 10.0, 0.0)])
        b.on_candle(self.c(103, 13.0), 103 * 60 + 59.999, now=6235)   # last update of 103
        out = b.closed_candles(105 * 60 + 2)                # 103 closed, 104 had no trades either
        self.assertEqual([(c.ts // 60, c.close) for c in out][-3:], [(102, 10.0), (103, 13.0), (104, 13.0)])

    def test_silent_stream_is_not_filled(self):
        b = self.buf()
        b.on_candle(self.c(100, 10.0), 100 * 60 + 59.999, now=6001)
        b.candle_msg = 0.0                                  # nothing heard for a long time: an outage, not quiet
        out = b.closed_candles(110 * 60)
        self.assertEqual([c.ts // 60 for c in out], [100])

    def test_reconnect_stops_filling_until_backfill(self):
        b = self.buf()
        b.on_candle(self.c(100, 10.0), 100 * 60 + 59.999, now=6001)
        b.on_candle(self.c(101, 11.0), 101 * 60 + 59.999, now=6061)
        b.on_reconnect()
        self.assertTrue(b.needs_backfill)
        self.assertIsNone(b.forming)
        b.on_candle(self.c(105, 15.0), 105 * 60 + 59.999, now=6301)   # minutes 101-104 unknown: not invented
        b.on_candle(self.c(106, 16.0), 106 * 60 + 59.999, now=6361)
        self.assertEqual([c.ts // 60 for c in b.candles], [100, 105])
        out = b.closed_candles(106 * 60 + 30)
        self.assertEqual([c.ts // 60 for c in out], [100, 105])

    def test_late_update_replaces_the_closed_candle(self):
        b = self.buf()
        b.on_candle(self.c(100, 10.0), 100 * 60 + 59.999, now=6001)
        b.on_candle(self.c(101, 11.0), 101 * 60 + 59.999, now=6061)
        b.on_candle(self.c(100, 10.5), 100 * 60 + 59.999, now=6062)
        self.assertEqual(b.candles[-1].close, 10.5)

    def test_one_sided_bbo(self):
        b = self.buf()
        b.on_bbo((99.0, 1.0), None, 1.0, 1.0)               # no book yet: ignored
        self.assertEqual(b.bids, [])
        b.on_book([(100.0, 1.0), (99.0, 1.0)], [(101.0, 1.0), (102.0, 1.0)], 1.0, "ws", 1.0)
        b.on_bbo(None, (100.5, 2.0), 2.0, 2.0)               # only the ask moved
        self.assertEqual(b.bids[0], (100.0, 1.0))
        self.assertEqual(b.asks, [(100.5, 2.0), (101.0, 1.0), (102.0, 1.0)])
        b.on_bbo((101.5, 1.0), None, 3.0, 3.0)               # a bid through the ask is not a valid book
        self.assertEqual(b.bids[0], (100.0, 1.0))


class HealthRaceTests(unittest.TestCase):
    def test_health_while_the_stream_appends(self):
        """health() runs on the engine thread while the WebSocket thread appends message times; iterating the live
        deque raised 'deque mutated during iteration' and failed engine ticks."""
        feed = HyperliquidFeed(["BTC"], rest=FakeRest(), clock=time.time)
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                feed.stats["msg_times"].append(time.time())
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            for _ in range(3000):
                feed.health()
        finally:
            stop.set()
            t.join(2)


class WeightBudgetTests(unittest.TestCase):
    def test_weights(self):
        from helix.hyperliquid import request_weight
        self.assertEqual(request_weight({"type": "l2Book", "coin": "BTC"}), 2.0)
        self.assertEqual(request_weight({"type": "metaAndAssetCtxs"}), 20.0)
        req = {"interval": "1m", "startTime": 0, "endTime": 600 * 60 * 1000}
        self.assertAlmostEqual(request_weight({"type": "candleSnapshot", "req": req}), 30.0)   # 600 candles: +10

    def test_bucket_waits_and_penalizes(self):
        from helix.hyperliquid import WeightBudget
        b = WeightBudget(600.0)                              # 10 weight per second
        t0 = time.monotonic()
        b.acquire(595)
        b.acquire(10)                                        # needs ~0.5 s of refill
        self.assertGreater(time.monotonic() - t0, 0.3)
        b.penalize(0.4)
        t1 = time.monotonic()
        b.acquire(1)
        self.assertGreater(time.monotonic() - t1, 0.35)

    def test_rest_feeds_the_budget(self):
        from helix.hyperliquid import HLRest, WeightBudget

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"levels": [[], []]}

        class Session:
            headers = {}

            def post(self, *a, **k):
                return Resp()
        b = WeightBudget(1200.0)
        HLRest("http://x", session=Session(), budget=b).post({"type": "l2Book", "coin": "BTC"})
        self.assertLess(b.tokens, 1200.0)


class FakeRest:
    def __init__(self):
        self.calls = []

    def asset_ctxs(self):
        self.calls.append("ctx")
        return {"BTC": parse_ctx(CTX_MSG["data"]["ctx"])}

    def candles(self, coin, interval, minutes, now):
        self.calls.append("candles")
        return []

    def l2_book(self, coin):
        self.calls.append("book")
        bids, asks = parse_levels(BOOK_MSG["data"]["levels"])
        return bids, asks, time.time()


class FeedIntegrationTests(unittest.TestCase):
    def test_stream_over_websocket(self):
        def script(conn, srv):
            subs = []
            for _ in range(3):
                _, data = srv.read_client_frame(conn)
                subs.append(json.loads(data))
            srv.received.extend(json.dumps(s) for s in subs)
            conn.sendall(server_frame(0x1, json.dumps({"channel": "subscriptionResponse", "data": subs[0]}).encode()))
            for msg in (BOOK_MSG, TRADES_MSG, CTX_MSG):
                conn.sendall(server_frame(0x1, json.dumps(msg).encode()))
            time.sleep(1.5)

        srv = MiniServer(script)
        rest = FakeRest()
        feed = HyperliquidFeed(["BTC"], rest=rest, ws_url=srv.url, stale_after_s=30)
        try:
            feed.start()
            v = None
            for _ in range(50):
                try:
                    v = feed.view("BTC")
                    if v.trades:
                        break
                except DataError:
                    pass
                time.sleep(0.05)
            self.assertIsNotNone(v)
            self.assertEqual(v.source, "ws")
            self.assertEqual(len(v.trades), 2)
            self.assertAlmostEqual(v.ctx["funding"], 0.0000125)
            subs = [json.loads(x) for x in srv.received]
            self.assertEqual({s["subscription"]["type"] for s in subs}, {"l2Book", "trades", "activeAssetCtx"})
            self.assertTrue(all(s["method"] == "subscribe" for s in subs))
            self.assertTrue(feed.health()["connected"])
        finally:
            feed.stop()
            srv.close()

    def test_rest_backup_when_stream_is_down(self):
        rest = FakeRest()
        feed = HyperliquidFeed(["BTC"], rest=rest, ws_url="ws://127.0.0.1:9/ws", stale_after_s=0.5)
        try:
            feed.start()
            v = None
            for _ in range(60):
                try:
                    v = feed.view("BTC")
                    break
                except DataError:
                    time.sleep(0.05)
            self.assertIsNotNone(v)
            self.assertEqual(v.source, "rest")
            self.assertIn("book", rest.calls)
            self.assertFalse(feed.health()["connected"])
        finally:
            feed.stop()


class SimTests(unittest.TestCase):
    def test_sim_views(self):
        clock = [1_791_200_000.0]
        feed = SimPerpFeed(["BTC", "ETH", "SOL"], lambda: clock[0], seed=2)
        for coin in ("BTC", "ETH", "SOL"):
            v = feed.view(coin)
            self.assertGreater(len(v.mids), 1000)
            self.assertEqual(len(v.bids), 20)
            self.assertLess(v.best_bid, v.best_ask)
            self.assertGreater(len(v.candles), 100)
        clock[0] += 10
        q = feed.quote("BTC")
        self.assertEqual(q.trades, [])
        self.assertIsNotNone(q.mid)


if __name__ == "__main__":
    unittest.main()
