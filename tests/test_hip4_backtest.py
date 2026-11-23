"""HIP-4 replay backtester on a synthetic tape: the sniper only fills against the book that exists AFTER its latency,
the arb needs both legs to survive, the market maker in replay fills only on trades through its quotes, settlement
records pay out, open positions are marked at the market mid (never our own fair), and the web endpoints answer."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.hip4_backtest import run_backtest  # noqa: E402
from helix.market import Candle  # noqa: E402

T0 = 1_791_300_000.0                                    # 15:20 UTC
EXP = 1_791_301_500                                     # 15:45 UTC
M1 = {"id": 9204, "coin": "BTC", "venue": "out", "strike": 85_000.0, "expiry": EXP, "rule": ">",
      "source": "BTC-USDC perp mark", "twap_s": 60}
M2 = {**M1, "id": 9205, "strike": 85_100.0}


class FakeRest:
    def __init__(self, settled=None):
        self.settled = settled or {}

    def candles(self, coin, interval, minutes, now):
        out, end = [], int(now) // 300 * 300
        for k in range(minutes // 5):
            t = end - (minutes // 5 - k) * 300
            px = 85_000 * (1 + 0.002 * math.sin(t / 1500) + 0.001 * math.sin(t / 211))
            out.append(Candle(t, px, px, px, px, 1.0))
        return out

    def post(self, body):
        return self.settled.get(body.get("outcome"))


def book(t, coin, bid, ask, sz=50):
    return {"t": t, "k": "ob", "c": coin, "et": int(t * 1000), "b": [[bid, sz]] if bid is not None else [],
            "a": [[ask, sz]] if ask is not None else []}


def write_tape(d: Path, records):
    (d / "tape").mkdir(parents=True, exist_ok=True)
    with open(d / "tape" / "20261006-15.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def base(t=T0):
    return [{"t": t, "k": "m", "mk": [M1, M2]},
            {"t": t + 0.1, "k": "pb", "c": "BTC", "et": 0, "b": 84_999.5, "bs": 1, "a": 85_000.5, "as": 1}]


class SniperTests(unittest.TestCase):
    def test_fills_only_if_the_stale_quote_survives_the_latency(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs = base() + [
                book(T0 + 1, "#92040", 0.10, 0.20),          # ask 20c with fair ~50c: a stale quote
                book(T0 + 1.1, "#92040", 0.10, 0.20),
                book(T0 + 1.5, "#92040", 0.10, 0.20),        # still there after 300 ms: filled at 20c
                book(T0 + 30, "#92040", 0.10, 0.20),
                book(T0 + 30.1, "#92040", 0.45, 0.55),       # gone 100 ms later: the order arrives too late
                book(T0 + 31, "#92040", 0.45, 0.55),
                {"t": T0 + 1600, "k": "x", "id": 9204, "y": 1.0},
                {"t": T0 + 1601, "k": "hb"},
            ]
            write_tape(d, recs)
            res = run_backtest(d, "sniper", {"sides": 0, "edge": 0.05, "latency_ms": 300, "size": 10, "cooldown_s": 20,
                                             "min_notional": 0, "band_lo": 0.0}, rest=FakeRest())
            self.assertEqual(res["fills"], 1)
            self.assertEqual(res["missed"], 1)
            self.assertAlmostEqual(res["sample"][0]["px"], 0.20)
            self.assertEqual(res["settled_markets"], 1)
            self.assertAlmostEqual(res["pnl_settled"], 10 * (1 - 0.20) - 10 * 0.0005)
            self.assertAlmostEqual(res["pnl"], res["pnl_settled"], places=2)


class SniperMinimumTests(unittest.TestCase):
    def test_orders_are_raised_to_the_exchange_minimum(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_tape(d, base() + [book(T0 + 1, "#92040", 0.10, 0.20), book(T0 + 1.5, "#92040", 0.10, 0.20),
                                    {"t": T0 + 2, "k": "hb"}])
            res = run_backtest(d, "sniper", {"sides": 0, "edge": 0.05, "latency_ms": 300, "size": 10, "band_lo": 0.0},
                               rest=FakeRest())
            self.assertEqual(res["sample"][0]["qty"], 50)               # 10 x 20c = $2 -> 50 contracts = $10
            banded = run_backtest(d, "sniper", {"sides": 0, "edge": 0.05, "latency_ms": 300, "size": 10, "band_lo": 0.30,
                                                "band_hi": 0.70}, rest=FakeRest())
            self.assertEqual(banded["fills"], 0)                        # a 20c contract is outside the 30-70c band


class SniperAutoSidesTests(unittest.TestCase):
    def test_auto_takes_only_the_trend_side(self):
        class Falling(FakeRest):                     # BTC fell 6% over the last day: 6 h ago it was well above spot
            def candles(self, coin, interval, minutes, now):
                out = super().candles(coin, interval, minutes, now)
                return [Candle(c.ts, c.open, c.high, c.low, c.close * (1 + 0.06 * (c.ts > now - 86400)
                                                                          * (now - 3600 - c.ts) / 86400), 1.0)
                        for c in out]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs = base() + [book(T0 + 1, "#92040", 0.10, 0.20), book(T0 + 1.5, "#92040", 0.10, 0.20),   # a BUY
                             book(T0 + 30, "#92050", 0.80, 0.90), book(T0 + 30.5, "#92050", 0.80, 0.90),  # a SELL
                             {"t": T0 + 31, "k": "hb"}]
            write_tape(d, recs)
            p = {"edge": 0.05, "latency_ms": 300, "size": 10, "min_notional": 0, "sides": 2,
                 "trend_lookback_min": 360, "trend_threshold": 0.005, "trend_neutral": 9}
            res = run_backtest(d, "sniper", p, rest=Falling())
            self.assertEqual([f["side"] for f in res["sample"]], [-1])          # the buy is skipped in a downtrend
            shutil.rmtree(d / "tape" / "bars")                                   # the falling bars were cached
            flat = run_backtest(d, "sniper", p, rest=FakeRest())
            self.assertEqual(flat["fills"], 0)                                   # no trend + neutral none: nothing
            both = run_backtest(d, "sniper", {**p, "trend_neutral": 0}, rest=FakeRest())
            self.assertEqual(sorted(f["side"] for f in both["sample"]), [-1, 1])


class ArbTests(unittest.TestCase):
    def test_yes_no_lock_needs_both_legs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs = base() + [
                book(T0 + 1, "#92040", 0.40, 0.45),
                book(T0 + 1.05, "#92041", 0.40, 0.50),       # Yes 45 + No 50 = 95c < $1: buy both
                {"t": T0 + 2, "k": "hb"},                    # both still there after 300 ms
                book(T0 + 5, "#92050", 0.30, 0.35),
                book(T0 + 5.05, "#92051", 0.30, 0.40),       # 75c: another lock
                book(T0 + 5.1, "#92051", 0.30, 0.70),        # ... No leg gone before we arrive
                book(T0 + 6, "#92050", 0.30, 0.35),
                {"t": T0 + 1600, "k": "x", "id": 9204, "y": 0.0},
                {"t": T0 + 1601, "k": "hb"},
            ]
            write_tape(d, recs)
            res = run_backtest(d, "arb", {"latency_ms": 300, "size": 10}, rest=FakeRest())
            self.assertEqual(res["signals"]["yes_no_buy"], 2)
            self.assertEqual(res["missed"], 1)
            self.assertEqual(res["fills"], 2)
            self.assertEqual(res["settled_markets"], 1)
            # paid 45c + 50c for $1 whatever happens, minus two fees
            self.assertAlmostEqual(res["pnl_settled"], 10 * (1 - 0.95) - 2 * 10 * 0.0005)


class MarketMakerReplayTests(unittest.TestCase):
    def test_live_market_maker_code_in_replay(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs = base() + [book(T0 + 1, "#92040", 0.30, 0.70)]
            recs += [{"t": T0 + 10, "k": "ot", "c": "#92040", "et": int((T0 + 10) * 1000), "p": 0.05, "s": 15,
                      "sd": "A", "tid": 1},                 # a seller far through any sane bid
                     {"t": T0 + 12, "k": "ot", "c": "#92040", "et": int((T0 + 12) * 1000), "p": 0.50, "s": 5,
                      "sd": "B", "tid": 2},                 # a buyer below our ask (fair ~50c + 2c): no fill
                     {"t": T0 + 1600, "k": "x", "id": 9204, "y": 1.0}, {"t": T0 + 1601, "k": "hb"}]
            write_tape(d, recs)
            res = run_backtest(d, "mm", {"quote_every_s": 1.0, "market_weight": 0.0}, rest=FakeRest())
            self.assertEqual(res["fills"], 1)
            self.assertEqual(res["sample"][0]["side"], 1)
            self.assertEqual(res["sample"][0]["qty"], 15)
            self.assertEqual(res["settled_markets"], 1)


class OpenMarkTests(unittest.TestCase):
    def test_open_positions_are_marked_at_the_market_not_our_fair(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs = base() + [book(T0 + 1, "#92040", 0.10, 0.20), book(T0 + 2, "#92040", 0.10, 0.20),
                             book(T0 + 3, "#92040", 0.18, 0.22), {"t": T0 + 4, "k": "hb"}]
            write_tape(d, recs)
            res = run_backtest(d, "sniper", {"sides": 0, "edge": 0.05, "latency_ms": 300, "size": 10, "min_notional": 0,
                                             "band_lo": 0.0}, rest=FakeRest())
            self.assertEqual(res["open_markets"], 1)
            # bought 10 at 20c, the market mid is 20c: worth what we paid, minus the fee - not fair (~50c)
            self.assertAlmostEqual(res["pnl"], -10 * 0.0005, places=3)


class EndpointTests(unittest.TestCase):
    def test_backtest_api_validates_and_runs(self):
        from starlette.testclient import TestClient
        from helix.server import create_app
        from helix import server as srv
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_tape(d / "data", base() + [book(T0 + 1, "#92040", 0.4, 0.45), {"t": T0 + 2, "k": "hb"}])

            class Eng:
                cfg = {"network": "mainnet"}

                def snapshot(self):
                    return {"engine": {"loop": 0, "status": "running"}}

            class Tape:
                coins, markets, last_books, recent_trades = ["BTC"], {9204: {**M1, "expiry": int(time.time()) + 600}}, {}, {}
                last_perp = {"BTC": (0, 85_000.0)}

                def to_dict(self):
                    return {"enabled": True}

            old = srv.WEB_DIR
            srv.WEB_DIR = d / "web"
            (d / "web").mkdir()
            (d / "web" / "hip4.html").write_text("<p>hip4</p>", encoding="utf-8")
            try:
                app = create_app(Eng(), run_loop=False, tape=Tape())
                with TestClient(app) as c:
                    self.assertIn("hip4", c.get("/hip4").text)
                    self.assertEqual(c.post("/api/backtest", json={"strategy": "nope"}).status_code, 400)
                    self.assertEqual(c.post("/api/backtest", json={"strategy": "arb", "params": {"bogus": 1}}).status_code, 400)
                    self.assertEqual(c.post("/api/backtest", json={"strategy": "arb", "params": {"size": "x"}}).status_code, 400)
                    self.assertEqual(c.get("/api/hip4/market?id=1").status_code, 404)
                    self.assertEqual(c.get("/api/hip4/ladder?coin=BTC").json()["groups"][0]["rows"][0]["id"], 9204)
                    self.assertIn("coins", c.get("/api/hip4/overview").json())
                    u = c.get("/api/hip4/universe?since=0").json()
                    self.assertEqual(u["markets"][0]["id"], 9204)
                    self.assertEqual(u["coins"]["BTC"]["spot"], 85_000.0)
                    self.assertTrue(c.get("/api/backtest").json()["tape_from"])
            finally:
                srv.WEB_DIR = old


if __name__ == "__main__":
    unittest.main()
