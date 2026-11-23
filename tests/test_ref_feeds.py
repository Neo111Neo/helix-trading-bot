"""Outside reference prices (helix/ref_feeds.py) and the lab rules that use them: Kalshi quotes are parsed, aged and
recorded; Coinbase moves are measured on fresh data only; the Kalshi variant bets only on the same 15-minute question
and only when HIP-4 sits >= 5c through Kalshi; the Coinbase veto skips bets right after a move against them; no lab
bets on xyz: perps across a weekday US open (with US daylight saving)."""

from __future__ import annotations

import calendar
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.ref_feeds import CexFeed, KalshiFeed  # noqa: E402
from helix.shadow_lab import EXTRA_VARIANTS, ShadowLab, _spans_us_open, _us_open  # noqa: E402
from tests.test_shadow_lab import SELL, Sniper  # noqa: E402

EXP = calendar.timegm((2026, 10, 12, 22, 45, 0))                   # a quarter-hour expiry
NOW = EXP - 10 * 60


def kalshi_api(bid="0.4000", ask="0.4200", strike=82_479.7, close="2026-10-12T22:45:00Z", list_age=9.0, mkt_age=0.0):
    """A fake Kalshi: the market LIST (CDN-cached, Age list_age, gives the strike) and the market's ORDER BOOK (Age
    mkt_age): Yes bids, and No bids whose best is 1 - the Yes ask. Levels ascending, as Kalshi sends them."""
    calls = []

    def fetch(url):
        calls.append(url)
        if "series_ticker=KXBTC15M" in url:
            return {"markets": [{"ticker": "KXBTC15M-X", "close_time": close, "floor_strike": strike}]}, list_age
        if url.endswith("/markets/KXBTC15M-X/orderbook"):
            no_best = f"{1 - float(ask):.4f}"
            return {"orderbook_fp": {"yes_dollars": [["0.0100", "9000.00"], [bid, "500.00"]],
                                     "no_dollars": [["0.0100", "8000.00"], [no_best, "700.00"]]}}, mkt_age
        if url.endswith("/markets/KXBTC15M-X"):
            raise AssertionError("prices must come from the order book, not the ~10 s market snapshot")
        return {"markets": []}, 0.0
    return fetch, calls


class KalshiFeedTests(unittest.TestCase):
    def test_prices_come_from_the_uncached_market_page_stamped_conservatively(self):
        clock = [NOW]
        with tempfile.TemporaryDirectory() as d:
            fetch, calls = kalshi_api(mkt_age=0.5)
            f = KalshiFeed(Path(d), fetch=fetch, clock=lambda: clock[0], spot_fn=lambda c: 85_000.0)
            f.poll_coin("BTC")
            self.assertTrue(any(u.endswith("/markets/KXBTC15M-X/orderbook") for u in calls))   # live order book
            q = f.quote("BTC", EXP, NOW)
            self.assertEqual((q["bid"], q["ask"], q["strike"], q["spot"]), (0.40, 0.42, 82_479.7, 85_000.0))
            self.assertEqual((q["bid_sz"], q["ask_sz"]), (500.0, 700.0))
            self.assertAlmostEqual(q["ts"], NOW - 0.5)                                # send time minus cache age
            self.assertIsNone(f.quote("ETH", EXP, NOW))
            self.assertIsNone(f.quote("BTC", EXP, NOW + 2))                          # stale after MAX_AGE_S
            f.poll_coin("BTC")                                                       # unchanged: not re-recorded
            lines = (Path(d) / "kalshi" / "20261012.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["e"], EXP)
            self.assertEqual(sum("series_ticker" in u for u in calls), 1)            # the list is re-read rarely
            fetch, _ = kalshi_api(mkt_age=12.0)                                      # a cached price is refused
            f2 = KalshiFeed(None, fetch=fetch, clock=lambda: NOW)
            with self.assertRaises(RuntimeError):
                f2.poll_coin("BTC")
            self.assertIsNone(f2.quote("BTC", EXP, NOW))

    def test_failure_marks_stale_never_raises(self):
        stop = threading.Event()

        def boom(url):
            stop.set()
            raise OSError("HTTP Error 429: Too Many Requests")
        f = KalshiFeed(None, fetch=boom, clock=lambda: NOW)
        f._loop("BTC", stop)
        self.assertEqual(f.fails, 1)
        self.assertIn("429", f.status()["error"])
        self.assertFalse(f.status()["ok"])


class CexFeedTests(unittest.TestCase):
    def tick(self, f, t, mid, clock):
        clock[0] = t
        f.on_message(json.dumps({"type": "ticker", "product_id": "BTC-USD", "best_bid": str(mid - 0.5),
                                 "best_ask": str(mid + 0.5)}))

    def test_move_needs_fresh_data(self):
        clock = [NOW]
        f = CexFeed(None, clock=lambda: clock[0])
        self.assertIsNone(f.move("BTC", NOW, 2.0))
        self.tick(f, NOW, 100_000.0, clock)
        self.tick(f, NOW + 1.0, 100_005.0, clock)
        self.tick(f, NOW + 2.5, 100_010.0, clock)
        self.assertAlmostEqual(f.move("BTC", NOW + 2.5, 2.0), 100_010 / 100_000 - 1, places=7)
        self.assertIsNone(f.move("BTC", NOW + 9, 2.0))                               # latest price is stale
        f.on_message(json.dumps({"type": "subscriptions"}))                          # other messages ignored
        f.on_message("not json")
        f.on_message(json.dumps({"type": "error", "message": "Failed to subscribe", "reason": "bad product"}))
        self.assertIn("Failed to subscribe", f.status()["error"])

    def test_a_silent_stream_is_reconnected(self):
        """Review r7: a half-open socket that only ever times out used to keep the feed stale until a restart."""
        clock, stop, opened = [NOW], threading.Event(), []

        class Silent:
            def settimeout(self, s):
                pass

            def recv(self):
                clock[0] += 10.0
                raise TimeoutError()

            def close(self):
                pass

        def connect():
            opened.append(clock[0])
            if len(opened) >= 2:
                stop.set()
            return Silent()
        f = CexFeed(None, clock=lambda: clock[0], connect=connect)
        f.run(stop)
        self.assertEqual(len(opened), 2)
        self.assertEqual(f.reconnects, 1)
        self.assertIn("no Coinbase ticker", f.error)


class LabRefTests(unittest.TestCase):
    class FakeKalshi:
        def __init__(self, q):
            self.q = q

        def quote(self, coin, expiry, now):
            return self.q

        def status(self):
            return {"ok": True}

    class FakeCex:
        def __init__(self, mv):
            self.mv = mv

        def move(self, coin, now, window_s):
            return self.mv

        def status(self):
            return {"ok": True}

    class SpotModel:
        """Fair 50c at 85,000 and +1c per +1 bp of the coin's price."""
        def fair(self, coin, spot, strike, now, expiry, twap_s=0):
            return 0.5 + (spot / 85_000.0 - 1) * 100

    def lab(self, name, refs, book, market=None, spot=85_000.0):
        sn = Sniper()
        sn.model, sn.spot_fn = self.SpotModel(), (lambda coin: spot)
        m = market or {"id": 1, "coin": "BTC", "strike": 82_480.0, "expiry": EXP, "twap_s": 60, "venue": "out"}
        sn.markets, sn.models, sn.books = {1: m}, {1: 0.5}, {1: book}
        L = ShadowLab(sn, None, variants={name: EXTRA_VARIANTS[name]}, clock=lambda: NOW, refs=refs)
        return L, sn

    def test_kalshi_quote_is_moved_with_the_price_since_and_refused_after_a_big_move(self):
        q = {"bid": 0.40, "ask": 0.42, "strike": 82_479.7, "ts": NOW, "spot": 85_000.0 * (1 - 0.0002)}
        book = {"bid": 0.48, "ask": 0.70, "bid_sz": 100, "ask_sz": 100}
        # +2 bp since Kalshi's quote: Kalshi's ask is really 44c, so a 48c bid is NOT 5c through it
        L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi(q)}, book)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 0)
        L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi(q)}, SELL)            # 58c bid: still through 44c + 5c
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 1)
        # moved 5 bp since the quote: not trusted at all
        L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi({**q, "spot": 85_000.0 * (1 - 0.0005)})}, SELL)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 0)
        # the model calls the sale 10c+ bad (fair 75c vs a 58c bid): backstop
        q2 = {**q, "spot": 85_000.0 * 1.0025}
        L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi(q2)}, SELL, spot=85_000.0 * 1.0025)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 0)

    def test_kalshi_variant_bets_only_through_kalshi_on_the_same_question(self):
        q = {"bid": 0.40, "ask": 0.42, "strike": 82_479.7, "ts": NOW, "spot": 85_000.0}
        # HIP-4 bid 58c >= Kalshi ask 42c + 5c: sell Yes, with Kalshi's price as the fair
        L, sn = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi(q)}, SELL)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 1)
        self.assertEqual(L._checks[1][0]["o"]["side"], -1)
        # HIP-4 ask 42c <= Kalshi bid 60c - 5c: buy Yes
        L, sn = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi({**q, "bid": 0.60, "ask": 0.62})},
                         {"bid": 0.30, "ask": 0.42, "bid_sz": 100, "ask_sz": 100})
        L.on_book(1, NOW)
        self.assertEqual((L.led["kalshi_5c"]["sent"], L._checks[1][0]["o"]["side"]), (1, 1))
        # within 5c of Kalshi: nothing
        L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi({**q, "bid": 0.54, "ask": 0.56})}, SELL)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 0)
        # a different question: strike 30 bp away, a non-'out' market, or not a quarter-hour expiry
        for mk in ({"strike": 82_730.0}, {"venue": "hl"}, {"expiry": EXP + 60}):
            m = {"id": 1, "coin": "BTC", "strike": 82_480.0, "expiry": EXP, "twap_s": 60, "venue": "out", **mk}
            L, _ = self.lab("kalshi_5c", {"kalshi": self.FakeKalshi(q)}, SELL, market=m)
            L.on_book(1, NOW)
            self.assertEqual(L.led["kalshi_5c"]["sent"], 0, mk)
        # no Kalshi feed: never bets
        L, _ = self.lab("kalshi_5c", {}, SELL)
        L.on_book(1, NOW)
        self.assertEqual(L.led["kalshi_5c"]["sent"], 0)

    def test_coinbase_veto(self):
        for mv, sent in ((0.0001, 0), (-0.0001, 1), (None, 0), (0.00004, 1)):        # +1 bp up vetoes a Yes sale;
            #                                                                          no fresh price: no bet
            L, _ = self.lab("both_last15_cbveto", {"cex": self.FakeCex(mv)}, SELL)
            L.on_book(1, NOW)
            self.assertEqual(L.led["both_last15_cbveto"]["sent"], sent, mv)
            self.assertEqual(L.led["both_last15_cbveto"]["vetoed"], 1 - sent)
        L, _ = self.lab("both_last15_cbveto", {"cex": self.FakeCex(-0.0001)},
                        {"bid": 0.30, "ask": 0.42, "bid_sz": 100, "ask_sz": 100})      # a Yes buy after a drop
        L.on_book(1, NOW)
        self.assertEqual(L.led["both_last15_cbveto"]["vetoed"], 1)


class UsOpenTests(unittest.TestCase):
    def test_open_follows_us_daylight_saving(self):
        self.assertEqual(_us_open(calendar.timegm((2026, 10, 12, 3, 0, 0))), calendar.timegm((2026, 10, 12, 13, 30, 0)))
        self.assertEqual(_us_open(calendar.timegm((2026, 12, 1, 3, 0, 0))), calendar.timegm((2026, 12, 1, 14, 30, 0)))
        self.assertEqual(_us_open(calendar.timegm((2026, 3, 6, 3, 0, 0))), calendar.timegm((2026, 3, 6, 14, 30, 0)))
        self.assertEqual(_us_open(calendar.timegm((2026, 3, 9, 3, 0, 0))), calendar.timegm((2026, 3, 9, 13, 30, 0)))
        self.assertEqual(_us_open(calendar.timegm((2026, 11, 2, 3, 0, 0))), calendar.timegm((2026, 11, 2, 14, 30, 0)))

    def test_xyz_bets_sit_the_open_out(self):
        t = lambda *a: calendar.timegm((2026, 10, 13) + a + (0,))                 # noqa: E731 - a Tuesday
        self.assertTrue(_spans_us_open(t(13, 25), t(13, 35)))                       # across the open
        self.assertTrue(_spans_us_open(t(13, 40), t(13, 50)))                        # first 15 minutes after it
        self.assertFalse(_spans_us_open(t(14, 0), t(14, 15)))
        self.assertTrue(_spans_us_open(t(20, 0), t(20, 0) + 86400))                  # a 24 h bet spans tomorrow's
        sat = lambda *a: calendar.timegm((2026, 10, 17) + a + (0,))                # noqa: E731
        self.assertFalse(_spans_us_open(sat(13, 25), sat(13, 35)))                   # weekend: no open
        sn = Sniper()
        sn.markets = {1: {"id": 1, "coin": "xyz:MU", "strike": 85_000.0, "expiry": t(13, 35), "twap_s": 60}}
        sn.models, sn.books = {1: 0.5}, {1: SELL}
        L = ShadowLab(sn, None, clock=lambda: t(13, 25))
        L.on_book(1, t(13, 25))
        self.assertEqual(sum(x["sent"] for x in L.led.values()), 0)
        L2 = ShadowLab(sn, None, clock=lambda: t(13, 25), xyz_open_guard=False)
        L2.on_book(1, t(13, 25))
        self.assertGreater(sum(x["sent"] for x in L2.led.values()), 0)


if __name__ == "__main__":
    unittest.main()
