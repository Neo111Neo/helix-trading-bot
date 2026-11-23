"""HIP-4 market maker (paper): the cached fair model equals hip4.strike_state's model_har, quotes never cross the
market and lean against inventory, fills need a real trade THROUGH our price, settlement pays from settledOutcome,
the risk cap and kill switch hold, and the book survives a restart."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import DEFAULTS, ConfigError, validate  # noqa: E402
from helix.hip4 import strike_state  # noqa: E402
from helix.hip4_mm import FairModel, HIP4MarketMaker, trend_return, trend_sides  # noqa: E402
from helix.market import Candle  # noqa: E402

NOW = 1_791_300_000.0


def candles(n, end, base=85_000.0):
    out = []
    for k in range(n):
        t = end - (n - k) * 300
        px = base * (1 + 0.004 * math.sin(t / 2000) + 0.0015 * math.sin(t / 333) + 0.0007 * math.sin(t / 97))
        out.append(Candle(t, px, px * 1.0005, px * 0.9995, px, 1.0))
    return out


class FakeRest:
    def __init__(self, settled=None, outcomes=None):
        self.settled, self.outcomes, self.calls = settled or {}, outcomes or [], []

    def post(self, body):
        self.calls.append(body["type"])
        if body["type"] == "outcomeMeta":
            return {"outcomes": self.outcomes}
        if body["type"] == "settledOutcome":
            return self.settled.get(body["outcome"])
        return None

    def candles(self, coin, interval, minutes, now):
        return [c for c in candles(15 * 288 + 20, int(now) // 300 * 300) if c.ts >= now - minutes * 60]


def cfg(**over):
    c = copy.deepcopy(DEFAULTS)
    c["market"]["coins"] = ["BTC", "ETH", "SOL", "HYPE"]
    c["hip4_mm"].update({"enabled": True, **over})
    return c


def maker(tmp=None, rest=None, clock=lambda: NOW, **over):
    return HIP4MarketMaker(cfg(**over), tmp, spot_fn=lambda coin: 85_000.0, rest=rest or FakeRest(), clock=clock)


MKT = {"id": 9204, "coin": "BTC", "venue": "out", "strike": 85_000.0, "expiry": NOW + 10 * 60, "rule": ">",
       "source": "BTC-USDC perp mark", "twap_s": 60}


def armed(mm, fair=0.50, bid=0.47, ask=0.53, market=MKT):
    """One market with a fresh book and a known fair, then one requote."""
    mm.markets = {market["id"]: dict(market)}
    mm.books = {market["id"]: {"bid": bid, "ask": ask, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}
    mm.fairs = {market["id"]: (fair, NOW)}
    mm._requote(NOW)
    return mm.quotes.get(market["id"])


class FairModelTests(unittest.TestCase):
    def test_matches_strike_state_exactly(self):
        c5 = candles(15 * 288 + 4, int(NOW) // 300 * 300)
        fm = FairModel()
        fm.set_bars("BTC", c5)
        spot = c5[-1].close
        for tte_min, k_mult, twap in ((14, 1.001, 60), (14, 0.998, 90), (55, 1.0, 0), (9, 1.0005, 60), (62, 0.995, 0)):
            ref = strike_state("BTC", c5, [], spot, NOW, NOW + tte_min * 60, spot * k_mult, {}, twap)["model_har"]
            got = fm.fair("BTC", spot, spot * k_mult, NOW, NOW + tte_min * 60, twap)
            self.assertAlmostEqual(got, ref, places=12, msg=(tte_min, k_mult, twap))

    def test_no_bars_no_fair(self):
        self.assertIsNone(FairModel().fair("BTC", 85_000, 85_000, NOW, NOW + 600))


class QuoteTests(unittest.TestCase):
    def test_symmetric_quotes_around_fair(self):
        q = armed(maker(), fair=0.50, bid=0.40, ask=0.60)
        self.assertAlmostEqual(q["bid"], 0.48)
        self.assertAlmostEqual(q["ask"], 0.52)
        self.assertEqual((q["bid_sz"], q["ask_sz"]), (20, 20))
        self.assertTrue(q["bid_best"] and q["ask_best"])       # inside a 20-cent market

    def test_never_crosses_the_market(self):
        q = armed(maker(), fair=0.60, bid=0.50, ask=0.55)       # fair above the whole book
        self.assertAlmostEqual(q["bid"], 0.549)                 # one tick under the ask, not 0.58
        self.assertAlmostEqual(q["ask"], 0.62)

    def test_rounds_away_from_fair(self):
        q = armed(maker(), fair=0.5004, bid=0.30, ask=0.70)
        self.assertAlmostEqual(q["bid"], 0.480)
        self.assertAlmostEqual(q["ask"], 0.521)

    def test_inventory_skew_and_position_cap(self):
        mm = maker()
        mm.pos["9204"] = {"coin": "BTC", "strike": 85_000, "expiry": MKT["expiry"], "venue": "out", "qty": 100.0,
                          "cash": -50.0, "fees": 0.0, "fills": 5, "volume": 100}
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertAlmostEqual(q["bid"], 0.46)                  # 0.50 - 0.02 edge - 0.02 skew at max position
        self.assertAlmostEqual(q["ask"], 0.50)
        self.assertEqual(q["bid_sz"], 0)                        # full long: no more bids
        self.assertEqual(q["ask_sz"], 20)

    def test_no_quotes_near_expiry_tails_stale_books_or_killed(self):
        self.assertIsNone(armed(maker(), market={**MKT, "expiry": NOW + 120}))       # inside stop_min
        self.assertIsNone(armed(maker(), market={**MKT, "expiry": NOW + 80 * 60}))   # beyond max_tte_min
        self.assertIsNone(armed(maker(), fair=0.03))
        mm = maker()
        mm.markets = {9204: dict(MKT)}
        mm.books = {9204: {"bid": 0.4, "ask": 0.6, "bid_sz": 1, "ask_sz": 1, "ts": NOW - 31}}
        mm.fairs = {9204: (0.5, NOW)}
        mm._requote(NOW)
        self.assertEqual(mm.quotes, {})
        mm = maker()
        mm.killed = True
        self.assertIsNone(armed(mm))

    def test_risk_cap_allows_only_reducing_quotes(self):
        mm = maker(max_risk_usd=10)
        mm.pos["9204"] = {"coin": "BTC", "strike": 85_000, "expiry": MKT["expiry"], "venue": "out", "qty": 40.0,
                          "cash": -20.0, "fees": 0.0, "fills": 2, "volume": 40}
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertEqual(q["bid_sz"], 0)
        self.assertEqual(q["ask_sz"], 20)


class FillTests(unittest.TestCase):
    def test_fill_needs_a_trade_through_our_price(self):
        mm = maker()
        armed(mm, fair=0.50, bid=0.47, ask=0.53)                # our 0.48 bid improves the 0.47 market bid
        mm.on_trade(9204, 0.49, 10, aggressor_buy=False, ts=NOW + 1)     # seller above our bid: not us
        self.assertEqual(mm.fills, [])
        mm.on_trade(9204, 0.475, 30, aggressor_buy=False, ts=NOW + 1)    # seller through our bid
        self.assertEqual(len(mm.fills), 1)
        f = mm.fills[0]
        self.assertEqual((f["side"], f["px"], f["qty"]), (1, 0.48, 20))  # our price, our size (trade was 30)
        self.assertAlmostEqual(f["edge"], 0.02)
        mm.on_trade(9204, 0.40, 30, aggressor_buy=False, ts=NOW + 2)     # quote used up until requoted
        self.assertEqual(len(mm.fills), 1)

    def test_at_our_price_only_when_alone_at_the_top(self):
        mm = maker()
        armed(mm, fair=0.50, bid=0.47, ask=0.53)
        mm.on_trade(9204, 0.52, 5, aggressor_buy=True, ts=NOW + 1)       # buyer at our 0.52 ask; we improve 0.53
        self.assertEqual((mm.fills[-1]["side"], mm.fills[-1]["qty"]), (-1, 5))
        mm2 = maker()
        armed(mm2, fair=0.50, bid=0.48, ask=0.52)                        # we only JOIN the market at 0.48 / 0.52
        mm2.on_trade(9204, 0.52, 5, aggressor_buy=True, ts=NOW + 1)
        mm2.on_trade(9204, 0.48, 5, aggressor_buy=False, ts=NOW + 1)
        self.assertEqual(mm2.fills, [])                                  # others are ahead of us in the queue

    def test_trades_before_the_quote_do_not_fill(self):
        mm = maker()
        armed(mm)
        mm.on_trade(9204, 0.40, 10, aggressor_buy=False, ts=NOW - 5)
        self.assertEqual(mm.fills, [])

    def test_handle_routes_yes_trades_dedupes_and_skips_history(self):
        mm = maker()
        armed(mm)
        mm._snapshot_due.add("#92040")
        t = {"coin": "#92040", "side": "A", "px": "0.40", "sz": "5", "time": int((NOW + 99) * 1000), "tid": 1}
        old = {**t, "tid": 2}
        mm.handle({"channel": "trades", "data": [old]}, now=NOW + 1)          # history sent on subscribe
        self.assertEqual(mm.fills, [])
        no_side = {**t, "coin": "#92041", "tid": 3}
        mm.handle({"channel": "trades", "data": [no_side, t, t]}, now=NOW + 1)
        self.assertEqual(len(mm.fills), 1)
        self.assertEqual(mm.fills[0]["ts"], NOW + 1)          # our receive time, not the exchange's clock

    def test_quote_needs_a_round_trip_before_it_can_fill(self):
        mm = maker()
        armed(mm)                                              # quotes placed at NOW
        mm.on_trade(9204, 0.30, 5, aggressor_buy=False, ts=NOW + 0.1)   # arrived before our quote could be live
        self.assertEqual(mm.fills, [])
        mm.on_trade(9204, 0.30, 5, aggressor_buy=False, ts=NOW + 0.3)
        self.assertEqual(len(mm.fills), 1)
        mm.handle({"channel": "l2Book", "data": {"coin": "#92040", "time": 0,
                                                 "levels": [[{"px": "0.41", "sz": "9"}], [{"px": "0.44", "sz": "3"}]]}},
                  now=NOW + 2)
        self.assertEqual(mm.books[9204]["bid"], 0.41)


class SettlementTests(unittest.TestCase):
    def test_round_trip_and_settlement(self):
        rest = FakeRest(settled={9204: {"settleFraction": "1.0", "details": "x"}})
        mm = maker(rest=rest)
        armed(mm, fair=0.50, bid=0.40, ask=0.60)
        mm.on_trade(9204, 0.45, 20, aggressor_buy=False, ts=NOW + 1)   # buy 20 @ 0.48
        mm.on_trade(9204, 0.55, 10, aggressor_buy=True, ts=NOW + 1)    # sell 10 @ 0.52
        p = mm.pos["9204"]
        self.assertEqual(p["qty"], 10)
        self.assertAlmostEqual(mm.cash, 10_000 - 20 * 0.48 + 10 * 0.52 - 30 * 0.0005)
        mm.markets = {}
        mm._settle(MKT["expiry"] + 100)
        self.assertNotIn("9204", mm.pos)
        s = mm.settled[-1]
        self.assertAlmostEqual(s["pnl"], -9.6 + 5.2 - 0.015 + 10 * 1.0)
        self.assertAlmostEqual(mm.cash, 10_000 + s["pnl"])

    def test_unsettled_backs_off_and_void_refunds(self):
        rest = FakeRest()
        mm = maker(rest=rest)
        armed(mm, fair=0.50, bid=0.40, ask=0.60)
        mm.on_trade(9204, 0.45, 20, aggressor_buy=False, ts=NOW + 1)
        mm.markets = {}
        mm._settle(MKT["expiry"] + 100)
        self.assertIn("9204", mm.pos)
        self.assertIn("9204", mm.retry)
        mm._settle(MKT["expiry"] + 8 * 86400)                          # gave up: void
        self.assertAlmostEqual(mm.cash, 10_000 - 20 * 0.0005)          # premium back, fees paid


class RiskAndPersistenceTests(unittest.TestCase):
    def test_drawdown_kills_and_pulls_quotes(self):
        mm = maker()
        armed(mm)
        mm.cash = 9_100.0
        mm._risk(NOW)
        self.assertTrue(mm.killed)
        self.assertEqual(mm.quotes, {})
        mm.resume()
        self.assertFalse(mm.killed)
        self.assertAlmostEqual(mm.peak, 9_100.0)

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            mm = maker(Path(d))
            armed(mm, fair=0.50, bid=0.40, ask=0.60)
            mm.on_trade(9204, 0.45, 20, aggressor_buy=False, ts=NOW + 1)
            mm.kill("test")
            again = maker(Path(d))
            self.assertEqual(again.pos["9204"]["qty"], 20)
            self.assertAlmostEqual(again.cash, mm.cash)
            self.assertTrue(again.killed)
            self.assertEqual(len(again.fills), 1)
            json.dumps(again.to_dict())

    def test_step_end_to_end_with_fake_exchange(self):
        o = {"outcome": 9204, "name": "template:binaryPrice", "venue": "out",
             "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:85000|time:20261007-0000"}
        exp = 1_791_331_200   # 2026-10-07 00:00 UTC
        mm = HIP4MarketMaker(cfg(), None, spot_fn=lambda c: 85_000.0, rest=FakeRest(outcomes=[o]),
                             clock=lambda: exp - 600)
        mm.books[9204] = {"bid": 0.30, "ask": 0.70, "bid_sz": 5, "ask_sz": 5, "ts": exp - 600}
        mm.step()
        self.assertIn(9204, mm.markets)
        self.assertIn(9204, mm.quotes)
        q = mm.quotes[9204]
        self.assertLess(q["bid"], q["ask"])
        self.assertEqual(mm.status, "quoting")


class BlendTests(unittest.TestCase):
    def test_tight_market_mid_is_blended_wide_or_one_sided_is_not(self):
        mm = maker()
        mm.markets = {9204: dict(MKT)}
        mm.model.set_bars("BTC", candles(15 * 288 + 4, int(NOW) // 300 * 300))
        mm._update_fairs(NOW)
        model = mm.models[9204]
        self.assertAlmostEqual(mm.fairs[9204][0], model)                          # no book: model only
        mm.books = {9204: {"bid": 0.30, "ask": 0.34, "bid_sz": 1, "ask_sz": 1, "ts": NOW}}
        mm._update_fairs(NOW)
        self.assertAlmostEqual(mm.fairs[9204][0], 0.5 * 0.32 + 0.5 * model)
        mm.books = {9204: {"bid": 0.0003, "ask": 0.9997, "bid_sz": 1, "ask_sz": 1, "ts": NOW}}   # placeholder book
        mm._update_fairs(NOW)
        self.assertAlmostEqual(mm.fairs[9204][0], model)


EXITS = {"enabled": True, "take_profit": 0.15, "stop_loss": 0.30, "min_hold_s": 15, "cooldown_s": 60, "taker_fee": 0.001}


class LevelTwoTests(unittest.TestCase):
    def test_a_sweep_fills_level_one_then_level_two(self):
        mm = maker(levels=2)
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertAlmostEqual(q["bid2"], 0.46)                 # 2 x edge from fair
        self.assertEqual(q["bid2_sz"], 20)
        mm.on_trade(9204, 0.45, 50, aggressor_buy=False, ts=NOW + 1)   # a seller sweeps down to 45c
        self.assertEqual([(f["px"], f["qty"]) for f in mm.fills], [(0.48, 20), (0.46, 20)])
        self.assertTrue(mm.fills[1]["how"].startswith("level 2"))

    def test_a_small_trade_only_reaches_level_one(self):
        mm = maker(levels=2)
        armed(mm, fair=0.50, bid=0.30, ask=0.70)
        mm.on_trade(9204, 0.47, 50, aggressor_buy=False, ts=NOW + 1)   # through 48c, not through 46c
        self.assertEqual([(f["px"], f["qty"]) for f in mm.fills], [(0.48, 20)])


class ExitTests(unittest.TestCase):
    def long_position(self, mm, bid, ask, t_fill=NOW + 1):
        armed(mm, fair=0.50, bid=0.30, ask=0.70)
        mm.on_trade(9204, 0.45, 20, aggressor_buy=False, ts=t_fill)    # long 20 at 48c
        mm.books[9204] = {"bid": bid, "ask": ask, "bid_sz": 50, "ask_sz": 50, "ts": t_fill + 20}

    def test_take_profit_on_a_big_jump(self):
        mm = maker(exits=EXITS)
        self.long_position(mm, bid=0.66, ask=0.68)
        mm._exits(NOW + 21)
        f = mm.fills[-1]
        self.assertEqual((f["how"], f["side"], f["px"], f["qty"]), ("take-profit", -1, 0.66, 20))
        self.assertEqual(mm.pos["9204"]["qty"], 0)
        self.assertAlmostEqual(mm.pos["9204"]["cash"], 20 * (0.66 - 0.48) - 20 * 0.0005 - 20 * 0.001)
        mm._requote(NOW + 22)
        self.assertNotIn(9204, mm.quotes)                       # sitting the market out

    def test_stop_loss_and_hold_in_between(self):
        mm = maker(exits=EXITS)
        self.long_position(mm, bid=0.40, ask=0.42)                # -8c: inside both limits, hold
        mm._exits(NOW + 21)
        self.assertEqual(len(mm.fills), 1)
        mm.books[9204].update(bid=0.15, ask=0.17)                  # -33c: stop out
        mm._exits(NOW + 22)
        self.assertEqual(mm.fills[-1]["how"], "stop-loss")

    def test_no_exit_right_after_a_fill_or_into_an_empty_book(self):
        mm = maker(exits=EXITS)
        self.long_position(mm, bid=0.66, ask=0.68)
        mm._exits(NOW + 5)                                         # inside min_hold_s
        self.assertEqual(len(mm.fills), 1)
        mm.books[9204].update(bid=0.66, ask=0.9997)               # one-sided placeholder book
        mm._exits(NOW + 30)
        self.assertEqual(len(mm.fills), 1)


class GuardTests(unittest.TestCase):
    def test_no_buying_a_falling_knife(self):
        mm = maker()
        mm.pos["9204"] = {"coin": "BTC", "strike": 85_000, "expiry": MKT["expiry"], "venue": "out", "qty": 40.0,
                          "cash": -12.0, "fees": 0.0, "fills": 2, "volume": 40}
        q = armed(mm, fair=0.30, bid=0.10, ask=0.50)                # fair 30c now ...
        self.assertGreater(q["bid_sz"], 0)
        mm.fairs = {9204: (0.24, NOW + 30)}                          # ... 24c 30 s later: -6c against our long
        mm._requote(NOW + 30)
        q = mm.quotes[9204]
        self.assertEqual(q["bid_sz"], 0)                             # stop adding
        self.assertGreater(q["ask_sz"], 0)                           # still happy to sell some back
        mm.fairs = {9204: (0.24, NOW + 200)}                         # move is old news after guard_s
        mm.books[9204]["ts"] = NOW + 200
        mm._requote(NOW + 200)
        self.assertGreater(mm.quotes[9204]["bid_sz"], 0)


class SniperTests(unittest.TestCase):
    def sniper(self, **over):
        mm = maker(strategy="sniper", snipe_edge=0.05, snipe_latency_ms=300, min_notional=10, max_tte_min=60, **over)
        mm.markets = {9204: dict(MKT)}
        mm.models = {9204: 0.50}
        return mm

    def test_takes_a_mispriced_ask_after_the_latency(self):
        mm = self.sniper()
        mm.books = {9204: {"bid": 0.30, "ask": 0.42, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}
        mm._snipe(NOW)                                             # 42c ask vs 50c fair: 8c past -> signal
        self.assertEqual(mm.fills, [])                             # nothing yet: the order is in flight
        mm.books[9204]["ts"] = NOW + 0.3
        mm._snipe(NOW + 0.3)
        f = mm.fills[-1]
        self.assertEqual((f["how"], f["side"], f["px"], f["qty"]), ("snipe", 1, 0.42, 24))   # 20 -> $10 minimum = 24
        self.assertAlmostEqual(f["edge"], 0.08)
        self.assertEqual(mm.snipe_stats["filled"], 1)

    def test_misses_when_the_quote_is_gone_and_ignores_small_edges(self):
        mm = self.sniper()
        mm.books = {9204: {"bid": 0.30, "ask": 0.42, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}
        mm._snipe(NOW)
        mm.books[9204].update(ask=0.49, ts=NOW + 0.3)               # someone else took it
        mm._snipe(NOW + 0.3)
        self.assertEqual((mm.fills, mm.snipe_stats["missed"]), ([], 1))
        mm2 = self.sniper()
        mm2.books = {9204: {"bid": 0.30, "ask": 0.47, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}   # only 3c past
        mm2._snipe(NOW)
        self.assertEqual(mm2.snipe_stats["signals"], 0)

    def test_sells_a_rich_bid_and_respects_the_horizon(self):
        mm = self.sniper()
        mm.books = {9204: {"bid": 0.58, "ask": 0.70, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}
        mm._snipe(NOW)
        mm.books[9204]["ts"] = NOW + 0.3
        mm._snipe(NOW + 0.3)
        self.assertEqual((mm.fills[-1]["side"], mm.fills[-1]["px"]), (-1, 0.58))
        far = self.sniper()
        far.markets = {9204: {**MKT, "expiry": NOW + 2 * 3600}}     # 2 h out: outside the 60 min rule
        far.books = {9204: {"bid": 0.30, "ask": 0.40, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}
        far._snipe(NOW)
        self.assertEqual(far.snipe_stats["signals"], 0)

    def test_price_band_and_sides(self):
        mm = self.sniper(price_band=[0.30, 0.70])
        mm.models = {9204: 0.30}
        mm.books = {9204: {"bid": 0.10, "ask": 0.22, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}   # 22c: a long shot
        mm._snipe(NOW)
        self.assertEqual(mm.snipe_stats["signals"], 0)
        mm.models = {9204: 0.50}
        mm.books = {9204: {"bid": 0.58, "ask": 0.70, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}   # rich bid, in band
        only_buy = self.sniper(price_band=[0.30, 0.70], sides="buy")
        only_buy.books = mm.books
        only_buy._snipe(NOW)
        self.assertEqual(only_buy.snipe_stats["signals"], 0)       # a sell signal, but only buys are allowed
        mm._snipe(NOW)
        self.assertEqual(mm.snipe_stats["signals"], 1)

    def test_live_follows_fills_not_dust_signals(self):
        calls = []
        mm = self.sniper(sides="sell", max_position=500)          # $10 at 8c needs 125 contracts
        mm.on_signal = lambda *a: calls.append(a)
        mm.models = {9204: 0.85}
        # the 2026-10-07 case: a 12-contract dust bid at 92c (~$1) - the $10 minimum can't be met
        mm.books = {9204: {"bid": 0.92, "ask": 0.99, "bid_sz": 12, "ask_sz": 5, "ts": NOW}}
        mm._snipe(NOW)
        mm.books[9204]["ts"] = NOW + 0.3
        mm._snipe(NOW + 0.3)
        self.assertEqual((calls, mm.snipe_stats["skipped"]), ([], 1))
        mm._snipe(NOW + 1)                                         # and it sits the market out, no signal spam
        self.assertEqual(mm.snipe_stats["signals"], 1)
        mm.books[9204].update(bid_sz=500, ts=NOW + 10)             # real depth later
        mm._snipe(NOW + 10)
        mm.books[9204]["ts"] = NOW + 10.3
        mm._snipe(NOW + 10.3)
        self.assertEqual(len(calls), 1)                            # the pilot is called once, on the fill
        self.assertEqual(calls[0][1:3], (-1, 0.92))

    def test_fire_on_signal_sends_at_once_on_depth_never_on_dust(self):
        calls = []
        mm = self.sniper(sides="sell", max_position=500)
        mm.on_signal, mm.live_fire_on = (lambda *a: calls.append(a)), "signal"
        mm.models = {9204: 0.85}
        mm.books = {9204: {"bid": 0.92, "ask": 0.99, "bid_sz": 12, "ask_sz": 5, "ts": NOW}}   # dust: 12 x 8c = 96c
        mm._snipe(NOW)
        self.assertEqual((mm.snipe_stats["signals"], calls), (1, []))
        mm2 = self.sniper(sides="sell", max_position=500)
        mm2.on_signal, mm2.live_fire_on = (lambda *a: calls.append(a)), "signal"
        mm2.models = {9204: 0.85}
        mm2.books = {9204: {"bid": 0.92, "ask": 0.99, "bid_sz": 400, "ask_sz": 5, "ts": NOW}}  # $32 of No: real
        mm2._snipe(NOW)
        self.assertEqual([c[1:3] for c in calls], [(-1, 0.92)])          # sent at the signal, not 300 ms later
        mm2.books[9204]["ts"] = NOW + 0.3
        mm2._snipe(NOW + 0.3)
        self.assertEqual((len(mm2.fills), len(calls)), (1, 1))            # the paper fill does not send it again

    def test_live_fast_path_fires_on_the_book_update_itself(self):
        calls = []
        mm = self.sniper(sides="sell", max_position=500)
        mm.on_signal, mm.live_fire_on = (lambda *a: calls.append(a)), "signal"
        mm.models = {9204: 0.50}

        def book(bid, bid_sz):
            return {"channel": "l2Book", "data": {"coin": "#92040", "time": int(NOW * 1000) - 120,
                                                   "levels": [[{"px": str(bid), "sz": str(bid_sz)}],
                                                              [{"px": "0.70", "sz": "50"}]]}}
        mm.handle(book(0.52, 500), now=NOW)                      # 2c rich: no signal
        mm.handle(book(0.58, 10), now=NOW)                       # 8c rich but dust (10 x 42c = $4.20): no signal
        self.assertEqual(calls, [])
        mm.handle(book(0.58, 500), now=NOW + 0.01)               # 8c rich with depth: fires at once, no 0.5 s wait
        self.assertEqual(len(calls), 1)
        mid, side, limit, fair, market = calls[0]
        self.assertEqual((mid, side, limit, fair), (9204, -1, 0.58, 0.50))
        self.assertEqual((market["path"], market["detect"], market["book_et"]), ("fast", NOW + 0.01, int(NOW * 1000) - 120))
        self.assertEqual((mm.fills, mm._pending, mm.snipe_stats["signals"]), ([], {}, 0))   # paper untouched
        mm.live_fire_on = "fill"                                  # the fast path is only for fire_on=signal
        mm.handle(book(0.60, 500), now=NOW + 0.02)
        self.assertEqual(len(calls), 1)

    def test_dry_run_watch_reports_when_the_quote_goes(self):
        mm = self.sniper()
        got = []
        mm.watch_quote(9204, -1, 0.57, lambda et, local: got.append(("A", et, local)))
        mm.watch_quote(9204, -1, 0.50, lambda et, local: got.append(("B", et, local)))

        def book(bid, t):
            return {"channel": "l2Book", "data": {"coin": "#92040", "time": t,
                                                   "levels": [[{"px": str(bid), "sz": "100"}], [{"px": "0.70", "sz": "50"}]]}}
        mm.handle(book(0.58, 1000), now=NOW + 0.01)               # still >= 57c: both still there
        self.assertEqual(got, [])
        mm.handle(book(0.55, 1050), now=NOW + 0.06)               # 55c: A (needs 57c) is gone, B (50c) is not
        self.assertEqual(got, [("A", 1050.0, NOW + 0.06)])
        mm.clock = lambda: NOW + 10
        mm.expire_watches(NOW + 10)                               # B outlived the watch window: it would have filled
        self.assertEqual(got[-1], ("B", None, None))
        self.assertEqual(mm._watch[9204], [])

    def test_regime_alarm_watches_sells_against_the_model(self):
        notes = []
        mm = self.sniper(sides="sell")
        mm.notify = lambda level, text: notes.append((level, text))
        self.assertEqual(mm.regime()["status"], "collecting")
        for k in range(30):                                        # sold Yes at 60c with fair 50c, Yes won
            key = str(9000 + k)
            mm.pos[key] = {"coin": "BTC", "strike": 1, "expiry": NOW, "venue": "out", "qty": -20.0,
                           "cash": 12.0, "fees": 0.01, "fills": 1, "volume": 20}
            mm.fills.append({"id": 9000 + k, "side": -1, "px": 0.60, "qty": 20.0, "fair": 0.50})
            mm.settle_one(key, 1.0 if k % 2 else 0.0, None, NOW)
        r = mm.regime()                                            # half lost: side*(y-fair) = -50c / +50c -> 0
        self.assertEqual((r["status"], r["markets"], r["surprise_c"]), ("ok", 30, 0.0))
        for k in range(30, 60):                                    # then Yes wins every time
            key = str(9000 + k)
            mm.pos[key] = {"coin": "BTC", "strike": 1, "expiry": NOW, "venue": "out", "qty": -20.0,
                           "cash": 12.0, "fees": 0.01, "fills": 1, "volume": 20}
            mm.fills.append({"id": 9000 + k, "side": -1, "px": 0.60, "qty": 20.0, "fair": 0.50})
            mm.settle_one(key, 1.0, None, NOW)
        self.assertEqual(mm.regime()["status"], "turned")
        self.assertEqual(len(notes), 1)                            # one warning when it turns, not one per market
        self.assertIn("regime", mm.to_dict()["snipe"])

    def test_regime_alarm_ignores_a_noisy_dip(self):
        mm = self.sniper(sides="sell")
        for k in range(30):                                        # 13 won, 17 lost: -6.7c, but t ~ -0.7 (luck)
            key = str(9100 + k)
            mm.pos[key] = {"coin": "BTC", "strike": 1, "expiry": NOW, "venue": "out", "qty": -20.0,
                           "cash": 12.0, "fees": 0.01, "fills": 1, "volume": 20}
            mm.fills.append({"id": 9100 + k, "side": -1, "px": 0.60, "qty": 20.0, "fair": 0.50})
            mm.settle_one(key, 0.0 if k < 13 else 1.0, None, NOW)
        r = mm.regime()
        self.assertAlmostEqual(r["surprise_c"], -6.67, places=2)
        self.assertGreater(r["t"], -2)
        self.assertEqual(r["status"], "watch")                     # big but not proven: watch, not turned
        # size gauge: 13 x +$12 and 17 x -$8 (settled pnl as logged: cash 12 + qty*y) over 600 contracts
        self.assertEqual((r["edge_markets"], r["size_up_ok"]), (30, False))
        self.assertAlmostEqual(r["edge_c"], (13 * 12 + 17 * (12 - 20)) / 600 * 100, places=2)

    def test_no_kill_switch_when_kill_drawdown_is_zero(self):
        mm = self.sniper(kill_drawdown=0)
        mm.peak, mm.cash = 20_000.0, 10_000.0                      # a 50% drawdown
        mm._risk(NOW)
        self.assertFalse(mm.killed)
        on = self.sniper(kill_drawdown=0.08)
        on.peak, on.cash = 20_000.0, 10_000.0
        on._risk(NOW)
        self.assertTrue(on.killed)

    def test_auto_sides_follow_the_trend(self):
        bars = [Candle(NOW - 24 * 3600 + k * 300, 0, 0, 0, 86_000.0 if k < 240 else 85_000.0, 1.0) for k in range(288)]
        # 6 h ago BTC closed at 86,000; the spot is 85,000: a -1.2% downtrend -> sell Yes only
        self.assertAlmostEqual(trend_return(bars, 85_000.0, NOW, 6 * 3600), math.log(85_000 / 86_000))
        self.assertIsNone(trend_return(bars, 85_000.0, NOW, 48 * 3600))          # no bar that old
        self.assertEqual(trend_sides(-0.012, 0.003), "sell")
        self.assertEqual(trend_sides(0.004, 0.003), "buy")
        self.assertEqual(trend_sides(0.001, 0.003, "none"), "none")
        self.assertEqual(trend_sides(None, 0.003, "both"), "both")
        mm = self.sniper(sides="auto", trend={"lookback_min": 360, "threshold": 0.003, "neutral": "none"})
        mm.model.bars["BTC"] = bars
        mm.books = {9204: {"bid": 0.30, "ask": 0.42, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}   # a cheap ask: a BUY
        mm._snipe(NOW)
        self.assertEqual(mm.snipe_stats["signals"], 0)             # ... but BTC is falling: buys are off
        mm.books = {9204: {"bid": 0.58, "ask": 0.70, "bid_sz": 50, "ask_sz": 50, "ts": NOW}}   # a rich bid: a SELL
        mm._snipe(NOW)
        self.assertEqual((mm.snipe_stats["signals"], mm._pending[9204]["side"]), (1, -1))
        self.assertEqual(mm.to_dict()["snipe"]["trend"]["coins"]["BTC"]["sides"], "sell")
        flat = self.sniper(sides="auto", trend={"lookback_min": 360, "threshold": 0.003, "neutral": "none"})
        flat.model.bars["BTC"] = [Candle(b.ts, 0, 0, 0, 85_000.0, 1.0) for b in bars]
        flat.books = mm.books
        flat._snipe(NOW)
        self.assertEqual(flat.snipe_stats["signals"], 0)           # no trend, neutral "none": sit out

    def test_step_never_quotes_in_sniper_mode(self):
        mm = self.sniper()
        armed(mm)
        mm.quotes = {9204: {"bid": 0.48}}
        mm.step = HIP4MarketMaker.step.__get__(mm)
        mm.rest = FakeRest()
        mm._meta_at = NOW                                          # skip the listing refresh
        mm._bars_thread = True
        mm.step()
        self.assertEqual(mm.quotes, {})
        self.assertEqual(mm.status, "sniping")


class SizeTests(unittest.TestCase):
    def test_multiplier_scales_quotes_limits_and_risk_together(self):
        mm = maker()
        info = mm.set_size(3)
        self.assertEqual((info["quote_size"], info["max_position"], info["max_risk_usd"]), (60, 300, 3000))
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertEqual((q["bid_sz"], q["ask_sz"]), (60, 60))
        with self.assertRaises(ValueError):
            mm.set_size(6)                      # above max_size_multiplier (5)
        with self.assertRaises(ValueError):
            mm.set_size(0)
        self.assertEqual(mm.size_mult, 3)

    def test_size_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            maker(Path(d)).set_size(2)
            again = maker(Path(d))
            self.assertEqual((again.size_mult, again.quote_size, again.max_pos), (2, 40, 200))

    def test_smaller_size_stops_adding_to_a_bigger_position(self):
        mm = maker()
        mm.set_size(2)
        mm.pos["9204"] = {"coin": "BTC", "strike": 85_000, "expiry": MKT["expiry"], "venue": "out", "qty": 150.0,
                          "cash": -70.0, "fees": 0.0, "fills": 5, "volume": 150}
        mm.set_size(1)                          # limit back to 100 while we hold 150
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertEqual(q["bid_sz"], 0)
        self.assertEqual(q["ask_sz"], 20)


class ConfigTests(unittest.TestCase):
    def test_coins_must_have_a_live_price(self):
        c = cfg(coins=["BTC", "NOTACOIN"])
        with self.assertRaises(ConfigError):
            validate(c)
        validate(cfg())


if __name__ == "__main__":
    unittest.main()
