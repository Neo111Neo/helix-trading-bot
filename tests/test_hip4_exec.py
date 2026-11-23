"""Live pilot executor (against a FAKE exchange - no test can reach Hyperliquid): it refuses to send unless every
guard passes, maps signals to the right HIP-4 orders, stays inside the pilot limits, settles positions correctly,
stops itself at the daily loss limit, and never writes the private key anywhere."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import DEFAULTS  # noqa: E402
from helix.hip4_exec import LiveExecutor  # noqa: E402

NOW = 1_791_300_000.0
MKT = {"id": 9204, "coin": "BTC", "strike": 85_000.0, "expiry": NOW + 600}
KEY = "0x" + "ab" * 32
ENV = {"HL_ACCOUNT_ADDRESS": "0x" + "11" * 20, "HL_AGENT_PRIVATE_KEY": KEY}


class FakeExchange:
    def __init__(self, fill=True):
        self.sent, self.fill = [], fill

    def ioc(self, coin, is_buy, sz, px):
        self.sent.append((coin, is_buy, sz, px))
        st = {"filled": {"totalSz": str(sz), "avgPx": str(px), "oid": 1}} if self.fill else {"error": "no match"}
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [st]}}}


class Gate:
    def __init__(self, ready=True):
        self.ready = ready

    def evaluate(self):
        return {"ready": self.ready}


class Sniper:
    killed = False
    rest = None
    on_signal = on_step = None


def pilot(d=None, armed=True, ready=True, env=ENV, ex=None, **pilot_over):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["hip4_live"]["live_enabled"] = armed
    cfg["hip4_live"]["pilot"].update(pilot_over)
    return LiveExecutor(cfg, Sniper(), Gate(ready), d, exchange=ex if ex is not None else FakeExchange(),
                        env=env, clock=lambda: NOW)


class ArmingTests(unittest.TestCase):
    def test_every_guard_must_pass(self):
        self.assertIn("hip4_live.live_enabled is false in config.yaml", pilot(armed=False).blockers())
        self.assertIn("the go-live gate is not ready", pilot(ready=False).blockers())
        self.assertIn("HL_ACCOUNT_ADDRESS / HL_AGENT_PRIVATE_KEY are not set in .env", pilot(env={}).blockers())
        p = pilot()
        self.assertEqual(p.blockers(), [])
        p.sniper.killed = True
        self.assertIn("kill switch engaged", p.blockers())

    def test_operator_can_override_only_the_gate(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["hip4_live"].update(live_enabled=True, require_gate=False)
        ex = FakeExchange()
        p = LiveExecutor(cfg, Sniper(), Gate(False), None, exchange=ex, env=ENV, clock=lambda: NOW)
        self.assertEqual(p.blockers(), [])
        rec = p._send(9204, -1, 0.58, 0.50, MKT)
        self.assertTrue(rec["gate_overridden"])
        self.assertTrue(p.to_dict()["gate_overridden"])
        p_env = LiveExecutor(cfg, Sniper(), Gate(False), None, exchange=ex, env={}, clock=lambda: NOW)
        self.assertIn("HL_ACCOUNT_ADDRESS / HL_AGENT_PRIVATE_KEY are not set in .env", p_env.blockers())   # still guarded
        big = LiveExecutor(cfg, Sniper(), Gate(False), None, exchange=FakeExchange(), env=ENV, clock=lambda: NOW)
        big.contracts = 100
        self.assertIsNone(big._send(9204, 1, 0.42, 0.50, MKT))   # limits still apply

    def test_nothing_is_sent_while_blocked(self):
        ex = FakeExchange()
        p = pilot(ready=False, ex=ex)
        self.assertIsNone(p._send(9204, 1, 0.42, 0.50, MKT))
        self.assertEqual(ex.sent, [])
        self.assertIn("the go-live gate is not ready", p.last_refusal["why"])


class OrderTests(unittest.TestCase):
    def test_buy_yes_at_the_minimum(self):
        ex = FakeExchange()
        rec = pilot(ex=ex)._send(9204, 1, 0.42, 0.50, MKT)
        self.assertEqual(ex.sent, [("#92040", True, 24, 0.42)])        # ceil($10 / 42c) = 24 contracts
        self.assertEqual((rec["status"], rec["filled"]), ("filled", 24.0))

    def test_sell_without_yes_is_a_no_buy(self):
        ex = FakeExchange()
        pilot(ex=ex)._send(9204, -1, 0.58, 0.50, MKT)
        self.assertEqual(ex.sent, [("#92041", True, 24, 0.42)])        # buy No at 1 - 58c

    def test_limits(self):
        ex = FakeExchange()
        p = pilot(ex=ex, max_order_usd=11, max_open_usd=21, capital_usd=150)
        self.assertIsNotNone(p._send(9204, 1, 0.42, 0.50, MKT))         # $10.08
        self.assertIsNotNone(p._send(9205, 1, 0.42, 0.50, {**MKT, "id": 9205}))   # $20.16 open
        self.assertIsNone(p._send(9206, 1, 0.42, 0.50, {**MKT, "id": 9206}))      # would be $30.24 > $21
        self.assertEqual(len(ex.sent), 2)
        big = pilot(ex=FakeExchange(), contracts=100, max_order_usd=25)
        self.assertIsNone(big._send(9204, 1, 0.42, 0.50, MKT))         # 100 x 42c = $42 > $25


class PerMarketCapTests(unittest.TestCase):
    def test_one_market_cannot_soak_up_the_open_limit(self):
        ex = FakeExchange()
        p = pilot(ex=ex, max_market_usd=30, max_open_usd=100)
        for _ in range(4):                                                # 24 No at 42c = $10.08 each
            p._send(9204, -1, 0.58, 0.50, MKT)
        self.assertEqual(len(ex.sent), 2)                                  # a third would make $30.24 > $30
        self.assertIn("max_market_usd", p.last_refusal["why"][0])
        self.assertIsNotNone(p._send(9205, -1, 0.58, 0.50, {**MKT, "id": 9205}))   # other markets still have room
        self.assertEqual(p.set_size(2)["max_market_usd"], 60)              # scales with the live size
        self.assertIsNotNone(p._send(9204, -1, 0.58, 0.50, MKT))


class LongDatedBudgetTests(unittest.TestCase):
    def test_long_dated_markets_cannot_crowd_out_short_ones(self):
        ex = FakeExchange()
        p = pilot(ex=ex, max_market_usd=30, max_open_usd=100, long_tte_min=60, long_budget_usd=30)
        far = {**MKT, "expiry": NOW + 3 * 3600}
        for i in range(4):                                                 # $10.08 each, 3 h out
            p._send(9300 + i, -1, 0.58, 0.50, {**far, "id": 9300 + i})
        self.assertEqual(len(ex.sent), 2)                                   # a third would make $30.24 > $30
        self.assertIn("long_budget_usd", p.last_refusal["why"][0])
        for i in range(5):                                                  # 10-minute markets: the rest of the $100
            self.assertIsNotNone(p._send(9400 + i, -1, 0.58, 0.50, {**MKT, "id": 9400 + i}))
        self.assertEqual(p.to_dict()["long_cost"], 20.16)
        p.clock = lambda: NOW + 2.5 * 3600                                  # 30 min left: now a short-dated market
        self.assertEqual(p.long_cost(p.clock()), 0)
        self.assertEqual(p.set_size(2)["long_budget_usd"], 60)


class CompoundTests(unittest.TestCase):
    def test_limits_follow_the_account_up_and_down(self):
        class Wallet(FakeExchange):
            usdc = 400.0

            def balances(self):
                return {"USDC": self.usdc}
        ex = Wallet()
        comp = {"enabled": True, "open": 0.65, "order": 0.03, "market": 0.15, "long": 0.15, "daily_loss": 0.15,
                "max_open_frac": 0.9}
        p = pilot(ex=ex, compound=comp)
        self.assertEqual(p.size_info()["max_open_usd"], 100)              # account unknown yet: the fixed limits ...
        self.assertIsNone(p._send(9204, 1, 0.42, 0.50, MKT))              # ... so NO order until it has been read
        self.assertEqual(ex.sent, [])
        self.assertIn("account", p.last_refusal["why"][0])
        p._refresh_equity()                                                # $400 account
        z = p.size_info()
        self.assertEqual((z["max_open_usd"], z["max_market_usd"], z["daily_loss_usd"], z["order_usd"]), (260, 60, 60, 12))
        p._send(9204, 1, 0.42, 0.50, MKT)
        self.assertEqual(ex.sent[-1][2], 29)                               # 24 contracts x $12/$10 -> 29 (~$12)
        ex.usdc, p._wallet["ts"] = 100.0, 0.0                              # the account shrinks to ~$112
        p._refresh_equity()
        z = p.size_info()
        self.assertEqual((z["order_usd"], z["max_open_usd"]), (10, round(0.65 * (100 + 29 * 0.42), 2)))   # never < $10
        self.assertEqual(p.set_size(2)["order_usd"], 20)                   # the multiplier still works on top
        fixed = pilot(ex=Wallet())
        fixed._refresh_equity()
        self.assertEqual(fixed.size_info()["max_open_usd"], 100)           # compounding off: config limits


class CompoundFloorTests(unittest.TestCase):
    def test_floor_keeps_room_for_more_than_one_order_on_a_small_account(self):
        class Wallet(FakeExchange):
            def balances(self):
                return {"USDC": 120.0}
        comp = {"enabled": True, "open": 0.53, "order": 0.02, "market": 0.16, "long": 0.21, "daily_loss": 0,
                "max_open_frac": 0.9}
        off = pilot(ex=Wallet(), compound=comp)
        off._refresh_equity()
        self.assertEqual(off.size_info()["max_market_usd"], 19.2)         # 16% of $120: ONE $10 order per market
        on = pilot(ex=Wallet(), compound={**comp, "floor_to_fixed": True})
        on._refresh_equity()
        z = on.size_info()
        self.assertEqual((z["max_market_usd"], z["max_open_usd"]), (30, 100))   # never below the fixed config
        on.set_size(2)
        self.assertEqual(on.size_info()["max_open_usd"], 108)              # still capped at 90% of the account


class LiveValueTests(unittest.TestCase):
    def test_a_won_settlement_never_shows_as_a_drop(self):
        class Account(FakeExchange):                       # what Hyperliquid reports: USDC + outcome tokens, one read
            def __init__(self):
                super().__init__()
                self.usdc, self.outcomes = 100.0, {}

            def snapshot(self):
                return {"usdc": self.usdc, "outcomes": dict(self.outcomes)}
        ex = Account()
        p = pilot(ex=ex)
        p.sniper.books = {9204: {"bid": 0.30, "ask": 0.40}}           # Yes mid 35c -> No worth 65c
        p._send(9204, -1, 0.58, 0.50, MKT)                            # 24 No for $10.08
        ex.usdc, ex.outcomes = 100.0 - 10.08, {92041: (24.0, 10.08)}
        p._wallet["ts"] = 0.0
        before = p.to_dict()["live_equity"]
        self.assertAlmostEqual(before, 89.92 + 24 * 0.65, places=2)
        # the market settles No on the exchange: the tokens are gone and $24 is in the wallet - but the pilot has not
        # polled settledOutcome yet and still holds the position. The old page counted it twice here, then "dropped"
        ex.usdc, ex.outcomes = 89.92 + 24.0, {}
        p._wallet["ts"] = 0.0
        p.sniper.books = {}
        mid_window = p.to_dict()["live_equity"]
        self.assertAlmostEqual(mid_window, 113.92, places=2)          # once, not 113.92 + 10.08
        p.settle(9204, 0.0)                                           # the pilot records it: +$13.92
        after = p.to_dict()["live_equity"]
        self.assertAlmostEqual(after, mid_window, places=2)           # no drop when the win is recorded
        self.assertAlmostEqual(p.realized_total, 24 - 10.08, places=2)

    def test_settlement_polling_never_books_an_early_or_odd_reply(self):
        class Rest:
            def __init__(self):
                self.replies, self.calls = [], 0

            def post(self, body):
                self.calls += 1
                return self.replies.pop(0) if self.replies else None
        p = pilot()
        p.sniper.rest = Rest()
        p._send(9204, -1, 0.58, 0.50, MKT)                              # 24 No, expires NOW + 600
        t = MKT["expiry"] + 61
        p.poll(t)
        self.assertEqual(p.sniper.rest.calls, 0)                       # first ask only at expiry + 120 s
        p.sniper.rest.replies = [{"status": "pending"}, {"settleFraction": None}]
        p.poll(MKT["expiry"] + 121)                                    # not settled yet
        p.poll(MKT["expiry"] + 121.5)                                  # half a second later: no new call
        self.assertEqual(p.sniper.rest.calls, 1)
        p.poll(MKT["expiry"] + 152)                                    # backed off 30 s: asks again, gets null
        self.assertIn(9204, p.pos)                                      # neither reply was booked as a void
        self.assertEqual(p.realized_total, 0)
        p.sniper.rest.replies = [{"settleFraction": "0.0"}]
        p.poll(MKT["expiry"] + 300)                                    # next back-off step (60 s): the real result
        self.assertNotIn(9204, p.pos)
        self.assertAlmostEqual(p.realized_total, 24 - 10.08, places=2)

    def test_last_good_mark_near_expiry_and_reads_in_order(self):
        class Account(FakeExchange):
            def __init__(self):
                super().__init__()
                self.usdc, self.outcomes = 89.92, {92041: (24.0, 10.08)}

            def snapshot(self):
                return {"usdc": self.usdc, "outcomes": dict(self.outcomes)}
        ex = Account()
        p = pilot(ex=ex)
        p.sniper.books = {9204: {"bid": 0.02, "ask": 0.04}}            # Yes 3c: the No side is worth 97c
        live = p.to_dict()["live_equity"]
        self.assertAlmostEqual(live, 89.92 + 24 * 0.97, places=2)
        p.sniper.books = {9204: {"bid": 0.0003, "ask": 0.99}}          # one-sided near expiry: keep the last good mark
        self.assertAlmostEqual(p.to_dict()["live_equity"], live, places=2)
        p.sniper.books = {}                                            # delisted: still the last good mark, not cost
        self.assertAlmostEqual(p.to_dict()["live_equity"], live, places=2)
        # a newer read (started at NOW + 5, after the settlement) is stored; then a slow read that STARTED earlier (at
        # NOW, before the settlement) finishes: it must not overwrite the newer one
        p._wallet = {**p._wallet, "started": NOW + 5, "ts": 0.0, "usdc": 113.92, "outcomes": {}}
        p.account()                                                    # reads at NOW (< NOW + 5) -> discarded
        self.assertEqual((p._wallet["usdc"], p._wallet["outcomes"]), (113.92, {}))

    def test_no_daily_stop_when_it_is_zero(self):
        p = pilot(daily_loss_usd=0)
        for i in range(5):
            p._send(9300 + i, 1, 0.50, 0.60, {**MKT, "id": 9300 + i})
            p.settle(9300 + i, 0.0)                                   # -$50 in a day
        self.assertFalse(p.halted)
        self.assertEqual(p.blockers(), [])


class DryRunTests(unittest.TestCase):
    def dry(self, **over):
        class WatchingSniper(Sniper):
            def __init__(self):
                self.watches, self.books = [], {}

            def watch_quote(self, mid, side, yes_limit, cb):
                self.watches.append((mid, side, yes_limit, cb))

            def expire_watches(self, now):
                pass
        ex = FakeExchange()
        cfg = copy.deepcopy(DEFAULTS)
        cfg["hip4_live"].update(live_enabled=True, require_gate=False)
        cfg["hip4_live"]["pilot"].update({"dry_run": True, "slip": 0.01, "min_edge": 0.03, **over})   # live's settings
        p = LiveExecutor(cfg, WatchingSniper(), Gate(True), None, exchange=ex, env=ENV, clock=lambda: NOW)
        return p, ex

    def test_dry_run_never_sends_and_scores_the_race(self):
        p, ex = self.dry()
        mkt = {**MKT, "detect": NOW, "book_et": NOW * 1000 - 40, "path": "fast"}
        o1 = p._send(9204, -1, 0.58, 0.50, mkt)                          # No at 42c (+1c slip: 43c), Yes limit 57c
        o2 = p._send(9205, -1, 0.58, 0.50, {**mkt, "id": 9205})
        o3 = p._send(9206, -1, 0.58, 0.50, {**mkt, "id": 9206})
        self.assertEqual(ex.sent, [])                                     # NOTHING reached the exchange
        self.assertEqual(p.to_dict()["mode"], "DRY RUN")
        self.assertEqual([w[2] for w in p.sniper.watches], [0.57, 0.57, 0.57])
        arrive = NOW * 1000 + 0 + p.DRY_NET_MS                            # detection + overhead (0 here) + network
        p.sniper.watches[0][3](arrive + 50, NOW + 0.05)                   # gone 50 ms after we'd have arrived
        p.sniper.watches[1][3](arrive - 10, NOW)                          # gone before we'd have arrived
        p.sniper.watches[2][3](None, None)                                # still there 5 s later
        self.assertEqual(o1["fills"], {"0": True, "100": False, "300": False})
        self.assertEqual(o2["fills"], {"0": False, "100": False, "300": False})
        self.assertEqual(o3["fills"], {"0": True, "100": True, "300": True})
        self.assertEqual(o1["reaction_ms"], 40.0)
        d = p.dry_summary()
        self.assertEqual((d["sent"], d["margins"]["0"]["filled"], d["margins"]["100"]["filled"]), (3, 2, 1))

    def test_dry_positions_settle_into_net_dollars(self):
        p, ex = self.dry()

        class Rest:
            def post(self, body):
                return {"settleFraction": "0.0" if body["outcome"] == 9204 else "1.0"}
        p.sniper.rest = Rest()
        for mid in (9204, 9205):
            p._send(mid, -1, 0.58, 0.50, {**MKT, "id": mid, "detect": NOW, "book_et": NOW * 1000})
        for w in p.sniper.watches:
            w[3](None, None)                                              # both would have filled
        p.poll(MKT["expiry"] + 121)
        d = p.dry_summary()["margins"]["100"]                             # 24 No at 43c = $10.32 each
        self.assertEqual((d["settled_markets"], d["won"]), (2, 1))
        # won: 24 x (1 - 0.43); lost: -24 x 0.43; fee 0.0005 per contract on both
        self.assertAlmostEqual(d["pnl"], round(24 * (1 - 0.43) - 24 * 0.43 - 2 * 24 * 0.0005, 2), places=2)
        self.assertEqual(len(p.dry["settled"]), 2)
        self.assertEqual(ex.sent, [])

    def test_shadow_scores_alongside_real_orders(self):
        p, ex = self.dry(dry_run=False, dry_shadow=True, max_open_usd=15, capital_usd=15)
        self.assertEqual(p.to_dict()["mode"], "LIVE")
        a = p._send(9204, -1, 0.58, 0.50, {**MKT, "detect": NOW, "book_et": NOW * 1000})
        b = p._send(9205, -1, 0.58, 0.50, {**MKT, "id": 9205, "detect": NOW, "book_et": NOW * 1000})
        self.assertEqual(a["status"], "filled")                          # the REAL order went out ...
        self.assertIsNone(b)                                             # ... the 2nd is over the $15 real cap ...
        self.assertEqual(len(ex.sent), 1)
        self.assertEqual(p.dry_summary()["sent"], 2)                     # ... but the shadow scored BOTH
        self.assertTrue(p.dry_summary()["alongside_live"])

    def test_dry_run_keeps_the_live_caps(self):
        p, ex = self.dry()
        for _ in range(4):                                                # $10.32 each, $30 per market at 1x
            if p._send(9204, -1, 0.58, 0.50, {**MKT, "detect": NOW, "book_et": NOW * 1000}):
                p.sniper.watches[-1][3](None, None)
        d = p.dry_summary()
        self.assertEqual((d["sent"], d["refused"]), (2, 2))               # a 3rd would make $30.96 > $30
        self.assertEqual(ex.sent, [])


class SpeedTests(unittest.TestCase):
    def test_slip_pays_up_to_fair_minus_min_edge_never_more(self):
        ex = FakeExchange()
        p = pilot(ex=ex, slip=0.02, min_edge=0.03)
        p._send(9204, 1, 0.42, 0.50, MKT)                                 # ask 42c, fair 50c: limit 44c (< 47c)
        p._send(9205, -1, 0.58, 0.50, {**MKT, "id": 9205})                # bid 58c = No 42c, No fair 50c: 44c
        p._send(9206, 1, 0.45, 0.50, {**MKT, "id": 9206})                 # only 5c of edge: capped at 47c
        self.assertEqual([s[3] for s in ex.sent], [0.44, 0.44, 0.47])
        self.assertEqual(ex.sent[0][2], 23)                               # sized at the worst price: ceil($10/44c)
        self.assertEqual(pilot(ex=FakeExchange())._worst_buy(0.42, 0.50), 0.42)    # slip 0: exactly the quote

    def test_sdk_client_is_warmed_up_once_in_the_background(self):
        import time as _t
        import helix.hip4_exec as hx
        built = []

        class FakeSDK:
            def __init__(self, network, account, key):
                built.append(network)
                _t.sleep(0.05)                                          # the real one downloads market tables
        orig = hx.SDKExchange
        hx.SDKExchange = FakeSDK
        try:
            cfg = copy.deepcopy(DEFAULTS)
            cfg["hip4_live"].update(live_enabled=True, require_gate=False)
            p = LiveExecutor(cfg, Sniper(), Gate(True), None, exchange=None, env=ENV, clock=lambda: NOW)
            if p.blockers():                                            # the SDK is not installed here: nothing to warm
                self.skipTest("hyperliquid SDK not installed")
            p.poll(NOW)
            p.poll(NOW + 0.5)                                           # a second step while warming: no second build
            for _ in range(100):
                if p._exchange is not None and not p._warming:
                    break
                _t.sleep(0.01)
            self.assertIsInstance(p._exchange, FakeSDK)
            p.poll(NOW + 1)
            self.assertEqual(built, ["mainnet"])                        # built exactly once
            off = LiveExecutor(copy.deepcopy(DEFAULTS), Sniper(), Gate(True), None, exchange=None, env=ENV,
                               clock=lambda: NOW)
            off.poll(NOW)
            self.assertIsNone(off._exchange)                            # not armed: never built
        finally:
            hx.SDKExchange = orig

    def test_latency_is_logged_on_every_order(self):
        p = pilot()
        rec = p._send(9204, -1, 0.58, 0.50, {**MKT, "detect": NOW - 0.25, "book_et": (NOW - 0.4) * 1000, "path": "fast"})
        self.assertEqual((rec["path"], rec["detect_to_send_ms"], rec["book_to_send_ms"]), ("fast", 250, 400))
        self.assertIn("order_rtt_ms", rec)
        plain = p._send(9205, -1, 0.58, 0.50, {**MKT, "id": 9205})             # older callers: no latency fields
        self.assertNotIn("detect_to_send_ms", plain)

    def test_fire_on_is_handed_to_the_sniper(self):
        self.assertEqual(pilot(fire_on="signal").sniper.live_fire_on, "signal")
        self.assertEqual(pilot().sniper.live_fire_on, "fill")


class SettlementTests(unittest.TestCase):
    def test_settles_both_legs_and_stops_at_the_daily_loss(self):
        p = pilot(daily_loss_usd=15)
        p._send(9204, 1, 0.42, 0.50, MKT)                                # 24 Yes for $10.08
        p._send(9205, -1, 0.58, 0.50, {**MKT, "id": 9205})               # 24 No for $10.08
        p.settle(9204, 1.0)                                              # Yes paid: +$13.92
        self.assertAlmostEqual(p.realized_today, 24 - 10.08)
        p.settle(9205, 1.0)                                              # No lost: -$10.08
        self.assertFalse(p.halted)
        p._send(9206, 1, 0.50, 0.60, {**MKT, "id": 9206})                # 20 Yes for $10
        p._send(9207, 1, 0.50, 0.60, {**MKT, "id": 9207})
        p.settle(9206, 0.0)
        p.settle(9207, 0.0)                                              # -$20 more: -$16.16 today
        self.assertTrue(p.halted)
        self.assertIn("daily loss limit", p.halt_reason)
        self.assertIn("pilot halted", " ".join(p.blockers()))


class LiveSizeTests(unittest.TestCase):
    def test_multiplier_scales_orders_and_limits_inside_the_capital_cap(self):
        ex = FakeExchange()
        p = pilot(ex=ex)
        info = p.set_size(3)
        self.assertEqual((info["max_order_usd"], info["max_open_usd"], info["daily_loss_usd"]), (75, 150, 180))  # open capped at $150
        p._send(9204, 1, 0.42, 0.50, MKT)
        self.assertEqual(ex.sent[-1], ("#92040", True, 72, 0.42))     # 3 x 24 contracts (~$30)
        with self.assertRaises(ValueError):
            p.set_size(6)
        with self.assertRaises(ValueError):
            p.set_size(0.5)                                              # never below the exchange minimum

    def test_size_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            pilot(Path(d)).set_size(2)
            self.assertEqual(pilot(Path(d)).size_mult, 2)


class SecretTests(unittest.TestCase):
    def test_the_key_is_never_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = pilot(Path(d))
            p._send(9204, 1, 0.42, 0.50, MKT)
            p.settle(9204, 1.0)
            dump = json.dumps(p.to_dict(), default=str)
            for f in Path(d).iterdir():
                dump += f.read_text(encoding="utf-8")
            self.assertNotIn(KEY, dump)
            self.assertNotIn(KEY[2:], dump)


if __name__ == "__main__":
    unittest.main()
