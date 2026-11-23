"""Unit tests: config, features, state, policy, risk, paper broker, calibration and the Jev client.
Run everything with:  python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.broker import PerpPaperBook  # noqa: E402
from helix.calibration import CalibrationTracker, outcome_label  # noqa: E402
from helix.config import DEFAULTS, ConfigError, load_config, validate  # noqa: E402
from helix.features import compute_features, resample, sigma_bps  # noqa: E402
from helix.jev import JevClient, build_questions, normalize_answers, setup_prob  # noqa: E402
from helix.market import Candle, PerpView, Trade  # noqa: E402
from helix.policy import flow_sign, jev_policy, reason_key, rules_policy  # noqa: E402
from helix.risk import RiskEngine  # noqa: E402
from helix.state import build_state, estimate_tokens  # noqa: E402

NOW = 1_791_200_000.0


def make_view(mid=100.0, spread=0.02, size=50.0, buy_share=0.5, coin="X", drift=0.0, levels=20) -> PerpView:
    bids = [(mid - spread / 2 - i * 0.01, size) for i in range(levels)]
    asks = [(mid + spread / 2 + i * 0.01, size) for i in range(levels)]
    mids = [(NOW - 600 + i * 0.25, mid * (1 + drift * (i - 2400) / 2400) + 0.01 * ((i % 7) - 3)) for i in range(2401)]
    trades = []
    for i in range(300):
        side = "b" if (i % 10) < buy_share * 10 else "s"
        trades.append(Trade(NOW - 300 + i, mid, 1.0, side))
    candles = [Candle(int(NOW - 7200 + i * 60), mid, mid + 0.1, mid - 0.1, mid + 0.01 * (i % 5), 10) for i in range(120)]
    ctx = {"mark": mid, "oracle": mid, "funding": 0.00001, "prev_day_px": mid * 0.98, "open_interest": 1e6}
    return PerpView(coin, bids, asks, NOW, NOW, trades, mids, candles, ctx, "sim")


class ConfigTests(unittest.TestCase):
    def test_defaults_and_shipped_file(self):
        validate(load_config(None))
        cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
        self.assertEqual(cfg["venue"], "hyperliquid")

    def test_live_mode_rejected(self):
        cfg = load_config(None)
        cfg["mode"] = "live"
        with self.assertRaises(ConfigError):
            validate(cfg)

    def test_deadline_must_fit_interval(self):
        cfg = load_config(None)
        cfg["jev"]["deadline_ms"] = 1500
        with self.assertRaises(ConfigError):
            validate(cfg)


class FeatureTests(unittest.TestCase):
    def test_resample_carries_forward_and_ignores_future(self):
        mids = [(0.0, 1.0), (2.5, 2.0), (10.0, 99.0)]
        self.assertEqual(resample(mids, 4.0, 4), [1.0, 1.0, 1.0, 2.0, 2.0])

    def test_needs_two_minutes(self):
        v = make_view()
        v.mids = v.mids[-200:]  # only 50 seconds
        self.assertIsNone(compute_features(v, NOW))

    def test_book_and_flow(self):
        v = make_view(buy_share=0.8)
        v.bids = [(99.99, 30.0)] + v.bids[1:]
        f = compute_features(v, NOW)
        self.assertAlmostEqual(f["spread_bps"], 0.02 / 100 * 1e4, places=6)
        self.assertLess(f["imbalance_5"], 0)  # thinner best bid
        self.assertAlmostEqual(f["flow_15s"], 0.8, delta=0.15)
        self.assertGreater(f["depth_10bps_usd"], 0)
        self.assertEqual(flow_sign(f), 1)

    def test_state_compact_numeric(self):
        f = compute_features(make_view(drift=0.002), NOW)
        s = build_state(f, {"side": 1, "upnl_bps": 3.2, "hold_s": 12}, {"day_pnl_pct": 0.1, "drawdown_pct": 0.3},
                        {"jev_latency_ms": 180})
        self.assertLess(estimate_tokens(s), 400)
        self.assertTrue(all(isinstance(x, (int, float)) for x in s.values()))
        self.assertGreater(s["return_300s_bps"], 0)

    def test_state_carries_scoring_band_and_long_context(self):
        f = compute_features(make_view(drift=0.002), NOW)
        s = build_state(f, {}, {}, {}, {"sigma_bps": 12.0, "band_bps": 6.0})
        self.assertEqual(s["neutral_band_bps"], 6.0)
        self.assertEqual(s["typical_move_bps"], 12.0)
        self.assertIn("return_15m_bps", s)
        self.assertIn("return_60m_bps", s)
        self.assertLess(estimate_tokens(s), 400)
        self.assertIn("neutral_band_bps", build_questions(300)["direction"]["criteria"]["neutral"])
        t = CalibrationTracker(300)
        self.assertEqual(t.threshold(12.0), 6.0)
        self.assertEqual(t.threshold(1.0), 2.0)  # floor

    def test_sigma_scales_with_horizon(self):
        f = {"vol_1s_bps": 1.0}
        self.assertAlmostEqual(sigma_bps(f, 64), 8.0)


def setup(supported: float) -> dict:
    """A score answer whose top two levels ("Supported"/"Strong") carry `supported` probability."""
    lo = 1 - supported
    return {"score": 0.0, "confidence": 0.5, "levels": 4,
            "probabilities": {0: lo * 0.5, 1: lo * 0.5, 2: supported * 0.6, 3: supported * 0.4}}


def answers(p_up=0.72, p_down=0.1, crisis=0.05, toxic=0.2, stress=0.1, long_sup=None, short_sup=None, inv=0.3):
    if long_sup is None:
        long_sup = 0.7 if p_up >= p_down else 0.1
    if short_sup is None:
        short_sup = 0.7 if p_down > p_up else 0.1
    return {
        "regime": {"choice": "trending", "p": 0.6, "probabilities": {"trending": 0.6, "crisis": crisis}, "confidence": 0.6},
        "direction": {"choice": "up" if p_up >= p_down else "down", "p": max(p_up, p_down),
                      "probabilities": {"up": p_up, "down": p_down, "neutral": max(0.0, 1 - p_up - p_down)}},
        "toxic_flow": {"p": toxic}, "liquidity_stressed": {"p": stress},
        "long_setup": setup(long_sup), "short_setup": setup(short_sup),
        "inventory_pressure": {"score": inv, "confidence": 0.6},
    }


POL = DEFAULTS["policy"]


class PolicyTests(unittest.TestCase):
    def test_full_and_half_size(self):
        self.assertEqual(jev_policy(answers(), 0, 0, POL, True).kind, "ENTER_LONG")
        self.assertEqual(jev_policy(answers(), 0, 0, POL, True).size, 1.0)
        self.assertEqual(jev_policy(answers(p_up=0.6), 0, 0, POL, True).size, 0.5)

    def test_short_when_allowed(self):
        a = answers(p_up=0.1, p_down=0.75)
        self.assertEqual(jev_policy(a, 0, 0, POL, True).kind, "ENTER_SHORT")
        self.assertEqual(jev_policy(a, 0, 0, POL, False).kind, "STAND_ASIDE")

    def test_stand_aside_paths(self):
        for kw in ({"crisis": 0.8}, {"stress": 0.9}, {"p_up": 0.4}, {"long_sup": 0.2}):
            self.assertEqual(jev_policy(answers(**kw), 0, 0, POL, True).kind, "STAND_ASIDE", kw)

    def test_setup_is_judged_per_side(self):
        # Direction says down, but only the LONG setup is supported: no short, and no long against the bias.
        a = answers(p_up=0.1, p_down=0.75, long_sup=0.9, short_sup=0.2)
        act = jev_policy(a, 0, 0, POL, True)
        self.assertEqual(act.kind, "STAND_ASIDE")
        self.assertTrue(any("SHORT setup" in r for r in act.reasons), act.reasons)

    def test_expected_edge_must_cover_cost(self):
        st = {"typical_move_bps": 16.0}
        weak = answers(p_up=0.56, p_down=0.30)   # (0.26) x 1.14 x 16 = 4.7 bps < 10
        self.assertEqual(jev_policy(weak, 0, 0, POL, True, state=st).kind, "STAND_ASIDE")
        strong = answers(p_up=0.80, p_down=0.05)  # (0.75) x 1.14 x 16 = 13.7 bps
        self.assertEqual(jev_policy(strong, 0, 0, POL, True, state=st).kind, "ENTER_LONG")
        short = answers(p_up=0.05, p_down=0.80)
        self.assertEqual(jev_policy(short, 0, 0, POL, True, state=st).kind, "ENTER_SHORT")

    def test_min_hold_blocks_signal_exits_but_not_crisis(self):
        pol = {**POL, "min_hold_s": 150}
        early, late = {"position_hold_s": 10}, {"position_hold_s": 200}
        self.assertEqual(jev_policy(answers(inv=2.8), 0, 1, pol, True, state=early).kind, "HOLD")
        self.assertEqual(jev_policy(answers(p_up=0.1, p_down=0.7), 0, 1, pol, True, state=early).kind, "HOLD")
        self.assertEqual(jev_policy(answers(inv=2.8), 0, 1, pol, True, state=late).kind, "EXIT")
        self.assertEqual(jev_policy(answers(crisis=0.8), 0, -1, pol, True, state=early).kind, "EXIT")

    def test_setup_prob_uses_level_probabilities(self):
        self.assertAlmostEqual(setup_prob({"levels": 4, "probabilities": {0: 0.05, 1: 0.15, 2: 0.45, 3: 0.35}}), 0.8)
        self.assertEqual(setup_prob(None), 0.0)

    def test_toxic_flow_against_blocks_with_allows(self):
        self.assertEqual(jev_policy(answers(toxic=0.8), -1, 0, POL, True).kind, "STAND_ASIDE")
        self.assertEqual(jev_policy(answers(toxic=0.8), 1, 0, POL, True).kind, "ENTER_LONG")

    def test_exits(self):
        self.assertEqual(jev_policy(answers(inv=2.8), 0, 1, POL, True).kind, "EXIT")
        self.assertEqual(jev_policy(answers(p_up=0.1, p_down=0.7), 0, 1, POL, True).kind, "EXIT")
        self.assertEqual(jev_policy(answers(), 0, 1, POL, True).kind, "HOLD")

    def test_rules(self):
        rp = DEFAULTS["rules"]
        f = {"ret_30s": 6.0, "imbalance_5": 0.3, "flow_15s": 0.7, "vol_ratio": 1.0}
        self.assertEqual(rules_policy(f, 0, rp, True).kind, "ENTER_LONG")
        self.assertEqual(rules_policy({**f, "ret_30s": -6.0, "imbalance_5": -0.3, "flow_15s": 0.2}, 0, rp, True).kind,
                         "ENTER_SHORT")
        self.assertEqual(rules_policy({**f, "ret_30s": -6.0}, 1, rp, True).kind, "EXIT")
        self.assertEqual(rules_policy({**f, "ret_30s": 1.0}, 0, rp, True).kind, "STAND_ASIDE")

    def test_reason_key(self):
        self.assertEqual(reason_key("Bias UP only 41% (< 55%)"), reason_key("Bias DOWN only 12% (< 55%)"))


def risk(**over):
    return RiskEngine(dict(DEFAULTS["risk"], **over), 4.5, 1.5, 0.5)


def evaluate(engine, book, view, feats, **kw):
    args = dict(book=book, view=view, feats=feats, side=1, size_mult=1.0, marks={view.coin: view.mid}, now=NOW,
                horizon_s=60, decision_latency_ms=200, errors_in_row=0, engine_paused=False)
    args.update(kw)
    return engine.evaluate_entry(**args)


FEATS = {"vol_1s_bps": 2.0, "data_age_s": 0.1}


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.book = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.5)
        self.view = make_view(size=500.0)

    def test_happy_path_sizes_by_risk_and_caps_leverage(self):
        rd = evaluate(risk(), self.book, self.view, FEATS)
        self.assertTrue(rd.approved, rd.reason)
        self.assertAlmostEqual(rd.stop_bps, 1.5 * 2.0 * 60 ** 0.5, places=4)
        self.assertLessEqual(rd.notional, 10_000 * DEFAULTS["risk"]["max_position_leverage"] + 1e-6)
        self.assertLess(rd.stop, self.view.mid)
        self.assertGreater(rd.target, self.view.mid)

    def test_sigma_floor_from_the_signal(self):
        base = evaluate(risk(), self.book, self.view, FEATS)                      # book sigma 2 x sqrt(60) = 15.5 bps
        wide = evaluate(risk(), self.book, self.view, FEATS, sigma_floor_bps=40.0)
        self.assertAlmostEqual(wide.stop_bps, 1.5 * 40.0)                         # candle-based sigma is larger: used
        self.assertAlmostEqual(wide.target_bps, DEFAULTS["risk"]["target_sigma"] * 40.0)
        self.assertLess(wide.notional, base.notional)                              # same $ risk, wider stop: smaller bet
        for low in (5.0, None, float("nan"), 0.0):                                 # a floor never lowers sigma
            self.assertAlmostEqual(evaluate(risk(), self.book, self.view, FEATS, sigma_floor_bps=low).stop_bps,
                                   base.stop_bps)

    def test_fee_gate(self):
        rd = evaluate(risk(), self.book, self.view, {"vol_1s_bps": 0.2, "data_age_s": 0.1})
        self.assertFalse(rd.approved)
        self.assertIn("Fees vs target", rd.reason)

    def test_blocks(self):
        self.assertFalse(evaluate(risk(), self.book, self.view, FEATS, engine_paused=True).approved)
        self.assertFalse(evaluate(risk(), self.book, self.view, {"vol_1s_bps": 2.0, "data_age_s": 9}).approved)
        self.assertFalse(evaluate(risk(), self.book, self.view, FEATS, decision_latency_ms=5000).approved)
        self.assertFalse(evaluate(risk(allow_short=False), self.book, self.view, FEATS, side=-1).approved)
        thin = make_view(size=0.01)
        rd = evaluate(risk(), self.book, thin, FEATS)
        self.assertFalse(rd.approved)
        self.assertIn("Book depth", rd.reason)

    def test_entry_rate_limit(self):
        for i in range(DEFAULTS["risk"]["max_entries_per_min"]):
            self.book.entry_times.append(NOW - 1)
        rd = evaluate(risk(), self.book, self.view, FEATS)
        self.assertFalse(rd.approved)
        self.assertIn("Entry rate", rd.reason)

    def test_bet_multiplier_scales_risk_and_caps_within_ceilings(self):
        wide = {"vol_1s_bps": 4.0, "data_age_s": 0.1}              # wide stop: 1x stays under the leverage cap
        big = make_view(size=5000.0)
        one = evaluate(risk(), self.book, big, wide)
        two = evaluate(risk(), self.book, big, wide, bet_mult=2.0)
        self.assertTrue(one.approved and two.approved, (one.reason, two.reason))
        self.assertLess(one.notional, 10_000 * DEFAULTS["risk"]["max_position_leverage"])
        self.assertAlmostEqual(two.notional, 2 * one.notional, delta=1e-6)
        self.assertTrue(any(c.name == "Bet size" for c in two.checks))
        huge = evaluate(risk(), self.book, big, wide, bet_mult=50.0)     # capped by the position ceiling (5x equity)
        self.assertLessEqual(huge.notional, 10_000 * DEFAULTS["risk"]["leverage_ceiling_position"] + 1e-6)
        self.assertGreater(huge.notional, 10_000 * DEFAULTS["risk"]["max_position_leverage"])
        half = evaluate(risk(), self.book, big, wide, bet_mult=0.5)
        self.assertAlmostEqual(half.notional, one.notional / 2, delta=1e-6)
        tight = {"vol_1s_bps": 0.6, "data_age_s": 0.1}             # tight stop: the leverage cap binds at 1x
        capped = evaluate(risk(min_edge_multiple=0), self.book, big, tight)
        self.assertAlmostEqual(capped.notional, 10_000 * DEFAULTS["risk"]["max_position_leverage"], delta=1e-6)
        halved = evaluate(risk(min_edge_multiple=0), self.book, big, tight, bet_mult=0.5)
        self.assertAlmostEqual(halved.notional, capped.notional / 2, delta=1e-6)   # 0.5x still halves the bet

    def test_no_room_left_is_reported_as_leverage(self):
        bk = PerpPaperBook("t", 10_000, 0, 0, 0)
        v = make_view(size=10_000)
        bk.open(v, 1, 350, 1, 1000, NOW, "")                       # ~3.5x already committed
        rd = evaluate(risk(), bk, make_view(size=5000.0, coin="Y"), {"vol_1s_bps": 4.0, "data_age_s": 0.1},
                      marks={"X": v.mid, "Y": v.mid})
        self.assertFalse(rd.approved)
        self.assertIn("Leverage", rd.reason)

    def test_guards(self):
        eng = risk()
        bk = PerpPaperBook("t", 10_000, 0, 0, 0)
        v = make_view(size=10_000)
        bk.open(v, 1, 100, 1, 1000, NOW)
        bk.mark({"X": v.mid}, NOW)
        self.assertEqual(eng.guard(bk, {"X": v.mid}), (None, None))
        self.assertEqual(eng.guard(bk, {"X": v.mid - 2.5})[0], "daily")
        self.assertEqual(eng.guard(bk, {"X": v.mid - 9})[0], "kill")


class BrokerTests(unittest.TestCase):
    def test_round_trip_accounting(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        pos = bk.open(v, 1, 10, 99, 101, NOW)
        self.assertAlmostEqual(pos.entry, v.best_ask)
        t = bk.close_market(v, NOW + 5, "test")
        expected = 10 * (v.best_bid - v.best_ask) - 10 * v.best_ask * 4.5e-4 - 10 * v.best_bid * 4.5e-4
        self.assertAlmostEqual(t["pnl"], expected, places=6)
        self.assertAlmostEqual(bk.cash, 10_000 + t["pnl"], places=6)

    def test_book_walk_slippage(self):
        bk = PerpPaperBook("t", 10_000, 0, 0, 0)
        v = make_view(size=1.0)
        pos = bk.open(v, 1, 5, 90, 110, NOW)
        self.assertGreater(pos.entry, v.best_ask)  # walked five levels
        self.assertGreater(pos.entry_slip_bps, 0)
        self.assertIsNone(bk.open(make_view(size=0.1, coin="Y"), 1, 50, 90, 110, NOW))  # too thin: no fill

    def test_take_profit_is_maker_stop_is_market(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        bk.open(v, 1, 1, 99.0, 100.5, NOW)
        up = make_view(mid=100.6, size=1000)
        t = bk.manage(up, NOW + 10, 999)
        self.assertEqual((t["reason"], t["exit"], t["maker_exit"]), ("Take profit", 100.5, True))
        bk.open(v, -1, 1, 100.4, 99.0, NOW)
        t = bk.manage(make_view(mid=100.5, size=1000), NOW + 20, 999)
        self.assertEqual(t["reason"], "Stop loss")
        self.assertFalse(t["maker_exit"])

    def test_max_hold_and_funding(self):
        bk = PerpPaperBook("t", 10_000, 0, 0, 0)
        v = make_view(size=1000)
        start = 3600 * 1000 + 10.0
        bk.open(v, 1, 10, 50, 150, start)
        paid = bk.apply_funding("X", 100.0, 0.0001, start + 3600)
        self.assertAlmostEqual(paid, 10 * 100 * 0.0001)  # long pays when funding is positive
        self.assertEqual(bk.apply_funding("X", 100.0, 0.0001, start + 3601), 0.0)
        t = bk.manage(v, start + 4000, 300)
        self.assertTrue(t["reason"].startswith("Max hold"))
        self.assertAlmostEqual(t["funding"], 0.1)

    def test_maker_entry_fills_only_on_trade_through(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        o = bk.place_entry(v, 1, 2.0, 20.0, 40.0, NOW, ttl_s=60, reason="test")
        self.assertEqual(o["price"], v.best_bid)
        self.assertIsNone(bk.place_entry(v, 1, 2.0, 20.0, 40.0, NOW, 60))           # one order per coin
        self.assertEqual(bk.check_entry(v, NOW + 1)[0], "resting")                    # touch unchanged: no fill
        touch = make_view(mid=v.best_bid + 0.01, size=1000)                         # ask lands ON our bid: still no fill
        self.assertGreaterEqual(touch.best_ask, o["price"])
        self.assertEqual(bk.check_entry(touch, NOW + 2)[0], "resting")
        state, pos = bk.check_entry(make_view(mid=v.best_bid - 0.05, size=1000), NOW + 3)  # traded through
        self.assertEqual(state, "filled")
        self.assertEqual(pos.entry, o["price"])
        self.assertAlmostEqual(pos.entry_fee, 2.0 * o["price"] * 1.5e-4)              # maker fee, no slippage
        self.assertAlmostEqual(pos.stop, o["price"] * (1 - 20e-4))
        self.assertAlmostEqual(pos.target, o["price"] * (1 + 40e-4))
        self.assertEqual(bk.pending, {})

    def test_maker_entry_expires_and_short_side(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        bk.place_entry(v, -1, 1.0, 20.0, 40.0, NOW, ttl_s=30)
        self.assertEqual(bk.check_entry(make_view(mid=v.best_ask - 0.005, size=1000), NOW + 5)[0], "resting")
        self.assertEqual(bk.check_entry(v, NOW + 31)[0], "expired")
        self.assertEqual(bk.maker_stats, {"placed": 1, "filled": 0, "expired": 1})
        bk.place_entry(v, -1, 1.0, 20.0, 40.0, NOW + 40, ttl_s=30)
        self.assertEqual(bk.check_entry(make_view(mid=v.best_ask + 0.05, size=1000), NOW + 45)[0], "filled")
        again = PerpPaperBook.from_dict(bk.to_dict())                               # persists with its counters
        self.assertEqual(again.maker_stats["filled"], 1)

    def test_maker_exit_near_the_horizon(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        bk.open(v, 1, 2.0, 90.0, 120.0, NOW)                          # long, far stop and target
        self.assertIsNone(bk.manage(v, NOW + 700, 900, maker_exit=True))      # before the 120 s window: nothing
        self.assertEqual(bk.positions["X"].exit_px, 0.0)
        self.assertIsNone(bk.manage(v, NOW + 790, 900, maker_exit=True))      # window opens: offer at the best ask
        pos = bk.positions["X"]
        self.assertEqual(pos.exit_px, v.best_ask)
        touch = make_view(mid=v.best_ask - 0.01, size=1000)                  # bid only reaches our offer: no fill
        self.assertIsNone(bk.manage(touch, NOW + 795, 900, maker_exit=True))
        lower = make_view(mid=v.mid - 0.2, size=1000)                         # price fell: re-priced after 15 s
        bk.manage(lower, NOW + 806, 900, maker_exit=True)
        self.assertEqual(bk.positions["X"].exit_px, lower.best_ask)
        through = make_view(mid=lower.best_ask + 0.05, size=1000)            # bid trades through our offer
        t = bk.manage(through, NOW + 810, 900, maker_exit=True)
        self.assertTrue(t["maker_exit"])
        self.assertEqual(t["exit"], lower.best_ask)
        self.assertIn("Maker exit", t["reason"])
        bk.open(v, -1, 2.0, 110.0, 80.0, NOW)                                 # short never traded through: taker at max hold
        bk.manage(v, NOW + 800, 900, maker_exit=True)
        t = bk.manage(v, NOW + 900, 900, maker_exit=True)
        self.assertFalse(t["maker_exit"])
        self.assertTrue(t["reason"].startswith("Max hold"))
        bk.open(v, 1, 2.0, 99.99, 120.0, NOW)                                 # stops are never delayed
        t = bk.manage(make_view(mid=99.9, size=1000), NOW + 850, 900, maker_exit=True)
        self.assertEqual(t["reason"], "Stop loss")

    def test_stale_maker_exit_is_void(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        bk.open(v, 1, 2.0, 90.0, 120.0, NOW)
        bk.manage(v, NOW + 790, 900, maker_exit=True)                        # offer rests at the best ask
        old = bk.positions["X"].exit_px
        through = make_view(mid=old + 0.5, size=1000)                         # a minute later, no calls in between
        self.assertIsNone(bk.manage(through, NOW + 850, 900, maker_exit=True))   # not filled at the stale price
        self.assertEqual(bk.positions["X"].exit_px, through.best_ask)       # re-placed at today's touch

    def test_clock_runs_from_the_signal_minute(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        sig_minute = NOW - 45                                               # the fill comes 45 s after the signal
        bk.place_entry(v, 1, 2.0, 900.0, 900.0, NOW - 5, ttl_s=60, clock_start=sig_minute)
        self.assertEqual(bk.check_entry(make_view(mid=v.best_bid - 0.05, size=1000), NOW)[0], "filled")
        self.assertEqual(bk.positions["X"].clock_start, sig_minute)
        self.assertIsNone(bk.manage(v, NOW + 730, 900, maker_exit=True))  # age 775 s: window not open yet
        self.assertEqual(bk.positions["X"].exit_px, 0.0)
        bk.manage(v, NOW + 736, 900, maker_exit=True)                      # age 781 s: maker offer rests
        self.assertEqual(bk.positions["X"].exit_px, v.best_ask)
        for k in range(1, 8):                                               # re-priced every 15 s, never traded through
            self.assertIsNone(bk.manage(v, NOW + 736 + 15 * k, 900, maker_exit=True))
        t = bk.manage(v, NOW + 855, 900, maker_exit=True)                  # 900 s after the SIGNAL, 855 s after the fill
        self.assertTrue(t["reason"].startswith("Max hold"))
        self.assertAlmostEqual(t["hold_s"], 855)

    def test_expired_order_never_fills_late(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        bk.place_entry(v, 1, 2.0, 20.0, 40.0, NOW, ttl_s=60, clock_start=NOW - 10)
        through = make_view(mid=v.best_bid - 0.05, size=1000)
        self.assertEqual(bk.check_entry(through, NOW + 1200)[0], "expired")   # e.g. the first tick after a restart
        self.assertNotIn("X", bk.positions)
        self.assertEqual(bk.maker_stats, {"placed": 1, "filled": 0, "expired": 1})

    def test_positions_from_a_newer_version_still_load(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        bk.open(make_view(size=1000), 1, 1.0, 90.0, 120.0, NOW)
        d = bk.to_dict()
        d["positions"]["X"]["some_future_field"] = 1
        self.assertIn("X", PerpPaperBook.from_dict(d).positions)

    def test_saved_positions_without_a_clock_still_load(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        bk.open(make_view(size=1000), 1, 1.0, 90.0, 120.0, NOW)
        d = bk.to_dict()
        for p in d["positions"].values():                                   # a state.json from before this change
            p.pop("clock_start")
        again = PerpPaperBook.from_dict(d)
        self.assertEqual(again.positions["X"].age(NOW + 100), 100)          # falls back to the fill time

    def test_future_clock_start_is_clamped_and_taker_opens_carry_it(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        pos = bk.open(make_view(size=1000), 1, 1.0, 90.0, 120.0, NOW, clock_start=NOW + 60)
        self.assertEqual(pos.clock_start, NOW)                              # never a clock that starts in the future
        self.assertEqual(pos.age(NOW + 10), 10)

    def test_take_profit_needs_trade_through(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.0)
        v = make_view(size=1000)
        touch = make_view(mid=100.51, size=1000)
        bk.open(v, 1, 1, 99.0, touch.best_bid, NOW)                         # target exactly at the touch's best bid
        self.assertIsNone(bk.manage(touch, NOW + 10, 999))                  # a touch alone doesn't fill a resting limit
        t = bk.manage(make_view(mid=100.53, size=1000), NOW + 11, 999)     # bid above it: filled at the target
        self.assertEqual((t["reason"], t["exit"]), ("Take profit", touch.best_bid))

    def test_exit_breakdown_separates_move_from_fees(self):
        from helix.broker import exit_breakdown
        mk = lambda reason, gross, fees: {"reason": reason, "qty": 1.0, "entry": 100.0, "gross": gross,  # noqa: E731
                                          "fees": fees, "pnl": gross - fees, "pnl_bps": (gross - fees) * 100}
        rows = exit_breakdown([mk("Max hold 900s", -0.1, 0.06), mk("Maker exit at horizon (790s)", 0.02, 0.03),
                               mk("Maker exit at horizon (785s)", -0.02, 0.03), mk("Take profit", 0.6, 0.03),
                               mk("Signal exit: Crisis regime 55%", -0.05, 0.06)])
        by = {r["kind"]: r for r in rows}
        self.assertEqual([r["kind"] for r in rows], ["time_maker", "time_market", "target", "other"])
        self.assertEqual(by["time_maker"]["n"], 2)
        self.assertAlmostEqual(by["time_maker"]["gross_bps"], 0.0)
        self.assertAlmostEqual(by["time_maker"]["fees_bps"], 3.0)
        self.assertAlmostEqual(by["time_market"]["net_bps"], -16.0)
        self.assertEqual(by["target"]["wins"], 1)

    def test_persistence(self):
        bk = PerpPaperBook("t", 10_000, 4.5, 1.5, 0.5)
        v = make_view(size=1000)
        bk.open(v, 1, 3, 99, 101, NOW)
        bk.mark({"X": 100.2}, NOW)
        again = PerpPaperBook.from_dict(bk.to_dict())
        self.assertAlmostEqual(again.equity({"X": 100.2}), bk.equity({"X": 100.2}))
        self.assertEqual(again.entries_last(NOW), 1)


class CalibrationTests(unittest.TestCase):
    def test_outcomes_and_report(self):
        self.assertEqual(outcome_label(5, 2), "up")
        self.assertEqual(outcome_label(-5, 2), "down")
        self.assertEqual(outcome_label(1, 2), "neutral")
        t = CalibrationTracker(60, keep=3)
        for i in range(5):
            t.add("X", i, 100.0, "up", {"up": 0.7, "down": 0.1, "neutral": 0.2}, 4.0)
        t.resolve(30, {"X": 101})
        self.assertEqual(t.report()["n"], 0)
        t.resolve(100, {"X": 101})
        rep = t.report()
        self.assertEqual(rep["n"], 3)
        self.assertEqual(rep["accuracy"], 1.0)
        self.assertEqual(sum(b["n"] for b in rep["bins"]), 3)


class FakeResponse:
    def __init__(self, status, payload, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.headers, self.calls = response, exc, {}, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response


API_ANSWERS = {
    "regime": {"type": "choice", "choice": "trending", "confidence": 0.6,
               "probabilities": {"trending": 0.6, "mean_reverting": 0.2, "high_vol": 0.15, "crisis": 0.05}},
    "direction": {"type": "choice", "choice": "up", "confidence": 0.5, "probabilities": {"up": 0.6, "down": 0.1, "neutral": 0.3}},
    "toxic_flow": {"type": "noul", "noul": 0.83},
    "liquidity_stressed": {"type": "noul", "noul": 0.1},
    "long_setup": {"type": "score", "score": 2.1, "confidence": 0.55,
                   "probabilities": {"0": 0.05, "1": 0.15, "2": 0.45, "3": 0.35}, "legend": {"0": "Against"}},
    "short_setup": {"type": "score", "score": 0.5, "confidence": 0.6,
                    "probabilities": {"0": 0.6, "1": 0.3, "2": 0.08, "3": 0.02}, "legend": {"0": "Against"}},
    "inventory_pressure": {"type": "score", "score": 0.4, "confidence": 0.7, "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0}},
}


class JevTests(unittest.TestCase):
    def test_battery_matches_the_guide(self):
        import json
        q = build_questions(60)
        self.assertEqual(set(q), {"regime", "direction", "toxic_flow", "liquidity_stressed", "long_setup",
                                  "short_setup", "inventory_pressure"})
        self.assertEqual(len(q["inventory_pressure"]["criteria"]), 4)
        self.assertIn("60 seconds", q["direction"]["instructions"])
        self.assertIn("`neutral_band_bps`", q["direction"]["criteria"]["neutral"])
        wire = json.loads(json.dumps(q))  # noul criteria must reach the API as "true"/"false" keys
        self.assertEqual(set(wire["toxic_flow"]["criteria"]), {"true", "false"})

    def test_normalize(self):
        a = normalize_answers(API_ANSWERS, build_questions(60))
        self.assertAlmostEqual(a["direction"]["p"], 0.6)
        self.assertAlmostEqual(a["toxic_flow"]["p"], 0.83)
        self.assertAlmostEqual(a["long_setup"]["probabilities"][2], 0.45)
        self.assertAlmostEqual(setup_prob(a["long_setup"]), 0.8)
        self.assertAlmostEqual(setup_prob(a["short_setup"]), 0.1)

    def test_request_shape_and_statuses(self):
        import requests
        ok = FakeSession(FakeResponse(200, {"model": "jev-1.13.0", "answers": API_ANSWERS, "usage": {"input_tokens": 512}}))
        c = JevClient("k", "jev-1.13.0", 800, base_url="https://api.typesafe.ai", http=ok)
        res = c.ask({"spread_bps": 0.1}, build_questions(60))
        self.assertEqual(res.status, "ok")
        call = ok.calls[0]
        self.assertEqual(call["url"], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(call["headers"]["Authorization"], "Bearer k")
        self.assertEqual(call["json"]["model"], "jev-1.13.0")
        self.assertAlmostEqual(call["timeout"], 0.8)
        bad_model = FakeSession(FakeResponse(200, {"model": "jev-2.0.0", "answers": API_ANSWERS}))
        self.assertEqual(JevClient("k", "jev-1.13.0", 800, http=bad_model).ask({}, build_questions(60)).status, "error")
        alias = FakeSession(FakeResponse(200, {"model": "jev-2.0.0", "answers": API_ANSWERS}))
        self.assertEqual(JevClient("k", "jev-latest", 800, http=alias).ask({}, build_questions(60)).status, "ok")
        slow = FakeSession(exc=requests.Timeout())
        self.assertEqual(JevClient("k", "jev-1.13.0", 800, http=slow).ask({}, build_questions(60)).status, "late")
        denied = FakeSession(FakeResponse(429, {}, "rate limited"))
        res = JevClient("k", "jev-1.13.0", 800, http=denied).ask({}, build_questions(60))
        self.assertEqual(res.status, "error")
        self.assertIn("429", res.error)


if __name__ == "__main__":
    unittest.main()
