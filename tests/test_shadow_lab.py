"""Shadow lab: the zero-risk tournament scores each variant like the live pilot would trade it, settles at the real
outcome, learns a bias only from an unconditional checkpoint sample (and only when it is clear), and the autopilot
hands live to a variant only on real evidence - a paper-betting score that stays honest however often it is checked,
over enough outcomes and days, in rising AND falling markets, positive on the latest hours. With NO edge anywhere (or
an edge that is only the market's drift) it must almost never qualify, even checked after every outcome."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.shadow_lab import VARIANTS, ShadowLab, _blank_ep, _log_wealth  # noqa: E402

NOW = 1_791_900_000.0


class Model:
    def fair(self, *a, **k):
        return None                               # fall back to the sniper's models[] in these tests


class Rest:
    def __init__(self):
        self.y = {}

    def post(self, body):
        y = self.y.get(body["outcome"])
        return None if y is None else {"settleFraction": str(y)}


class Sniper:
    def __init__(self):
        self.markets, self.books, self.models = {}, {}, {}
        self.min_p, self.model, self.rest = 0.03, Model(), Rest()
        self.band_lo, self.band_hi, self.stop_s, self.max_tte_s = 0.0, 1.0, 180.0, 1440 * 60.0
        self.book_hooks = []
        self.spot_fn = lambda coin: 85_000.0


def update(L, sn, mid, t, book=None):
    """The next Yes-book update of `mid` arrives at local time t (the same book unless one is given)."""
    if book is not None:
        sn.books[mid] = book
    L.on_book(mid, t)


def market(mid, tte_min):
    return {"id": mid, "coin": "BTC", "strike": 85_000.0, "expiry": NOW + tte_min * 60, "twap_s": 60}


def lab(**kw):
    sn = Sniper()
    return ShadowLab(sn, None, clock=lambda: NOW, **kw), sn


SELL = {"bid": 0.58, "ask": 0.70, "bid_sz": 100, "ask_sz": 100}      # rich bid vs a 50c fair: a sell
BUY = {"bid": 0.30, "ask": 0.42, "bid_sz": 100, "ask_sz": 100}       # cheap ask vs a 50c fair: a buy


class TournamentTests(unittest.TestCase):
    def test_variants_signal_by_side_and_horizon(self):
        L, sn = lab()
        sn.markets, sn.models, sn.books = {1: market(1, 10), 2: market(2, 300)}, {1: 0.5, 2: 0.5}, {1: SELL, 2: BUY}
        self.assertEqual(sn.book_hooks, [L.on_book])
        L.on_book(1, NOW)
        L.on_book(2, NOW)
        sent = {n: x["sent"] for n, x in L.led.items()}
        self.assertEqual(sent, {"live_rules": 1, "sell_last15": 1, "both_last15": 1, "both_last15_7c": 1,
                                "both_24h": 2, "selfcal_24h": 2, "selfcal_60": 1})
        L.on_book(1, NOW + 1)                                                          # 15 s cooldown per market
        self.assertEqual(L.led["live_rules"]["sent"], 1)

    def test_fills_are_booked_at_the_quote_and_settle_net(self):
        L, sn = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}})
        sn.markets, sn.models, sn.books = {1: market(1, 10), 2: market(2, 10)}, {1: 0.5, 2: 0.5}, {1: SELL, 2: SELL}
        L.on_book(1, NOW)
        L.on_book(2, NOW)
        self.assertEqual(L._checks[1][0]["o"]["yes_limit"], 0.57)                     # Yes limit = 1 - (42c + 1c)
        v = L.led["v"]
        self.assertEqual((v["resolved"], v["pending"].get(1, 0) > 0), (0, True))      # in flight until checked
        update(L, sn, 1, NOW + 1)                                                      # still there: filled
        update(L, sn, 2, NOW + 1, {**SELL, "bid": 0.55})                               # below the 57c limit: missed
        self.assertEqual((v["filled"], v["resolved"], v["pending"]), (1, 2, {}))
        sz = math.ceil(10 / 0.43)                                                      # sized at the 43c limit: 24
        self.assertAlmostEqual(v["pos"][1]["no"]["cost"], sz * 0.42)                  # booked at the 42c that was there
        sn.rest.y = {1: 0.0}
        L.poll(NOW + 10 * 60 + 121)
        self.assertAlmostEqual(v["realized"], sz * 1.0 - sz * 0.42 - sz * L.FEE, places=4)
        self.assertEqual((v["markets"], v["won"], len(v["settled"])), (1, 1, 1))

    def test_caps_count_both_legs_and_in_flight_orders(self):
        L, sn = lab(variants={"v": {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}})
        sn.markets, sn.models = {1: market(1, 10)}, {1: 0.5}
        et0 = NOW * 1000 - 280                                                         # exchange time of the snapshot
        sn.books = {1: {**SELL, "et": et0}}
        L.on_book(1, NOW)                                                              # $10.08, still in flight
        sn.books = {1: {**BUY, "et": et0}}                                             # an old snapshot: no check yet
        L.on_book(1, NOW + 20)                                                         # the other side, same market
        L.on_book(1, NOW + 40)                                                         # a 3rd: over $30 incl. pending
        self.assertEqual((L.led["v"]["sent"], L.led["v"]["refused"]), (2, 1))
        both = {"bid": 0.58, "ask": 0.43, "bid_sz": 100, "ask_sz": 100, "et": (NOW + 41) * 1000}   # shows both quotes
        update(L, sn, 1, NOW + 41.3, both)
        self.assertEqual(set(L.led["v"]["pos"][1]), {"no", "yes"})                    # one market, two legs
        sn.rest.y = {1: 1.0}
        L.poll(NOW + 10 * 60 + 121)
        self.assertEqual(len(L.led["v"]["settled"]), 1)                              # ONE outcome for the market

    def test_long_dated_budget(self):
        L, sn = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False}})
        for i in range(6):                                                             # 5 h out: long-dated
            sn.markets[i], sn.models[i], sn.books[i] = market(i, 300), 0.5, SELL
            L.on_book(i, NOW)
        self.assertEqual((L.led["v"]["sent"], L.led["v"]["refused"]), (3, 3))         # $40 / $10.08 = 3


class FillRuleTests(unittest.TestCase):
    """Rule "et300" (2026-10-22): a would-be IOC fills only if the FIRST Yes-book update whose exchange time is at least
    300 ms after the send still shows our price - for at most the size shown, at the price shown. On 387 real live
    IOCs this matched reality 91% of the time; the old 5 s watch matched 32% (it booked 92-97% as filled)."""
    V = {"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}}
    ET0 = NOW * 1000 - 280                                                             # the signal's snapshot

    def setup(self, n=1):
        L, sn = lab(variants=self.V)
        for i in range(1, n + 1):
            sn.markets[i], sn.models[i], sn.books[i] = market(i, 10), 0.5, {**SELL, "et": self.ET0}
            L.on_book(i, NOW)
        return L, sn

    def test_only_the_first_update_300ms_after_the_send_decides(self):
        L, sn = self.setup(2)
        due = NOW * 1000 + L.OVERHEAD_MS + L.FILL_DELAY_MS
        gone = {**SELL, "bid": 0.50}
        update(L, sn, 1, NOW + 0.2, {**gone, "et": due - 1})                          # too early: does not count
        self.assertEqual(L.led["v"]["resolved"], 0)
        update(L, sn, 1, NOW + 0.6, {**SELL, "et": due})                              # the first one at +300 ms
        update(L, sn, 2, NOW + 0.6, {**gone, "et": due})                              # gone there: a miss ...
        update(L, sn, 2, NOW + 6.0, {**SELL, "et": due + 5400})                       # ... even if it comes back
        v = L.led["v"]
        self.assertEqual((v["filled"], v["resolved"], set(v["pos"]), v["pending"]), (1, 2, {1}, {}))

    def test_fill_is_capped_by_the_size_shown_and_booked_at_the_price_shown(self):
        L, sn = self.setup(2)
        sz = math.ceil(10 / 0.43)                                                      # 24 contracts wanted
        update(L, sn, 1, NOW + 1, {**SELL, "bid_sz": 10, "et": (NOW + 1) * 1000})     # only 10 left at 58c
        update(L, sn, 2, NOW + 1, {**SELL, "bid": 0.60, "et": (NOW + 1) * 1000})      # a better bid than seen
        v = L.led["v"]
        self.assertEqual((v["pos"][1]["no"]["contracts"], v["partial"]), (10, 1))
        self.assertAlmostEqual(v["pos"][1]["no"]["cost"], 10 * 0.42)
        self.assertEqual(v["pos"][2]["no"]["contracts"], sz)
        self.assertAlmostEqual(v["pos"][2]["no"]["cost"], sz * 0.40)                  # an IOC fills at the resting price
        self.assertEqual(v["pending"], {})

    def test_an_ioc_sweeps_the_levels_inside_its_limit(self):
        """Review 2026-10-22: capping at the top level dropped the deeper (worse) contracts a real IOC also takes."""
        L, sn = self.setup()
        sz = math.ceil(10 / 0.43)                                                      # 24 wanted, Yes limit 57c
        deep = {**SELL, "bid": 0.60, "bid_sz": 10, "et": (NOW + 5) * 1000,
                "bid_lv": [(0.60, 10.0), (0.58, 20.0), (0.56, 50.0)]}                  # 56c is through the limit
        update(L, sn, 1, NOW + 5.3, deep)
        leg = L.led["v"]["pos"][1]["no"]
        self.assertEqual(leg["contracts"], sz)                                         # 10 at 60c + 14 at 58c
        self.assertAlmostEqual(leg["cost"], 10 * 0.40 + (sz - 10) * 0.42)
        self.assertEqual(L.led["v"]["partial"], 0)

    def test_caps_are_reserved_at_the_limit_so_fills_never_overshoot(self):
        """Review 2026-10-22: reserved at the seen price but booked up to the limit, 2c asks overshot the $100 budget by
        40%. The live pilot reserves sz x limit; so does the lab now."""
        L, sn = lab(variants={"b": {"sides": "buy", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False}})
        for i in range(1, 21):
            sn.markets[i], sn.models[i] = market(i, 60), 0.08
            sn.books[i] = {"bid": 0.01, "ask": 0.02, "bid_sz": 2000, "ask_sz": 2000, "et": NOW * 1000}
            L.on_book(i, NOW)
        for i in range(1, 21):                                                         # each fills AT the 3c limit
            update(L, sn, i, NOW + 5.4, {**sn.books[i], "ask": 0.03, "et": (NOW + 5.4) * 1000})
        open_cost = sum(sum(x["cost"] for x in p.values()) for p in L.led["b"]["pos"].values())
        self.assertLessEqual(open_cost, L.BUDGET + 1e-9)
        self.assertEqual(L.led["b"]["sent"], 9)                                         # 9 x $10.02 at the limit

    def test_a_missed_order_does_not_label_or_date_the_market(self):
        """Review 2026-10-22: a miss set the variant's first-bet time, so a later independent cluster was skipped."""
        L, sn = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False}})
        sn.markets, sn.models, sn.books = {1: market(1, 60)}, {1: 0.5}, {1: {**SELL, "et": self.ET0}}
        L.on_book(1, NOW)
        update(L, sn, 1, NOW + 5.4, {**SELL, "bid": 0.50, "et": self.ET0 + 5400})    # missed
        self.assertNotIn(1, L.led["v"]["first"])
        update(L, sn, 1, NOW + 40, {**SELL, "et": (NOW + 40) * 1000})                 # a new signal ...
        update(L, sn, 1, NOW + 45.4, {**SELL, "et": (NOW + 45.4) * 1000})             # ... that fills
        self.assertEqual(L.led["v"]["first"][1][0], NOW + 40)

    def test_meta_of_markets_that_only_missed_is_dropped_after_expiry(self):
        L, sn = self.setup()
        update(L, sn, 1, NOW + 5.4, {**SELL, "bid": 0.50, "et": self.ET0 + 5400})    # missed: nothing held
        L.ckpt.pop(1, None)
        self.assertIn(1, L.meta)
        L.poll(NOW + 10 * 60 + 121)
        self.assertNotIn(1, L.meta)

    def test_no_qualifying_update_is_a_miss_and_frees_the_budget(self):
        L, sn = self.setup()
        L.poll(NOW + L.CHECK_MAX_S - 1)
        self.assertEqual(L.led["v"]["resolved"], 0)
        L.poll(NOW + L.CHECK_MAX_S + 1)                                                # nothing showed the quote
        v = L.led["v"]
        self.assertEqual((v["filled"], v["resolved"], v["pos"], v["pending"], L._checks), (0, 1, {}, {}, {}))

    def test_without_exchange_time_the_local_clock_stands_in(self):
        L, sn = lab(variants=self.V)
        sn.markets, sn.models, sn.books = {1: market(1, 10)}, {1: 0.5}, {1: SELL}       # no "et" in these books
        L.on_book(1, NOW)
        update(L, sn, 1, NOW + 0.2)                                                    # < 305 ms after detection
        self.assertEqual(L.led["v"]["resolved"], 0)
        update(L, sn, 1, NOW + 0.31)
        self.assertEqual(L.led["v"]["filled"], 1)

    def test_board_and_log_say_which_rule_scored_the_fills(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sn = Sniper()
            L = ShadowLab(sn, Path(d), variants=self.V, clock=lambda: NOW)
            sn.markets, sn.models, sn.books = {1: market(1, 10)}, {1: 0.5}, {1: {**SELL, "et": self.ET0}}
            L.on_book(1, NOW)
            update(L, sn, 1, NOW + 5.4, {**SELL, "bid": 0.50, "et": self.ET0 + 5400})
            rec = [json.loads(x) for x in (Path(d) / "shadow_lab.jsonl").read_text().splitlines()][-1]
            self.assertEqual((rec["event"], rec["rule"], rec["why"], rec["got"]), ("miss", "et300", "gone", 0.0))
            self.assertEqual((rec["check_px"], rec["check_et"]), (0.50, self.ET0 + 5400))
            self.assertEqual(L.to_dict()["fill_rule"]["name"], "et300")
            self.assertEqual(L.to_dict()["board"][0]["fill_rate"], 0.0)

    def test_a_new_fill_rule_restarts_evidence_once_without_new_slots(self):
        import json
        import tempfile
        a = {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}
        with tempfile.TemporaryDirectory() as d:
            L1 = ShadowLab(Sniper(), Path(d), variants={"a": a, "b": {**a, "sides": "both"}}, clock=lambda: NOW,
                           season_size=7)
            for _ in range(5):
                L1._ep_update(L1.led["a"]["ep"], 0.3, 0.01)                            # a looked good ...
            L1.led["a"]["settled"].append([NOW - 50, 7, NOW - 100, 9.0, 10.0, 0.01, NOW - 400])
            L1.led["a"]["born"] = NOW - 20 * 86400
            L1._save()
            p = Path(d) / "shadow_lab.json"
            st = json.loads(p.read_text())
            st.pop("fill_rule")                                                        # ... under the old 5 s watch
            p.write_text(json.dumps(st))
            later = NOW + 3600
            L2 = ShadowLab(Sniper(), Path(d), variants={"a": a, "b": {**a, "sides": "both"}}, clock=lambda: later,
                           season_size=7)
            A = L2.led["a"]
            self.assertEqual((A["ep"]["n"], A["settled"], A["born"]), (0, [], later))  # from 1x, fresh age clock
            self.assertEqual((len(L2.tried), L2.bar(), L2.season["started"]), (2, 140, st["season"]["started"]))
            self.assertEqual(L2.fill_rule_since, later)
            bak = json.loads((Path(d) / "shadow_lab.pre-et300.json").read_text())       # the old state, untouched
            self.assertEqual(bak["led"]["a"]["ep"]["n"], 5)
            self.assertEqual(L2.archive_rules["watch"]["archive"][f"a:{A['params']}"]["ep"]["n"], 5)   # kept ...
            L2._close_clusters(later)
            self.assertEqual(A["ep"]["n"], 0)                                          # ... never scored again
            L2._save()
            L3 = ShadowLab(Sniper(), Path(d), variants={"a": a}, clock=lambda: later + 60, season_size=7)
            self.assertEqual((L3.led["a"]["ep"]["n"], L3.fill_rule_since), (0, later))  # same rule: no 2nd restart
            self.assertEqual(L3.archive_rules["watch"]["archive"][f"a:{A['params']}"]["ep"]["n"], 5)


class CalibrationTests(unittest.TestCase):
    def setup_obs(self, L, sn, yes_rate, clusters=20, per=3):
        k = 0
        for c in range(clusters):
            for j in range(per):
                k += 1
                sn.markets[k] = {**market(k, 12), "expiry": NOW + 12 * 60 + c}         # one expiry per cluster
                sn.models[k], sn.books[k] = 0.50, {"bid": 0.49, "ask": 0.51, "bid_sz": 100, "ask_sz": 100}
                L.on_book(k, NOW)
                sn.rest.y[k] = 1.0 if (c * per + j) % 10 < yes_rate * 10 else 0.0
        L.poll(NOW + 12 * 60 + 200)

    def test_clear_drift_shifts_the_fair_noise_does_not(self):
        L, sn = lab(variants={"sc": {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": True}})
        self.assertEqual(L.bias(), 0.0)
        self.setup_obs(L, sn, 0.8)                                                     # Yes won 80% at fair 50%
        self.assertAlmostEqual(L.bias(), 0.10, places=6)                               # clear: +30c, clipped to 10c
        self.assertAlmostEqual(L._shift(0.5, 0.10), 0.5987, places=3)                  # log-odds: ~ +10c at 50c ...
        self.assertLess(L._shift(0.05, 0.10) - 0.05, 0.03)                             # ... much less in the tails
        sn.markets[999], sn.models[999], sn.books[999] = market(999, 30), 0.5, SELL
        L.on_book(999, NOW + 800)
        self.assertEqual(L.led["sc"]["sent"], 0)                                       # 58c is no longer rich
        L2, sn2 = lab()
        self.setup_obs(L2, sn2, 0.5)                                                   # no drift: no shift
        self.assertEqual(L2.bias(), 0.0)


def per_budget(r_per_dollar, stake=25.0):
    """A cluster's score input: its net P&L over the lab's fixed $100 budget, for a typical $25 at stake."""
    return r_per_dollar * stake / ShadowLab.BUDGET


class AutopilotTests(unittest.TestCase):
    def feed(self, L, name, rets, trends, span_d=7.5, cost=30.0, now=NOW, first_before_s=60.0):
        """One settled market per expiry cluster, oldest first over the last `span_d` days: rets[i] is its return
        per $ staked, trends[i] the coin's 6 h trend before the bet (first bet `first_before_s` before expiry). Then
        score the clusters as poll() does."""
        n = len(rets)
        for i, (r, tr) in enumerate(zip(rets, trends)):
            ts = now - span_d * 86400 * (n - i) / n
            exp = ts - 200
            L.led[name]["settled"].append([ts, 10_000 + i, exp, r * cost, cost, tr, exp - first_before_s])
        L.led[name]["born"] = now - span_d * 86400 - 3600
        L._close_clusters(now)

    @staticmethod
    def row(L, ep, confirm=True):
        """The score row _qualifies sees, from a bare score state (the recent-hours check passed or not)."""
        lw, lu, ld = _log_wealth(ep["lw"]), _log_wealth(ep["up"]), _log_wealth(ep["dn"])
        return {"score": math.exp(min(lw, 50)), "log_score": lw, "clusters": ep["n"], "age_d": 30.0,
                "up": ep["n_up"], "down": ep["n_dn"], "log_up": lu, "log_down": ld,
                "confirm_clusters": 10 if confirm else 0, "net_confirm": 1.0 if confirm else 0.0}

    def ever_qualifies(self, L, outcomes):
        """Feed (r, trend) pairs one at a time; True if the gate would pass after ANY of them."""
        lb, ep = math.log(L.bar()), _blank_ep()
        for r, tr in outcomes:
            L._ep_update(ep, r, tr)
            if ep["n"] >= L.MIN_CLUSTERS and _log_wealth(ep["lw"]) >= lb and L._qualifies(self.row(L, ep))[0]:
                return True
        return False

    def test_sits_out_without_evidence(self):
        L, sn = lab(autopilot=True)
        self.assertIsNone(L.select(NOW))
        self.assertIn("sits out", L.selection["why"])
        self.assertAlmostEqual(L.bar(), 15 / 0.05)                                     # a season of 15 at 5%: 300x

    def test_picks_strong_two_way_evidence(self):
        L, sn = lab(autopilot=True)
        good = [0.3, 0.25, -0.2, 0.35, 0.2, 0.3] * 70                                 # 420 outcomes, mean +0.2 per $
        trend = [0.01, -0.01] * 210                                                    # half rising, half falling
        self.feed(L, "both_last15", good, trend)
        self.assertEqual(L.select(NOW), "both_last15")
        s = next(s for s in L.selection["scores"] if s["name"] == "both_last15")
        self.assertEqual((s["clusters"], s["up"], s["down"], s["skipped"]), (420, 210, 210, 0))
        self.assertGreater(s["score"], L.bar())

    def test_refuses_one_way_young_thin_or_late_losing_evidence(self):
        trend = [0.01, -0.01] * 210
        # wins only in falling markets (the sell-only trap): a huge overall score, still refused
        L, _ = lab(autopilot=True)
        self.feed(L, "live_rules", [0.8 if t < 0 else -0.3 for t in trend], trend)
        self.assertIsNone(L.select(NOW))
        s = next(s for s in L.selection["scores"] if s["name"] == "live_rules")
        self.assertGreater(s["score"], L.bar())
        self.assertIn("rising or in falling", s["why"])
        good = [0.3, 0.25, -0.2, 0.35, 0.2, 0.3] * 70
        # only falling markets seen: not enough rising-market outcomes
        L, _ = lab(autopilot=True)
        self.feed(L, "both_last15", good, [-0.01] * 420)
        self.assertIsNone(L.select(NOW))
        # too young (2 days)
        L, _ = lab(autopilot=True)
        self.feed(L, "both_last15", good, trend, span_d=2.0)
        self.assertIsNone(L.select(NOW))
        # losing on the last 24 h
        L, _ = lab(autopilot=True)
        self.feed(L, "both_24h", [r if i < 380 else -0.5 for i, r in enumerate(good)], trend)
        self.assertIsNone(L.select(NOW))
        self.assertIn("last 24 h", next(s for s in L.selection["scores"] if s["name"] == "both_24h")["why"])

    def test_null_checked_after_every_outcome_rarely_qualifies(self):
        """No variant has an edge (fair-priced bets at 50-90c), 7 variants, 30 days x 48 outcomes, the gate checked
        after EVERY outcome (more often than the hourly autopilot): any variant ever qualifying must stay < 5%. The
        old t-gate re-checked hourly let one through ~31% of the time per week."""
        rng = random.Random(11)
        L, _ = lab()
        picks, trials = 0, 100

        def outcome():
            q = rng.uniform(0.5, 0.9)
            return per_budget((1 / q - 1) if rng.random() < q else -1.0), rng.choice([0.01, -0.01])
        for _ in range(trials):
            picks += any(self.ever_qualifies(L, [outcome() for _i in range(1440)]) for _v in VARIANTS)
        self.assertLess(picks / trials, 0.05, f"false pick rate {picks / trials:.2%}")

    def test_drift_trap_rarely_qualifies(self):
        """Three 'sell-like' variants with no long-run edge that win while the market falls and lose while it rises
        (regimes of ~1-2 days), checked after every outcome for 30 days: must stay < 5%."""
        rng = random.Random(5)
        L, _ = lab()
        picks, trials = 0, 100
        for _ in range(trials):
            hit = False
            for _v in range(3):
                seq, down, left = [], True, 0
                for _i in range(1440):
                    if left <= 0:
                        down, left = rng.random() < 0.5, rng.randint(48, 96)
                    left -= 1
                    q = rng.uniform(0.5, 0.9)
                    p_win = min(0.99, max(0.01, q + (0.06 if down else -0.06)))
                    seq.append((per_budget((1 / q - 1) if rng.random() < p_win else -1.0), -0.01 if down else 0.01))
                hit = hit or self.ever_qualifies(L, seq)
            picks += hit
        self.assertLess(picks / trials, 0.05, f"drift-trap pick rate {picks / trials:.2%}")

    def test_a_real_edge_gets_through(self):
        """+15% per $ staked (both directions) at a typical $25 per cluster is found within 30 days in most runs: the
        bar is not impossible."""
        rng = random.Random(3)
        L, _ = lab()
        found = 0
        for _ in range(20):
            seq = []
            for _i in range(1440):
                q = rng.uniform(0.5, 0.9)
                seq.append((per_budget((1 / q - 1) if rng.random() < min(0.99, q * 1.15) else -1.0),
                            rng.choice([0.01, -0.01])))
            found += self.ever_qualifies(L, seq)
        self.assertGreaterEqual(found, 15)

    def test_clusters_score_once_and_wait_for_open_bets(self):
        L, sn = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}})
        sn.markets = {1: market(1, 10), 2: {**market(2, 10), "strike": 86_000.0}}         # same expiry: one cluster
        sn.models, sn.books = {1: 0.5, 2: 0.5}, {1: SELL, 2: SELL}
        L.on_book(1, NOW)
        L.on_book(2, NOW)
        update(L, sn, 1, NOW + 1)
        update(L, sn, 2, NOW + 1)
        sn.rest.y = {1: 0.0}                                                           # only market 1 has settled
        L.poll(NOW + 10 * 60 + 121)
        L._close_clusters(NOW + 10 * 60 + 121)
        self.assertEqual(L.led["v"]["ep"]["n"], 0)                                     # market 2 still open: wait
        sn.rest.y[2] = 0.0
        L.poll(NOW + 10 * 60 + 400)
        L._close_clusters(NOW + 10 * 60 + 400)
        L._close_clusters(NOW + 10 * 60 + 500)
        self.assertEqual(L.led["v"]["ep"]["n"], 1)                                     # one outcome, scored once
        self.assertGreater(_log_wealth(L.led["v"]["ep"]["lw"]), 0)                     # both won: the score rose

    def test_no_double_count_after_a_quiet_spell_or_an_outage(self):
        """Review r6: 'done' keys used to expire 9 days after scoring while a quiet variant's rows lived on, so the
        same outcomes were scored again (W 3x -> 372x on no new data)."""
        import tempfile
        L, _ = lab()
        trend = [0.01, -0.01] * 48
        self.feed(L, "both_last15", [0.3, -0.2] * 48, trend, span_d=1.0)
        n0, lw0 = L.led["both_last15"]["ep"]["n"], list(L.led["both_last15"]["ep"]["lw"])
        for d in (5, 9.5, 10, 19, 28):                                                 # quiet for up to 28 days
            L._close_clusters(NOW + d * 86400)
        self.assertEqual((L.led["both_last15"]["ep"]["n"], L.led["both_last15"]["ep"]["lw"]), (n0, lw0))
        with tempfile.TemporaryDirectory() as dd:                                      # a 2-day outage, then restart
            clock = [NOW]
            A = ShadowLab(Sniper(), Path(dd), clock=lambda: clock[0])
            self.feed(A, "both_last15", [0.3, -0.2] * 48, trend, span_d=8.5)
            n1 = A.led["both_last15"]["ep"]["n"]
            A._save()
            clock[0] = NOW + 2 * 86400
            B = ShadowLab(Sniper(), Path(dd), clock=lambda: clock[0])
            B.poll(clock[0])
            B._close_clusters(clock[0] + 31)
            self.assertEqual(B.led["both_last15"]["ep"]["n"], n1)

    def test_overlapping_cluster_is_skipped_not_scored(self):
        """A cluster whose first bet was placed before the last scored expiry is counted but never scored (the earlier
        stakes did not know its outcome would arrive)."""
        L, _ = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False}})
        led = L.led["v"]
        led["settled"] += [[NOW - 3000, 1, NOW - 3600, 5.0, 10.0, 0.01, NOW - 4000],      # short: scored
                           [NOW - 1000, 2, NOW - 1200, 5.0, 10.0, 0.01, NOW - 30000]]     # bet 8 h earlier: overlaps
        L._close_clusters(NOW)
        self.assertEqual((led["ep"]["n"], led["ep"]["skipped"]), (1, 1))
        led["settled"].append([NOW + 500, 3, NOW + 300, -5.0, 10.0, None, None])         # unknown first bet: skipped
        L._close_clusters(NOW + 600)
        self.assertEqual((led["ep"]["n"], led["ep"]["skipped"]), (1, 2))

    def test_trend_label_comes_from_the_price_before_the_bet(self):
        from collections import namedtuple
        Bar = namedtuple("Bar", "ts close")
        L, sn = lab(variants={"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}})
        sn.model.bars = {"BTC": [Bar(NOW - 9 * 3600 + 300 * i, 80_000.0) for i in range(36)]}   # 6 h ago: 80k
        sn.markets, sn.models, sn.books = {1: market(1, 10)}, {1: 0.5}, {1: SELL}
        L.on_book(1, NOW)                                                              # spot 85k now: rising
        update(L, sn, 1, NOW + 1)
        sn.rest.y = {1: 0.0}
        L.poll(NOW + 10 * 60 + 121)
        self.assertAlmostEqual(L.led["v"]["settled"][0][5], math.log(85_000 / 80_000), places=4)
        self.assertAlmostEqual(L.led["v"]["settled"][0][6], NOW)                        # first bet time
        L._close_clusters(NOW + 10 * 60 + 121)
        self.assertEqual((L.led["v"]["ep"]["n_up"], L.led["v"]["ep"]["n_dn"]), (1, 0))

    def test_each_variant_labels_its_own_first_bet(self):
        """Review r6: the label was shared, so a last-15-min variant got a 24 h variant's label from hours earlier."""
        from collections import namedtuple
        Bar = namedtuple("Bar", "ts close")
        L, sn = lab(variants={"long": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False},
                              "short": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}})
        sn.markets, sn.models, sn.books = {1: market(1, 300)}, {1: 0.5}, {1: SELL}
        sn.model.bars = {"BTC": [Bar(NOW - 12 * 3600 + 300 * i, 90_000.0) for i in range(80)]}  # 6 h ago: 90k
        L.on_book(1, NOW)                                                              # 5 h out: only 'long' bets
        update(L, sn, 1, NOW + 1)                                                      # filled
        sn.model.bars = {"BTC": [Bar(NOW + 3600 - 300 * i, 80_000.0) for i in range(80)][::-1]}
        L.on_book(1, NOW + 290 * 60)                                                   # 10 min out: 'short' bets
        update(L, sn, 1, NOW + 290 * 60 + 1)
        self.assertLess(L.led["long"]["first"][1][1], 0)                               # 85k vs 90k: falling
        self.assertGreater(L.led["short"]["first"][1][1], 0)                           # 85k vs 80k: rising

    def test_season_rules_are_fixed_and_evidence_is_never_restarted_for_free(self):
        import tempfile
        a = {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}
        with tempfile.TemporaryDirectory() as d:
            L1 = ShadowLab(Sniper(), Path(d), variants={"a": a, "b": {**a, "sides": "both"}}, clock=lambda: NOW,
                           season_size=7)
            self.assertAlmostEqual(L1.bar(), 140)
            for _ in range(5):
                L1._ep_update(L1.led["a"]["ep"], -0.2, 0.01)                           # a has been losing
            L1._save()
            # a later config tries to loosen the rules: the season keeps its own
            L2 = ShadowLab(Sniper(), Path(d), variants={"a": {**a, "edge": 0.04}}, clock=lambda: NOW, alpha=0.10,
                           min_clusters=50, season_size=7)
            self.assertEqual((L2.ALPHA, L2.MIN_CLUSTERS, L2.bar()), (0.05, 300, 140))
            self.assertEqual(L2.led["a"]["ep"]["n"], 0)                                # a tweak is a new variant
            L2._save()
            # back to the original a: its losing record returns (no free restart)
            L3 = ShadowLab(Sniper(), Path(d), variants={"a": a}, clock=lambda: NOW, season_size=7)
            self.assertEqual(L3.led["a"]["ep"]["n"], 5)
            self.assertLess(_log_wealth(L3.led["a"]["ep"]["lw"]), 0)
            self.assertEqual(len(L3.tried), 3)                                          # a, b, a' all count

    def test_only_the_seasons_first_variants_may_qualify(self):
        many = {f"v{i}": {"sides": "sell", "edge": 0.05 + i / 1000, "min_tte": 3, "max_tte": 15, "selfcal": False}
                for i in range(9)}
        L, _ = lab(variants=many, season_size=7)
        self.assertEqual([L.eligible(n) for n in many], [True] * 7 + [False] * 2)
        self.feed(L, "v8", [0.3, 0.25, -0.2, 0.35, 0.2, 0.3] * 70, [0.01, -0.01] * 210)
        L.autopilot = True
        self.assertIsNone(L.select(NOW))
        self.assertIn("display only", next(s for s in L.selection["scores"] if s["name"] == "v8")["why"])

    def test_direction_check_takes_the_drift_out(self):
        """Display: a strategy that wins $10 per result in falling markets and loses $8 in rising ones makes +$1 per
        result direction-neutral, whatever the mix of the window."""
        L, _ = lab()
        rows = L.led["both_24h"]["settled"]
        for i in range(30):
            rows.append([NOW - 1000 + i, 100 + i, NOW - 5000 + 60 * i, 10.0 if i % 3 else -8.0, 20.0,
                         -0.01 if i % 3 else 0.01, NOW - 6000])
        rows.append([NOW, 999, NOW - 100, 3.0, 20.0, None, None])                     # no label: "flat"
        bt = L.to_dict()["board"][[b["name"] for b in L.to_dict()["board"]].index("both_24h")]["by_trend"]
        self.assertEqual((bt["falling"]["n"], bt["rising"]["n"], bt["flat"]["n"]), (20, 10, 1))
        self.assertEqual((bt["falling"]["mean"], bt["rising"]["mean"]), (10.0, -8.0))
        self.assertEqual(bt["neutral"]["mean"], 1.0)

    def test_bet_size_ladder_fills_only_what_the_book_held(self):
        """Display only: each fill is replayed at 2x..5x the ticket, capped by the top-of-book size, settled with the
        same outcome, and 1x is counted from the ladder's start so all sizes cover the same time."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sn = Sniper()
            v = {"v": {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False}}
            L = ShadowLab(sn, Path(d), variants=v, clock=lambda: NOW)
            sn.markets, sn.models = {1: market(1, 10)}, {1: 0.5}
            sn.books = {1: {"bid": 0.58, "ask": 0.70, "bid_sz": 60, "ask_sz": 100}}   # 60 Yes contracts at 58c
            L.on_book(1, NOW)
            update(L, sn, 1, NOW + 1)                                                 # still there: filled
            lad = L.ladder["v"]["rungs"]
            want = {m: math.ceil(10 * m / 0.43) for m in (2, 3, 4, 5)}                # sized at the 43c limit
            self.assertEqual({m: lad[m]["got"] for m in want}, {m: min(want[m], 60) for m in want})
            self.assertEqual([lad[m]["partial"] for m in (2, 3, 4, 5)], [0, 1, 1, 1])  # 3x+ wanted more than 60
            self.assertEqual(lad[1]["got"], math.ceil(10 / 0.43))                     # 1x: the variant's own ticket
            L._save()
            L2 = ShadowLab(Sniper(), Path(d), variants=v, clock=lambda: NOW)          # survives a restart
            self.assertEqual(L2.ladder["v"]["rungs"][5]["got"], 60)
            sn.rest.y = {1: 0.0}                                                      # No wins
            L.poll(NOW + 10 * 60 + 121)
            rows = {s["x"]: s for s in L.to_dict()["board"][0]["ladder"]["sizes"]}
            self.assertEqual(rows[1]["markets"], 1)
            self.assertAlmostEqual(rows[5]["net"], 60 * (1 - 0.42) - 60 * L.FEE, places=2)
            self.assertAlmostEqual(rows[2]["net"] / rows[1]["net"], want[2] / math.ceil(10 / 0.43), places=2)
            self.assertLess(rows[5]["fill_share"], 1.0)

    def test_huge_scores_do_not_crash_the_gate(self):
        """Review r6: exp(log score) overflowed past ~710 and stopped the lab for good."""
        L, _ = lab(autopilot=True)
        L.led["both_last15"]["ep"]["lw"] = [800.0] * 6
        L.select(NOW)
        L.to_dict()
        s = next(s for s in L.selection["scores"] if s["name"] == "both_last15")
        self.assertEqual(s["score"], round(math.exp(50), 3))

    def test_live_gets_only_the_selected_variant_within_the_guards(self):
        L, sn = lab(autopilot=True)
        calls = []
        L.live_hook = lambda *a: calls.append(a)
        sn.markets, sn.models, sn.books = {1: market(1, 10)}, {1: 0.5}, {1: SELL}
        L.on_book(1, NOW)                                                              # nothing selected: no live
        self.assertEqual(calls, [])
        L.selected = "both_last15"
        L.on_book(1, NOW + 20)
        self.assertEqual(len(calls), 1)
        mid, side, limit, fair, mk = calls[0]
        self.assertEqual((mid, side, limit, fair, mk["path"]), (1, -1, 0.58, 0.5, "lab:both_last15"))
        sn.band_lo = 0.60                                                              # operator's price band: 58c out
        L.on_book(1, NOW + 40)
        self.assertEqual(len(calls), 1)
        sn.band_lo = 0.0
        L.autopilot = False                                                            # autopilot off: never live
        L.on_book(1, NOW + 60)
        self.assertEqual(len(calls), 1)

    def test_selection_is_never_restored_from_disk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sn = Sniper()
            a = ShadowLab(sn, Path(d), clock=lambda: NOW, autopilot=True)
            a.selected = "both_last15"
            a._save()
            b = ShadowLab(Sniper(), Path(d), clock=lambda: NOW, autopilot=True)
            self.assertIsNone(b.selected)


if __name__ == "__main__":
    unittest.main()
