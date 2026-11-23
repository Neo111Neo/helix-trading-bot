"""Hourly probability test: model maths, round lifecycle, resolution from the expiry candle, persistence, scoring."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import DEFAULTS  # noqa: E402
from helix.hourly import HourlyTest, lognormal_above, sigma_per_hour, logloss  # noqa: E402
from helix.jev import JevResult  # noqa: E402
from helix.market import Candle  # noqa: E402

HOUR = 1_791_219_600  # an exact UTC hour mark


class FakeRest:
    """5m and 1m candles on a gentle sine walk; the 1m candle closing at expiry has a known close."""

    def __init__(self, expiry_close: float = 101.0):
        self.expiry_close = expiry_close

    def candles(self, coin, interval, minutes, now):
        step = 300 if interval == "5m" else 60
        start = int(now - minutes * 60) // step * step
        out, t = [], start
        while t + step <= now:
            px = 100 * (1 + 0.002 * math.sin(t / 900))
            if interval == "1m" and t == HOUR + 3600 - 60:
                px = self.expiry_close
            out.append(Candle(t, px, px * 1.001, px * 0.999, px, 1.0))
            t += step
        return out

    def l2_book(self, coin):
        return [(99.99, 5.0)], [(100.01, 5.0)], HOUR

    def asset_ctxs(self):
        return {"BTC": {"funding": 0.0000125, "premium": -0.0003}}


class FakeJev:
    def __init__(self):
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        answers = {q: {"type": "noul", "p": p} for q, p in zip(questions, (0.8, 0.5, 0.2))}
        return JevResult("ok", "jev-test", 150.0, answers, {"input_tokens": 300})


def make(tmp, clock, rest=None):
    cfg = {**DEFAULTS, "hourly_test": {"enabled": True, "coins": ["BTC"]}}
    return HourlyTest(cfg, Path(tmp), rest=rest or FakeRest(), jev_client=FakeJev(), clock=clock)


class ModelTests(unittest.TestCase):
    def test_lognormal(self):
        self.assertAlmostEqual(lognormal_above(100, 100, 0.01), 0.498, places=3)   # tiny -sigma^2/2 drift
        self.assertGreater(lognormal_above(101, 100, 0.01), 0.8)
        self.assertLess(lognormal_above(99, 100, 0.01), 0.2)
        self.assertEqual(lognormal_above(101, 100, 0.0), 1.0)

    def test_sigma_per_hour_scales_bars(self):
        closes = [100 * (1.001 if i % 2 else 0.999) for i in range(40)]
        self.assertAlmostEqual(sigma_per_hour(closes, 5) / sigma_per_hour(closes, 1), math.sqrt(12 / 60), places=6)
        self.assertEqual(sigma_per_hour([100, 101], 5), 0.0)

    def test_logloss_is_clipped(self):
        self.assertTrue(math.isfinite(logloss(1.0, 0)))


class LifecycleTests(unittest.TestCase):
    def test_round_opens_once_resolves_from_expiry_candle_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [HOUR + 120]
            h = make(tmp, lambda: now[0])
            h.step(); h.step()
            self.assertEqual(len(h.rounds), 1, "only one round per hour")
            c = h.rounds[0]["coins"]["BTC"]
            self.assertEqual(len(c["strikes"]), 3)
            self.assertTrue(c["strikes"][0] < c["strikes"][1] < c["strikes"][2])
            self.assertTrue(c["model"][0] > c["model"][1] > c["model"][2], "P(above) must fall as the strike rises")
            self.assertTrue(c["model_emp"][0] >= c["model_emp"][1] >= c["model_emp"][2])
            self.assertEqual(c["jev"]["probs"], [0.8, 0.5, 0.2])
            self.assertIn("strike_mid", c["state"])
            # not resolved before the expiry candle has closed
            now[0] = HOUR + 3600 + 30
            h.step()
            self.assertNotIn("outcome", c)
            # resolved from the 1-minute candle closing at expiry (close 101.0 > all strikes near 100)
            now[0] = HOUR + 3600 + 200
            h.step()
            self.assertEqual(h.rounds[0]["coins"]["BTC"]["outcome"]["price"], 101.0)
            self.assertEqual(h.rounds[0]["coins"]["BTC"]["outcome"]["above"], [1, 1, 1])
            self.assertTrue(h.rounds[0]["resolved"])
            # the next hour's round opened too (we're 3 minutes into it)
            self.assertEqual([r["hour"] for r in h.rounds], [HOUR, HOUR + 3600])
            # restart: state reloads from disk, and stepping again never duplicates an hour
            h2 = make(tmp, lambda: now[0])
            h2.step()
            self.assertEqual([r["hour"] for r in h2.rounds], [HOUR, HOUR + 3600])
            self.assertEqual(h2.score(h2.rows(), "jev")["n"], 3)

    def test_missing_coin_is_retried_inside_the_window(self):
        class FlakyRest(FakeRest):
            fails = 1

            def candles(self, coin, interval, minutes, now):
                if coin == "ETH" and self.fails:
                    self.fails -= 1
                    raise RuntimeError("429 rate limited")
                return super().candles(coin, interval, minutes, now)
        with tempfile.TemporaryDirectory() as tmp:
            now = [HOUR + 120]
            cfg = {**DEFAULTS, "hourly_test": {"enabled": True, "coins": ["BTC", "ETH"]}}
            jev = FakeJev()
            h = HourlyTest(cfg, Path(tmp), rest=FlakyRest(), jev_client=jev, clock=lambda: now[0])
            h.step()
            self.assertEqual(list(h.rounds[0]["coins"]), ["BTC"])
            now[0] += 30
            h.step()                                         # at most one retry a minute
            self.assertEqual(list(h.rounds[0]["coins"]), ["BTC"])
            now[0] += 31
            h.step()
            self.assertEqual(sorted(h.rounds[0]["coins"]), ["BTC", "ETH"])
            self.assertEqual(len(h.rounds), 1)
            self.assertEqual(jev.calls, 2)                   # BTC was not asked again

    def test_no_round_late_in_the_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = make(tmp, lambda: HOUR + 1200)
            h.step()
            self.assertEqual(h.rounds, [])


class ScoreJev(FakeJev):
    """Answers noul questions like FakeJev and score questions with most weight on the first level."""

    def ask(self, state, questions):
        self.calls += 1
        self.last_state = state
        raw = {}
        for qid, q in questions.items():
            if q["type"] == "score":
                n = len(q["criteria"])
                raw[qid] = {"type": "score", "probabilities": {str(i): (0.7 if i == 0 else 0.3 / (n - 1)) for i in range(n)}}
            else:
                raw[qid] = {"type": "noul", "noul": 0.5}
        from helix.jev import normalize_answers
        return JevResult("ok", "jev-test", 150.0, normalize_answers(raw, questions), {"input_tokens": 300})


class VariantTests(unittest.TestCase):
    def test_lab_variants_are_asked_decoded_and_scored(self):
        from helix import lab
        cfg = {**DEFAULTS, "hourly_test": {"enabled": True, "coins": ["BTC"], "jev_variants": {
            "jev_v2": "helix/prompts/present_stretch_reversion.json",
            "jev_v3": "helix/prompts/calibrated_anchor_bounded_score.json"}}}
        with tempfile.TemporaryDirectory() as tmp:
            now = [HOUR + 120]
            jev = ScoreJev()
            h = HourlyTest(cfg, Path(tmp), rest=FakeRest(), jev_client=jev, clock=lambda: now[0])
            self.assertEqual(h.forecasters, ["model", "model_emp", "model_har", "jev", "jev_v2", "jev_v3", "claude"])
            h.step()
            c = h.rounds[0]["coins"]["BTC"]
            spec, _, dec = h.variants["jev_v2"]
            n = len(dec["above_low"]["criteria"])
            want = lab.decode(dec["above_low"], {"probabilities": {i: (0.7 if i == 0 else 0.3 / (n - 1)) for i in range(n)}})
            self.assertEqual(c["jev_v2"]["status"], "ok")
            self.assertAlmostEqual(c["jev_v2"]["probs"][0], want)
            self.assertNotIn("baseline_p_above_low", jev.last_state)  # v2 asks for no extra blocks
            self.assertEqual(c["jev_v3"]["status"], "no claude")    # needs Claude's numbers; none configured here
            now[0] = HOUR + 3600 + 200
            h.step()
            rows = h.rows()
            self.assertTrue(all("jev_v2" in r for r in rows))
            self.assertEqual(h.to_dict()["scores"]["jev_v2"]["n"], 3)


class ScoringTests(unittest.TestCase):
    @staticmethod
    def rows(n_hours, good=True):
        out = []
        for k in range(n_hours):
            y = k % 2
            p = (0.8 if y else 0.2) if good else (0.2 if y else 0.8)
            out.append({"hour": k * 3600, "coin": "BTC", "strike": "mid", "y": y, "model": 0.5, "model_emp": 0.5, "jev": p})
        return out

    def test_no_verdict_before_fixed_horizon(self):
        s = HourlyTest.score(self.rows(60), "jev", n_boot=400)
        self.assertIsNone(s["beats_bar"])
        self.assertIn("collecting", s["verdict"])
        self.assertGreater(s["primary_gain"], 0)
        self.assertEqual(s["bar"], "model_emp")
        self.assertIn("primary_ci", s)

    def test_verdict_at_horizon_uses_block_bootstrap(self):
        good = HourlyTest.score(self.rows(60), "jev", n_boot=400, target_hours=50)
        self.assertTrue(good["beats_bar"])
        bad = HourlyTest.score(self.rows(60, good=False), "jev", n_boot=400, target_hours=50)
        self.assertFalse(bad["beats_bar"])
        self.assertEqual(bad["verdict"], "worse than the baseline")
        self.assertEqual(HourlyTest.score([], "jev"), {"n": 0})

    def test_baselines_have_no_comparison(self):
        s = HourlyTest.score(self.rows(10), "model_emp")
        self.assertNotIn("bar", s)


class BaselineTests(unittest.TestCase):
    def test_empirical_baseline_keeps_fat_tails(self):
        from helix.hourly import empirical_above, past_z, seasonal_factors
        zs = [0.0] * 50 + [2.0] * 25 + [-2.0] * 25     # fatter than normal: half the mass at +/-2
        self.assertAlmostEqual(empirical_above(zs, 0.0), 0.5, places=2)
        self.assertGreater(empirical_above(zs, 0.5), 0.2)
        c5 = [Candle(HOUR - 3600 * 30 + i * 300, 100 + math.sin(i / 7), 0, 0, 100 + math.sin(i / 7), 1) for i in range(360)]
        f = seasonal_factors(c5)
        self.assertEqual(len(f), 24)
        self.assertTrue(all(0.5 <= x <= 2.5 for x in f))
        self.assertGreater(len(past_z(c5, f)), 0)


if __name__ == "__main__":
    unittest.main()
