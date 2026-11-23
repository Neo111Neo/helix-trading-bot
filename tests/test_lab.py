"""Jev Lab: no look-ahead in replayed states, the replay matches the live hourly test, variant checks, statistics."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix import lab  # noqa: E402
from helix.config import DEFAULTS  # noqa: E402
from helix.hourly import HourlyTest  # noqa: E402
from helix.market import Candle  # noqa: E402

T0 = 1_790_000_000 // 3600 * 3600


def walk(n: int, step: int, seed: int, start: int) -> list[Candle]:
    rng, px, out = random.Random(seed), 100.0, []
    for k in range(n):
        o = px
        px *= math.exp(rng.gauss(0, 0.002 * math.sqrt(step / 300)))
        out.append(Candle(start + k * step, o, max(o, px) * 1.0005, min(o, px) * 0.9995, px, 1 + rng.random()))
    return out


def raw(seed: int = 3) -> dict:
    c5 = walk(4990, 300, seed, T0)
    end = c5[-1].ts + 300
    c1 = walk(4990, 60, seed + 1, end - 4990 * 60)
    funding = [[float(T0 + h * 3600), 1e-5 * math.sin(h), 2e-4] for h in range(4990 * 300 // 3600)]
    return {"coin": "BTC", "c1": c1, "c5": c5, "funding": funding}


def perturb_after(r: dict, t: float) -> dict:
    """Same history up to t, a completely different future after it."""
    bump = lambda cs, step: [c if c.ts + step <= t else Candle(c.ts, c.open * 1.1, c.high * 1.1, c.low * 1.1,  # noqa: E731
                                                                     c.close * 1.1, c.volume * 3) for c in cs]
    return {**r, "c1": bump(r["c1"], 60), "c5": bump(r["c5"], 300),
            "funding": [f if f[0] <= t else [f[0], 0.01, 0.01] for f in r["funding"]]}


class NoLookAheadTests(unittest.TestCase):
    def test_direction_states_ignore_the_future(self):
        r = raw()
        a = lab.build_direction(r)
        self.assertGreater(len(a), 100)
        t = a[len(a) // 2]["ts"]
        b = {x["id"]: x for x in lab.build_direction(perturb_after(r, t))}
        for x in a:
            if x["ts"] <= t:
                y = b[x["id"]]
                self.assertEqual((x["state"], x["stats"], x["anchor"]), (y["state"], y["stats"], y["anchor"]), x["id"])
        same_fwd = [x["fwd"] == b[x["id"]]["fwd"] for x in a if x["ts"] + 900 <= t]
        self.assertTrue(all(same_fwd))
        self.assertNotIn("price", a[0]["state"])            # no price level in the direction task

    def test_hourly_states_ignore_the_future(self):
        r = raw()
        a = lab.build_hourly(r)
        self.assertGreater(len(a), 50)
        t = a[len(a) // 2]["ts"]
        b = {x["id"]: x for x in lab.build_hourly(perturb_after(r, t))}
        for x in a:
            if x["ts"] <= t:
                self.assertEqual((x["state"], x["model_emp"]), (b[x["id"]]["state"], b[x["id"]]["model_emp"]))


class FakeRest:
    def __init__(self, c5, now):
        self.c5, self.now = c5, now

    def candles(self, coin, interval, minutes, now):
        step = 300 if interval == "5m" else 60
        lo = now - minutes * 60
        if interval == "5m":
            return [c for c in self.c5 if lo <= c.ts and c.ts + step <= now]
        last = [c for c in self.c5 if c.ts + 300 <= now][-1]
        return [Candle(int(now - (20 - k) * 60), last.close, last.close, last.close, last.close, 1) for k in range(20)]

    def l2_book(self, coin):
        last = [c for c in self.c5 if c.ts + 300 <= self.now][-1]
        return [(last.close, 1.0)], [(last.close, 1.0)], self.now


class ReplayMatchesLiveTests(unittest.TestCase):
    def test_hourly_replay_equals_live_baselines(self):
        r = raw(7)
        states = lab.build_hourly(r)
        s = states[-1]
        live = HourlyTest(DEFAULTS, Path("."), coins=["BTC"], rest=FakeRest(r["c5"], s["ts"]), jev_client=False)
        cs = live._coin_state("BTC", s["ts"], s["ts"] + 3600, {})
        self.assertEqual(cs["strikes"], [s["state"][f"strike_{n}"] for n in ("low", "mid", "high")])
        for a, b in zip(cs["model"], s["model"]):
            self.assertAlmostEqual(a, b, places=9)
        for a, b in zip(cs["model_emp"], s["model_emp"]):
            self.assertAlmostEqual(a, b, places=9)
        self.assertGreaterEqual(s["n_z"], 100)


SPEC = {
    "name": "t", "blocks_direction": ["anchor"], "blocks_above": [],
    "direction": {"type": "score", "instructions": "Higher in {H} minutes? See `baseline_p_up_{H}m`.",
                  "criteria": ["lower", "same", "higher"], "level_values": [0.3, 0.5, 0.7]},
    "extras_direction": {"stretched": {"type": "noul", "instructions": "Is `move_15m_in_sigmas` beyond 2?"}},
    "above": {"type": "choice", "instructions": "Above `strike_{S}` at expiry?",
              "criteria": {"above": "ends above", "below": "ends at or below"}, "up_key": "above", "down_key": "below"},
    "extras_above": {},
}


class VariantTests(unittest.TestCase):
    def test_render_and_decode(self):
        api, dec, extras = lab.render(SPEC, "D")
        self.assertEqual(set(api), {"p_up_5m", "p_up_15m", "stretched"})
        self.assertIn("`baseline_p_up_15m`", api["p_up_15m"]["instructions"])
        self.assertNotIn("level_values", api["p_up_5m"])               # decoding hints never reach the API
        self.assertAlmostEqual(lab.decode(dec["p_up_5m"], {"probabilities": {0: 0.2, 1: 0.3, 2: 0.5}}), 0.56)
        # the same answer after a JSON round-trip through the cache (level keys become strings)
        self.assertAlmostEqual(lab.decode(dec["p_up_5m"], {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}), 0.56)
        _, kdec, _ = lab.render(SPEC, "K")
        self.assertAlmostEqual(lab.decode(kdec["above_mid"], {"probabilities": {"above": 0.3, "below": 0.1}}), 0.75)

    def test_nudge_decode_moves_the_anchor_in_log_odds(self):
        spec = {"name": "n", "blocks_above": ["har"], "extras_above": {},
                "above": {"type": "score", "instructions": "Nudge `har_p_above_{S}`?", "criteria": ["below", "none", "above"],
                          "level_shifts": [-0.4, 0.0, 0.4], "anchor": "har_p_above_{S}"}}
        _, dec, _ = lab.render(spec, "K15")
        q = dec["above_low"]
        self.assertEqual(q["anchor"], "har_p_above_low")
        st = {"har_p_above_low": 0.9}
        self.assertAlmostEqual(lab.decode(q, {"probabilities": {"1": 1.0}}, st), 0.9, places=6)   # no reason: the anchor
        up = lab.decode(q, {"probabilities": {"2": 1.0}}, st)
        self.assertAlmostEqual(math.log(up / (1 - up)) - math.log(0.9 / 0.1), 0.4, places=6)
        self.assertIsNone(lab.decode(q, {"probabilities": {"2": 1.0}}, {}))                        # no anchor, no answer
        bad = {**spec, "above": {**spec["above"], "anchor": "not_a_field"}}
        with self.assertRaises(ValueError):
            lab.render(bad, "K")

    def test_k15_task_and_recent_vol_baseline(self):
        r = raw()
        k15 = lab.build_strikes(r, 3, 900, 5.0, "K15")
        self.assertGreater(len(k15), 50)
        s = k15[-1]
        self.assertEqual(s["state"]["minutes_to_expiry"], 15.0)
        self.assertTrue(s["id"].startswith("K15:"))
        self.assertTrue(s["model_har"][0] > s["model_har"][1] > s["model_har"][2])
        self.assertEqual(set(s["har"]), set(lab.K_BLOCKS["har"]))
        t = k15[len(k15) // 2]["ts"]                                    # no look-ahead in the 15-minute task either
        b = {x["id"]: x for x in lab.build_strikes(perturb_after(r, t), 3, 900, 5.0, "K15")}
        for x in k15:
            if x["ts"] <= t:
                self.assertEqual((x["state"], x["model_emp"], x["model_har"]), (b[x["id"]]["state"], b[x["id"]]["model_emp"],
                                                                              b[x["id"]]["model_har"]))

    def test_unknown_field_is_rejected(self):
        bad = {**SPEC, "blocks_direction": []}                           # baseline field without its block
        with self.assertRaises(ValueError):
            lab.render(bad, "D")


class StatsTests(unittest.TestCase):
    def test_logit_recovers_signal(self):
        rng = random.Random(1)
        X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(1500)]
        y = [1 if rng.random() < 1 / (1 + math.exp(-1.5 * x[0])) else 0 for x in X]
        m = lab.Logit(l2=1.0).fit(X, y)
        self.assertGreater(m.predict([2.0, 0.0]), 0.9)
        self.assertLess(m.predict([-2.0, 0.0]), 0.1)
        self.assertAlmostEqual(m.predict([0.0, 3.0]), 0.5, delta=0.08)

    def test_stacked_is_out_of_fold_on_design(self):
        rng = random.Random(2)
        recs = [{"id": i, "ts": i, "x": rng.gauss(0, 1)} for i in range(800)]
        for r in recs:
            r["y"] = 1 if rng.random() < 0.5 else 0          # pure noise: an in-sample fit would still "learn" it
        des, hold = recs[:600], recs[600:]
        noise = lambda r: [r["x"]] + [((r["id"] * 7919 + k) % 101) / 101 for k in range(12)]  # noqa: E731
        p = lab.stacked(des, hold, noise, lambda r: r["y"])
        self.assertEqual(set(p), {r["id"] for r in recs})
        ll_des = sum(lab.logloss(p[r["id"]], r["y"]) for r in des) / len(des)
        self.assertGreater(ll_des, math.log(2) - 0.01)       # no in-sample optimism on noise

    def test_rank_stats(self):
        self.assertAlmostEqual(lab.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(lab.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        lo, hi = lab.block_ci([1.0] * 40, list(range(40)))
        self.assertAlmostEqual(lo, 1.0)


if __name__ == "__main__":
    unittest.main()
