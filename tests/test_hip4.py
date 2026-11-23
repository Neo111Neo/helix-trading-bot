"""HIP-4 recorder: market parsing (UTC expiries), checkpoint snapshots, forecasters never see the price, settlement
from settledOutcome, scoring against the market and the bid/ask, and the same baseline as the hourly test."""

from __future__ import annotations

import calendar
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import DEFAULTS  # noqa: E402
from helix.hip4 import FEE, HIP4Recorder, book_top, parse_market, score, strike_state  # noqa: E402
from helix.hourly import HourlyTest  # noqa: E402
from helix.jev import JevResult, normalize_answers  # noqa: E402
from helix.market import Candle  # noqa: E402

HOUR = 1_791_226_800          # 2026-10-05 19:00 UTC
COINS = ["BTC", "ETH", "SOL", "HYPE"]


def utc(s):
    return calendar.timegm((int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[9:11]), int(s[11:13]), 0, 0, 0, 0))


class ParseTests(unittest.TestCase):
    def test_both_market_families_and_utc_expiry(self):
        hl = {"outcome": 8596, "name": "Recurring",
              "description": "class:priceBinary|underlying:BTC|expiry:20261006-0600|targetPrice:85749|period:1d"}
        out = {"outcome": 8831, "name": "template:binaryPrice", "venue": "out",
               "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:85597|time:20261005-1900"}
        a, b = parse_market(hl, COINS), parse_market(out, COINS)
        self.assertEqual((a["coin"], a["strike"], a["rule"], a["venue"]), ("BTC", 85749.0, ">=", "HL"))
        self.assertEqual(a["expiry"], utc("20261006-0600"))
        self.assertEqual((b["venue"], b["rule"], b["twap_s"], b["expiry"]), ("out", ">", 60, HOUR))
        self.assertIsNone(parse_market({**out, "description": out["description"].replace("perp:BTC", "perp:xyz:SP500")}, COINS))
        self.assertIsNone(parse_market({"outcome": 1, "name": "template:priceTouch", "description": "perp:BTC|target:1|time:20261101-0000"}, COINS))

    def test_book_top(self):
        t = book_top([[{"px": "0.62", "sz": "40"}], [{"px": "0.66", "sz": "25"}]])
        self.assertAlmostEqual(t["mid"], 0.64)
        self.assertTrue(t["usable"])
        self.assertFalse(book_top([[{"px": "0.0002", "sz": "1"}], [{"px": "0.98", "sz": "1"}]])["usable"])
        self.assertFalse(book_top([[], [{"px": "0.5", "sz": "1"}]])["usable"])


def candles(step, n, end, base=85_000.0):
    out = []
    for k in range(n):
        t = end - (n - k) * step
        px = base * (1 + 0.003 * math.sin(t / 2000) + 0.001 * math.sin(t / 333))
        out.append(Candle(t, px, px * 1.0005, px * 0.9995, px, 1.0))
    return out


class FakeRest:
    def __init__(self, now, markets, settled=None):
        self.now, self.markets, self.settled = now, markets, settled or {}

    def candles(self, coin, interval, minutes, now):
        step = 300 if interval == "5m" else 60
        end = int(now) // step * step
        return [c for c in candles(step, min(minutes * 60 // step, 5000), end) if c.ts >= now - minutes * 60]

    def l2_book(self, coin):
        px = candles(60, 1, int(self.now) // 60 * 60)[0].close
        return [(px - 0.5, 1.0)], [(px + 0.5, 1.0)], self.now

    def asset_ctxs(self):
        return {"BTC": {"funding": 0.00001, "premium": 0.0001}}

    def post(self, body):
        if body["type"] == "outcomeMeta":
            return {"outcomes": self.markets}
        if body["type"] == "l2Book":
            return {"levels": [[{"px": "0.55", "sz": "100"}], [{"px": "0.58", "sz": "80"}]], "time": int(self.now * 1000)}
        if body["type"] == "settledOutcome":
            return self.settled.get(body["outcome"])
        raise AssertionError(body)


class FakeJev:
    def __init__(self):
        self.states = []

    def ask(self, state, questions):
        self.states.append(state)
        raw = {q: ({"type": "noul", "noul": 0.7} if v["type"] == "noul" else
                   {"type": "score", "probabilities": {str(i): 1.0 / len(v["criteria"]) for i in range(len(v["criteria"]))}})
               for q, v in questions.items()}
        return JevResult("ok", "jev-test", 100.0, normalize_answers(raw, questions), {})


def market(oid, minutes_left, now, venue="out"):
    t = now + minutes_left * 60
    stamp = "%04d%02d%02d-%02d%02d" % tuple(__import__("time").gmtime(t)[:5])
    return {"outcome": oid, "name": "template:binaryPrice", "venue": venue,
            "description": f"perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:85000|time:{stamp}"}


class RecorderTests(unittest.TestCase):
    def test_checkpoints_no_price_leak_settlement_and_scores(self):
        cfg = {**DEFAULTS, "hip4": {"enabled": True, "coins": ["BTC"], "claude": False, "edge": 0.03, "variants": ["jev_v2"]},
               "hourly_test": {**DEFAULTS["hourly_test"], "jev_variants": {"jev_v2": "helix/prompts/present_stretch_reversion.json"}}}
        now = [float(HOUR - 11 * 60)]                         # 18:49: the 19:00 market is 11 min away
        mk = [market(1, 11, now[0]), market(2, 53, now[0], "skew"), market(3, 30, now[0])]
        rest = FakeRest(now[0], mk)
        jev = FakeJev()
        variants = HourlyTest._load_variants(cfg["hourly_test"]["jev_variants"])
        with tempfile.TemporaryDirectory() as tmp:
            r = HIP4Recorder(cfg, Path(tmp), rest=rest, jev_client=jev, variants=variants, clock=lambda: now[0])
            r.step()
            self.assertEqual(sorted(s["key"] for s in r.snaps), ["1:12m", "2:55m"])   # market 3 is outside both windows
            r._meta_at = 0
            r.step()
            self.assertEqual(len(r.snaps), 2, "one snapshot per market per checkpoint")
            s = next(x for x in r.snaps if x["id"] == 1)
            self.assertEqual((s["bid"], s["ask"], s["usable"]), (0.55, 0.58, True))
            self.assertAlmostEqual(s["jev"], 0.7)
            self.assertIsNotNone(s["jev_v2"])
            for st in jev.states:                              # nobody is shown the market price
                self.assertFalse({"bid", "ask", "mid", "market"} & set(st))
                self.assertEqual(st["strike_mid"], 85000.0)
            self.assertTrue(s["usable"] and s["book_gap_s"] == 0)
            # settlement: market 1 (Yes) settles and drops out of outcomeMeta; market 2 is still listed (not polled)
            now[0] = HOUR + 120
            rest.markets = [x for x in mk if x["outcome"] != 1]
            rest.settled = {1: {"settleFraction": "1.0", "details": "template"}}
            r._meta_at = 0
            r.step()
            self.assertEqual(next(x for x in r.snaps if x["id"] == 1)["y"], 1.0)
            self.assertNotIn("y", next(x for x in r.snaps if x["id"] == 2))
            d = r.to_dict()
            self.assertEqual(d["scores_all"]["market"]["n"], 1)
            self.assertIn("gain_vs_market", d["scores_all"]["jev"])
            self.assertEqual(d["forecasters"], ["market", "model", "model_emp", "model_har", "jev", "jev_v2", "claude"])
            self.assertIn("har_p_above_mid", s["har"])
            again = HIP4Recorder(cfg, Path(tmp), rest=rest, jev_client=jev, variants=variants, clock=lambda: now[0])
            self.assertEqual(len(again.snaps), 2)              # persisted

    def test_unsettled_markets_back_off_and_cannot_starve_newer_ones(self):
        cfg = {**DEFAULTS, "hip4": {**DEFAULTS["hip4"], "enabled": True, "coins": ["BTC"], "claude": False}}
        now = [float(HOUR + 3600)]
        calls = []

        class StuckRest(FakeRest):
            def post(self, body):
                if body["type"] == "settledOutcome":
                    calls.append(body["outcome"])
                    return {"settleFraction": "0.0"} if body["outcome"] == 99 else None
                return super().post(body)

        with tempfile.TemporaryDirectory() as tmp:
            r = HIP4Recorder(cfg, Path(tmp), rest=StuckRest(now[0], []), clock=lambda: now[0])
            r.snaps = [{"v": 2, "id": i, "key": f"{i}:12m", "expiry": HOUR + i, "usable": True} for i in range(1, 30)]
            r.snaps.append({"v": 2, "id": 99, "key": "99:12m", "expiry": HOUR + 600, "usable": True})
            r.resolve_due(now[0])
            self.assertLessEqual(len(calls), 4)                       # capped per step
            self.assertEqual(calls[0], 99)                            # newest expiry first: never starved
            self.assertEqual(next(s for s in r.snaps if s["id"] == 99)["y"], 0.0)
            n = len(calls)
            r.resolve_due(now[0] + 20)                                # the three that came back null wait (backoff)
            self.assertTrue(all(c not in calls[1:n] for c in calls[n:]))

    def test_short_window_baseline_matches_the_normal_model_on_normal_moves(self):
        """With normally distributed moves, the empirical 12-minute baseline must agree with the lognormal model built
        from the same volatility. Before the fix (unscaled 10-minute z's, kernel bandwidth meant for 1-hour z's) it
        overshot by ~+0.04 for a strike 1 sigma above the price."""
        import random
        diffs = []
        for seed in (4, 5, 6, 7):
            rng, px, c5 = random.Random(seed), 100.0, []
            start = HOUR - 15 * 86400
            for k in range(15 * 288):                                 # constant vol: 20 bps per 5 minutes
                o = px
                px *= math.exp(rng.gauss(0, 0.002))
                c5.append(Candle(start + k * 300, o, max(o, px), min(o, px), px, 1.0))
            now = start + 15 * 288 * 300
            c1 = [Candle(now - (20 - k) * 60, px, px, px, px, 1.0) for k in range(20)]
            f = strike_state("BTC", c5, c1, px, now, now + 12 * 60, px * math.exp(0.002 * math.sqrt(12 / 5)), {})
            self.assertGreater(f["n_z"], 1000)                        # every non-overlapping window, not just hour marks
            diffs.append(f["model_emp"] - f["model"])
        self.assertLess(abs(sum(diffs) / len(diffs)), 0.012)

    def test_score_trade_math(self):
        rows = [{"id": 1, "expiry": HOUR, "y": 1, "bid": 0.55, "ask": 0.60, "market": 0.575, "jev": 0.8},
                {"id": 2, "expiry": HOUR + 900, "y": 0, "bid": 0.50, "ask": 0.53, "market": 0.515, "jev": 0.2},
                {"id": 3, "expiry": HOUR + 1800, "y": 1, "bid": 0.50, "ask": 0.53, "market": 0.515, "jev": 0.52}]
        s = score(rows, "jev", 0.03)
        self.assertEqual(s["trades"], 2)                        # the third is inside the spread +- edge: no trade
        want = ((1 - 0.60 - FEE) + (0.50 - 0 - FEE)) / 2 * 100
        self.assertAlmostEqual(s["cents_per_trade"], round(want, 2))
        self.assertGreater(s["gain_vs_market"], 0)
        self.assertEqual(score(rows, "market", 0.03)["n"], 3)


class WeeklyLabTests(unittest.TestCase):
    def test_alert_when_the_reversal_fades(self):
        from helix.lab_schedule import LabSchedule
        ok = {c: {"autocorr": -0.1, "ic": 0.08, "gross_bps": 3.0, "n": 400, "continuation": 0.47, "trades": 50,
                  "gain_vs_half": 0.002} for c in ("BTC", "ETH", "SOL")}
        s = LabSchedule.summarize(ok, HOUR)
        self.assertFalse(s["alert"])
        self.assertEqual(s["reversing"], 3)
        faded = {c: dict(m, autocorr=0.03) for c, m in ok.items()}
        self.assertTrue(LabSchedule.summarize(faded, HOUR)["alert"])
        blind = {c: dict(m, ic=-0.02) for c, m in ok.items()}
        self.assertTrue(LabSchedule.summarize(blind, HOUR)["headline"].startswith("ALERT"))

    def test_probe_failure_retries_in_6h_away_from_the_top_of_the_hour(self):
        import copy
        import tempfile
        from helix.jev import JevResult
        from helix.lab_schedule import RETRY_S, LabSchedule

        class DownJev:
            calls = 0

            def ask(self, state, questions):
                DownJev.calls += 1
                return JevResult("error", None, None, {}, None, error="401 invalid key")

        class NoRest:
            def __getattr__(self, name):
                raise AssertionError("no download before the probe passes")
        cfg = copy.deepcopy(DEFAULTS)
        cfg["lab_schedule"]["enabled"] = True
        notes = []
        with tempfile.TemporaryDirectory() as tmp:
            now = [HOUR + 5 * 60]
            s = LabSchedule(cfg, Path(tmp), DownJev(), notify=lambda lvl, t: notes.append(t), rest=NoRest(),
                            clock=lambda: now[0])
            self.assertTrue((Path(tmp) / "lab" / "weekly" / "schedule.json").exists())   # seed saved at once
            s.state["last_run"] = 0.0
            s.tick()
            self.assertEqual(DownJev.calls, 0)               # :05 past the hour: waits for :10
            now[0] = HOUR + 15 * 60
            s.tick()
            self.assertEqual(DownJev.calls, 1)
            self.assertEqual(s.status, "error")
            self.assertEqual(s.state["retry_at"], now[0] + RETRY_S)
            self.assertIn("retrying in 6 h", notes[-1])
            now[0] += 3600
            s.tick()
            self.assertEqual(DownJev.calls, 1)               # not before the retry time
            self.assertEqual(s.to_dict()["next_run"], s.state["retry_at"])

    def test_jev_circuit_breaker(self):
        import tempfile
        from helix import lab
        from helix.jev import JevResult

        class DownJev:
            calls = 0

            def ask(self, state, questions):
                DownJev.calls += 1
                return JevResult("error", None, None, {}, None, error="503")
        spec = json.loads((Path(__file__).resolve().parent.parent / "helix" / "prompts" /
                           "present_stretch_reversion.json").read_text(encoding="utf-8"))
        states = [{"id": f"s{i}", "task": "D", "state": {}} for i in range(500)]
        orig = lab.time.sleep
        lab.time.sleep = lambda s: None
        try:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
                lab.run_jev(DownJev(), Path(tmp), "D", spec, states, {}, workers=2, log=lambda *_: None,
                            max_fail_streak=5)
        finally:
            lab.time.sleep = orig
        self.assertLess(DownJev.calls, 60)                   # stopped early, not 500 x 5 attempts


class RenderValidationTests(unittest.TestCase):
    def spec(self, name):
        return json.loads((Path(__file__).resolve().parent.parent / "helix" / "prompts" / f"{name}.json")
                          .read_text(encoding="utf-8"))

    def test_nudge_rules(self):
        from helix import lab
        good = self.spec("positioning_premium_nudge")
        lab.render(good, "K")                                # the live HIP-4 prompt is valid
        bad = self.spec("positioning_premium_nudge")
        bad["above"]["anchor"] = "realized_vol_last_4h_bps"  # a state field, but not a probability
        with self.assertRaises(ValueError):
            lab.render(bad, "K")
        d = self.spec("present_stretch_reversion")
        d["direction"]["level_shifts"] = [0.1] * len(d["direction"]["criteria"])
        d["direction"]["anchor"] = "return_5m_bps"
        with self.assertRaises(ValueError):
            lab.render(d, "D")


class SameBaselineAsHourlyTest(unittest.TestCase):
    def test_next_hour_strikes_match_the_hourly_test(self):
        now = float(HOUR - 3600 + 120)                          # 18:02, expiry 19:00
        rest = FakeRest(now, [])
        live = HourlyTest(DEFAULTS, Path("."), coins=["BTC"], rest=rest, jev_client=False)
        cs = live._coin_state("BTC", now, HOUR, {})
        c5 = rest.candles("BTC", "5m", 15 * 1440 + 10, now)
        c1 = rest.candles("BTC", "1m", 20, now)
        spot = cs["spot"]
        for j, k in enumerate(cs["strikes"]):
            f = strike_state("BTC", c5, c1, spot, now, HOUR, k, {})
            self.assertAlmostEqual(f["model_emp"], cs["model_emp"][j], places=9)
            self.assertAlmostEqual(f["model"], cs["model"][j], places=9)
            self.assertAlmostEqual(f["model_har"], cs["model_har"][j], places=9)        # the recent-vol baseline too
            self.assertEqual(f["har"]["har_p_above_mid"], cs["har"][f"har_p_above_{('low', 'mid', 'high')[j]}"])
        self.assertEqual(f["state"]["expected_move_to_expiry_bps"], cs["state"]["expected_move_to_expiry_bps"])


if __name__ == "__main__":
    unittest.main()
