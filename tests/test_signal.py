"""Jev v2: the live signal sees exactly the lab's features, decodes like the lab, goes stale on time; the policy;
and the engine in signal mode with resting maker entries."""

from __future__ import annotations

import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix import lab  # noqa: E402
from helix.config import load_config  # noqa: E402
from helix.engine import Engine, StepClock  # noqa: E402
from helix.jev import JevResult, MockJev, normalize_answers  # noqa: E402
from helix.jev_signal import JevSignal  # noqa: E402
from helix.journal import Journal  # noqa: E402
from helix.market import Candle  # noqa: E402
from helix.policy import signal_policy  # noqa: E402
from helix.simfeed import SimPerpFeed  # noqa: E402

MIN = 1_791_219_600   # a minute (and hour) boundary


def walk(n, step, end, seed):
    rng, px, out = random.Random(seed), 100.0, []
    for k in range(n):
        o = px
        px *= math.exp(rng.gauss(0, 0.001 * math.sqrt(step / 60)))
        out.append(Candle(end - (n - k) * step, o, max(o, px), min(o, px), px, 1 + rng.random()))
    return out


class FakeRest:
    def __init__(self, end):
        self.c1, self.c5 = walk(400, 60, end, 1), walk(lab.C5_NEED + 400, 300, end, 2)

    def candles(self, coin, interval, minutes, now):
        src, step = (self.c1, 60) if interval == "1m" else (self.c5, 300)
        return [c for c in src if c.ts >= now - minutes * 60 and c.ts + step <= now]

    def asset_ctxs(self):
        return {"BTC": {"funding": 0.0000125, "premium": -0.0002}}


class FakeJev:
    """Answers the score question with most weight on level `lvl` (0 = stretched far up ... 4 = far down)."""

    def __init__(self, lvl=4):
        self.lvl, self.states = lvl, []

    def ask(self, state, questions):
        self.states.append(state)
        raw = {}
        for qid, q in questions.items():
            if q["type"] == "score":
                raw[qid] = {"type": "score", "probabilities": {str(i): (0.8 if i == self.lvl else 0.05) for i in range(5)}}
            else:
                raw[qid] = {"type": "noul", "noul": 0.1}
        return JevResult("ok", "jev-1.13.0", 120.0, normalize_answers(raw, questions), {"input_tokens": 900})


def signal_cfg():
    cfg = load_config(None)
    cfg["market"]["coins"] = ["BTC"]
    cfg["signal"].update(enabled=True)
    return cfg


class SignalTests(unittest.TestCase):
    def test_live_state_equals_lab_state_and_decodes(self):
        now = [MIN + 5.0]
        rest, jev = FakeRest(MIN), FakeJev(lvl=4)
        sig = JevSignal(signal_cfg(), jev, rest=rest, clock=lambda: now[0])
        sig.step()
        s = sig.fresh("BTC", now[0])
        self.assertIsNotNone(s)
        self.assertEqual(s["ts"], MIN)
        snap = lab.direction_snapshot("BTC", rest.candles("BTC", "1m", 250, now[0]),
                                      rest.candles("BTC", "5m", 99999, now[0]), 0.125, -2.0)
        self.assertEqual(jev.states[0], snap["state"])            # this prompt uses no extra blocks
        # 0.8 on "stretched far downward" (value 0.56), 0.05 on each other level (0.44, 0.47, 0.50, 0.53)
        self.assertAlmostEqual(s["p_up"], (0.8 * 0.56 + 0.05 * (0.44 + 0.47 + 0.50 + 0.53)) / 1.0, places=4)
        self.assertTrue(s["class"].startswith("Stretched far downward"))
        self.assertEqual(s["sigma_bps"], snap["state"]["expected_move_15m_bps"])   # candle-based, for risk sizing
        sig.step()                                                 # same minute: no second call
        self.assertEqual(sig.calls, 1)
        now[0] = MIN + 5 + 200                                     # past max_age_s with no new candle
        self.assertIsNone(sig.fresh("BTC", now[0]))

    def test_funding_is_the_last_settled_record_like_the_backtest(self):
        class FundedRest(FakeRest):
            def post(self, body):
                assert body["type"] == "fundingHistory"
                return [{"time": (MIN - 7200) * 1000, "fundingRate": "0.00001", "premium": "0.0001"},
                        {"time": (MIN - 3600) * 1000, "fundingRate": "0.00003", "premium": "-0.0004"}]
        jev = FakeJev()
        sig = JevSignal(signal_cfg(), jev, rest=FundedRest(MIN), clock=lambda: MIN + 5.0)
        sig.step()
        self.assertAlmostEqual(jev.states[0]["funding_hr_bps"], 0.3)    # 0.00003 settled, not the 0.0000125 live ctx
        self.assertAlmostEqual(jev.states[0]["premium_bps"], -4.0)

    def test_waits_for_the_closed_candle(self):
        now = [MIN + 5.0]
        rest, jev = FakeRest(MIN - 60), FakeJev()                  # newest candle is a minute old
        sig = JevSignal(signal_cfg(), jev, rest=rest, clock=lambda: now[0])
        sig.step()
        self.assertEqual(sig.calls, 0)                             # maybe not published yet: wait
        self.assertIsNone(sig.fresh("BTC", now[0]))
        now[0] = MIN + 12.0                                        # still missing: a minute with no trades
        sig.step()
        self.assertEqual(sig.calls, 1)
        self.assertEqual(jev.states[0]["return_1m_bps"], 0.0)      # flat candle, as the exchange records it

    def test_kept_signal_keeps_its_own_minute(self):
        now = [MIN + 5.0]
        rest = FakeRest(MIN)
        sig = JevSignal(signal_cfg(), FakeJev(), rest=rest, clock=lambda: now[0])
        sig.step()
        self.assertEqual(sig.latest["BTC"]["signal_ts"], MIN)

        def boom(*a, **k):
            raise ConnectionError("REST timeout")
        rest.candles = boom                                        # next minute: a transient data error
        now[0] = MIN + 75.0
        sig.step()
        kept = sig.fresh("BTC", now[0])
        self.assertEqual(kept["ts"], MIN + 60)                     # re-stamped so it isn't retried every 2 s ...
        self.assertEqual(kept["signal_ts"], MIN)                   # ... but the forecast's minute is unchanged
        pol = load_config(None)["policy"]
        ok = {"regime": {"probabilities": {"crisis": 0.0}}, "liquidity_stressed": {"p": 0.1}}
        act = signal_policy(dict(kept, p_up=0.6, momentum_regime=False), ok, 0, pol, 0.54, True)
        self.assertEqual(act.kind, "ENTER_LONG")
        self.assertEqual(act.meta["signal_ts"], MIN)               # so the exit clock starts at the forecast minute

    def test_rest_gap_too_long_is_not_filled(self):
        rest = FakeRest(MIN - 600)                                 # ten minutes missing: a data problem, not quiet
        sig = JevSignal(signal_cfg(), FakeJev(), rest=rest, clock=lambda: MIN + 12.0)
        sig.step()
        self.assertEqual(sig.calls, 0)
        self.assertEqual(sig.latest["BTC"]["status"], "no candle")

    def test_stream_with_a_gap_falls_back_to_rest(self):
        rest = FakeRest(MIN)
        gappy = [c for c in rest.c1 if c.ts != MIN - 30 * 60]      # the stream missed one minute
        calls = []

        class CountingRest(FakeRest):
            def candles(self, coin, interval, minutes, at):
                calls.append(interval)
                return super().candles(coin, interval, minutes, at)
        rest = CountingRest(MIN)
        now = [MIN + 5.0]
        sig = JevSignal(signal_cfg(), FakeJev(), rest=rest, clock=lambda: now[0], candle_source=lambda c, t: gappy)
        sig.step()
        self.assertEqual(sig.calls, 0)                             # first a few seconds for the stream
        now[0] = MIN + 11.0
        sig.step()
        self.assertEqual(sig.calls, 1)
        self.assertIn("1m", calls)                                 # then the REST candles

    def test_aggregate_5m_only_complete_bars(self):
        from helix.jev_signal import aggregate_5m
        c1 = walk(30, 60, MIN, 3)                                  # 30 one-minute candles ending at MIN
        start = MIN - 30 * 60
        bars = aggregate_5m(c1, start - 300, MIN)
        self.assertEqual([b.ts for b in bars], [start + 300 * k for k in range(6)])
        five = c1[:5]
        self.assertEqual((bars[0].open, bars[0].close), (five[0].open, five[-1].close))
        self.assertEqual(bars[0].high, max(c.high for c in five))
        self.assertAlmostEqual(bars[0].volume, sum(c.volume for c in five))
        holed = [c for c in c1 if c.ts != start + 660]             # a missing minute (in bar 3) ends the run
        self.assertEqual(len(aggregate_5m(holed, start - 300, MIN)), 2)
        self.assertEqual(aggregate_5m(c1, start - 300, MIN - 1), bars[:5])   # the forming bar is never included

    def test_regime_hysteresis(self):
        sig = JevSignal(signal_cfg(), FakeJev(), rest=FakeRest(MIN), clock=lambda: MIN + 5.0)
        self.assertFalse(sig._regime("BTC", 0.50))
        self.assertTrue(sig._regime("BTC", 0.505))
        self.assertTrue(sig._regime("BTC", 0.495))                 # stays in until below 0.49
        self.assertFalse(sig._regime("BTC", 0.489))
        self.assertFalse(sig._regime("BTC", 0.50))

    def test_funding_cutoff_is_the_minute(self):
        class FundedRest(FakeRest):
            def post(self, body):
                return [{"time": (MIN - 3600) * 1000, "fundingRate": "0.00001", "premium": "0"},
                        {"time": (MIN + 1) * 1000, "fundingRate": "0.00009", "premium": "0"}]   # after the state
        jev = FakeJev()
        sig = JevSignal(signal_cfg(), jev, rest=FundedRest(MIN), clock=lambda: MIN + 5.0)
        sig.step()
        self.assertAlmostEqual(jev.states[0]["funding_hr_bps"], 0.1)

    def test_policy(self):
        pol = load_config(None)["policy"]
        ok = {"regime": {"probabilities": {"crisis": 0.0}}, "liquidity_stressed": {"p": 0.1}}
        sig = lambda p: {"p_up": p, "horizon_min": 15, "class": "Stretched far downward"}  # noqa: E731
        self.assertEqual(signal_policy(sig(0.56), ok, 0, pol, 0.54, True).kind, "ENTER_LONG")
        self.assertEqual(signal_policy(sig(0.44), ok, 0, pol, 0.54, True).kind, "ENTER_SHORT")
        self.assertEqual(signal_policy(sig(0.44), ok, 0, pol, 0.54, False).kind, "STAND_ASIDE")
        self.assertEqual(signal_policy(sig(0.52), ok, 0, pol, 0.54, True).kind, "STAND_ASIDE")
        self.assertEqual(signal_policy(None, ok, 0, pol, 0.54, True).kind, "STAND_ASIDE")
        thin = dict(ok, liquidity_stressed={"p": 0.9})
        self.assertEqual(signal_policy(sig(0.6), thin, 0, pol, 0.54, True).kind, "STAND_ASIDE")
        crisis = dict(ok, regime={"probabilities": {"crisis": 0.9}})
        self.assertEqual(signal_policy(sig(0.6), crisis, 1, pol, 0.54, True).kind, "EXIT")
        self.assertEqual(signal_policy(sig(0.3), ok, 1, pol, 0.54, True).kind, "HOLD")   # no early signal exits
        self.assertEqual(signal_policy(sig(0.44), ok, 0, pol, 0.56, True).kind, "ENTER_SHORT")   # 1 - 0.56 != 0.44 in floats
        self.assertEqual(signal_policy(sig(0.56), ok, 0, pol, 0.56, True).kind, "ENTER_LONG")
        go = signal_policy(dict(sig(0.56), ts=MIN, sigma_bps=31.5), ok, 0, pol, 0.54, True)
        self.assertEqual(go.meta, {"signal_ts": MIN, "sigma_bps": 31.5})     # execution gets the signal minute + sigma
        trending = dict(sig(0.6), momentum_regime=True, continuation=0.53)    # regime guard: sit a trending coin out
        act = signal_policy(trending, ok, 0, pol, 0.54, True)
        self.assertEqual(act.kind, "STAND_ASIDE")
        self.assertIn("Momentum regime", act.reasons[0])

    def test_stream_candles_are_used_and_regime_is_flagged(self):
        now = [MIN + 5.0]
        rest, jev = FakeRest(MIN), FakeJev(lvl=2)
        calls = []

        class CountingRest(FakeRest):
            def candles(self, coin, interval, minutes, at):
                calls.append(interval)
                return super().candles(coin, interval, minutes, at)
        rest = CountingRest(MIN)
        stream = lambda coin, at: rest.c1                     # noqa: E731  (the feed's closed 1-minute candles)
        sig = JevSignal(signal_cfg(), jev, rest=rest, clock=lambda: now[0], candle_source=stream)
        sig.step()
        s = sig.fresh("BTC", now[0])
        self.assertIsNotNone(s)
        self.assertNotIn("1m", calls)                          # no REST call for 1-minute candles
        self.assertIn("continuation", s)
        self.assertEqual(s["momentum_regime"], s["continuation"] > 0.5)


class FakeSignal:
    enabled, entry_prob = True, 0.54

    def __init__(self, clock):
        self.clock, self.latest = clock, {}

    def fresh(self, coin, now):
        p = {"BTC": 0.6, "ETH": 0.4}.get(coin, 0.5)
        rec = {"ts": int(now // 60) * 60, "made": now, "status": "ok", "p_up": p, "horizon_min": 15, "class": "test",
               "sigma_bps": 12.0}
        self.latest[coin] = rec
        return rec

    def to_dict(self):
        return {"enabled": True, "status": "ok", "prompt": "fake", "latest": self.latest}


class EngineSignalModeTests(unittest.TestCase):
    def test_signal_drives_maker_entries(self):
        cfg = load_config(None)
        cfg["paper"]["entry_mode"] = "maker"
        cfg["market"]["horizon_s"] = 900
        cfg["risk"]["max_hold_s"] = 900
        clock = StepClock(MIN)
        feed = SimPerpFeed(cfg["market"]["coins"], clock.now, seed=5)
        tmp = tempfile.mkdtemp()
        j = Journal(Path(tmp) / "t.db")
        try:
            eng = Engine(cfg, feed, MockJev(), j, None, clock, tmp, demo=True, persist=False, signal=FakeSignal(clock))
            self.assertTrue(eng.signal_mode)
            next_dec, end, saw_pending, clocks = clock.t, clock.t + 900, False, []
            while clock.t < end:
                clock.t += eng.monitor_interval
                eng.monitor(clock.t)
                if clock.t >= next_dec:
                    eng.decide(clock.t)
                    next_dec += eng.interval
                    saw_pending = saw_pending or bool(eng.books["jev"].pending)
                    clocks += [(p.clock_start, p.opened) for p in eng.books["jev"].positions.values()]
            self.assertTrue(clocks)
            for start, opened in clocks:                                # the holding clock starts at the signal minute
                self.assertEqual(start % 60, 0)
                self.assertLessEqual(start, opened)
                self.assertLess(opened - start, 180)
            s = eng.snapshot(fresh=True)
            jb = s["books"]["jev"]
            self.assertTrue(saw_pending, "entries should rest as maker orders first")
            self.assertGreater(jb["maker"]["placed"], 0)
            self.assertEqual(jb["maker"]["filled"], jb["funnel"]["fills"])
            sides = {p["coin"]: p["side"] for p in jb["positions"]}
            sides.update({t["coin"]: t["side"] for t in eng.books["jev"].trades})
            self.assertNotIn("SOL", sides)                         # P(up) 0.5: never traded
            for coin, side in sides.items():
                self.assertEqual(side, 1 if coin == "BTC" else -1)
            for t in eng.books["jev"].trades:
                self.assertTrue(t["entry_reason"].startswith("Maker entry"))
            d = s["markets"]["BTC"]["decision"]
            self.assertEqual(d["answers"]["direction"]["source"], "jev_v2")
            self.assertAlmostEqual(d["answers"]["direction"]["probabilities"]["up"], 0.6)
            self.assertIn("direction_5s", d["answers"])
            self.assertEqual(s["config"]["entry_mode"], "maker")
            self.assertLess(s["config"]["round_trip_bps"], 10)
        finally:
            j.close()


if __name__ == "__main__":
    unittest.main()
