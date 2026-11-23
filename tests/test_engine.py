"""Engine tests on the simulated market: every rung of the fallback ladder, the Jev budget and
circuit breaker, the kill switch, data outages, persistence, and the HTTP API."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import load_config  # noqa: E402
from helix.engine import C_LATE, C_PAUSED, Engine, StepClock  # noqa: E402
from helix.jev import JevResult, MockJev  # noqa: E402
from helix.journal import Journal  # noqa: E402
from helix.market import DataError  # noqa: E402
from helix.simfeed import SimPerpFeed  # noqa: E402

T0 = 1_791_200_000.0
_JOURNALS: list[Journal] = []


def tearDownModule():
    for j in _JOURNALS:
        j.close()


class SlowJev:
    provider, model, pinned = "typesafe", "jev-1.13.0", True

    def ask(self, state, questions):
        time.sleep(1.2)  # longer than the 800 ms deadline
        return JevResult("ok", "jev-1.13.0", 1200, {})


class BrokenJev:
    provider, model, pinned = "typesafe", "jev-1.13.0", True

    def __init__(self):
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        return JevResult("error", latency_ms=50, error="HTTP 500")


class PricedJev(MockJev):
    """Mock answers but reports real-looking usage, so the budget meter moves."""

    provider = "typesafe"

    def ask(self, state, questions):
        res = MockJev.ask(self, state, questions)
        res.status, res.model, res.usage = "ok", "jev-1.13.0", {"input_tokens": 1_000_000}
        return res


class FlakyFeed:
    def __init__(self, inner):
        self.inner, self.fail, self.name = inner, False, "flaky"

    def start(self):
        pass

    def stop(self):
        pass

    def view(self, coin):
        if self.fail:
            raise DataError("connection reset")
        return self.inner.view(coin)

    def quote(self, coin):
        if self.fail:
            raise DataError("connection reset")
        return self.inner.quote(coin)

    def health(self):
        return self.inner.health()


def make(jev=None, cfg=None, persist=False, data_dir=None, flaky=False, seed=5):
    cfg = cfg or load_config(None)
    clock = StepClock(T0)
    feed = SimPerpFeed(cfg["market"]["coins"], clock.now, seed=seed)
    if flaky:
        feed = FlakyFeed(feed)
    tmp = data_dir or tempfile.mkdtemp()
    # the journal prunes by the REAL clock (72 h); the test's simulated T0 is a fixed past date, so without this the
    # pruner started deleting the test's own rows once T0 was more than 72 h ago (2026-10-08)
    j = Journal(Path(tmp) / "t.db", retain_hours=1e6)
    _JOURNALS.append(j)
    eng = Engine(cfg, feed, jev, j, None, clock, tmp, demo=True, persist=persist)
    return eng, clock


def simulate(eng, clock, seconds: float):
    next_dec = clock.t
    end = clock.t + seconds
    while clock.t < end:
        clock.t += eng.monitor_interval
        eng.monitor(clock.t)
        if clock.t >= next_dec:
            eng.decide(clock.t)
            next_dec += eng.interval


class LoopTests(unittest.TestCase):
    def test_full_loop_with_mock_jev(self):
        eng, clock = make(MockJev())
        simulate(eng, clock, 300)
        s = eng.snapshot(fresh=True)
        self.assertGreaterEqual(s["engine"]["decisions_per_min"], 170)  # 3 coins x ~60 per minute
        self.assertGreater(s["calibration"]["n"], 300)
        for coin, m in s["markets"].items():
            self.assertTrue(m["ready"], coin)
            self.assertLess(m["state_tokens"], 400)
            self.assertEqual(len(m["book"]["bids"]), 20)
            self.assertGreater(len(m["mids"]["v"]), 250)
            self.assertIsNotNone(m["decision"]["answers"])
        for name, b in s["books"].items():
            f = b["funnel"]
            self.assertGreaterEqual(f["decisions"], f["signals"])
            self.assertGreaterEqual(f["signals"], f["risk_pass"])
            self.assertGreaterEqual(f["risk_pass"], f["fills"])
            self.assertAlmostEqual(b["equity"], b["start"] + b["pnl"], places=1)
        self.assertEqual(len(s["pipeline"]), 9)
        rows = eng.journal.query("SELECT COUNT(*) FROM decisions")[0][0]
        self.assertEqual(rows, s["engine"]["decisions_total"])

    def test_late_answers_hold(self):
        eng, clock = make(SlowJev())
        simulate(eng, clock, 4)
        s = eng.snapshot(fresh=True)
        self.assertEqual(s["books"]["jev"]["funnel"]["signals"], 0)
        d = s["markets"]["BTC"]["decision"]
        self.assertEqual(d["jev_status"], "late")
        self.assertEqual(d["jev_action"]["kind"], "HOLD")
        self.assertEqual(s["markets"]["BTC"]["ribbon"][-1][3], C_LATE)

    def test_no_key_runs_fallback_rules(self):
        eng, clock = make(None)
        simulate(eng, clock, 200)
        s = eng.snapshot(fresh=True)
        self.assertEqual(s["markets"]["ETH"]["decision"]["jev_status"], "offline")
        self.assertEqual(s["markets"]["ETH"]["decision"]["jev_action"]["source"], "fallback")
        j, b = s["books"]["jev"], s["books"]["baseline"]
        self.assertEqual(j["trades"], b["trades"])  # same rules, same data, same costs
        self.assertAlmostEqual(j["equity"], b["equity"], places=2)

    def test_circuit_breaker(self):
        jev = BrokenJev()
        eng, clock = make(jev)
        simulate(eng, clock, 160)
        s = eng.snapshot(fresh=True)
        self.assertEqual(s["markets"]["SOL"]["decision"]["jev_action"]["source"], "fallback")
        self.assertLess(jev.calls, 3 * 160 / 2)  # paused for 30 s after every 5 errors in a row
        self.assertTrue(any("in a row" in e["text"] for e in s["events"]))

    def test_daily_budget(self):
        cfg = load_config(None)
        cfg["jev"]["daily_budget_usd"] = 0.2  # each fake call reports 1M tokens = $0.042
        eng, clock = make(PricedJev(), cfg=cfg)
        simulate(eng, clock, 150)
        s = eng.snapshot(fresh=True)
        self.assertTrue(s["jev"]["budget_hit"])
        self.assertLessEqual(s["jev"]["calls"], 9)
        self.assertEqual(s["markets"]["BTC"]["decision"]["jev_status"], "budget")

    def test_kill_switch(self):
        eng, clock = make(MockJev(), seed=9)
        simulate(eng, clock, 400)
        eng.kill()
        s = eng.snapshot(fresh=True)
        self.assertEqual(s["engine"]["status"], "paused")
        for b in s["books"].values():
            self.assertEqual(b["positions"], [])
        fills = {n: b["funnel"]["fills"] for n, b in s["books"].items()}
        simulate(eng, clock, 120)
        s = eng.snapshot(fresh=True)
        self.assertEqual({n: b["funnel"]["fills"] for n, b in s["books"].items()}, fills)
        self.assertEqual(s["markets"]["BTC"]["ribbon"][-1][3], C_PAUSED)
        eng.resume()
        self.assertEqual(eng.snapshot(fresh=True)["engine"]["status"], "running")

    def test_data_outage_and_recovery(self):
        eng, clock = make(MockJev(), flaky=True)
        simulate(eng, clock, 150)
        eng.feed.fail = True
        simulate(eng, clock, 12)
        self.assertGreaterEqual(eng.errors_in_row, 10)
        self.assertEqual(eng.snapshot(fresh=True)["engine"]["status"], "degraded")
        eng.feed.fail = False
        simulate(eng, clock, 3)
        self.assertEqual(eng.errors_in_row, 0)

    def test_persistence(self):
        tmp = tempfile.mkdtemp()
        eng, clock = make(MockJev(), persist=True, data_dir=tmp, seed=9)
        simulate(eng, clock, 400)
        eng._persist()
        eq = eng.snapshot(fresh=True)["books"]["jev"]["equity"]
        eng2 = Engine(eng.cfg, eng.feed, MockJev(), None, None, clock, tmp, demo=True, persist=True)
        marks = {c: eng.cs[c].last.mid for c in eng.coins}
        self.assertAlmostEqual(eng2.books["jev"].equity(marks), eq, delta=abs(eq) * 0.01)


class SizingTests(unittest.TestCase):
    @staticmethod
    def trades(pnls):
        return [{"pnl": p, "pnl_bps": p * 10, "book": "jev"} for p in pnls]

    def test_streak_boost_needs_wins_and_a_profitable_record(self):
        eng, _ = make(MockJev())
        jb = eng.books["jev"]
        eng.sizing.update(manual=2.0, streak=True)
        jb.trades = self.trades([1.0] * 3)                       # 3 wins but only 3 trades: gate closed (needs 30)
        s = eng.sizing_info()
        self.assertEqual((s["wins_in_row"], s["gate_ok"], s["boost"], s["effective"]), (3, False, 1.0, 2.0))
        jb.trades = self.trades([1.0] * 27 + [-0.5, 1.0, 1.0, 1.0])   # profitable record, 3 wins in a row
        s = eng.sizing_info()
        self.assertTrue(s["gate_ok"])
        self.assertEqual((s["boost"], s["effective"]), (2.0, 4.0))   # 3 wins -> 2x, times manual 2x
        jb.trades = self.trades([1.0] * 35)                     # long run: boost capped at 3x, total capped at 5x
        s = eng.sizing_info()
        self.assertEqual((s["boost"], s["effective"]), (3.0, 5.0))
        jb.trades = self.trades([1.0] * 35 + [-1.0])            # any loss resets
        self.assertEqual(eng.sizing_info()["boost"], 1.0)
        jb.trades = self.trades([-3.0] * 28 + [1.0, 1.0, 1.0])  # a run inside a losing record: no boost
        s = eng.sizing_info()
        self.assertEqual((s["wins_in_row"], s["gate_ok"], s["boost"]), (3, False, 1.0))
        # the audit's case: 26 small wins, one big boosted loss, 3 small wins. Average bps is positive (+1.5) but the
        # last 30 trades lost $155, so the gate must stay closed
        jb.trades = ([{"pnl": 5.0, "pnl_bps": 5.0}] * 26 + [{"pnl": -300.0, "pnl_bps": -100.0}]
                     + [{"pnl": 5.0, "pnl_bps": 5.0}] * 3)
        s = eng.sizing_info()
        self.assertEqual((s["wins_in_row"], s["gate_pnl"], s["gate_ok"], s["boost"]), (3, -155.0, False, 1.0))
        eng.r = dict(eng.r, streak=dict(eng.r["streak"], gate_trades=0))   # 0 = no gate
        self.assertEqual(eng.sizing_info()["boost"], 2.0)
        eng.sizing["streak"] = False
        jb.trades = self.trades([1.0] * 35)
        self.assertEqual(eng.sizing_info()["effective"], 2.0)
        eng.sizing.update(manual=5.0, streak=True)                  # at the ceiling the boost is clipped, and says so
        s = eng.sizing_info()
        self.assertEqual((s["boost"], s["effective"], s["capped"]), (3.0, 5.0, True))

    def test_lowering_the_bet_cancels_resting_entries_and_bad_input_is_refused(self):
        eng, clock = make(MockJev())
        simulate(eng, clock, 130)
        eng.set_sizing(multiplier=4.0)
        v = eng.cs["BTC"].last
        eng.books["jev"].place_entry(v, 1, 1.0, 30.0, 50.0, clock.t, 60)
        eng.set_sizing(multiplier=4.0)                              # not lower: order kept
        self.assertIn("BTC", eng.books["jev"].pending)
        eng.set_sizing(multiplier=1.0)
        self.assertEqual(eng.books["jev"].pending, {})
        for bad in ({"streak": "false"}, {"multiplier": float("nan")}, {"multiplier": 0.0}):
            with self.assertRaises(ValueError):
                eng.set_sizing(**bad)
        self.assertEqual(eng.sizing, {"manual": 1.0, "streak": False})

    def test_bad_saved_sizing_keeps_the_restored_books(self):
        tmp = tempfile.mkdtemp()
        eng, clock = make(MockJev(), persist=True, data_dir=tmp, seed=9)
        simulate(eng, clock, 300)
        eng._persist()
        path = Path(tmp) / "demo_state.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sizing"] = {"manual": "abc", "streak": "false"}
        path.write_text(json.dumps(data), encoding="utf-8")
        eng2 = Engine(eng.cfg, eng.feed, MockJev(), None, None, clock, tmp, demo=True, persist=True)
        self.assertEqual(eng2.books["jev"].cash, eng.books["jev"].cash)          # books restored
        self.assertEqual(eng2.sizing, {"manual": 1.0, "streak": False})           # bad values ignored
        self.assertTrue(any("Ignored saved bet multiplier" in e["text"] for e in eng2.events))

    def test_set_sizing_validates_persists_and_only_scales_jev(self):
        tmp = tempfile.mkdtemp()
        eng, clock = make(MockJev(), persist=True, data_dir=tmp)
        with self.assertRaises(ValueError):
            eng.set_sizing(multiplier=50)
        info = eng.set_sizing(multiplier=3.0, streak=True)
        self.assertEqual((info["manual"], info["streak_enabled"], info["effective"]), (3.0, True, 3.0))
        eng2 = Engine(eng.cfg, eng.feed, MockJev(), None, None, clock, tmp, demo=True, persist=True)
        self.assertEqual(eng2.sizing, {"manual": 3.0, "streak": True})
        simulate(eng, clock, 600)
        s = eng.snapshot(fresh=True)
        self.assertEqual(s["sizing"]["effective"], 3.0)
        eng.journal.flush()
        rows = eng.journal.query("SELECT book, risk_json FROM signals WHERE risk_json IS NOT NULL")
        jev = [r for b, r in rows if b == "jev"]
        base = [r for b, r in rows if b == "baseline"]
        self.assertTrue(jev and base, "both books should have evaluated entries")
        self.assertTrue(all('"Bet size"' in r and '"3.00x"' in r for r in jev))   # every Jev entry sized at 3x
        self.assertFalse(any('"Bet size"' in r for r in base))                   # the baseline never is


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerTests(unittest.TestCase):
    def test_http_api(self):
        import requests
        import uvicorn

        from helix.server import create_app

        eng, clock = make(MockJev())
        simulate(eng, clock, 130)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(eng, run_loop=False, token="s3cret"), host="127.0.0.1",
                                               port=port, log_level="error"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        try:
            self.assertEqual(requests.get(f"{base}/api/state", timeout=5).status_code, 401)
            h = {"x-helix-token": "s3cret"}
            body = requests.get(f"{base}/api/state", headers=h, timeout=5).json()
            self.assertIn("books", body)
            self.assertEqual(body["venue"], "hyperliquid")
            page = requests.get(f"{base}/", timeout=5)
            self.assertEqual(page.status_code, 200)
            self.assertIn("HELIX", page.text)
            requests.post(f"{base}/api/kill", headers=h, timeout=5)
            self.assertEqual(requests.get(f"{base}/api/state", headers=h, timeout=5).json()["engine"]["status"], "paused")
            ok = requests.post(f"{base}/api/sizing", headers=h, json={"multiplier": 2.5}, timeout=5).json()
            self.assertEqual((ok["ok"], ok["effective"]), (True, 2.5))
            bad = requests.post(f"{base}/api/sizing", headers=h, json={"multiplier": 99}, timeout=5)
            self.assertEqual(bad.status_code, 400)
            for body in ({"streak": "false"}, {"multiplier": True}, [1, 2], 5):
                self.assertEqual(requests.post(f"{base}/api/sizing", headers=h, json=body, timeout=5).status_code, 400, body)
            self.assertEqual(requests.post(f"{base}/api/sizing", json={"multiplier": 2}, timeout=5).status_code, 401)
            # another website in the same browser: refused even with the right token
            evil = {**h, "Origin": "https://evil.example"}
            self.assertEqual(requests.post(f"{base}/api/sizing", headers=evil, data='{"multiplier": 4.5}', timeout=5).status_code, 403)
            same = {**h, "Origin": f"http://127.0.0.1:{port}"}
            self.assertEqual(requests.post(f"{base}/api/sizing", headers=same, json={"multiplier": 2.5}, timeout=5).status_code, 200)
            requests.post(f"{base}/api/resume", headers=h, timeout=5)
            self.assertEqual(requests.get(f"{base}/api/state?token=s3cret", timeout=5).json()["engine"]["status"],
                             "running")
        finally:
            server.should_exit = True
            th.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
