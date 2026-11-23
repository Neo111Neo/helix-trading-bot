"""HIP-4 tape recorder: records every message kind compactly, dedupes trades across reconnects, rotates hourly
files and gzips the closed ones, reads a torn last line, and remembers settlements across restarts."""

from __future__ import annotations

import copy
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.config import DEFAULTS  # noqa: E402
from helix.hip4_tape import HIP4Tape, read_tape, tape_files  # noqa: E402

T0 = 1_791_300_000.0     # 2026-10-06 15:20 UTC


class FakeRest:
    def __init__(self):
        self.outcomes = [{"outcome": 9204, "name": "template:binaryPrice", "venue": "out",
                          "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:85000|time:20261006-1530"}]
        self.settled = {}

    def post(self, body):
        if body["type"] == "outcomeMeta":
            return {"outcomes": self.outcomes}
        if body["type"] == "settledOutcome":
            return self.settled.get(body["outcome"])


def tape(d, clock, rest=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["hip4_tape"]["enabled"] = True
    return HIP4Tape(cfg, Path(d), rest=rest or FakeRest(), clock=clock)


class TapeTests(unittest.TestCase):
    def test_records_dedupes_rotates_and_reads(self):
        with tempfile.TemporaryDirectory() as d:
            now = [T0]
            tp = tape(d, lambda: now[0])
            tp.handle({"channel": "bbo", "data": {"coin": "BTC", "time": 1, "bbo": [{"px": "85000", "sz": "1"}, {"px": "85001", "sz": "2"}]}})
            tp.handle({"channel": "l2Book", "data": {"coin": "#92040", "time": 2, "levels": [[{"px": "0.4", "sz": "5"}], [{"px": "0.45", "sz": "3"}]]}})
            tr = {"coin": "#92040", "side": "A", "px": "0.41", "sz": "7", "time": 3, "tid": 99}
            tp.handle({"channel": "trades", "data": [tr, tr]})
            tp.handle({"channel": "trades", "data": [{"coin": "BTC", "side": "B", "px": "85001", "sz": "0.1", "time": 4, "tid": 5}]})
            now[0] = T0 + 3600                                     # next hour: new file, old one gzipped
            tp.handle({"channel": "trades", "data": [tr]})         # same trade again after a reconnect: dropped
            tp.handle({"channel": "bbo", "data": {"coin": "BTC", "time": 9, "bbo": [None, {"px": "85002", "sz": "1"}]}})
            tp.close()
            for _ in range(200):                                   # until the gzip thread has finished (and let go)
                if list(Path(d, "tape").glob("*.gz")) and not Path(d, "tape", "20261006-15.jsonl").exists():
                    break
                time.sleep(0.05)
            files = tape_files(Path(d, "tape"))
            self.assertEqual([f.name for f in files], ["20261006-15.jsonl.gz", "20261006-16.jsonl"])
            first = list(read_tape(files[0]))
            self.assertEqual([r["k"] for r in first], ["pb", "ob", "ot", "pt"])
            self.assertEqual(first[1]["b"], [[0.4, 5.0]])
            self.assertEqual(first[2]["tid"], 99)
            second = list(read_tape(files[1]))
            self.assertEqual([r["k"] for r in second], ["pb"])
            self.assertIsNone(second[0]["b"])
            with open(files[1], "a", encoding="utf-8") as fh:
                fh.write('{"k":"pb","c":"BT')                      # torn line from a crash
            self.assertEqual(len(list(read_tape(files[1]))), 1)
            self.assertEqual(tp.counts["ot"], 1)

    def test_listing_delisting_and_settlement_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            now = [T0]
            rest = FakeRest()
            tp = tape(d, lambda: now[0], rest)
            tp._refresh_meta(now[0])
            self.assertIn(9204, tp.markets)
            self.assertIn(("l2Book", "#92041"), tp._subs())        # No side too
            rest.outcomes = []
            now[0] += 600
            tp._refresh_meta(now[0])                               # delisted: settling
            rest.settled[9204] = {"settleFraction": "1.0"}
            now[0] += 120
            tp._settle(now[0])
            self.assertEqual(tp.settled[9204], 1.0)
            tp.close()
            again = tape(d, lambda: now[0], rest)
            self.assertEqual(again.settled.get(9204), 1.0)
            kinds = [r["k"] for p in tape_files(Path(d, "tape")) for r in read_tape(p)]
            self.assertEqual(kinds.count("m"), 2)
            self.assertEqual(kinds.count("x"), 1)


class AllCoinsTests(unittest.TestCase):
    def test_all_records_every_underlying_and_its_perp(self):
        with tempfile.TemporaryDirectory() as d:
            rest = FakeRest()
            rest.outcomes.append({"outcome": 9300, "name": "template:binaryPrice", "venue": "out",
                                  "description": "perp:xyz:SP500|priceDescription:SP500 mark|seconds:60|threshold:6700|time:20261006-1600"})
            cfg = copy.deepcopy(DEFAULTS)
            cfg["hip4_tape"].update(enabled=True, coins=["all"])
            tp = HIP4Tape(cfg, Path(d), rest=rest, clock=lambda: T0)
            tp._refresh_meta(T0)
            self.assertEqual(set(tp.markets), {9204, 9300})
            subs = tp._subs()
            self.assertIn(("bbo", "xyz:SP500"), subs)
            self.assertIn(("bbo", "BTC"), subs)
            self.assertIn(("l2Book", "#93001"), subs)
            self.assertEqual(tp.to_dict()["coins"], ["BTC", "xyz:SP500"])
            tp.close()


if __name__ == "__main__":
    unittest.main()
