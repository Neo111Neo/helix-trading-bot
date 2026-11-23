"""Live readiness: the shadow executor turns paper quotes into the exact Hyperliquid orders (post-only, whole contracts,
$10 minimum, asks we can't sell become No bids), only re-prices on a real move, and never has a way to send; the
paper maker obeys the same minimum; the go-live gate reports evidence and the operator's own confirmations."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix import hip4_live  # noqa: E402
from helix.config import DEFAULTS  # noqa: E402
from helix.golive import GoLiveGate  # noqa: E402
from helix.hip4_live import ShadowExecutor, order_action  # noqa: E402
from tests.test_hip4_mm import MKT, NOW, armed, maker  # noqa: E402


def live_cfg(**gate):
    c = copy.deepcopy(DEFAULTS)
    c["hip4_live"]["shadow"] = True
    c["hip4_live"]["gate"].update(gate)
    return c


class ShadowTests(unittest.TestCase):
    def test_quotes_become_exact_post_only_orders(self):
        mm = maker(quote_size=40)
        armed(mm, fair=0.50, bid=0.30, ask=0.70)                      # our 48c bid / 52c ask, 40 contracts
        sh = ShadowExecutor(live_cfg(), mm, None, clock=lambda: NOW)
        want, rejected = sh.desired()
        bid, ask = want[(9204, "bid")], want[(9204, "ask")]
        self.assertEqual((bid["coin"], bid["is_buy"], bid["px"], bid["sz"]), ("#92040", True, 0.48, 40))
        # no Yes held: the 52c ask goes in as a BID on the No book at 48c (same payoff)
        self.assertEqual((ask["coin"], ask["is_buy"], ask["px"], ask["asset"]), ("#92041", True, 0.48, 100_092_041))
        self.assertEqual(rejected, [])
        self.assertEqual(order_action(bid["asset"], True, 0.48, 40),
                         {"type": "order", "orders": [{"a": 100_092_040, "b": True, "p": "0.48", "s": "40", "r": False,
                                                       "t": {"limit": {"tif": "Alo"}}}], "grouping": "na"})

    def test_held_yes_is_sold_and_small_orders_are_rejected(self):
        mm = maker(quote_size=40)
        mm.pos["9204"] = {"coin": "BTC", "strike": 85_000, "expiry": MKT["expiry"], "venue": "out", "qty": 60.0,
                          "cash": -28.0, "fees": 0.0, "fills": 2, "volume": 60}
        armed(mm, fair=0.50, bid=0.30, ask=0.70)
        want, _ = ShadowExecutor(live_cfg(), mm, None, clock=lambda: NOW).desired()
        self.assertEqual((want[(9204, "ask")]["coin"], want[(9204, "ask")]["is_buy"]), ("#92040", False))
        mm2 = maker(quote_size=10)                                     # 10 contracts at 48c = $4.80 < $10
        armed(mm2, fair=0.50, bid=0.30, ask=0.70)
        want, rejected = ShadowExecutor(live_cfg(), mm2, None, clock=lambda: NOW).desired()
        self.assertEqual(want, {})
        self.assertTrue(all("minimum" in r["reason"] for r in rejected))

    def test_only_a_real_move_spends_budget(self):
        mm = maker(quote_size=40)
        armed(mm, fair=0.50, bid=0.30, ask=0.70)
        sh = ShadowExecutor(live_cfg(), mm, None, clock=lambda: NOW)
        first = sh.step(NOW)
        self.assertEqual(sorted(k for k, *_ in first), ["order", "order"])
        armed(mm, fair=0.505, bid=0.30, ask=0.70)                     # half a cent: below the 1c repost threshold
        self.assertEqual(sh.step(NOW + 1), [])
        armed(mm, fair=0.52, bid=0.30, ask=0.70)                      # 2c: re-price both
        self.assertEqual(sorted(k for k, *_ in sh.step(NOW + 2)), ["modify", "modify"])
        mm.quotes = {}
        self.assertEqual(sorted(k for k, *_ in sh.step(NOW + 3)), ["cancel", "cancel"])
        d = sh.to_dict()
        self.assertEqual(d["totals"], {"order": 2, "modify": 2, "cancel": 2})
        self.assertEqual(d["mode"], "SHADOW (nothing is sent)")

    def test_nothing_can_send(self):
        """The shadow module imports no networking, signing or exchange code, so it physically cannot place orders."""
        import ast
        tree = ast.parse(Path(hip4_live.__file__).read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
               {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - {"__future__", "json", "math", "threading", "time", "collections", "pathlib"}, set())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {"post", "send", "sign", "sign_message", "order", "cancel", "connect"})


class MinNotionalTests(unittest.TestCase):
    def test_paper_quotes_follow_the_live_minimum(self):
        mm = maker(min_notional=10, quote_size=20)
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        self.assertEqual((q["bid_sz"], q["ask_sz"]), (21, 21))          # 20 x 48c = $9.60 -> raised to 21 ($10.08)
        mm2 = maker(min_notional=10, quote_size=20, max_position=20)
        q = armed(mm2, fair=0.50, bid=0.30, ask=0.70)
        self.assertEqual((q["bid_sz"], q["ask_sz"]), (0, 0))            # no room for 21: not quoted at all


class StickyTests(unittest.TestCase):
    def test_small_moves_keep_the_resting_price_big_or_unsafe_ones_move_it(self):
        mm = maker(requote_threshold=0.02)
        q = armed(mm, fair=0.50, bid=0.30, ask=0.70)
        placed = q["placed_bid"]
        self.assertEqual((q["bid"], q["ask"]), (0.48, 0.52))
        q = armed(mm, fair=0.51, bid=0.30, ask=0.70)                  # target 49/53: within 2c, still safe -> keep
        self.assertEqual((q["bid"], q["ask"]), (0.48, 0.52))
        self.assertEqual(q["placed_bid"], placed)                      # and its place in the queue
        q = armed(mm, fair=0.535, bid=0.30, ask=0.70)                  # ask 52c is now within 1c of fair: unsafe
        self.assertAlmostEqual(q["ask"], 0.555)
        q = armed(mm, fair=0.47, bid=0.30, ask=0.70)                   # bid 48c above fair 47c: unsafe -> moves
        self.assertAlmostEqual(q["bid"], 0.45)


class GateTests(unittest.TestCase):
    def test_reports_evidence_and_confirmations(self):
        with tempfile.TemporaryDirectory() as d:
            mm = maker()
            g = GoLiveGate(live_cfg(days=7, settled=2, markout_n=2), mm, None, None, Path(d), clock=lambda: NOW)
            r = g.evaluate()
            self.assertFalse(r["ready"])
            self.assertFalse(r["live_enabled"])
            self.assertEqual({c["key"] for c in r["checks"]},
                             {"paper_days", "settled", "settled_pnl", "markout", "no_kills", "two_sided", "backtest", "shadow"})
            mm.since = NOW - 8 * 86400
            mm.settled = [{"pnl": 3.0, "ts": NOW, "id": i, "y": 1.0} for i in range(25)]
            mm.fills = [{"markout": 0.02, "ts": NOW, "id": i, "side": 1, "px": 0.4, "qty": 10} for i in range(3)]
            r = {c["key"]: c for c in g.evaluate()["checks"]}
            self.assertTrue(r["paper_days"]["ok"] and r["settled"]["ok"] and r["settled_pnl"]["ok"] and r["markout"]["ok"])
            self.assertTrue(r["no_kills"]["ok"])
            self.assertFalse(r["two_sided"]["ok"])                       # only ever bought Yes: not proven both ways
            self.assertFalse(r["backtest"]["ok"])
            g.confirm("legal", True)
            with self.assertRaises(ValueError):
                g.confirm("nope", True)
            again = GoLiveGate(live_cfg(), mm, None, None, Path(d), clock=lambda: NOW)
            self.assertTrue(next(c for c in again.evaluate()["confirmations"] if c["key"] == "legal")["ok"])


if __name__ == "__main__":
    unittest.main()
