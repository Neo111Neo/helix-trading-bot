"""HIP-4 live execution, SHADOW mode: turns the paper market maker's quotes into the exact orders Hyperliquid would
receive, keeps a model of what would be resting on the exchange, and logs every place / modify / cancel with its
exact payload. It NEVER signs and NEVER sends anything: there is no key, no wallet and no exchange client here.

What it measures, so going live is a decision based on evidence:
  validity  how many paper quotes would be legal live orders: whole contracts, price 0.001-0.999 on the tick, and the
            exchange's $10 minimum per order (a 20-contract quote at 25c is $5 and would be REJECTED)
  churn     order actions per minute against the address budget: 10,000 requests to start, then 1 per $1 traded
            (cancels get a bigger allowance). A maker that re-prices constantly burns that in minutes, so orders are
            only re-priced when the price moves by >= repost_threshold
  payloads  the exact exchange action JSON for every change (data/shadow_orders.jsonl)

HIP-4 order rules (Hyperliquid docs + Chainstack's HIP-4 guide, checked 2026-10-06):
  - coin "#<10*outcome + side>", side 0 = Yes, 1 = No; asset id for the action = 100_000_000 + encoding
  - fully collateralised in USDC (USDH until it was sunset in Oct 2026); no leverage; settles automatically
  - you cannot sell Yes you do not hold: an ask on Yes is placed as a BID on No at 1 - price (same payoff)
  - post-only (tif "Alo") keeps every order a maker order: zero maker fee on outcome markets"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

ASSET_BASE = 100_000_000
PX_MIN, PX_MAX = 0.001, 0.999
ROLES = ("bid", "ask", "bid2", "ask2")


def fmt_px(px: float) -> str:
    return f"{px:.4f}".rstrip("0").rstrip(".")


def order_action(asset: int, is_buy: bool, px: float, sz: int) -> dict:
    """The exchange 'order' action exactly as Hyperliquid expects it (before signing): post-only limit order."""
    return {"type": "order", "orders": [{"a": asset, "b": is_buy, "p": fmt_px(px), "s": str(int(sz)), "r": False,
                                         "t": {"limit": {"tif": "Alo"}}}], "grouping": "na"}


def cancel_action(asset: int, oid) -> dict:
    return {"type": "cancel", "cancels": [{"a": asset, "o": oid}]}


def modify_action(asset: int, oid, is_buy: bool, px: float, sz: int) -> dict:
    return {"type": "batchModify", "modifies": [{"oid": oid, "order": order_action(asset, is_buy, px, sz)["orders"][0]}]}


class ShadowExecutor:
    def __init__(self, cfg: dict, mm, data_dir: Path | None, clock=time.time):
        lc = cfg["hip4_live"]
        self.enabled = bool(lc["shadow"])
        self.repost = float(lc["repost_threshold"])
        self.min_notional = float(lc["min_notional"])
        self.tick = float(lc["tick"])
        self.budget = float(lc["address_budget"])
        self.mm, self.clock = mm, clock
        self.path = Path(data_dir) / "shadow_orders.jsonl" if data_dir is not None else None
        self.working: dict[tuple, dict] = {}     # (market id, role) -> the order we would have resting
        self._oid = 0
        self.actions: deque = deque(maxlen=200)  # recent actions (for the page)
        self.rate: deque = deque()               # (ts, kind) for the last hour
        self.totals = {"order": 0, "modify": 0, "cancel": 0}
        self.checks: deque = deque()             # (ts, valid, reason) for every desired order in the last hour
        self.started = clock()
        self._last_fill_ts = max((f["ts"] for f in getattr(mm, "fills", [])), default=0.0)   # mirror new fills only
        self.status, self.error = ("waiting" if self.enabled else "off"), None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paper quotes -> live orders
    def _snap(self, px: float, up: bool) -> float:
        n = px / self.tick
        return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * self.tick, 6)

    def desired(self) -> tuple[dict, list]:
        """What we would want resting right now, and why any paper quote could NOT be a live order."""
        want, rejected = {}, []
        with self.mm.lock:
            quotes = {k: dict(v) for k, v in self.mm.quotes.items()}
            hold = {int(k): p["qty"] for k, p in self.mm.pos.items()}
        for mid, q in quotes.items():
            yes = max(0.0, hold.get(mid, 0.0))
            for role in ROLES:
                px, sz = q.get(role), int(math.floor(q.get(role + "_sz", 0) + 1e-9))
                if px is None or sz <= 0:
                    continue
                if role.startswith("bid"):
                    coin_side, is_buy, opx = 0, True, self._snap(px, up=False)             # buy Yes
                elif yes >= sz:
                    coin_side, is_buy, opx = 0, False, self._snap(px, up=True)             # sell Yes we hold
                    yes -= sz
                else:
                    coin_side, is_buy, opx = 1, True, self._snap(1 - px, up=False)         # no Yes to sell: buy No
                notional = opx * sz
                reason = None
                if not (PX_MIN <= opx <= PX_MAX):
                    reason = "price outside 0.001-0.999"
                elif notional < self.min_notional:
                    reason = f"under the ${self.min_notional:g} minimum (${notional:.2f})"
                if reason:
                    rejected.append({"mid": mid, "role": role, "px": opx, "sz": sz, "reason": reason})
                    continue
                enc = mid * 10 + coin_side
                want[(mid, role)] = {"mid": mid, "role": role, "coin": f"#{enc}", "asset": ASSET_BASE + enc,
                                     "is_buy": is_buy, "px": opx, "sz": sz}
        return want, rejected

    def _snipes(self, now: float) -> tuple[list, list]:
        """The sniper takes the book: every paper snipe is ONE immediate-or-cancel order at its limit price. A Yes sell
        we can't cover from held Yes goes in as a No buy at 1 - price. Checked against the same live rules."""
        out, rejected = [], []
        with self.mm.lock:
            new = [f for f in self.mm.fills if f.get("how") == "snipe" and f["ts"] > self._last_fill_ts]
            hold = {int(k): p["qty"] for k, p in self.mm.pos.items()}
        for f in new:
            self._last_fill_ts = max(self._last_fill_ts, f["ts"])
            mid, sz = f["id"], int(f["qty"])
            yes_before = hold.get(mid, 0.0) + (f["qty"] if f["side"] < 0 else -f["qty"])
            if f["side"] > 0:
                side, is_buy, px = 0, True, self._snap(f["px"], up=True)
            elif yes_before >= sz:
                side, is_buy, px = 0, False, self._snap(f["px"], up=False)
            else:
                side, is_buy, px = 1, True, self._snap(1 - f["px"], up=True)
            if px * sz < self.min_notional - 1e-9 or not (PX_MIN <= px <= PX_MAX):
                rejected.append({"mid": mid, "role": "snipe", "px": px, "sz": sz,
                                 "reason": f"under the ${self.min_notional:g} minimum" if px * sz < self.min_notional else "price range"})
                continue
            enc = mid * 10 + side
            o = {"mid": mid, "role": "snipe", "coin": f"#{enc}", "asset": ASSET_BASE + enc, "is_buy": is_buy, "px": px, "sz": sz}
            act = order_action(o["asset"], is_buy, px, sz)
            act["orders"][0]["t"] = {"limit": {"tif": "Ioc"}}                  # take what is there, never rest
            out.append(("order", o, act))
        return out, rejected

    # ------------------------------------------------------------------ the diff a live order manager would send
    def step(self, now: float | None = None) -> list:
        now = self.clock() if now is None else now
        if getattr(self.mm, "strategy", "maker") == "sniper":
            out, rejected = self._snipes(now)
            self._record(now, out, [None] * len(out), rejected)
            return out
        want, rejected = self.desired()
        out = []
        with self._lock:
            for k in [k for k in self.working if k not in want]:
                o = self.working.pop(k)
                out.append(("cancel", o, cancel_action(o["asset"], o["oid"])))
            for k, o in want.items():
                cur = self.working.get(k)
                if cur is None:
                    self._oid += 1
                    o["oid"] = f"shadow-{self._oid}"
                    self.working[k] = o
                    out.append(("order", o, order_action(o["asset"], o["is_buy"], o["px"], o["sz"])))
                elif cur["asset"] != o["asset"] or cur["is_buy"] != o["is_buy"]:
                    # Yes-sell <-> No-buy switch: a different book, so cancel and place
                    out.append(("cancel", cur, cancel_action(cur["asset"], cur["oid"])))
                    self._oid += 1
                    o["oid"] = f"shadow-{self._oid}"
                    self.working[k] = o
                    out.append(("order", o, order_action(o["asset"], o["is_buy"], o["px"], o["sz"])))
                elif abs(cur["px"] - o["px"]) >= self.repost - 1e-9 or o["sz"] > cur["sz"] * 1.2:
                    # re-price only on a real move (or a real top-up): every modify spends address budget
                    o["oid"] = cur["oid"]
                    self.working[k] = o
                    out.append(("modify", o, modify_action(o["asset"], o["oid"], o["is_buy"], o["px"], o["sz"])))
        self._record(now, out, want, rejected)
        return out

    def _record(self, now: float, out: list, valid, rejected: list) -> None:
        with self._lock:
            for kind, o, _ in out:
                self.totals[kind] += 1
                self.rate.append((now, kind))
                self.actions.append({"ts": now, "kind": kind, **{k: o[k] for k in ("coin", "is_buy", "px", "sz", "role", "mid")}})
            for _ in valid:
                self.checks.append((now, True, None))
            for r in rejected:
                self.checks.append((now, False, r["reason"].split(" (")[0]))
            while self.rate and now - self.rate[0][0] > 3600:
                self.rate.popleft()
            while self.checks and now - self.checks[0][0] > 3600:
                self.checks.popleft()
        if out and self.path is not None:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    for kind, o, payload in out:
                        fh.write(json.dumps({"t": round(now, 3), "kind": kind, "mid": o["mid"], "role": o["role"],
                                             "action": payload}, separators=(",", ":")) + "\n")
                if self.path.stat().st_size > 20e6:                  # keep the log bounded
                    self.path.replace(self.path.with_suffix(".old.jsonl"))
            except OSError as exc:
                self.error = f"shadow log: {exc}"[:200]

    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        while not stop.is_set():
            try:
                self.step()
                self.status = "shadowing"
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:240]
            stop.wait(max(0.25, getattr(self.mm, "every_s", 1.0)))

    # ------------------------------------------------------------------ reporting
    def to_dict(self) -> dict:
        now = self.clock()
        with self._lock:
            span = max(60.0, min(3600.0, now - self.started))
            n = {"order": 0, "modify": 0, "cancel": 0}
            for t, k in self.rate:
                n[k] += 1
            per_min = {k: round(v / span * 60, 1) for k, v in n.items()}
            spend_day = (n["order"] + n["modify"]) / span * 86400        # cancels have their own, bigger allowance
            valid = sum(1 for c in self.checks if c[1])
            reasons: dict[str, int] = {}
            for c in self.checks:
                if not c[1]:
                    reasons[c[2]] = reasons.get(c[2], 0) + 1
            checked = len(self.checks)
            working = len(self.working)
            recent = list(self.actions)[-25:][::-1]
        # the budget refills 1 request per $1 traded: estimate today's paper volume from the maker's fills
        vol_day = 0.0
        with self.mm.lock:
            day = [f for f in self.mm.fills if now - f["ts"] <= 86400]
            if day:
                vol_day = sum(f["px"] * f["qty"] for f in day) / max(3600.0, now - day[0]["ts"]) * 86400
        net = spend_day - vol_day
        days_left = (self.budget / net) if net > 0 else None
        validity = valid / checked if checked else None
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "mode": "SHADOW (nothing is sent)",
                "mirrors": f"{getattr(self.mm, 'name', 'hip4_mm')} ({getattr(self.mm, 'strategy', 'maker')})",
                "working_orders": working, "per_min": per_min, "totals": dict(self.totals),
                "budget": {"initial": self.budget, "spend_per_day": round(spend_day), "earn_per_day": round(vol_day),
                           "days_until_empty": round(days_left, 1) if days_left is not None else None,
                           "ok": days_left is None or days_left >= 30},
                "validity": round(validity, 4) if validity is not None else None, "checked": checked,
                "rejected_reasons": reasons, "recent": recent,
                "rules": {"min_notional": self.min_notional, "tick": self.tick, "repost_threshold": self.repost,
                          "tif": "Alo (post-only)"}}
