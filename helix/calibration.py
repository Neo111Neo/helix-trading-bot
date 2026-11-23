"""Calibration: when Jev said 70%, did it happen about 70% of the time on this market?
Every direction answer is checked against the real mid after the horizon has passed."""

from __future__ import annotations

from collections import deque

BINS = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0001)]


def outcome_label(ret_bps: float, threshold_bps: float) -> str:
    if ret_bps > threshold_bps:
        return "up"
    if ret_bps < -threshold_bps:
        return "down"
    return "neutral"


def calibration_report(rows) -> dict:
    rows = list(rows)
    bins = []
    for lo, hi in BINS:
        inside = [r for r in rows if lo <= r["prob"] < hi]
        n = len(inside)
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": n,
                     "avg_pred": round(sum(r["prob"] for r in inside) / n, 3) if n else None,
                     "hit_rate": round(sum(1 for r in inside if r["correct"]) / n, 3) if n else None})
    n = len(rows)
    brier = None
    if n:
        total = 0.0
        for r in rows:
            for k in ("up", "down", "neutral"):
                total += (float(r["probs"].get(k, 0.0)) - (1.0 if r["outcome"] == k else 0.0)) ** 2
        brier = round(total / n, 4)
    return {
        "bins": bins, "n": n,
        "accuracy": round(sum(1 for r in rows if r["correct"]) / n, 3) if n else None,
        "brier": brier,
        "base_rates": {k: (round(sum(1 for r in rows if r["outcome"] == k) / n, 3) if n else None)
                       for k in ("up", "down", "neutral")},
    }


def _bin_index(prob: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= prob < hi:
            return i
    return -1


class CalibrationTracker:
    """Rolling window of resolved predictions with running totals, so the report is O(1)."""

    def __init__(self, horizon_s: float, min_threshold_bps: float = 2.0, keep: int = 20000):
        self.horizon_s = horizon_s
        self.min_threshold_bps = min_threshold_bps
        self.keep = keep
        self.pending: deque = deque()
        self.window: deque = deque()
        self._bins = [[0, 0.0, 0] for _ in BINS]  # n, sum of stated prob, hits
        self._outcomes = {"up": 0, "down": 0, "neutral": 0}
        self._hits = 0
        self._brier = 0.0

    def threshold(self, sigma_bps: float) -> float:
        """The neutral band: moves smaller than this count as 'neutral'. Jev is shown the same number."""
        return max(self.min_threshold_bps, 0.5 * sigma_bps)

    def add(self, coin: str, now: float, mid: float, choice: str, probs: dict, sigma_bps: float) -> None:
        self.pending.append((now + self.horizon_s, coin, mid, self.threshold(sigma_bps), choice, dict(probs)))

    def _apply(self, row: dict, sign: int) -> None:
        i = _bin_index(row["prob"])
        if i >= 0:
            b = self._bins[i]
            b[0] += sign
            b[1] += sign * row["prob"]
            b[2] += sign * int(row["correct"])
        self._outcomes[row["outcome"]] += sign
        self._hits += sign * int(row["correct"])
        self._brier += sign * row["brier"]

    def resolve(self, now: float, mids: dict) -> list[dict]:
        """Resolve every prediction whose horizon has passed, using the current mid."""
        done = []
        while self.pending and self.pending[0][0] <= now:
            due, coin, ref, thr, choice, probs = self.pending.popleft()
            mid = mids.get(coin)
            if not mid:
                continue  # market missing at resolution time: drop it rather than guess
            ret = (mid / ref - 1) * 1e4
            outcome = outcome_label(ret, thr)
            brier = sum((float(probs.get(k, 0.0)) - (1.0 if outcome == k else 0.0)) ** 2 for k in ("up", "down", "neutral"))
            row = {"coin": coin, "prob": probs.get(choice, 0.0), "choice": choice, "outcome": outcome,
                   "correct": outcome == choice, "ret_bps": ret, "brier": brier}
            if len(self.window) >= self.keep:
                self._apply(self.window.popleft(), -1)
            self.window.append(row)
            self._apply(row, +1)
            done.append(row)
        return done

    def report(self) -> dict:
        n = len(self.window)
        bins = [{"lo": lo, "hi": min(hi, 1.0), "n": b[0],
                 "avg_pred": round(b[1] / b[0], 3) if b[0] else None,
                 "hit_rate": round(b[2] / b[0], 3) if b[0] else None} for (lo, hi), b in zip(BINS, self._bins)]
        return {
            "bins": bins, "n": n, "pending": len(self.pending),
            "accuracy": round(self._hits / n, 3) if n else None,
            "brier": round(self._brier / n, 4) if n else None,
            "base_rates": {k: (round(v / n, 3) if n else None) for k, v in self._outcomes.items()},
        }
