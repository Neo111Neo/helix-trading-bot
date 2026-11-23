"""Weekly Jev Lab re-run, inside Helix (paper research; never trades, never changes config).

Every `every_days` it downloads fresh history for the live coins into data/lab/weekly/<date>/, rebuilds the direction
states and asks Jev the live prompt (Jev v2) on all of them - roughly 9,500 calls, about $1 - then checks, per coin and
pooled, whether short moves still REVERSE and whether Jev still reads them (rank IC > 0). Nothing is fitted, so all
states are used. The result goes into the event log and a summary (/api/labweekly, shown in the terminal); an
ALERT is raised when the reversal has faded or Jev's IC has turned negative, because Jev v2 assumes it."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from statistics import fmean, median

from . import lab
from .hyperliquid import URLS, HLRest

RETRY_S = 6 * 3600

class LabSchedule:
    def __init__(self, cfg: dict, data_dir: Path, jev_client, notify=None, rest: HLRest | None = None, clock=time.time):
        ls = cfg["lab_schedule"]
        self.enabled = bool(ls["enabled"]) and jev_client is not None
        self.every_s = float(ls["every_days"]) * 86400
        self.variant = ls["variant"]
        self.workers = int(ls.get("workers", 2))
        self.coins = list(cfg["market"]["coins"])
        self.root = Path(data_dir) / "lab" / "weekly"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "schedule.json"
        self.jev = jev_client
        self.notify = notify or (lambda level, text: None)
        self.rest = rest or HLRest(URLS[cfg.get("network", "mainnet")][0], timeout=20)
        self.clock = clock
        self.prompt = Path(__file__).resolve().parent / "prompts" / f"{self.variant}.json"
        self.status, self.error, self.running = ("waiting" if self.enabled else "off"), None, False
        self.state = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        # seed from the 20-coin study run by hand on 2026-10-05, so the first weekly run is a week after it; saved at
        # once so the schedule doesn't move if that report is later rewritten
        full = self.path.parent.parent / "full" / "report.json"
        self.state = {"last_run": full.stat().st_mtime if full.exists() else 0.0, "history": []}
        try:
            self._save()
        except OSError:
            pass
        return self.state

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def due(self, now: float) -> bool:
        """Due a week after the last success (or at the retry time after a failure), and only between :10 and :40
        past the hour, away from the hourly test's round opening and resolution at the top of the hour."""
        when = self.state.get("retry_at") or self.state.get("last_run", 0.0) + self.every_s
        return now >= when and 10 <= (now % 3600) / 60 < 40

    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        while not stop.wait(60.0):
            self.tick()

    def tick(self) -> None:
        if not self.due(self.clock()):
            return
        try:
            self.run_once()
        except Exception as exc:  # report and try again in 6 hours (a later run resumes the same day's data)
            self.status, self.error, self.running = "error", f"{type(exc).__name__}: {exc}"[:300], False
            self.state["retry_at"] = self.clock() + RETRY_S
            self.notify("warn", f"Weekly Jev Lab failed, retrying in 6 h: {self.error}")
            self._save()

    def probe(self) -> None:
        """One cheap Jev call before 20 minutes of downloads and ~9,500 calls: fail fast if the key or API is down."""
        res = self.jev.ask({"x": 1}, {"probe": {"type": "noul", "instructions": "Is `x` greater than zero?"}})
        if res.status != "ok":
            raise RuntimeError(f"Jev probe failed: {res.error}")

    def run_once(self) -> dict:
        now = self.clock()
        self.running, self.status = True, "running"
        self.probe()
        out = self.root / time.strftime("%Y-%m-%d", time.gmtime(now))
        (out / "variants").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.prompt, out / "variants" / self.prompt.name)
        spec = json.loads(self.prompt.read_text(encoding="utf-8"))
        lab.fetch_all(self.rest, self.coins, out, now, skip_existing=True)   # a same-day retry reuses what it has
        lab.build_all(out, self.coins, ("D",))
        states = lab.load_states(out, "D")
        lab.run_jev(self.jev, out, "D", spec, states, {}, workers=self.workers, log=lambda *_: None,
                    max_fail_streak=25)
        per = lab.per_coin_direction(out, states, {spec["name"]: spec}, spec["name"], holdout_only=False)
        summary = self.summarize(per, now)
        self.state["last_run"] = now
        self.state.pop("retry_at", None)
        self.state["history"] = (self.state.get("history", []) + [summary])[-26:]
        self._save()
        self.running, self.status, self.error = False, "ok", None
        self.notify("risk" if summary["alert"] else "info", summary["headline"])
        return summary

    @staticmethod
    def summarize(per: dict, now: float) -> dict:
        if not per:
            return {"ts": now, "alert": True, "headline": "Weekly Jev Lab: no usable coins", "coins": {}}
        ac = [m["autocorr"] for m in per.values()]
        ic = [m["ic"] for m in per.values()]
        gross = [m["gross_bps"] for m in per.values() if m.get("gross_bps") is not None]
        reversing = sum(1 for a in ac if a < 0)
        alert = median(ac) > -0.02 or median(ic) <= 0
        best = sorted(((m["gross_bps"], c) for c, m in per.items() if m.get("gross_bps") is not None), reverse=True)[:3]
        head = (f"Weekly Jev Lab: short moves reversed in {reversing}/{len(per)} coins (median {median(ac):+.3f}), "
                f"Jev v2 rank IC median {median(ic):+.3f}, gross {fmean(gross) if gross else 0:+.1f} bps/trade; best "
                + ", ".join(f"{c} {g:+.1f}" for g, c in best))
        if alert:
            head = "ALERT - " + head + ". The reversal Jev v2 relies on has faded: consider pausing it."
        return {"ts": now, "alert": alert, "headline": head, "median_autocorr": median(ac), "median_ic": median(ic),
                "reversing": reversing, "n_coins": len(per), "coins": per}

    def to_dict(self) -> dict:
        hist = self.state.get("history", [])
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "running": self.running,
                "last_run": self.state.get("last_run"),
                "next_run": self.state.get("retry_at") or self.state.get("last_run", 0) + self.every_s,
                "latest": hist[-1] if hist else None}
