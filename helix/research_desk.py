"""Runs the research desk (research_agent/daily.py) from the PC's Helix every `every_min` minutes (2026-10-13).

The desk only READS the Tokyo server (a forced-command key that can print status and nothing else) and writes reports
under research_agent/. Running it here means the hourly snapshots and reports keep coming whenever this PC's Helix is
up (it runs 24/7 under keepalive.ps1), with no Claude session and no permission prompts. It runs as a separate,
low-priority process with a timeout, so a hang or crash in the desk can never touch Helix. Off by default
(config research_desk.enabled); the Tokyo server never gets research_agent/, so it never runs there."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

BELOW_NORMAL, NO_WINDOW = 0x00004000, 0x08000000          # Windows: low priority, never a console window


class DeskRunner:
    FIRST_DELAY_S, TIMEOUT_S = 90.0, 300.0

    def __init__(self, root: Path, every_min: float = 60.0, python: str = sys.executable, clock=time.time,
                 run=subprocess.run):
        self.root, self.every_s, self.python, self.clock, self._run = Path(root), every_min * 60.0, python, clock, run
        self.script = self.root / "research_agent" / "daily.py"
        self.last_ts, self.last_ok, self.last_out, self.runs = None, None, None, 0

    def available(self) -> bool:
        return self.script.exists()

    def run_once(self) -> bool:
        flags = (BELOW_NORMAL | NO_WINDOW) if sys.platform == "win32" else 0
        self.runs += 1
        try:
            p = self._run([self.python, "-I", str(self.script)], cwd=str(self.root), capture_output=True, text=True,
                          timeout=self.TIMEOUT_S, creationflags=flags, stdin=subprocess.DEVNULL)
            self.last_ok = p.returncode == 0
            self.last_out = ((p.stdout or "") + (p.stderr or "")).strip()[-300:]
        except Exception as exc:                               # noqa: BLE001 - never into Helix
            self.last_ok, self.last_out = False, f"{type(exc).__name__}: {exc}"[:300]
        self.last_ts = self.clock()
        self._write_status()
        return bool(self.last_ok)

    def _write_status(self) -> None:
        """research_agent/runner_status.json: the runner's own view (a hung or unstartable run leaves desk_status.json
        old, and this says why)."""
        try:
            p = self.root / "research_agent" / "runner_status.json"
            tmp = p.with_name(f"runner_status.json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass

    def to_dict(self) -> dict:
        return {"every_min": self.every_s / 60, "last_ts": self.last_ts, "last_ok": self.last_ok,
                "last_out": self.last_out, "runs": self.runs}

    def run(self, stop: threading.Event) -> None:
        if stop.wait(self.FIRST_DELAY_S):                  # let Helix finish starting first
            return
        while not stop.is_set():
            self.run_once()
            stop.wait(self.every_s)
