"""Opus research layer (the feedback loop). Sends the run's numbers to Claude Opus and gets
back a written review: what's working, failure patterns, and the next experiments to run.
It never trades and never changes config on its own; you decide what to change."""

from __future__ import annotations

import json
import os
import threading
import time

import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are the research lead reviewing a paper-trading run of HELIX on Hyperliquid perps.
Every second, code turns the live order book and trade flow into a compact state, Jev (a fast decision
model) answers six typed questions (regime, direction over the horizon, toxic flow, liquidity stress,
entry quality, inventory pressure), code turns those into actions, and a risk engine (fees vs target,
slippage from the real book, leverage, entry rate, daily loss, drawdown) can veto anything. A rules-only
BASELINE book trades the same markets with the same costs and limits.

Write a short, blunt review for the owner, in plain English:
1. Verdict: is there evidence Jev adds edge over the baseline after costs? Say "not enough data" if
   either book has fewer than 50 closed trades.
2. What the numbers show: net vs gross P&L, fees and funding as a share of gross, win rate, average
   trade in bps, holding time, slippage.
3. Calibration: do Jev's direction probabilities match reality on this market? Point to specific bins.
4. The funnel: how many decisions became signals, passed risk and filled, and which blocks dominate.
5. Next experiments: exactly 3, each a specific config.yaml change with the expected effect and how to
   tell if it worked.
Never recommend going live or raising leverage. Use short sections and bullet points. No hype."""


class Analyst:
    def __init__(self, model: str, max_tokens: int = 2000, api_key: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.status = "idle" if self.api_key else "unavailable"
        self.report: str | None = None
        self.error: str | None = None
        self.ts: float | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def to_dict(self) -> dict:
        return {"status": self.status, "report": self.report, "error": self.error, "ts": self.ts,
                "model": self.model}

    def start(self, summary: dict) -> bool:
        if not self.available:
            self.status, self.error = "unavailable", "Add ANTHROPIC_API_KEY to .env to enable Opus reviews"
            return False
        with self._lock:
            if self.status == "running":
                return False
            self.status, self.error = "running", None
        threading.Thread(target=self._run, args=(summary,), daemon=True).start()
        return True

    def _run(self, summary: dict) -> None:
        try:
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()  # needed when the key isn't workspace-scoped
            if workspace:
                headers["anthropic-workspace-id"] = workspace
            resp = requests.post(
                ANTHROPIC_URL,
                headers=headers,
                json={"model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM_PROMPT,
                      "messages": [{"role": "user", "content": "Run summary (JSON):\n" + json.dumps(summary, default=str)}]},
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
            self.report, self.status, self.ts = text.strip() or "(empty response)", "done", time.time()
        except Exception as exc:  # network, auth, parsing: surface it in the dashboard
            self.status, self.error, self.ts = "error", str(exc), time.time()
