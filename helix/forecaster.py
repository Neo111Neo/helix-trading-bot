"""Claude forecaster: a second opinion next to Jev. It never trades and never changes config.

Every `interval_s` it sends every coin's compact state (the same numbers Jev reads) to Claude in one
call and gets back, per coin, up/down/neutral probabilities against the same neutral band Jev is
scored on, plus an expected move and an 80% range over the horizon. Each forecast is scored with the
same CalibrationTracker logic as Jev, so the two can be compared on equal terms."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from statistics import fmean
from typing import Literal

import anthropic
from pydantic import BaseModel

from .calibration import CalibrationTracker

SYSTEM_PROMPT = """You forecast short-horizon price moves of crypto perpetual futures for a paper-trading
research tool. Nothing you say is traded; your forecasts are scored against what actually happens.

For each coin you get a compact numeric state: returns from 5 seconds to 60 minutes, 1-second
volatility, order-book imbalance and depth, aggressive buy share of recent trades, trade rate,
funding, premium, 1-minute trend and RSI, plus typical_move_bps (the expected 1-sigma move over the
horizon) and neutral_band_bps. You also get recent_outcome_rates: how often moves over this horizon
recently ended up, down or inside the band.

For every coin, forecast where the mid price will be {horizon_s} seconds from now:
- p_up: probability the mid rises by more than neutral_band_bps
- p_down: probability the mid falls by more than neutral_band_bps
- p_neutral: probability it ends within neutral_band_bps of the current mid (the three sum to 1)
- direction: the most likely of up / down / neutral
- expected_move_bps: your expected signed change in basis points
- low_bps and high_bps: an 80% range for the change
- reason: one short sentence naming the inputs that drove the call

Short-horizon moves are mostly noise. Be calibrated rather than confident: with weak or conflicting
evidence, stay close to recent_outcome_rates, and say neutral when that is the most likely outcome.
A strong longer-term trend is information, not a guarantee that it continues over the next few minutes."""


class CoinForecast(BaseModel):
    coin: str
    direction: Literal["up", "down", "neutral"]
    p_up: float
    p_down: float
    p_neutral: float
    expected_move_bps: float
    low_bps: float
    high_bps: float
    reason: str


class ForecastBatch(BaseModel):
    forecasts: list[CoinForecast]


def _utc_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


class ClaudeForecaster:
    def __init__(self, cfg: dict, engine, data_dir: Path, api_key: str | None = None):
        fc = cfg["claude_forecast"]
        self.enabled = bool(fc["enabled"])
        self.model = fc["model"]
        self.interval_s = float(fc["interval_s"])
        self.effort = fc["effort"]
        self.budget = float(fc["daily_budget_usd"])
        self.price_in = float(fc["price_in_per_million"])
        self.price_out = float(fc["price_out_per_million"])
        self.horizon_s = float(cfg["market"]["horizon_s"])
        self.engine = engine
        self.path = Path(data_dir) / "claude_forecasts.json"
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
        workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()  # needed when the key isn't workspace-scoped
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        self.client = anthropic.Anthropic(api_key=key, timeout=120.0, max_retries=2, default_headers=headers) if key else None
        self.status = "off" if not self.enabled else ("waiting" if self.client else "no key")
        self.error: str | None = None
        self.last_run: float | None = None
        self.next_run = time.time() + 150  # let the engine warm up first
        self.last_latency_s: float | None = None
        self.cost_day, self.cost_date, self.calls_day = 0.0, _utc_day(time.time()), 0
        self.latest: dict[str, dict] = {}
        self.pending: deque = deque()
        self.resolved: deque = deque(maxlen=3000)
        self.tracker = CalibrationTracker(self.horizon_s)
        self._lock = threading.Lock()
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.latest = data.get("latest", {})
        self.pending = deque(data.get("pending", []))
        self.resolved = deque(data.get("resolved", []), maxlen=3000)
        if data.get("cost_date") == _utc_day(time.time()):
            self.cost_day, self.calls_day, self.cost_date = data.get("cost_day", 0.0), data.get("calls_day", 0), data["cost_date"]

    def _save(self) -> None:
        data = {"latest": self.latest, "pending": list(self.pending), "resolved": list(self.resolved),
                "cost_day": self.cost_day, "calls_day": self.calls_day, "cost_date": self.cost_date}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled or not self.client:
            return
        while not stop.wait(2.0):
            now = time.time()
            try:
                self._resolve(now)
                if _utc_day(now) != self.cost_date:
                    self.cost_day, self.calls_day, self.cost_date = 0.0, 0, _utc_day(now)
                if now >= self.next_run:
                    self.next_run = now + self.interval_s
                    if self.budget > 0 and self.cost_day >= self.budget:  # 0 = no daily cap
                        self.status = "budget reached"
                        continue
                    self._forecast(now)
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:300]

    def _inputs(self) -> dict[str, dict]:
        snap = self.engine.snapshot()
        out = {}
        for coin, m in (snap.get("markets") or {}).items():
            st = m.get("state")
            if m.get("ready") and st and m.get("mid") and "neutral_band_bps" in st:
                out[coin] = {"state": st, "mid": m["mid"]}
        return out

    def _forecast(self, now: float) -> None:
        inputs = self._inputs()
        if not inputs:
            self.status = "waiting for market data"
            return
        rates = self.tracker.report()["base_rates"] if self.tracker.window else None
        if rates is None:
            with self.engine.lock:
                rep = self.engine.calib.report()
            rates = rep["base_rates"] if rep["n"] else None
        payload = {"horizon_s": self.horizon_s, "recent_outcome_rates": rates,
                   "coins": {c: v["state"] for c, v in inputs.items()}}
        self.status = "forecasting"
        t0 = time.time()
        try:
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=SYSTEM_PROMPT.format(horizon_s=int(self.horizon_s)),
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                output_format=ForecastBatch,
            )
        except anthropic.AuthenticationError:
            self.status, self.error = "auth error", "ANTHROPIC_API_KEY was rejected"
            self.next_run = float("inf")
            return
        except anthropic.RateLimitError:
            self.status, self.error = "rate limited", "Rate limited; retrying next interval"
            return
        except anthropic.APIStatusError as exc:
            if "anthropic-workspace-id" in str(exc.message):
                self.status = "needs workspace id"
                self.error = "This API key isn't scoped to a workspace: add ANTHROPIC_WORKSPACE_ID to .env, or use a workspace key"
                self.next_run = float("inf")  # fails the same way every time until the key/.env changes
                return
            self.status, self.error = "api error", f"HTTP {exc.status_code}: {exc.message}"[:300]
            return
        except anthropic.APIConnectionError:
            self.status, self.error = "network error", "Could not reach the Anthropic API"
            return
        self.last_latency_s = time.time() - t0
        usage = resp.usage
        self.cost_day += (usage.input_tokens * self.price_in + usage.output_tokens * self.price_out) / 1e6
        self.calls_day += 1
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            self.status, self.error = "no answer", f"stop_reason={resp.stop_reason}"
            return
        made = time.time()
        with self._lock:
            for f in resp.parsed_output.forecasts:
                if f.coin not in inputs:
                    continue
                probs = {"up": max(0.0, f.p_up), "down": max(0.0, f.p_down), "neutral": max(0.0, f.p_neutral)}
                total = sum(probs.values()) or 1.0
                probs = {k: v / total for k, v in probs.items()}
                choice = max(probs, key=probs.get)
                st, mid = inputs[f.coin]["state"], inputs[f.coin]["mid"]
                rec = {"coin": f.coin, "ts": made, "due": made + self.horizon_s, "mid0": mid,
                       "band_bps": st["neutral_band_bps"], "sigma_bps": st.get("typical_move_bps", 0.0),
                       "probs": probs, "choice": choice, "expected_bps": f.expected_move_bps,
                       "low_bps": min(f.low_bps, f.high_bps), "high_bps": max(f.low_bps, f.high_bps),
                       "reason": f.reason[:200]}
                self.latest[f.coin] = rec
                self.pending.append(rec)
        self.last_run, self.status, self.error = made, "ok", None
        self._save()

    def _resolve(self, now: float) -> None:
        if not self.pending or self.pending[0]["due"] > now:
            return
        mids = {c: v["mid"] for c, v in self._inputs().items()}
        changed = False
        with self._lock:
            while self.pending and self.pending[0]["due"] <= now:
                rec = self.pending.popleft()
                mid = mids.get(rec["coin"])
                if not mid:
                    continue
                ret = (mid / rec["mid0"] - 1) * 1e4
                band = rec["band_bps"]
                outcome = "up" if ret > band else ("down" if ret < -band else "neutral")
                self.resolved.append({**rec, "actual_bps": ret, "outcome": outcome, "correct": outcome == rec["choice"]})
                changed = True
        if changed:
            self._save()

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _metrics(rows: list[dict]) -> dict:
        """rows: choice, outcome, ret_bps (and optionally expected_bps)."""
        if not rows:
            return {"n": 0}
        calls = [r for r in rows if r["choice"] in ("up", "down")]
        moved = [r for r in calls if r["outcome"] in ("up", "down")]
        signed = [r["ret_bps"] if r["choice"] == "up" else -r["ret_bps"] for r in calls]
        out = {
            "n": len(rows),
            "hit_rate": round(sum(1 for r in rows if r["choice"] == r["outcome"]) / len(rows), 3),
            "dir_right_when_moved": round(sum(1 for r in moved if r["choice"] == r["outcome"]) / len(moved), 3) if moved else None,
            "moved_n": len(moved),
            "avg_move_called_dir_bps": round(fmean(signed), 2) if signed else None,
            "calls": {k: sum(1 for r in rows if r["choice"] == k) for k in ("up", "down", "neutral")},
            "outcomes": {k: sum(1 for r in rows if r["outcome"] == k) for k in ("up", "down", "neutral")},
        }
        exp = [abs(r["expected_bps"] - r["ret_bps"]) for r in rows if r.get("expected_bps") is not None]
        if exp:
            out["mae_bps"] = round(fmean(exp), 2)
        return out

    def scoreboard(self) -> dict:
        with self._lock:
            claude_rows = [{"choice": r["choice"], "outcome": r["outcome"], "ret_bps": r["actual_bps"],
                            "expected_bps": r["expected_bps"]} for r in self.resolved]
        with self.engine.lock:
            jev_rows = [{"choice": r["choice"], "outcome": r["outcome"], "ret_bps": r["ret_bps"]}
                        for r in list(self.engine.calib.window)[-20000:]]
        return {"claude": self._metrics(claude_rows), "jev": self._metrics(jev_rows)}

    def to_dict(self) -> dict:
        with self._lock:
            latest = dict(self.latest)
            history = list(self.resolved)[-300:]
            pending = list(self.pending)
        return {
            "enabled": self.enabled, "status": self.status, "error": self.error, "model": self.model,
            "effort": self.effort, "interval_s": self.interval_s, "horizon_s": self.horizon_s,
            "last_run": self.last_run, "next_run": self.next_run if self.next_run != float("inf") else None,
            "latency_s": self.last_latency_s, "cost_day": round(self.cost_day, 4), "calls_day": self.calls_day,
            "budget_usd": self.budget, "latest": latest, "pending": pending, "history": history,
            "scoreboard": self.scoreboard(),
        }
