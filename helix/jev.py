"""Jev decision layer.

Calls TypeSafe's System One API directly (POST /v1/systemone): one call per market per
decision, carrying the six atomic questions from the guide's judgment battery. Jev returns
typed answers with probabilities. It never places trades and never computes prices. The
model version is pinned and logged, and an answer that misses the deadline is discarded."""

from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "https://api.typesafe.ai"


def build_questions(horizon_s: int) -> dict:
    """The judgment battery, written to TypeSafe's API rules (docs.typesafe.ai/api):
    - Questions are answered independently against the same state, so no question may depend on
      another's answer. Setup quality is therefore asked separately for a LONG and for a SHORT
      (the old "entry quality in the direction of the bias" was unanswerable - Jev never sees its bias).
    - Criteria describe situations, not degrees; state fields are referenced in backticks.
    - Noul questions define what yes and no mean."""
    return {
        "regime": {
            "type": "choice",
            "instructions": "Which regime describes this market right now, judging by the returns, volatility and order book in the state",
            "criteria": {
                "trending": "Returns over 5 to 60 minutes point the same way and price keeps extending in that direction",
                "mean_reverting": "Price oscillates around a stable level; short moves keep reversing",
                "high_vol": "Large moves both ways; `vol_60s_vs_300s` well above 1",
                "crisis": "Disorderly: liquidity vanishing (`depth_vs_5m_median` far below 1) with violent moves",
            },
        },
        "direction": {
            # Worded exactly as calibration scores it: moves inside the band count as neutral.
            "type": "choice",
            "instructions": f"Where will the mid price be {horizon_s} seconds from now, compared with the current mid? "
                            "`neutral_band_bps` is the threshold; `typical_move_bps` is the usual size of a move over that time.",
            "criteria": {
                "up": "The mid ends more than `neutral_band_bps` above where it is now",
                "down": "The mid ends more than `neutral_band_bps` below where it is now",
                "neutral": "The mid ends within `neutral_band_bps` of where it is now",
            },
        },
        "toxic_flow": {
            "type": "noul",
            "instructions": "Is the recent aggressive order flow informed, likely to keep pushing price the way it is going",
            "criteria": {
                True: "One-sided aggressive flow (`aggr_buy_share_15s` far from 0.5) with the book and price moving the same way",
                False: "Flow is balanced, or one-sided but absorbed by the book without moving price",
            },
        },
        "liquidity_stressed": {
            "type": "noul",
            "instructions": "Is the order book thinner than normal right now",
            "criteria": {
                True: "Depth near the mid is well below normal (`depth_vs_5m_median` clearly under 1) or the spread is wide",
                False: "Depth and spread are around their normal levels",
            },
        },
        "long_setup": {
            "type": "score",
            "instructions": f"How favourable is opening a LONG right now, a bet that price rises over the next {horizon_s} seconds",
            "criteria": [
                "Against: price is falling or sellers dominate the flow and the book",
                "Weak: mixed signals; nothing in the flow, book or trend supports a rise",
                "Supported: flow, book imbalance and short-term trend lean towards a rise",
                "Strong: buying pressure, bid-heavy book and trend all clearly point up",
            ],
        },
        "short_setup": {
            "type": "score",
            "instructions": f"How favourable is opening a SHORT right now, a bet that price falls over the next {horizon_s} seconds",
            "criteria": [
                "Against: price is rising or buyers dominate the flow and the book",
                "Weak: mixed signals; nothing in the flow, book or trend supports a fall",
                "Supported: flow, book imbalance and short-term trend lean towards a fall",
                "Strong: selling pressure, ask-heavy book and trend all clearly point down",
            ],
        },
        "inventory_pressure": {
            "type": "score",
            "instructions": "How urgently should the open position (`position_side`, `position_upnl_bps`, `position_hold_s`) be reduced",
            "criteria": [
                "None: no position, or the position is working and conditions still support it",
                "Mild: the position is flat or slightly against, conditions turning neutral",
                "Skew hard: conditions now favour the other side",
                "Reduce now: the position is clearly wrong or the market is disorderly",
            ],
        },
    }


def setup_prob(answer: dict | None) -> float:
    """P(setup is at least 'Supported') = probability mass on the top two levels.
    Jev's `score` is a probability-weighted mean of level indices (docs.typesafe.ai/api), so a bar on
    the mean (e.g. 1.8) needs near-certainty; the level probabilities are the honest signal."""
    if not answer:
        return 0.0
    probs = answer.get("probabilities") or {}
    levels = answer.get("levels") or len(probs) or 4
    return sum(float(probs.get(i, 0.0)) for i in range(levels // 2, levels))


@dataclass
class JevResult:
    status: str  # ok | mock | late | error | offline
    model: str | None = None
    latency_ms: float | None = None
    answers: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "mock") and bool(self.answers)


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def normalize_answers(raw: dict, questions: dict) -> dict:
    """Turn the API's answer objects into plain dicts the policy can rely on."""
    out: dict = {}
    for qid, q in questions.items():
        a = raw.get(qid)
        if not isinstance(a, dict):
            continue
        qtype = a.get("type") or q["type"]
        if qtype == "noul":
            out[qid] = {"type": "noul", "p": min(1.0, max(0.0, _f(a.get("noul"), 0.5)))}
        elif qtype == "choice":
            options = list(q["criteria"].keys())
            probs_raw = a.get("probabilities") or {}
            probs = {k: _f(probs_raw.get(k)) for k in options}
            choice = a.get("choice") if a.get("choice") in options else max(probs, key=probs.get)
            if sum(probs.values()) <= 0:
                conf = _f(a.get("confidence"), 0.0)
                probs = {k: (conf if k == choice else (1 - conf) / max(1, len(options) - 1)) for k in options}
            out[qid] = {"type": "choice", "choice": choice, "probabilities": probs,
                        "p": probs.get(choice, 0.0), "confidence": _f(a.get("confidence"), probs.get(choice, 0.0))}
        elif qtype == "score":
            levels = len(q["criteria"])
            probs_raw = a.get("probabilities") or {}
            probs = {i: _f(probs_raw.get(str(i), probs_raw.get(i))) for i in range(levels)}
            score = a.get("score")
            if score is None and sum(probs.values()) > 0:
                score = sum(i * p for i, p in probs.items()) / sum(probs.values())
            out[qid] = {"type": "score", "score": _f(score), "confidence": _f(a.get("confidence")),
                        "probabilities": probs, "levels": levels}
    return out


class JevClient:
    """Direct HTTP client for System One. Pinned model, hard deadline, one session per thread
    so calls for several markets can run in parallel."""

    provider = "typesafe"

    def __init__(self, api_key: str, model: str, timeout_ms: int, base_url: str | None = None,
                 http: requests.Session | None = None):
        self.api_key = api_key
        self.model = model
        self.timeout_ms = timeout_ms
        self.url = (base_url or os.getenv("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/") + "/v1/systemone"
        self._shared = http
        self._local = threading.local()
        self.pinned = not model.endswith(("-latest", "-preview"))

    def _session(self) -> requests.Session:
        if self._shared is not None:
            return self._shared
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": "helix-paper-engine/2.0"})
            self._local.session = s
        return s

    def ask(self, state: dict, questions: dict) -> JevResult:
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            resp = self._session().post(self.url, json=payload, headers=headers, timeout=self.timeout_ms / 1000)
        except requests.Timeout:
            return JevResult("late", latency_ms=(time.perf_counter() - t0) * 1000,
                             error=f"No answer within {self.timeout_ms} ms")
        except requests.RequestException as exc:
            return JevResult("error", latency_ms=(time.perf_counter() - t0) * 1000, error=str(exc)[:200])
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            body = resp.text[:200].replace("\n", " ")
            return JevResult("error", latency_ms=latency, error=f"HTTP {resp.status_code}: {body}")
        try:
            data = resp.json()
        except ValueError:
            return JevResult("error", latency_ms=latency, error="Response was not JSON")
        answers = normalize_answers(data.get("answers") or {}, questions)
        model = data.get("model")
        usage = data.get("usage") or {}
        if len(answers) < len(questions):
            return JevResult("error", model, latency, answers, usage, error="Response was missing answers")
        if latency > self.timeout_ms:
            return JevResult("late", model, latency, answers, usage,
                             error=f"Answer took {latency:.0f} ms (deadline {self.timeout_ms} ms)")
        if self.pinned and model and model != self.model:
            return JevResult("error", model, latency, answers, usage,
                             error=f"Model changed to {model}; thresholds were tuned on {self.model}")
        return JevResult("ok", model, latency, answers, usage)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


class MockJev:
    """Offline stand-in returning Jev-shaped answers from simple heuristics on the state.
    Only used with --demo or JEV_PROVIDER=mock. The dashboard always labels it MOCK."""

    provider = "mock"
    pinned = True

    def __init__(self, seed: int = 11, latency_ms: tuple[float, float] = (70, 240)):
        self.model = "mock-jev"
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._latency = latency_ms

    def ask(self, state: dict, questions: dict) -> JevResult:
        with self._lock:
            noise = [self._rng.gauss(0, 1) for _ in range(8)]
            latency = self._rng.uniform(*self._latency)
        vol = max(0.05, state["vol_1s_bps"])
        flow = (state["aggr_buy_share_15s"] - 0.5) * 2
        book = state["book_imbalance_top5"]
        mom = state["return_15s_bps"] / (vol * 4)
        s = 0.45 * math.tanh(2 * book) + 0.55 * math.tanh(2.5 * flow) + 0.25 * math.tanh(mom) + 0.15 * noise[0]
        direction = _softmax({"up": 1.8 * s, "down": -1.8 * s, "neutral": 0.35 + 0.4 * (1 - min(1.0, abs(s)))})
        trend = abs(state["return_300s_bps"]) / (vol * math.sqrt(300))
        vr = state["vol_60s_vs_300s"]
        depth = state["depth_vs_5m_median"]
        regime = _softmax({"trending": 1.4 * trend - 0.4 + 0.2 * noise[1], "mean_reverting": 0.8 - 0.8 * trend + 0.2 * noise[2],
                           "high_vol": 2.2 * (vr - 1.3) + 0.2 * noise[3], "crisis": 3.0 * (vr - 2.2) + 3.0 * (0.55 - depth)})
        toxic = _sigmoid(3.0 * (abs(flow) - 0.45) + 1.2 * (state["trade_rate_vs_5m"] - 1.2) + 0.3 * noise[4])
        stressed = _sigmoid(7.0 * (0.72 - depth) + 0.2 * noise[5])
        base_q = -1.2 * stressed - 0.4 * max(0.0, state["spread_bps"] - 1.0) + 0.3 * noise[6]
        long_q = 3.2 * max(0.0, s) + base_q
        short_q = 3.2 * max(0.0, -s) + base_q
        long_probs = _softmax({str(i): -abs(long_q - i) * 1.8 for i in range(4)})
        short_probs = _softmax({str(i): -abs(short_q - i) * 1.8 for i in range(4)})
        side = state["position_side"]
        inv = 0.0
        if side:
            against = direction["down"] if side > 0 else direction["up"]
            inv = 3.5 * against + max(0.0, -state["position_upnl_bps"]) / (vol * 6) + state["position_hold_s"] / 240
        i_probs = _softmax({str(i): -abs(min(3.2, inv + 0.2 * noise[7]) - i) * 1.8 for i in range(4)})
        raw = {
            "regime": {"type": "choice", "choice": max(regime, key=regime.get), "probabilities": regime,
                       "confidence": max(regime.values())},
            "direction": {"type": "choice", "choice": max(direction, key=direction.get), "probabilities": direction,
                          "confidence": max(direction.values())},
            "toxic_flow": {"type": "noul", "noul": toxic},
            "liquidity_stressed": {"type": "noul", "noul": stressed},
            "long_setup": {"type": "score", "probabilities": long_probs,
                           "score": sum(int(k) * p for k, p in long_probs.items()), "confidence": max(long_probs.values())},
            "short_setup": {"type": "score", "probabilities": short_probs,
                            "score": sum(int(k) * p for k, p in short_probs.items()), "confidence": max(short_probs.values())},
            "inventory_pressure": {"type": "score", "probabilities": i_probs,
                                   "score": sum(int(k) * p for k, p in i_probs.items()), "confidence": max(i_probs.values())},
        }
        return JevResult("mock", self.model, latency, normalize_answers(raw, questions),
                         {"input_tokens": 0, "output_tokens": 0})
