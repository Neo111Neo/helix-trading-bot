"""Policy engine: code turns Jev's answers into an action. Thresholds live in config.yaml.
The reasons shown on the dashboard are written here, from Jev's numbers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .jev import setup_prob


@dataclass
class Action:
    kind: str  # ENTER_LONG | ENTER_SHORT | EXIT | HOLD | STAND_ASIDE
    size: float = 0.0  # 0..1 multiplier on the risk engine's size
    reasons: list[str] = field(default_factory=list)
    source: str = "jev"  # jev | mock | rules | fallback | late | budget | circuit
    confidence: float | None = None
    meta: dict = field(default_factory=dict)  # entry context for execution (Jev v2: signal minute, 15-min sigma)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "size": round(self.size, 2), "reasons": self.reasons, "source": self.source,
                "confidence": None if self.confidence is None else round(self.confidence, 3)}


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def flow_sign(feats: dict, dead_zone: float = 0.05) -> int:
    x = feats["flow_15s"] - 0.5
    return 0 if abs(x) < dead_zone else (1 if x > 0 else -1)


# Expected |move| given that the mid ends outside the neutral band (0.5 sigma) for a normal move:
# sigma * pdf(0.5) / (1 - cdf(0.5)) = 1.14 sigma. Expected signed move = (P(up) - P(down)) * EDGE_K * sigma.
EDGE_K = 1.14


def jev_policy(ans: dict, fsign: int, side: int, pol: dict, allow_short: bool, source: str = "jev",
               state: dict | None = None) -> Action:
    d = ans["direction"]["probabilities"]
    p_up, p_down = d.get("up", 0.0), d.get("down", 0.0)
    crisis = ans["regime"]["probabilities"].get("crisis", 0.0)
    toxic = ans["toxic_flow"]["p"]
    stress = ans["liquidity_stressed"]["p"]
    inv = ans["inventory_pressure"]
    sigma = float((state or {}).get("typical_move_bps") or 0.0)

    if side != 0:  # managing a position: exits have a lower bar than entries
        against = p_down if side > 0 else p_up
        word = "down" if side > 0 else "up"
        if crisis >= pol["max_crisis_prob"]:
            return Action("EXIT", 1.0, [f"Crisis regime {_pct(crisis)}"], source, crisis)
        hold = float((state or {}).get("position_hold_s") or 0.0)
        min_hold = float(pol.get("min_hold_s", 0))
        if hold < min_hold:
            # Jev's per-second answers jitter, and a fresh position always shows a small loss (spread + fee),
            # which reads as "reduce now". Signal exits inside this window paid round-trip costs on noise.
            # The broker's stop and target stay active the whole time.
            return Action("HOLD", 0.0, [f"Min hold {min_hold:.0f}s ({hold:.0f}s so far): stop and target still active"],
                          source, 1 - against)
        if inv["score"] >= pol["exit_inventory_pressure"]:
            return Action("EXIT", 1.0, [f"Inventory pressure {inv['score']:.1f}/3 ≥ {pol['exit_inventory_pressure']}"],
                          source, inv["confidence"])
        if against >= pol["exit_direction_prob"]:
            return Action("EXIT", 1.0, [f"Bias flipped {word} {_pct(against)} ≥ {_pct(pol['exit_direction_prob'])}"],
                          source, against)
        return Action("HOLD", 0.0, [f"Holding: {word} risk {_pct(against)}, pressure {inv['score']:.1f}/3"], source,
                      1 - against)

    if crisis >= pol["max_crisis_prob"]:
        return Action("STAND_ASIDE", 0.0, [f"Crisis regime {_pct(crisis)} ≥ {_pct(pol['max_crisis_prob'])}"], source, crisis)
    if stress >= pol["max_liquidity_stress"]:
        return Action("STAND_ASIDE", 0.0, [f"Thin book: liquidity stress {_pct(stress)} ≥ {_pct(pol['max_liquidity_stress'])}"],
                      source, stress)

    # Evaluate LONG and SHORT symmetrically; each side has its own setup question.
    candidates = []
    for want in ((1, -1) if allow_short else (1,)):
        p_dir, p_opp = (p_up, p_down) if want > 0 else (p_down, p_up)
        word, move = ("LONG", "UP") if want > 0 else ("SHORT", "DOWN")
        setup = setup_prob(ans.get("long_setup" if want > 0 else "short_setup"))
        edge = (p_dir - p_opp) * EDGE_K * sigma
        fails, notes = [], []
        if p_dir >= pol["min_direction_prob"]:
            notes.append(f"Bias {move} {_pct(p_dir)} ≥ {_pct(pol['min_direction_prob'])}")
        else:
            fails.append(f"Bias {move} only {_pct(p_dir)} (< {_pct(pol['min_direction_prob'])})")
        if setup >= pol["min_setup_prob"]:
            notes.append(f"{word} setup supported {_pct(setup)}")
        else:
            fails.append(f"{word} setup supported only {_pct(setup)} (< {_pct(pol['min_setup_prob'])})")
        if sigma > 0:
            if edge >= pol["min_edge_bps"]:
                notes.append(f"Expected edge {edge:+.1f} bps ≥ {pol['min_edge_bps']:g}")
            else:
                fails.append(f"{word} expected edge {edge:+.1f} bps (< {pol['min_edge_bps']:g} bps, about the round-trip cost)")
        if toxic >= pol["toxic_flow_threshold"]:
            if fsign == -want:
                fails.append(f"Informed flow {_pct(toxic)} is against the {word.lower()}")
            elif fsign == want:
                notes.append(f"Informed flow {_pct(toxic)} is with the trade")
        candidates.append((want, p_dir, setup, edge, fails, notes))

    ready = [c for c in candidates if not c[4]]
    if ready:
        want, p_dir, setup, edge, _, notes = max(ready, key=lambda c: c[3])
        full = p_dir >= pol["full_size_prob"]
        notes.append("Full size" if full else f"Half size: bias under {_pct(pol['full_size_prob'])}")
        return Action("ENTER_LONG" if want > 0 else "ENTER_SHORT", 1.0 if full else 0.5, notes, source, p_dir)
    # Nothing qualifies: report the side that came closest (highest expected edge).
    want, p_dir, _setup, _edge, fails, _notes = max(candidates, key=lambda c: (c[3], c[1]))
    return Action("STAND_ASIDE", 0.0, fails, source, p_dir)


def signal_policy(sig: dict | None, ans: dict | None, side: int, pol: dict, entry_prob: float, allow_short: bool,
                  source: str = "jev") -> Action:
    """Jev v2: entries from the once-a-minute signal (P(up over the horizon), decoded in code from the present-tense
    class Jev picked); the 5-second battery only vetoes (crisis regime, thin book). Exits are the broker's stop,
    take-profit and max hold (= the forecast horizon), plus a crisis exit: the backtest measured the move AT the
    horizon, so exiting earlier on a re-read signal would trade a different, untested strategy."""
    crisis = (ans or {}).get("regime", {}).get("probabilities", {}).get("crisis", 0.0)
    stress = (ans or {}).get("liquidity_stressed", {}).get("p", 0.0)
    if side != 0:
        if crisis >= pol["max_crisis_prob"]:
            return Action("EXIT", 1.0, [f"Crisis regime {_pct(crisis)}"], source, crisis)
        return Action("HOLD", 0.0, ["Holding to the forecast horizon: stop, target and max hold active"], source)
    if sig is None:
        return Action("STAND_ASIDE", 0.0, ["Jev v2 signal not ready (waiting for the next 1-minute candle)"], source)
    if crisis >= pol["max_crisis_prob"]:
        return Action("STAND_ASIDE", 0.0, [f"Crisis regime {_pct(crisis)} ≥ {_pct(pol['max_crisis_prob'])}"], source, crisis)
    if stress >= pol["max_liquidity_stress"]:
        return Action("STAND_ASIDE", 0.0, [f"Thin book: liquidity stress {_pct(stress)} ≥ {_pct(pol['max_liquidity_stress'])}"],
                      source, stress)
    if sig.get("momentum_regime"):
        return Action("STAND_ASIDE", 0.0, [f"Momentum regime: moves kept going {_pct(sig.get('continuation') or 0)} of the "
                                           f"time this week; Jev v2 fades moves, so it sits out"], source)
    p, h = sig["p_up"], sig.get("horizon_min", 15)
    cls = f" · Jev: {sig['class']}" if sig.get("class") else ""
    # the forecast window starts at the signal's minute (the holding clock follows it), and the signal's candle-based
    # 15-min move floors the stop/target/size sigma (the 1-s book estimate runs low on volatile alts)
    meta = {"signal_ts": sig.get("signal_ts") or sig.get("ts"), "sigma_bps": sig.get("sigma_bps")}
    eps = 1e-9   # 1 - 0.56 is 0.43999999999999995 in floating point: without this, p = 0.44 could never short
    if p >= entry_prob - eps:
        return Action("ENTER_LONG", 1.0, [f"Jev v2 P(up {h}m) {_pct(p)} ≥ {_pct(entry_prob)}{cls}"], source, p, meta)
    if allow_short and p <= 1 - entry_prob + eps:
        return Action("ENTER_SHORT", 1.0, [f"Jev v2 P(down {h}m) {_pct(1 - p)} ≥ {_pct(entry_prob)}{cls}"], source,
                      1 - p, meta)
    return Action("STAND_ASIDE", 0.0, [f"Bias not confident: Jev v2 P(up {h}m) {_pct(p)}, needs ≥ {_pct(entry_prob)} "
                                       f"or ≤ {_pct(1 - entry_prob)}{cls}"], source, max(p, 1 - p))


def rules_policy(f: dict, side: int, rp: dict, allow_short: bool, source: str = "rules", hold_s: float = 0.0) -> Action:
    """Plain hand-written rules: the baseline book, and Jev's fallback."""
    mom, imb, flow = f["ret_30s"], f["imbalance_5"], f["flow_15s"]
    thr = rp["momentum_bps"]
    min_hold = float(rp.get("min_hold_s", 0))
    if side != 0 and hold_s < min_hold:
        # 30 s momentum flips constantly; exiting on it paid round-trip fees within seconds.
        # Stop-loss and take-profit stay active in the broker during this window.
        return Action("HOLD", 0.0, [f"Min hold {min_hold:.0f}s ({hold_s:.0f}s so far): stop and target still active"], source)
    if side > 0:
        if mom < -thr or flow < 1 - rp["min_flow"]:
            return Action("EXIT", 1.0, [f"Momentum reversed ({mom:+.1f} bps / 30s)"], source)
        return Action("HOLD", 0.0, ["Momentum intact"], source)
    if side < 0:
        if mom > thr or flow > rp["min_flow"]:
            return Action("EXIT", 1.0, [f"Momentum reversed ({mom:+.1f} bps / 30s)"], source)
        return Action("HOLD", 0.0, ["Momentum intact"], source)
    calm = f["vol_ratio"] < 2.5
    if mom > thr and imb > rp["min_imbalance"] and flow > rp["min_flow"] and calm:
        return Action("ENTER_LONG", 1.0, [f"30s momentum {mom:+.1f} bps, bids heavier, buyers {flow:.0%}"], source)
    if allow_short and mom < -thr and imb < -rp["min_imbalance"] and flow < 1 - rp["min_flow"] and calm:
        return Action("ENTER_SHORT", 1.0, [f"30s momentum {mom:+.1f} bps, asks heavier, sellers {1 - flow:.0%}"], source)
    if abs(mom) <= thr:
        why = f"Momentum {mom:+.1f} bps under {thr:g}"
    elif not calm:
        why = "Volatility spike"
    elif (mom > 0 and imb <= rp["min_imbalance"]) or (mom < 0 and imb >= -rp["min_imbalance"]):
        why = "Book imbalance doesn't confirm"
    else:
        why = "Flow doesn't confirm"
    return Action("STAND_ASIDE", 0.0, [why], source)


_REASON_KEYS = (
    ("crisis", "Crisis regime"), ("thin book", "Thin book"), ("informed flow", "Informed flow against"),
    ("bias", "Bias not confident"), ("setup", "Weak setup"), ("expected edge", "Edge below cost"),
    ("entry quality", "Low entry quality"), ("momentum regime", "Momentum regime (Jev v2 sits out)"),
    ("not ready", "Jev v2 signal not ready"), ("momentum", "No momentum"),
    ("imbalance", "Book doesn't confirm"), ("flow doesn't", "Flow doesn't confirm"), ("volatility", "Volatility spike"),
)


def reason_key(text: str) -> str:
    """Groups reasons that differ only in their numbers, for the 'why no trade' counters."""
    low = text.lower()
    for needle, label in _REASON_KEYS:
        if needle in low:
            return label
    return text[:40]
