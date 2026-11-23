"""Go-live gate: the evidence the HIP-4 market maker must show on paper (and in shadow) before live trading can even
be switched on, plus the operator's own confirmations. Nothing here trades; it only reports.

Automatic checks (thresholds in config hip4_live.gate):
  paper_days      at least `days` of continuous paper trading (a reset restarts the clock)
  settled         at least `settled` markets settled
  settled_pnl     settled P&L above zero (what was actually paid out, not marks)
  markout         60 s markout above zero over the last `markout_n` fills (we are not being picked off)
  no_kills        the drawdown kill switch did not fire in the last `days` days
  backtest        the latest automatic market-maker backtest covers >= `backtest_hours` of tape and made money
  shadow          >= `shadow_validity` of paper quotes would be legal live orders, and the order churn would not run
                  the exchange's address budget dry within 30 days

Operator confirmations (the page's checkboxes; nobody else can tick them):
  legal           HIP-4 prediction markets are legal for me to trade where I live
  wallet          I created a dedicated API (agent) wallet that can trade but cannot withdraw
  funded          the pilot account is funded with USDC in spot (USDH was sunset in Oct 2026)
  loss_ok         I can afford to lose the whole pilot amount"""

from __future__ import annotations

import json
import time
from pathlib import Path

CONFIRMATIONS = {
    "legal": "HIP-4 prediction markets are legal for me to trade where I live",
    "wallet": "I created a dedicated API (agent) wallet that can trade but cannot withdraw, and I keep its key myself",
    "funded": "The pilot account is funded with USDC in spot (HIP-4's collateral since USDH was sunset)",
    "loss_ok": "I can afford to lose the whole pilot amount",
}


class GoLiveGate:
    def __init__(self, cfg: dict, mm, backtests, shadow, data_dir: Path | None, clock=time.time):
        self.g = cfg["hip4_live"]["gate"]
        self.mm, self.backtests, self.shadow, self.clock = mm, backtests, shadow, clock
        self.strategy = "sniper" if getattr(mm, "strategy", "maker") == "sniper" else "mm"
        # only backtests on tape recorded after the rules were fixed count (out-of-sample)
        self.holdout_from = float((cfg.get("hip4_sniper") or {}).get("holdout_from", 0)) if self.strategy == "sniper" else 0.0
        self.path = Path(data_dir) / "golive.json" if data_dir is not None else None
        self.confirmed: dict[str, float | None] = {k: None for k in CONFIRMATIONS}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}
            self.confirmed.update({k: v for k, v in saved.get("confirmed", {}).items() if k in CONFIRMATIONS})
        except (OSError, ValueError):
            pass

    def confirm(self, key: str, value: bool) -> None:
        if key not in CONFIRMATIONS:
            raise ValueError(f"unknown confirmation {key!r}")
        self.confirmed[key] = self.clock() if value else None
        if self.path is not None:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"confirmed": self.confirmed}), encoding="utf-8")
            tmp.replace(self.path)

    @staticmethod
    def _check(key, label, ok, progress, detail):
        return {"key": key, "label": label, "ok": bool(ok), "progress": round(max(0.0, min(1.0, progress)), 3),
                "detail": detail}

    def evaluate(self) -> dict:
        now, g, mm = self.clock(), self.g, self.mm
        checks = []
        with mm.lock:
            since = mm.since
            settled = list(mm.settled)
            mo = [f["markout"] for f in mm.fills if f.get("markout") is not None][-int(g["markout_n"]):]
            kills = [t for t in mm.kill_log if now - t <= g["days"] * 86400]
        days = (now - since) / 86400 if since else 0.0
        checks.append(self._check("paper_days", f"{g['days']:g} days of paper trading", days >= g["days"], days / g["days"],
                                  f"{days:.1f} of {g['days']:g} days" if since else "no paper fills yet"))
        checks.append(self._check("settled", f"{g['settled']:g} settled markets", len(settled) >= g["settled"],
                                  len(settled) / g["settled"], f"{len(settled)} of {g['settled']:g}"))
        spnl = sum(s["pnl"] for s in settled)
        checks.append(self._check("settled_pnl", "Settled P&L above zero", spnl > 0 and len(settled) >= 20,
                                  1.0 if spnl > 0 else 0.0, f"${spnl:+,.2f} over {len(settled)} markets"))
        avg_mo = sum(mo) / len(mo) if mo else None
        checks.append(self._check("markout", f"60 s markout above zero ({g['markout_n']:g} fills)",
                                  avg_mo is not None and avg_mo > 0 and len(mo) >= g["markout_n"],
                                  len(mo) / g["markout_n"] if (avg_mo or 0) > 0 else 0.0,
                                  (f"{avg_mo * 100:+.2f}c over {len(mo)} fills" if avg_mo is not None else "no markouts yet")))
        checks.append(self._check("no_kills", f"No kill-switch trip in {g['days']:g} days", not kills and days >= g["days"],
                                  0.0 if kills else min(1.0, days / g["days"]),
                                  f"{len(kills)} trip(s)" if kills else "none so far"))
        # one-sided profit is the warning sign from the first sniper backtest (selling Yes made it all): settled P&L
        # has to be positive on BOTH buying and selling Yes
        sides = {1: 0.0, -1: 0.0}
        with mm.lock:
            ys = {s["id"]: s.get("y") for s in mm.settled}
            for f in mm.fills:
                y = ys.get(f["id"])
                if y is not None and f.get("how") != "take-profit" and f.get("how") != "stop-loss":
                    sides[f["side"]] += f["side"] * (y - f["px"]) * f["qty"]
        checks.append(self._check("two_sided", "Makes money buying AND selling Yes", sides[1] > 0 and sides[-1] > 0,
                                  (sides[1] > 0) * 0.5 + (sides[-1] > 0) * 0.5,
                                  f"buy Yes ${sides[1]:+,.2f} · sell Yes ${sides[-1]:+,.2f}"))
        bt = None
        if self.backtests is not None:
            with self.backtests.lock:
                runs = [r for r in self.backtests.results if r.get("strategy") == self.strategy and r.get("label") == "auto"
                        and r.get("from", 0) >= self.holdout_from - 1]
            bt = runs[-1] if runs else None
        bh = float(bt["hours"]) if bt else 0.0
        name = "Sniper" if self.strategy == "sniper" else "Market-maker"
        held = " held-out" if self.holdout_from else ""
        checks.append(self._check("backtest", f"{name} backtest on {g['backtest_hours']:g}+ h of{held} tape makes money",
                                  bt is not None and bh >= g["backtest_hours"] and bt["pnl"] > 0,
                                  (bh / g["backtest_hours"]) if bt is None or bt["pnl"] > 0 else 0.0,
                                  f"{bh:.1f} h, ${bt['pnl']:+,.2f}" if bt else "no automatic run yet"))
        sh = self.shadow.to_dict() if self.shadow is not None else None
        val = sh["validity"] if sh else None
        checks.append(self._check("shadow", f"Shadow: {g['shadow_validity'] * 100:g}%+ of quotes are legal live orders, budget lasts",
                                  sh is not None and val is not None and val >= g["shadow_validity"] and sh["budget"]["ok"],
                                  (val or 0) / g["shadow_validity"],
                                  (f"{val * 100:.0f}% legal · budget " + (f"empty in {sh['budget']['days_until_empty']:g} d"
                                                                           if sh["budget"]["days_until_empty"] is not None else "self-sustaining"))
                                  if sh and val is not None else "shadow executor not running yet"))
        manual = [{"key": k, "label": v, "ok": self.confirmed.get(k) is not None, "at": self.confirmed.get(k)}
                  for k, v in CONFIRMATIONS.items()]
        auto_ok = sum(c["ok"] for c in checks)
        ready = auto_ok == len(checks) and all(m["ok"] for m in manual)
        return {"ready": ready, "passed": auto_ok + sum(m["ok"] for m in manual), "total": len(checks) + len(manual),
                "checks": checks, "confirmations": manual, "live_enabled": False,
                "note": "Live trading is not built yet: passing the gate unlocks building and testing the tiny pilot."}
