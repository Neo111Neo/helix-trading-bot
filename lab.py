#!/usr/bin/env python3
"""Jev Lab - backtest how Jev is asked, on real Hyperliquid history (paper research, never trades).

  python lab.py fetch                 download candles + funding into data/lab/
  python lab.py build                 build point-in-time states for the tasks
  python lab.py claude --max-usd 15   Claude forecasts for the same states (capped)
  python lab.py jev [--only NAME]     ask every variant in <dir>/variants/ (~$0.00009 per call)
  python lab.py score                 design/holdout scores -> <dir>/report.json
  python lab.py all                   all of the above in order

  --dir data/lab/full --coins BTC,ETH,...   a separate study (its own candles, states, answers and report)
  --tasks D,K,K15                           D = direction 5/15 min, K = next-hour strikes, K15 = 15-min strikes"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Jev Lab")
    ap.add_argument("step", choices=["fetch", "build", "claude", "jev", "score", "all"])
    ap.add_argument("--max-usd", type=float, default=15.0, help="Claude spending cap for this lab (cumulative)")
    ap.add_argument("--only", default=None, help="run one variant")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dir", default="data/lab", help="study folder (relative to the project)")
    ap.add_argument("--coins", default=None, help="comma-separated perps (default: the original six)")
    ap.add_argument("--tasks", default="D,K", help="comma-separated: D, K, K15")
    ap.add_argument("--skip-existing", action="store_true", help="fetch: keep coins already downloaded")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from helix import lab
    from helix.config import load_config
    from helix.hyperliquid import URLS, HLRest

    cfg = load_config(str(ROOT / "config.yaml"))
    lab_dir = ROOT / args.dir
    lab_dir.mkdir(parents=True, exist_ok=True)
    coins = [c.strip() for c in args.coins.split(",")] if args.coins else list(lab.DIRECTION_COINS)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    steps = ["fetch", "build", "claude", "jev", "score"] if args.step == "all" else [args.step]

    if "fetch" in steps:
        print("Fetching history ...")
        lab.fetch_all(HLRest(URLS[cfg["network"]][0], timeout=20), coins, lab_dir, skip_existing=args.skip_existing)
    if "build" in steps:
        print("Building states ...", lab.build_all(lab_dir, coins, tasks))
    if not set(steps) & {"claude", "jev", "score"}:
        return 0
    states = {t: lab.load_states(lab_dir, t) for t in tasks}
    if "claude" in steps:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        ws = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        client = anthropic.Anthropic(api_key=key, timeout=180.0, max_retries=3,
                                     default_headers={"anthropic-workspace-id": ws} if ws else None)
        fc = cfg["claude_forecast"]
        print("Claude forecasts (strike tasks: every state; direction task: every other 10-minute state) ...")
        for t in tasks:
            lab.run_claude(client, fc["model"], lab_dir, t, states[t], args.max_usd,
                           fc["price_in_per_million"], fc["price_out_per_million"],
                           subsample=(lambda r: (r["ts"] // lab.STEP_D_S) % 2 == 0) if t == "D" else None)
    claude = {t: lab.claude_fields(lab_dir, t) for t in tasks}
    variants = lab.load_variants(lab_dir)
    if args.only:
        variants = {k: v for k, v in variants.items() if k == args.only}
    if "jev" in steps:
        from helix.jev import JevClient
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        if not key:
            print("No TYPESAFE_API_KEY in .env")
            return 2
        client = JevClient(key, cfg["jev"]["model"], 8000)
        for name, spec in variants.items():
            usable = [t for t in tasks if lab.supports(spec, t)]
            for task in usable:
                lab.render(spec, task)   # validate before spending anything
            print(f"Jev variant {name} ({', '.join(usable)}) ...")
            for task in usable:
                lab.run_jev(client, lab_dir, task, spec, states[task], claude[task], workers=args.workers)
    if "score" in steps:
        variants = lab.load_variants(lab_dir)
        print("Scoring ...")
        report = {}
        if "D" in tasks:
            dv = {n: v for n, v in variants.items() if lab.supports(v, "D")}
            report["D"] = lab.score_direction(lab_dir, states["D"], dv, claude["D"])
            report["D"]["per_coin"] = lab.per_coin_direction(lab_dir, states["D"], dv)
        for t in ("K", "K15"):
            if t in tasks:
                report[t] = lab.score_hourly(lab_dir, states[t], {n: v for n, v in variants.items() if lab.supports(v, t)},
                                             claude[t], task=t)
        (lab_dir / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(lab.summarize(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
