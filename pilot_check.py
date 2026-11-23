#!/usr/bin/env python3
"""Live pilot connection check - run it YOURSELF after setting up the API wallet (see LIVE_PILOT.md).

  .venv\\Scripts\\python.exe pilot_check.py              read-only: wallet, USDH balance, one live HIP-4 market
  .venv\\Scripts\\python.exe pilot_check.py --test-order a single immediate-or-cancel BUY of the minimum size at
                                                     0.1c - far below any real ask, so it cannot fill. It proves
                                                     signing and HIP-4 order routing work, without risking money.

It reads HL_ACCOUNT_ADDRESS and HL_AGENT_PRIVATE_KEY from .env and never prints the key."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-order", action="store_true", help="send one unfillable 0.1c IOC order")
    ap.add_argument("--network", default="mainnet", choices=["mainnet", "testnet"])
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    acct, key = os.getenv("HL_ACCOUNT_ADDRESS", "").strip(), os.getenv("HL_AGENT_PRIVATE_KEY", "").strip()
    if not acct or not key:
        print("  HL_ACCOUNT_ADDRESS / HL_AGENT_PRIVATE_KEY are not set in .env (see LIVE_PILOT.md)")
        return 2
    try:
        from helix.hip4_exec import SDKExchange
        ex = SDKExchange(args.network, acct, key)
    except ImportError as exc:
        print(f"  The Hyperliquid SDK is not installed ({exc}). Run: .venv\\Scripts\\pip install hyperliquid-python-sdk")
        return 2
    from eth_account import Account
    agent = Account.from_key(key).address
    print(f"  Account     : {acct}")
    print(f"  Agent wallet: {agent}  (this key can trade but not withdraw)")
    bal = ex.balances()
    print(f"  USDC (spot) : {bal.get('USDC', 0):,.2f}   (HIP-4 collateral since USDH was sunset, Oct 2026)")
    if bal.get("USDC", 0) < 10:
        print("  ! Less than $10 USDC in spot: HIP-4 orders need at least $10 each.")
    from helix.hip4 import parse_market, primary
    from helix.hip4_tape import _Every
    meta = ex.info.post("/info", {"type": "outcomeMeta"})
    now = time.time()
    ms = sorted((m for m in (parse_market(o, _Every()) for o in meta.get("outcomes", [])) if m and primary(m)
                 and m["expiry"] - now > 600), key=lambda m: m["expiry"])
    if not ms:
        print("  No HIP-4 price markets listed right now.")
        return 1
    m = ms[0]
    print(f"  Market      : #{m['id']} {m['coin']} > {m['strike']:g} at {time.strftime('%H:%M UTC', time.gmtime(m['expiry']))}")
    if not args.test_order:
        print("  Read-only check passed. Add --test-order to send one unfillable 0.1c order.")
        return 0
    coin, px, sz = f"#{m['id'] * 10}", 0.001, 10_000            # $10 at 0.1c: the minimum, and it cannot fill
    print(f"  Sending IOC BUY {sz} {coin} @ 0.1c (cannot fill) ...")
    resp = ex.ioc(coin, True, sz, px)
    print(f"  Response    : {resp}")
    st = (((resp or {}).get("response") or {}).get("data") or {}).get("statuses") or [{}]
    err = str(st[0].get("error", "")) if isinstance(st[0], dict) else ""
    if resp.get("status") == "ok" and "filled" not in st[0] and "could not immediately match" in err.lower():
        print("  Signing and HIP-4 order routing work (the exchange accepted the order and, as intended, found no match).")
        return 0
    print("  ! Unexpected response: do NOT enable live trading until this is understood.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
