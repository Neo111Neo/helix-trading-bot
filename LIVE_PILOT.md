# HIP-4 live pilot — runbook

The live pilot sends **real orders with real money**: one immediate-or-cancel order per sniper signal, inside hard
limits. It does **nothing** until every step below is done. You do these steps yourself. Nobody else should ever
handle your keys, Claude included.

## 0. Before you start
- [ ] The **Go-live readiness** meter on the HIP-4 page shows every automatic check passed:
  - 48 h of paper trading
  - 200 settled markets
  - settled P&L above zero
  - positive markout
  - no kill-switch trip
  - money made buying **and** selling Yes
  - a profitable held-out backtest
  - the shadow checks
- [ ] You've ticked the four confirmations on that page: legal where you live, an agent wallet, USDC funding, money you can lose.
- [ ] You accept that paper results can be better than live. Paper assumes we always get the price we saw, and live
      fills will be partial and sometimes missed. The pilot exists to measure that gap.

## 1. Create an API ("agent") wallet: it can trade but cannot withdraw
1. Open app.hyperliquid.xyz with your main wallet, then go to **More → API**.
2. Generate an API wallet, give it a name, and authorize it. Copy its **private key**; it's shown once.
3. Your **account address** is your main wallet's address, not the agent's.

## 2. Fund the pilot with USDC (in Spot)
- HIP-4 markets trade in **USDC**. USDH used to be the collateral, but it was sunset in October 2026 and all 297
  listed outcomes now quote in USDC. **Don't buy USDH**: its order book is thin and unreliable.
- Keep the USDC in your **Spot** balance. Put in **only the pilot amount** (`hip4_live.pilot.capital_usd`, $150 by
  default) plus a small buffer.

## 3. Install the SDK (one time)
```
.venv\Scripts\pip install hyperliquid-python-sdk
```
If it won't install on this Python (3.14), make a Python 3.12 virtualenv and install `requirements.txt` there too.

## 4. Add the keys to `.env` (never share them, never paste them into chat)
```
HL_ACCOUNT_ADDRESS=0xYourMainWalletAddress
HL_AGENT_PRIVATE_KEY=0xTheAgentWalletPrivateKey
```

## 5. Check the connection, safely
```
.venv\Scripts\python.exe pilot_check.py
.venv\Scripts\python.exe pilot_check.py --test-order
```
The second command sends **one unfillable order** (a bid at 0.1¢ that can never match). It proves signing and HIP-4
order routing work. Don't continue unless it says *"Signing and HIP-4 order routing work"*.

## 6. Switch it on
In `config.yaml`:
```yaml
hip4_live:
  live_enabled: true
  pilot:
    capital_usd: 150      # never more than this open in total
    max_order_usd: 25     # largest single order
    max_open_usd: 100
    daily_loss_usd: 30    # down this much in a UTC day -> it stops itself until tomorrow
    contracts: 0          # 0 = the $10 minimum per order
```
Restart Helix. The **Live pilot** panel on the HIP-4 page turns **● LIVE** when nothing blocks it. Check your first
order on app.hyperliquid.xyz.

## Stopping
- **KILL** on the terminal page stops the pilot and the paper books at once. The pilot only sends immediate-or-cancel
  orders, so nothing is left resting on the exchange.
- Or set `live_enabled: false` and restart.
- Open positions settle automatically at expiry. HIP-4 has no claim step.

## Scaling
Run the pilot at the minimum size for about two weeks. Only raise `contracts` / `capital_usd` if the live results
(the **Live pilot** panel and `data/live_orders.jsonl`) match paper closely: fill rate, edge taken, and settled P&L.
