"""HIP-4 LIVE PILOT executor for the sniper - OFF by default, and it stays off until EVERY one of these holds:

  1. config hip4_live.live_enabled: true        (only the operator edits config.yaml)
  2. the go-live gate reports ready             (paper evidence + the operator's four confirmations)
  3. HL_ACCOUNT_ADDRESS and HL_AGENT_PRIVATE_KEY are in .env  (an API/agent wallet: it can trade, it can NOT withdraw)
  4. the official hyperliquid-python-sdk is installed

Then, for every sniper signal, it sends ONE immediate-or-cancel order (it takes what is there, nothing rests) at the
price the sniper saw, sized at the exchange minimum ($10) unless the pilot size says more, and within hard pilot limits:

  max_order_usd     largest single order
  max_open_usd      total cost of open pilot positions
  daily_loss_usd    realised + marked loss for the UTC day -> the pilot stops itself until tomorrow (or a manual resume)
  capital_usd       the pilot never spends more than this in total open cost

The kill switch (the terminal's KILL button) stops it at once. Every order, response and refusal is logged to
data/live_orders.jsonl. The private key is read from the environment by the SDK client and never logged, printed,
stored or sent anywhere except as the signature on the operator's own orders.

HIP-4 specifics (Hyperliquid docs + Chainstack's HIP-4 guide): coin "#<10*outcome + side>", asset 100_000_000 +
encoding, whole contracts, $10 minimum, USDC collateral (USDH was sunset in Oct 2026), a Yes sell we can't cover from held Yes is a No buy at
1 - price. The SDK (0.2x) has no HIP-4 table, so the outcome coins are added to its coin -> asset map."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

ASSET_BASE = 100_000_000
GIVE_UP_S = 7 * 86400        # = hip4.GIVE_UP_S: builders must settle within a week


class SDKExchange:
    """The thin, real exchange client: hyperliquid-python-sdk, imported only when live trading is armed."""

    def __init__(self, network: str, account: str, key: str):
        from eth_account import Account                     # noqa: imported only when armed
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL
        wallet = Account.from_key(key)
        self.info = Info(url, skip_ws=True)
        self.exchange = Exchange(wallet, url, account_address=account)
        self.account = account

    def _register(self, coin: str) -> None:
        enc = int(coin[1:])
        for info in (self.info, self.exchange.info):
            info.coin_to_asset[coin] = ASSET_BASE + enc
            info.name_to_coin[coin] = coin
            info.asset_to_sz_decimals[ASSET_BASE + enc] = 0             # whole contracts

    def ioc(self, coin: str, is_buy: bool, sz: int, px: float) -> dict:
        self._register(coin)
        return self.exchange.order(coin, is_buy, sz, px, {"limit": {"tif": "Ioc"}})

    def balances(self) -> dict:
        st = self.info.spot_user_state(self.account)
        return {b["coin"]: float(b["total"]) for b in st.get("balances", [])}

    def snapshot(self) -> dict:
        """ONE read of the account: USDC and every outcome token held ("+<encoding>": contracts, entry cost). Cash and
        positions come from the same instant, so a settlement (tokens gone, payout in) can never be counted twice."""
        st = self.info.spot_user_state(self.account)
        usdc, outcomes = 0.0, {}
        for b in st.get("balances", []):
            coin, total = str(b.get("coin", "")), float(b.get("total") or 0)
            if coin == "USDC":
                usdc = total
            elif coin.startswith("+") and coin[1:].isdigit() and total > 0:
                outcomes[int(coin[1:])] = (total, float(b.get("entryNtl") or 0))
        return {"usdc": usdc, "outcomes": outcomes}


class LiveExecutor:
    def __init__(self, cfg: dict, sniper, gate, data_dir: Path | None, exchange=None, env=os.environ, clock=time.time):
        lc = cfg["hip4_live"]
        self.armed_cfg = bool(lc.get("live_enabled", False))
        # the operator may choose to go live before the gate passes; every other limit still applies, and the page
        # and the order log say so on every order
        self.require_gate = bool(lc.get("require_gate", True))
        p = lc["pilot"]
        self.max_order = float(p["max_order_usd"])
        self.max_open = float(p["max_open_usd"])
        self.daily_loss = float(p["daily_loss_usd"])
        self.capital = float(p["capital_usd"])
        self.contracts = int(p["contracts"])                 # 0 = just the $10 minimum
        self.cooldown_s = float(p.get("cooldown_s", 15))
        # speed: "fill" sends after the paper sniper's simulated fill (signal + its latency, then the trip to the
        # exchange: ~0.5-0.8 s, too late for most stale quotes); "signal" sends the moment the signal fires, when the top
        # of the book alone meets the $10 minimum. slip: how much worse than the quote we saw the IOC may pay, but never
        # closer than min_edge to our fair (0 = exactly the quote, as before)
        self.fire_on = str(p.get("fire_on", "fill"))
        self.slip = float(p.get("slip", 0.0))
        self.min_edge = float(p.get("min_edge", 0.03))
        # live size multiplier (the operator clicks it on the page): scales order size, the per-order and open limits
        # and the daily stop together; total open cost can never pass capital_usd
        # per market: stacking one market (2026-10-07: $91 of No in one BTC market) does not raise the win rate, it only
        # makes one bad market cost a day's wins and leaves no room under max_open_usd for other markets
        self.max_market = float(p.get("max_market_usd", 30))
        # long-dated markets (> long_tte_min to expiry) get only long_budget_usd of the open budget: they filled the
        # whole $100 and blocked the 3-60 min markets, which earn ~30% of capital per hour vs ~4% (research 2026-10-07)
        self.long_tte_s = float(p.get("long_tte_min", 120)) * 60
        self.long_budget = float(p.get("long_budget_usd", 40))
        self.base = {"max_order_usd": self.max_order, "max_open_usd": self.max_open, "daily_loss_usd": self.daily_loss,
                     "max_market_usd": self.max_market, "long_budget_usd": self.long_budget,
                     "capital_usd": self.capital}
        self.max_mult = float(p.get("max_multiplier", 5))
        self.size_mult = 1.0
        self.cool: dict[int, float] = {}
        # one account snapshot, replaced whole (never field by field) so USDC and the tokens always come from the same
        # read; "started" orders concurrent reads, "ts" is zeroed to force a fresh read after a fill or settlement
        self._wallet = {"ts": 0.0, "started": 0.0, "usdc": None, "error": None, "outcomes": None}
        self.min_notional = float(lc["min_notional"])
        self.order_usd = self.min_notional
        # compounding: the budgets follow the account (wallet + open positions at cost) as fractions of it, so the pilot
        # grows with its winnings and shrinks after losses. Research 2026-10-07: the open budget is what binds (it
        # refused most attempts); bigger single orders do NOT pay (thin books), so orders grow only slowly
        cp = p.get("compound") or {}
        self.compound = bool(cp.get("enabled", False))
        self.floor_fixed = bool(cp.get("floor_to_fixed", False))     # never below max_open_usd / max_market_usd
        self.cf = {k: float(cp.get(k, d)) for k, d in (("open", 0.53), ("order", 0.02), ("market", 0.16), ("long", 0.21),
                                                        ("daily_loss", 0.32), ("max_open_frac", 0.9))}
        self.equity_basis: float | None = None              # wallet + open cost, refreshed in the background
        self._eq_at, self._eq_busy = 0.0, False
        self.network = cfg.get("network", "mainnet")
        self.sniper, self.gate, self.clock, self.env = sniper, gate, clock, env
        self.path = Path(data_dir) / "live_orders.jsonl" if data_dir is not None else None
        self.state_path = Path(data_dir) / "live_pilot.json" if data_dir is not None else None
        self._exchange = exchange                            # tests inject a fake; live builds SDKExchange lazily
        self._ex_lock = threading.Lock()
        self._warming = False
        self.lock = threading.Lock()
        self.pos: dict[int, dict] = {}                       # market id -> {yes, no (contracts held), cost (cash spent)}
        self.realized_today, self.day = 0.0, int(clock() // 86400)
        self.realized_total = 0.0
        self.halted, self.halt_reason = False, None
        self.orders: list[dict] = []
        self.last_refusal = None
        self._load()
        self._settle_retry: dict[int, float] = {}
        self._settle_tries: dict[int, int] = {}
        # DRY RUN (zero-risk test, 2026-10-09): do everything a live order does except send it, and score whether it
        # would have filled and what it would have made (see _dry_send). Never sends while dry_run is true.
        self.dry_run = bool(p.get("dry_run", False))
        self.dry_shadow = bool(p.get("dry_shadow", False))      # live AND the zero-risk scoring side by side
        self.dry = {"pos": {}, "meta": {}, "orders": [], "settled": [], "realized": {}, "won": {}, "markets": {},
                    "since": clock(), "stats": {"signals": 0, "sent": 0, "refused": 0, "resolved": 0,
                                                **{f"filled_{mg}": 0 for mg in self.DRY_MARGINS_MS}}}
        self._dry_retry: dict[int, tuple] = {}
        if self.dry_run or self.dry_shadow:
            self._load_dry()
        self._marks: dict[int, float] = {}                   # last good Yes mid per market (books thin out near expiry)
        if sniper is not None:
            sniper.on_signal = self.on_signal
            sniper.on_step = self.poll
            sniper.live_fire_on = self.fire_on

    def poll(self, now: float) -> None:
        """Each sniper step: settle pilot positions whose market has expired (settledOutcome, backed off)."""
        self._roll_day(now)
        if self.armed_cfg and self._exchange is None and not self._warming and not self.blockers():
            # build the SDK client now, in the background: building it inside the first order cost that order ~30 s
            # (the SDK downloads the exchange's market tables) - 2026-10-08, the first live IOC after every restart
            self._warming = True
            threading.Thread(target=self._warm, daemon=True).start()
        if self.compound and now - self._eq_at >= 60 and not self._eq_busy:
            self._eq_at, self._eq_busy = now, True            # the wallet read is a network call: never in this thread
            threading.Thread(target=self._refresh_equity, daemon=True).start()
        rest = getattr(self.sniper, "rest", None)
        if self.dry_run or self.dry_shadow:
            self._dry_poll(now, rest)
        for mid, p in list(self.pos.items()):
            # first ask at expiry + 120 s (every live settlement so far answered on that first try), then back off
            if not p.get("expiry") or now < p["expiry"] + 120 or now < self._settle_retry.get(mid, 0):
                continue
            tries = self._settle_tries.get(mid, 0)
            self._settle_tries[mid] = tries + 1
            self._settle_retry[mid] = now + min(3600.0, 30.0 * 2 ** tries)      # set BEFORE parsing: never a 0.5 s loop
            try:
                res = rest.post({"type": "settledOutcome", "outcome": int(mid)}) if rest is not None else None
            except Exception:
                res = None
            try:
                y = float(res.get("settleFraction")) if isinstance(res, dict) else math.nan
            except (TypeError, ValueError):
                y = math.nan
            if math.isfinite(y):
                self.settle(mid, y)
            elif now > p["expiry"] + GIVE_UP_S:          # never settled: void (refunded), as the paper book does
                self.settle(mid, None)
            # otherwise not settled yet (or an unexpected reply): keep the position and ask again later - an early or
            # odd reply must never be booked as a void and lose the real result

    # ------------------------------------------------------------------ arming
    def blockers(self) -> list[str]:
        """Every reason live orders cannot be sent right now (empty = armed)."""
        out = []
        if not self.armed_cfg:
            out.append("hip4_live.live_enabled is false in config.yaml")
        if self.require_gate and (self.gate is None or not self.gate.evaluate()["ready"]):
            out.append("the go-live gate is not ready")
        if not self.env.get("HL_ACCOUNT_ADDRESS") or not self.env.get("HL_AGENT_PRIVATE_KEY"):
            out.append("HL_ACCOUNT_ADDRESS / HL_AGENT_PRIVATE_KEY are not set in .env")
        if self._exchange is None:
            import importlib.util
            if importlib.util.find_spec("hyperliquid") is None or importlib.util.find_spec("eth_account") is None:
                out.append("hyperliquid-python-sdk is not installed")
        if self.halted:
            out.append(f"pilot halted: {self.halt_reason}")
        if self.sniper is not None and self.sniper.killed:
            out.append("kill switch engaged")
        return out

    def exchange(self):
        with self._ex_lock:                              # one SDK client, built once (the warm-up or the first caller)
            if self._exchange is None:
                self._exchange = SDKExchange(self.network, self.env["HL_ACCOUNT_ADDRESS"], self.env["HL_AGENT_PRIVATE_KEY"])
            return self._exchange

    def _warm(self) -> None:
        try:
            self.exchange()
        except Exception as exc:
            self.last_refusal = {"ts": self.clock(), "mid": None, "why": [f"SDK warm-up failed: {type(exc).__name__}: {exc}"[:200]]}
        finally:
            self._warming = False

    # ------------------------------------------------------------------ orders
    def on_signal(self, mid: int, side: int, limit: float, fair: float, market: dict) -> None:
        """Called by the sniper for every signal. Sends in a background thread so the sniper never waits on it."""
        if not self.armed_cfg:                       # the master switch is off: nothing to do, not even a thread
            return
        now = self.clock()
        if now < self.cool.get(mid, 0):              # one attempt per market per cooldown (first live run sent 3 in 1 s)
            return
        self.cool[mid] = now + self.cooldown_s
        threading.Thread(target=self._send, args=(mid, side, limit, fair, market), daemon=True).start()

    def _send(self, mid, side, limit, fair, market) -> dict | None:
        now = self.clock()
        self._roll_day(now)
        block = self.blockers()
        if block:
            self.last_refusal = {"ts": now, "mid": mid, "why": block}
            return None
        with self.lock:
            held_yes = self.pos.get(mid, {}).get("yes", 0.0)
            if side > 0:                                                         # buy Yes at the ask we saw (+ slip)
                coin, is_buy, px = f"#{mid * 10}", True, self._worst_buy(limit, fair)
            elif held_yes > 0:                                                   # sell Yes we hold
                coin, is_buy, px = f"#{mid * 10}", False, round(1 - self._worst_buy(1 - limit, 1 - fair), 4)
            else:                                                                # no Yes: buy No at 1 - bid (+ slip)
                coin, is_buy, px = f"#{mid * 10 + 1}", True, self._worst_buy(1 - limit, 1 - fair)
            pay = px if is_buy else 1 - px
            # the $10 minimum's contracts, scaled to this order size (= size_mult when not compounding)
            sz = math.ceil(max(self.contracts, math.ceil(self.min_notional / max(pay, 1e-6) - 1e-9))
                           * self.order_usd / self.min_notional - 1e-9)
            if not is_buy:
                sz = min(sz, int(held_yes))
            cost = sz * pay
            open_cost = sum(p["cost"] for p in self.pos.values())
            why = None
            if self.dry_run:
                pass                                  # dry run: its own limits on its own ledger, below - never sends
            elif self.compound and self.equity_basis is None:
                # compounding sizes every limit to the account: until the account has been read, the fixed config
                # limits ($100 open / $150 cap) would apply - on 2026-10-09 a ~$10 account placed a $9.99 long-dated
                # bet in the first seconds after a restart that way. No order until the account is known.
                why = "sizing the limits to the account first (no account read yet)"
            elif sz <= 0:
                why = "nothing to sell"
            elif cost > self.max_order + 1e-9:
                why = f"order ${cost:.2f} > max_order_usd ${self.max_order:g}"
            elif open_cost + cost > min(self.max_open, self.capital) + 1e-9:
                why = f"open cost would be ${open_cost + cost:.2f} > ${min(self.max_open, self.capital):g}"
            elif is_buy and self.pos.get(mid, {}).get("cost", 0.0) + cost > self.max_market + 1e-9:
                why = (f"this market would hold ${self.pos[mid]['cost'] + cost:.2f} > max_market_usd "
                       f"${self.max_market:g}")
            elif is_buy and (market.get("expiry") or 0) - now > self.long_tte_s and \
                    self.long_cost(now) + cost > self.long_budget + 1e-9:
                why = (f"long-dated markets would hold ${self.long_cost(now) + cost:.2f} > long_budget_usd "
                       f"${self.long_budget:g} (budget kept for markets with <= {self.long_tte_s / 60:g} min left)")
        if self.dry_run:
            return self._dry_send(mid, side, fair, market, coin, is_buy, px, sz, pay, cost, now)
        if self.dry_shadow:
            # the zero-risk test keeps scoring EVERY opportunity on its own $100 ledger while real orders run as
            # normal - on a tiny live account most signals are refused for size, so this is where the evidence comes from
            self._dry_send(mid, side, fair, market, coin, is_buy, px, sz, pay, cost, now)
        if why:
            self.last_refusal = {"ts": now, "mid": mid, "why": [why]}
            return None
        rec = {"ts": now, "mid": mid, "coin": coin, "is_buy": is_buy, "px": px, "sz": sz, "fair": round(fair, 4),
               "gate_overridden": not self.require_gate,
               "strike": market.get("strike"), "expiry": market.get("expiry"), "underlying": market.get("coin")}
        # latency, measured: exchange time of the book that triggered it -> our detection -> send -> response (ms)
        if market.get("detect") is not None:
            rec["path"] = market.get("path")
            rec["detect_to_send_ms"] = round((now - market["detect"]) * 1000)
            if market.get("book_et"):
                rec["book_to_send_ms"] = round(now * 1000 - float(market["book_et"]))
        t_send = time.time()
        try:
            resp = self.exchange().ioc(coin, is_buy, sz, px)
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            self._log(rec)
            return rec
        rec["order_rtt_ms"] = round((time.time() - t_send) * 1000)
        filled, avg = self._parse(resp)
        rec.update(status="filled" if filled else "not filled", filled=filled, avg_px=avg, response=resp)
        if filled:
            with self.lock:
                # Yes and No are held separately (holding both is $1 a pair, whatever happens); cost = cash spent
                p = self.pos.setdefault(mid, {"yes": 0.0, "no": 0.0, "cost": 0.0, "expiry": market.get("expiry"),
                                              "underlying": market.get("coin"), "strike": market.get("strike")})
                leg = "yes" if coin.endswith("0") else "no"
                p[leg] += filled if is_buy else -filled
                p["cost"] += filled * avg if is_buy else -filled * avg
                p["at"] = self.clock()                  # last fill: a snapshot read before this can't show it yet
            self._wallet["ts"] = 0.0                    # cash left the wallet: re-read it with the new position
        self._log(rec)
        self._save()
        return rec

    def long_cost(self, now: float) -> float:
        """Cost held in markets that still have more than long_tte_min to run (they roll into the main budget later)."""
        return sum(p["cost"] for p in self.pos.values() if (p.get("expiry") or 0) - now > self.long_tte_s)

    # ------------------------------------------------------------------ dry run (zero-risk Tokyo test)
    DRY_MARGINS_MS = (0, 100, 300)       # how long past our estimated arrival the quote must have survived to count
    DRY_NET_MS = 3.0                     # Tokyo -> Hyperliquid one way (measured RTT 2.5 ms, 2026-10-09)

    def _dry_send(self, mid, side, fair, market, coin, is_buy, px, sz, pay, cost, now) -> dict | None:
        """Everything a live order does except sending it. Limits come from the config's fixed live limits (a $100
        open / $30 per market pilot at 1x, x the multiplier) on the dry ledger. The sniper then watches the book: the
        IOC 'would have filled' if the quote it needed was still there when it would have arrived (detection + our
        send overhead + the network), with 0 / 100 / 300 ms of extra margin."""
        b, m = self.base, self.size_mult
        MG = str(self.DRY_MARGINS_MS[1])                   # the caps follow the middle (100 ms) ledger
        with self.lock:
            held = self.dry["pos"]
            cost_of = lambda legs: legs.get(MG, {}).get("cost", 0.0)   # noqa: E731
            mcost = cost_of(held.get(mid, {}))
            open_cost = sum(cost_of(v) for v in held.values())
            is_long = (market.get("expiry") or 0) - now > self.long_tte_s
            long_cost = sum(cost_of(v) for k, v in held.items()
                            if (self.dry["meta"].get(k, {}).get("expiry") or 0) - now > self.long_tte_s)
            why = None
            if open_cost + cost > b["max_open_usd"] * m + 1e-9:
                why = "dry: open budget"
            elif mcost + cost > b["max_market_usd"] * m + 1e-9:
                why = "dry: per-market cap"
            elif is_long and long_cost + cost > b["long_budget_usd"] * m + 1e-9:
                why = "dry: long-dated budget"
            self.dry["stats"]["signals"] += 1
            if why:
                self.dry["stats"]["refused"] += 1
                return None
        yes_limit = round(1 - px, 4) if coin.endswith("1") else px     # the Yes price our IOC needed
        detect = market.get("detect") if market.get("detect") is not None else now
        book_et = market.get("book_et")
        order = {"ts": now, "mid": mid, "side": side, "coin": coin, "px": px, "sz": sz, "pay": round(pay, 4),
                 "cost": round(cost, 4), "fair": round(fair, 4), "yes_limit": yes_limit, "path": market.get("path"),
                 "underlying": market.get("coin"), "expiry": market.get("expiry"), "detect": detect,
                 "book_et": book_et, "overhead_ms": round((self.clock() - detect) * 1000, 1), "status": "watching"}
        with self.lock:
            self.dry["meta"].setdefault(mid, {"expiry": market.get("expiry"), "coin": market.get("coin"),
                                              "strike": market.get("strike")})
            self.dry["stats"]["sent"] += 1
            self.dry["orders"] = (self.dry["orders"] + [order])[-300:]
        watch = getattr(self.sniper, "watch_quote", None)
        if watch is None:
            self._dry_resolve(order, None, None)
        else:
            watch(mid, side, yes_limit, lambda gone_et, gone_local, o=order: self._dry_resolve(o, gone_et, gone_local))
        return order

    def _dry_resolve(self, o: dict, gone_et, gone_local) -> None:
        """The quote went away (exchange time gone_et, ms) or survived the watch window (None). Our order would have
        arrived at detection + overhead + network; it fills with margin M if the quote outlived that by M ms."""
        arrive_ms = o["detect"] * 1000 + o["overhead_ms"] + self.DRY_NET_MS
        o["life_ms"] = None if gone_et is None else round(gone_et - arrive_ms, 1)
        if o.get("book_et"):
            o["reaction_ms"] = round(o["detect"] * 1000 - float(o["book_et"]), 1)
        o["fills"] = {str(mg): bool(gone_et is None or gone_et > arrive_ms + mg) for mg in self.DRY_MARGINS_MS}
        o["status"] = "resolved"
        with self.lock:
            self.dry["stats"]["resolved"] = self.dry["stats"].get("resolved", 0) + 1
            for mg, ok in o["fills"].items():
                if ok:
                    self.dry["stats"]["filled_" + mg] += 1
                    leg = self.dry["pos"].setdefault(o["mid"], {}).setdefault(mg, {"contracts": 0.0, "cost": 0.0,
                                                                                     "side": o["side"]})
                    leg["contracts"] += o["sz"]
                    leg["cost"] += o["cost"]
        self._log_dry(o)

    def _dry_poll(self, now: float, rest) -> None:
        """Settle dry positions like live ones (settledOutcome from expiry + 120 s, backed off, never booked early)."""
        if self.sniper is not None and hasattr(self.sniper, "expire_watches"):
            self.sniper.expire_watches(now)
        for mid in list(self.dry["pos"]):
            exp = self.dry["meta"].get(mid, {}).get("expiry") or 0
            if not exp or now < exp + 120 or now < self._dry_retry.get(mid, (0, 0))[0]:
                continue
            tries = self._dry_retry.get(mid, (0, 0))[1]
            self._dry_retry[mid] = (now + min(3600.0, 30.0 * 2 ** tries), tries + 1)
            try:
                res = rest.post({"type": "settledOutcome", "outcome": int(mid)}) if rest is not None else None
                y = float(res.get("settleFraction")) if isinstance(res, dict) else math.nan
            except Exception:
                y = math.nan
            if not math.isfinite(y):
                continue
            with self.lock:
                legs = self.dry["pos"].pop(mid, {})
                row = {"ts": now, "mid": mid, "y": y, **{k: v for k, v in self.dry["meta"].get(mid, {}).items()}}
                for mg, leg in legs.items():
                    # a No buy pays 1 - y per contract, a Yes buy pays y; fee as the paper book (0.0005 per contract)
                    payout = leg["contracts"] * ((1 - y) if leg["side"] < 0 else y)
                    pnl = payout - leg["cost"] - 0.0005 * leg["contracts"]
                    row["pnl_" + mg] = round(pnl, 4)
                    self.dry["realized"][mg] = self.dry["realized"].get(mg, 0.0) + pnl
                    self.dry["won"][mg] = self.dry["won"].get(mg, 0) + (pnl > 0)
                    self.dry["markets"][mg] = self.dry["markets"].get(mg, 0) + 1
                self.dry["settled"] = (self.dry["settled"] + [row])[-200:]
            self._log_dry({"status": "settled", **row})
        self._save_dry()

    def _log_dry(self, rec: dict) -> None:
        if self.path is None:
            return
        try:
            with open(self.path.parent / "dry_orders.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({k: v for k, v in rec.items() if k != "cb"}, default=str, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _save_dry(self) -> None:
        if self.path is None:
            return
        p = self.path.parent / "dry_run.json"
        tmp = p.with_suffix(".tmp")
        try:
            with self.lock:
                data = json.dumps({k: self.dry[k] for k in ("pos", "meta", "stats", "realized", "won", "markets",
                                                            "settled", "since")}, default=str)
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass

    def _load_dry(self) -> None:
        if self.path is None:
            return
        try:
            d = json.loads((self.path.parent / "dry_run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        d["pos"] = {int(k): v for k, v in d.get("pos", {}).items()}
        d["meta"] = {int(k): v for k, v in d.get("meta", {}).items()}
        for k in ("pos", "meta", "stats", "realized", "won", "markets", "settled", "since"):
            if k in d:
                self.dry[k] = d[k]

    def dry_summary(self) -> dict:
        s = self.dry["stats"]
        res = [o for o in self.dry["orders"] if o.get("status") == "resolved"]
        react = sorted(o["reaction_ms"] for o in res if o.get("reaction_ms") is not None)
        over = sorted(o["overhead_ms"] for o in res if o.get("overhead_ms") is not None)
        med = lambda xs: xs[len(xs) // 2] if xs else None   # noqa: E731
        out = {"enabled": self.dry_run or self.dry_shadow, "alongside_live": self.dry_shadow and not self.dry_run,
               "since": self.dry.get("since"), "signals": s["signals"], "sent": s["sent"],
               "refused": s["refused"], "median_reaction_ms": med(react), "median_overhead_ms": med(over),
               "open_markets": len(self.dry["pos"]), "margins": {}}
        for mg in map(str, self.DRY_MARGINS_MS):
            n = self.dry["markets"].get(mg, 0)
            out["margins"][mg] = {"filled": s.get("filled_" + mg, 0),
                                  "fill_rate": round(s.get("filled_" + mg, 0) / s["resolved"], 3)
                                  if s.get("resolved") else None,
                                  "settled_markets": n, "won": self.dry["won"].get(mg, 0),
                                  "pnl": round(self.dry["realized"].get(mg, 0.0), 2)}
        out["recent"] = [{k: o.get(k) for k in ("ts", "mid", "underlying", "px", "sz", "path", "reaction_ms",
                                                 "overhead_ms", "life_ms", "fills")} for o in self.dry["orders"][-12:][::-1]]
        return out

    def _worst_buy(self, seen: float, fair: float) -> float:
        """The IOC limit for a buy at `seen`: up to `slip` worse, never above fair - min_edge (never below `seen`)."""
        return round(max(seen, min(seen + self.slip, fair - self.min_edge)), 4)

    @staticmethod
    def _parse(resp) -> tuple[float, float | None]:
        """Hyperliquid order response -> (filled contracts, average price)."""
        try:
            st = resp["response"]["data"]["statuses"][0]
        except (KeyError, IndexError, TypeError):
            return 0.0, None
        f = st.get("filled") if isinstance(st, dict) else None
        if not f:
            return 0.0, None
        return float(f.get("totalSz", 0)), float(f.get("avgPx", 0))

    # ------------------------------------------------------------------ settlement and the daily loss limit
    def settle(self, mid: int, y: float | None) -> None:
        """Called when the paper sniper sees a market settle: the pilot's position there pays qty x y on the Yes leg."""
        with self.lock:
            p = self.pos.pop(mid, None)
        if p is None:
            return
        payout = (p["yes"] * y + p["no"] * (1 - y)) if y is not None else p["cost"]     # void: refunded
        pnl = payout - p["cost"]
        self.realized_today += pnl
        self.realized_total += pnl                     # kept across restarts (live_pilot.json)
        self._log({"ts": self.clock(), "mid": mid, "status": "settled", "y": y, "pnl": round(pnl, 4)})
        self._wallet["ts"] = 0.0                       # the payout is in the wallet now: read it on the next look
        if self.daily_loss > 0 and self.realized_today <= -self.daily_loss:     # daily_loss 0 = no daily stop
            self.halt(f"daily loss limit: ${self.realized_today:.2f} today (limit -${self.daily_loss:g})")
        self._save()

    def _refresh_equity(self) -> None:
        try:
            w = self.account()
            if w["usdc"] is not None:
                # cash + positions at cost, both from ONE snapshot (see account / _held)
                self.equity_basis = w["usdc"] + self._held(w)[1]
                self._apply_limits()
        finally:
            self._eq_busy = False

    def _apply_limits(self) -> None:
        """Fixed: the config limits x the size multiplier, open capped at capital_usd. Compounding (with a known
        account value E): order max($10, order x E), open open x E (never above max_open_frac x E), per market, long-
        dated and daily stop as fractions of E - all x the multiplier, which stays the operator's aggression dial."""
        m, b, E = self.size_mult, self.base, self.equity_basis
        with self.lock:
            if self.compound and E:
                c = self.cf
                self.capital = c["max_open_frac"] * E
                self.order_usd = max(self.min_notional, c["order"] * E) * m
                self.max_order = max(b["max_order_usd"], 2.5 * self.order_usd)
                if self.floor_fixed:
                    # research 2026-10-08 (out of sample): compounding may grow the budgets but never shrinks them
                    # below the fixed config - on a ~$120 account 16% fits only ONE order per market
                    self.max_open = min(max(c["open"] * E * m, b["max_open_usd"] * m), self.capital)
                    self.max_market = max(self.order_usd, c["market"] * E * m, b["max_market_usd"] * m)
                else:
                    self.max_open = min(c["open"] * E * m, self.capital)
                    self.max_market = max(self.order_usd, c["market"] * E * m)
                self.long_budget = c["long"] * E * m
                self.daily_loss = c["daily_loss"] * E * m if c["daily_loss"] > 0 else 0.0     # 0 = no daily stop
            else:
                self.capital = b["capital_usd"]
                self.order_usd = self.min_notional * m
                self.max_order = b["max_order_usd"] * m
                self.max_open = min(b["max_open_usd"] * m, self.capital)
                self.daily_loss = b["daily_loss_usd"] * m
                self.max_market = b["max_market_usd"] * m
                self.long_budget = b["long_budget_usd"] * m

    def set_size(self, mult: float, log: bool = True) -> dict:
        mult = float(mult)
        if not (math.isfinite(mult) and 1 <= mult <= self.max_mult):
            raise ValueError(f"live size must be between 1 and {self.max_mult:g}")
        self.size_mult = mult
        self._apply_limits()
        if log:
            self._log({"ts": self.clock(), "status": "size", "reason": f"live size set to {mult:g}x by the operator"})
            self._save()
        return self.size_info()

    def size_info(self) -> dict:
        return {"multiplier": self.size_mult, "max_multiplier": self.max_mult, "max_order_usd": round(self.max_order, 2),
                "max_open_usd": round(self.max_open, 2), "daily_loss_usd": round(self.daily_loss, 2),
                "max_market_usd": round(self.max_market, 2), "long_budget_usd": round(self.long_budget, 2),
                "long_tte_min": self.long_tte_s / 60, "capital_usd": round(self.capital, 2),
                "order_usd": round(self.order_usd, 2),
                "compound": {"enabled": self.compound, "account": None if self.equity_basis is None
                             else round(self.equity_basis, 2), **self.cf}}

    def halt(self, reason: str) -> None:
        self.halted, self.halt_reason = True, reason
        self._log({"ts": self.clock(), "status": "halted", "reason": reason})
        self._save()

    def resume(self) -> None:
        self.halted, self.halt_reason = False, None
        self._save()

    def _roll_day(self, now: float) -> None:
        d = int(now // 86400)
        if d != self.day:
            self.day, self.realized_today = d, 0.0
            if self.halted and (self.halt_reason or "").startswith("daily loss"):
                self.resume()

    # ------------------------------------------------------------------ persistence / reporting
    def _log(self, rec: dict) -> None:
        self.orders = (self.orders + [{k: v for k, v in rec.items() if k != "response"}])[-200:]
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _save(self) -> None:
        if self.state_path is None:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pos": {str(k): v for k, v in self.pos.items()}, "realized_today": self.realized_today,
                                   "day": self.day, "halted": self.halted, "halt_reason": self.halt_reason,
                                   "size_mult": self.size_mult, "realized_total": self.realized_total}), encoding="utf-8")
        tmp.replace(self.state_path)

    def _total_from_log(self) -> float:
        if self.path is None or not self.path.exists():
            return 0.0
        total = 0.0
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("status") == "settled" and isinstance(r.get("pnl"), (int, float)):
                        total += r["pnl"]
        except OSError:
            return 0.0
        return total

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.pos = {int(k): v for k, v in d.get("pos", {}).items()}
        self.realized_today, self.day = float(d.get("realized_today", 0)), int(d.get("day", self.day))
        # before this was persisted, rebuild it from the order log's settlements
        self.realized_total = float(d["realized_total"]) if "realized_total" in d else self._total_from_log()
        self.halted, self.halt_reason = bool(d.get("halted")), d.get("halt_reason")
        try:
            self.set_size(float(d.get("size_mult", 1.0)), log=False)
        except (TypeError, ValueError):
            pass

    def account(self) -> dict:
        """The account snapshot (read-only): USDC and, with a snapshot-capable exchange, the outcome tokens held, from
        ONE read. Refreshed at most every 30 s, at once after a fill or settlement, and while a held market has just
        expired. Replaced whole, and only by a read that started later than the stored one, so two threads can never
        pair the USDC of one read with the tokens of another."""
        now = self.clock()
        w = self._wallet
        if now - w["ts"] >= 30 and self.env.get("HL_ACCOUNT_ADDRESS") and self.env.get("HL_AGENT_PRIVATE_KEY"):
            w["ts"] = now                                 # one reader at a time per 30 s
            try:
                ex = self.exchange()
                if hasattr(ex, "snapshot"):
                    snap = ex.snapshot()
                    new = {"usdc": float(snap["usdc"]), "outcomes": dict(snap["outcomes"]), "error": None}
                else:
                    new = {"usdc": float(ex.balances().get("USDC", 0.0)), "outcomes": None, "error": None}
                with self.lock:
                    if now >= self._wallet["started"]:
                        self._wallet = {**new, "ts": now, "started": now}
            except Exception as exc:
                with self.lock:
                    self._wallet = {**self._wallet, "error": f"{type(exc).__name__}: {exc}"[:160]}
        return self._wallet

    def wallet_usdc(self) -> float | None:
        """The real USDC spot balance (see account())."""
        return self.account()["usdc"]

    def _mark(self, mid: int) -> float | None:
        """The Yes mid of a two-sided book no wider than 50c; otherwise the last good one seen (books thin out and
        vanish near expiry, and falling back to cost there made wins flicker as drops)."""
        b = (getattr(self.sniper, "books", {}) or {}).get(mid) or {}
        if b.get("bid") is not None and b.get("ask") is not None and b["ask"] - b["bid"] <= 0.5:
            self._marks[mid] = (b["bid"] + b["ask"]) / 2
        return self._marks.get(mid)

    def _held(self, w: dict) -> tuple[float, float]:
        """(value at market, cost) of everything held, from the snapshot `w` when it has the tokens. Tokens are
        "+<encoding>" balances with their entry cost; a pilot position missing from a snapshot read after its last
        fill, or a token without an entry cost, falls back to the pilot's own record (and says so in wallet_error)."""
        now = self.clock()
        outs = w.get("outcomes")
        value = cost = 0.0
        if outs is None:
            for mid, p in self.pos.items():
                m = self._mark(mid)
                value += p["cost"] if m is None else p.get("yes", 0.0) * m + p.get("no", 0.0) * (1 - m)
                cost += p["cost"]
            return value, cost
        seen, warn = set(), None
        for enc, (total, entry) in outs.items():
            mid, leg = divmod(enc, 10)
            seen.add(mid)
            if entry <= 0:
                entry = (self.pos.get(mid) or {}).get("cost", 0.0)
                warn = f"no entry cost for +{enc} on the exchange: using the pilot's record"
            m = self._mark(mid)
            value += entry if m is None else total * (m if leg == 0 else 1 - m)
            cost += entry
            exp = (self.pos.get(mid) or {}).get("expiry") or 0
            if exp and now > exp and w.get("started", 0) < exp + 120:
                self._wallet["ts"] = 0.0                 # just expired: read again soon, it may have settled
        for mid, p in self.pos.items():
            if mid not in seen and now < (p.get("expiry") or 0) and w.get("started", 0) > p.get("at", 0) + 1:
                m = self._mark(mid)
                value += p["cost"] if m is None else p.get("yes", 0.0) * m + p.get("no", 0.0) * (1 - m)
                cost += p["cost"]
                warn = f"pilot position {mid} not in the exchange snapshot: using the pilot's record"
        if warn and not w.get("error"):
            w["error"] = warn
        return value, cost

    def open_value(self) -> float:
        """Open positions at the current market: Yes at the Yes mid, No at 1 - Yes mid. From the exchange's own token
        balances when we have them: they come from the same read as the USDC, so a settlement can't be counted twice
        (2026-10-08: the page counted a won market's payout AND the position for up to ~2.5 min, then "dropped" when
        the pilot recorded the settlement - and the same double count inflated the compounding account, which let one
        BTC market take $90 against a ~$69 cap)."""
        return self._held(self._wallet)[0]

    def to_dict(self) -> dict:
        block = self.blockers()
        w = self.account()
        usdc = w["usdc"]
        ov = self._held(w)[0]
        return {"mode": ("DRY RUN" if self.dry_run else "LIVE") if not block else "OFF", "blockers": block,
                "dry": self.dry_summary() if (self.dry_run or self.dry_shadow) else None,
                "gate_overridden": not self.require_gate,
                "wallet_usdc": round(usdc, 2) if usdc is not None else None, "wallet_error": w.get("error"),
                "open_value": round(ov, 2), "size": self.size_info(),
                "live_equity": round(usdc + ov, 2) if usdc is not None else None,   # cash + open positions at market
                "live_pnl_total": round(self.realized_total, 2),
                "limits": {"capital_usd": self.capital, "max_order_usd": self.max_order, "max_open_usd": self.max_open,
                           "daily_loss_usd": self.daily_loss, "max_market_usd": self.max_market,
                           "long_budget_usd": self.long_budget, "long_tte_min": self.long_tte_s / 60,
                           "contracts": self.contracts or "the $10 minimum"},
                "long_cost": round(self.long_cost(self.clock()), 2),
                "speed": {"fire_on": self.fire_on, "slip": self.slip, "min_edge": self.min_edge},
                "open": [{"mid": k, **v} for k, v in self.pos.items()],
                "open_cost": round(sum(p["cost"] for p in self.pos.values()), 2),
                "realized_today": round(self.realized_today, 2), "halted": self.halted, "halt_reason": self.halt_reason,
                "orders": self.orders[-20:][::-1], "last_refusal": self.last_refusal,
                "wallet_set": bool(self.env.get("HL_ACCOUNT_ADDRESS") and self.env.get("HL_AGENT_PRIVATE_KEY"))}
