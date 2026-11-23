"""Config loading and validation. Every threshold lives in config.yaml, never in a prompt."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "mode": "paper",
    "venue": "hyperliquid",
    "network": "mainnet",
    "market": {
        "coins": ["BTC", "ETH", "SOL"],
        "decision_interval_ms": 1000,
        "monitor_interval_ms": 200,
        "horizon_s": 60,
        "book_refresh_s": 1.0,
    },
    "paper": {
        "starting_cash": 10000.0,
        "taker_fee_bps": 4.5,
        "maker_fee_bps": 1.5,
        "extra_slippage_bps": 0.5,
        "entry_mode": "taker",     # Jev book entries: "taker" (market order) or "maker" (post-only limit at the touch)
        "entry_ttl_s": 60,         # a resting maker entry is cancelled after this long
        "exit_mode": "taker",      # time exits: "taker" (market order at max hold) or "maker" (rest at the touch first)
        "exit_window_s": 120,      # maker exits start resting this long before max hold; taker at max hold if unfilled
    },
    "jev": {
        "model": "jev-1.13.0",
        "deadline_ms": 800,
        "fallback": "rules",
        "daily_budget_usd": 10.0,
        "price_per_million_input": 0.042,
        "circuit_errors": 5,
        "circuit_cooldown_s": 30,
    },
    "policy": {
        "min_direction_prob": 0.55,
        "full_size_prob": 0.68,
        "min_setup_prob": 0.45,
        "min_edge_bps": 10.0,
        "max_liquidity_stress": 0.70,
        "toxic_flow_threshold": 0.60,
        "max_crisis_prob": 0.50,
        "exit_direction_prob": 0.60,
        "exit_inventory_pressure": 2.4,
        "min_hold_s": 0,
        "confirm_ticks": 1,
    },
    "rules": {
        "momentum_bps": 3.0,
        "min_imbalance": 0.15,
        "min_flow": 0.60,
        "min_hold_s": 0,
    },
    "risk": {
        "risk_per_trade": 0.002,
        "stop_sigma": 1.5,
        "target_sigma": 2.5,
        "min_stop_bps": 6.0,
        "min_target_bps": 10.0,
        "max_hold_s": 180,
        "max_position_leverage": 1.5,
        "max_total_leverage": 3.0,
        "max_open_positions": 3,
        "max_entries_per_min": 12,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.08,
        "max_data_age_ms": 3000,
        "max_decision_latency_ms": 800,
        "max_consecutive_errors": 10,
        "max_spread_bps": 5.0,
        "max_slippage_bps": 3.0,
        "min_edge_multiple": 3.0,
        "cooldown_s": 30,
        "reentry_cooldown_s": 0,
        "min_order_usd": 10.0,
        "allow_short": True,
        "flatten_on_kill": True,
        # Bet sizing for the Jev book (the baseline book always bets 1x so it stays a fair benchmark).
        "bet_multiplier": 1.0,               # your manual multiplier; changeable live from the terminal
        "max_bet_multiplier": 5.0,           # hard ceiling on manual x streak boost
        "leverage_ceiling_position": 5.0,    # the leverage caps scale with the multiplier, but never above these
        "leverage_ceiling_total": 10.0,
        "streak": {"enabled": False, "after_wins": 2, "step": 0.5, "max": 3.0, "gate_trades": 30},
    },
    "journal": {"retain_hours": 72},
    "server": {"host": "127.0.0.1", "port": 8787, "open_browser": True, "refresh_ms": 1000},
    "analyst": {"model": "claude-opus-5-5", "max_tokens": 2000},
    "hourly_test": {"enabled": False, "coins": ["BTC", "ETH", "SOL", "HYPE"], "jev_variants": {}},
    "hip4": {"enabled": False, "coins": ["BTC", "ETH", "SOL", "HYPE"], "claude": True, "edge": 0.03,
             "variants": ["jev_v3"]},
    "perp_trading": True,          # False = the engine only supplies market data (no perp orders, no Jev calls)
    "hip4_tape": {"enabled": False, "coins": ["BTC", "ETH", "SOL", "HYPE"], "keep_days": 30},
    "hip4_backtest": {"auto_every_h": 0},
    # the zero-risk strategy tournament (helix/shadow_lab.py). autopilot: LIVE trades only a variant whose paper-betting
    # score reached season_size / alpha, over >= min_clusters outcomes and >= min_age_d days, with >= min_side outcomes
    # in both rising and falling markets; it sits out when none qualifies. These are pinned when a season starts
    # (data/shadow_lab.json): a later change here can only make them stricter
    # kalshi / cex_feed: record an outside reference price and add its test strategy (helix/ref_feeds.py);
    # xyz_open_guard: no lab (or autopilot-live) bets on xyz: stock perps across a weekday US market open
    "hip4_lab": {"enabled": False, "autopilot": False, "alpha": 0.05, "season_size": 15, "min_clusters": 300,
                 "min_side": 100, "min_age_d": 7, "kalshi": False, "cex_feed": "off", "xyz_open_guard": True},
    # the PC's research desk (helix/research_desk.py -> research_agent/daily.py): read-only snapshots of the Tokyo
    # server and reports every every_min minutes. PC only - the server never has research_agent/
    "research_desk": {"enabled": False, "every_min": 60},
    "hip4_sniper": {               # paper taker on HIP-4 (helix/hip4_mm.py, strategy "sniper"): the rules are fixed
        "enabled": False,          # BEFORE the held-out data (pre-registered 2026-10-06), so later tests are honest
        "strategy": "sniper",
        "coins": ["all"],
        "snipe_edge": 0.05,        # take the book only when it is >= 5c past our model fair
        "max_tte_min": 60,         # ... in markets with <= 60 minutes left
        "stop_min": 3,             # ... and not in the last 3 minutes (settlement averaging)
        "snipe_latency_ms": 300,   # orders land this much later, against the book at that time
        "snipe_cooldown_s": 5,
        "quote_size": 20,          # contracts per snipe (the size selector scales this, the caps and the risk cap)
        "max_position": 100,       # contracts per market, either side
        "max_risk_usd": 1000,      # total worst-case loss across open markets
        "min_p": 0.03,
        "min_notional": 10,        # the exchange's minimum order
        "exits": {"enabled": False, "take_profit": 0.15, "stop_loss": 0.30, "min_hold_s": 15, "cooldown_s": 60,
                  "taker_fee": 0.0005},
        "holdout_from": 0,         # unix time the rules were fixed: only tape after it is out-of-sample
        "price_band": [0.0, 1.0],  # only take contracts whose Yes price is inside this band
        "sides": "both",           # both | buy (Yes) | sell (Yes) | auto (follow each underlying's trend)
        # for sides: auto - NOT recommended: on the 2026-10-07 research no trend setting beat sell-only (the trailing
        # trend did not predict the drift to expiry); neutral "sell" is the least bad fallback
        "trend": {"lookback_min": 360, "threshold": 0.003, "neutral": "sell"},
        "edge": 0.02, "skew": 0.0, "book_stale_s": 30, "fee": 0.0005, "start_cash": 10000, "kill_drawdown": 0.08,
        "quote_every_s": 0.5, "market_weight": 0.0, "blend_max_spread": 0.10, "max_size_multiplier": 5,
        "fast_model": True, "zs_refresh_s": 900,
    },
    "hip4_live": {                 # live readiness + the live pilot (helix/hip4_exec.py sends REAL orders when live_enabled)
        "shadow": False,           # mirror the paper maker as the exact live orders, log them, measure churn/validity
        "repost_threshold": 0.01,  # only re-price a resting order when the price moves this much (address budget)
        "min_notional": 10.0,      # the exchange's minimum order value (USDC)
        "tick": 0.001,
        "address_budget": 10000,   # Hyperliquid's initial per-address request buffer (then 1 per $1 traded)
        "gate": {"days": 7, "settled": 200, "markout_n": 200, "backtest_hours": 24, "shadow_validity": 0.9},
        "live_enabled": False,     # THE master switch for real orders (helix/hip4_exec.py); only the operator sets it
        "remote_pilot_url": "",    # live runs on another Helix (e.g. the Tokyo server via the SSH tunnel): show it here
        "remote_name": "Tokyo server",
        "require_gate": True,      # False = the operator goes live before the go-live gate passes (shown on the page)
        "pilot": {"capital_usd": 150, "max_order_usd": 25, "max_open_usd": 100, "daily_loss_usd": 60, "contracts": 0,
                  "cooldown_s": 15,
                  "max_market_usd": 30,  # most cost in ONE market (scales with the live size multiplier)
                  "long_tte_min": 120,   # markets with more than this left to expiry ...
                  "long_budget_usd": 40,  # ... may hold at most this much in total (the rest is for short-dated ones)
                  # compounding: when enabled, the limits above are replaced by fractions of the account (wallet + open
                  # positions at cost), so the pilot grows with its winnings and shrinks after losses
                  # (the default fractions are research stage 1 - $100 open, $30/market, $40 long, $60 stop - on a
                  # ~$190 account; orders stay at the $10 minimum until the account passes $500)
                  "compound": {"enabled": False, "open": 0.53, "order": 0.02, "market": 0.16, "long": 0.21,
                               "daily_loss": 0.32, "max_open_frac": 0.9,
                               "floor_to_fixed": False},   # never shrink below max_open_usd / max_market_usd
                  "dry_run": False,     # True = do everything a live order does EXCEPT send it, and score whether it
                                        #   would have filled and what it would have made (zero-risk test)
                  "dry_shadow": False,  # True = that zero-risk scoring runs ALONGSIDE real orders (live stays live)
                  "fire_on": "fill",    # fill = after the paper sniper's simulated fill; signal = the moment it fires
                  "slip": 0.0,          # the IOC may pay up to this much worse than the quote seen ...
                  "min_edge": 0.03},    # ... but never closer than this to our fair
    },   # >0: run the automatic strategy comparison this often (0 = off)
    "hip4_mm": {                   # paper market maker on HIP-4 outcome markets (helix/hip4_mm.py)
        "enabled": False,
        "coins": ["BTC", "ETH", "SOL", "HYPE"],
        "edge": 0.02,              # quote this far from fair on each side ($ per $1 contract)
        "skew": 0.02,              # at max_position, both quotes lean this far against the inventory
        "quote_size": 20,          # contracts per quote
        "max_position": 100,       # contracts per market, either side
        "max_risk_usd": 1000,      # total worst-case loss across open markets; above it, only risk-reducing quotes
        "min_p": 0.05,             # no quotes when fair is below this or above 1 - this
        "stop_min": 3,             # no quotes in the last minutes before expiry
        "max_tte_min": 65,         # only markets expiring within this (the fair model was tested on 15 min and 1 h)
        "book_stale_s": 30,
        "fee": 0.0005,             # per contract per fill
        "start_cash": 10000,
        "kill_drawdown": 0.08,
        "quote_every_s": 1.0,
        "market_weight": 0.5,      # fair = this x market mid + the rest x model, when the market is two-sided and tight
        "blend_max_spread": 0.10,  # ... i.e. its spread is at most this
        "latency_s": 0.25,         # a quote can only be filled by trades that arrive this long after it was placed
        "max_size_multiplier": 5,  # the bet-size selector's ceiling (scales quote size, position limit and risk cap)
        "quoting": True,           # False = pause the maker: no quotes, open positions still settle
        "min_notional": 0,         # live rule: minimum order value; quotes below it are raised to it or dropped
        "requote_threshold": 0,    # sticky quotes: keep a resting price until fair moves this far (0 = re-price always)
        "max_markets": 0,          # quote only the N busiest markets (0 = every eligible market)
        "guard_move": 0.05,        # stop adding to a position when fair moved this far against it ...
        "guard_s": 60,             # ... within this many seconds
        "levels": 1,               # 2 = a second, deeper quote on each side (level2_edge_mult x edge from fair)
        "level2_edge_mult": 2.0,
        "fast_model": False,       # grid lookup + 15-day table refreshed every zs_refresh_s (for many markets)
        "zs_refresh_s": 0,
        "exits": {"enabled": False, "take_profit": 0.15, "stop_loss": 0.30, "min_hold_s": 15, "cooldown_s": 60,
                  "taker_fee": 0.001},
    },
    "signal": {                    # Jev v2: the Jev Lab's question set, once a minute per coin, on lab-identical features
        "enabled": False,
        "prompt": "helix/prompts/present_stretch_reversion.json",
        "horizon_min": 15,
        "entry_prob": 0.54,        # P(up) >= this -> long, <= 1 - this -> short
        "max_age_s": 150,          # a signal older than this is ignored
        "max_continuation": 0.50,  # regime guard: sit a coin out when its 7-day continuation rate is above this
    },
    "lab_schedule": {"enabled": False, "every_days": 7, "variant": "present_stretch_reversion", "workers": 2},
    "claude_forecast": {
        "enabled": False,
        "model": "claude-opus-5-5",
        "interval_s": 300,
        "effort": "low",
        "daily_budget_usd": 5.0,
        "price_in_per_million": 4.0,
        "price_out_per_million": 20.0,
    },
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ConfigError("config.yaml must be a mapping")
        cfg = deep_merge(cfg, user)
    validate(cfg)
    return cfg


def _between(name: str, value, lo: float, hi: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi} (got {value!r})")


def validate(cfg: dict) -> None:
    if cfg.get("mode") != "paper":
        raise ConfigError("Only paper mode exists in this build. Live trading is intentionally not implemented.")
    if cfg.get("venue") != "hyperliquid":
        raise ConfigError("venue must be hyperliquid")
    if cfg.get("network") not in ("mainnet", "testnet"):
        raise ConfigError("network must be mainnet or testnet (market data only)")
    m = cfg["market"]
    if not m["coins"] or not all(isinstance(c, str) and c.isalnum() for c in m["coins"]):
        raise ConfigError("market.coins must be a list like [BTC, ETH, SOL]")
    if len(m["coins"]) > 30:
        raise ConfigError("market.coins: 30 markets at most (WebSocket subscriptions and request weight scale with it)")
    _between("market.decision_interval_ms", m["decision_interval_ms"], 200, 60000)
    _between("market.monitor_interval_ms", m["monitor_interval_ms"], 50, 5000)
    _between("market.horizon_s", m["horizon_s"], 5, 3600)
    _between("market.book_refresh_s", m["book_refresh_s"], 0.25, 30)
    _between("rules.min_hold_s", cfg["rules"]["min_hold_s"], 0, 3600)
    cf = cfg["claude_forecast"]
    _between("claude_forecast.interval_s", cf["interval_s"], 30, 86400)
    _between("claude_forecast.daily_budget_usd", cf["daily_budget_usd"], 0, 1000)
    if cf["effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise ConfigError("claude_forecast.effort must be low, medium, high, xhigh or max")
    p = cfg["paper"]
    _between("paper.starting_cash", p["starting_cash"], 100, 1e9)
    _between("paper.taker_fee_bps", p["taker_fee_bps"], 0, 100)
    _between("paper.maker_fee_bps", p["maker_fee_bps"], -10, 100)
    _between("paper.extra_slippage_bps", p["extra_slippage_bps"], 0, 100)
    if p["entry_mode"] not in ("taker", "maker"):
        raise ConfigError("paper.entry_mode must be 'taker' or 'maker'")
    _between("paper.entry_ttl_s", p["entry_ttl_s"], 5, 3600)
    root = Path(__file__).resolve().parent.parent
    for name, rel in (cfg["hourly_test"].get("jev_variants") or {}).items():
        if not (root / rel).is_file():
            raise ConfigError(f"hourly_test.jev_variants.{name}: {rel} not found")
    h4 = cfg["hip4"]
    if not h4["coins"] or not all(isinstance(c, str) and c.isalnum() for c in h4["coins"]):
        raise ConfigError("hip4.coins must be a list like [BTC, ETH]")
    _between("hip4.edge", h4["edge"], 0, 0.5)
    unknown = [v for v in h4["variants"] if v not in (cfg["hourly_test"].get("jev_variants") or {})]
    if h4["enabled"] and unknown:
        raise ConfigError(f"hip4.variants {unknown} must be names from hourly_test.jev_variants")
    tc = cfg["hip4_tape"]
    if not tc["coins"] or not all(isinstance(c, str) and c.replace(":", "").isalnum() for c in tc["coins"]):
        raise ConfigError("hip4_tape.coins must be a list like [BTC, ETH, xyz:SP500], or [all]")
    _between("hip4_tape.keep_days", tc["keep_days"], 1, 3650)
    _between("hip4_backtest.auto_every_h", cfg["hip4_backtest"]["auto_every_h"], 0, 24 * 7)
    if not isinstance(cfg["perp_trading"], bool):
        raise ConfigError("perp_trading must be true or false")
    mm = cfg["hip4_mm"]
    mm_all = [str(c).lower() for c in mm["coins"]] == ["all"]
    if mm["enabled"] and mm_all and not cfg["hip4_tape"]["enabled"]:
        raise ConfigError("hip4_mm.coins: [all] needs hip4_tape.enabled (it supplies prices for the xyz: perps)")
    if mm["enabled"] and not mm_all and (not mm["coins"] or not all(isinstance(c, str) and c in m["coins"] for c in mm["coins"])):
        raise ConfigError("hip4_mm.coins must be [all] or a list of coins that are also in market.coins (for the live price)")
    if mm["levels"] not in (1, 2):
        raise ConfigError("hip4_mm.levels must be 1 or 2")
    _between("hip4_mm.level2_edge_mult", mm["level2_edge_mult"], 1.1, 10)
    _between("hip4_mm.min_notional", mm["min_notional"], 0, 10000)
    _between("hip4_mm.requote_threshold", mm["requote_threshold"], 0, 0.5)
    _between("hip4_mm.max_markets", mm["max_markets"], 0, 10000)
    sn = cfg["hip4_sniper"]
    if sn["enabled"]:
        if [str(c).lower() for c in sn["coins"]] == ["all"] and not cfg["hip4_tape"]["enabled"]:
            raise ConfigError("hip4_sniper.coins: [all] needs hip4_tape.enabled (it supplies prices for the xyz: perps)")
        if sn["strategy"] != "sniper":
            raise ConfigError("hip4_sniper.strategy must be sniper")
    _between("hip4_sniper.snipe_edge", sn["snipe_edge"], 0.005, 0.5)
    _between("hip4_sniper.max_tte_min", sn["max_tte_min"], sn["stop_min"] + 1, 7 * 1440)
    _between("hip4_sniper.snipe_latency_ms", sn["snipe_latency_ms"], 0, 10000)
    _between("hip4_sniper.quote_size", sn["quote_size"], 1, 100000)
    _between("hip4_sniper.max_position", sn["max_position"], sn["quote_size"], 1e6)
    _between("hip4_sniper.max_risk_usd", sn["max_risk_usd"], 1, 1e9)
    _between("hip4_sniper.min_notional", sn["min_notional"], 0, 10000)
    _between("hip4_sniper.holdout_from", sn["holdout_from"], 0, 4e9)
    pb = sn["price_band"]
    if not (isinstance(pb, list) and len(pb) == 2 and all(isinstance(x, (int, float)) for x in pb) and 0 <= pb[0] < pb[1] <= 1):
        raise ConfigError("hip4_sniper.price_band must be [low, high] with 0 <= low < high <= 1")
    if sn["sides"] not in ("both", "buy", "sell", "auto"):
        raise ConfigError("hip4_sniper.sides must be both, buy, sell or auto")
    tr = sn["trend"]
    _between("hip4_sniper.trend.lookback_min", tr["lookback_min"], 5, 7 * 1440)
    _between("hip4_sniper.trend.threshold", tr["threshold"], 0, 0.5)
    if tr["neutral"] not in ("both", "buy", "sell", "none"):
        raise ConfigError("hip4_sniper.trend.neutral must be both, buy, sell or none")
    lab = cfg["hip4_lab"]
    if not isinstance(lab, dict):
        raise ConfigError("hip4_lab must be a mapping")
    for k in ("enabled", "autopilot", "kalshi", "xyz_open_guard"):
        if not isinstance(lab.get(k), bool):
            raise ConfigError(f"hip4_lab.{k} must be true or false")
    if lab["cex_feed"] in (False, None):                              # YAML reads a bare `off` as false
        lab["cex_feed"] = "off"
    if lab["cex_feed"] not in ("off", "coinbase"):
        raise ConfigError("hip4_lab.cex_feed must be off or coinbase")
    if lab["autopilot"] and not lab["enabled"]:
        raise ConfigError("hip4_lab.autopilot needs hip4_lab.enabled: true")
    if lab["enabled"] and not cfg["hip4_sniper"]["enabled"]:
        raise ConfigError("hip4_lab needs hip4_sniper.enabled: true (it reads the sniper's books and fair)")
    _between("hip4_lab.alpha", lab["alpha"], 0.01, 0.10)              # chance that luck alone ever qualifies
    _between("hip4_lab.season_size", lab["season_size"], 7, 50)       # strategies a season may ever try
    _between("hip4_lab.min_clusters", lab["min_clusters"], 50, 5000)  # independent outcomes (expiry clusters)
    _between("hip4_lab.min_side", lab["min_side"], 20, 1000)          # ... in rising AND in falling markets
    _between("hip4_lab.min_age_d", lab["min_age_d"], 0, 30)
    rd = cfg["research_desk"]
    if not isinstance(rd, dict) or not isinstance(rd.get("enabled"), bool):
        raise ConfigError("research_desk.enabled must be true or false")
    _between("research_desk.every_min", rd["every_min"], 10, 24 * 60)
    lv = cfg["hip4_live"]
    _between("hip4_live.repost_threshold", lv["repost_threshold"], 0.001, 0.5)
    _between("hip4_live.min_notional", lv["min_notional"], 0, 10000)
    _between("hip4_live.tick", lv["tick"], 0.0001, 0.01)
    _between("hip4_live.address_budget", lv["address_budget"], 0, 1e9)
    for k, lo, hi in (("days", 0, 365), ("settled", 1, 1e6), ("markout_n", 1, 1e6), ("backtest_hours", 0, 24 * 365),
                      ("shadow_validity", 0, 1)):
        _between(f"hip4_live.gate.{k}", lv["gate"][k], lo, hi)
    if not isinstance(lv["live_enabled"], bool) or not isinstance(lv["require_gate"], bool):
        raise ConfigError("hip4_live.live_enabled and hip4_live.require_gate must be true or false")
    pl = lv["pilot"]
    _between("hip4_live.pilot.capital_usd", pl["capital_usd"], 10, 1e7)
    _between("hip4_live.pilot.max_order_usd", pl["max_order_usd"], lv["min_notional"], pl["capital_usd"])
    _between("hip4_live.pilot.max_open_usd", pl["max_open_usd"], pl["max_order_usd"], pl["capital_usd"])
    _between("hip4_live.pilot.daily_loss_usd", pl["daily_loss_usd"], 0, pl["capital_usd"])     # 0 = no daily stop
    _between("hip4_live.pilot.contracts", pl["contracts"], 0, 1e6)
    _between("hip4_live.pilot.max_market_usd", pl["max_market_usd"], lv["min_notional"], pl["max_open_usd"])
    _between("hip4_live.pilot.long_tte_min", pl["long_tte_min"], 5, 7 * 1440)
    _between("hip4_live.pilot.long_budget_usd", pl["long_budget_usd"], 0, pl["max_open_usd"])
    cp = pl["compound"]
    if not isinstance(cp.get("enabled"), bool):
        raise ConfigError("hip4_live.pilot.compound.enabled must be true or false")
    # max_open_frac up to 1.0 = the whole account may be in open bets (the operator's call on a tiny account, 2026-10-09)
    for k, hi in (("open", 0.95), ("order", 0.2), ("market", 0.5), ("long", 0.5), ("max_open_frac", 1.0)):
        _between(f"hip4_live.pilot.compound.{k}", cp[k], 0.001, hi)
    _between("hip4_live.pilot.compound.daily_loss", cp["daily_loss"], 0, 0.5)                  # 0 = no daily stop
    if cp["open"] > cp["max_open_frac"]:
        raise ConfigError("hip4_live.pilot.compound.open must be <= max_open_frac")
    # a daily stop, if there is one, must be at least 2x the per-market cap (a $30 stop tripped on one ordinary losing
    # market; research 2026-10-08)
    if (0 < cp["daily_loss"] < 2 * cp["market"] - 1e-9) or (0 < pl["daily_loss_usd"] < 2 * pl["max_market_usd"] - 1e-9):
        raise ConfigError("a daily stop must be at least 2x the per-market cap, or 0 for none")
    _between("hip4_sniper.kill_drawdown", sn["kill_drawdown"], 0, 1)                            # 0 = no kill switch
    if pl["fire_on"] not in ("fill", "signal"):
        raise ConfigError("hip4_live.pilot.fire_on must be fill or signal")
    _between("hip4_live.pilot.slip", pl["slip"], 0, 0.2)
    _between("hip4_live.pilot.min_edge", pl["min_edge"], 0, 0.5)
    if lv["live_enabled"] and not sn["enabled"]:
        raise ConfigError("hip4_live.live_enabled trades the sniper's signals: enable hip4_sniper")
    if lv["shadow"] and not (mm["enabled"] or sn["enabled"]):
        raise ConfigError("hip4_live.shadow mirrors the HIP-4 sniper (or market maker): enable one of them")
    _between("hip4_mm.guard_move", mm["guard_move"], 0.005, 1)
    _between("hip4_mm.guard_s", mm["guard_s"], 5, 3600)
    _between("hip4_mm.zs_refresh_s", mm["zs_refresh_s"], 0, 86400)
    ex = mm["exits"]
    _between("hip4_mm.exits.take_profit", ex["take_profit"], 0.01, 1)
    _between("hip4_mm.exits.stop_loss", ex["stop_loss"], 0.01, 1)
    _between("hip4_mm.exits.min_hold_s", ex["min_hold_s"], 0, 3600)
    _between("hip4_mm.exits.cooldown_s", ex["cooldown_s"], 0, 3600)
    _between("hip4_mm.exits.taker_fee", ex["taker_fee"], 0, 0.05)
    _between("hip4_mm.edge", mm["edge"], 0.001, 0.3)
    _between("hip4_mm.skew", mm["skew"], 0, 0.3)
    _between("hip4_mm.quote_size", mm["quote_size"], 1, 100000)
    _between("hip4_mm.max_position", mm["max_position"], mm["quote_size"], 1000000)
    _between("hip4_mm.max_risk_usd", mm["max_risk_usd"], 1, 1e9)
    _between("hip4_mm.min_p", mm["min_p"], 0, 0.45)
    _between("hip4_mm.stop_min", mm["stop_min"], 0, 60)
    _between("hip4_mm.max_tte_min", mm["max_tte_min"], mm["stop_min"] + 1, 7 * 1440)
    _between("hip4_mm.book_stale_s", mm["book_stale_s"], 1, 600)
    _between("hip4_mm.fee", mm["fee"], 0, 0.05)
    _between("hip4_mm.start_cash", mm["start_cash"], 100, 1e9)
    _between("hip4_mm.kill_drawdown", mm["kill_drawdown"], 0.01, 1)
    _between("hip4_mm.quote_every_s", mm["quote_every_s"], 0.2, 60)
    _between("hip4_mm.market_weight", mm["market_weight"], 0, 1)
    _between("hip4_mm.blend_max_spread", mm["blend_max_spread"], 0.001, 1)
    _between("hip4_mm.latency_s", mm["latency_s"], 0, 10)
    _between("hip4_mm.max_size_multiplier", mm["max_size_multiplier"], 1, 50)
    s = cfg["signal"]
    if s["horizon_min"] not in (5, 15):
        raise ConfigError("signal.horizon_min must be 5 or 15 (the horizons the Jev Lab tested)")
    _between("signal.entry_prob", s["entry_prob"], 0.5, 0.99)
    _between("signal.max_age_s", s["max_age_s"], 30, 3600)
    _between("signal.max_continuation", s["max_continuation"], 0.3, 1.0)
    _between("lab_schedule.every_days", cfg["lab_schedule"]["every_days"], 1, 60)
    if p.get("exit_mode", "taker") not in ("taker", "maker"):
        raise ConfigError("paper.exit_mode must be 'taker' or 'maker'")
    _between("paper.exit_window_s", p.get("exit_window_s", 120), 10, max(10, cfg["risk"]["max_hold_s"] - 10))
    if s["enabled"] and not (root / s["prompt"]).is_file():
        raise ConfigError(f"signal.prompt: {s['prompt']} not found")
    j = cfg["jev"]
    if j["fallback"] not in ("rules", "hold"):
        raise ConfigError("jev.fallback must be 'rules' or 'hold'")
    _between("jev.deadline_ms", j["deadline_ms"], 50, 30000)
    _between("jev.daily_budget_usd", j["daily_budget_usd"], 0, 100000)
    if j["deadline_ms"] >= m["decision_interval_ms"]:
        raise ConfigError("jev.deadline_ms must be shorter than market.decision_interval_ms (late answer = no trade)")
    pol = cfg["policy"]
    for key in ("min_direction_prob", "full_size_prob", "min_setup_prob", "max_liquidity_stress",
                "toxic_flow_threshold", "max_crisis_prob", "exit_direction_prob"):
        _between(f"policy.{key}", pol[key], 0, 1)
    _between("policy.min_edge_bps", pol["min_edge_bps"], 0, 500)
    _between("policy.min_hold_s", pol["min_hold_s"], 0, 3600)
    _between("policy.confirm_ticks", pol["confirm_ticks"], 1, 120)
    _between("risk.reentry_cooldown_s", cfg["risk"]["reentry_cooldown_s"], 0, 86400)
    _between("policy.exit_inventory_pressure", pol["exit_inventory_pressure"], 0, 3)
    r = cfg["risk"]
    _between("risk.risk_per_trade", r["risk_per_trade"], 0.0001, 0.05)
    _between("risk.max_position_leverage", r["max_position_leverage"], 0.05, 10)
    _between("risk.max_total_leverage", r["max_total_leverage"], 0.05, 10)
    _between("risk.max_daily_loss_pct", r["max_daily_loss_pct"], 0, 0.5)  # 0 = no daily stop
    _between("risk.max_drawdown_pct", r["max_drawdown_pct"], 0.005, 0.9)
    _between("risk.min_edge_multiple", r["min_edge_multiple"], 0, 20)
    _between("risk.max_entries_per_min", r["max_entries_per_min"], 1, 600)
    if int(r["max_open_positions"]) < 1:
        raise ConfigError("risk.max_open_positions must be at least 1")
    _between("risk.max_bet_multiplier", r["max_bet_multiplier"], 1, 10)
    _between("risk.bet_multiplier", r["bet_multiplier"], 0.1, r["max_bet_multiplier"])
    _between("risk.leverage_ceiling_position", r["leverage_ceiling_position"], r["max_position_leverage"], 20)
    _between("risk.leverage_ceiling_total", r["leverage_ceiling_total"], r["max_total_leverage"], 40)
    st = r["streak"]
    _between("risk.streak.after_wins", st["after_wins"], 1, 20)
    _between("risk.streak.step", st["step"], 0.05, 5)
    _between("risk.streak.max", st["max"], 1, r["max_bet_multiplier"])
    _between("risk.streak.gate_trades", st["gate_trades"], 0, 1000)
