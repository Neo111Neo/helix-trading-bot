"""Shadow lab: a ZERO-RISK strategy tournament on the live feed, and an evidence-gated AUTOPILOT (2026-10-12).

Several variants of the sniper are scored side by side on every Yes-book update the sniper receives, as the live pilot
would trade them - $10 orders (1c slip, never closer than 3c to fair), a 15 s per-market cooldown, $30 per market,
$40 for markets > 120 min out, $100 open per variant - and NOTHING is sent. FILLS (rule "et300", 2026-10-22): a
would-be IOC sent at detection + OVERHEAD_MS fills only if the FIRST Yes-book update whose exchange time is at least
FILL_DELAY_MS after the send still shows a price at or through our limit; it gets at most the size shown there, at the
price shown there, and is held to the real settlement, net of fees. No such update within CHECK_MAX_S = a miss. Why:
HIP-4 books arrive as ~5.4 s snapshots, and the old rule ("filled unless seen gone within ~0.1 s of arrival", checked
on those snapshots) booked 92-97% of decisions as filled. On 387 real live IOCs it agreed with reality 32% of the time;
this rule agreed 91% (fills 21-22% vs 24.5% real; research EDGE_HUNT_R2_2026-10-21.md). The quotes that vanish are
the profitable ones, so the old rule scored phantom winners. LIMITS (review 2026-10-22): those IOCs were mostly sent on
stale snapshots; the lab always decides on a FRESH one, where the rule agreed only ~66% (29 IOCs: 66% really filled,
38% predicted) - it may undercount fills there, and the extra real fills looked like losers (-3.6c at 300 s, small n).
Replayed on the lab's own logged decisions it fills ~57% (Kalshi variants ~40%) and matches the research simulator on
98% of them. Measure it against real Tokyo IOCs before trusting the gate's fills fully.
Fills walk the levels shown (bid_lv / ask_lv from the sniper's book, top level if absent) up to the limit, as an IOC
sweeps them; caps are reserved at sz x the limit price, as the live pilot reserves.

Why: on 2026-10-09..12 the live sell-only rules lost while Tokyo's fills were fine (93-96%); the Oct 6-8 profits were
mostly the downtrend. The variants test what might hold in BOTH directions (the last 15 minutes, both sides, a
self-calibrated fair).

AUTOPILOT (hip4_lab.autopilot): live trades the variant with real evidence, else nothing. Evidence is an anytime-valid
PAPER-BETTING SCORE (an e-process), because the gate is re-checked every hour: a t-test re-run hourly on sliding windows
let a no-edge variant through ~31% of the time per WEEK (review 2026-10-12, scratchpad r5_agent / r5_verify), even
though one look was ~5%. Per variant, once every bet of an EXPIRY CLUSTER (all markets settling at the same time = one
outcome) has settled, r = its net P&L / BUDGET updates W = mean over LAMS of prod(1 + lam * min(r, 3)):
  * r is divided by the FIXED $100 budget (the most a variant can have open), not by the realized stake - with add-on
    bets that depend on interim moves, P&L / stake can average > 0 even when every bet has zero edge (review r6);
  * a cluster is scored only if its FIRST bet came at or after the expiry of the last scored cluster (overlapping long
    bets would make outcomes that the earlier stakes did not know about compound into each other); skipped ones are
    counted but never scored - which ones are skipped is fixed before their outcome;
  * every cluster is scored once (its key is kept while any of its rows exists).
With no edge W is a nonnegative supermartingale, so P(W EVER reaches K / alpha) <= alpha / K however often it is looked
at (Ville). K = the SEASON SIZE fixed up front (hip4_lab.season_size); at most K variants (name + params, counted for
good once run) may ever qualify in a season, so the chance that ANY no-edge variant qualifies stays <= alpha. alpha and
the minima are pinned in the season state: a later config can only make them stricter. A variant also needs
>= min_clusters scored outcomes over >= min_age_d days, >= min_side outcomes in both RISING and FALLING markets
(labelled per variant by the coin's 6 h trend BEFORE its own first bet in that market - not by the outcome, which
mechanically failed every one-sided variant) with W >= 1 on each side, and a positive last 24 h. A variant's evidence
is kept across restarts, retirements and re-additions (no free restarts). Passing means "eligible for a small live
trial", not "proven".
The ONE exception is a change of the fill rule (FILL_RULE): evidence measured with other fills is not evidence about
these, so on the first start with a new rule every variant starts again from 1x with a fresh ledger (and a fresh
min_age_d clock), the old evidence is kept under "archive_rules" for the record and is never resumed, and the season
(its tried list, bar and minima) carries on unchanged - no new slots. This is not a free restart: it is made once, for
a reason fixed before any outcome under the new rule, and no variant had qualified, so Ville's bound holds from the
switch for the rest of the season.
tests/test_shadow_lab.py runs zero-edge and drift-trap simulations checked after every outcome (< 5% false picks), an
outage / quiet-variant double-count check and the overlap rule.

Self-calibration: every market's RAW model fair is recorded once ~5-15 min before expiry (bet or not); after settlement
y - fair is averaged per expiry cluster over the last BIAS_CLUSTERS clusters; the shift is used only when it is beyond
2 standard errors, clipped to 10c, and applied in log-odds (smaller in the tails).

State: data/shadow_lab.json; log: data/shadow_lab.jsonl."""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import threading
import time
from pathlib import Path

from .hip4_mm import trend_return

VARIANTS = {
    "live_rules":      {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False},
    "sell_last15":     {"sides": "sell", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False},
    "both_last15":     {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False},
    "both_last15_7c":  {"sides": "both", "edge": 0.07, "min_tte": 3, "max_tte": 15, "selfcal": False},
    "both_24h":        {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": False},
    "selfcal_24h":     {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 1440, "selfcal": True},
    "selfcal_60":      {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 60, "selfcal": True},
}
# added by the server only when their outside feed is on (helix/ref_feeds.py): each counts against the season once run
EXTRA_VARIANTS = {
    # take HIP-4's quote when it sits >= 5c through Kalshi's on the same 15-minute BTC/ETH question. Only 7-15 min
    # before expiry: closer in, the drift of Hyperliquid's perp premium over Kalshi's index (the two settle on
    # different prices) moves the right HIP-4 price by more than 5c (review r7: p90 8-11c at 3-5 min)
    # src "orderbook" (2026-10-13): the first run read Kalshi's ~10 s market snapshot, a different (stale) test - its
    # evidence stays archived under the old key and this version starts from 1x
    "kalshi_5c":          {"sides": "both", "edge": 0.05, "min_tte": 7, "max_tte": 15, "selfcal": False, "ref": "kalshi",
                           "src": "orderbook"},
    # the same at 3c (2026-10-13): on the Oct 6-13 minute-candle backtest 3c made +10c/contract over 171 fills (t 2.4)
    # vs 5c's +6.4c (t 1.1) - chosen after looking, so it earns its evidence here from 1x like everything else
    "kalshi_3c":          {"sides": "both", "edge": 0.03, "min_tte": 7, "max_tte": 15, "selfcal": False, "ref": "kalshi",
                           "src": "orderbook"},
    # both_last15 on BTC/ETH, skipping a bet when Coinbase moved >= 0.5 bp against it in the last 2 s; no fresh
    # Coinbase price = no bet (the evidence measures the rule as written)
    "both_last15_cbveto": {"sides": "both", "edge": 0.05, "min_tte": 3, "max_tte": 15, "selfcal": False,
                           "veto_bp": 0.5, "veto_s": 2.0},
}
GIVE_UP_S = 7 * 86400
# BET-SIZE LADDER (2026-10-15, display only - never the gate, never live): every variant's own fills are replayed at
# 2x..5x the ticket ($20..$50 orders; $60..$150 per market, $200..$500 open, long-dated budget scaled the same) on the
# SAME decisions, each filled only up to the size that was on the book's top level at that moment (an IOC takes no
# more). 1x is a rung too (not the variant's own totals, which include bets placed before the ladder started), so
# every size covers the same bets. Shows how the money and the fill share change with the live size buttons.
LADDER, LADDER_VERSION = (1, 2, 3, 4, 5), 2
# the paper-betting fractions averaged in the score. r = P&L / $100 >= -1.02 (fees), so 1 + 0.9 r > 0 always; on the
# typical $20-30 per cluster these are 0.05-0.9% of the paper wealth per $ staked, which covers the Kelly fraction of a
# +5% to +15% per $ edge
LAMS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.9)
EP_VERSION = 2                                             # bump when LAMS or the scoring of r change
LOG_CAP = 50.0                                             # shown scores stop at e^50 (the comparisons use logs)


def _params_hash(v: dict) -> str:
    return hashlib.sha1(json.dumps(v, sort_keys=True).encode()).hexdigest()[:10]


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _tstat(xs: list) -> tuple[int, float, float]:
    n = len(xs)
    if n < 2:
        return n, (xs[0] if xs else 0.0), 0.0
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return n, m, (m / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def _blank_ep() -> dict:
    """Paper-betting score state: log wealth per lam (all outcomes, rising-market and falling-market ones), the expiry
    of the last scored cluster, and the clusters already handled (expiry -> when)."""
    z = [0.0] * len(LAMS)
    return {"v": EP_VERSION, "lw": list(z), "up": list(z), "dn": list(z), "n": 0, "n_up": 0, "n_dn": 0, "skipped": 0,
            "last_exp": 0.0, "done": {}}


def _ok_ep(ep) -> bool:
    """A saved score made with THESE LAMS and scoring rules (an older format starts again from 1x)."""
    return isinstance(ep, dict) and ep.get("v") == EP_VERSION and isinstance(ep.get("done"), dict) and \
        all(isinstance(ep.get(k), list) and len(ep[k]) == len(LAMS) for k in ("lw", "up", "dn"))


def _shown(lw: float) -> float:
    return round(math.exp(min(lw, LOG_CAP)), 3)


def _blank_rung() -> dict:
    """One rung of the bet-size ladder: positions and the totals shown on the board."""
    return {"pos": {}, "wanted": 0.0, "got": 0.0, "fills": 0, "partial": 0, "refused": 0, "markets": 0, "won": 0,
            "realized": 0.0, "staked": 0.0, "contracts": 0.0}


def _us_open(day_ts: float) -> float:
    """09:30 New York time on the UTC date of day_ts, as a UTC timestamp (US daylight saving: 2nd Sunday of March
    07:00 UTC to 1st Sunday of November 06:00 UTC)."""
    t = time.gmtime(day_ts)
    day0 = calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0))

    def sunday(month: int, n: int) -> int:
        first = calendar.timegm((t.tm_year, month, 1, 0, 0, 0))
        return first + ((6 - time.gmtime(first).tm_wday) % 7 + 7 * (n - 1)) * 86400

    dst = sunday(3, 2) + 7 * 3600 <= day0 + 13.5 * 3600 < sunday(11, 1) + 6 * 3600
    return day0 + (13.5 if dst else 14.5) * 3600


def _spans_us_open(now: float, expiry: float) -> bool:
    """True when a bet placed now and held to expiry spans a weekday US stock-market open, or is placed in the first
    15 minutes after one (the xyz: perps jumped there in 42% of 5-minute windows vs ~1% otherwise, 2026-10-06..12)."""
    for d in range(0, int(max(0.0, expiry - now) // 86400) + 2):
        op = _us_open(now + d * 86400)
        if time.gmtime(op).tm_wday >= 5:
            continue
        if now < op <= expiry or op <= now < op + 900:
            return True
    return False


def _log_wealth(lw: list) -> float:
    """log of the mean over lam of exp(lw): the score of betting each fraction with 1/len(LAMS) of the $1."""
    m = max(lw)
    return m + math.log(sum(math.exp(x - m) for x in lw) / len(lw))


class ShadowLab:
    ORDER_USD, MARKET_CAP, BUDGET = 10.0, 30.0, 100.0
    LONG_TTE_MIN, LONG_BUDGET = 120.0, 40.0
    COOL_S, SLIP, MIN_EDGE = 15.0, 0.01, 0.03
    OVERHEAD_MS, FEE = 5.0, 0.0005
    FILL_RULE, FILL_DELAY_MS, CHECK_MAX_S = "et300", 300.0, 20.0     # see the module docstring
    BIAS_CLUSTERS, BIAS_CLIP = 40, 0.10
    SELECT_EVERY_S, CONFIRM_H, MIN_CONFIRM, SWITCH_X = 3600.0, 24.0, 5, 2.0
    ALPHA, MIN_CLUSTERS, MIN_SIDE, MIN_AGE_D, SEASON_SIZE = 0.05, 300, 100, 7.0, 15
    R_CLIP, TREND_LOOKBACK_S, TREND_DEAD = 3.0, 6 * 3600.0, 0.002
    REF_MAX_MOVE, REF_MODEL_TOL = 0.0003, 0.10              # Kalshi: max 3 bp move since its quote; model backstop
    KEEP_S, SAVE_EVERY_S = 8 * 86400, 30.0

    def __init__(self, sniper, data_dir: Path | None, variants: dict | None = None, clock=time.time, rest=None,
                 autopilot: bool = False, alpha: float | None = None, min_clusters: int | None = None,
                 min_side: int | None = None, min_age_d: float | None = None, season_size: int | None = None,
                 refs: dict | None = None, xyz_open_guard: bool = True):
        self.sniper, self.clock = sniper, clock
        self.refs = dict(refs or {})                       # "kalshi" / "cex" feeds (helix/ref_feeds.py), optional
        self.xyz_open_guard = xyz_open_guard
        self.rest = rest if rest is not None else getattr(sniper, "rest", None)
        self.variants = dict(variants or VARIANTS)
        self.autopilot = autopilot
        self.live_hook = None                              # the live pilot's on_signal when autopilot drives live
        self.selected: str | None = None                   # never restored from disk: re-earned on every start
        self.selection: dict = {"ts": None, "why": "not chosen yet", "scores": []}
        self._sel_at = 0.0
        if alpha is not None:
            self.ALPHA = float(alpha)
        if min_clusters is not None:
            self.MIN_CLUSTERS = int(min_clusters)
        if min_side is not None:
            self.MIN_SIDE = int(min_side)
        if min_age_d is not None:
            self.MIN_AGE_D = float(min_age_d)
        if season_size is not None:
            self.SEASON_SIZE = int(season_size)
        self.path = Path(data_dir) / "shadow_lab.json" if data_dir is not None else None
        self.log_path = Path(data_dir) / "shadow_lab.jsonl" if data_dir is not None else None
        self.lock = threading.Lock()
        self.since = clock()
        self.error: str | None = None
        self._fails = 0
        self._saved_at = 0.0
        self.meta: dict[int, dict] = {}                    # market id -> coin, strike, expiry (while held / pending)
        self.ckpt: dict[int, dict] = {}                    # market id -> raw model fair ~5-15 min before expiry
        self.obs: list[list] = []                          # settled checkpoint markets: [ts, mid, expiry, y - raw fair]
        self.led = {n: self._blank(v) for n, v in self.variants.items()}
        for L in self.led.values():
            L["born"] = self.since
        self.ladder: dict = {}                             # built after _load (it needs the restored 1x totals)
        self._loaded_ladder: dict = {}
        self.tried: list[str] = []                         # every variant (name:params) run this season, in order
        self.archive: dict[str, dict] = {}                 # name:params -> {ep, born} of every variant ever run
        self.season: dict = {}                             # the rules pinned when the season began
        self._retry: dict[int, tuple] = {}
        self._checks: dict[int, list] = {}                 # market id -> would-be IOCs waiting for their fill check
        self.fill_rule_since = self.since                  # when evidence under FILL_RULE began
        self.archive_rules: dict[str, dict] = {}           # evidence measured under earlier fill rules (never resumed)
        self._load()
        for n, L in self.led.items():
            self._try(f"{n}:{L['params']}")
        self._pin_rules()
        for n, L in self.led.items():                      # the bet-size ladder: restored, or started now
            old = self._loaded_ladder.get(n)
            if isinstance(old, dict) and old.get("params") == L["params"] and isinstance(old.get("rungs"), dict) \
                    and old.get("v") == LADDER_VERSION:
                rungs = {}
                for m in LADDER:
                    R = {**_blank_rung(), **(old["rungs"].get(str(m)) or {})}
                    R["pos"] = {int(k): v for k, v in (R.get("pos") or {}).items() if str(k).isdigit()}
                    rungs[m] = R
                self.ladder[n] = {**old, "rungs": rungs}
            else:
                self.ladder[n] = {"params": L["params"], "v": LADDER_VERSION, "since": clock(),
                                  "rungs": {m: _blank_rung() for m in LADDER}}
        self._loaded_ladder = {}
        if sniper is not None:
            hooks = getattr(sniper, "book_hooks", None)
            if hooks is not None:
                hooks.append(self.on_book)

    @staticmethod
    def _blank(v: dict) -> dict:
        return {"params": _params_hash(v), "pos": {}, "pending": {}, "cool": {}, "signals": 0, "sent": 0, "refused": 0,
                "filled": 0, "resolved": 0, "realized": 0.0, "markets": 0, "won": 0, "contracts": 0.0, "staked": 0.0,
                "settled": [], "ep": _blank_ep(), "born": 0.0, "first": {}, "vetoed": 0}
        # settled: [ts, mid, expiry, pnl, cost, trend, first_bet_ts] - one row per market; first: mid -> [first bet ts,
        # the coin's 6 h log return before it] for THIS variant (None when unknown)

    # ------------------------------------------------------------------ evidence
    def _try(self, key: str) -> None:
        if key not in self.tried:
            self.tried.append(key)

    def _pin_rules(self) -> None:
        """The season's rules are fixed when it begins; a later config may only make them stricter."""
        s = self.season
        if s:
            self.ALPHA = min(self.ALPHA, float(s.get("alpha", self.ALPHA)))
            self.SEASON_SIZE = max(self.SEASON_SIZE, int(s.get("size", self.SEASON_SIZE)))
            self.MIN_CLUSTERS = max(self.MIN_CLUSTERS, int(s.get("min_clusters", self.MIN_CLUSTERS)))
            self.MIN_SIDE = max(self.MIN_SIDE, int(s.get("min_side", self.MIN_SIDE)))
            self.MIN_AGE_D = max(self.MIN_AGE_D, float(s.get("min_age_d", self.MIN_AGE_D)))
        self.season = {"alpha": self.ALPHA, "size": self.SEASON_SIZE, "min_clusters": self.MIN_CLUSTERS,
                       "min_side": self.MIN_SIDE, "min_age_d": self.MIN_AGE_D,
                       "started": s.get("started", self.since) if s else self.since}

    def bar(self) -> float:
        """The score a variant must reach: season size / alpha (fixed from the start of the season)."""
        return self.SEASON_SIZE / self.ALPHA

    def eligible(self, name: str) -> bool:
        """Only the first SEASON_SIZE variants run this season may ever qualify (the bar was set for that many)."""
        key = f"{name}:{self.led[name]['params']}"
        return key in self.tried[:self.SEASON_SIZE]

    @classmethod
    def _ep_update(cls, ep: dict, r: float, trend: float | None) -> None:
        """One scored expiry cluster: bet each fraction lam of the paper wealth on r = its net P&L / BUDGET."""
        x = min(r, cls.R_CLIP)
        inc = [math.log(max(1e-12, 1 + lam * x)) for lam in LAMS]
        ep["lw"] = [a + b for a, b in zip(ep["lw"], inc)]
        ep["n"] += 1
        side = "up" if trend is not None and trend >= cls.TREND_DEAD else \
            "dn" if trend is not None and trend <= -cls.TREND_DEAD else None
        if side:
            ep[side] = [a + b for a, b in zip(ep[side], inc)]
            ep["n_" + side] += 1

    def _close_clusters(self, now: float) -> None:
        """Score every expiry cluster whose bets have all settled (or voided), once, in expiry order. A cluster with a
        bet still open waits; one whose first bet came before the last scored expiry is skipped (see the docstring)."""
        with self.lock:
            for L in self.led.values():
                ep = L["ep"]
                L["settled"] = [s for s in L["settled"] if s[0] >= now - self.KEEP_S]
                busy = {(self.meta.get(k) or {}).get("expiry") for k in list(L["pos"]) + list(L["pending"])}
                agg: dict = {}
                for row in L["settled"]:
                    exp = row[2]
                    if exp is None or exp in busy or str(exp) in ep["done"]:
                        continue
                    a = agg.setdefault(exp, [0.0, 0.0, 0.0, 0.0, math.inf])  # pnl, cost, trend x cost, cost w/ trend, t0
                    a[0] += row[3]
                    a[1] += row[4]
                    tr = row[5] if len(row) > 5 else None
                    if tr is not None:
                        a[2] += tr * row[4]
                        a[3] += row[4]
                    t0 = row[6] if len(row) > 6 else None
                    a[4] = min(a[4], -math.inf if t0 is None else t0)          # unknown first bet: treat as overlapping
                for exp in sorted(agg):
                    pnl, cost, tw, tc, t0 = agg[exp]
                    ep["done"][str(exp)] = now
                    if cost <= 0:
                        continue
                    if t0 < ep.get("last_exp", 0.0):
                        ep["skipped"] = ep.get("skipped", 0) + 1
                        continue
                    ep["last_exp"] = max(ep.get("last_exp", 0.0), float(exp))
                    self._ep_update(ep, pnl / self.BUDGET, tw / tc if tc > 0 else None)
                live = {str(r[2]) for r in L["settled"]}           # a key lives while any of its rows does
                ep["done"] = {k: v for k, v in ep["done"].items() if k in live or v >= now - self.KEEP_S - 86400}
                L["first"] = {k: v for k, v in L["first"].items()      # firsts of bets that never filled
                              if k in L["pos"] or k in L["pending"] or v[0] >= now - 2 * 86400}

    def _side_stats(self, L: dict) -> dict:
        """Display only (2026-10-13): the variant's settled expiry clusters of the ledger (last KEEP_S) split by the
        market's 6 h trend before the bet - rising / falling / flat-or-unknown - with n, net $ and mean $ per result,
        and the DIRECTION-NEUTRAL mean = (rising mean + falling mean) / 2: what the strategy makes per result when rising
        and falling markets count equally, i.e. with the market's drift taken out. Caller holds the lock."""
        agg: dict = {}
        for row in L["settled"]:
            a = agg.setdefault(row[2], [0.0, 0.0, 0.0])           # pnl, trend x cost, cost with trend
            a[0] += row[3]
            tr = row[5] if len(row) > 5 else None
            if tr is not None:
                a[1] += tr * row[4]
                a[2] += row[4]
        groups: dict = {"rising": [], "falling": [], "flat": []}
        for pnl, tw, tc in agg.values():
            tr = tw / tc if tc > 0 else None
            k = "rising" if tr is not None and tr >= self.TREND_DEAD else \
                "falling" if tr is not None and tr <= -self.TREND_DEAD else "flat"
            groups[k].append(pnl)

        def summary(xs: list) -> dict:
            n = len(xs)
            m = sum(xs) / n if n else None
            sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else None
            return {"n": n, "net": round(sum(xs), 2), "mean": None if m is None else round(m, 2),
                    "se": None if sd is None else sd / math.sqrt(n)}
        out = {k: summary(v) for k, v in groups.items()}
        up, dn = out["rising"], out["falling"]
        if up["mean"] is not None and dn["mean"] is not None:
            m = (up["mean"] + dn["mean"]) / 2
            se = 0.5 * math.sqrt((up["se"] or 0.0) ** 2 + (dn["se"] or 0.0) ** 2) if up["n"] > 1 and dn["n"] > 1 else 0.0
            out["neutral"] = {"mean": round(m, 2), "t": round(m / se, 2) if se > 0 else None}
        else:
            out["neutral"] = None
        for k in ("rising", "falling", "flat"):
            out[k].pop("se", None)
        return out

    def scores(self, now: float) -> list[dict]:
        c0 = now - self.CONFIRM_H * 3600
        out = []
        with self.lock:
            for name, L in self.led.items():
                ep = L["ep"]
                recent: dict = {}
                for row in L["settled"]:
                    if row[0] >= c0:
                        recent[row[2]] = recent.get(row[2], 0.0) + row[3]
                lw, lu, ld = _log_wealth(ep["lw"]), _log_wealth(ep["up"]), _log_wealth(ep["dn"])
                out.append({"name": name, "score": _shown(lw), "log_score": round(lw, 4),
                            "clusters": ep["n"], "skipped": ep.get("skipped", 0),
                            "age_d": round((now - L.get("born", now)) / 86400, 2),
                            "up": ep["n_up"], "down": ep["n_dn"], "score_up": _shown(lu), "score_down": _shown(ld),
                            "log_up": round(lu, 4), "log_down": round(ld, 4),
                            "confirm_clusters": len(recent), "net_confirm": round(sum(recent.values()), 2)})
        return sorted(out, key=lambda r: r["log_score"], reverse=True)

    def _qualifies(self, s: dict) -> tuple[bool, str]:
        bar = self.bar()
        if s.get("name") in self.led and not self.eligible(s["name"]):
            return False, f"added after the season's {self.SEASON_SIZE} strategies: display only"
        if s["log_score"] < math.log(bar):
            return False, f"score {s['score']:.2f}x of {bar:.0f}x needed"
        if s["clusters"] < self.MIN_CLUSTERS or s["age_d"] < self.MIN_AGE_D:
            return False, f"{s['clusters']} outcomes over {s['age_d']:.1f} d (needs {self.MIN_CLUSTERS} over " \
                          f"{self.MIN_AGE_D:g} d)"
        if s["up"] < self.MIN_SIDE or s["down"] < self.MIN_SIDE:
            return False, f"{s['up']} rising / {s['down']} falling-market outcomes (needs {self.MIN_SIDE} of each)"
        if s["log_up"] < 0 or s["log_down"] < 0:
            return False, "loses in rising or in falling markets"
        if s["confirm_clusters"] < self.MIN_CONFIRM or s["net_confirm"] <= 0:
            return False, f"not positive on the last {self.CONFIRM_H:g} h"
        return True, "ok"

    def select(self, now: float) -> str | None:
        """Live trades the qualifying variant with the highest score (see the module docstring), or NOTHING; a
        challenger replaces the current choice only with SWITCH_X times its score. Safe to call as often as wanted."""
        self._close_clusters(now)
        sc = self.scores(now)
        for s in sc:
            s["ok"], s["why"] = self._qualifies(s)
        ok = [s for s in sc if s["ok"]]
        cur = next((s for s in ok if s["name"] == self.selected), None)
        best = ok[0] if ok else None
        bar = self.bar()
        if best is None:
            lead = f"; best so far {sc[0]['name']} at {sc[0]['score']:.2f}x" if sc else ""
            pick, why = None, (f"no strategy has earned enough evidence yet (score {bar:.0f}x, "
                               f"{self.MIN_CLUSTERS}+ outcomes over {self.MIN_AGE_D:g}+ days, wins in rising and "
                               f"falling markets, positive last {self.CONFIRM_H:g} h){lead}: live sits out")
        elif cur is not None and best["name"] != cur["name"] and \
                best["log_score"] < cur["log_score"] + math.log(self.SWITCH_X):
            pick, why = cur["name"], (f"keeping {cur['name']} ({cur['score']:.0f}x); {best['name']} "
                                      f"({best['score']:.0f}x) not clearly better")
        else:
            pick, why = best["name"], (f"{best['name']}: score {best['score']:.0f}x (needs {bar:.0f}x) over "
                                       f"{best['clusters']} outcomes, +${best['net_confirm']} on the last "
                                       f"{self.CONFIRM_H:g} h")
        self.selected, self._sel_at = pick, now
        self.selection = {"ts": now, "why": why, "scores": sc}
        self._log({"ts": now, "event": "select", "selected": pick, "why": why})
        return pick

    # ------------------------------------------------------------------ self-calibration
    def bias(self) -> float:
        """Mean (y - raw fair) over the last BIAS_CLUSTERS expiry clusters, used only beyond 2 standard errors."""
        with self.lock:
            by: dict = {}
            for o in self.obs:
                by.setdefault(o[2], []).append(o[3])
            xs = [sum(v) / len(v) for k, v in sorted(by.items())][-self.BIAS_CLUSTERS:]
        n, m, t = _tstat(xs)
        if n < 10 or abs(t) < 2:
            return 0.0
        return max(-self.BIAS_CLIP, min(self.BIAS_CLIP, m))

    @staticmethod
    def _shift(p: float, b: float) -> float:
        return _sigmoid(_logit(p) + 4 * b) if b else p    # ~ +b at p = 0.5, smaller in the tails

    # ------------------------------------------------------------------ signals (WebSocket thread)
    def _signal(self, v: dict, fair: float, b: dict):
        bid, ask = b.get("bid"), b.get("ask")
        if v["sides"] in ("both", "sell") and bid is not None and bid >= fair + v["edge"] \
                and b.get("bid_sz", 0) * (1 - bid) >= self.ORDER_USD - 1e-9:
            limit_pay = round(max(1 - bid, min(1 - bid + self.SLIP, (1 - fair) - self.MIN_EDGE)), 4)
            return -1, bid, 1 - bid, limit_pay, round(1 - limit_pay, 4)
        if v["sides"] in ("both", "buy") and ask is not None and ask <= fair - v["edge"] \
                and b.get("ask_sz", 0) * ask >= self.ORDER_USD - 1e-9:
            limit_pay = round(max(ask, min(ask + self.SLIP, fair - self.MIN_EDGE)), 4)
            return 1, ask, ask, limit_pay, limit_pay
        return None

    def on_book(self, mid: int, now: float) -> None:
        sn = self.sniper
        m, b = sn.markets.get(mid), sn.books.get(mid)
        if b is not None:
            self._run_checks(mid, now, b)                  # earlier would-be IOCs on this book first
        if m is None or b is None:
            return
        p, spot = sn.models.get(mid), None
        try:                                               # fresh fair on the perp price now (as the live fast path)
            spot = sn.spot_fn(m["coin"])
            fresh = sn.model.fair(m["coin"], spot, m["strike"], now, m["expiry"], m["twap_s"]) if spot else None
            if fresh is not None:
                p = fresh
        except Exception:
            pass
        if p is None or not (sn.min_p <= p <= 1 - sn.min_p):
            return
        tte_min = (m["expiry"] - now) / 60
        if 5 < tte_min <= 15 and mid not in self.ckpt:
            with self.lock:
                self.ckpt[mid] = {"fair": round(p, 4), "ts": now, "expiry": m["expiry"]}
        if self.xyz_open_guard and str(m["coin"]).startswith("xyz:") and _spans_us_open(now, m["expiry"]):
            return                                         # the model has no market-hours clock: sit the open out
        bias = None
        order = sorted(self.variants, key=lambda n: n != self.selected)      # the live variant first (latency)
        for name in order:
            v = self.variants[name]
            if not (v["min_tte"] < tte_min <= v["max_tte"]):
                continue
            if v.get("ref") == "kalshi":                   # Kalshi's price is the fair
                q = self._kalshi_quote(m, now, spot, p)
                if q is None:
                    continue
                sig, fair = (self._signal({**v, "sides": "sell"}, q["ask"], b), q["ask"]) \
                    if v["sides"] in ("both", "sell") else (None, None)
                if sig is None and v["sides"] in ("both", "buy"):
                    sig, fair = self._signal({**v, "sides": "buy"}, q["bid"], b), q["bid"]
                if sig is not None and ((sig[0] < 0 and sig[1] < p - self.REF_MODEL_TOL) or
                                        (sig[0] > 0 and sig[1] > p + self.REF_MODEL_TOL)):
                    continue                               # backstop: never a bet the model calls 10c+ bad
                raw = fair
            else:
                if v["selfcal"]:
                    bias = self.bias() if bias is None else bias
                    fair = self._shift(p, bias)
                else:
                    fair = p
                sig, raw = self._signal(v, fair, b), p
            if sig is None:
                continue
            if v.get("veto_bp") and self._vetoed(v, m["coin"], sig[0], now):
                self.led[name]["vetoed"] = self.led[name].get("vetoed", 0) + 1
                continue
            if self.autopilot and name == self.selected and self.live_hook is not None:
                self._to_live(name, mid, m, sig, fair, b, now, raw)
            self._book(name, mid, m, sig, fair, b, now)

    def _kalshi_quote(self, m: dict, now: float, spot: float | None, p: float) -> dict | None:
        """Kalshi's live quote for the same question - a HIP-4 15-minute BTC/ETH market ('out', quarter-hour expiry)
        whose strike is within 10 bp of Kalshi's reference price, with a sane Kalshi book - moved by what the model says
        the price move since Kalshi's quote is worth (Kalshi is ~1 s behind; review r7). None when it cannot be trusted:
        no Hyperliquid price at either time, or the coin moved > REF_MAX_MOVE since."""
        feed = self.refs.get("kalshi")
        if feed is None or m.get("coin") not in ("BTC", "ETH") or m.get("venue") != "out" or int(m["expiry"]) % 900:
            return None
        q = feed.quote(m["coin"], m["expiry"], now)
        if q is None or not (0 < q["bid"] < q["ask"] < 1) or q["ask"] - q["bid"] > 0.10:
            return None
        if not q.get("strike") or abs(m["strike"] / q["strike"] - 1) > 0.001:
            return None
        s0 = q.get("spot")
        if not spot or not s0 or abs(math.log(spot / s0)) > self.REF_MAX_MOVE:
            return None
        try:
            p0 = self.sniper.model.fair(m["coin"], s0, m["strike"], now, m["expiry"], m["twap_s"])
        except Exception:
            p0 = None
        if p0 is None:
            return None
        d = p - p0                                         # the model's value of the move since Kalshi's quote
        bid, ask = q["bid"] + d, q["ask"] + d
        if not (0 < bid < ask < 1):
            return None
        return {**q, "bid": bid, "ask": ask, "shift": d}

    def _vetoed(self, v: dict, coin: str, side: int, now: float) -> bool:
        """True when the outside price (Coinbase) already moved >= veto_bp against this bet in the last veto_s (Yes
        sold - side -1 - right after a jump up, or bought right after a drop), and ALSO when there is no fresh outside
        price for the coin: the variant then sits out rather than bet without its filter."""
        feed = self.refs.get("cex")
        mv = feed.move(coin, now, float(v.get("veto_s", 2.0))) if feed is not None else None
        if mv is None:
            return True
        lim = float(v["veto_bp"]) / 1e4
        return (side < 0 and mv >= lim) or (side > 0 and mv <= -lim)

    def _to_live(self, name, mid, m, sig, fair, b, now, raw) -> None:
        """Hand the selected variant's signal to the live pilot (its own cooldown, account-sized limits and caps
        apply). The operator's sniper guards still hold: price band, stop_min / max_tte_min window; and never an order
        worse than the RAW model fair (a self-calibrated fair may move, the raw-model expectation must stay >= 0)."""
        sn = self.sniper
        side, seen = sig[0], sig[1]
        tte_s = m["expiry"] - now
        band_ok = getattr(sn, "band_lo", 0.0) - 1e-9 <= seen <= getattr(sn, "band_hi", 1.0) + 1e-9
        window_ok = getattr(sn, "stop_s", 0.0) < tte_s <= getattr(sn, "max_tte_s", 1e18)
        raw_ok = (side < 0 and seen >= raw) or (side > 0 and seen <= raw)
        if not (band_ok and window_ok and raw_ok):
            return
        try:
            self.live_hook(mid, side, seen, fair, dict(m, detect=now, book_et=b.get("et"), path="lab:" + name))
        except Exception as exc:
            self.error = f"live hook ({name}): {type(exc).__name__}: {exc}"[:240]
            self._log({"ts": now, "event": "error", "where": "live_hook", "error": self.error})

    def _trend(self, coin: str, now: float) -> float | None:
        """The coin's log return over the TREND_LOOKBACK_S before now (rising vs falling market label)."""
        sn = self.sniper
        try:
            return trend_return((getattr(sn.model, "bars", None) or {}).get(coin), sn.spot_fn(coin), now,
                                self.TREND_LOOKBACK_S)
        except Exception:
            return None

    def _book(self, name, mid, m, sig, fair, b, now) -> None:
        side, seen, seen_pay, limit_pay, yes_limit = sig
        L = self.led[name]
        trend = self._trend(m["coin"], now) if mid not in L["first"] else None   # this variant's first bet labels it
        with self.lock:
            if now < L["cool"].get(mid, 0):
                return
            L["cool"][mid] = now + self.COOL_S
            L["signals"] += 1
            sz = math.ceil(self.ORDER_USD / max(limit_pay, 1e-6) - 1e-9)        # sized at the limit, as live
            cost = sz * limit_pay                       # reserved at the limit, as live; booked at what is shown later
            held = sum(x["cost"] for x in L["pos"].get(mid, {}).values()) + L["pending"].get(mid, 0.0)
            open_cost = sum(sum(x["cost"] for x in v.values()) for v in L["pos"].values()) + sum(L["pending"].values())
            is_long = (m["expiry"] - now) / 60 > self.LONG_TTE_MIN
            far = lambda k: ((self.meta.get(k) or {}).get("expiry") or 0) - now > self.LONG_TTE_MIN * 60   # noqa: E731
            long_cost = sum(sum(x["cost"] for x in v.values()) for k, v in L["pos"].items() if far(k)) + \
                sum(c for k, c in L["pending"].items() if far(k))                  # in-flight orders count too
            if held + cost > self.MARKET_CAP + 1e-9 or open_cost + cost > self.BUDGET + 1e-9 or \
                    (is_long and long_cost + cost > self.LONG_BUDGET + 1e-9):
                L["refused"] += 1
                return
            L["sent"] += 1
            L["pending"][mid] = L["pending"].get(mid, 0.0) + cost              # reserved until resolved
            self.meta.setdefault(mid, {"coin": m["coin"], "strike": m["strike"], "expiry": m["expiry"]})
        send = now + self.OVERHEAD_MS / 1000
        o = {"v": name, "mid": mid, "side": side, "seen": seen, "sz": sz, "cost": round(cost, 4),
             "fair": round(fair, 4), "detect": now, "yes_limit": yes_limit, "limit_pay": limit_pay,
             "seen_pay": seen_pay, "long": is_long, "book_et": b.get("et"),
             "trend": None if trend is None else round(trend, 5)}   # labels the market if this is its first fill
        with self.lock:
            self._checks.setdefault(mid, []).append({"o": o, "t": now, "due_t": send + self.FILL_DELAY_MS / 1000,
                                                     "due_ms": send * 1000 + self.FILL_DELAY_MS})

    def _run_checks(self, mid: int, now: float, b: dict) -> None:
        """A Yes-book update of `mid` arrived (local time now, exchange time b['et'] in ms): every waiting would-be IOC
        whose send + FILL_DELAY_MS this update is at or after gets its fill decided by THIS book. Without an exchange
        time, the local receive time stands in."""
        et = b.get("et")
        with self.lock:
            ws = self._checks.get(mid)
            if not ws:
                return
            due, keep = [], []
            for c in ws:
                late = float(et) >= c["due_ms"] if et is not None else now >= c["due_t"]
                (due if late else keep).append(c)
            if keep:
                self._checks[mid] = keep
            else:
                self._checks.pop(mid, None)
        for c in due:
            self._resolve(c["o"], b, now)

    def _expire_checks(self, now: float) -> None:
        """Would-be IOCs that got no qualifying book update within CHECK_MAX_S: misses (nothing showed the quote)."""
        gone = []
        with self.lock:
            for mid, ws in list(self._checks.items()):
                keep = [c for c in ws if now - c["t"] < self.CHECK_MAX_S]
                gone += [c for c in ws if now - c["t"] >= self.CHECK_MAX_S]
                if keep:
                    self._checks[mid] = keep
                else:
                    self._checks.pop(mid, None)
        for c in gone:
            self._resolve(c["o"], None, now)

    @staticmethod
    def _levels(o: dict, b: dict) -> list:
        """The levels an IOC at our limit could take on book b, best first, as [(pay per contract, size)]: Yes bids >=
        the Yes limit for a Yes sale (we pay 1 - bid for No), Yes asks <= it for a Yes buy. From the sniper's bid_lv /
        ask_lv (top 5); the top level alone when they are missing."""
        sell = o["side"] < 0
        lv = b.get("bid_lv" if sell else "ask_lv")
        if not lv:
            px, sz = (b.get("bid"), b.get("bid_sz")) if sell else (b.get("ask"), b.get("ask_sz"))
            lv = [] if px is None else [(px, sz or 0.0)]
        out = []
        for px, sz in lv:
            if px is None or (sell and px < o["yes_limit"] - 1e-9) or (not sell and px > o["yes_limit"] + 1e-9):
                break
            if sz and sz > 0:
                out.append((1 - px if sell else px, float(sz)))
        return out

    @staticmethod
    def _walk(levels: list, want: float) -> tuple[float, float]:
        """Sweep `levels` for up to `want` contracts: (contracts got, total paid)."""
        got = paid = 0.0
        for pay, sz in levels:
            q = min(want - got, sz)
            if q <= 0:
                break
            got, paid = got + q, paid + q * pay
        return got, paid

    def _resolve(self, o: dict, b: dict | None, now: float) -> None:
        """Decide one would-be IOC on the book `b` (the first update at least FILL_DELAY_MS after the send; None = none
        came): filled for what the levels at or through the limit show, at their prices, up to the order's size."""
        levels = self._levels(o, b) if b is not None else []
        got, cost = self._walk(levels, float(o["sz"]))
        filled = got > 0
        L = self.led[o["v"]]
        with self.lock:
            L["resolved"] += 1
            L["pending"][o["mid"]] = max(0.0, L["pending"].get(o["mid"], 0.0) - o["cost"])
            if L["pending"][o["mid"]] <= 1e-9:
                L["pending"].pop(o["mid"], None)
            if filled:
                L["filled"] += 1
                L["partial"] = L.get("partial", 0) + (got < o["sz"])
                L["first"].setdefault(o["mid"], [o["detect"], o.get("trend")])  # the first FILLED bet labels it
                leg = L["pos"].setdefault(o["mid"], {}).setdefault("no" if o["side"] < 0 else "yes",
                                                                    {"contracts": 0.0, "cost": 0.0})
                leg["contracts"] += got
                leg["cost"] += cost
                self._ladder_fill(o, levels)
        side_px = None if b is None else b.get("bid") if o["side"] < 0 else b.get("ask")
        why = None if filled else "no book update" if b is None else "gone"
        self._log({"ts": self.clock(), "event": "fill" if filled else "miss", "rule": self.FILL_RULE,
                   **{k: v for k, v in o.items() if k not in ("limit_pay", "seen_pay", "long")},
                   "got": got, "paid": None if not filled else round(cost, 4),
                   "check_et": None if b is None else b.get("et"), "check_px": side_px,
                   "check_sz": round(sum(s for _, s in levels), 4), "check_t": None if b is None else now,
                   **({"why": why} if why else {})})

    def _ladder_fill(self, o: dict, levels: list) -> None:
        """The same fill at 2x..5x the ticket (display only; caller holds the lock). Each rung asks for m x $10 at the
        same limit and sweeps the same levels the 1x check saw (bigger tickets reach worse prices); caps scale with m."""
        lad = self.ladder.get(o["v"])
        if lad is None or "limit_pay" not in o:
            return
        now = o["detect"]
        far = lambda k: ((self.meta.get(k) or {}).get("expiry") or 0) - now > self.LONG_TTE_MIN * 60   # noqa: E731
        for m, R in lad["rungs"].items():
            wanted = math.ceil(self.ORDER_USD * m / max(o["limit_pay"], 1e-6) - 1e-9)
            got, cost = self._walk(levels, wanted)
            held = sum(x["cost"] for x in R["pos"].get(o["mid"], {}).values())
            open_cost = sum(sum(x["cost"] for x in v.values()) for v in R["pos"].values())
            long_cost = sum(sum(x["cost"] for x in v.values()) for k, v in R["pos"].items() if far(k))
            if got <= 0 or held + cost > self.MARKET_CAP * m + 1e-9 or open_cost + cost > self.BUDGET * m + 1e-9 or \
                    (o.get("long") and long_cost + cost > self.LONG_BUDGET * m + 1e-9):
                R["refused"] += 1
                continue
            leg = R["pos"].setdefault(o["mid"], {}).setdefault("no" if o["side"] < 0 else "yes",
                                                               {"contracts": 0.0, "cost": 0.0})
            leg["contracts"] += got
            leg["cost"] += cost
            R["wanted"] += wanted
            R["got"] += got
            R["fills"] += 1
            R["partial"] += got < wanted

    # ------------------------------------------------------------------ settlement (its own thread)
    def poll(self, now: float) -> None:
        if self.autopilot and now - self._sel_at >= self.SELECT_EVERY_S:
            self.select(now)                               # first, before any slow settlement calls
        self._expire_checks(now)
        due = {}
        with self.lock:
            held = {k for L in self.led.values() for k in L["pos"]}
            for mid in held:
                exp = (self.meta.get(mid) or {}).get("expiry") or 0
                if exp and now > exp + GIVE_UP_S:
                    due[mid] = None                        # never settled: void (refunded), as the live pilot does
                elif exp and now >= exp + 120 and now >= self._retry.get(mid, (0, 0))[0]:
                    due[mid] = exp
            for mid, c in list(self.ckpt.items()):
                if now > c["expiry"] + GIVE_UP_S:
                    self.ckpt.pop(mid, None)
                elif now >= c["expiry"] + 120 and now >= self._retry.get(mid, (0, 0))[0]:
                    due.setdefault(mid, c["expiry"])
            # markets where every would-be order missed: nothing will settle them, so their meta goes after expiry
            need = held | {k for L in self.led.values() for k in L["pending"]} | set(self.ckpt) | set(self._checks)
            for mid, mm in list(self.meta.items()):
                if mid not in need and now > ((mm or {}).get("expiry") or 0) + 120:
                    self.meta.pop(mid, None)
                    self._retry.pop(mid, None)
        for mid, exp in due.items():
            if exp is None:
                self._settle(mid, None, now)
                continue
            tries = self._retry.get(mid, (0, 0))[1]
            self._retry[mid] = (now + min(3600.0, 30.0 * 2 ** tries), tries + 1)
            try:
                res = self.rest.post({"type": "settledOutcome", "outcome": int(mid)}) if self.rest is not None else None
                y = float(res.get("settleFraction")) if isinstance(res, dict) else math.nan
            except Exception:
                y = math.nan
            if math.isfinite(y):
                self._settle(mid, y, now)
        if now - self._saved_at >= self.SAVE_EVERY_S:
            self._saved_at = now
            self._close_clusters(now)
            self._save()

    def _settle(self, mid: int, y: float | None, now: float) -> None:
        rows = []
        with self.lock:
            c = self.ckpt.pop(mid, None)
            if c is not None and y is not None:
                self.obs.append([now, mid, c["expiry"], round(y - c["fair"], 4)])
                self.obs = self.obs[-1500:]
            exp = (self.meta.get(mid) or {}).get("expiry") or (c or {}).get("expiry")
            for name, L in self.led.items():
                legs = L["pos"].pop(mid, None)
                t0, trend = L["first"].pop(mid, None) or (None, None)
                if not legs:
                    continue
                cost = sum(x["cost"] for x in legs.values())
                ct = sum(x["contracts"] for x in legs.values())
                if y is None:                              # void: premiums back, nothing scored
                    continue
                payout = legs.get("yes", {}).get("contracts", 0.0) * y + legs.get("no", {}).get("contracts", 0.0) * (1 - y)
                pnl = payout - cost - self.FEE * ct
                L["realized"] += pnl
                L["markets"] += 1
                L["won"] += pnl > 0
                L["contracts"] += ct
                L["staked"] += cost
                L["settled"].append([now, mid, exp, round(pnl, 4), round(cost, 4), trend, t0])
                L["settled"] = [s for s in L["settled"] if s[0] >= now - self.KEEP_S]
                rows.append({"v": name, "pnl": round(pnl, 4)})
            for lad in self.ladder.values():               # the bet-size ladder settles with the same outcome
                for R in lad["rungs"].values():
                    legs = R["pos"].pop(mid, None)
                    if not legs or y is None:
                        continue
                    cost = sum(x["cost"] for x in legs.values())
                    ct = sum(x["contracts"] for x in legs.values())
                    payout = legs.get("yes", {}).get("contracts", 0.0) * y + \
                        legs.get("no", {}).get("contracts", 0.0) * (1 - y)
                    pnl = payout - cost - self.FEE * ct
                    R["realized"] += pnl
                    R["markets"] += 1
                    R["won"] += pnl > 0
                    R["staked"] += cost
                    R["contracts"] += ct
            self.meta.pop(mid, None)
            self._retry.pop(mid, None)
        if rows:
            self._log({"ts": now, "event": "settled", "mid": mid, "y": y, "rows": rows})

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.poll(self.clock())
                self._fails = 0
            except Exception as exc:
                self._fails += 1
                self.error = f"lab poll: {type(exc).__name__}: {exc}"[:240]
                if self._fails >= 3 and self.selected is not None:
                    self.selected = None                  # fail safe: no fresh evidence -> live sits out
                    self._log({"ts": self.clock(), "event": "error", "where": "poll", "error": self.error,
                               "action": "live sits out"})
            stop.wait(2.0)
        self._save()

    # ------------------------------------------------------------------ reporting / persistence
    def to_dict(self) -> dict:
        now = self.clock()
        with self.lock:
            board = []
            for name, L in self.led.items():
                v = self.variants[name]
                board.append({"name": name, **v, "signals": L["signals"], "sent": L["sent"], "refused": L["refused"],
                              "fill_rate": round(L["filled"] / L["resolved"], 3) if L["resolved"] else None,
                              "partial": L.get("partial", 0),
                              "markets": L["markets"], "won": L["won"], "net": round(L["realized"], 2),
                              "c_per_contract": round(L["realized"] / L["contracts"] * 100, 2) if L["contracts"] else None,
                              "return_on_staked": round(L["realized"] / L["staked"], 4) if L["staked"] else None,
                              "open_markets": len(L["pos"]),
                              "open_cost": round(sum(sum(x["cost"] for x in p.values()) for p in L["pos"].values()), 2),
                              "last24h": round(sum(s[3] for s in L["settled"] if s[0] > now - 86400), 2),
                              "score": _shown(_log_wealth(L["ep"]["lw"])), "outcomes": L["ep"]["n"],
                              "skipped": L["ep"].get("skipped", 0), "rising": L["ep"]["n_up"],
                              "falling": L["ep"]["n_dn"], "eligible": self.eligible(name),
                              "vetoed": L.get("vetoed", 0), "by_trend": self._side_stats(L),
                              "ladder": self._ladder_rows(name, L)})
            board.sort(key=lambda r: r["net"], reverse=True)
            n_obs = len({o[2] for o in self.obs})
            tried = len(self.tried)
        bar = self.bar()
        return {"since": self.since, "bias_c": round(self.bias() * 100, 2), "bias_markets": n_obs, "board": board,
                "error": self.error, "xyz_open_guard": self.xyz_open_guard,
                "fill_rule": {"name": self.FILL_RULE, "since": self.fill_rule_since, "delay_ms": self.FILL_DELAY_MS,
                              "rule": f"a would-be IOC fills only if the first Yes-book update at least "
                                      f"{self.FILL_DELAY_MS:g} ms (exchange time) after the send still shows our "
                                      "price, for at most the size shown there"},
                "refs": {k: f.status() for k, f in self.refs.items()},
                "season": {**self.season, "tried": tried, "bar": round(bar, 1)},
                "autopilot": {"enabled": self.autopilot, "selected": self.selected, "why": self.selection["why"],
                              "chosen_at": self.selection["ts"], "scores": self.selection["scores"], "bar": round(bar, 1),
                              "rule": (f"paper-betting score >= {bar:.0f}x (a season of up to {self.SEASON_SIZE} "
                                       f"strategies at {self.ALPHA:g}; {tried} tried so far), >= {self.MIN_CLUSTERS} "
                                       f"outcomes over >= {self.MIN_AGE_D:g} days, >= {self.MIN_SIDE} outcomes in both "
                                       f"rising and falling markets without losing in either, positive on the last "
                                       f"{self.CONFIRM_H:g} h; re-checked every {self.SELECT_EVERY_S / 60:g} min (the "
                                       "score stays honest however often it is checked)"),
                              "drives_live": self.autopilot and self.live_hook is not None}}

    def _ladder_rows(self, name: str, L: dict) -> dict | None:
        """The bet-size ladder for the board (caller holds the lock): since when, and per size x the settled markets,
        net $, cents per contract, return on money staked and the share of the wanted contracts the book had."""
        lad = self.ladder.get(name)
        if not lad:
            return None

        def row(x, realized, markets, won, staked, contracts, share):
            return {"x": x, "net": round(realized, 2), "markets": markets, "won": won,
                    "c_per_contract": round(realized / contracts * 100, 2) if contracts else None,
                    "return_on_staked": round(realized / staked, 4) if staked else None,
                    "fill_share": None if share is None else round(share, 3)}
        rows = []
        for m, R in sorted(lad["rungs"].items()):
            rows.append(row(m, R["realized"], R["markets"], R["won"], R["staked"], R["contracts"],
                            R["got"] / R["wanted"] if R["wanted"] else None))
        return {"since": lad.get("since"), "sizes": rows}

    def _log(self, rec: dict) -> None:
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _save(self) -> None:
        if self.path is None:
            return
        with self.lock:
            for n, L in self.led.items():                  # every variant's evidence outlives its retirement
                self.archive[f"{n}:{L['params']}"] = {"ep": {**L["ep"], "done": dict(L["ep"]["done"])}, "born": L["born"]}
            ladder = {n: {**lad, "rungs": {str(m): {**R, "pos": {str(k): {g: dict(x) for g, x in v.items()}
                                                                     for k, v in R["pos"].items()}}
                                           for m, R in lad["rungs"].items()}} for n, lad in self.ladder.items()}
            snap = {"since": self.since, "meta": dict(self.meta), "obs": list(self.obs), "ckpt": dict(self.ckpt),
                    "tried": list(self.tried), "season": dict(self.season), "archive": dict(self.archive),
                    "fill_rule": {"name": self.FILL_RULE, "since": self.fill_rule_since},
                    "archive_rules": dict(self.archive_rules), "ladder": ladder,
                    "led": {n: {**L, "pos": {k: {g: dict(x) for g, x in v.items()} for k, v in L["pos"].items()},
                                "pending": {}, "cool": {}, "settled": list(L["settled"]), "first": dict(L["first"]),
                                "ep": {**L["ep"], "done": dict(L["ep"]["done"])}} for n, L in self.led.items()}}
        data = json.dumps({**snap, "meta": {str(k): v for k, v in snap["meta"].items()},
                           "ckpt": {str(k): v for k, v in snap["ckpt"].items()},
                           "led": {n: {**L, "pos": {str(k): v for k, v in L["pos"].items()},
                                       "first": {str(k): v for k, v in L["first"].items()}}
                                   for n, L in snap["led"].items()}},
                          default=str)
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.since = d.get("since", self.since)
        self.meta = {int(k): v for k, v in d.get("meta", {}).items()}
        self.obs = [o for o in d.get("obs", []) if len(o) == 4]
        self.ckpt = {int(k): v for k, v in d.get("ckpt", {}).items()}
        for x in d.get("tried", []):
            if isinstance(x, str):
                self._try(x)
        self.season = d.get("season") if isinstance(d.get("season"), dict) else {}
        self.archive_rules = {k: v for k, v in (d.get("archive_rules") or {}).items() if isinstance(v, dict)}
        fr = d.get("fill_rule") if isinstance(d.get("fill_rule"), dict) else {}
        old_rule = fr.get("name") or "watch"               # state from before fill rules were named: the 5 s watch
        if old_rule != self.FILL_RULE:
            # evidence measured with other fills is not evidence about these (module docstring): kept for the record,
            # never resumed; every variant starts again from 1x with a fresh ledger and ladder. The season goes on.
            # First a copy of the whole old state that no build rewrites (an older build would drop archive_rules).
            bak = self.path.with_name(f"shadow_lab.pre-{self.FILL_RULE}.json")
            try:
                if not bak.exists():
                    bak.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                pass
            old = {k: v for k, v in (d.get("archive") or {}).items() if isinstance(v, dict)}
            for n, L in (d.get("led") or {}).items():
                if isinstance(L, dict) and L.get("params") and isinstance(L.get("ep"), dict):
                    old[f"{n}:{L['params']}"] = {"ep": L["ep"], "born": L.get("born")}
            self.archive_rules[old_rule] = {"until": self.clock(), "archive": old}
            self.fill_rule_since = self.clock()
            self.meta = {k: v for k, v in self.meta.items() if k in self.ckpt}
            self._log({"ts": self.clock(), "event": "fill_rule", "from": old_rule, "to": self.FILL_RULE,
                       "restarted": sorted(old)})
            return
        self.fill_rule_since = float(fr.get("since") or self.since)
        self._loaded_ladder = d.get("ladder") if isinstance(d.get("ladder"), dict) else {}
        self.archive = {k: v for k, v in (d.get("archive") or {}).items()
                        if isinstance(v, dict) and _ok_ep(v.get("ep"))}
        for n, L in d.get("led", {}).items():
            if not L.get("params"):
                continue
            key = f"{n}:{L['params']}"
            self._try(key)
            if _ok_ep(L.get("ep")):
                self.archive.setdefault(key, {"ep": L["ep"], "born": L.get("born") or d.get("since")})
        cutoff = self.clock() - self.KEEP_S
        for n, L in d.get("led", {}).items():
            if n in self.led and L.get("params") == self.led[n]["params"]:     # rules changed -> a fresh ledger
                pos = {int(k): v for k, v in L.get("pos", {}).items() if str(k).isdigit() and isinstance(v, dict)
                       and all(isinstance(x, dict) for x in v.values())}
                first = {int(k): v for k, v in (L.get("first") or {}).items()
                         if str(k).isdigit() and isinstance(v, list) and len(v) == 2}
                self.led[n].update({**L, "pos": pos, "pending": {}, "cool": {}, "first": first,
                                    "settled": [s for s in L.get("settled", []) if 5 <= len(s) <= 7 and s[0] >= cutoff],
                                    "ep": L["ep"] if _ok_ep(L.get("ep")) else _blank_ep(),
                                    "born": L.get("born") or d.get("since") or self.since})
        for n, L in self.led.items():                      # a returning variant resumes its old evidence
            a = self.archive.get(f"{n}:{L['params']}")
            if a is not None and L["ep"]["n"] == 0 and a["ep"].get("n", 0) > 0:
                L["ep"], L["born"] = a["ep"], a.get("born") or L["born"]
