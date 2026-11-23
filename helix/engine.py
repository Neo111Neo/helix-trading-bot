"""The fast loop.

Every decision tick (1 s by default), for every market:
  live book + trades -> features -> compact state -> Jev battery (all markets in parallel)
  -> policy -> risk engine -> paper execution on the real book -> journal
Between ticks a monitor checks stops, targets, holding time and funding every 200 ms.

Fallback ladder (from the guide):
  healthy + confident   -> trade normally
  healthy + unsure      -> half size or stand aside (policy thresholds)
  answer arrived late   -> hold: no new orders this tick
  Jev unavailable       -> code-only rules (or hold), also when the daily budget is spent
                           or after repeated errors (circuit breaker)
  hard limit breached   -> kill switch: flatten and stop new entries

Two paper books trade the same data with the same costs and limits:
  "jev"       Jev's answers through the policy engine
  "baseline"  plain momentum/flow rules only
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from .broker import PerpPaperBook
from .calibration import CalibrationTracker
from .features import compute_features, resample, sigma_bps
from .jev import JevResult, build_questions, setup_prob
from .market import DataError
from .policy import Action, flow_sign, jev_policy, reason_key, rules_policy, signal_policy
from .risk import RiskEngine
from .state import build_state, estimate_tokens

BOOKS = ("jev", "baseline")
LABEL = {"jev": "Jev", "baseline": "Baseline"}
PIPELINE = [("data", "Market data"), ("features", "Features"), ("state", "State"), ("jev", "Jev"),
            ("policy", "Policy"), ("risk", "Risk"), ("execution", "Execution"), ("monitor", "Monitor"),
            ("feedback", "Opus")]
# decision codes, used by the dashboard's decision raster
C_IDLE, C_HOLD, C_LONG, C_SHORT, C_EXIT, C_BLOCK, C_LATE, C_FALLBACK, C_PAUSED = range(9)
QUESTION_TOKENS = 170  # rough size of the six questions, for cost estimates when usage isn't returned


def _decimals(px: float | None) -> int:
    """About seven significant digits: 84110.1, 2841.42, 150.123."""
    if not px or px <= 0:
        return 4
    digits = len(str(int(px)))
    return max(1, 7 - digits)


class RealClock:
    speed = 1.0

    def now(self) -> float:
        return time.time()


class SimClock:
    """Clock for --demo. speed=1 is real time."""

    def __init__(self, start: float, speed: float = 1.0):
        self.speed = speed
        self._start = start
        self._real0 = time.time()

    def now(self) -> float:
        return self._start + (time.time() - self._real0) * self.speed


class StepClock:
    """Manually advanced clock for tests and preview recordings."""

    speed = 1.0

    def __init__(self, t: float):
        self.t = t

    def now(self) -> float:
        return self.t


class CoinState:
    def __init__(self, coin: str):
        self.coin = coin
        self.view = None  # last full view (decision tick)
        self.last = None  # freshest view or quote
        self.feats: dict | None = None
        self.state: dict | None = None
        self.state_tokens: int | None = None
        self.decision: dict | None = None
        self.depth_hist: deque = deque(maxlen=300)
        self.ribbon: deque = deque(maxlen=600)  # (ts, p_up, p_down, jev_code, base_code)
        self.error: str | None = None
        self.jev_confirm: tuple[str, int] | None = None  # (ENTER_LONG|ENTER_SHORT, consecutive decisions)
        self.last_sig_ts: int | None = None  # minute of the last Jev v2 signal fed to calibration
        self.momentum: bool | None = None    # regime guard state last seen (None = not yet known)


class Engine:
    def __init__(self, cfg: dict, feed, jev=None, journal=None, analyst=None, clock=None,
                 data_dir: str | Path = "data", demo: bool = False, persist: bool = True, signal=None):
        self.cfg = cfg
        self.feed = feed
        self.jev = jev
        # Jev v2: entries come from the once-a-minute JevSignal; the 5-second battery only vetoes and is displayed
        self.signal = signal
        self.signal_mode = signal is not None and signal.enabled
        # off = market data only (it feeds the HIP-4 bots): no new perp orders and no Jev calls; open positions still
        # run to their normal exits. Retired 2026-10-06: every perp variant tested lost 5-10 bps/trade after costs.
        self.perp_trading = bool(cfg.get("perp_trading", True))
        self.maker_entry = cfg["paper"]["entry_mode"] == "maker"
        self.entry_ttl = float(cfg["paper"]["entry_ttl_s"])
        self.maker_exit = cfg["paper"].get("exit_mode", "taker") == "maker"
        self.exit_window = float(cfg["paper"].get("exit_window_s", 120))
        self.journal = journal
        self.analyst = analyst
        self.clock = clock or RealClock()
        self.demo = demo
        self.persist_enabled = persist
        self.data_dir = Path(data_dir)
        m = cfg["market"]
        self.coins: list[str] = list(m["coins"])
        self.interval = m["decision_interval_ms"] / 1000
        self.monitor_interval = m["monitor_interval_ms"] / 1000
        self.horizon = m["horizon_s"]
        self.questions = build_questions(self.horizon)
        self.r, self.pol, self.rules = cfg["risk"], cfg["policy"], cfg["rules"]
        self.allow_short = bool(self.r["allow_short"])
        p = cfg["paper"]
        self.books = {b: self._new_book(b) for b in BOOKS}
        self.risk = RiskEngine(self.r, p["taker_fee_bps"], p["maker_fee_bps"], p["extra_slippage_bps"])
        self.calib = CalibrationTracker(self.horizon)
        self.cs = {c: CoinState(c) for c in self.coins}
        self.pool = ThreadPoolExecutor(max_workers=max(4, 2 * len(self.coins)), thread_name_prefix="jev")
        self.lock = threading.RLock()
        self.events: deque = deque(maxlen=300)
        self._gate: dict = {}
        self.block_counts = {b: Counter() for b in BOOKS}
        self.decision_times: deque = deque(maxlen=30000)
        self.loop = self.skipped = self.errors_in_row = self.decisions_total = 0
        self.paused, self.pause_reason = False, None
        self.started = self.clock.now()
        self.last_tick: float | None = None
        self.tick_ms: float | None = None
        self.last_risk: dict | None = None
        self.jev_stats = {"calls": 0, "ok": 0, "late": 0, "errors": 0, "latencies": deque(maxlen=400),
                          "status": "offline" if jev is None else "waiting", "last_error": None, "model_seen": None,
                          "tokens_day": 0, "cost_day": 0.0, "day": int(self.started // 86400),
                          "consecutive_errors": 0, "circuit_until": 0.0, "budget_hit": False,
                          "call_times": deque(maxlen=20000)}
        self.pipeline = {k: {"label": v, "status": "idle", "detail": "Starting", "ts": None} for k, v in PIPELINE}
        # Jev book bet sizing: your manual multiplier x an optional streak boost (saved with the books)
        self.sizing = {"manual": float(self.r["bet_multiplier"]), "streak": bool(self.r["streak"]["enabled"])}
        self._last_boost = 1.0
        self._snapshot: dict | None = None
        self._snapshot_mono = 0.0
        self._last_persist = 0.0
        self._last_equity_log = 0.0
        self._load_state()
        provider = "no key, fallback rules" if jev is None else getattr(jev, "provider", "jev")
        self.event("info", f"Engine ready: {', '.join(self.coins)} perps, decisions every {self.interval * 1000:.0f} ms, "
                           f"paper mode, Jev: {provider}")

    def _new_book(self, name: str) -> PerpPaperBook:
        p = self.cfg["paper"]
        return PerpPaperBook(name, float(p["starting_cash"]), float(p["taker_fee_bps"]), float(p["maker_fee_bps"]),
                             float(p["extra_slippage_bps"]))

    # ------------------------------------------------------------------ utilities
    def event(self, level: str, text: str, key: str | None = None, every: float = 0.0) -> None:
        now = self.clock.now()
        if key:
            last = self._gate.get(key)
            if last is not None and now - last < every:
                return
            self._gate[key] = now
        self.events.appendleft({"ts": now, "level": level, "text": text})
        if self.journal:
            self.journal.log_event(now, level, text)

    def _state_file(self) -> Path:
        return self.data_dir / ("demo_state.json" if self.demo else "state.json")

    def _load_state(self) -> None:
        path = self._state_file()
        if not self.persist_enabled or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != 2 or data.get("starting_cash") != self.cfg["paper"]["starting_cash"]:
                self.event("warn", "Saved state is from another version or starting cash; starting fresh")
                return
            for b in BOOKS:
                if b in data.get("books", {}):
                    self.books[b] = PerpPaperBook.from_dict(data["books"][b])
            self.paused = bool(data.get("paused"))
            self.pause_reason = data.get("pause_reason")
            self.event("info", "Restored paper books from the last session")
        except Exception as exc:
            self.event("warn", f"Could not restore saved state ({exc}); starting fresh")
            return
        sz = data.get("sizing") if isinstance(data.get("sizing"), dict) else {}
        m = sz.get("manual")
        if isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m):
            self.sizing["manual"] = min(float(self.r["max_bet_multiplier"]), max(0.1, float(m)))
        elif m is not None:
            self.event("warn", f"Ignored saved bet multiplier {m!r}; using {self.sizing['manual']:g}x from config")
        if isinstance(sz.get("streak"), bool):
            self.sizing["streak"] = sz["streak"]
        self._last_boost = self.sizing_info()["boost"]

    def _persist(self) -> None:
        if not self.persist_enabled:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {"version": 2, "starting_cash": self.cfg["paper"]["starting_cash"],
                "books": {b: bk.to_dict() for b, bk in self.books.items()},
                "paused": self.paused, "pause_reason": self.pause_reason, "sizing": self.sizing,
                "saved": self.clock.now()}
        tmp = self._state_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self._state_file())

    # ------------------------------------------------------------------ main loop
    def run(self, stop: threading.Event) -> None:
        self.feed.start()
        self.event("info", "Loop started")
        next_dec = self.clock.now()
        try:
            while not stop.is_set():
                now = self.clock.now()
                try:
                    self.monitor(now)
                    if now >= next_dec:
                        self.decide(now)
                        next_dec += self.interval
                        after = self.clock.now()
                        if after > next_dec:  # overran: skip the missed ticks instead of queueing them
                            missed = int((after - next_dec) // self.interval) + 1
                            self.skipped += missed
                            next_dec += missed * self.interval
                except Exception as exc:  # one bad tick must never stop a 24/7 loop
                    with self.lock:
                        self.errors_in_row += 1
                        self.event("error", f"Tick failed: {type(exc).__name__}: {exc}", key="tickfail", every=10)
                    traceback.print_exc()
                pause = min(self.monitor_interval, max(0.0, next_dec - self.clock.now()))
                stop.wait(max(0.005, pause / max(1e-6, self.clock.speed)))
        finally:
            self.feed.stop()
            self.pool.shutdown(wait=False, cancel_futures=True)

    def monitor(self, now: float) -> None:
        """Resting maker entries, stops, take-profits, max hold and hourly funding, on the freshest book."""
        with self.lock:
            open_coins = {c for bk in self.books.values() for c in (*bk.positions, *bk.pending)}
        if not open_coins:
            return
        quotes = {}
        for coin in open_coins:
            try:
                quotes[coin] = self.feed.quote(coin)
            except DataError:
                continue
        marks = {c: q.mid for c, q in quotes.items() if q.mid}
        with self.lock:
            for coin, q in quotes.items():
                self.cs[coin].last = q
                for name, bk in self.books.items():
                    if coin in bk.pending:
                        self._check_entry(name, bk, q, now)
                    if coin not in bk.positions:
                        continue
                    bk.apply_funding(coin, q.ctx.get("oracle"), q.ctx.get("funding"), now)
                    trade = bk.manage(q, now, float(self.r["max_hold_s"]),
                                      maker_exit=self.maker_exit and name == "jev", exit_window_s=self.exit_window)
                    if trade:
                        self._on_trade_closed(trade, now)
            if marks:
                self._mark_and_guard(marks, now)

    def _check_entry(self, name: str, bk: PerpPaperBook, q, now: float) -> None:
        kind, obj = bk.check_entry(q, now)
        if kind == "filled":
            bk.funnel["fills"] += 1
            word = "LONG" if obj.side > 0 else "SHORT"
            self.event("trade", f"{LABEL[name]} {word} {q.coin} ${obj.qty * obj.entry:,.0f} @ {self._fmt(obj.entry)} · "
                                f"maker fill (price traded through) · SL {self._fmt(obj.stop)} / TP {self._fmt(obj.target)}")
        elif kind == "expired":
            self.event("info", f"{LABEL[name]} {q.coin} resting {'bid' if obj['side'] > 0 else 'offer'} expired unfilled "
                               f"after {self.entry_ttl:.0f}s", key=f"exp-{name}-{q.coin}", every=120)

    def decide(self, now: float) -> None:
        t0 = time.perf_counter()
        views, failures = {}, []
        for coin in self.coins:
            try:
                views[coin] = self.feed.view(coin)
            except DataError as exc:
                failures.append((coin, str(exc)))
            except Exception as exc:
                failures.append((coin, f"{type(exc).__name__}: {exc}"))
        marks = {c: v.mid for c, v in views.items() if v.mid}

        prepared = []
        with self.lock:
            if failures:
                self.errors_in_row += 1
                for coin, err in failures:
                    self.cs[coin].error = err
                    self.event("warn", f"{coin} data: {err}", key=f"data-{coin}", every=30)
            else:
                if self.errors_in_row >= self.r["max_consecutive_errors"]:
                    self.event("info", "Market data recovered")
                self.errors_in_row = 0
            jb = self.books["jev"]
            for coin, v in views.items():
                cs = self.cs[coin]
                cs.view = cs.last = v
                cs.error = None
                feats = compute_features(v, now, list(cs.depth_hist))
                cs.feats = feats
                if feats is None:
                    continue
                cs.depth_hist.append(feats["depth_10bps_usd"])
                sig = sigma_bps(feats, self.horizon)
                state = build_state(feats, self._pos_info(jb, coin, v.mid, now),
                                    {"day_pnl_pct": jb.day_pnl_pct(marks), "drawdown_pct": jb.drawdown_pct(marks)},
                                    {"jev_latency_ms": self._recent_latency()},
                                    {"sigma_bps": sig, "band_bps": self.calib.threshold(sig)})
                cs.state, cs.state_tokens = state, estimate_tokens(state)
                prepared.append((coin, v, feats, state))

        results = self._ask_jev_batch(prepared, now)  # network, outside the lock

        with self.lock:
            for coin, v, feats, state in prepared:
                self._decide_one(coin, v, feats, state, results[coin], marks, now)
            if marks:
                for row in self.calib.resolve(now, marks):
                    if self.journal:
                        self.journal.log_prediction(now, row)
                self._mark_and_guard(marks, now)
            self.loop += 1
            self.last_tick = now
            self.tick_ms = (time.perf_counter() - t0) * 1000
            self._update_pipeline(now, views, failures, prepared, results)
            if now - self._last_persist >= 10:
                self._last_persist = now
                try:
                    self._persist()
                except OSError as exc:
                    self.event("warn", f"Could not save state: {exc}", key="persist", every=300)
            self._snapshot = None

    # ------------------------------------------------------------------ Jev
    def _recent_latency(self) -> float:
        lat = list(self.jev_stats["latencies"])[-10:]
        return sum(lat) / len(lat) if lat else 0.0

    def _ask_jev_batch(self, prepared: list, now: float) -> dict[str, JevResult]:
        if not prepared:
            return {}
        st, j = self.jev_stats, self.cfg["jev"]
        with self.lock:
            day = int(now // 86400)
            if day != st["day"]:
                st.update(day=day, tokens_day=0, cost_day=0.0, budget_hit=False)
            blocked = None
            if not self.perp_trading:
                blocked = JevResult("offline", error="Perp trading is off (perp_trading: false)")
            elif self.jev is None:
                blocked = JevResult("offline", error="No TYPESAFE_API_KEY set")
            elif j["daily_budget_usd"] > 0 and st["cost_day"] >= j["daily_budget_usd"]:  # 0 = no daily cap
                blocked = JevResult("budget", error=f"Daily Jev budget ${j['daily_budget_usd']:g} used")
                if not st["budget_hit"]:
                    st["budget_hit"] = True
                    self.event("risk", f"Jev daily budget ${j['daily_budget_usd']:g} reached. Fallback until 00:00 UTC.")
            elif now < st["circuit_until"]:
                blocked = JevResult("circuit", error="Paused after repeated Jev errors")
        if blocked is not None:
            with self.lock:
                if self.jev is not None:
                    st["status"] = blocked.status
            return {coin: blocked for coin, *_ in prepared}

        futures = {coin: self.pool.submit(self.jev.ask, state, self.questions) for coin, _v, _f, state in prepared}
        deadline = j["deadline_ms"] / 1000 + 0.05
        done, _ = wait(list(futures.values()), timeout=deadline)
        out = {}
        for coin, fut in futures.items():
            if fut in done:
                try:
                    out[coin] = fut.result()
                except Exception as exc:
                    out[coin] = JevResult("error", error=f"{type(exc).__name__}: {exc}")
            else:
                out[coin] = JevResult("late", latency_ms=deadline * 1000, error="No answer before the deadline")

        with self.lock:
            for coin, res in out.items():
                st["calls"] += 1
                st["call_times"].append(now)
                st["status"] = res.status
                if res.latency_ms is not None:
                    st["latencies"].append(res.latency_ms)
                if res.status in ("ok", "mock"):
                    st["ok"] += 1
                    st["consecutive_errors"] = 0
                    st["last_error"] = None
                    if res.status == "ok":
                        tokens = int(res.usage.get("input_tokens") or 0) or (self.cs[coin].state_tokens or 0) + QUESTION_TOKENS
                        st["tokens_day"] += tokens
                        st["cost_day"] += tokens * j["price_per_million_input"] / 1e6
                    if res.model and st["model_seen"] and res.model != st["model_seen"]:
                        self.event("warn", f"Jev model changed from {st['model_seen']} to {res.model}. Re-check thresholds.")
                    st["model_seen"] = res.model or st["model_seen"]
                elif res.status == "late":
                    st["late"] += 1
                    st["last_error"] = res.error
                else:
                    st["errors"] += 1
                    st["consecutive_errors"] += 1
                    st["last_error"] = res.error
                    self.event("warn", f"Jev error: {res.error}", key="jev-error", every=30)
                    if st["consecutive_errors"] >= j["circuit_errors"]:
                        st["circuit_until"] = now + j["circuit_cooldown_s"]
                        st["consecutive_errors"] = 0
                        self.event("risk", f"Jev failed {j['circuit_errors']} times in a row. Fallback for "
                                           f"{j['circuit_cooldown_s']}s, then retrying.")
        return out

    # ------------------------------------------------------------------ decisions
    @staticmethod
    def _pos_info(book: PerpPaperBook, coin: str, mid: float | None, now: float) -> dict:
        pos = book.positions.get(coin)
        if not pos or not mid:
            return {"side": 0, "upnl_bps": 0.0, "hold_s": 0}
        return {"side": pos.side, "upnl_bps": pos.side * (mid / pos.entry - 1) * 1e4, "hold_s": now - pos.opened}

    def _decide_one(self, coin, v, feats, state, res: JevResult, marks, now) -> None:
        cs = self.cs[coin]
        side_j = self._side(self.books["jev"], coin)
        side_b = self._side(self.books["baseline"], coin)
        fs = flow_sign(feats)
        sig = self.signal.fresh(coin, now) if self.signal_mode else None
        if sig is not None and sig.get("momentum_regime") != cs.momentum:    # regime monitor: say when a coin flips
            if cs.momentum is not None:
                self.event("risk" if sig.get("momentum_regime") else "info",
                           f"{coin}: {'MOMENTUM regime (moves kept going ' if sig.get('momentum_regime') else 'reversal regime back ('}"
                           f"{(sig.get('continuation') or 0) * 100:.1f}% continuation over 7 days) - Jev v2 "
                           f"{'sits this coin out' if sig.get('momentum_regime') else 'trades it again'}",
                           key=f"regime-{coin}", every=1800)
            cs.momentum = sig.get("momentum_regime")
        if not self.perp_trading:
            act_j = Action("HOLD", 0.0, ["Perp trading is off: market data only"], "off")
            act_b = Action("HOLD", 0.0, ["Perp trading is off: market data only"], "off")
        elif self.paused:
            act_j = Action("HOLD", 0.0, ["Kill switch engaged: no new orders"], "paused")
            act_b = Action("HOLD", 0.0, ["Kill switch engaged: no new orders"], "paused")
        else:
            if self.signal_mode:
                if res.usable:
                    act_j = signal_policy(sig, res.answers, side_j, self.pol, self.signal.entry_prob, self.allow_short,
                                          source="mock" if res.status == "mock" else "jev")
                elif res.status == "late":
                    act_j = Action("HOLD", 0.0, [f"Jev answer late ({res.latency_ms or 0:.0f} ms): no new orders this tick"],
                                   "late")
                else:  # never the momentum rules here: the Jev Lab showed they trade the wrong side of short moves
                    act_j = Action("HOLD" if side_j else "STAND_ASIDE", 0.0,
                                   [f"Jev unavailable ({res.status}): no new entries"], "fallback")
            elif res.usable:
                act_j = jev_policy(res.answers, fs, side_j, self.pol, self.allow_short,
                                   source="mock" if res.status == "mock" else "jev", state=state)
                # Entries must repeat for confirm_ticks consecutive decisions: Jev's per-second answers jitter.
                need = int(self.pol.get("confirm_ticks", 1))
                if act_j.kind in ("ENTER_LONG", "ENTER_SHORT"):
                    prev = cs.jev_confirm
                    cs.jev_confirm = (act_j.kind, prev[1] + 1 if prev and prev[0] == act_j.kind else 1)
                    if cs.jev_confirm[1] < need:
                        side_word = "long" if act_j.kind == "ENTER_LONG" else "short"
                        act_j = Action("STAND_ASIDE", 0.0, [f"Confirming {side_word} signal {cs.jev_confirm[1]}/{need}"] + act_j.reasons[:2],
                                       act_j.source, act_j.confidence)
                else:
                    cs.jev_confirm = None
            elif res.status == "late":
                act_j = Action("HOLD", 0.0, [f"Jev answer late ({res.latency_ms or 0:.0f} ms): no new orders this tick"],
                               "late")
            elif self.cfg["jev"]["fallback"] == "rules":
                act_j = rules_policy(feats, side_j, self.rules, self.allow_short, source="fallback",
                                     hold_s=self._pos_info(self.books["jev"], coin, v.mid, now)["hold_s"])
            else:
                act_j = Action("HOLD", 0.0, [f"Jev unavailable ({res.status}): holding"], "fallback")
            act_b = rules_policy(feats, side_b, self.rules, self.allow_short, source="rules",
                                 hold_s=self._pos_info(self.books["baseline"], coin, v.mid, now)["hold_s"])

        latency = res.latency_ms if res.usable else None
        codes, risk_out = {}, {}
        for name, act in (("jev", act_j), ("baseline", act_b)):
            codes[name], risk_out[name] = self._execute(name, coin, act, v, feats, state, res, marks, latency, now)

        p_up = p_down = p_neu = None
        a = res.answers if res.usable else None
        if a and self.signal_mode:
            # Everything downstream (ribbon, Cortex, journal, calibration) shows Jev v2's lean; the 5-second
            # battery's own direction answer is kept alongside as direction_5s.
            a = dict(a, direction_5s=a.get("direction"))
            pu = sig["p_up"] if sig else 0.5
            a["direction"] = {"type": "choice", "choice": ("up" if pu > 0.5 else "down" if pu < 0.5 else "neutral"),
                              "probabilities": {"up": pu, "down": 1 - pu, "neutral": 0.0}, "p": max(pu, 1 - pu),
                              "confidence": abs(2 * pu - 1), "source": "jev_v2" if sig else "jev_v2 (waiting)"}
        if a:
            d = a["direction"]
            p_up, p_down = d["probabilities"].get("up", 0.0), d["probabilities"].get("down", 0.0)
            p_neu = d["probabilities"].get("neutral", 0.0)
            if not self.signal_mode:
                self.calib.add(coin, now, v.mid, d["choice"], d["probabilities"], sigma_bps(feats, self.horizon))
            elif sig and sig["ts"] != cs.last_sig_ts:   # one calibration entry per new signal, not per 5 s tick
                cs.last_sig_ts = sig["ts"]
                self.calib.add(coin, now, v.mid, d["choice"], d["probabilities"], sigma_bps(feats, self.horizon))
        cs.ribbon.append((now, p_up, p_down, codes["jev"], codes["baseline"]))
        cs.decision = {
            "ts": now, "jev_status": res.status, "model": res.model,
            "latency_ms": None if res.latency_ms is None else round(res.latency_ms, 1), "error": res.error,
            "answers": a, "jev_action": act_j.to_dict(), "baseline_action": act_b.to_dict(), "risk": risk_out,
            "flow_sign": fs, "signal": sig,
        }
        self.decision_times.append(now)
        self.decisions_total += 1
        if self.journal:
            self.journal.log_decision((
                now, coin, res.status, res.latency_ms, res.model, p_up, p_down, p_neu,
                a["regime"]["choice"] if a else None, a["regime"]["p"] if a else None,
                a["toxic_flow"]["p"] if a else None, a["liquidity_stressed"]["p"] if a else None,
                # "quality" column: signed setup lean, P(long supported) - P(short supported), -1..1
                (setup_prob(a.get("long_setup")) - setup_prob(a.get("short_setup"))) if a else None,
                a["inventory_pressure"]["score"] if a else None,
                act_j.kind, act_b.kind, codes["jev"], codes["baseline"]))

    @staticmethod
    def _side(book: PerpPaperBook, coin: str) -> int:
        pos = book.positions.get(coin)
        return pos.side if pos else 0

    def _execute(self, name, coin, act: Action, v, feats, state, res, marks, latency, now):
        book = self.books[name]
        book.funnel["decisions"] += 1
        pos = book.positions.get(coin)
        maker = self.maker_entry and name == "jev"
        resting = book.pending.get(coin)
        if resting is not None:
            want = {"ENTER_LONG": 1, "ENTER_SHORT": -1}.get(act.kind)
            if want == resting["side"]:
                return (C_LONG if want > 0 else C_SHORT), None       # the signal still wants it: keep resting
            if act.kind in ("STAND_ASIDE", "EXIT", "ENTER_LONG", "ENTER_SHORT") or act.source == "paused":
                book.cancel_entry(coin)
                self.event("info", f"{LABEL[name]} {coin} resting {'bid' if resting['side'] > 0 else 'offer'} cancelled: "
                                   f"{(act.reasons or [act.kind])[0][:80]}", key=f"cxl-{name}-{coin}", every=60)
        if act.source == "paused":
            return C_PAUSED, None
        if act.kind == "EXIT":
            if pos:
                t = book.close_market(v, now, "Signal exit: " + "; ".join(act.reasons))
                if t:
                    book.funnel["exits"] += 1
                    self._on_trade_closed(t, now)
                return C_EXIT, None
            return C_IDLE, None
        if act.kind in ("ENTER_LONG", "ENTER_SHORT"):
            if pos:
                return C_HOLD, None
            book.funnel["signals"] += 1
            side = 1 if act.kind == "ENTER_LONG" else -1
            rd = self.risk.evaluate_entry(book=book, view=v, feats=feats, side=side, size_mult=act.size, marks=marks,
                                          now=now, horizon_s=self.horizon,
                                          decision_latency_ms=latency if name == "jev" else None,
                                          errors_in_row=self.errors_in_row, engine_paused=self.paused,
                                          maker_entry=maker,
                                          bet_mult=self.sizing_info()["effective"] if name == "jev" else 1.0,
                                          sigma_floor_bps=act.meta.get("sigma_bps"))
            clock_start = float(act.meta.get("signal_ts") or 0.0)   # Jev v2: hold 15 min from the signal minute
            rdd = rd.to_dict()
            self.last_risk = {"ts": now, "coin": coin, "book": name, **rdd}
            if self.journal:
                self.journal.log_signal(now, coin, name, state if name == "jev" else None,
                                        res.answers if name == "jev" and res.usable else None, act.to_dict(), rdd)
            word = "long" if side > 0 else "short"
            if not rd.approved:
                key = rd.reason.split(":")[0]
                self.block_counts[name][f"Risk: {key}"] += 1
                self.event("risk", f"{LABEL[name]} {coin} {word} blocked. {rd.reason}",
                           key=f"blk-{name}-{coin}-{key}", every=60)
                return C_BLOCK, rdd
            book.funnel["risk_pass"] += 1
            if maker:
                o = book.place_entry(v, side, rd.qty, rd.stop_bps, rd.target_bps, now, self.entry_ttl, "; ".join(act.reasons),
                                     clock_start=clock_start)
                if o is None:
                    self.block_counts[name]["Risk: Book empty at order time"] += 1
                    return C_BLOCK, rdd
                self.event("trade", f"{LABEL[name]} {word.upper()} {coin}: resting {'bid' if side > 0 else 'offer'} "
                                    f"${o['qty'] * o['price']:,.0f} @ {self._fmt(o['price'])} (post-only maker, "
                                    f"{self.entry_ttl:.0f}s) · SL {rd.stop_bps:.1f} / TP {rd.target_bps:.1f} bps",
                           key=f"rest-{name}-{coin}", every=30)
                return (C_LONG if side > 0 else C_SHORT), rdd
            p = book.open(v, side, rd.qty, rd.stop, rd.target, now, "; ".join(act.reasons), clock_start=clock_start)
            if p is None:
                self.block_counts[name]["Risk: Book too thin at fill"] += 1
                return C_BLOCK, rdd
            book.funnel["fills"] += 1
            self.event("trade", f"{LABEL[name]} {word.upper()} {coin} ${p.qty * p.entry:,.0f} @ {self._fmt(p.entry)} · "
                                f"SL {rd.stop_bps:.1f} / TP {rd.target_bps:.1f} bps · slip {p.entry_slip_bps:.2f} bps")
            return (C_LONG if side > 0 else C_SHORT), rdd
        if act.kind == "STAND_ASIDE":
            if act.reasons:
                self.block_counts[name][f"Policy: {reason_key(act.reasons[0])}"] += 1
            return (C_FALLBACK if act.source == "fallback" else C_IDLE), None
        if act.source == "late":
            return C_LATE, None
        if pos:
            return C_HOLD, None
        return (C_FALLBACK if act.source == "fallback" else C_IDLE), None

    @staticmethod
    def _fmt(p: float | None) -> str:
        if p is None:
            return "-"
        return f"{p:,.4f}" if p < 1000 else f"{p:,.1f}"

    def _on_trade_closed(self, t: dict, now: float) -> None:
        sign = "+" if t["pnl"] >= 0 else "-"
        self.event("trade", f"{LABEL[t['book']]} closed {t['coin']} {sign}${abs(t['pnl']):,.2f} ({t['pnl_bps']:+.1f} bps, "
                            f"{t['hold_s']:.0f}s) · {t['reason'][:70]}")
        bk = self.books[t["book"]]
        if t["reason"] == "Stop loss":
            bk.cooldown_until[t["coin"]] = now + float(self.r["cooldown_s"])
        reentry = float(self.r.get("reentry_cooldown_s", 0))
        if reentry > 0:  # any exit: no immediate re-entry in the same coin (stops fee-burning flip-flops)
            bk.cooldown_until[t["coin"]] = max(bk.cooldown_until.get(t["coin"], 0.0), now + reentry)
        if self.journal:
            self.journal.log_trade(t)
        if t["book"] == "jev" and self.sizing["streak"]:
            s = self.sizing_info()
            if s["boost"] != self._last_boost:
                why = (f"{s['wins_in_row']} wins in a row" if s["boost"] > 1 else
                       "a loss reset the streak" if s["wins_in_row"] == 0 else
                       f"gate closed: last {s['gate_trades']} trades average {s['gate_mean_bps']} bps")
                self.event("risk", f"Streak boost {self._last_boost:g}x -> {s['boost']:g}x ({why}); "
                                   f"Jev bets now {s['effective']:g}x")
                self._last_boost = s["boost"]

    def _views_for_flatten(self) -> dict:
        return {c: cs.last for c, cs in self.cs.items() if cs.last is not None}

    def _mark_and_guard(self, marks: dict, now: float) -> None:
        log_equity = now - self._last_equity_log >= 30
        for name, bk in self.books.items():
            eq = bk.mark(marks, now)
            if self.journal and log_equity:
                self.journal.log_equity(now, name, eq)
            kind, reason = self.risk.guard(bk, marks)
            if kind == "kill" and not bk.killed:
                bk.killed, bk.kill_reason = True, reason
                closed = bk.flatten(self._views_for_flatten(), now, "Kill switch") if self.r["flatten_on_kill"] else []
                for t in closed:
                    self._on_trade_closed(t, now)
                self.event("risk", f"KILL SWITCH ({LABEL[name]}): {reason}. Flattened {len(closed)} position(s). "
                                   "Resume manually after review.")
            elif kind == "daily" and not bk.daily_stop:
                bk.daily_stop = True
                closed = bk.flatten(self._views_for_flatten(), now, "Daily loss limit") if self.r["flatten_on_kill"] else []
                for t in closed:
                    self._on_trade_closed(t, now)
                self.event("risk", f"Daily stop ({LABEL[name]}): {reason}. No new entries until 00:00 UTC.")
        if log_equity:
            self._last_equity_log = now

    # ------------------------------------------------------------------ controls
    def kill(self, reason: str = "Manual kill switch") -> None:
        with self.lock:
            self.paused, self.pause_reason = True, reason
            now = self.clock.now()
            n = 0
            if self.r["flatten_on_kill"]:
                for bk in self.books.values():
                    for t in bk.flatten(self._views_for_flatten(), now, reason):
                        self._on_trade_closed(t, now)
                        n += 1
            self.event("risk", f"{reason}: trading paused, {n} position(s) closed")
            self._persist()
            self._snapshot = None

    def resume(self) -> None:
        with self.lock:
            self.paused, self.pause_reason = False, None
            marks = {c: cs.last.mid for c, cs in self.cs.items() if cs.last is not None and cs.last.mid}
            for bk in self.books.values():
                bk.killed, bk.kill_reason = False, None
                bk.peak_equity = max(bk.equity(marks), 1e-9)
            self.event("info", "Resumed by operator. Drawdown peak reset to current equity.")
            self._persist()
            self._snapshot = None

    # ------------------------------------------------------------------ bet sizing (Jev book)
    def sizing_info(self) -> dict:
        """manual x streak boost, capped. The boost only applies after `after_wins` wins in a row AND when the last
        `gate_trades` trades made money IN DOLLARS (net of fees and funding; a boosted loss can't hide behind many
        small wins), so a lucky run inside a losing record isn't doubled up. Any losing trade resets it.
        gate_trades 0 = no gate."""
        st, trades = self.r["streak"], self.books["jev"].trades
        wins = 0
        for t in reversed(trades):
            if t["pnl"] > 0:
                wins += 1
            else:
                break
        n_gate = int(st["gate_trades"])
        recent = trades[-n_gate:] if n_gate else []
        gate_pnl = sum(t["pnl"] for t in recent)
        gate_notional = sum(t["qty"] * t["entry"] for t in recent if "qty" in t and "entry" in t)
        gate_ok = n_gate == 0 or (len(trades) >= n_gate and gate_pnl > 0)
        boost = 1.0
        if self.sizing["streak"] and wins >= st["after_wins"] and gate_ok:
            boost = min(float(st["max"]), 1.0 + st["step"] * (wins - st["after_wins"] + 1))
        cap = float(self.r["max_bet_multiplier"])
        eff = min(cap, self.sizing["manual"] * boost)
        return {"manual": self.sizing["manual"], "streak_enabled": self.sizing["streak"], "wins_in_row": wins,
                "boost": round(boost, 2), "effective": round(eff, 2), "max": cap,
                "capped": self.sizing["manual"] * boost > cap + 1e-9,
                "gate_ok": gate_ok, "gate_trades": n_gate, "trades": len(trades),
                "gate_pnl": round(gate_pnl, 2) if recent else None,
                "gate_mean_bps": round(gate_pnl / gate_notional * 1e4, 2) if gate_notional else None,
                "after_wins": st["after_wins"], "step": st["step"], "streak_max": st["max"]}

    def set_sizing(self, multiplier: float | None = None, streak: bool | None = None) -> dict:
        with self.lock:
            before = self.sizing_info()["effective"]
            if multiplier is not None:
                m = float(multiplier)
                if not (math.isfinite(m) and 0.1 <= m <= float(self.r["max_bet_multiplier"])):
                    raise ValueError(f"multiplier must be between 0.1 and {self.r['max_bet_multiplier']:g}")
                self.sizing["manual"] = m
                self.event("risk", f"Jev bet multiplier set to {m:g}x by operator")
            if streak is not None:
                if not isinstance(streak, bool):
                    raise ValueError("streak must be true or false")
                self.sizing["streak"] = streak
                self.event("risk", f"Streak boost {'ON' if streak else 'OFF'} by operator")
            info = self.sizing_info()
            self._last_boost = info["boost"]
            jb = self.books["jev"]
            if info["effective"] < before and jb.pending:   # resting entries were sized at the old, bigger bet
                for coin in list(jb.pending):
                    jb.cancel_entry(coin)
                self.event("risk", f"Cancelled resting Jev entries sized at {before:g}x (now {info['effective']:g}x)")
            try:
                self._persist()
            except OSError as exc:   # the change is live; only the save failed
                self.event("warn", f"Could not save state: {exc}", key="persist", every=300)
            self._snapshot = None
            return info

    def reset_paper(self) -> None:
        with self.lock:
            self.books = {b: self._new_book(b) for b in BOOKS}
            self.block_counts = {b: Counter() for b in BOOKS}
            self._last_boost = 1.0
            self.paused, self.pause_reason = False, None
            self.event("info", "Paper books reset to starting cash")
            self._persist()
            self._snapshot = None

    # ------------------------------------------------------------------ dashboard
    def _update_pipeline(self, now, views, failures, prepared, results) -> None:
        P = self.pipeline

        def setp(key, status, detail):
            P[key].update(status=status, detail=detail, ts=now)

        h = self.feed.health()
        n_ok = len(views)
        src = "sim" if h.get("venue") == "simulated" else ("WS" if h.get("connected") else "REST backup")
        setp("data", "fail" if n_ok == 0 else ("warn" if failures or src == "REST backup" else "ok"),
             f"{n_ok}/{len(self.coins)} books · {src}" + (f" · {h['msgs_per_s']:.0f} msg/s" if h.get("msgs_per_s") else ""))
        setp("features", "ok" if prepared else "idle",
             f"{len(prepared)}/{len(self.coins)} ready" if prepared else "Warming up (2 min of mids)")
        toks = [cs.state_tokens for cs in self.cs.values() if cs.state_tokens]
        setp("state", "ok" if toks else "idle", f"~{max(toks)} tokens" if toks else "Waiting")
        st = self.jev_stats
        lat = sorted(list(st["latencies"])[-200:])
        p50 = f"p50 {lat[len(lat) // 2]:.0f}ms" if lat else "no calls"
        status = st["status"]
        if self.jev is None:
            setp("jev", "warn", "No key · fallback rules")
        elif status == "mock":
            setp("jev", "warn", f"MOCK (demo) · {p50}")
        elif status == "ok":
            setp("jev", "ok", f"{p50} · {st['model_seen'] or self.cfg['jev']['model']}")
        elif status == "late":
            setp("jev", "warn", f"Late answers · {p50}")
        elif status == "budget":
            setp("jev", "warn", "Daily budget used · fallback")
        elif status == "circuit":
            setp("jev", "fail", "Circuit open · fallback")
        elif status == "error":
            setp("jev", "fail", (st["last_error"] or "error")[:48])
        else:
            setp("jev", "idle", "Waiting")
        dpm = sum(1 for t in self.decision_times if now - t <= 60)
        setp("policy", "ok" if prepared else "idle", f"{dpm}/min decisions")
        killed = self.paused or any(bk.killed for bk in self.books.values())
        daily = any(bk.daily_stop for bk in self.books.values())
        if killed:
            setp("risk", "fail", "Kill switch engaged")
        elif daily or self.errors_in_row >= self.r["max_consecutive_errors"]:
            setp("risk", "warn", "Entries blocked")
        else:
            blocks = sum(sum(c.values()) for c in self.block_counts.values() if c)
            setp("risk", "ok", f"Armed · {blocks:,} filtered")
        n_pos = sum(len(bk.positions) for bk in self.books.values())
        fills = sum(bk.funnel["fills"] for bk in self.books.values())
        setp("execution", "ok", f"{fills} fills · {n_pos} open")
        setp("monitor", "ok" if self.errors_in_row == 0 else "warn",
             f"{self.tick_ms or 0:.0f}ms tick · {self.skipped} skipped")
        if self.analyst is None or not self.analyst.available:
            setp("feedback", "idle", "Add ANTHROPIC_API_KEY")
        else:
            a = self.analyst.status
            setp("feedback", {"running": "warn", "error": "fail"}.get(a, "ok"),
                 {"idle": "Ready", "running": "Reviewing…", "done": "Review ready", "error": "Failed"}.get(a, a))

    def snapshot(self, fresh: bool = False) -> dict:
        mono = time.monotonic()
        with self.lock:
            if fresh or self._snapshot is None or mono - self._snapshot_mono > 0.5:
                self._snapshot = self._build_snapshot()
                self._snapshot_mono = mono
            snap = dict(self._snapshot)
        if self.analyst is not None:
            a = self.analyst.to_dict()
            snap["analyst"] = a
            if self.analyst.available:
                fb = {"key": "feedback", "label": "Opus", "ts": a["ts"],
                      "status": {"running": "warn", "error": "fail"}.get(a["status"], "ok"),
                      "detail": {"idle": "Ready", "running": "Reviewing…", "done": "Review ready",
                                 "error": "Failed"}.get(a["status"], a["status"])}
                snap["pipeline"] = [fb if p["key"] == "feedback" else p for p in snap["pipeline"]]
        return snap

    def _build_snapshot(self) -> dict:
        now = self.clock.now()
        marks = {c: cs.last.mid for c, cs in self.cs.items() if cs.last is not None and cs.last.mid}
        books = {}
        for name, bk in self.books.items():
            stats = bk.stats(marks, now)
            positions = []
            for coin, pos in bk.positions.items():
                px = marks.get(coin, pos.entry)
                positions.append({"coin": coin, "side": pos.side, "qty": pos.qty, "entry": pos.entry, "price": px,
                                  "stop": pos.stop, "target": pos.target, "opened": pos.opened,
                                  "hold_s": now - pos.opened, "age_s": pos.age(now),   # age: since the signal (the exit clock)
                                  "notional": pos.qty * px, "upnl": pos.upnl(px),
                                  "upnl_bps": pos.side * (px / pos.entry - 1) * 1e4, "funding": pos.funding_paid})
            curve = bk.equity_curve[-4000:]
            if len(curve) > 360:
                step = len(curve) // 360 + 1
                curve = curve[::step] + [curve[-1]]
            pending = [{"coin": o["coin"], "side": o["side"], "price": o["price"], "notional": o["qty"] * o["price"],
                        "age_s": now - o["placed"], "ttl_s": max(0.0, o["expires"] - now)} for o in bk.pending.values()]
            books[name] = {**stats, "positions": positions, "equity_curve": curve, "pending": pending,
                           "maker": dict(bk.maker_stats), "blocks": self.block_counts[name].most_common(7)}

        markets = {}
        for coin, cs in self.cs.items():
            v = cs.view
            last = cs.last or v
            if last is None:
                markets[coin] = {"coin": coin, "ready": False, "error": cs.error}
                continue
            ctx = last.ctx or {}
            grid = resample(v.mids, now, 300) if v else []
            mid = last.mid
            prev = ctx.get("prev_day_px")
            markers = []
            for name, bk in self.books.items():
                for t in bk.trades[-80:]:
                    if t["coin"] == coin and now - t["closed"] <= 330:
                        markers.append({"ts": t["opened"], "px": t["entry"], "kind": "entry", "book": name, "side": t["side"]})
                        markers.append({"ts": t["closed"], "px": t["exit"], "kind": "exit", "book": name,
                                        "win": t["pnl"] > 0})
                pos = bk.positions.get(coin)
                if pos and now - pos.opened <= 330:
                    markers.append({"ts": pos.opened, "px": pos.entry, "kind": "entry", "book": name, "side": pos.side})
            f = cs.feats or {}
            ribbon = [[round(now - t, 1), None if pu is None else round(pu, 2), None if pd is None else round(pd, 2), cj, cb]
                      for t, pu, pd, cj, cb in list(cs.ribbon)[-300:]]  # [seconds ago, P(up), P(down), codes], 5 min
            nd = _decimals(mid)
            tape = [[round(now - t.ts, 1), round(t.px, nd + 1), round(t.sz, 5), t.side] for t in (v.trades[-24:] if v else [])]
            markets[coin] = {
                "coin": coin, "ready": cs.feats is not None, "mid": mid, "mark": ctx.get("mark"), "oracle": ctx.get("oracle"),
                "funding_hr": ctx.get("funding"), "open_interest": ctx.get("open_interest"),
                "day_volume": ctx.get("day_volume"), "change_24h_pct": (mid / prev - 1) * 100 if prev and mid else None,
                "spread_bps": last.spread_bps, "data_age_s": max(0.0, now - last.received), "source": last.source,
                "mids": {"t0": now - max(0, len(grid) - 1), "dt": 1.0, "v": [round(x, nd) for x in grid]},
                "ribbon": ribbon, "tape": tape,
                "book": {"bids": [[round(p, nd + 1), round(q, 5)] for p, q in last.bids[:20]],
                         "asks": [[round(p, nd + 1), round(q, 5)] for p, q in last.asks[:20]]},
                "features": {k: f.get(k) for k in ("imbalance_5", "imbalance_20", "flow_15s", "flow_60s", "vol_1s_bps",
                                                   "depth_10bps_usd", "depth_ratio", "trade_rate_15s", "ret_60s",
                                                   "ret_300s", "funding_hr_bps", "premium_bps", "trend_1m_bps")} if f else None,
                "state": cs.state, "state_tokens": cs.state_tokens, "decision": cs.decision, "markers": markers,
                "error": cs.error,
                "signal": self.signal.latest.get(coin) if self.signal is not None else None,
            }

        st = self.jev_stats
        lat = sorted(list(st["latencies"])[-300:])
        edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 10 ** 9]
        hist = [[edges[i], edges[i + 1] if edges[i + 1] < 10 ** 9 else None,
                 sum(1 for x in lat if edges[i] <= x < edges[i + 1])] for i in range(len(edges) - 1)]
        dpm = sum(1 for t in self.decision_times if now - t <= 60)
        cpm = sum(1 for t in st["call_times"] if now - t <= 60)
        fills_min = sum(bk.entries_last(now) for bk in self.books.values())
        limits = []
        for name in BOOKS:
            bk = self.books[name]
            if not marks:
                break
            day, dd, lev = bk.day_pnl_pct(marks), bk.drawdown_pct(marks), bk.leverage(marks)
            n_e = bk.entries_last(now)
            r = self.r
            bm = self.sizing_info()["effective"] if name == "jev" else 1.0
            lev_cap = min(r["max_total_leverage"] * bm, r["leverage_ceiling_total"])   # same cap the risk engine uses
            limits.append({"book": name, "items": [
                {"name": "Daily loss", "value": day, "limit": -r["max_daily_loss_pct"] * 100, "unit": "%",
                 # limit 0 = daily stop off: show the meter empty instead of dividing by zero
                 "use": max(0.0, -day / (r["max_daily_loss_pct"] * 100)) if r["max_daily_loss_pct"] > 0 else 0.0},
                {"name": "Drawdown", "value": dd, "limit": r["max_drawdown_pct"] * 100, "unit": "%",
                 "use": dd / (r["max_drawdown_pct"] * 100)},
                {"name": "Leverage", "value": lev, "limit": lev_cap, "unit": "x", "use": lev / lev_cap},
                {"name": "Positions", "value": len(bk.positions), "limit": r["max_open_positions"], "unit": "",
                 "use": len(bk.positions) / r["max_open_positions"]},
                {"name": "Entries/min", "value": n_e, "limit": r["max_entries_per_min"], "unit": "",
                 "use": n_e / r["max_entries_per_min"]},
            ]})
        trades = sorted((t for bk in self.books.values() for t in bk.trades[-60:]), key=lambda t: t["closed"],
                        reverse=True)[:40]
        jbs, bbs = books["jev"], books["baseline"]
        j = self.cfg["jev"]
        return {
            "ts": now, "version": 2, "mode": "paper", "demo": self.demo, "venue": "hyperliquid",
            "feed": self.feed.health(),
            "engine": {"status": "paused" if self.paused else ("degraded" if self.errors_in_row else "running"),
                       "pause_reason": self.pause_reason, "loop": self.loop, "interval_ms": self.interval * 1000,
                       "tick_ms": self.tick_ms, "skipped": self.skipped, "errors_in_row": self.errors_in_row,
                       "started": self.started, "decisions_total": self.decisions_total,
                       "decisions_per_min": dpm, "jev_calls_per_min": cpm, "entries_per_min": fills_min,
                       "speed": getattr(self.clock, "speed", 1.0), "refresh_ms": self.cfg["server"]["refresh_ms"]},
            "jev": {"provider": None if self.jev is None else getattr(self.jev, "provider", "typesafe"),
                    "status": st["status"], "model_pinned": j["model"], "model_seen": st["model_seen"],
                    "calls": st["calls"], "ok": st["ok"], "late": st["late"], "errors": st["errors"],
                    "last_error": st["last_error"], "p50_ms": lat[len(lat) // 2] if lat else None,
                    "p95_ms": lat[max(0, int(len(lat) * 0.95) - 1)] if len(lat) >= 2 else None,
                    "hist": hist, "deadline_ms": j["deadline_ms"], "fallback": j["fallback"],
                    "cost_day": round(st["cost_day"], 4), "tokens_day": st["tokens_day"],
                    "budget_usd": j["daily_budget_usd"], "budget_hit": st["budget_hit"],
                    "circuit_s": max(0.0, st["circuit_until"] - now),
                    "questions": {k: {"type": q["type"], "instructions": q["instructions"],
                                      "options": list(q["criteria"].keys()) if isinstance(q.get("criteria"), dict)
                                      else q.get("criteria")} for k, q in self.questions.items()}},
            "config": {"coins": self.coins, "interval_ms": self.interval * 1000, "horizon_s": self.horizon,
                       "taker_fee_bps": self.cfg["paper"]["taker_fee_bps"], "maker_fee_bps": self.cfg["paper"]["maker_fee_bps"],
                       "round_trip_bps": self.risk.round_trip_bps(0.5, self.maker_entry),
                       "starting_cash": self.cfg["paper"]["starting_cash"], "entry_mode": self.cfg["paper"]["entry_mode"],
                       "exit_mode": self.cfg["paper"].get("exit_mode", "taker"),
                       "risk": self.r, "policy": self.pol, "rules": self.rules, "network": self.cfg["network"]},
            "signal": ({k: v for k, v in self.signal.to_dict().items() if k != "latest"} | {"mode": self.signal_mode})
            if self.signal is not None else {"enabled": False, "mode": False},
            "sizing": self.sizing_info(),
            "books": books,
            "edge": {"pnl": round(jbs["pnl"] - bbs["pnl"], 2), "pnl_pct": round(jbs["pnl_pct"] - bbs["pnl_pct"], 3)},
            "markets": markets,
            "limits": limits,
            "last_risk": self.last_risk,
            "trades": trades,
            "events": list(self.events)[:60],
            "calibration": self.calib.report(),
            "pipeline": [{"key": k, **v} for k, v in self.pipeline.items()],
            "analyst": self.analyst.to_dict() if self.analyst else {"status": "unavailable", "report": None,
                                                                   "error": None, "ts": None, "model": None},
        }

    def analyst_summary(self) -> dict:
        snap = self.snapshot(fresh=True)
        return {
            "venue": "Hyperliquid perps (paper)", "config": snap["config"],
            "jev": {k: snap["jev"][k] for k in ("provider", "model_pinned", "model_seen", "calls", "ok", "late",
                                                "errors", "p50_ms", "p95_ms", "cost_day")},
            "throughput": {k: snap["engine"][k] for k in ("decisions_per_min", "jev_calls_per_min", "decisions_total",
                                                          "skipped")},
            "books": {n: {k: v for k, v in b.items() if k not in ("equity_curve", "positions")}
                      for n, b in snap["books"].items()},
            "calibration": snap["calibration"],
            "jev_bet_sizing": snap["sizing"],   # Jev's $ results are scaled by this; the baseline always bets 1x
            "recent_trades": [{k: t[k] for k in ("book", "coin", "side", "pnl", "pnl_bps", "fees", "funding", "slip_bps",
                                                 "hold_s", "reason", "entry_reason")} for t in snap["trades"][:40]],
            "run_hours": round((self.clock.now() - self.started) / 3600, 2), "demo": self.demo,
        }
