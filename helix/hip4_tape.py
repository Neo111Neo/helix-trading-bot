"""HIP-4 tape recorder: every book update and trade on the HIP-4 outcome markets of `coins` (Yes and No sides, every
venue), next to the perp best bid/ask and trades of the same coins, plus market listings and settlements. It is the
raw material for replay backtests (helix/hip4_backtest.py) and the latency / cross-strike studies.

Files: data/tape/YYYYMMDD-HH.jsonl (UTC hour, plain while it is written, gzipped once the hour is over). One JSON
record per line, `t` = local receive time (s), `et` = exchange time (ms):

  pb  perp best bid/ask     {c, et, b, bs, a, as}
  pt  perp trade            {c, et, p, s, sd}          sd "B" = aggressor bought
  ob  outcome book (top 5)  {c, et, b: [[px, sz]..], a: [[px, sz]..]}      c = "#<id*10 + side>", side 0 Yes, 1 No
  ot  outcome trade         {c, et, p, s, sd, tid}
  m   listed markets        {mk: [{id, coin, venue, strike, expiry, rule, source, twap_s}, ..]}  when the set changes
  x   settlement            {id, y}                     y = settle fraction (1 = Yes paid), null = void
  hb  heartbeat             {subs, n}                   once a minute, so gaps in the tape are visible

Read-only market data; nothing here can trade."""

from __future__ import annotations

import calendar
import gzip
import json
import math
import shutil
import socket
import threading
import time
from collections import deque
from pathlib import Path

from .hip4 import GIVE_UP_S, parse_market
from .hyperliquid import URLS, HLRest
from .wsclient import WebSocket, WebSocketError

META_EVERY_S = 60
LEVELS = 5


def hour_name(ts: float) -> str:
    return time.strftime("%Y%m%d-%H", time.gmtime(ts))


def read_tape(path: Path):
    """Yield records from one tape file (.jsonl or .jsonl.gz), skipping a torn last line."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except (EOFError, OSError):
        return


def tape_files(tape_dir: Path) -> list[Path]:
    return sorted(list(tape_dir.glob("*.jsonl.gz")) + list(tape_dir.glob("*.jsonl")), key=lambda p: p.name)


class _Every:
    """`coins: [all]`: every underlying that has HIP-4 price markets (crypto and the xyz: stock / commodity perps)."""

    def __contains__(self, coin) -> bool:
        return isinstance(coin, str) and bool(coin)


class HIP4Tape:
    def __init__(self, cfg: dict, data_dir: Path, rest: HLRest | None = None, ws_factory=WebSocket, clock=time.time):
        tc = cfg["hip4_tape"]
        self.enabled = bool(tc["enabled"])
        self.all_coins = [c.lower() for c in tc["coins"]] == ["all"]
        self.coins = [] if self.all_coins else list(tc["coins"])
        self.keep_days = float(tc["keep_days"])
        net = cfg.get("network", "mainnet")
        self.rest = rest or HLRest(URLS[net][0], timeout=10)
        self.ws_url, self.ws_factory = URLS[net][1], ws_factory
        self.clock = clock
        self.dir = Path(data_dir) / "tape"
        self.lock = threading.Lock()
        self.markets: dict[int, dict] = {}
        self.settled: dict[int, float | None] = {}
        self.retry: dict[int, list] = {}
        self._fh = None
        self._hour = None
        self._tids: deque = deque(maxlen=20000)
        self._tid_set: set = set()
        self.last_books: dict[str, tuple] = {}        # "#<id*10+side>" -> (t, bids, asks), for the visualizer
        self.recent_trades: dict[str, deque] = {}     # "#<id*10+side>" -> last 60 trades
        self.last_perp: dict[str, tuple] = {}         # underlying perp -> (t, mid): spot for every coin, xyz: too
        self.counts = {"pb": 0, "pt": 0, "ob": 0, "ot": 0, "x": 0}
        self.status, self.error = ("waiting" if self.enabled else "off"), None
        self.ws_state = {"connected": False, "subs": 0, "reconnects": 0, "error": None}
        self.started = clock()
        self._meta_at = 0.0
        self._load_settled()

    # ------------------------------------------------------------------ files
    def _load_settled(self) -> None:
        """Settlements already on tape, so a restart does not ask for them again."""
        if not self.dir.exists():
            return
        for path in tape_files(self.dir)[-48:]:
            for r in read_tape(path):
                if r.get("k") == "x":
                    self.settled[int(r["id"])] = r.get("y")

    def write(self, rec: dict) -> None:
        now = rec.setdefault("t", round(self.clock(), 3))
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self.lock:
            h = hour_name(now)
            if h != self._hour:
                self._rotate(h)
            self._fh.write(line)

    def _rotate(self, h: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._fh is not None:
            self._fh.close()
        self._hour = h
        self._fh = open(self.dir / f"{h}.jsonl", "a", encoding="utf-8", buffering=1 << 16)
        threading.Thread(target=self._compress_old, args=(h,), name="tape-gzip", daemon=True).start()

    def _compress_old(self, current: str) -> None:
        cutoff = self.clock() - self.keep_days * 86400
        for p in list(self.dir.glob("*.jsonl")):
            # only hours strictly older than the one that triggered this job AND than the one being written now: a
            # slow job from an earlier rotation must never touch a newer (open) file
            if p.stem >= current or p.stem == self._hour:
                continue
            gz = p.with_suffix(".jsonl.gz")
            try:
                with open(p, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                p.unlink()
            except OSError as exc:
                self.error = f"compress {p.name}: {exc}"[:240]
        for p in self.dir.glob("*.jsonl.gz"):
            try:
                if calendar.timegm(time.strptime(p.name[:11], "%Y%m%d-%H")) < cutoff:
                    p.unlink()
            except (ValueError, OSError):
                continue

    def flush(self) -> None:
        with self.lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self.lock:
            if self._fh is not None:
                self._fh.close()
                self._fh, self._hour = None, None

    # ------------------------------------------------------------------ WebSocket
    def handle(self, msg: dict) -> None:
        ch, d = msg.get("channel"), msg.get("data")
        if ch == "bbo" and isinstance(d, dict):
            b, a = ((d.get("bbo") or []) + [None, None])[:2]
            rec = {"k": "pb", "c": d.get("coin"), "et": d.get("time"),
                   "b": float(b["px"]) if b else None, "bs": float(b["sz"]) if b else None,
                   "a": float(a["px"]) if a else None, "as": float(a["sz"]) if a else None}
            self.write(rec)
            if rec["b"] is not None and rec["a"] is not None:
                self.last_perp[rec["c"]] = (rec["t"], (rec["b"] + rec["a"]) / 2)
            self.counts["pb"] += 1
        elif ch == "l2Book" and isinstance(d, dict):
            lv = d.get("levels") or [[], []]
            side = lambda xs: [[float(x["px"]), float(x["sz"])] for x in xs[:LEVELS]]
            rec = {"k": "ob", "c": d.get("coin"), "et": d.get("time"),
                   "b": side(lv[0] if lv else []), "a": side(lv[1] if len(lv) > 1 else [])}
            self.write(rec)
            self.last_books[rec["c"]] = (rec["t"], rec["b"], rec["a"])
            self.counts["ob"] += 1
        elif ch == "trades" and isinstance(d, list):
            for t in d:
                tid = (t.get("coin"), t.get("tid"))
                if tid in self._tid_set:
                    continue
                if len(self._tids) == self._tids.maxlen:
                    self._tid_set.discard(self._tids[0])
                self._tids.append(tid)
                self._tid_set.add(tid)
                k = "ot" if str(t.get("coin", "")).startswith("#") else "pt"
                rec = {"k": k, "c": t.get("coin"), "et": t.get("time"), "p": float(t["px"]), "s": float(t["sz"]),
                       "sd": t.get("side")}
                if k == "ot":
                    rec["tid"] = t.get("tid")
                self.write(rec)
                if k == "ot":
                    self.recent_trades.setdefault(rec["c"], deque(maxlen=60)).append(
                        {"et": rec["et"], "p": rec["p"], "s": rec["s"], "sd": rec["sd"], "tid": rec["tid"], "t": rec["t"]})
                self.counts[k] += 1

    def _subs(self) -> set:
        perps = set(self.coins) | {m["coin"] for m in self.markets.values()}
        out = {("bbo", c) for c in perps} | {("trades", c) for c in perps}
        for mid in self.markets:
            for s in (0, 1):
                out |= {("l2Book", f"#{mid * 10 + s}"), ("trades", f"#{mid * 10 + s}")}
        return out

    def _ws_loop(self, stop: threading.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            ws, subbed = None, set()
            try:
                ws = self.ws_factory(self.ws_url, timeout=10)
                ws.settimeout(1.0)
                self.ws_state.update(connected=True, error=None)
                backoff, last_ping = 1.0, time.monotonic()
                while not stop.is_set():
                    want = self._subs()
                    for typ, coin in want - subbed:
                        ws.send(json.dumps({"method": "subscribe", "subscription": {"type": typ, "coin": coin}}))
                    for typ, coin in subbed - want:
                        ws.send(json.dumps({"method": "unsubscribe", "subscription": {"type": typ, "coin": coin}}))
                    subbed = want
                    self.ws_state["subs"] = len(subbed)
                    if time.monotonic() - last_ping > 30:
                        ws.send(json.dumps({"method": "ping"}))
                        last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except socket.timeout:
                        continue
                    try:
                        self.handle(json.loads(raw))
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        self.ws_state["error"] = f"bad message: {exc}"[:200]
            except (OSError, WebSocketError, ValueError) as exc:
                self.ws_state["error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                if self.ws_state["connected"]:
                    self.ws_state["reconnects"] += 1
                self.ws_state["connected"] = False
                if ws is not None:
                    ws.close()
            stop.wait(backoff)
            backoff = min(30.0, backoff * 2)

    # ------------------------------------------------------------------ main loop
    def run(self, stop: threading.Event) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self._ws_loop, args=(stop,), name="hip4-tape-ws", daemon=True).start()
        last_hb = 0.0
        while not stop.is_set():
            try:
                now = self.clock()
                if now - self._meta_at >= META_EVERY_S:
                    self._refresh_meta(now)
                self._settle(now)
                if now - last_hb >= 60:
                    self.write({"k": "hb", "subs": self.ws_state["subs"], "n": sum(self.counts.values())})
                    last_hb = now
                self.flush()
                self.status = "recording" if self.ws_state["connected"] else "reconnecting"
            except Exception as exc:  # keep the thread alive no matter what
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:240]
            stop.wait(1.0)
        self.close()

    def _refresh_meta(self, now: float) -> None:
        meta = self.rest.post({"type": "outcomeMeta"}) or {}
        wanted = _Every() if self.all_coins else self.coins
        listed = {m["id"]: m for m in (parse_market(o, wanted) for o in meta.get("outcomes", [])) if m}
        if set(listed) != set(self.markets):
            gone = [mid for mid in self.markets if mid not in listed]
            for mid in gone:                          # delisted = settling: ask for the result
                self.retry.setdefault(mid, [now + 60, 0, self.markets[mid]["expiry"]])
            self.markets = listed
            self.write({"k": "m", "mk": sorted(listed.values(), key=lambda m: (m["expiry"], m["coin"], m["strike"]))})
        self._meta_at = now

    def _settle(self, now: float) -> None:
        calls = 0
        for mid, (nxt, tries, expiry) in list(self.retry.items()):
            if mid in self.settled:
                self.retry.pop(mid)
                continue
            if now < nxt or calls >= 2:
                continue
            if now > expiry + GIVE_UP_S:
                y = None
            else:
                calls += 1
                try:
                    res = self.rest.post({"type": "settledOutcome", "outcome": mid})
                except Exception as exc:
                    res = None
                    self.error = f"settle #{mid}: {exc}"[:240]
                if not res:
                    self.retry[mid] = [now + min(3600.0, 90.0 * 2 ** tries), tries + 1, expiry]
                    continue
                y = float(res.get("settleFraction", "nan"))
                y = y if math.isfinite(y) else None
            self.settled[mid] = y
            self.retry.pop(mid, None)
            self.write({"k": "x", "id": mid, "y": y})
            self.counts["x"] += 1

    def to_dict(self) -> dict:
        files = tape_files(self.dir) if self.dir.exists() else []
        return {"enabled": self.enabled, "status": self.status, "error": self.error, "ws": dict(self.ws_state),
                "coins": self.coins or sorted({m["coin"] for m in self.markets.values()}), "markets": len(self.markets), "counts": dict(self.counts),
                "since": self.started, "files": len(files),
                "mb_on_disk": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                "first_hour": files[0].name[:11] if files else None}
