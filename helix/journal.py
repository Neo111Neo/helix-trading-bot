"""Journal: decisions, entry signals (with the full state Jev saw), trades, equity, events and
resolved predictions go to SQLite (data/helix.db). Writes are batched once a second on a
background thread so logging never slows the loop, and old rows are pruned hourly."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    ts REAL, coin TEXT, jev_status TEXT, latency_ms REAL, model TEXT, p_up REAL, p_down REAL, p_neutral REAL,
    regime TEXT, regime_p REAL, toxic REAL, stress REAL, quality REAL, inventory REAL,
    jev_action TEXT, base_action TEXT, jev_code INTEGER, base_code INTEGER
);
CREATE TABLE IF NOT EXISTS signals (ts REAL, coin TEXT, book TEXT, state_json TEXT, answers_json TEXT,
    action_json TEXT, risk_json TEXT);
CREATE TABLE IF NOT EXISTS trades (ts REAL, book TEXT, coin TEXT, side INTEGER, qty REAL, entry REAL, exit REAL,
    opened REAL, hold_s REAL, pnl REAL, pnl_bps REAL, fees REAL, funding REAL, slip_bps REAL, reason TEXT,
    entry_reason TEXT);
CREATE TABLE IF NOT EXISTS equity (ts REAL, book TEXT, equity REAL);
CREATE TABLE IF NOT EXISTS events (ts REAL, level TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS predictions (ts REAL, coin TEXT, choice TEXT, prob REAL, outcome TEXT, correct INTEGER,
    ret_bps REAL);
CREATE INDEX IF NOT EXISTS idx_dec_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
"""

_INSERT = {
    "decisions": "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "signals": "INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
    "trades": "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "equity": "INSERT INTO equity VALUES (?,?,?)",
    "events": "INSERT INTO events VALUES (?,?,?)",
    "predictions": "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
}


class Journal:
    def __init__(self, path: str | Path, retain_hours: float = 72, flush_every: float = 1.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retain_s = retain_hours * 3600
        self.flush_every = flush_every
        self._q: queue.Queue = queue.Queue()
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._stop = threading.Event()
        self._last_prune = 0.0
        self._thread = threading.Thread(target=self._writer, name="journal", daemon=True)
        self._thread.start()

    # ---- producers (called from the engine; never block on disk) ----------------------
    def log_decision(self, row: tuple) -> None:
        self._q.put(("decisions", row))

    def log_signal(self, ts, coin, book, state, answers, action, risk) -> None:
        self._q.put(("signals", (ts, coin, book, json.dumps(state), json.dumps(answers), json.dumps(action),
                                 json.dumps(risk))))

    def log_trade(self, t: dict) -> None:
        self._q.put(("trades", (t["closed"], t["book"], t["coin"], t["side"], t["qty"], t["entry"], t["exit"],
                                t["opened"], t["hold_s"], t["pnl"], t["pnl_bps"], t["fees"], t["funding"],
                                t["slip_bps"], t["reason"], t.get("entry_reason", ""))))

    def log_equity(self, ts: float, book: str, equity: float) -> None:
        self._q.put(("equity", (ts, book, equity)))

    def log_event(self, ts: float, level: str, text: str) -> None:
        self._q.put(("events", (ts, level, text)))

    def log_prediction(self, ts: float, row: dict) -> None:
        self._q.put(("predictions", (ts, row["coin"], row["choice"], row["prob"], row["outcome"], int(row["correct"]),
                                     row["ret_bps"])))

    # ---- writer --------------------------------------------------------------------------
    def flush(self) -> int:
        batches: dict[str, list] = {}
        n = 0
        while True:
            try:
                table, row = self._q.get_nowait()
            except queue.Empty:
                break
            batches.setdefault(table, []).append(row)
            n += 1
        if batches:
            with self._db_lock:
                for table, rows in batches.items():
                    self._db.executemany(_INSERT[table], rows)
                self._db.commit()
        return n

    def _prune(self) -> None:
        cutoff = time.time() - self.retain_s
        with self._db_lock:
            for table in ("decisions", "signals", "predictions", "equity", "events"):
                self._db.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            self._db.commit()

    def _writer(self) -> None:
        while not self._stop.wait(self.flush_every):
            try:
                self.flush()
                if time.time() - self._last_prune > 3600:
                    self._last_prune = time.time()
                    self._prune()
            except sqlite3.Error:
                pass

    def query(self, sql: str, args: tuple = ()) -> list[tuple]:
        self.flush()
        with self._db_lock:
            return self._db.execute(sql, args).fetchall()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self.flush()
        with self._db_lock:
            self._db.close()
