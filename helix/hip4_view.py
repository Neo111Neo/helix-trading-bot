"""Read-only views for the HIP-4 Cortex (web/hip4.html):

  universe  every listed HIP-4 price market at once (the 3D scene): book, fair, our quotes and position, the latest
            Jev / Claude forecasts, new trades since the last poll, plus spot for every underlying and the maker's
            fills and settlements (the scene turns these into photons and shockwaves)
  ladder    one coin's strike ladders ("the curve": the market's probability across strikes next to our fair curve)
  market    one market's detail: depth, trades, our fills, 90 minutes of price vs fair

Built from the tape recorder's latest books and trades, the market maker's fair prices, quotes and history, and the
HIP-4 recorder's Jev / Claude forecasts."""

from __future__ import annotations

import time

from .hip4 import primary

FORECASTERS = ("jev", "jev_v3", "jev_v4", "claude", "model_har")


def _top(book):
    if not book:
        return None, None, 0.0, 0.0
    _, bids, asks = book
    return (bids[0][0] if bids else None, asks[0][0] if asks else None,
            bids[0][1] if bids else 0.0, asks[0][1] if asks else 0.0)


def _real(bid, ask):
    """Hyperliquid seeds empty outcome books with 0.0003 / 0.9997 placeholder orders: not a market."""
    return bid is not None and ask is not None and ask - bid <= 0.5


def _spot(tape, mm, coin):
    if tape is not None and coin in tape.last_perp:
        return tape.last_perp[coin][1]
    try:
        return mm.spot_fn(coin) if mm is not None else None
    except Exception:
        return None


def _fair(mm, m, spot, now):
    if mm is None or not spot or now >= m["expiry"] or m["expiry"] - now > 86400:
        return None
    try:
        return mm.model.fair(m["coin"], spot, m["strike"], now, m["expiry"], m["twap_s"])
    except Exception:
        return None


def forecasts(rec) -> dict[int, dict]:
    """market id -> the latest Jev v1/v3/v4, Claude and HAR-model forecasts the HIP-4 recorder asked for it."""
    if rec is None:
        return {}
    out: dict[int, dict] = {}
    with rec._lock:
        snaps = rec.snaps[-3000:]
    for s in reversed(snaps):
        if s["id"] in out:
            continue
        out[s["id"]] = {"ts": s["ts"], "checkpoint": s.get("checkpoint"),
                        **{k: (round(s[k], 4) if isinstance(s.get(k), float) else s.get(k)) for k in FORECASTERS}}
    return out


def _markets(tape, mm):
    return (tape.markets if tape is not None else {}) or (mm.markets if mm is not None else {})


def _qty(books, mid) -> float:
    return sum(b.pos.get(str(mid), {}).get("qty", 0.0) for b in books)


def _book_summary(b) -> dict:
    eq = b.equity()
    return {"name": b.name, "strategy": b.strategy, "status": b.status, "killed": b.killed,
            "pnl": round(eq - b.start_cash, 2), "risk": round(b.risk(), 2), "fills": len(b.fills), "quoting": len(b.quotes)}


def universe(tape, mm, rec=None, since: float = 0.0, books=None, pilot=None) -> dict:
    """`mm` supplies live books / fair prices / quotes; `books` are the paper books whose positions, fills and
    settlements are shown (the main strategy first)."""
    books = books if books is not None else ([mm] if mm is not None else [])
    now = time.time()
    markets = _markets(tape, mm)
    fc = forecasts(rec)
    coins: dict[str, dict] = {}
    rows = []
    for m in sorted(markets.values(), key=lambda m: (m["coin"], m["expiry"], m["strike"])):
        if m["expiry"] <= now - 60:
            continue
        c = m["coin"]
        if c not in coins:
            coins[c] = {"spot": _spot(tape, mm, c), "markets": 0, "live": 0, "volume": 0.0, "new_trades": 0}
        yb, ya, ybs, yas = _top(tape.last_books.get(f"#{m['id'] * 10}") if tape is not None else None)
        real = _real(yb, ya)
        q = mm.quotes.get(m["id"]) if mm is not None else None
        f = mm.fairs.get(m["id"]) if mm is not None else None
        trades = list(tape.recent_trades.get(f"#{m['id'] * 10}", ())) if tape is not None else []
        new = [{"p": t["p"], "s": t["s"], "sd": t["sd"]} for t in trades if t.get("t", 0) > since][-12:]
        coins[c]["markets"] += 1
        coins[c]["live"] += real
        coins[c]["new_trades"] += len(new)
        coins[c]["volume"] += sum(t["s"] for t in trades)
        rows.append({
            "id": m["id"], "coin": c, "venue": m["venue"], "strike": m["strike"], "expiry": m["expiry"],
            "rule": m["rule"], "twap_s": m["twap_s"], "primary": primary(m),
            "tte_min": round((m["expiry"] - now) / 60, 2),
            "bid": yb if real else None, "ask": ya if real else None, "depth": round(ybs + yas, 1) if real else 0.0,
            "fair": round(f[0], 4) if f else None,
            "model": round(mm.models[m["id"]], 4) if mm is not None and m["id"] in mm.models else None,
            "our_bid": q["bid"] if q and q["bid_sz"] > 0 else None, "our_ask": q["ask"] if q and q["ask_sz"] > 0 else None,
            "qty": _qty(books, m["id"]), "upnl": None,
            "trades": len(trades), "new": new, "fc": fc.get(m["id"]),
        })
    fills = sorted(({**{k: f.get(k) for k in ("ts", "id", "coin", "strike", "side", "px", "qty", "edge", "markout", "how",
                                               "exit_pc")}, "book": b.name}
                    for b in books for f in b.fills if f["ts"] > since), key=lambda f: f["ts"])[-20:]
    settled = sorted(({**{k: s.get(k) for k in ("ts", "id", "coin", "strike", "y", "qty", "pnl")}, "book": b.name}
                      for b in books for s in b.settled if s["ts"] > since), key=lambda s: s["ts"])[-10:]
    summaries = [_book_summary(b) for b in books]
    # REAL-money pilot events (orders sent, fills, settlements) since the last poll, for the Cortex
    live = [{k: o.get(k) for k in ("ts", "mid", "coin", "is_buy", "px", "sz", "status", "filled", "avg_px", "pnl",
                                   "underlying", "strike", "reason")}
            for o in (list(pilot.orders) if pilot is not None else []) if o.get("ts", 0) > since]
    return {"ts": now, "coins": coins, "markets": rows, "fills": fills, "settled": settled,
            "mm": summaries[0] if summaries else None, "books": summaries, "live": live,
            "live_mode": (("LIVE" if not pilot.blockers() else "OFF") if pilot is not None else "OFF")}


def overview(tape, mm, books=None) -> dict:
    books = books if books is not None else ([mm] if mm is not None else [])
    now = time.time()
    markets = _markets(tape, mm)
    coins = (tape.coins if tape is not None else None) or sorted({m["coin"] for m in markets.values()},
                                                                  key=lambda c: (":" in c, c)) \
        or (mm.coins if mm is not None else [])
    out = {}
    for c in coins:
        ms = [m for m in markets.values() if m["coin"] == c and m["expiry"] > now]
        live = 0
        for m in ms:
            b, a, _, _ = _top(tape.last_books.get(f"#{m['id'] * 10}") if tape is not None else None)
            live += _real(b, a)
        pos = [p for b in books for p in b.pos.values() if p["coin"] == c and abs(p["qty"]) > 1e-9]
        exps = sorted({(m["expiry"], m["venue"]) for m in ms})
        out[c] = {"spot": _spot(tape, mm, c), "markets": len(ms), "live_books": live,
                  "quoting": sum(1 for mid, q in (mm.quotes.items() if mm is not None else []) if
                                 (mm.markets.get(mid) or {}).get("coin") == c),
                  "positions": len(pos), "contracts": sum(abs(p["qty"]) for p in pos),
                  "next": [{"expiry": e, "venue": v, "tte_min": round((e - now) / 60, 1)} for e, v in exps[:6]]}
    return {"ts": now, "coins": out}


def ladder(tape, mm, coin: str, rec=None, books=None) -> dict:
    books = books if books is not None else ([mm] if mm is not None else [])
    now = time.time()
    markets = _markets(tape, mm)
    spot = _spot(tape, mm, coin)
    fc = forecasts(rec)
    groups: dict[tuple, list] = {}
    for m in markets.values():
        if m["coin"] != coin or m["expiry"] <= now:
            continue
        yb, ya, ybs, yas = _top(tape.last_books.get(f"#{m['id'] * 10}") if tape is not None else None)
        nb, na, _, _ = _top(tape.last_books.get(f"#{m['id'] * 10 + 1}") if tape is not None else None)
        q = mm.quotes.get(m["id"]) if mm is not None else None
        f = mm.fairs.get(m["id"]) if mm is not None else None
        row = {"id": m["id"], "strike": m["strike"], "rule": m["rule"], "source": m["source"], "twap_s": m["twap_s"],
               "primary": primary(m), "bid": yb, "ask": ya, "bid_sz": ybs, "ask_sz": yas, "real": _real(yb, ya),
               "no_bid": nb, "no_ask": na, "model": _fair(mm, m, spot, now),
               "fair": round(f[0], 4) if f else None,
               "our_bid": q["bid"] if q and q["bid_sz"] > 0 else None, "our_ask": q["ask"] if q and q["ask_sz"] > 0 else None,
               "qty": _qty(books, m["id"]), "fc": fc.get(m["id"]),
               "trades": len(tape.recent_trades.get(f"#{m['id'] * 10}", ())) if tape is not None else 0}
        groups.setdefault((m["expiry"], m["venue"]), []).append(row)
    out = []
    for (exp, venue), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["strike"])
        out.append({"expiry": exp, "venue": venue, "tte_min": round((exp - now) / 60, 1), "rows": rows})
    return {"ts": now, "coin": coin, "spot": spot, "groups": out}


def market(tape, mm, mid: int, rec=None, books=None) -> dict | None:
    books = books if books is not None else ([mm] if mm is not None else [])
    now = time.time()
    m = (tape.markets.get(mid) if tape is not None else None) or (mm.markets.get(mid) if mm is not None else None)
    held = next((b.pos[str(mid)] for b in books if str(mid) in b.pos), None)
    if m is None and held is not None:
        m = {"id": mid, "coin": held["coin"], "strike": held["strike"], "expiry": held["expiry"], "venue": held.get("venue"),
             "rule": "?", "source": "?", "twap_s": 0}
    if m is None:
        return None
    yes = tape.last_books.get(f"#{mid * 10}") if tape is not None else None
    no = tape.last_books.get(f"#{mid * 10 + 1}") if tape is not None else None
    spot = _spot(tape, mm, m["coin"])
    p = held
    q = mm.quotes.get(mid) if mm is not None else None
    return {
        "ts": now, "market": m, "tte_min": round((m["expiry"] - now) / 60, 2), "spot": spot,
        "primary": primary(m), "model": _fair(mm, m, spot, now), "fc": forecasts(rec).get(mid),
        "fair": round(mm.fairs[mid][0], 4) if mm is not None and mid in mm.fairs else None,
        "book": {"bids": yes[1], "asks": yes[2], "t": yes[0]} if yes else None,
        "no_book": {"bids": no[1], "asks": no[2], "t": no[0]} if no else None,
        "trades": [{k: t[k] for k in ("et", "p", "s", "sd")} for t in
                   (list(tape.recent_trades.get(f"#{mid * 10}", []))[::-1] if tape is not None else [])],
        "history": list(mm.hist.get(mid, [])) if mm is not None else [],
        "history_cols": ["t", "mkt_bid", "mkt_ask", "fair", "model", "our_bid", "our_ask", "spot"],
        "quote": {k: q[k] for k in ("bid", "ask", "bid_sz", "ask_sz")} if q else None,
        "position": {k: p[k] for k in ("qty", "cash", "fees", "fills")} if p else None,
        "fills": sorted(({**f, "book": b.name} for b in books for f in b.fills if f["id"] == mid),
                        key=lambda f: -f["ts"])[:30],
    }
