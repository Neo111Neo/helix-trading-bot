"""Dashboard + JSON API (Starlette). The engine runs in a background thread; the API only
reads a cached snapshot, so the dashboard never slows the trading loop down."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PAGE_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
             '<meta name="color-scheme" content="dark"></head><body>')


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class TokenGate(BaseHTTPMiddleware):
    """If HELIX_TOKEN is set, every /api call must carry it (header or ?token=)."""

    def __init__(self, app, token: str | None):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/api/"):
            # Any website open in the same browser could otherwise POST here (kill, reset, bet size). Browsers always
            # send Origin on cross-site POSTs, so only same-origin pages (and tools without an Origin, like curl) pass.
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1].rstrip("/") != request.headers.get("host", ""):
                return SafeJSON({"error": "cross-site request refused"}, status_code=403)
        if self.token and request.url.path.startswith("/api/"):
            supplied = request.headers.get("x-helix-token") or request.query_params.get("token")
            if supplied != self.token:
                return SafeJSON({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def create_app(engine, run_loop: bool = True, token: str | None = None, forecaster=None, hourly=None,
               signal=None, hip4=None, weekly=None, hip4_mm=None, tape=None, shadow=None, sniper=None) -> Starlette:
    stop = threading.Event()
    hip4_books = [b for b in (hip4_mm, sniper) if b is not None]     # paper HIP-4 books: the kill switch covers all

    @contextlib.asynccontextmanager
    async def lifespan(app):
        thread = None
        if run_loop:
            thread = threading.Thread(target=engine.run, args=(stop,), name="helix-engine", daemon=True)
            thread.start()
            if forecaster is not None:
                threading.Thread(target=forecaster.run, args=(stop,), name="claude-forecast", daemon=True).start()
            if hourly is not None:
                threading.Thread(target=hourly.run, args=(stop,), name="hourly-test", daemon=True).start()
            if signal is not None:
                threading.Thread(target=signal.run, args=(stop,), name="jev-signal", daemon=True).start()
            if hip4 is not None:
                threading.Thread(target=hip4.run, args=(stop,), name="hip4-recorder", daemon=True).start()
            if weekly is not None:
                threading.Thread(target=weekly.run, args=(stop,), name="lab-weekly", daemon=True).start()
            for b in hip4_books:
                threading.Thread(target=b.run, args=(stop,), name=b.name, daemon=True).start()
            if tape is not None:
                threading.Thread(target=tape.run, args=(stop,), name="hip4-tape", daemon=True).start()
            if shadow is not None:
                threading.Thread(target=shadow.run, args=(stop,), name="hip4-shadow", daemon=True).start()
            if lab is not None:
                threading.Thread(target=lab.run, args=(stop,), name="shadow-lab", daemon=True).start()
                for name, feed in lab.refs.items():
                    threading.Thread(target=feed.run, args=(stop,), name=f"ref-{name}", daemon=True).start()
            if desk is not None:
                threading.Thread(target=desk.run, args=(stop,), name="research-desk", daemon=True).start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=5)

    async def index(request: Request) -> Response:
        body = (WEB_DIR / "dashboard.html").read_text(encoding="utf-8")
        return HTMLResponse(PAGE_HEAD + body + "</body></html>", headers={"Cache-Control": "no-store"})

    state_cache = {"ts": 0.0, "body": None}

    async def state(request: Request) -> Response:
        # ~0.25 s and ~280 KB on the live server: built off the event loop and shared for 2 s between pollers
        if state_cache["body"] is None or time.time() - state_cache["ts"] > 2.0:
            snap = await asyncio.to_thread(engine.snapshot)
            state_cache.update(ts=time.time(), body=SafeJSON(snap).body)
        return Response(state_cache["body"], media_type="application/json", headers={"Cache-Control": "no-store"})

    async def health(request: Request) -> Response:
        snap = engine.snapshot()
        return SafeJSON({"ok": True, "loop": snap["engine"]["loop"], "status": snap["engine"]["status"]})

    async def kill(request: Request) -> Response:
        engine.kill("Manual kill switch")
        for b in hip4_books:
            b.kill("Manual kill switch")
        if pilot is not None:
            pilot.halt("Manual kill switch")
        return SafeJSON({"ok": True})

    async def resume(request: Request) -> Response:
        engine.resume()
        for b in hip4_books:
            b.resume()
        if pilot is not None:
            pilot.resume()
        return SafeJSON({"ok": True})

    async def reset(request: Request) -> Response:
        engine.reset_paper()
        for b in hip4_books:
            b.reset()
        return SafeJSON({"ok": True})

    def book_api(book, what):
        async def api(request: Request) -> Response:
            if book is None:
                return SafeJSON({"enabled": False, "status": "off"})
            return SafeJSON(book.to_dict(), headers={"Cache-Control": "no-store"})

        async def size(request: Request) -> Response:
            """POST {"multiplier": 2}: this paper book's bet size."""
            if book is None:
                return SafeJSON({"ok": False, "error": f"the HIP-4 {what} is off"}, status_code=400)
            try:
                body = await request.json()
            except ValueError:
                body = None
            mult = body.get("multiplier") if isinstance(body, dict) else None
            if isinstance(mult, bool) or not isinstance(mult, (int, float)):
                return SafeJSON({"ok": False, "error": "send {\"multiplier\": 2}"}, status_code=400)
            try:
                return SafeJSON({"ok": True, **book.set_size(float(mult))})
            except ValueError as exc:
                return SafeJSON({"ok": False, "error": str(exc)}, status_code=400)
        return api, size

    hip4_mm_api, hip4_mm_size = book_api(hip4_mm, "market maker")
    sniper_api, sniper_size = book_api(sniper, "sniper")

    async def tape_api(request: Request) -> Response:
        if tape is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(tape.to_dict(), headers={"Cache-Control": "no-store"})

    from . import hip4_view
    from .hip4_backtest import BacktestRunner
    backtests = BacktestRunner(WEB_DIR.parent / "data", network=engine.cfg.get("network", "mainnet"),
                               auto_every_h=float((engine.cfg.get("hip4_backtest") or {}).get("auto_every_h", 0))
                               if run_loop else 0.0,
                               holdout_from=float((engine.cfg.get("hip4_sniper") or {}).get("holdout_from", 0) or 0)) \
        if (tape is not None or hip4_books) else None

    from .golive import GoLiveGate
    main_book = sniper or hip4_mm                                     # the strategy that would go live
    data_book = hip4_mm or sniper                                     # the widest live books and fair prices
    gate = GoLiveGate(engine.cfg, main_book, backtests, shadow, WEB_DIR.parent / "data") \
        if main_book is not None and "hip4_live" in engine.cfg else None
    view_books = [b for b in (sniper, hip4_mm) if b is not None]
    from .hip4_exec import LiveExecutor
    pilot = LiveExecutor(engine.cfg, sniper, gate, WEB_DIR.parent / "data") \
        if sniper is not None and "hip4_live" in engine.cfg else None
    # the zero-risk strategy tournament (helix/shadow_lab.py); with autopilot, LIVE follows its best-evidenced variant
    lab = None
    lc = engine.cfg.get("hip4_lab") or {}
    if sniper is not None and lc.get("enabled"):
        from .ref_feeds import CexFeed, KalshiFeed
        from .shadow_lab import EXTRA_VARIANTS, VARIANTS, ShadowLab
        refs, variants = {}, dict(VARIANTS)               # outside reference prices: recorded + tested, never traded on
        if lc.get("kalshi"):
            refs["kalshi"] = KalshiFeed(WEB_DIR.parent / "data", spot_fn=sniper.spot_fn)
            variants["kalshi_5c"], variants["kalshi_3c"] = EXTRA_VARIANTS["kalshi_5c"], EXTRA_VARIANTS["kalshi_3c"]
        if lc.get("cex_feed") == "coinbase":
            refs["cex"] = CexFeed(WEB_DIR.parent / "data")
            variants["both_last15_cbveto"] = EXTRA_VARIANTS["both_last15_cbveto"]
        lab = ShadowLab(sniper, WEB_DIR.parent / "data", variants=variants, autopilot=bool(lc.get("autopilot")),
                        alpha=lc.get("alpha"), min_clusters=lc.get("min_clusters"), min_side=lc.get("min_side"),
                        min_age_d=lc.get("min_age_d"), season_size=lc.get("season_size"), refs=refs,
                        xyz_open_guard=bool(lc.get("xyz_open_guard", True)))
        if lab.autopilot and pilot is not None:
            # the lab drives live: the sniper's own sell-only rules stop sending, the pilot's checks all still apply
            lab.live_hook, sniper.on_signal = sniper.on_signal, None
            if lab.live_hook is None:
                lab.live_hook = pilot.on_signal

    # the research desk (PC only): read-only hourly snapshots of the Tokyo server + reports, run as its own process
    desk = None
    dc = engine.cfg.get("research_desk") or {}
    if dc.get("enabled"):
        from .research_desk import DeskRunner
        desk = DeskRunner(WEB_DIR.parent, every_min=float(dc.get("every_min", 60)))
        if not desk.available():
            desk = None

    async def pilot_size(request: Request) -> Response:
        """POST {"multiplier": 2}: the LIVE pilot's bet size - real money, set by the operator on the page."""
        if pilot is None:
            return SafeJSON({"ok": False, "error": "the live pilot is off"}, status_code=400)
        try:
            body = await request.json()
        except ValueError:
            body = None
        mult = body.get("multiplier") if isinstance(body, dict) else None
        if isinstance(mult, bool) or not isinstance(mult, (int, float)):
            return SafeJSON({"ok": False, "error": "send {\"multiplier\": 2}"}, status_code=400)
        try:
            return SafeJSON({"ok": True, **pilot.set_size(float(mult))})
        except ValueError as exc:
            return SafeJSON({"ok": False, "error": str(exc)}, status_code=400)

    remote_cache = {"t": 0.0, "data": None}
    lab_cache = {"t": 0.0, "data": None}

    def _desk_snapshot() -> tuple[dict, str] | None:
        """The research desk's hourly READ-ONLY snapshot of the Tokyo server (research_agent/latest.json), when the
        tunnel is closed: (snapshot, 'hourly snapshot HH:MM UTC'), or None when missing, failed or > 2 h old."""
        try:
            snap = json.loads((WEB_DIR.parent / "research_agent" / "latest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ts = snap.get("pulled_at") or snap.get("ts") or 0
        if snap.get("error") or time.time() - ts > 7200:
            return None
        return snap, "hourly snapshot " + time.strftime("%H:%M UTC", time.gmtime(ts))

    def _remote_pilot() -> dict | None:
        """When live runs on ANOTHER Helix (the Tokyo server, reached through the dashboard SSH tunnel), this PC's page
        shows that bot's live status read-only: hip4_live.remote_pilot_url, e.g. http://127.0.0.1:8788."""
        url = (engine.cfg.get("hip4_live", {}) or {}).get("remote_pilot_url") or ""
        if not url:
            return None
        if time.time() - remote_cache["t"] < 3:
            return remote_cache["data"]
        remote_cache["t"] = time.time()
        try:
            import requests
            d = requests.get(url.rstrip("/") + "/api/pilot", timeout=3).json()
            keep = ("mode", "blockers", "wallet_usdc", "open_value", "live_equity", "live_pnl_total", "realized_today",
                    "open", "open_cost", "orders", "size", "halted", "halt_reason", "speed", "limits", "dry")
            data = {k: d.get(k) for k in keep}
            data.update(where=(engine.cfg["hip4_live"].get("remote_name") or "Tokyo server"), page=url.rstrip("/") + "/hip4",
                        error=None)
        except Exception as exc:
            name = engine.cfg["hip4_live"].get("remote_name") or "Tokyo server"
            desk = _desk_snapshot()
            if desk is not None and isinstance(desk[0].get("pilot"), dict):
                data = {**desk[0]["pilot"], "where": f"{name} · {desk[1]}", "page": url.rstrip("/") + "/hip4",
                        "error": None, "snapshot": True}
            else:
                data = {"where": name, "page": url.rstrip("/") + "/hip4", "mode": None,
                        "error": f"not reachable ({type(exc).__name__}) - open the tunnel: deploy\\dashboard.ps1"}
        remote_cache["data"] = data
        return data

    def _remote_lab() -> dict | None:
        url = (engine.cfg.get("hip4_live", {}) or {}).get("remote_pilot_url") or ""
        if not url:
            return None
        if time.time() - lab_cache["t"] < 3:               # one Tokyo request per 3 s however many tabs poll
            return lab_cache["data"]
        lab_cache["t"] = time.time()
        name = engine.cfg["hip4_live"].get("remote_name") or "Tokyo server"
        try:
            import requests
            d = requests.get(url.rstrip("/") + "/api/shadowlab", timeout=3).json()
            data = {**d, "where": name} if d.get("enabled") else None
        except Exception:
            desk = _desk_snapshot()                        # tunnel closed: the research desk's hourly snapshot
            lab_snap = (desk[0].get("lab") or {}) if desk else {}
            data = {**lab_snap, "where": f"{name} · {desk[1]}"} if desk and lab_snap.get("enabled") else None
        lab_cache["data"] = data
        return data

    async def shadow_lab_api(request: Request) -> Response:
        if lab is None:                      # this PC has no lab: show the Tokyo server's (read-only), if configured
            remote = await asyncio.to_thread(_remote_lab)
            return SafeJSON(remote or {"enabled": False}, headers={"Cache-Control": "no-store"})
        return SafeJSON({"enabled": True, **lab.to_dict()}, headers={"Cache-Control": "no-store"})

    async def pilot_api(request: Request) -> Response:
        if pilot is None:
            return SafeJSON({"mode": "OFF", "blockers": ["the sniper is off"]})
        d = pilot.to_dict()
        if d.get("mode") == "OFF":
            d["remote"] = await asyncio.to_thread(_remote_pilot)
        return SafeJSON(d, headers={"Cache-Control": "no-store"})

    async def live_api(request: Request) -> Response:
        if shadow is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(shadow.to_dict(), headers={"Cache-Control": "no-store"})

    async def golive_api(request: Request) -> Response:
        if gate is None:
            return SafeJSON({"enabled": False})
        return SafeJSON(gate.evaluate(), headers={"Cache-Control": "no-store"})

    async def golive_confirm(request: Request) -> Response:
        """POST {"key": "legal", "value": true}: the operator's own confirmations for the go-live gate."""
        if gate is None:
            return SafeJSON({"ok": False, "error": "go-live gate is off"}, status_code=400)
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("value"), bool) or not isinstance(body.get("key"), str):
            return SafeJSON({"ok": False, "error": "send {\"key\": \"legal\", \"value\": true}"}, status_code=400)
        try:
            gate.confirm(body["key"], body["value"])
        except ValueError as exc:
            return SafeJSON({"ok": False, "error": str(exc)}, status_code=400)
        return SafeJSON({"ok": True, **gate.evaluate()})

    async def hip4_page(request: Request) -> Response:
        body = (WEB_DIR / "hip4.html").read_text(encoding="utf-8")
        return HTMLResponse(PAGE_HEAD + body + "</body></html>", headers={"Cache-Control": "no-store"})

    async def h4_overview(request: Request) -> Response:
        return SafeJSON(hip4_view.overview(tape, data_book, view_books), headers={"Cache-Control": "no-store"})

    async def h4_ladder(request: Request) -> Response:
        coin = request.query_params.get("coin", "BTC")
        return SafeJSON(hip4_view.ladder(tape, data_book, coin, hip4, view_books), headers={"Cache-Control": "no-store"})

    async def h4_universe(request: Request) -> Response:
        try:
            since = float(request.query_params.get("since", "0"))
        except ValueError:
            since = 0.0
        return SafeJSON(hip4_view.universe(tape, data_book, hip4, since if math.isfinite(since) else 0.0, view_books,
                                           pilot),
                        headers={"Cache-Control": "no-store"})

    async def h4_market(request: Request) -> Response:
        try:
            mid = int(request.query_params.get("id", ""))
        except ValueError:
            return SafeJSON({"error": "id must be a market number"}, status_code=400)
        res = hip4_view.market(tape, data_book, mid, hip4, view_books)
        if res is None:
            return SafeJSON({"error": f"market #{mid} is not listed"}, status_code=404)
        return SafeJSON(res, headers={"Cache-Control": "no-store"})

    bt_cache = {"ts": 0.0, "n": -1, "body": None}

    async def backtest_api(request: Request) -> Response:
        if backtests is None:
            return SafeJSON({"error": "backtester needs the HIP-4 tape"}, status_code=400)
        if request.method == "GET":
            # built off the event loop and reused for 15 s (unless a run is in progress or a new result landed): the
            # payload is ~1 MB and the dashboard polls it, which used to block the web thread on the live server
            n = len(backtests.results)
            if bt_cache["body"] is None or backtests.running or n != bt_cache["n"] or time.time() - bt_cache["ts"] > 15:
                data = await asyncio.to_thread(backtests.to_dict)
                bt_cache.update(ts=time.time(), n=n, body=SafeJSON(data).body)
            return Response(bt_cache["body"], media_type="application/json", headers={"Cache-Control": "no-store"})
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return SafeJSON({"ok": False, "error": "send {\"strategy\": \"sniper\", \"params\": {...}, \"hours\": 24}"},
                            status_code=400)
        if body.get("suite") is True:
            backtests.queue_auto()
            return SafeJSON({"ok": True, "message": "full suite queued: sniper 10/100/300/1000 ms, arb, maker"})
        params = body.get("params") or {}
        hours = body.get("hours")
        if not isinstance(params, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                   and math.isfinite(v) for v in params.values()):
            return SafeJSON({"ok": False, "error": "params must be numbers"}, status_code=400)
        if hours is not None and (isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= 24 * 60):
            return SafeJSON({"ok": False, "error": "hours must be between 0 and 1440"}, status_code=400)
        ok, msg = backtests.start(str(body.get("strategy")), params, hours)
        return SafeJSON({"ok": ok, "message": msg}, status_code=200 if ok else 400)

    async def sizing(request: Request) -> Response:
        """POST {"multiplier": 2.0} and/or {"streak": true}: the Jev book's bet size (paper)."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return SafeJSON({"ok": False, "error": "send a JSON object like {\"multiplier\": 2}"}, status_code=400)
        mult, streak = body.get("multiplier"), body.get("streak")
        if mult is None and streak is None:
            return SafeJSON({"ok": False, "error": "send multiplier and/or streak"}, status_code=400)
        if mult is not None and (isinstance(mult, bool) or not isinstance(mult, (int, float, str))):
            return SafeJSON({"ok": False, "error": "multiplier must be a number"}, status_code=400)
        if streak is not None and not isinstance(streak, bool):
            return SafeJSON({"ok": False, "error": "streak must be true or false"}, status_code=400)
        try:
            info = engine.set_sizing(None if mult is None else float(mult), streak)
        except (TypeError, ValueError) as exc:
            return SafeJSON({"ok": False, "error": str(exc)}, status_code=400)
        return SafeJSON({"ok": True, **info})

    async def analyst(request: Request) -> Response:
        if engine.analyst is None:
            return SafeJSON({"ok": False, "error": "Opus analyst not configured"}, status_code=400)
        started = engine.analyst.start(engine.analyst_summary())
        return SafeJSON({"ok": started, **engine.analyst.to_dict()})

    async def live(request: Request) -> Response:
        body = (WEB_DIR / "live.html").read_text(encoding="utf-8")
        return HTMLResponse(PAGE_HEAD + body + "</body></html>", headers={"Cache-Control": "no-store"})

    async def claude(request: Request) -> Response:
        if forecaster is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(forecaster.to_dict(), headers={"Cache-Control": "no-store"})

    async def hourly_api(request: Request) -> Response:
        if hourly is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(hourly.to_dict(), headers={"Cache-Control": "no-store"})

    async def signal_api(request: Request) -> Response:
        if signal is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(signal.to_dict(), headers={"Cache-Control": "no-store"})

    async def hip4_api(request: Request) -> Response:
        if hip4 is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(hip4.to_dict(), headers={"Cache-Control": "no-store"})

    async def lab_weekly(request: Request) -> Response:
        if weekly is None:
            return SafeJSON({"enabled": False, "status": "off"})
        return SafeJSON(weekly.to_dict(), headers={"Cache-Control": "no-store"})

    async def lab_api(request: Request) -> Response:
        path = Path(__file__).resolve().parent.parent / "data" / "lab" / "report.json"
        if not path.exists():
            return SafeJSON({"available": False})
        return Response(path.read_bytes(), media_type="application/json", headers={"Cache-Control": "no-store"})

    routes = [
        Route("/", index),
        Route("/live", live),
        Route("/api/claude", claude),
        Route("/api/hourly", hourly_api),
        Route("/api/signal", signal_api),
        Route("/api/hip4", hip4_api),
        Route("/api/hip4mm", hip4_mm_api),
        Route("/api/hip4mm/size", hip4_mm_size, methods=["POST"]),
        Route("/api/hip4sniper", sniper_api),
        Route("/api/hip4sniper/size", sniper_size, methods=["POST"]),
        Route("/api/tape", tape_api),
        Route("/api/hip4live", live_api),
        Route("/api/pilot", pilot_api),
        Route("/api/shadowlab", shadow_lab_api),
        Route("/api/pilot/size", pilot_size, methods=["POST"]),
        Route("/api/golive", golive_api),
        Route("/api/golive/confirm", golive_confirm, methods=["POST"]),
        Route("/hip4", hip4_page),
        Route("/api/hip4/overview", h4_overview),
        Route("/api/hip4/ladder", h4_ladder),
        Route("/api/hip4/universe", h4_universe),
        Route("/api/hip4/market", h4_market),
        Route("/api/backtest", backtest_api, methods=["GET", "POST"]),
        Route("/api/labweekly", lab_weekly),
        Route("/api/lab", lab_api),
        Route("/api/state", state),
        Route("/api/health", health),
        Route("/api/kill", kill, methods=["POST"]),
        Route("/api/resume", resume, methods=["POST"]),
        Route("/api/reset", reset, methods=["POST"]),
        Route("/api/sizing", sizing, methods=["POST"]),
        Route("/api/analyst", analyst, methods=["POST"]),
    ]
    token = token if token is not None else (os.getenv("HELIX_TOKEN") or None)
    class FreshState(BaseHTTPMiddleware):
        """Any command (POST: kill, resume, reset, sizing, ...) drops the 2 s /api/state cache, so the page never shows
        the state from before the operator's own click."""
        async def dispatch(self, request: Request, call_next):
            if request.method != "GET":
                state_cache["body"] = None
            resp = await call_next(request)
            if request.method != "GET":
                state_cache["body"] = None
            return resp

    return Starlette(routes=routes, lifespan=lifespan,
                     middleware=[Middleware(TokenGate, token=token), Middleware(FreshState)])
