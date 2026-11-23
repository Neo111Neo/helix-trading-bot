#!/usr/bin/env python3
"""Start HELIX.

  python run.py            Live Hyperliquid market data (read-only), paper trading, terminal in your browser
  python run.py --demo     Simulated market + mock Jev, to see everything move with no keys and no internet
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _lan_ip() -> str:
    """This PC's address on the local network (no packets are sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "YOUR-PC-IP"


def main() -> int:
    ap = argparse.ArgumentParser(description="HELIX: Jev x Opus paper-trading engine for Hyperliquid perps")
    ap.add_argument("--demo", action="store_true", help="simulated market + mock Jev (nothing real)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    import uvicorn

    from helix.analyst import Analyst
    from helix.config import ConfigError, load_config
    from helix.engine import Engine, RealClock, SimClock
    from helix.hyperliquid import HyperliquidFeed
    from helix.jev import JevClient, MockJev
    from helix.journal import Journal
    from helix.server import create_app
    from helix.simfeed import SimPerpFeed

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"\n  Config problem: {exc}\n")
        return 2

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    coins = cfg["market"]["coins"]
    if args.demo:
        clock = SimClock(start=time.time(), speed=1.0)
        feed = SimPerpFeed(coins, clock.now)
        jev = MockJev()
        db = data_dir / "demo.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        journal = Journal(db, retain_hours=6)
        jev_line = "MOCK answers (demo only)"
        feed_line = "SIMULATED market (not real prices)"
    else:
        clock = RealClock()
        feed = HyperliquidFeed(coins, network=cfg["network"], stale_after_s=float(cfg["market"]["book_refresh_s"]))
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        if os.getenv("JEV_PROVIDER", "").lower() == "mock":
            jev, jev_line = MockJev(), "MOCK answers (JEV_PROVIDER=mock)"
        elif key:
            jev = JevClient(key, cfg["jev"]["model"], int(cfg["jev"]["deadline_ms"]))
            jev_line = f"ON, pinned to {cfg['jev']['model']}, deadline {cfg['jev']['deadline_ms']} ms, " \
                       f"budget ${cfg['jev']['daily_budget_usd']:g}/day"
        else:
            jev, jev_line = None, f"no TYPESAFE_API_KEY yet, using fallback ({cfg['jev']['fallback']})"
        journal = Journal(data_dir / "helix.db", retain_hours=cfg["journal"]["retain_hours"])
        feed_line = f"Hyperliquid {cfg['network']} order books + trades (WebSocket, REST backup)"

    analyst = Analyst(cfg["analyst"]["model"], int(cfg["analyst"]["max_tokens"]))
    signal = None
    if cfg["signal"]["enabled"] and not args.demo:
        from helix.jev_signal import JevSignal
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        # its own client: once a minute per coin, not latency-critical, so a longer timeout than the 5 s battery.
        # 1-minute candles come from the feed's WebSocket stream (no REST call per coin per minute).
        source = (lambda coin, now: feed.buffers[coin].closed_candles(now) if coin in feed.buffers else None) \
            if isinstance(feed, HyperliquidFeed) else None
        signal = JevSignal(cfg, JevClient(key, cfg["jev"]["model"], 5000) if key else None, candle_source=source)
    engine = Engine(cfg, feed, jev, journal, analyst, clock, data_dir, demo=args.demo, persist=not args.demo,
                    signal=signal)
    forecaster = None
    if cfg["claude_forecast"]["enabled"] and not args.demo:
        from helix.forecaster import ClaudeForecaster
        forecaster = ClaudeForecaster(cfg, engine, data_dir)
    hourly = None
    if cfg["hourly_test"]["enabled"] and not args.demo:
        from helix.hourly import HourlyTest
        hourly = HourlyTest(cfg, data_dir, claude_client=forecaster.client if forecaster else None,
                            claude_model=cfg["claude_forecast"]["model"])
    hip4 = None
    if cfg["hip4"]["enabled"] and not args.demo:
        from helix.hip4 import HIP4Recorder
        from helix.hourly import HourlyTest
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        hip4 = HIP4Recorder(cfg, data_dir, jev_client=JevClient(key, cfg["jev"]["model"], 5000) if key else None,
                            variants=HourlyTest._load_variants(cfg["hourly_test"].get("jev_variants") or {}),
                            claude_client=forecaster.client if forecaster else None,
                            claude_model=cfg["claude_forecast"]["model"])
    tape = None
    if cfg["hip4_tape"]["enabled"] and not args.demo:
        from helix.hip4_tape import HIP4Tape
        tape = HIP4Tape(cfg, data_dir)
    def spot(coin):
        """The engine's live book for the perps it streams; the tape's perp bid/ask for the rest (xyz:)."""
        if isinstance(feed, HyperliquidFeed) and coin in feed.buffers:
            return feed.quote(coin).mid
        got = tape.last_perp.get(coin) if tape is not None else None
        return got[1] if got and time.time() - got[0] < 30 else None

    from helix.hip4_mm import HIP4MarketMaker
    hip4_mm = None
    if cfg["hip4_mm"]["enabled"] and not args.demo:
        hip4_mm = HIP4MarketMaker(cfg, data_dir, spot_fn=spot, notify=lambda level, text: engine.event(level, text))
    sniper = None
    if cfg["hip4_sniper"]["enabled"] and not args.demo:
        sniper = HIP4MarketMaker(cfg, data_dir, spot_fn=spot, notify=lambda level, text: engine.event(level, text),
                                 section="hip4_sniper", name="hip4_sniper")
    shadow = None
    if (sniper or hip4_mm) is not None and cfg["hip4_live"]["shadow"]:
        from helix.hip4_live import ShadowExecutor
        shadow = ShadowExecutor(cfg, sniper or hip4_mm, data_dir)     # live readiness follows the main strategy
    weekly = None
    if cfg["lab_schedule"]["enabled"] and not args.demo:
        from helix.lab_schedule import LabSchedule
        key = os.getenv("TYPESAFE_API_KEY", "").strip()
        weekly = LabSchedule(cfg, data_dir, JevClient(key, cfg["jev"]["model"], 8000) if key else None,
                             notify=lambda level, text: engine.event(level, text))
    app = create_app(engine, forecaster=forecaster, hourly=hourly, signal=signal, hip4=hip4, weekly=weekly,
                     hip4_mm=hip4_mm, tape=tape, shadow=shadow, sniper=sniper)

    host = args.host or cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])
    local_only = host in ("127.0.0.1", "localhost", "::1")
    token = os.getenv("HELIX_TOKEN", "").strip()
    if not local_only and not token:
        print(f"\n  The terminal is set to listen on {host}, which lets other devices open it (and press the kill")
        print("  switch). That needs a password. Add HELIX_TOKEN=some-long-password to .env,")
        print("  or set server.host back to 127.0.0.1 in config.yaml.\n")
        journal.close()
        return 2
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print("\n  HELIX 2 - Jev x Opus paper engine for Hyperliquid perps")
    print(f"  Terminal  : {url}")
    print(f"  Market    : {feed_line}")
    print(f"  Coins     : {', '.join(coins)} - a decision every {cfg['market']['decision_interval_ms']} ms per coin")
    print(f"  Jev       : {jev_line}")
    print(f"  Opus      : {'ready' if analyst.available else 'add ANTHROPIC_API_KEY to .env to enable reviews'}")
    if forecaster is not None:
        fc = cfg["claude_forecast"]
        print(f"  Forecast  : {forecaster.status} - {fc['model']} every {fc['interval_s']} s, budget ${fc['daily_budget_usd']:g}/day")
    if signal is not None:
        sc = cfg["signal"]
        print(f"  Jev v2    : {signal.name} ({signal.status}) - P(up {sc['horizon_min']}m) once a minute per coin, "
              f"enter at {sc['entry_prob']:.0%} / {1 - sc['entry_prob']:.0%}, {cfg['paper']['entry_mode']} entries")
    if hourly is not None:
        print(f"  Hourly    : P(above K at next hour) for {', '.join(hourly.coins)} - {', '.join(hourly.forecasters)}, "
              f"{hourly.status}")
    if hip4 is not None:
        print(f"  HIP-4     : recording Hyperliquid outcome markets for {', '.join(hip4.coins)} vs our forecasters "
              f"(paper only, {len(hip4.snaps)} snapshots so far)")
    if not cfg["perp_trading"]:
        print("  Perps     : trading OFF (perp_trading: false) - live prices only, no perp orders"
              + (" (Jev v2 signal shown, never traded)" if cfg["signal"]["enabled"] else ", no Jev calls"))
    if tape is not None:
        print(f"  Tape      : recording every HIP-4 book and trade on {', '.join(tape.coins)} to data/tape/")
    if hip4_mm is not None:
        mc = cfg["hip4_mm"]
        print(f"  HIP-4 MM  : PAPER market maker on {', '.join(mc['coins'])} outcome markets up to {mc['max_tte_min']:g} min out - "
              f"fair +-{mc['edge']:g}{' (2 levels)' if mc['levels'] == 2 else ''}, {mc['quote_size']:g} contracts/quote, "
              f"max {mc['max_position']:g}/market, risk cap ${mc['max_risk_usd']:g}"
              + (f", exits TP {mc['exits']['take_profit'] * 100:g}c / SL {mc['exits']['stop_loss'] * 100:g}c" if mc['exits']['enabled'] else ""))
    if sniper is not None:
        sc = cfg["hip4_sniper"]
        print(f"  Sniper    : PAPER taker (main strategy) - >= {sc['snipe_edge'] * 100:g}c past model fair, <= {sc['max_tte_min']:g} min "
              f"left, {sc['snipe_latency_ms']:g} ms latency, rules fixed at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(sc['holdout_from']))}")
    if shadow is not None:
        print(f"  Live      : SHADOW only - the {shadow.mm.name} book mirrored as exact Hyperliquid orders, logged, never sent")
    if weekly is not None:
        nxt = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(weekly.to_dict()["next_run"]))
        print(f"  Lab check : weekly Jev v2 re-test on {len(weekly.coins)} coins ({weekly.status}), next {nxt}")
    print(f"  Live view : {url}/live")
    if cfg["hip4_live"]["live_enabled"]:
        print("  Mode      : LIVE PILOT ENABLED - REAL MONEY once its remaining checks pass (see LIVE_PILOT.md)")
    else:
        print("  Mode      : PAPER - no real money, no wallet, no account needed")
    if not local_only:
        print(f"  Phone     : http://{_lan_ip()}:{port}/?token=YOUR_HELIX_TOKEN (same Wi-Fi)")
    print("  Stop      : Ctrl+C\n")

    if cfg["server"].get("open_browser", True) and not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        journal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
