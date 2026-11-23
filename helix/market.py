"""Shared market-data types. Every feed (live Hyperliquid or the simulator) hands the engine
the same PerpView, so the engine never knows or cares where data came from."""

from __future__ import annotations

from dataclasses import dataclass, field


class DataError(Exception):
    pass


@dataclass(frozen=True)
class Candle:
    ts: int  # open time, unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Trade:
    ts: float  # seconds
    px: float
    sz: float
    side: str  # "b" = aggressive buy (taker bought), "s" = aggressive sell


@dataclass
class PerpView:
    coin: str
    bids: list[tuple[float, float]]  # (price, size), best first
    asks: list[tuple[float, float]]
    book_ts: float  # exchange timestamp of the book, seconds
    received: float  # local time the book arrived, seconds
    trades: list[Trade] = field(default_factory=list)  # recent, oldest first
    mids: list[tuple[float, float]] = field(default_factory=list)  # (ts, mid), oldest first
    candles: list[Candle] = field(default_factory=list)  # closed 1m candles, oldest first
    ctx: dict = field(default_factory=dict)  # mark, oracle, funding (hourly), open_interest, day_volume, prev_day_px
    source: str = "ws"

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float | None:
        m = self.mid
        if not m:
            return None
        return (self.asks[0][0] - self.bids[0][0]) / m * 1e4


def walk_book(levels: list[tuple[float, float]], qty: float) -> tuple[float, float]:
    """Average fill price for a market order of `qty` against `levels`.
    Returns (avg_price, filled_qty). filled_qty < qty means the book ran out."""
    left, cost = qty, 0.0
    for px, sz in levels:
        take = min(left, sz)
        cost += take * px
        left -= take
        if left <= 1e-12:
            break
    filled = qty - max(left, 0.0)
    return (cost / filled if filled > 0 else 0.0), filled
