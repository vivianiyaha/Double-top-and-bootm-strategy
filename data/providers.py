"""
data/providers.py
------------------
Market data provider abstraction.

Design goal: the rest of the application (strategy, signals, charts,
backtest) only ever talks to the `MarketDataProvider` interface. Adding a
new data source (a new broker, a new exchange, a paid data vendor, etc.)
means writing one new class and registering it in `get_provider()` -
nothing else in the app needs to change.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


# --------------------------------------------------------------------------
# Instrument catalogue
# --------------------------------------------------------------------------

FOREX_PAIRS: List[str] = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "XAU/USD",
]

CRYPTO_PAIRS: List[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]

MARKET_PAIRS: Dict[str, List[str]] = {
    "Forex": FOREX_PAIRS,
    "Crypto": CRYPTO_PAIRS,
}

# Timeframe label -> (yfinance interval, Binance interval, pandas freq for resampling)
TIMEFRAMES: Dict[str, Dict[str, str]] = {
    "1m":  {"yf": "1m",  "binance": "1m",  "period": "7d"},
    "5m":  {"yf": "5m",  "binance": "5m",  "period": "60d"},
    "15m": {"yf": "15m", "binance": "15m", "period": "60d"},
    "30m": {"yf": "30m", "binance": "30m", "period": "60d"},
    "1h":  {"yf": "60m", "binance": "1h",  "period": "730d"},
    "4h":  {"yf": "60m", "binance": "4h",  "period": "730d"},   # resampled from 1h for yfinance
    "1D":  {"yf": "1d",  "binance": "1d",  "period": "5y"},
    "1W":  {"yf": "1wk", "binance": "1w",  "period": "10y"},
}


@dataclass
class DataFetchResult:
    df: pd.DataFrame
    source: str
    is_synthetic: bool = False
    note: str = ""


class MarketDataProvider(ABC):
    """Abstract base class every data provider must implement."""

    name: str = "base"

    @abstractmethod
    def supports(self, pair: str) -> bool:
        ...

    @abstractmethod
    def fetch_ohlcv(self, pair: str, timeframe: str, lookback_bars: int) -> DataFetchResult:
        """Return a DataFrame indexed by datetime with columns:
        open, high, low, close, volume
        """
        ...


# --------------------------------------------------------------------------
# Forex provider (yfinance)
# --------------------------------------------------------------------------

_YF_SYMBOL_MAP = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X", "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "XAU/USD": "GC=F",
}


class ForexDataProvider(MarketDataProvider):
    """Fetches forex (and spot gold) OHLCV data via yfinance."""

    name = "yfinance"

    def supports(self, pair: str) -> bool:
        return pair in _YF_SYMBOL_MAP

    def fetch_ohlcv(self, pair: str, timeframe: str, lookback_bars: int) -> DataFetchResult:
        import yfinance as yf

        if pair not in _YF_SYMBOL_MAP:
            raise ValueError(f"Unsupported forex pair: {pair}")
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        symbol = _YF_SYMBOL_MAP[pair]
        tf = TIMEFRAMES[timeframe]
        yf_interval = tf["yf"]
        period = tf["period"]

        try:
            raw = yf.download(
                symbol, period=period, interval=yf_interval,
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception as exc:  # network / rate-limit issues
            return DataFetchResult(
                df=_synthetic_ohlcv(pair, lookback_bars),
                source="synthetic-fallback",
                is_synthetic=True,
                note=f"yfinance error ({exc}); showing synthetic sample data.",
            )

        if raw is None or raw.empty:
            return DataFetchResult(
                df=_synthetic_ohlcv(pair, lookback_bars),
                source="synthetic-fallback",
                is_synthetic=True,
                note="yfinance returned no data for this pair/timeframe; showing synthetic sample data.",
            )

        # yfinance can return MultiIndex columns (ticker, field) - flatten them.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "datetime"

        # yfinance has no native 4h interval - resample from 60m bars.
        if timeframe == "4h":
            df = _resample(df, "4h")

        df = df.dropna()
        if len(df) > lookback_bars:
            df = df.iloc[-lookback_bars:]

        return DataFetchResult(df=df, source=self.name)


# --------------------------------------------------------------------------
# Crypto provider (Binance public REST API - no auth required)
# --------------------------------------------------------------------------

_BINANCE_SYMBOL_MAP = {p: p.replace("/", "") for p in CRYPTO_PAIRS}

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class CryptoDataProvider(MarketDataProvider):
    """Fetches crypto OHLCV data via Binance's public klines endpoint."""

    name = "binance"

    def supports(self, pair: str) -> bool:
        return pair in _BINANCE_SYMBOL_MAP

    def fetch_ohlcv(self, pair: str, timeframe: str, lookback_bars: int) -> DataFetchResult:
        if pair not in _BINANCE_SYMBOL_MAP:
            raise ValueError(f"Unsupported crypto pair: {pair}")
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        symbol = _BINANCE_SYMBOL_MAP[pair]
        interval = TIMEFRAMES[timeframe]["binance"]
        limit = min(max(lookback_bars, 50), 1000)  # Binance caps a single call at 1000

        try:
            resp = requests.get(
                _BINANCE_KLINES_URL,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            return DataFetchResult(
                df=_synthetic_ohlcv(pair, lookback_bars),
                source="synthetic-fallback",
                is_synthetic=True,
                note=f"Binance API error ({exc}); showing synthetic sample data.",
            )

        if not raw:
            return DataFetchResult(
                df=_synthetic_ohlcv(pair, lookback_bars),
                source="synthetic-fallback",
                is_synthetic=True,
                note="Binance returned no data for this pair/timeframe; showing synthetic sample data.",
            )

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("datetime")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]

        return DataFetchResult(df=df, source=self.name)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg)
    return out.dropna()


def _synthetic_ohlcv(pair: str, n: int) -> pd.DataFrame:
    """
    Deterministic-ish synthetic OHLCV used ONLY as a fallback when a live
    provider is unreachable (offline / rate-limited), so the UI never
    crashes. Clearly flagged to the user via DataFetchResult.is_synthetic.
    """
    import zlib
    seed = zlib.crc32(pair.encode("utf-8")) % (2 ** 32)
    rng = np.random.default_rng(seed)
    n = max(n, 120)
    base_price = 1.10 if "USD" in pair and "BTC" not in pair and "ETH" not in pair else 100.0
    if "XAU" in pair:
        base_price = 2400.0
    if "BTC" in pair:
        base_price = 65000.0
    if "ETH" in pair:
        base_price = 3200.0

    steps = rng.normal(0, base_price * 0.002, n).cumsum()
    # inject a double-top / double-bottom-like wiggle so the demo has something to find
    wiggle = np.sin(np.linspace(0, 6 * np.pi, n)) * base_price * 0.01
    close = base_price + steps + wiggle
    close = np.maximum(close, base_price * 0.5)

    high = close + np.abs(rng.normal(0, base_price * 0.001, n))
    low = close - np.abs(rng.normal(0, base_price * 0.001, n))
    open_ = close + rng.normal(0, base_price * 0.0007, n)
    volume = np.abs(rng.normal(1000, 300, n))

    now = datetime.now(timezone.utc)
    idx = pd.date_range(end=now, periods=n, freq="15min")

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"
    return df


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

_PROVIDERS: Dict[str, MarketDataProvider] = {
    "Forex": ForexDataProvider(),
    "Crypto": CryptoDataProvider(),
}


def get_provider(market_type: str) -> MarketDataProvider:
    if market_type not in _PROVIDERS:
        raise ValueError(f"No data provider registered for market type '{market_type}'")
    return _PROVIDERS[market_type]


def fetch_market_data(market_type: str, pair: str, timeframe: str, lookback_bars: int = 400) -> DataFetchResult:
    provider = get_provider(market_type)
    return provider.fetch_ohlcv(pair, timeframe, lookback_bars)
