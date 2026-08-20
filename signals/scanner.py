"""
signals/scanner.py
--------------------
Scans every pair belonging to a given market type (Forex or Crypto) and
returns one row per pair summarizing its current signal status - BUY,
SELL, or NO TRADE - using the same detection -> confidence -> validation
pipeline as the main chart.

Kept separate from app.py so the scanning logic can be reused (e.g. by a
future scheduled job or CLI tool) without touching Streamlit code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from data.providers import MARKET_PAIRS, DataFetchResult
from strategy.double_pattern import detect_patterns, most_recent_pattern
from risk.risk_manager import build_trade_setup
from signals.signal_generator import generate_signal


@dataclass
class ScanRow:
    pair: str
    status: str                 # "BUY", "SELL", "NO TRADE", or "ERROR"
    pattern: str = "-"
    confidence: Optional[float] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    reason: str = ""
    is_synthetic: bool = False


def scan_market(
    market_type: str,
    timeframe: str,
    fetch_fn: Callable[[str, str, str, int], DataFetchResult],
    tolerance_pct: float,
    sl_buffer_pct: float,
    min_confidence: float,
    min_risk_reward: float,
    tp_method: str,
    lookback_bars: int = 300,
) -> List[ScanRow]:
    """
    fetch_fn should have the signature (market_type, pair, timeframe, lookback_bars) -> DataFetchResult
    (this lets the caller pass in a cached Streamlit loader instead of hitting the network every time).
    """
    rows: List[ScanRow] = []

    for pair in MARKET_PAIRS.get(market_type, []):
        try:
            result = fetch_fn(market_type, pair, timeframe, lookback_bars)
            df = result.df if hasattr(result, "df") else result[0]
            is_synthetic = result.is_synthetic if hasattr(result, "is_synthetic") else result[2]
        except Exception as exc:
            rows.append(ScanRow(pair=pair, status="ERROR", reason=str(exc)))
            continue

        if df is None or df.empty or len(df) < 20:
            rows.append(ScanRow(pair=pair, status="ERROR", reason="No data available"))
            continue

        try:
            patterns = detect_patterns(df, tolerance_pct=tolerance_pct, lookback_bars=lookback_bars)
            pattern = most_recent_pattern(patterns)

            if pattern is None:
                rows.append(ScanRow(
                    pair=pair, status="NO TRADE", reason="No confirmed pattern",
                    is_synthetic=is_synthetic,
                ))
                continue

            trade = build_trade_setup(
                pattern, sl_buffer_pct=sl_buffer_pct,
                min_risk_reward=min_risk_reward, tp_method=tp_method,
            )
            signal = generate_signal(
                df, pattern, trade,
                min_confidence=min_confidence, min_risk_reward=min_risk_reward,
            )
            pattern_label = "Double Bottom" if pattern.pattern_type == "double_bottom" else "Double Top"

            if signal.is_valid_signal:
                rows.append(ScanRow(
                    pair=pair, status=trade.direction, pattern=pattern_label,
                    confidence=signal.confidence, entry=trade.entry,
                    stop_loss=trade.stop_loss, take_profit=trade.take_profit,
                    risk_reward=trade.risk_reward_ratio, is_synthetic=is_synthetic,
                ))
            else:
                rows.append(ScanRow(
                    pair=pair, status="NO TRADE", pattern=pattern_label,
                    confidence=signal.confidence, reason=signal.reject_reason,
                    is_synthetic=is_synthetic,
                ))
        except Exception as exc:
            rows.append(ScanRow(pair=pair, status="ERROR", reason=str(exc)))

    return rows


def scan_rows_to_dataframe(rows: List[ScanRow]) -> pd.DataFrame:
    records = []
    for r in rows:
        records.append({
            "Pair": r.pair,
            "Signal": r.status,
            "Pattern": r.pattern,
            "Confidence": f"{r.confidence}%" if r.confidence is not None else "-",
            "Entry": f"{r.entry:,.5f}" if r.entry is not None else "-",
            "Stop Loss": f"{r.stop_loss:,.5f}" if r.stop_loss is not None else "-",
            "Take Profit": f"{r.take_profit:,.5f}" if r.take_profit is not None else "-",
            "Risk/Reward": f"1:{r.risk_reward:.2f}" if r.risk_reward is not None else "-",
            "Note": ("Synthetic data" if r.is_synthetic else "") or r.reason,
        })
    return pd.DataFrame(records)
