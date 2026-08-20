"""
strategy/double_pattern.py
---------------------------
Detects confirmed Double Top and Double Bottom formations on OHLCV data
using scipy's peak finder, then validates the pattern through a strict
multi-step checklist:

    1. Pattern formation   (two peaks / two troughs found)
    2. Pattern validation  (peaks/troughs within tolerance of each other)
    3. Neckline identified (the swing low/high between the two extremes)
    4. Breakout confirmed  (a candle closes through the neckline)
    5. (Confidence scoring and RR validation happen in signals/ and risk/)

Only patterns that pass steps 1-4 are returned to the caller at all -
"maybe" patterns that never break the neckline are discarded, in line
with the "quality over quantity" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PatternType = Literal["double_top", "double_bottom"]


@dataclass
class DoublePattern:
    pattern_type: PatternType
    first_extreme_idx: int
    second_extreme_idx: int
    neckline_idx: int
    breakout_idx: int
    first_extreme_price: float
    second_extreme_price: float
    neckline_price: float
    breakout_price: float
    breakout_close: float

    # descriptive metrics used later for confidence scoring
    extreme_similarity_pct: float = 0.0     # how close the two peaks/troughs are, 0-100
    bar_distance: int = 0                   # bars between the two extremes
    pattern_height: float = 0.0             # |extreme - neckline|
    breakout_strength_pct: float = 0.0      # how far price closed beyond neckline, relative to height
    avg_volume_extremes: float = 0.0
    breakout_volume: float = 0.0
    notes: List[str] = field(default_factory=list)


def _find_swing_points(series: pd.Series, distance: int, prominence: float):
    """Return indices of local maxima and minima in a price series."""
    peaks, _ = find_peaks(series.values, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-series.values, distance=distance, prominence=prominence)
    return peaks, troughs


def detect_patterns(
    df: pd.DataFrame,
    tolerance_pct: float = 0.15,
    min_bar_distance: int = 5,
    max_bar_distance: int = 80,
    lookback_bars: int = 300,
) -> List[DoublePattern]:
    """
    Scan the most recent `lookback_bars` candles for confirmed Double Top
    and Double Bottom patterns.

    tolerance_pct       max % difference allowed between the two peaks/troughs
    min_bar_distance    minimum candles between the two extremes (avoids noise)
    max_bar_distance    maximum candles between the two extremes (keeps patterns relevant)
    """
    if df is None or len(df) < 20:
        return []

    data = df.tail(lookback_bars).copy()
    data = data.reset_index()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"] if "volume" in data.columns else pd.Series(np.zeros(len(data)))

    # prominence scales with the instrument's own volatility so this works
    # across FX pairs (small decimals) and crypto/gold (large numbers).
    price_range = high.max() - low.min()
    prominence = max(price_range * 0.01, close.std() * 0.3, 1e-9)

    peak_idx, trough_idx = _find_swing_points(close, distance=min_bar_distance, prominence=prominence)

    patterns: List[DoublePattern] = []

    patterns.extend(
        _scan_double_tops(data, high, low, close, volume, peak_idx, trough_idx,
                           tolerance_pct, min_bar_distance, max_bar_distance)
    )
    patterns.extend(
        _scan_double_bottoms(data, high, low, close, volume, peak_idx, trough_idx,
                              tolerance_pct, min_bar_distance, max_bar_distance)
    )

    return patterns


def _scan_double_tops(data, high, low, close, volume, peak_idx, trough_idx,
                       tolerance_pct, min_dist, max_dist) -> List[DoublePattern]:
    results = []
    for i in range(len(peak_idx) - 1):
        p1, p2 = peak_idx[i], peak_idx[i + 1]
        bar_dist = p2 - p1
        if not (min_dist <= bar_dist <= max_dist):
            continue

        h1, h2 = high.iloc[p1], high.iloc[p2]
        similarity_pct = 100 * (1 - abs(h1 - h2) / max(h1, h2))
        pct_diff = abs(h1 - h2) / max(h1, h2) * 100
        if pct_diff > tolerance_pct:
            continue

        # neckline = lowest low between the two peaks
        between = low.iloc[p1:p2 + 1]
        if between.empty:
            continue
        neckline_idx = between.idxmin()
        neckline_price = between.min()

        # breakout: first close AFTER the second peak that closes below neckline
        breakout_idx = None
        for j in range(p2 + 1, len(close)):
            if close.iloc[j] < neckline_price:
                breakout_idx = j
                break
        if breakout_idx is None:
            continue  # not yet confirmed - discard (quality over quantity)

        pattern_height = max(h1, h2) - neckline_price
        if pattern_height <= 0:
            continue
        breakout_strength_pct = min(
            100.0, (neckline_price - close.iloc[breakout_idx]) / pattern_height * 100
        )

        avg_vol_extremes = float((volume.iloc[p1] + volume.iloc[p2]) / 2) if len(volume) else 0.0
        breakout_vol = float(volume.iloc[breakout_idx]) if len(volume) else 0.0

        results.append(DoublePattern(
            pattern_type="double_top",
            first_extreme_idx=int(p1),
            second_extreme_idx=int(p2),
            neckline_idx=int(neckline_idx),
            breakout_idx=int(breakout_idx),
            first_extreme_price=float(h1),
            second_extreme_price=float(h2),
            neckline_price=float(neckline_price),
            breakout_price=float(neckline_price),
            breakout_close=float(close.iloc[breakout_idx]),
            extreme_similarity_pct=float(similarity_pct),
            bar_distance=int(bar_dist),
            pattern_height=float(pattern_height),
            breakout_strength_pct=float(breakout_strength_pct),
            avg_volume_extremes=avg_vol_extremes,
            breakout_volume=breakout_vol,
        ))
    return results


def _scan_double_bottoms(data, high, low, close, volume, peak_idx, trough_idx,
                          tolerance_pct, min_dist, max_dist) -> List[DoublePattern]:
    results = []
    for i in range(len(trough_idx) - 1):
        t1, t2 = trough_idx[i], trough_idx[i + 1]
        bar_dist = t2 - t1
        if not (min_dist <= bar_dist <= max_dist):
            continue

        l1, l2 = low.iloc[t1], low.iloc[t2]
        similarity_pct = 100 * (1 - abs(l1 - l2) / max(l1, l2))
        pct_diff = abs(l1 - l2) / max(l1, l2) * 100
        if pct_diff > tolerance_pct:
            continue

        # neckline = highest high between the two troughs
        between = high.iloc[t1:t2 + 1]
        if between.empty:
            continue
        neckline_idx = between.idxmax()
        neckline_price = between.max()

        breakout_idx = None
        for j in range(t2 + 1, len(close)):
            if close.iloc[j] > neckline_price:
                breakout_idx = j
                break
        if breakout_idx is None:
            continue

        pattern_height = neckline_price - min(l1, l2)
        if pattern_height <= 0:
            continue
        breakout_strength_pct = min(
            100.0, (close.iloc[breakout_idx] - neckline_price) / pattern_height * 100
        )

        avg_vol_extremes = float((volume.iloc[t1] + volume.iloc[t2]) / 2) if len(volume) else 0.0
        breakout_vol = float(volume.iloc[breakout_idx]) if len(volume) else 0.0

        results.append(DoublePattern(
            pattern_type="double_bottom",
            first_extreme_idx=int(t1),
            second_extreme_idx=int(t2),
            neckline_idx=int(neckline_idx),
            breakout_idx=int(breakout_idx),
            first_extreme_price=float(l1),
            second_extreme_price=float(l2),
            neckline_price=float(neckline_price),
            breakout_price=float(neckline_price),
            breakout_close=float(close.iloc[breakout_idx]),
            extreme_similarity_pct=float(similarity_pct),
            bar_distance=int(bar_dist),
            pattern_height=float(pattern_height),
            breakout_strength_pct=float(breakout_strength_pct),
            avg_volume_extremes=avg_vol_extremes,
            breakout_volume=breakout_vol,
        ))
    return results


def most_recent_pattern(patterns: List[DoublePattern]) -> Optional[DoublePattern]:
    """Return the pattern whose breakout happened most recently (freshest signal)."""
    if not patterns:
        return None
    return max(patterns, key=lambda p: p.breakout_idx)
