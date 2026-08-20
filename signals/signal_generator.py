"""
signals/signal_generator.py
-----------------------------
Turns a validated DoublePattern + TradeSetup into a scored, explainable
trading Signal. This is where the 75%-confidence "quality gate" lives.

Confidence is a transparent, additive score built from named factors that
each contribute a bounded number of points, capped at 100. It is a
*heuristic quality score*, not a statistically-measured win probability -
that distinction is surfaced in the UI and in `explanation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from strategy.double_pattern import DoublePattern
from risk.risk_manager import TradeSetup


@dataclass
class ConfidenceBreakdown:
    symmetry_score: float = 0.0
    distance_score: float = 0.0
    similarity_score: float = 0.0
    breakout_strength_score: float = 0.0
    volume_score: float = 0.0
    trend_context_score: float = 0.0
    candle_confirmation_score: float = 0.0
    volatility_score: float = 0.0
    risk_reward_score: float = 0.0

    @property
    def total(self) -> float:
        return sum([
            self.symmetry_score, self.distance_score, self.similarity_score,
            self.breakout_strength_score, self.volume_score, self.trend_context_score,
            self.candle_confirmation_score, self.volatility_score, self.risk_reward_score,
        ])


@dataclass
class Signal:
    pattern: DoublePattern
    trade: TradeSetup
    confidence: float
    breakdown: ConfidenceBreakdown
    explanation: List[str] = field(default_factory=list)
    is_valid_signal: bool = False
    reject_reason: str = ""


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _trend_context_score(df: pd.DataFrame, pattern: DoublePattern) -> tuple[float, str]:
    """Reward patterns that reverse an existing trend (the classical setup
    for a double top/bottom) using a simple SMA-slope check."""
    data = df.reset_index()
    if len(data) < 30:
        return 5.0, "Insufficient history to fully assess prior trend"

    window = data["close"].iloc[max(0, pattern.first_extreme_idx - 20):pattern.first_extreme_idx + 1]
    if len(window) < 5:
        return 5.0, "Limited pre-pattern data for trend context"

    slope = np.polyfit(np.arange(len(window)), window.values, 1)[0]

    if pattern.pattern_type == "double_top" and slope > 0:
        return 10.0, "Pattern formed after a clear prior uptrend (favorable reversal context)"
    if pattern.pattern_type == "double_bottom" and slope < 0:
        return 10.0, "Pattern formed after a clear prior downtrend (favorable reversal context)"
    return 4.0, "Prior trend context is weak or unclear"


def _candle_confirmation_score(df: pd.DataFrame, pattern: DoublePattern) -> tuple[float, str]:
    """Reward a decisive (large-bodied) breakout candle in the direction of the signal."""
    data = df.reset_index()
    idx = pattern.breakout_idx
    if idx >= len(data):
        return 0.0, "Breakout candle unavailable"

    row = data.iloc[idx]
    body = abs(row["close"] - row["open"])
    rng = max(row["high"] - row["low"], 1e-9)
    body_ratio = body / rng

    directional = (
        row["close"] < row["open"] if pattern.pattern_type == "double_top"
        else row["close"] > row["open"]
    )

    if directional and body_ratio > 0.6:
        return 10.0, "Breakout candle closed strongly in the signal direction"
    if directional:
        return 6.0, "Breakout candle closed in the signal direction"
    return 2.0, "Breakout candle body is weak or against the signal direction"


def score_confidence(df: pd.DataFrame, pattern: DoublePattern, trade: TradeSetup) -> tuple[ConfidenceBreakdown, List[str]]:
    notes: List[str] = []
    b = ConfidenceBreakdown()

    # 1. Symmetry: how visually even the pattern looks (distance vs height ratio)
    ideal_ratio = 3.0  # a "textbook" double top/bottom is roughly 3x wider than it is tall
    actual_ratio = pattern.bar_distance / max(pattern.pattern_height / max(pattern.neckline_price, 1e-9) * 1000, 1e-6)
    symmetry_penalty = min(abs(actual_ratio - ideal_ratio) / ideal_ratio, 1.0)
    b.symmetry_score = round(10 * (1 - symmetry_penalty), 1)
    notes.append(f"Pattern symmetry score: {b.symmetry_score}/10")

    # 2. Distance between the two peaks/troughs (favor a healthy separation, not too tight/too wide)
    if 8 <= pattern.bar_distance <= 40:
        b.distance_score = 12.0
        notes.append("Distance between the two extremes is within an ideal range")
    elif 5 <= pattern.bar_distance <= 60:
        b.distance_score = 8.0
        notes.append("Distance between the two extremes is acceptable")
    else:
        b.distance_score = 3.0
        notes.append("Distance between the two extremes is outside the ideal range")

    # 3. Similarity of the two highs/lows
    b.similarity_score = round(15 * (pattern.extreme_similarity_pct / 100), 1)
    notes.append(f"Two {'peaks' if pattern.pattern_type == 'double_top' else 'troughs'} are "
                 f"{pattern.extreme_similarity_pct:.2f}% similar")

    # 4. Neckline breakout strength
    b.breakout_strength_score = round(min(pattern.breakout_strength_pct, 100) / 100 * 15, 1)
    notes.append(f"Neckline breakout strength: {pattern.breakout_strength_pct:.1f}% of pattern height")

    # 5. Volume confirmation (only if real volume data exists, i.e. not all-zero)
    if pattern.breakout_volume > 0 and pattern.avg_volume_extremes > 0:
        vol_ratio = pattern.breakout_volume / pattern.avg_volume_extremes
        if vol_ratio >= 1.3:
            b.volume_score = 10.0
            notes.append("Breakout volume is notably higher than volume at the two extremes")
        elif vol_ratio >= 1.0:
            b.volume_score = 6.0
            notes.append("Breakout volume is roughly in line with prior volume")
        else:
            b.volume_score = 2.0
            notes.append("Breakout volume is below prior volume (weaker confirmation)")
    else:
        b.volume_score = 5.0
        notes.append("Volume data unavailable for this pair/timeframe - neutral score applied")

    # 6. Trend/context confirmation
    trend_score, trend_note = _trend_context_score(df, pattern)
    b.trend_context_score = trend_score
    notes.append(trend_note)

    # 7. Candle confirmation
    candle_score, candle_note = _candle_confirmation_score(df, pattern)
    b.candle_confirmation_score = candle_score
    notes.append(candle_note)

    # 8. Volatility / ATR confirmation - penalize patterns that are tiny relative to ATR (noise)
    atr_series = _atr(df)
    atr_val = atr_series.iloc[-1] if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else None
    if atr_val and atr_val > 0:
        height_to_atr = pattern.pattern_height / atr_val
        if height_to_atr >= 2.0:
            b.volatility_score = 10.0
            notes.append("Pattern height is well above current volatility (ATR) - meaningful structure")
        elif height_to_atr >= 1.0:
            b.volatility_score = 6.0
            notes.append("Pattern height is reasonable relative to current volatility (ATR)")
        else:
            b.volatility_score = 2.0
            notes.append("Pattern height is small relative to current volatility - may be noise")
    else:
        b.volatility_score = 5.0
        notes.append("ATR unavailable - neutral volatility score applied")

    # 9. Risk/reward quality
    if trade.risk_reward_ratio >= 2.5:
        b.risk_reward_score = 18.0
        notes.append(f"Risk/Reward of 1:{trade.risk_reward_ratio:.2f} is excellent")
    elif trade.risk_reward_ratio >= 1.5:
        b.risk_reward_score = 12.0
        notes.append(f"Risk/Reward of 1:{trade.risk_reward_ratio:.2f} meets minimum quality bar")
    elif trade.risk_reward_ratio >= 1.0:
        b.risk_reward_score = 6.0
        notes.append(f"Risk/Reward of 1:{trade.risk_reward_ratio:.2f} is marginal")
    else:
        b.risk_reward_score = 0.0
        notes.append(f"Risk/Reward of 1:{trade.risk_reward_ratio:.2f} is poor")

    return b, notes


def generate_signal(
    df: pd.DataFrame,
    pattern: DoublePattern,
    trade: TradeSetup,
    min_confidence: float = 75.0,
    min_risk_reward: float = 1.5,
) -> Signal:
    breakdown, notes = score_confidence(df, pattern, trade)
    confidence = round(min(breakdown.total, 100.0), 1)

    is_valid = True
    reject_reason = ""

    if trade.risk_reward_ratio < min_risk_reward:
        is_valid = False
        reject_reason = (
            f"Risk/Reward of 1:{trade.risk_reward_ratio:.2f} is below the minimum "
            f"required 1:{min_risk_reward:.2f}"
        )
    elif confidence < min_confidence:
        is_valid = False
        reject_reason = f"Confidence {confidence}% is below the {min_confidence}% minimum threshold"

    return Signal(
        pattern=pattern,
        trade=trade,
        confidence=confidence,
        breakdown=breakdown,
        explanation=notes,
        is_valid_signal=is_valid,
        reject_reason=reject_reason,
    )
