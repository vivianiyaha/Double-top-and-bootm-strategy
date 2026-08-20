"""
backtest/backtester.py
-------------------------
Walk-forward-style backtest for the Double Top / Double Bottom strategy.

Methodology (kept intentionally simple and transparent):
  - Slide a growing window across the historical data.
  - At each step, run the same detection + confidence pipeline used live.
  - When a fresh, valid (>= min confidence, >= min RR) signal appears that
    hasn't been evaluated yet, "open" a trade at its entry price.
  - Walk forward bar-by-bar until price hits the Stop Loss or Take Profit
    (whichever comes first) to close the trade.
  - Track win rate, profit factor, drawdown, and how many candidate
    patterns were rejected for being below the confidence threshold.

This is a simplified, single-position-at-a-time simulation intended for
strategy exploration - not a substitute for a full execution-aware
backtesting engine (it ignores spread, slippage, and swap costs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from strategy.double_pattern import detect_patterns
from risk.risk_manager import build_trade_setup
from signals.signal_generator import generate_signal


@dataclass
class BacktestTrade:
    entry_idx: int
    exit_idx: int
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    exit_price: float
    outcome: str          # "WIN" or "LOSS"
    r_multiple: float
    confidence: float


@dataclass
class BacktestResult:
    trades: List[BacktestTrade] = field(default_factory=list)
    rejected_low_confidence: int = 0
    total_patterns_found: int = 0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "WIN")

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "LOSS")

    @property
    def win_rate(self) -> float:
        return round(100 * self.winning_trades / self.total_trades, 2) if self.trades else 0.0

    @property
    def avg_risk_reward(self) -> float:
        if not self.trades:
            return 0.0
        return round(float(np.mean([abs(t.r_multiple) for t in self.trades])), 2)

    @property
    def profit_factor(self) -> float:
        gains = sum(t.r_multiple for t in self.trades if t.r_multiple > 0)
        losses = abs(sum(t.r_multiple for t in self.trades if t.r_multiple < 0))
        if losses == 0:
            return round(gains, 2) if gains > 0 else 0.0
        return round(gains / losses, 2)

    @property
    def total_return_r(self) -> float:
        return round(sum(t.r_multiple for t in self.trades), 2)

    @property
    def average_trade_r(self) -> float:
        return round(self.total_return_r / self.total_trades, 3) if self.trades else 0.0

    @property
    def max_drawdown_r(self) -> float:
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.r_multiple for t in self.trades])
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        return round(float(drawdown.min()), 2)


def run_backtest(
    df: pd.DataFrame,
    tolerance_pct: float = 0.15,
    min_bar_distance: int = 5,
    max_bar_distance: int = 80,
    sl_buffer_pct: float = 0.1,
    min_risk_reward: float = 1.5,
    min_confidence: float = 75.0,
    tp_method: str = "measured_move",
) -> BacktestResult:
    result = BacktestResult()
    if df is None or len(df) < 60:
        return result

    patterns = detect_patterns(
        df, tolerance_pct=tolerance_pct,
        min_bar_distance=min_bar_distance, max_bar_distance=max_bar_distance,
        lookback_bars=len(df),
    )
    result.total_patterns_found = len(patterns)

    data = df.reset_index()
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # process patterns in chronological order of their breakout bar
    patterns = sorted(patterns, key=lambda p: p.breakout_idx)

    last_trade_end = -1
    for pattern in patterns:
        if pattern.breakout_idx <= last_trade_end:
            continue  # avoid overlapping trades in this simplified simulation

        trade = build_trade_setup(
            pattern, sl_buffer_pct=sl_buffer_pct,
            min_risk_reward=min_risk_reward, tp_method=tp_method,
        )
        signal = generate_signal(
            df, pattern, trade,
            min_confidence=min_confidence, min_risk_reward=min_risk_reward,
        )

        if not signal.is_valid_signal:
            result.rejected_low_confidence += 1
            continue

        entry_idx = pattern.breakout_idx
        outcome = None
        exit_idx = None
        exit_price = None

        for j in range(entry_idx + 1, len(data)):
            bar_high, bar_low = high.iloc[j], low.iloc[j]
            if trade.direction == "BUY":
                hit_sl = bar_low <= trade.stop_loss
                hit_tp = bar_high >= trade.take_profit
            else:
                hit_sl = bar_high >= trade.stop_loss
                hit_tp = bar_low <= trade.take_profit

            if hit_sl and hit_tp:
                # conservative assumption: stop loss hit first within the same bar
                outcome, exit_idx, exit_price = "LOSS", j, trade.stop_loss
                break
            elif hit_sl:
                outcome, exit_idx, exit_price = "LOSS", j, trade.stop_loss
                break
            elif hit_tp:
                outcome, exit_idx, exit_price = "WIN", j, trade.take_profit
                break

        if outcome is None:
            continue  # trade never resolved within available data - skip

        r_multiple = (
            1 * trade.risk_reward_ratio if outcome == "WIN" else -1.0
        )

        result.trades.append(BacktestTrade(
            entry_idx=entry_idx, exit_idx=exit_idx, direction=trade.direction,
            entry=trade.entry, stop_loss=trade.stop_loss, take_profit=trade.take_profit,
            exit_price=exit_price, outcome=outcome, r_multiple=r_multiple,
            confidence=signal.confidence,
        ))
        last_trade_end = exit_idx

    return result
