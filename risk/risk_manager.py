"""
risk/risk_manager.py
---------------------
Converts a confirmed DoublePattern into concrete Entry / Stop Loss /
Take Profit levels, and computes the resulting risk-to-reward ratio.

Two take-profit methods are supported (configurable in the sidebar):
  - "measured_move": classic technical-analysis target = neckline +/- pattern height
  - "risk_reward":   TP placed at a multiple of the initial risk (e.g. 1:2)

Both numbers are shown to the user so they can see how the target was
derived either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from strategy.double_pattern import DoublePattern

TPMethod = Literal["measured_move", "risk_reward"]


@dataclass
class TradeSetup:
    direction: str          # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    risk_per_unit: float
    reward_per_unit: float
    risk_reward_ratio: float
    tp_method: TPMethod


def build_trade_setup(
    pattern: DoublePattern,
    sl_buffer_pct: float = 0.1,
    min_risk_reward: float = 2.0,
    tp_method: TPMethod = "measured_move",
) -> TradeSetup:
    """
    sl_buffer_pct    extra buffer beyond the second peak/trough, as a % of price
    min_risk_reward  used only when tp_method == "risk_reward"
    """
    entry = pattern.breakout_price

    if pattern.pattern_type == "double_top":
        direction = "SELL"
        buffer = pattern.second_extreme_price * (sl_buffer_pct / 100)
        stop_loss = pattern.second_extreme_price + buffer
        risk_per_unit = stop_loss - entry

        measured_move_tp = entry - pattern.pattern_height
        rr_tp = entry - (risk_per_unit * min_risk_reward)
        take_profit = measured_move_tp if tp_method == "measured_move" else rr_tp
        reward_per_unit = entry - take_profit

    else:  # double_bottom
        direction = "BUY"
        buffer = pattern.second_extreme_price * (sl_buffer_pct / 100)
        stop_loss = pattern.second_extreme_price - buffer
        risk_per_unit = entry - stop_loss

        measured_move_tp = entry + pattern.pattern_height
        rr_tp = entry + (risk_per_unit * min_risk_reward)
        take_profit = measured_move_tp if tp_method == "measured_move" else rr_tp
        reward_per_unit = take_profit - entry

    risk_per_unit = max(risk_per_unit, 1e-9)
    rr_ratio = reward_per_unit / risk_per_unit

    return TradeSetup(
        direction=direction,
        entry=float(entry),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        risk_per_unit=float(risk_per_unit),
        reward_per_unit=float(reward_per_unit),
        risk_reward_ratio=float(rr_ratio),
        tp_method=tp_method,
    )
