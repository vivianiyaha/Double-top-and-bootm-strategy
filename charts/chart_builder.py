"""
charts/chart_builder.py
-------------------------
Builds the interactive Plotly candlestick chart with pattern overlays:
first/second extreme markers, neckline, breakout point, and Entry / SL / TP
levels for the active signal (if any).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from strategy.double_pattern import DoublePattern
from risk.risk_manager import TradeSetup

# Theme colors (Blue / Green / Red / White only, per spec)
COLOR_BLUE = "#1a56db"
COLOR_GREEN = "#0e9f6e"
COLOR_RED = "#e02424"
COLOR_WHITE = "#ffffff"
COLOR_GRID = "#e5e9f2"
COLOR_TEXT = "#1f2937"


def build_price_chart(
    df: pd.DataFrame,
    pattern: Optional[DoublePattern],
    trade: Optional[TradeSetup],
    pair: str,
    timeframe: str,
) -> go.Figure:
    data = df.reset_index()
    dt_col = data.columns[0]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data[dt_col],
        open=data["open"], high=data["high"], low=data["low"], close=data["close"],
        increasing_line_color=COLOR_GREEN, decreasing_line_color=COLOR_RED,
        increasing_fillcolor=COLOR_GREEN, decreasing_fillcolor=COLOR_RED,
        name=pair,
    ))

    if pattern is not None:
        _overlay_pattern(fig, data, dt_col, pattern)

    if trade is not None:
        _overlay_trade_levels(fig, data, dt_col, trade)

    fig.update_layout(
        title=f"{pair} - {timeframe}",
        template="plotly_white",
        paper_bgcolor=COLOR_WHITE,
        plot_bgcolor=COLOR_WHITE,
        font=dict(color=COLOR_TEXT),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=560,
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, showspikes=True)
    fig.update_yaxes(gridcolor=COLOR_GRID, showspikes=True)

    return fig


def _overlay_pattern(fig: go.Figure, data: pd.DataFrame, dt_col: str, pattern: DoublePattern):
    is_top = pattern.pattern_type == "double_top"
    marker_color = COLOR_RED if is_top else COLOR_GREEN
    label = "Peak" if is_top else "Trough"

    x1 = data[dt_col].iloc[pattern.first_extreme_idx]
    x2 = data[dt_col].iloc[pattern.second_extreme_idx]
    x_neck = data[dt_col].iloc[pattern.neckline_idx]
    x_breakout = data[dt_col].iloc[pattern.breakout_idx] if pattern.breakout_idx < len(data) else data[dt_col].iloc[-1]

    fig.add_trace(go.Scatter(
        x=[x1, x2], y=[pattern.first_extreme_price, pattern.second_extreme_price],
        mode="markers+text", marker=dict(size=12, color=marker_color, symbol="diamond"),
        text=[f"1st {label}", f"2nd {label}"], textposition="top center" if is_top else "bottom center",
        name=f"{label}s", showlegend=True,
    ))

    # neckline as a horizontal segment spanning from first extreme to breakout point
    fig.add_trace(go.Scatter(
        x=[x1, x_breakout], y=[pattern.neckline_price, pattern.neckline_price],
        mode="lines", line=dict(color=COLOR_BLUE, width=2, dash="dash"),
        name="Neckline",
    ))

    fig.add_trace(go.Scatter(
        x=[x_breakout], y=[pattern.breakout_close],
        mode="markers+text", marker=dict(size=13, color=COLOR_BLUE, symbol="star"),
        text=["Breakout"], textposition="top center",
        name="Breakout",
    ))


def _overlay_trade_levels(fig: go.Figure, data: pd.DataFrame, dt_col: str, trade: TradeSetup):
    x0 = data[dt_col].iloc[0]
    x1 = data[dt_col].iloc[-1]

    levels = [
        ("Entry", trade.entry, COLOR_BLUE),
        ("Stop Loss", trade.stop_loss, COLOR_RED),
        ("Take Profit", trade.take_profit, COLOR_GREEN),
    ]
    for name, price, color in levels:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[price, price],
            mode="lines", line=dict(color=color, width=1.5, dash="dot"),
            name=f"{name} ({price:,.4f})",
        ))
