"""
app.py
-------
Double Top / Double Bottom Signal Terminal
A professional Streamlit trading-signal dashboard.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.providers import MARKET_PAIRS, TIMEFRAMES, fetch_market_data
from strategy.double_pattern import detect_patterns, most_recent_pattern
from risk.risk_manager import build_trade_setup
from signals.signal_generator import generate_signal
from charts.chart_builder import build_price_chart, COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_WHITE
from backtest.backtester import run_backtest

# --------------------------------------------------------------------------
# Page config & theme
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Double Top / Bottom Signal Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {COLOR_WHITE}; }}
    .signal-card {{
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e5e9f2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }}
    .signal-card-buy {{ border-left: 6px solid {COLOR_GREEN}; }}
    .signal-card-sell {{ border-left: 6px solid {COLOR_RED}; }}
    .signal-card-none {{ border-left: 6px solid {COLOR_BLUE}; }}
    .metric-label {{ color:#6b7280; font-size:0.85rem; }}
    .metric-value {{ font-weight:700; font-size:1.05rem; }}
    .app-header {{
        background: linear-gradient(90deg, {COLOR_BLUE} 0%, #1e3a8a 100%);
        color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px;
    }}
    .disclaimer {{ font-size: 0.8rem; color: #6b7280; }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_BLUE}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []  # list of dict rows
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# --------------------------------------------------------------------------
# Sidebar - Settings
# --------------------------------------------------------------------------

st.sidebar.markdown("## ⚙️ Settings")

market_type = st.sidebar.selectbox("Market Type", list(MARKET_PAIRS.keys()), key="market_type")
pair_options = MARKET_PAIRS[market_type]
pair = st.sidebar.selectbox("Trading Pair", pair_options, key="pair")
timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=2, key="timeframe")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Strategy Parameters")
min_confidence = st.sidebar.slider("Minimum Confidence (%)", 50, 95, 75, step=1)
min_risk_reward = st.sidebar.slider("Minimum Risk/Reward", 1.0, 4.0, 1.5, step=0.1)
tolerance_pct = st.sidebar.slider("Pattern Tolerance (%)", 0.05, 1.0, 0.15, step=0.05)
sl_buffer_pct = st.sidebar.slider("Stop-Loss Buffer (%)", 0.0, 1.0, 0.1, step=0.05)
lookback_bars = st.sidebar.slider("Lookback Period (bars)", 100, 500, 300, step=25)
tp_method = st.sidebar.radio(
    "Take-Profit Method",
    options=["measured_move", "risk_reward"],
    format_func=lambda x: "Pattern Height (measured move)" if x == "measured_move" else "Risk/Reward Multiple",
)
refresh_interval = st.sidebar.selectbox(
    "Refresh Interval", ["Manual", "30s", "1 min", "5 min"], index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="disclaimer">Signals are probabilistic. A confidence score of '
    '75% is a heuristic quality score, not a guaranteed 75% win rate. '
    'This tool is for educational purposes and is not financial advice.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown(
        f"""
        <div class="app-header">
            <h2 style="margin:0;">📈 Double Top / Bottom Signal Terminal</h2>
            <p style="margin:4px 0 0 0; opacity:0.9;">
                Detects confirmed Double Top and Double Bottom formations and generates
                quality-gated trade setups.
            </p>
            <p style="margin:8px 0 0 0; font-size:0.9rem;">
                <b>Market:</b> {market_type} &nbsp;|&nbsp;
                <b>Pair:</b> {pair} &nbsp;|&nbsp;
                <b>Timeframe:</b> {timeframe}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.write("")
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)

# --------------------------------------------------------------------------
# Data fetch (cached)
# --------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _load_data(market_type: str, pair: str, timeframe: str, lookback_bars: int):
    result = fetch_market_data(market_type, pair, timeframe, lookback_bars=max(lookback_bars, 200))
    return result.df, result.source, result.is_synthetic, result.note


if refresh_clicked:
    _load_data.clear()

with st.spinner(f"Fetching {pair} {timeframe} data..."):
    df, source, is_synthetic, note = _load_data(market_type, pair, timeframe, lookback_bars)

st.session_state.last_refresh = datetime.now(timezone.utc)

if is_synthetic:
    st.warning(
        f"⚠️ Live data unavailable ({note}) Showing **synthetic sample data** so the "
        f"interface remains usable - do not trade on this data."
    )

if df is None or df.empty:
    st.error("No data available for this pair/timeframe. Try a different selection.")
    st.stop()

st.caption(
    f"Data source: **{source}** &nbsp;|&nbsp; {len(df)} candles loaded &nbsp;|&nbsp; "
    f"Last refresh: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

# --------------------------------------------------------------------------
# Pattern detection & signal generation
# --------------------------------------------------------------------------

patterns = detect_patterns(
    df,
    tolerance_pct=tolerance_pct,
    lookback_bars=lookback_bars,
)
active_pattern = most_recent_pattern(patterns)

active_signal = None
if active_pattern is not None:
    trade = build_trade_setup(
        active_pattern,
        sl_buffer_pct=sl_buffer_pct,
        min_risk_reward=min_risk_reward,
        tp_method=tp_method,
    )
    active_signal = generate_signal(
        df, active_pattern, trade,
        min_confidence=min_confidence, min_risk_reward=min_risk_reward,
    )

# --------------------------------------------------------------------------
# Main layout: chart + signal panel
# --------------------------------------------------------------------------

chart_col, panel_col = st.columns([2.4, 1])

with chart_col:
    fig = build_price_chart(
        df,
        active_pattern,
        active_signal.trade if (active_signal and active_signal.is_valid_signal) else None,
        pair, timeframe,
    )
    st.plotly_chart(fig, use_container_width=True)

with panel_col:
    st.markdown("#### Signal")
    if active_signal is None:
        st.markdown(
            '<div class="signal-card signal-card-none">'
            '<h4 style="margin:0;">NO TRADE</h4>'
            '<p style="margin:6px 0 0 0;">No confirmed Double Top or Double Bottom pattern detected '
            'on the current chart.</p></div>',
            unsafe_allow_html=True,
        )
    elif not active_signal.is_valid_signal:
        st.markdown(
            f'<div class="signal-card signal-card-none">'
            f'<h4 style="margin:0;">NO TRADE</h4>'
            f'<p style="margin:6px 0 0 0;">{active_signal.reject_reason}.</p>'
            f'<p class="metric-label" style="margin-top:8px;">Pattern confidence: {active_signal.confidence}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        s = active_signal
        t = s.trade
        is_buy = t.direction == "BUY"
        card_class = "signal-card-buy" if is_buy else "signal-card-sell"
        signal_color = COLOR_GREEN if is_buy else COLOR_RED
        pattern_label = "Double Bottom" if active_pattern.pattern_type == "double_bottom" else "Double Top"

        st.markdown(
            f"""
            <div class="signal-card {card_class}">
                <h3 style="margin:0; color:{signal_color};">{t.direction} SIGNAL</h3>
                <p style="margin:2px 0 10px 0;"><b>{pattern_label} Confirmed</b></p>
                <p style="margin:0;"><b>Confidence:</b> {s.confidence}%</p>
                <p style="margin:0;"><b>Market:</b> {market_type} &nbsp; <b>Pair:</b> {pair}</p>
                <p style="margin:0 0 10px 0;"><b>Timeframe:</b> {timeframe}</p>
                <p style="margin:0;"><b>Entry:</b> {t.entry:,.5f}</p>
                <p style="margin:0;"><b>Stop Loss:</b> {t.stop_loss:,.5f}</p>
                <p style="margin:0;"><b>Take Profit:</b> {t.take_profit:,.5f}</p>
                <p style="margin:0;"><b>Risk/Reward:</b> 1:{t.risk_reward_ratio:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # append to signal history once per fresh breakout bar
        history_key = (pair, timeframe, active_pattern.breakout_idx, active_pattern.pattern_type)
        already_logged = any(
            row.get("_key") == history_key for row in st.session_state.signal_history
        )
        if not already_logged:
            st.session_state.signal_history.append({
                "_key": history_key,
                "Date/Time": st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M UTC"),
                "Market": market_type,
                "Pair": pair,
                "Timeframe": timeframe,
                "Signal": t.direction,
                "Pattern": pattern_label,
                "Entry": round(t.entry, 5),
                "Stop Loss": round(t.stop_loss, 5),
                "Take Profit": round(t.take_profit, 5),
                "Confidence": s.confidence,
                "Risk/Reward": f"1:{t.risk_reward_ratio:.2f}",
                "Outcome": "Pending",
            })

    st.markdown(
        '<p class="disclaimer">Confidence reflects pattern quality, not a guaranteed win rate.</p>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Pattern analysis explanation
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("#### 🔍 Pattern Analysis")

if active_signal is None:
    st.info("No pattern to analyze. Adjust the pattern tolerance or lookback period in the sidebar, "
            "or try a different pair/timeframe.")
else:
    pattern_label = "Double Bottom" if active_pattern.pattern_type == "double_bottom" else "Double Top"
    lines = [f"**{pattern_label} detected**"] + [f"- {n}" for n in active_signal.explanation]
    lines.append(f"- **Final confidence: {active_signal.confidence}%**")
    if not active_signal.is_valid_signal:
        lines.append(f"- **Rejected:** {active_signal.reject_reason}")
    st.markdown("\n".join(lines))

    with st.expander("Confidence score breakdown"):
        b = active_signal.breakdown
        bd = pd.DataFrame([
            ("Pattern symmetry", b.symmetry_score, 10),
            ("Distance between extremes", b.distance_score, 12),
            ("High/low similarity", b.similarity_score, 15),
            ("Neckline breakout strength", b.breakout_strength_score, 15),
            ("Volume confirmation", b.volume_score, 10),
            ("Trend/context confirmation", b.trend_context_score, 10),
            ("Candle confirmation", b.candle_confirmation_score, 10),
            ("Volatility (ATR) confirmation", b.volatility_score, 10),
            ("Risk/Reward quality", b.risk_reward_score, 18),
        ], columns=["Factor", "Score", "Max"])
        st.dataframe(bd, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Signal history
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("#### 📜 Signal History")

if st.session_state.signal_history:
    hist_df = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "_key"} for row in st.session_state.signal_history]
    )
    st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)
    if st.button("Clear signal history"):
        st.session_state.signal_history = []
        st.rerun()
else:
    st.caption("No signals logged yet this session. Signals are logged automatically as they are confirmed.")

# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("#### 🧪 Backtest")
st.caption(f"Backtests the Double Top/Bottom strategy for **{market_type} → {pair} → {timeframe}** "
           f"using the currently loaded historical candles and the settings in the sidebar.")

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        bt = run_backtest(
            df,
            tolerance_pct=tolerance_pct,
            sl_buffer_pct=sl_buffer_pct,
            min_risk_reward=min_risk_reward,
            min_confidence=min_confidence,
            tp_method=tp_method,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trades", bt.total_trades)
    m2.metric("Win Rate", f"{bt.win_rate}%")
    m3.metric("Profit Factor", bt.profit_factor)
    m4.metric("Max Drawdown (R)", bt.max_drawdown_r)

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Winning Trades", bt.winning_trades)
    m6.metric("Losing Trades", bt.losing_trades)
    m7.metric("Avg Risk/Reward", bt.avg_risk_reward)
    m8.metric("Total Return (R)", bt.total_return_r)

    st.caption(
        f"Patterns found: {bt.total_patterns_found} &nbsp;|&nbsp; "
        f"Rejected for confidence < {min_confidence}% or RR < 1:{min_risk_reward:.2f}: "
        f"{bt.rejected_low_confidence}"
    )

    if bt.trades:
        trades_df = pd.DataFrame([{
            "Direction": t.direction,
            "Entry": round(t.entry, 5),
            "Stop Loss": round(t.stop_loss, 5),
            "Take Profit": round(t.take_profit, 5),
            "Exit Price": round(t.exit_price, 5),
            "Outcome": t.outcome,
            "R Multiple": round(t.r_multiple, 2),
            "Confidence": t.confidence,
        } for t in bt.trades])
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades in this backtest window with the current settings.")

    st.markdown(
        '<p class="disclaimer">Backtest results are simplified (no spread, slippage, or swap costs '
        'modeled) and are not a guarantee of future performance. A 75% confidence threshold does '
        'not imply a 75% historical win rate unless demonstrated above.</p>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "⚠️ This application is for educational and informational purposes only and does not "
    "constitute financial advice. Trading forex, crypto, and other instruments carries "
    "substantial risk of loss. Past patterns and backtested results do not guarantee future results."
)
