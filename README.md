# Double Top / Double Bottom Signal Terminal

A Streamlit trading-signal dashboard that detects confirmed **Double Top**
and **Double Bottom** chart patterns across Forex and Crypto pairs, and
turns them into quality-gated trade setups (Entry / Stop Loss / Take
Profit / Risk-Reward / Confidence).

---

## 1. Project Structure

```
double_pattern_app/
├── app.py                     # Streamlit UI - entry point
├── requirements.txt
├── data/
│   └── providers.py           # ForexDataProvider (yfinance), CryptoDataProvider (Binance)
├── strategy/
│   └── double_pattern.py      # Double Top / Bottom detection + neckline + breakout logic
├── signals/
│   └── signal_generator.py    # Confidence scoring + signal validation (75% gate)
├── risk/
│   └── risk_manager.py        # Entry / Stop Loss / Take Profit / R:R calculation
├── charts/
│   └── chart_builder.py       # Plotly candlestick chart with pattern overlays
└── backtest/
    └── backtester.py          # Historical strategy simulation
```

Each layer only depends on the layer below it (`app.py` → `signals` →
`strategy`/`risk` → `data`), so any single piece can be swapped out. For
example, to add a new data vendor, write a new class implementing
`MarketDataProvider` in `data/providers.py` and register it in
`get_provider()` - nothing else needs to change.

---

## 2. Installation (local)

```bash
# 1. Create the project folder and copy these files into it (already done if you're reading this in the delivered folder)
cd double_pattern_app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 3. Deployment on Streamlit Community Cloud

1. Push this folder to a GitHub repository (keep the folder structure intact).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select your repo/branch, and set the main file path to `app.py`.
4. Streamlit Cloud will read `requirements.txt` automatically and install everything.
5. Deploy. No API keys are required — `yfinance` and Binance's public REST
   endpoint are both used without authentication.

> If you deploy from an Android phone (Streamlit Cloud's mobile-friendly
> GitHub web editor works fine for this), just make sure the repo keeps the
> `data/`, `strategy/`, `signals/`, `risk/`, `charts/`, and `backtest/`
> folders alongside `app.py` — Streamlit Cloud clones the whole repo.

---

## 4. Strategy Logic

### Double Top (SELL)
1. Two swing highs (peaks) form within a configurable **tolerance %** of
   each other's price.
2. The bars between the two peaks fall within a configurable
   **min/max distance** (avoids both noise and stale, irrelevant patterns).
3. The **neckline** is the lowest low recorded between the two peaks.
4. The pattern is only confirmed once a candle **closes below the
   neckline** — unconfirmed "maybe" patterns are discarded entirely, never
   shown as a signal.
5. On confirmation: **SELL** at the neckline breakout price, **Stop Loss**
   above the second peak (+ buffer), **Take Profit** from either the
   measured-move (pattern height) or a Risk/Reward multiple (your choice
   in the sidebar).

### Double Bottom (BUY)
Mirror image of the above: two similar swing lows, a neckline at the
highest high between them, confirmed only on a close **above** the
neckline, generating a **BUY** signal.

### Confidence Scoring (0–100, transparent & additive)

| Factor | Max points | What it measures |
|---|---|---|
| Pattern symmetry | 10 | How close the pattern's width:height ratio is to a "textbook" shape |
| Distance between extremes | 12 | Whether the two peaks/troughs are neither too close nor too far apart |
| High/low similarity | 15 | How closely the two peak/trough prices match |
| Neckline breakout strength | 15 | How decisively price broke through the neckline, relative to pattern height |
| Volume confirmation | 10 | Whether breakout volume exceeds volume at the pattern's extremes (neutral if volume data is unavailable) |
| Trend/context confirmation | 10 | Whether the pattern reverses a genuine prior trend |
| Candle confirmation | 10 | Whether the breakout candle has a strong, directional body |
| Volatility (ATR) confirmation | 10 | Whether the pattern is large relative to current volatility (filters out noise) |
| Risk/Reward quality | 18 | Whether the resulting trade offers a favorable Risk/Reward ratio |

A signal is only shown when **confidence ≥ your configured minimum
(default 75%) AND Risk/Reward ≥ your configured minimum**. Otherwise the
app displays **NO TRADE**.

**Important:** the confidence score is a heuristic measure of pattern
*quality*, not a statistically measured win probability. A 75% confidence
score is **not** a claim that the strategy wins 75% of the time — actual
historical performance is only shown after you run the **Backtest**
panel on real data.

---

## 5. Backtesting

The **Run Backtest** button replays the same detection → confidence →
validation pipeline used for the live signal across the entire loaded
history for the selected Market → Pair → Timeframe, opening one
simulated trade at a time and walking forward until the Stop Loss or
Take Profit is hit. It reports:

- Total / winning / losing trades, win rate
- Profit factor, average Risk/Reward, average trade (in R)
- Maximum drawdown (in R)
- Total return (in R)
- Number of candidate patterns rejected for confidence below the threshold

This is a simplified single-position simulation (no spread, slippage, or
swap costs modeled) meant for strategy exploration, not a production
execution model.

---

## 6. Data Providers

| Market | Provider | Notes |
|---|---|---|
| Forex + XAU/USD | `yfinance` (Yahoo Finance) | No API key required. 1m/5m data has limited history (a Yahoo Finance limitation, not this app's) |
| Crypto | Binance public REST API (`/api/v3/klines`) | No API key required for public market data |

If a live provider is temporarily unreachable, the app falls back to
clearly-labeled **synthetic sample data** so the UI stays usable — it
never silently presents synthetic candles as real market data.

---

## 7. Configuration Reference (sidebar)

- **Market Type** — Forex or Crypto
- **Trading Pair** — dynamically filtered by Market Type
- **Timeframe** — 1m, 5m, 15m, 30m, 1h, 4h, 1D, 1W
- **Minimum Confidence** — signal quality gate (default 75%)
- **Minimum Risk/Reward** — signal quality gate (default 1.5)
- **Pattern Tolerance (%)** — max allowed price difference between the two peaks/troughs
- **Stop-Loss Buffer (%)** — extra room beyond the second peak/trough
- **Lookback Period** — how many recent candles to scan
- **Take-Profit Method** — measured move (pattern height) vs. Risk/Reward multiple
- **Refresh Interval** — manual or auto-refresh cadence (data is cached for 60s either way)

---

## 8. Disclaimer

This application is for **educational and informational purposes only**
and does not constitute financial advice. Trading forex, crypto, and
other leveraged instruments carries substantial risk of loss. Confidence
scores and backtested results are not guarantees of future performance.
