import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

from config import SYMBOLS, WINDOW_SIZE
from src.data import build_features, prepare_sequences
from src.predict import predict, compute_metrics, next_day_prediction

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Return Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.main { background-color: #0a0a0f; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #6060a0;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #e8e8f0;
}

.direction-up {
    background: linear-gradient(135deg, #0d2b1a, #13311f);
    border: 1px solid #1a5c35;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.direction-down {
    background: linear-gradient(135deg, #2b0d0d, #311313);
    border: 1px solid #5c1a1a;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.direction-label {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
    color: #9090c0;
}

.direction-arrow {
    font-size: 64px;
    line-height: 1;
}

.direction-pct {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
}

.up-color { color: #2ecc71; }
.down-color { color: #e74c3c; }

.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: #e8e8f0;
    margin-bottom: 4px;
}

.sidebar-sub {
    font-size: 12px;
    color: #6060a0;
    margin-bottom: 24px;
}

.stSelectbox > div > div {
    background-color: #13131f !important;
    border: 1px solid #2a2a3f !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3d3d8f, #5050c0);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5050c0, #6060d0);
    transform: translateY(-1px);
}

.info-box {
    background: #13131f;
    border-left: 3px solid #5050c0;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #9090c0;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📈 Stock Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">LSTM · Marketaux Sentiment</div>', unsafe_allow_html=True)

    symbol = st.selectbox("Select Symbol", SYMBOLS)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "saved_models", f"{symbol}_best_model.keras")
    model_exists = os.path.exists(model_path)

    if not model_exists:
        st.warning(f"No saved model found for {symbol}.\n\nExpected: `{model_path}`")

    run = st.button("▶  Run Prediction", disabled=not model_exists)

    st.markdown('<div class="info-box">Loads the saved LSTM model, fetches latest price + sentiment data, and predicts next-day return direction.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span style="font-family: Space Mono; font-size:11px; color:#3a3a6a;">MODEL · LSTM(50) + Dropout(0.3)<br>FEATURES · Close, Returns, Sentiment MA5<br>TARGET · Daily Return</span>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(f"# {symbol}")
st.markdown('<p style="color:#6060a0; font-family: Space Mono; font-size:13px; margin-top:-12px;">RETURN DIRECTION FORECAST</p>', unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #3a3a6a;">
        <div style="font-size: 64px; margin-bottom: 16px;">⬡</div>
        <div style="font-family: Space Mono; font-size: 14px; letter-spacing: 2px;">SELECT A SYMBOL AND RUN PREDICTION</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & predict ────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data and running model for {symbol}..."):
    try:
        model = load_model(model_path)

        df_full = build_features(symbol)
        X_test, y_test, target_scaler, test_df, df_all = prepare_sequences(df_full)

        y_true, pred = predict(model, X_test, y_test, target_scaler)
        metrics = compute_metrics(y_true, pred)
        next_return = next_day_prediction(model, df_full, target_scaler, WINDOW_SIZE)

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── Next day prediction card ──────────────────────────────────────────────────
st.markdown("### Next Trading Day Forecast")

col_dir, col_metrics = st.columns([1, 2])

with col_dir:
    is_up = next_return > 0
    css_class = "direction-up" if is_up else "direction-down"
    arrow = "↑" if is_up else "↓"
    color_class = "up-color" if is_up else "down-color"
    label = "BULLISH" if is_up else "BEARISH"
    pct = f"{next_return * 100:+.3f}%"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="direction-label">{label}</div>
        <div class="direction-arrow {color_class}">{arrow}</div>
        <div class="direction-pct {color_class}">{pct}</div>
        <div style="font-size:11px; color:#6060a0; margin-top:8px; font-family: Space Mono;">PREDICTED RETURN</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.markdown("**Model Performance on Test Set**")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{metrics['mae']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{metrics['rmse']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        da_pct = metrics['directional_accuracy'] * 100
        da_color = "#2ecc71" if da_pct >= 55 else "#e8e8f0" if da_pct >= 50 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Direction Acc.</div>
            <div class="metric-value" style="color:{da_color}">{da_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown("### Actual vs Predicted Returns (Test Period)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=y_true,
    mode="lines",
    name="Actual",
    line=dict(color="#5050c0", width=1.5),
))

fig.add_trace(go.Scatter(
    y=pred,
    mode="lines",
    name="Predicted",
    line=dict(color="#2ecc71", width=1.5, dash="dot"),
))

fig.add_hline(y=0, line_dash="dash", line_color="#3a3a6a", line_width=1)

fig.update_layout(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=11),
    legend=dict(bgcolor="#13131f", bordercolor="#2a2a3f", borderwidth=1),
    margin=dict(l=0, r=0, t=20, b=0),
    height=350,
    xaxis=dict(gridcolor="#1a1a2f", showgrid=True, title="Test Day"),
    yaxis=dict(gridcolor="#1a1a2f", showgrid=True, title="Return"),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Recent price chart ────────────────────────────────────────────────────────
st.markdown("### Recent Price History (60 days)")

recent = df_full["Close"].tail(60)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=recent.index,
    y=recent.values,
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(80,80,192,0.08)",
    line=dict(color="#5050c0", width=2),
    name="Close Price"
))

fig2.update_layout(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(family="Space Mono", color="#e8e8f0", size=11),
    margin=dict(l=0, r=0, t=20, b=0),
    height=280,
    xaxis=dict(gridcolor="#1a1a2f", showgrid=True),
    yaxis=dict(gridcolor="#1a1a2f", showgrid=True, title="Price (USD)"),
    showlegend=False,
)

st.plotly_chart(fig2, use_container_width=True)
