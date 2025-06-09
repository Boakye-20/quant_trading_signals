"""
Trading Signal Dashboard
Real-time display of trading signals from the API
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import websocket
import threading
from datetime import datetime, timedelta
from collections import deque
import time
import numpy as np

# Page config
st.set_page_config(
    page_title="Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .signal-buy { 
        background-color: #1f4529;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #00ff00;
    }
    .signal-sell { 
        background-color: #4a1f1f;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #ff0000;
    }
    .signal-hold { 
        background-color: #3a3a3a;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #888;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "BAC"]

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = {}
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=200)
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Helper functions
def fetch_signals():
    """Fetch current signals from API"""
    try:
        response = requests.get(f"{API_URL}/signals", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('signals', {})
    except Exception as e:
        st.error(f"Failed to fetch signals: {e}")
    return {}

def fetch_risk_metrics():
    """Fetch risk metrics from API"""
    try:
        response = requests.get(f"{API_URL}/api/risk", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('metrics', {})
    except Exception as e:
        st.error(f"Failed to fetch risk metrics: {e}")
    return {}

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Dashboard header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("📈 Advanced Trading Signal Dashboard")
    st.markdown("Real-time market microstructure analysis and signal generation")

with col_status:
    if check_api_health():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")

# Top metrics row
st.markdown("### Dashboard Metrics")
metric_cols = st.columns(6)

# Fetch current data
current_signals = fetch_signals()
current_risk = fetch_risk_metrics()

# Update session state
st.session_state.signals = current_signals
st.session_state.risk_metrics = current_risk
st.session_state.last_update = datetime.now()

# Calculate metrics
active_signals = len(current_signals)
buy_signals = sum(1 for s in current_signals.values() if s.get('action') == 'BUY')
sell_signals = sum(1 for s in current_signals.values() if s.get('action') == 'SELL')
avg_confidence = sum(s.get('confidence', 0) for s in current_signals.values()) / max(active_signals, 1)
total_exposure = current_risk.get('total_exposure', 0)
portfolio_value = current_risk.get('portfolio_value', 100000)

# Display metrics
with metric_cols[0]:
    st.metric("Active Signals", active_signals, 
              delta=f"{buy_signals} buy, {sell_signals} sell")

with metric_cols[1]:
    st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%",
              delta="High" if avg_confidence > 0.7 else "Normal")

with metric_cols[2]:
    st.metric("Portfolio Value", f"${portfolio_value:,.0f}")

with metric_cols[3]:
    st.metric("Cash Available", f"${current_risk.get('cash_available', 100000):,.0f}")

with metric_cols[4]:
    st.metric("Total Exposure", f"${total_exposure:,.0f}",
              delta=f"{current_risk.get('exposure_percentage', 0):.1f}%")

with metric_cols[5]:
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

# Signal display section
st.markdown("---")
st.markdown("### 📊 Current Trading Signals")

# Create signal grid
if current_signals:
    # Sort signals by confidence
    sorted_signals = sorted(current_signals.items(), 
                          key=lambda x: x[1].get('confidence', 0), 
                          reverse=True)
    
    # Display in grid
    num_cols = 4
    for i in range(0, len(sorted_signals), num_cols):
        cols = st.columns(num_cols)
        
        for j, col in enumerate(cols):
            if i + j < len(sorted_signals):
                symbol, signal = sorted_signals[i + j]
                
                with col:
                    # Determine signal type and styling
                    action = signal.get('action', 'HOLD')
                    confidence = signal.get('confidence', 0) * 100
                    
                    if action == 'BUY':
                        container_class = "signal-buy"
                        emoji = "🟢"
                        action_color = "#00ff00"
                    elif action == 'SELL':
                        container_class = "signal-sell"
                        emoji = "🔴"
                        action_color = "#ff0000"
                    else:
                        container_class = "signal-hold"
                        emoji = "⚪"
                        action_color = "#888888"
                    
                    # Create signal card
                    st.markdown(f"""
                    <div class="{container_class}">
                        <h3 style="margin:0; color:{action_color};">{emoji} {symbol} - {action}</h3>
                        <p style="margin:5px 0; color:white;">Entry: ${signal.get('current_price', 0):.2f}</p>
                        <p style="margin:5px 0; color:white;">Stop: ${signal.get('stop_loss', 0):.2f}</p>
                        <p style="margin:5px 0; color:white;">Target: ${signal.get('take_profit', 0):.2f}</p>
                        <p style="margin:5px 0; color:white;">Position: {signal.get('position_size', 0)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Also update the CSS at the top of the file (around line 30-50) to ensure all text is visible:
                    st.markdown("""
                    <style>
                        .signal-buy { 
                            background-color: #1f4529;
                            padding: 10px;
                            border-radius: 5px;
                            border: 2px solid #00ff00;
                            color: white;
                        }
                        .signal-sell { 
                            background-color: #4a1f1f;
                            padding: 10px;
                            border-radius: 5px;
                            border: 2px solid #ff0000;
                            color: white;
                        }
                        .signal-hold { 
                            background-color: #3a3a3a;
                            padding: 10px;
                            border-radius: 5px;
                            border: 1px solid #888;
                            color: white;
                        }
                        .metric-card {
                            background-color: #262730;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                        }
                        .stMetric > div {
                            text-align: center;
                        }
                        /* Ensure all paragraph text in signal cards is white */
                        .signal-buy p, .signal-sell p, .signal-hold p {
                            color: white !important;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.progress(confidence / 100)
                    st.caption(f"Confidence: {confidence:.1f}% | Score: {signal.get('score', 0):.2f}")
                    
                    # Add to history
                    st.session_state.signal_history.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': action,
                        'confidence': confidence / 100,
                        'score': signal.get('score', 0),
                        'price': signal.get('current_price', 0)
                    })
else:
    st.info("No active signals. Waiting for signal generation...")
    
    # Show waiting state for each symbol
    cols = st.columns(4)
    for i, symbol in enumerate(SYMBOLS):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="signal-hold">
                <h4 style="margin:0; color:#888;">⏳ {symbol}</h4>
                <p style="margin:5px 0; color:#666;">Analyzing...</p>
            </div>
            """, unsafe_allow_html=True)

# Risk Metrics Section
st.markdown("---")
st.markdown("### 📊 Risk Metrics")

risk_cols = st.columns(4)
with risk_cols[0]:
    st.metric("Risk Status", 
              current_risk.get('risk_status', 'NORMAL'),
              delta="Within limits" if current_risk.get('risk_status') == 'NORMAL' else "Check exposure")

with risk_cols[1]:
    st.metric("Active Positions", 
              current_risk.get('position_count', 0))

with risk_cols[2]:
    st.metric("Max Position Size", 
              f"{current_risk.get('max_position_size', 0.1)*100:.0f}%")

with risk_cols[3]:
    st.metric("Signal Confidence", 
              f"{current_risk.get('average_signal_confidence', 0)*100:.1f}%")

# Signal History Chart
if len(st.session_state.signal_history) > 0:
    st.markdown("---")
    st.markdown("### 📈 Signal History")
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(list(st.session_state.signal_history))
    
    # Create visualization
    tab1, tab2, tab3 = st.tabs(["Signal Timeline", "Confidence Trends", "Action Distribution"])
    
    with tab1:
        # Timeline chart
        fig = go.Figure()
        
        for action in ['BUY', 'SELL']:
            action_df = history_df[history_df['action'] == action]
            if not action_df.empty:
                fig.add_trace(go.Scatter(
                    x=action_df['timestamp'],
                    y=action_df['symbol'],
                    mode='markers',
                    name=action,
                    marker=dict(
                        size=action_df['confidence'] * 20,
                        color='green' if action == 'BUY' else 'red',
                        symbol='triangle-up' if action == 'BUY' else 'triangle-down'
                    ),
                    text=[f"{row['symbol']}<br>{row['action']}<br>Confidence: {row['confidence']*100:.1f}%" 
                          for _, row in action_df.iterrows()],
                    hoverinfo='text'
                ))
        
        fig.update_layout(
            title="Signal Generation Timeline",
            xaxis_title="Time",
            yaxis_title="Symbol",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Confidence over time
        fig2 = go.Figure()
        
        for symbol in history_df['symbol'].unique():
            symbol_df = history_df[history_df['symbol'] == symbol]
            fig2.add_trace(go.Scatter(
                x=symbol_df['timestamp'],
                y=symbol_df['confidence'],
                mode='lines+markers',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig2.add_hline(y=0.7, line_dash="dash", line_color="green", 
                      annotation_text="High Confidence Threshold")
        fig2.add_hline(y=0.5, line_dash="dash", line_color="yellow", 
                      annotation_text="Min Confidence")
        
        fig2.update_layout(
            title="Signal Confidence Trends",
            xaxis_title="Time",
            yaxis_title="Confidence",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Action distribution
        action_counts = history_df['action'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = go.Figure(data=[go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                hole=0.3,
                marker_colors=['green', 'red']
            )])
            fig3.update_layout(title="Signal Distribution", height=300)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Symbol performance
            symbol_stats = history_df.groupby('symbol').agg({
                'action': 'count',
                'confidence': 'mean'
            }).round(3)
            symbol_stats.columns = ['Signal Count', 'Avg Confidence']
            st.dataframe(symbol_stats, use_container_width=True)

# Auto-refresh
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

# Auto-refresh every 5 seconds
time.sleep(2)
st.rerun()