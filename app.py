import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.graph_objects as go

# ------------------------
# Gamma Calculation (Black-Scholes)
# ------------------------
def bs_gamma(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except:
        return 0

# ------------------------
# Get next Friday expiry
# ------------------------
def get_next_friday():
    today = datetime.today()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

# ------------------------
# GEX Calculation
# ------------------------
def calculate_gex(ticker_symbol):
    r = 0.05
    multiplier = 100
    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.history(period='1d')['Close'].iloc[-1]
    expiries = ticker.options
    expiry = get_next_friday()
    if expiry not in expiries:
        expiry = expiries[0]
    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    def process(df, is_call=True):
        df = df[df['openInterest'] > 0].copy()
        df['impliedVolatility'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.5)
        df['T'] = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365
        df['gamma'] = df.apply(lambda x: bs_gamma(spot, x['strike'], x['T'], r, x['impliedVolatility']), axis=1)
        df['gex'] = df['openInterest'] * multiplier * df['gamma'] * spot ** 2 / 10000
        if not is_call:
            df['gex'] *= -1
        return df[['strike', 'gex']]

    call_gex = process(calls, is_call=True)
    put_gex = process(puts, is_call=False)
    total_gex = pd.concat([call_gex, put_gex]).groupby('strike').sum().reset_index()
    total_gex = total_gex.sort_values('strike')

    # Focus on ±30 strikes
    closest_strikes = total_gex['strike'].sub(spot).abs().sort_values().index
    filtered_gex = total_gex.loc[closest_strikes].head(61).sort_values('strike')
    return filtered_gex, spot, expiry

# ------------------------
# Price Chart (5-min)
# ------------------------
def get_intraday_chart(ticker_symbol):
    df = yf.download(ticker_symbol, period="1d", interval="5m", progress=False)
    
    # Ensure timestamps are formatted
    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    fig = go.Figure(data=[go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='5-min Candle'
    )])

    fig.update_layout(
        title=f"{ticker_symbol.upper()} 5-Min Intraday Chart",
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig
# ------------------------
# Streamlit Layout
# ------------------------
st.set_page_config(page_title="GEX Estimator", layout="wide")
st.title("📊 GEX Estimator (SpotGamma-style layout)")

ticker_input = st.text_input("Enter stock ticker:", value="SPY")

if st.button("Run GEX Analysis") and ticker_input:
    try:
        gex_df, spot, expiry = calculate_gex(ticker_input.upper())
        col1, col2 = st.columns([1, 2])  # 1/3 GEX - 2/3 chart

        with col1:
            st.subheader("GEX by Strike")
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=gex_df['gex'],
                y=gex_df['strike'],
                orientation='h',
                marker_color='gray',
                name='GEX'
            ))

            # Flip zone (red line)
            try:
                flip_zone = gex_df[gex_df['gex'] >= 0].iloc[0]['strike']
                fig1.add_vline(x=0, line_dash="dash", line_color="black")
                fig1.add_hline(y=flip_zone, line_dash="dash", line_color="red",
                               annotation_text=f"GEX Flip ≈ {flip_zone}", annotation_position="top left")
            except:
                pass

            # Spot price (blue line)
            fig1.add_hline(y=spot, line_dash="dash", line_color="blue",
                           annotation_text=f"Spot: {spot:.2f}", annotation_position="bottom left")

            fig1.update_layout(
                height=500,
                yaxis_title="Strike",
                xaxis_title="GEX Estimate",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("5-Min Price Chart")
            st.plotly_chart(get_intraday_chart(ticker_input.upper()), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
