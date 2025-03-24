import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.graph_objects as go

# ------------------------
# Black-Scholes Gamma
# ------------------------
def bs_gamma(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except:
        return 0

# ------------------------
# Get Closest Friday Expiry
# ------------------------
def get_next_friday():
    today = datetime.today()
    weekday = today.weekday()
    if weekday < 4:
        days_ahead = 4 - weekday
    elif weekday == 4:
        days_ahead = 0
    else:
        days_ahead = (7 - weekday) + 4
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

# ------------------------
# Calculate GEX
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

    # ±50 strikes around spot
    closest_strikes = total_gex['strike'].sub(spot).abs().sort_values().index
    filtered_gex = total_gex.loc[closest_strikes].head(101).sort_values('strike')
    
    # Add cumulative GEX
    filtered_gex['cumulative_gex'] = filtered_gex['gex'].cumsum()
    
    return filtered_gex, spot, expiry

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="GEX Estimator", layout="wide")
st.title("📊 GEX Estimator")

ticker_input = st.text_input("Enter stock ticker:", value="SPY")

if st.button("Run GEX Analysis") and ticker_input:
    try:
        gex_df, spot, expiry = calculate_gex(ticker_input.upper())

        st.subheader("GEX by Strike")
        fig1 = go.Figure()

        # GEX bars
        fig1.add_trace(go.Bar(
            x=gex_df['gex'],
            y=gex_df['strike'],
            orientation='h',
            marker_color='gray',
            name='GEX'
        ))

        # Cumulative GEX overlay
        fig1.add_trace(go.Scatter(
            x=gex_df['cumulative_gex'],
            y=gex_df['strike'],
            mode='lines+markers',
            line=dict(color='orange', width=2),
            name='Cumulative GEX',
            yaxis='y1'
        ))

        try:
            flip_zone = gex_df[gex_df['gex'] >= 0].iloc[0]['strike']
            fig1.add_hline(y=flip_zone, line_dash="dash", line_color="red",
                           annotation_text=f"GEX Flip ≈ {flip_zone}", annotation_position="top left")
        except:
            pass

        fig1.add_hline(y=spot, line_dash="dash", line_color="blue",
                       annotation_text=f"Spot: {spot:.2f}", annotation_position="bottom left")

        # Highlight max GEX zone
        max_gex_row = gex_df.iloc[gex_df['gex'].abs().idxmax()]
        fig1.add_shape(type="rect",
                      x0=0, x1=max_gex_row['gex'],
                      y0=max_gex_row['strike'] - 0.5, y1=max_gex_row['strike'] + 0.5,
                      line=dict(color="green", width=0),
                      fillcolor="green", opacity=0.2,
                      layer="below")

        fig1.update_layout(
            height=650,
            yaxis=dict(title="Strike", tickmode='linear', dtick=2),
            xaxis_title="GEX Estimate",
            hovermode='y unified',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
