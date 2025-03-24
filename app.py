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

    total_gex['distance'] = abs(total_gex['strike'] - spot)
    filtered_gex = total_gex.sort_values('distance').head(101).sort_values('strike')

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

        colors = ['green' if val > 0 else 'red' for val in gex_df['gex']]
        spacing_factor = 10
        gex_df['strike_spaced'] = np.arange(len(gex_df)) * spacing_factor
        strike_labels = gex_df['strike'].astype(str)

        fig1.add_trace(go.Bar(
            x=gex_df['gex'],
            y=gex_df['strike_spaced'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=1)),
            width=spacing_factor * 0.6,
            name='GEX',
            text=strike_labels,
            hovertemplate='Strike: %{text}<br>GEX: %{x:,.0f}'
        ))

        spot_pos = gex_df.iloc[(gex_df['strike'] - spot).abs().idxmin()]['strike_spaced']
        fig1.add_hline(y=spot_pos, line_dash="dash", line_color="blue",
                       annotation_text=f"Spot: {spot:.2f}", annotation_position="bottom left")

        max_gex_row = gex_df.iloc[gex_df['gex'].abs().idxmax()]
        fig1.add_shape(type="rect",
                      x0=0, x1=max_gex_row['gex'],
                      y0=max_gex_row['strike_spaced'] - spacing_factor / 2,
                      y1=max_gex_row['strike_spaced'] + spacing_factor / 2,
                      line=dict(color="green", width=0),
                      fillcolor="green", opacity=0.2,
                      layer="below")

        fig1.update_layout(
            height=450,
            width=350,
            yaxis=dict(
                title="Strike",
                tickmode='array',
                tickvals=gex_df['strike_spaced'],
                ticktext=gex_df['strike'],
                tickfont=dict(size=11)
            ),
            xaxis_title="GEX Estimate",
            hovermode='y unified',
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig1)

    except Exception as e:
        st.error(f"Error: {e}")
