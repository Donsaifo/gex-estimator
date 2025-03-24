import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------
# Gamma calculation (Black-Scholes)
# ------------------------
def bs_gamma(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
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

    # Focus on ±30 strikes around spot
    closest_strikes = total_gex['strike'].sub(spot).abs().sort_values().index
    filtered_gex = total_gex.loc[closest_strikes].head(61).sort_values('strike')

    return filtered_gex, spot, expiry

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="GEX Estimator", layout="wide")

st.title("📊 GEX Estimator")
ticker_input = st.text_input("Enter stock ticker (e.g. SPY, TSLA, NVDA):", value="SPY")

if st.button("Run GEX Analysis") and ticker_input:
    with st.spinner("Calculating GEX..."):
        try:
            gex_df, spot, expiry = calculate_gex(ticker_input.upper())

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(gex_df['strike'], gex_df['gex'], width=2.0, color='gray')
            ax.axhline(0, linestyle='--', color='black')

            # GEX flip line
            try:
                flip_zone = gex_df[gex_df['gex'] >= 0].iloc[0]['strike']
                ax.axvline(flip_zone, linestyle='--', color='red', label=f"GEX Flip ≈ {flip_zone}")
            except:
                pass

            # Spot price line
            ax.axvline(spot, linestyle='--', color='blue', label=f"Spot Price: {spot:.2f}")
            ax.set_xticks(gex_df['strike'][::2])
            ax.set_xticklabels(gex_df['strike'][::2].astype(int), rotation=45)
            ax.set_xlim(gex_df['strike'].min() - 5, gex_df['strike'].max() + 5)
            ax.set_title(f"{ticker_input.upper()} GEX by Strike (Expiry: {expiry})")
            ax.set_xlabel("Strike Price")
            ax.set_ylabel("Estimated GEX")
            ax.grid(axis='y')
            ax.legend()

            st.pyplot(fig)
            st.dataframe(gex_df)

        except Exception as e:
            st.error(f"Error: {e}")
