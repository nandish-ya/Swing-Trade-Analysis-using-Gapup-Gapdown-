import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# =====================================================
# CONFIG
# =====================================================
STOCK_TICKER = "CANBK.NS"          # Canara Bank
INDEX_TICKER = "^NSEBANK"          # BankNifty

DATA_FILE = "D:/swing_proj/src/data/labeled.csv"

ATR_PERIOD = 14
RSI_PERIOD = 14
MA_PERIOD = 20
VOL_PERIOD = 20

# =====================================================
# HELPER INDICATORS
# =====================================================
def compute_atr(df, period=14):
    # Ensure inputs are Series, not DataFrames
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_obv(close, volume):
    # Ensure inputs are Series
    obv = [0]
    # Use .values to avoid index alignment issues during loop
    c_vals = close.values
    v_vals = volume.values
    
    for i in range(1, len(c_vals)):
        if c_vals[i] > c_vals[i - 1]:
            obv.append(obv[-1] + v_vals[i])
        elif c_vals[i] < c_vals[i - 1]:
            obv.append(obv[-1] - v_vals[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

# =====================================================
# FETCH LATEST DATA
# =====================================================
def fetch_latest_data(ticker, days=60):
    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        progress=False,
        auto_adjust=True  # Handle split/div adjustments automatically
    )
    
    # === CRITICAL FIX: Flatten MultiIndex Columns ===
    # yfinance often returns columns like ('Close', 'CANBK.NS')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

# =====================================================
# MAIN
# =====================================================
def main():
    print("Fetching latest market data...")

    stock = fetch_latest_data(STOCK_TICKER)
    index = fetch_latest_data(INDEX_TICKER)

    if stock.empty or index.empty:
        print("❌ Failed to fetch data.")
        return

    # Align dates using an inner join
    # We only need the Close price of the index
    df = stock.join(
        index[["Close"]],
        rsuffix="_BN",
        how="inner"
    )

    # Rename the joined index column clearly
    df.rename(columns={"Close_BN": "BN_Close"}, inplace=True)

    # =================================================
    # FEATURE ENGINEERING
    # =================================================
    # Now df['Close'] is guaranteed to be a Series (1 column)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Gap_pct"] = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"]

    df["S_ATR14"] = compute_atr(df, ATR_PERIOD)
    df["S_RSI14"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["S_MA20"] = df["Close"].rolling(MA_PERIOD).mean()
    df["S_VOL20"] = df["Volume"].rolling(VOL_PERIOD).mean()
    df["S_OBV"] = compute_obv(df["Close"], df["Volume"])

    # Bank Nifty Features
    df["BN_Return"] = df["BN_Close"].pct_change()
    
    # Note: For BN_ATR, we need High/Low from the index dataframe
    # We must compute it on the original 'index' df before it loses columns in the join
    index_atr = compute_atr(index, ATR_PERIOD)
    
    # Align the computed ATR with the main df
    df = df.join(index_atr.rename("BN_ATR14"), how="left")

    df["BN_Direction"] = np.where(df["BN_Return"] > 0, 1, -1)

    df = df.dropna()

    if df.empty:
        print("⚠️ Not enough data to calculate indicators (df is empty after dropna).")
        return

    latest_row = df.iloc[-1:].copy()
    latest_row.reset_index(inplace=True)
    latest_row.rename(columns={"index": "Date"}, inplace=True) 
    # yfinance index name is usually "Date" already, but safe to standardize

    # =================================================
    # APPEND TO DATASET
    # =================================================
    if not os.path.exists(DATA_FILE):
        print(f" {DATA_FILE} not found. Creating new file...")
        # Create directory if missing
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        latest_row["Gap_Success"] = np.nan
        latest_row.to_csv(DATA_FILE, index=False)
        print(f"Created new file with row for: {latest_row['Date'].iloc[0]}")
        return

    existing = pd.read_csv(DATA_FILE)
    existing["Date"] = pd.to_datetime(existing["Date"])

    # Check for duplicates
    last_date = latest_row["Date"].iloc[0]
    if last_date in existing["Date"].values:
        print(f"ℹ Data already exists for {last_date.date()}. Skipping.")
        return

    latest_row["Gap_Success"] = np.nan  # unknown (pseudo-live)

    updated = pd.concat([existing, latest_row], ignore_index=True)
    updated.to_csv(DATA_FILE, index=False)

    print(f" Added new pseudo-live row for date: {last_date.date()}")

# =====================================================
if __name__ == "__main__":
    main()