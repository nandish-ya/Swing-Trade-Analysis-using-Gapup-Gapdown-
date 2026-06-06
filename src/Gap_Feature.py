import pandas as pd
import numpy as np
import os

# =====================================================
# AUTO-CREATE DATA FOLDER
# =====================================================
os.makedirs("data", exist_ok=True)

# =====================================================
# INDICATOR FUNCTIONS
# =====================================================
def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

# =====================================================
# LOAD & MERGE DATA
# =====================================================
def load_data():
    stock_path = "D:/swing_proj/CANBK.csv"
    bn_path = "D:/swing_proj/BN.csv"

    stock = pd.read_csv(stock_path)
    bn = pd.read_csv(bn_path)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in stock.columns:
            stock[col] = pd.to_numeric(stock[col], errors="coerce")
        if col in bn.columns:
            bn[col] = pd.to_numeric(bn[col], errors="coerce")

    stock["Date"] = pd.to_datetime(stock["Date"])
    bn["Date"] = pd.to_datetime(bn["Date"])

    stock.rename(columns={
        "Open": "S_Open", "High": "S_High", "Low": "S_Low",
        "Close": "S_Close", "Volume": "S_Volume"
    }, inplace=True)

    bn.rename(columns={
        "Open": "BN_Open", "High": "BN_High", "Low": "BN_Low",
        "Close": "BN_Close", "Volume": "BN_Volume"
    }, inplace=True)

    df = pd.merge(stock, bn, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)

    return df

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def add_features(df):

    # ---- Stock Features ----
    df["S_ATR14"] = compute_atr(df["S_High"], df["S_Low"], df["S_Close"])
    df["S_RSI14"] = compute_rsi(df["S_Close"])
    df["S_MA20"] = df["S_Close"].rolling(20).mean()
    df["S_VOL20"] = df["S_Volume"].rolling(20).mean()
    df["S_OBV"] = compute_obv(df["S_Close"], df["S_Volume"])

    # ---- BankNifty Features ----
    df["BN_Return"] = df["BN_Close"].pct_change()
    df["BN_ATR14"] = compute_atr(df["BN_High"], df["BN_Low"], df["BN_Close"])
    df["BN_Direction"] = np.sign(df["BN_Return"]).fillna(0)

    # ---- Gap Features ----
    df["Prev_Close"] = df["S_Close"].shift(1)
    df["Gap_pct"] = (df["S_Open"] - df["Prev_Close"]) / df["Prev_Close"]

    # Remove rows with incomplete indicators
    df = df.dropna().reset_index(drop=True)

    return df

# =====================================================
# GENERATE SINGLE FEATURE FILE
# =====================================================
def generate_feature_file():
    df = load_data()
    df = add_features(df)

    df.to_csv("data/CANBK_features.csv", index=False)

    print("Feature file created: data/CANBK_features.csv")
    print(f"Total rows: {len(df)}")
    print(df.head())

    return df

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    generate_feature_file()
