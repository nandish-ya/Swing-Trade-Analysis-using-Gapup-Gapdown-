import yfinance as yf
import pandas as pd
import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# ================= CONFIG =================
SYMBOL = "CANBK.NS"
BN_SYMBOL = "^NSEBANK"

DATA_FILE = r"D:\swing_proj\src\data\labeled.csv"

# =========================================
def fetch_next_trading_day():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("labeled.csv not found")

    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    last_date = df["Date"].iloc[-1]
    next_date = last_date + timedelta(days=1)

    while True:
        # Skip weekends
        if next_date.weekday() >= 5:
            next_date += timedelta(days=1)
            continue

        # Fetch stock
        stock = yf.download(
            SYMBOL,
            start=next_date.strftime("%Y-%m-%d"),
            end=(next_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False
        )

        # Fetch BankNifty
        bn = yf.download(
            BN_SYMBOL,
            start=next_date.strftime("%Y-%m-%d"),
            end=(next_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False
        )

        # If either is missing → holiday → skip
        if stock.empty or bn.empty:
            next_date += timedelta(days=1)
            continue

        break  # ✅ Valid trading day found

    # Build new row
    new_row = {
        "Date": next_date,
        "Open": stock["Open"].iloc[0],
        "High": stock["High"].iloc[0],
        "Low": stock["Low"].iloc[0],
        "Close": stock["Close"].iloc[0],
        "Volume": stock["Volume"].iloc[0],
        "BN_Close": bn["Close"].iloc[0],
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    print(f"Added valid trading day: {next_date.date()}")

# ================= RUN ====================
if __name__ == "__main__":
    fetch_next_trading_day()
