import pandas as pd
import numpy as np
import os

# =====================================================
# CONFIG (ATR-BASED, INTRADAY-SAFE)
# =====================================================
FEATURE_FILE = "data/CANBK_features.csv"
OUTPUT_FILE = "data/labeled.csv"

# Volatility-adjusted TP / SL (Correct for gap trading)
TP_ATR_MULTIPLE = 0.5    # Target = 0.5 × ATR
SL_ATR_MULTIPLE = 0.25   # Stop   = 0.25 × ATR

# =====================================================
# LOAD FEATURES
# =====================================================
def load_features():
    df = pd.read_csv(FEATURE_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# =====================================================
# GAP SUCCESS LABEL (ATR-BASED, DIRECTION-AWARE)
# =====================================================
def create_gap_labels(df, tp_mult, sl_mult):

    out = df.copy()

    # Trade direction
    out["Trade_Type"] = np.where(out["Gap_pct"] > 0, "Long", "Short")

    entry = out["S_Open"]
    atr = out["S_ATR14"]

    # Target & Stop (absolute price levels)
    target_price = np.where(
        out["Trade_Type"] == "Long",
        entry + atr * tp_mult,
        entry - atr * tp_mult
    )

    stop_price = np.where(
        out["Trade_Type"] == "Long",
        entry - atr * sl_mult,
        entry + atr * sl_mult
    )

    # Hit logic
    hit_target = np.where(
        out["Trade_Type"] == "Long",
        out["S_High"] >= target_price,
        out["S_Low"] <= target_price
    )

    hit_stop = np.where(
        out["Trade_Type"] == "Long",
        out["S_Low"] <= stop_price,
        out["S_High"] >= stop_price
    )

    # Pessimistic tie-breaker
    label = np.where(hit_target & ~hit_stop, 1, 0)
    label = np.where(hit_target & hit_stop, 0, label)

    out["Gap_Success"] = label

    # Clean dataset
    out = out[out["Gap_pct"] != 0]
    out["Gap_Success"] = out["Gap_Success"].astype(int)

    return out

# =====================================================
# RUN
# =====================================================
def main():
    os.makedirs("data", exist_ok=True)

    df = load_features()
    df_labeled = create_gap_labels(df, TP_ATR_MULTIPLE, SL_ATR_MULTIPLE)

    df_labeled.to_csv(OUTPUT_FILE, index=False)

    print(" ATR-based gap labels created")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df_labeled)}")

    print("\nLabel Distribution:")
    print(df_labeled["Gap_Success"].value_counts(normalize=True))

    print("\nBy Trade Type:")
    print(df_labeled.groupby("Trade_Type")["Gap_Success"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
