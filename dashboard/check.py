import pandas as pd
import joblib
import os

# ==============================
# PATHS
# ==============================
DATA_FILE = r"D:\swing_proj\src\data\labeled.csv"
MODEL_DIR = r"D:\swing_proj\src\models"

FEATURES = [
    "Gap_pct", "Prev_Close",
    "S_ATR14", "S_RSI14", "S_MA20", "S_VOL20", "S_OBV",
    "BN_Return", "BN_ATR14", "BN_Direction"
]

PROB_THRESHOLD = 0.60

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ==============================
# LOAD MODELS (LONG ONLY here)
# ==============================
xgb = joblib.load(os.path.join(MODEL_DIR, "gap_long_xgb.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "gap_long_rf.pkl"))
lr = joblib.load(os.path.join(MODEL_DIR, "gap_long_logreg.pkl"))

sc_xgb = joblib.load(os.path.join(MODEL_DIR, "gap_long_scaler_xgb.pkl"))
sc_rf = joblib.load(os.path.join(MODEL_DIR, "gap_long_scaler_rf.pkl"))
sc_lr = joblib.load(os.path.join(MODEL_DIR, "gap_long_scaler_logreg.pkl"))

# ==============================
# GENERATE PREDICTIONS
# ==============================
X = df[FEATURES]

df["XGB_Pred"] = xgb.predict(sc_xgb.transform(X))
df["RF_Pred"] = rf.predict(sc_rf.transform(X))
df["LR_Prob"] = lr.predict_proba(sc_lr.transform(X))[:, 1]

# ==============================
# STRONG SIGNAL DAYS
# ==============================
strong_days = df[
    (df["XGB_Pred"] == 1) &
    (df["RF_Pred"] == 1) &
    (df["LR_Prob"] >= PROB_THRESHOLD)
]

# ==============================
# OUTPUT
# ==============================
print("STRONG SIGNAL DAYS ")
print(strong_days[["Date", "Gap_pct", "LR_Prob"]].tail(10))

print(f"\nTotal Strong Signal Days: {len(strong_days)}")
