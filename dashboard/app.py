from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "..", "src", "data", "labeled.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "src", "models")
LOG_FILE = os.path.join(BASE_DIR, "..", "reports", "prediction_log.csv")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
FEATURES = [
    "Gap_pct", "Prev_Close",
    "S_ATR14", "S_RSI14", "S_MA20", "S_VOL20", "S_OBV",
    "BN_Return", "BN_ATR14", "BN_Direction"
]

TP_ATR = 0.5
SL_ATR = 0.25
PROB_THRESHOLD = 0.55

# =====================================================
# LOAD MODELS
# =====================================================
def load_models(direction):
    return (
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_xgb.pkl")),
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_rf.pkl")),
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_logreg.pkl")),
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_scaler_xgb.pkl")),
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_scaler_rf.pkl")),
        joblib.load(os.path.join(MODEL_DIR, f"gap_{direction}_scaler_logreg.pkl")),
    )

# =====================================================
# SIGNAL STRENGTH
# =====================================================
def signal_strength(prob):
    if prob >= 0.75:
        return "STRONG BUY"
    elif prob >= 0.65:
        return "BUY"
    elif prob >= 0.55:
        return "WEAK BUY"
    else:
        return "NO TRADE"

# =====================================================
# MAIN ROUTE (STRICT PSEUDO-LIVE)
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():

    # -------- Load Data --------
    if not os.path.exists(DATA_FILE):
        return f"Data file not found: {DATA_FILE}"

    df = pd.read_csv(DATA_FILE, on_bad_lines="skip")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 🔥 VALID PSEUDO-LIVE DATA ONLY
    valid_df = df.dropna(subset=FEATURES).copy()

    if valid_df.empty:
        return "No valid pseudo-live data available."

    # -------- Date Selection (VALID ONLY) --------
    selected_date = request.form.get("date")

    if selected_date:
        mask = valid_df["Date"] == pd.to_datetime(selected_date)
        if not mask.any():
            return f"No valid pseudo-live data for {selected_date}"
        row = valid_df[mask].iloc[0]
    else:
        # Default = last valid pseudo-live day
        row = valid_df.iloc[-1]

    selected_date = row["Date"]

    # -------- Direction --------
    direction = "long" if row["Gap_pct"] > 0 else "short"

    # -------- Load Models --------
    try:
        xgb, rf, lr, sc_xgb, sc_rf, sc_lr = load_models(direction)
    except FileNotFoundError as e:
        return f"Model loading error: {e}"

    # -------- Feature Prep (NUMERIC SAFE) --------
    X = (
        pd.to_numeric(row[FEATURES], errors="coerce")
        .values
        .reshape(1, -1)
    )

    # -------- Scaling --------
    X_xgb = sc_xgb.transform(X)
    X_rf = sc_rf.transform(X)
    X_lr = sc_lr.transform(X)

    # -------- Predictions --------
    xgb_raw = int(xgb.predict(X_xgb)[0])
    rf_raw = int(rf.predict(X_rf)[0])
    prob = float(lr.predict_proba(X_lr)[0][1])

    xgb_vote = "BUY" if xgb_raw == 1 else "NO"
    rf_vote = "BUY" if rf_raw == 1 else "NO"

    final_signal = signal_strength(prob)
    trade_allowed = (xgb_raw == 1) and (rf_raw == 1) and (prob >= PROB_THRESHOLD)

    # -------- Trade Levels --------
    entry = row["S_Open"]
    atr = row["S_ATR14"]

    if direction == "long":
        target = entry + TP_ATR * atr
        stop = entry - SL_ATR * atr
    else:
        target = entry - TP_ATR * atr
        stop = entry + SL_ATR * atr

    # -------- Logging --------
    log_row = pd.DataFrame([{
        "Date": selected_date.date(),
        "Direction": direction,
        "XGB": xgb_vote,
        "RF": rf_vote,
        "Prob": round(prob, 3),
        "Signal": final_signal
    }])

    log_row.to_csv(
        LOG_FILE,
        mode="a",
        header=not os.path.exists(LOG_FILE),
        index=False
    )

    history = []
    if os.path.exists(LOG_FILE):
        history = (
            pd.read_csv(LOG_FILE, on_bad_lines="skip")
            .tail(10)
            .iloc[::-1]
            .to_dict("records")
        )

    # -------- Render UI --------
    return render_template(
        "index.html",
        dates=valid_df["Date"].dt.date.unique(),
        selected=selected_date.date(),
        direction=direction.upper(),
        xgb=xgb_vote,
        rf=rf_vote,
        prob=round(prob, 2),
        prob_pct=int(prob * 100),
        signal=final_signal,
        trade="YES" if trade_allowed else "NO",
        entry=round(entry, 2),
        target=round(target, 2),
        stop=round(stop, 2),
        history=history,
        last_valid_date=valid_df["Date"].max().date()
    )

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
