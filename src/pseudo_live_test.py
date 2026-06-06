import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = "data/labeled.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

START_YEAR = 2022
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.01
BROKERAGE = 0.0005

TP_ATR = 0.5
SL_ATR = 0.25
LOGREG_PROB_THRESHOLD = 0.60

FEATURES = [
    "Gap_pct", "Prev_Close",
    "S_ATR14", "S_RSI14", "S_MA20", "S_VOL20", "S_OBV",
    "BN_Return", "BN_ATR14", "BN_Direction"
]

os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# LOAD MODELS & SCALERS
# =====================================================
def load_models(direction):

    models = {
        "xgb": joblib.load(f"{MODEL_DIR}/gap_{direction}_xgb.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/gap_{direction}_rf.pkl"),
        "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_logreg.pkl"),
    }

    scalers = {
        "xgb": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_xgb.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_rf.pkl"),
        "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_logreg.pkl"),
    }

    return models, scalers

# =====================================================
# PSEUDO-LIVE WALK-FORWARD SIMULATION
# =====================================================
def run_pseudo_live(df):

    capital = INITIAL_CAPITAL
    equity_curve = []
    equity_dates = []
    trades = []

    models_long, scalers_long = load_models("long")
    models_short, scalers_short = load_models("short")

    for _, row in df.iterrows():

        # -------------------------------
        # Decide trade direction
        # -------------------------------
        direction = "long" if row["Gap_pct"] > 0 else "short"
        models = models_long if direction == "long" else models_short
        scalers = scalers_long if direction == "long" else scalers_short

       # -------------------------------
        # Feature vector (NUMERIC SAFE)
        # -------------------------------
        X = (
        pd.to_numeric(row[FEATURES], errors="coerce")
        .values
        .reshape(1, -1)
)

        # 🚨 NaN SAFETY CHECK
        if np.isnan(X).any():
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        # -------------------------------
        # Scaling
        # -------------------------------
        X_xgb = scalers["xgb"].transform(X)
        X_rf = scalers["rf"].transform(X)
        X_lr = scalers["logreg"].transform(X)

        # -------------------------------
        # Model predictions
        # -------------------------------
        xgb_pred = models["xgb"].predict(X_xgb)[0]
        rf_pred = models["rf"].predict(X_rf)[0]
        logreg_prob = models["logreg"].predict_proba(X_lr)[0][1]

        # -------------------------------
        # Ensemble decision
        # -------------------------------
        trade_signal = (
            xgb_pred == 1 and
            rf_pred == 1 and
            logreg_prob >= LOGREG_PROB_THRESHOLD
        )

        if not trade_signal:
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        # -------------------------------
        # Trade execution
        # -------------------------------
        entry = row["S_Open"]
        atr = row["S_ATR14"]

        if direction == "long":
            target = entry + TP_ATR * atr
            stop = entry - SL_ATR * atr
            hit_target = row["S_High"] >= target
            hit_stop = row["S_Low"] <= stop
        else:
            target = entry - TP_ATR * atr
            stop = entry + SL_ATR * atr
            hit_target = row["S_Low"] <= target
            hit_stop = row["S_High"] >= stop

        # -------------------------------
        # Position sizing
        # -------------------------------
        risk_amount = capital * RISK_PER_TRADE
        position_size = risk_amount / abs(entry - stop)

        # -------------------------------
        # PnL calculation
        # -------------------------------
        pnl = 0.0
        if hit_target and not hit_stop:
            pnl = position_size * abs(target - entry)
        elif hit_stop:
            pnl = -risk_amount

        # Brokerage cost
        pnl -= entry * position_size * BROKERAGE
        capital += pnl

        trades.append({
            "Date": row["Date"],
            "Direction": direction,
            "PnL": pnl,
            "Capital": capital
        })

        equity_curve.append(capital)
        equity_dates.append(row["Date"])

    equity = pd.Series(equity_curve, index=equity_dates)
    trades_df = pd.DataFrame(trades)

    trades_df.to_csv(f"{REPORT_DIR}/pseudo_live_trades.csv", index=False)

    return trades_df, equity

# =====================================================
# PERFORMANCE METRICS
# =====================================================
def evaluate(trades, equity):

    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    win_rate = (trades["PnL"] > 0).mean() * 100

    profit_factor = (
        trades[trades["PnL"] > 0]["PnL"].sum()
        / abs(trades[trades["PnL"] < 0]["PnL"].sum())
        if not trades[trades["PnL"] < 0].empty else np.nan
    )

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min() * 100

    daily_returns = equity.pct_change().dropna()
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else 0
    )

    print("\n====== PSEUDO-LIVE WALK-FORWARD RESULTS ======")
    print(f"Trades        : {len(trades)}")
    print(f"Win Rate      : {win_rate:.2f}%")
    print(f"Total Return  : {total_return:.2f}%")
    print(f"Profit Factor : {profit_factor:.2f}")
    print(f"Max Drawdown  : {max_dd:.2f}%")
    print(f"Sharpe Ratio  : {sharpe:.2f}")
    print("============================================")

# =====================================================
# EQUITY CURVE PLOT
# =====================================================
def plot_equity(equity):

    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values)
    plt.title("Pseudo-Live Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.grid(True)
    plt.show()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # PSEUDO-LIVE PERIOD
    df_live = df[df["Date"].dt.year >= START_YEAR].reset_index(drop=True)

    print(f"Pseudo-Live Period: {df_live['Date'].min().date()} = {df_live['Date'].max().date()}")
    print(f"Total Days: {len(df_live)}")

    trades, equity = run_pseudo_live(df_live)
    evaluate(trades, equity)
    plot_equity(equity)
