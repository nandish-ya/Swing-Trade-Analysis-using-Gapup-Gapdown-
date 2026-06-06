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

INITIAL_CAPITAL = 1_00_000   # 10 Lakhs
RISK_PER_TRADE = 0.01         # 1% per trade

TP_ATR = 0.5
SL_ATR = 0.25

LOGREG_PROB_THRESHOLD = 0.60

FEATURES = [
    "Gap_pct", "Prev_Close",
    "S_ATR14", "S_RSI14", "S_MA20", "S_VOL20", "S_OBV",
    "BN_Return", "BN_ATR14", "BN_Direction"
]

os.makedirs("reports", exist_ok=True)

# =====================================================
# LOAD MODELS & SCALERS
# =====================================================
def load_models(direction: str):

    models = {
        "xgb": joblib.load(f"{MODEL_DIR}/gap_{direction}_xgb.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/gap_{direction}_rf.pkl"),
        "svm": joblib.load(f"{MODEL_DIR}/gap_{direction}_svm.pkl"),
        "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_logreg.pkl"),
    }

    scalers = {
        "xgb": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_xgb.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_rf.pkl"),
        "svm": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_svm.pkl"),
        "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_logreg.pkl"),
    }

    return models, scalers

# =====================================================
# BACKTEST ENGINE
# =====================================================
def run_backtest():

    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    capital = INITIAL_CAPITAL
    equity_curve = []
    equity_dates = []
    trades = []

    models_long, scalers_long = load_models("long")
    models_short, scalers_short = load_models("short")

    for _, row in df.iterrows():

        direction = "long" if row["Gap_pct"] > 0 else "short"
        models = models_long if direction == "long" else models_short
        scalers = scalers_long if direction == "long" else scalers_short

        X = row[FEATURES].values.reshape(1, -1)

        # --- Scale separately per model ---
        X_xgb = scalers["xgb"].transform(X)
        X_rf  = scalers["rf"].transform(X)
        X_svm = scalers["svm"].transform(X)
        X_lr  = scalers["logreg"].transform(X)

        # --- Predictions ---
        xgb_pred = models["xgb"].predict(X_xgb)[0]
        rf_pred  = models["rf"].predict(X_rf)[0]
        logreg_prob = models["logreg"].predict_proba(X_lr)[0][1]

        # =================================================
        # ENSEMBLE DECISION RULE
        # =================================================
        trade_signal = (
            xgb_pred == 1
            and rf_pred == 1
            and logreg_prob >= LOGREG_PROB_THRESHOLD
        )

        if not trade_signal:
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        # =================================================
        # TRADE EXECUTION
        # =================================================
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

        risk_amount = capital * RISK_PER_TRADE
        position_size = risk_amount / abs(entry - stop)

        pnl = 0
        if hit_target and not hit_stop:
            pnl = position_size * abs(target - entry)
        elif hit_stop:
            pnl = -risk_amount

        capital += pnl
        equity_curve.append(capital)
        equity_dates.append(row["Date"])

        trades.append({
            "Date": row["Date"],
            "Direction": direction,
            "Entry": entry,
            "PnL": pnl,
            "Capital": capital
        })

    equity_series = pd.Series(equity_curve, index=equity_dates)
    trades_df = pd.DataFrame(trades)

    return trades_df, equity_series

# =====================================================
# PERFORMANCE METRICS + SHARPE
# =====================================================
def evaluate(trades: pd.DataFrame, equity: pd.Series):

    if trades.empty:
        print("No trades executed.")
        return

    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    win_rate = (trades["PnL"] > 0).mean() * 100

    profit_factor = (
        trades[trades["PnL"] > 0]["PnL"].sum()
        / abs(trades[trades["PnL"] < 0]["PnL"].sum())
    )

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min() * 100

    daily_returns = equity.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() != 0 else 0.0
    )

    print("\n================ BACKTEST RESULTS ================")
    print(f"Total Trades     : {len(trades)}")
    print(f"Win Rate         : {win_rate:.2f}%")
    print(f"Total Return     : {total_return:.2f}%")
    print(f"Profit Factor    : {profit_factor:.2f}")
    print(f"Max Drawdown     : {max_dd:.2f}%")
    print(f"Sharpe Ratio     : {sharpe:.2f}")
    print("=================================================")

# =====================================================
# EQUITY CURVE PLOT
# =====================================================
def plot_equity_curve(equity: pd.Series):

    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve – Ensemble Gap Trading Strategy")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.grid(True)

    save_path = "reports/equity_curve.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Equity curve saved to {save_path}")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    trades, equity = run_backtest()
    evaluate(trades, equity)
    plot_equity_curve(equity)
