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

INITIAL_CAPITAL = 1_000_000        # 10 Lakhs
RISK_PER_TRADE  = 0.01             # 1% risk per trade
BROKERAGE       = 0.0005           # 0.05% per trade

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
def load_models(direction: str):
    """
    Load all trained models and their corresponding scalers
    for the given trade direction ('long' or 'short').
    """
    try:
        models = {
            "xgb":    joblib.load(f"{MODEL_DIR}/gap_{direction}_xgb.pkl"),
            "rf":     joblib.load(f"{MODEL_DIR}/gap_{direction}_rf.pkl"),
            "svm":    joblib.load(f"{MODEL_DIR}/gap_{direction}_svm.pkl"),
            "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_logreg.pkl"),
        }
        scalers = {
            "xgb":    joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_xgb.pkl"),
            "rf":     joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_rf.pkl"),
            "svm":    joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_svm.pkl"),
            "logreg": joblib.load(f"{MODEL_DIR}/gap_{direction}_scaler_logreg.pkl"),
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model file not found for direction='{direction}'. "
            f"Make sure all .pkl files exist in '{MODEL_DIR}/'. Error: {e}"
        )

    return models, scalers


# =====================================================
# VALIDATE FEATURES
# =====================================================
def has_valid_features(row: pd.Series) -> bool:
    """
    Returns True only if all required feature columns
    are present and contain no NaN or Inf values.
    """
    values = row[FEATURES].values.astype(float)

    if np.isnan(values).any():
        return False

    if np.isinf(values).any():
        return False

    return True


# =====================================================
# ENSEMBLE SIGNAL
# =====================================================
def get_trade_signal(row: pd.Series, models: dict, scalers: dict) -> bool:
    """
    Returns True if XGBoost, RandomForest both predict success
    AND LogisticRegression probability >= threshold.
    SVM is used as an optional confirmation (soft filter).
    """
    X = row[FEATURES].values.reshape(1, -1).astype(float)

    X_xgb    = scalers["xgb"].transform(X)
    X_rf     = scalers["rf"].transform(X)
    X_svm    = scalers["svm"].transform(X)
    X_lr     = scalers["logreg"].transform(X)

    xgb_pred    = models["xgb"].predict(X_xgb)[0]
    rf_pred     = models["rf"].predict(X_rf)[0]
    svm_pred    = models["svm"].predict(X_svm)[0]
    logreg_prob = models["logreg"].predict_proba(X_lr)[0][1]

    # Core signal: XGB + RF must agree, LR must be confident
    core_signal = (
        xgb_pred == 1
        and rf_pred == 1
        and logreg_prob >= LOGREG_PROB_THRESHOLD
    )

    # Optional: SVM as a soft confirmation (comment out if too restrictive)
    # core_signal = core_signal and svm_pred == 1

    return core_signal


# =====================================================
# TRADE OUTCOME
# =====================================================
def calculate_pnl(row: pd.Series, direction: str, capital: float):
    """
    Calculates trade PnL using ATR-based TP and SL.
    Returns (pnl, hit_target, hit_stop).
    """
    entry = row["S_Open"]
    atr   = row["S_ATR14"]

    if atr <= 0 or np.isnan(atr):
        return 0.0, False, False

    if direction == "long":
        target    = entry + TP_ATR * atr
        stop      = entry - SL_ATR * atr
        hit_target = row["S_High"] >= target
        hit_stop   = row["S_Low"]  <= stop
    else:
        target    = entry - TP_ATR * atr
        stop      = entry + SL_ATR * atr
        hit_target = row["S_Low"]  <= target
        hit_stop   = row["S_High"] >= stop

    risk_amount   = capital * RISK_PER_TRADE
    sl_distance   = abs(entry - stop)

    if sl_distance == 0:
        return 0.0, False, False

    position_size = risk_amount / sl_distance

    pnl = 0.0
    if hit_target and not hit_stop:
        # Target hit first → full profit
        pnl = position_size * abs(target - entry)
    elif hit_stop:
        # Stop hit → full loss
        pnl = -risk_amount
    # else: neither hit → flat (no PnL, no trade recorded)

    # Brokerage cost (both entry + exit)
    brokerage_cost = entry * position_size * BROKERAGE * 2
    pnl -= brokerage_cost

    return pnl, hit_target, hit_stop


# =====================================================
# BACKTEST ENGINE
# =====================================================
def run_backtest(df: pd.DataFrame):
    """
    Iterates over all gap events, generates ensemble signals,
    executes trades, and tracks equity.
    """
    capital      = INITIAL_CAPITAL
    equity_curve = []
    equity_dates = []
    trades       = []

    skipped_nan  = 0
    skipped_atr  = 0
    no_signal    = 0

    models_long,  scalers_long  = load_models("long")
    models_short, scalers_short = load_models("short")

    print(f"\nRunning backtest on {len(df)} rows...")
    print("-" * 50)

    for _, row in df.iterrows():

        # ── 1. Skip rows with missing/invalid features ──────
        if not has_valid_features(row):
            skipped_nan += 1
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        # ── 2. Determine direction ───────────────────────────
        direction = "long" if row["Gap_pct"] > 0 else "short"
        models    = models_long  if direction == "long" else models_short
        scalers   = scalers_long if direction == "long" else scalers_short

        # ── 3. Get ensemble trade signal ─────────────────────
        try:
            trade_signal = get_trade_signal(row, models, scalers)
        except Exception as e:
            print(f"  [WARNING] Signal error on {row['Date']}: {e}")
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        if not trade_signal:
            no_signal += 1
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        # ── 4. Calculate PnL ─────────────────────────────────
        pnl, hit_target, hit_stop = calculate_pnl(row, direction, capital)

        if not hit_target and not hit_stop:
            # Neither level hit — skip, no trade
            skipped_atr += 1
            equity_curve.append(capital)
            equity_dates.append(row["Date"])
            continue

        capital += pnl

        trades.append({
            "Date":       row["Date"],
            "Direction":  direction,
            "Entry":      row["S_Open"],
            "ATR":        row["S_ATR14"],
            "Target":     row["S_Open"] + TP_ATR * row["S_ATR14"] if direction == "long"
                          else row["S_Open"] - TP_ATR * row["S_ATR14"],
            "Stop":       row["S_Open"] - SL_ATR * row["S_ATR14"] if direction == "long"
                          else row["S_Open"] + SL_ATR * row["S_ATR14"],
            "Hit_Target": hit_target,
            "Hit_Stop":   hit_stop,
            "PnL":        round(pnl, 2),
            "Capital":    round(capital, 2),
        })

        equity_curve.append(capital)
        equity_dates.append(row["Date"])

    # ── Build results ────────────────────────────────────────
    equity    = pd.Series(equity_curve, index=equity_dates)
    trades_df = pd.DataFrame(trades)

    # Save trade log
    trades_df.to_csv(f"{REPORT_DIR}/trades.csv", index=False)

    print(f"  Rows skipped (NaN/Inf features) : {skipped_nan}")
    print(f"  Rows skipped (no TP/SL hit)     : {skipped_atr}")
    print(f"  Rows with no trade signal       : {no_signal}")
    print(f"  Total trades executed           : {len(trades_df)}")

    return trades_df, equity


# =====================================================
# PERFORMANCE METRICS
# =====================================================
def evaluate(trades: pd.DataFrame, equity: pd.Series):
    """
    Prints a full performance summary of the backtest.
    """
    if trades.empty:
        print("\n[WARNING] No trades were executed. Check your data and models.")
        return

    total_return  = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    win_rate      = (trades["PnL"] > 0).mean() * 100

    wins          = trades[trades["PnL"] > 0]["PnL"].sum()
    losses        = abs(trades[trades["PnL"] < 0]["PnL"].sum())
    profit_factor = wins / losses if losses > 0 else float("inf")

    drawdown      = equity / equity.cummax() - 1
    max_dd        = drawdown.min() * 100

    daily_returns = equity.pct_change().dropna()
    sharpe        = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else 0.0
    )

    long_trades   = trades[trades["Direction"] == "long"]
    short_trades  = trades[trades["Direction"] == "short"]

    print("\n" + "=" * 50)
    print("         BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Initial Capital   : {INITIAL_CAPITAL:,.0f}")
    print(f"  Final Capital     : {equity.iloc[-1]:,.2f}")
    print(f"  Total Return      : {total_return:.2f}%")
    print("-" * 50)
    print(f"  Total Trades      : {len(trades)}")
    print(f"    Long  Trades    : {len(long_trades)}")
    print(f"    Short Trades    : {len(short_trades)}")
    print(f"  Win Rate          : {win_rate:.2f}%")
    print(f"  Profit Factor     : {profit_factor:.2f}")
    print(f"  Max Drawdown      : {max_dd:.2f}%")
    print(f"  Sharpe Ratio      : {sharpe:.2f}")
    print("=" * 50)

    # Save summary to file
    summary = {
        "Initial Capital":  INITIAL_CAPITAL,
        "Final Capital":    round(equity.iloc[-1], 2),
        "Total Return (%)": round(total_return, 2),
        "Total Trades":     len(trades),
        "Long Trades":      len(long_trades),
        "Short Trades":     len(short_trades),
        "Win Rate (%)":     round(win_rate, 2),
        "Profit Factor":    round(profit_factor, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Sharpe Ratio":     round(sharpe, 2),
    }
    pd.DataFrame([summary]).to_csv(f"{REPORT_DIR}/summary.csv", index=False)
    print(f"\n  Summary saved → {REPORT_DIR}/summary.csv")
    print(f"  Trade log saved → {REPORT_DIR}/trades.csv")


# =====================================================
# EQUITY CURVE PLOT
# =====================================================
def plot_equity_curve(equity: pd.Series, trades: pd.DataFrame):
    """
    Plots the equity curve with win/loss trade markers.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # ── Top: Equity Curve ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, color="#1E2761", linewidth=1.5, label="Portfolio Value")
    ax1.set_title("Equity Curve – Ensemble Gap Trading Strategy", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Capital (₹)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # Mark wins and losses on the curve
    if not trades.empty:
        wins  = trades[trades["PnL"] > 0]
        losses = trades[trades["PnL"] < 0]
        ax1.scatter(wins["Date"],   wins["Capital"],   color="green", s=15, zorder=5, label="Win",  alpha=0.6)
        ax1.scatter(losses["Date"], losses["Capital"], color="red",   s=15, zorder=5, label="Loss", alpha=0.6)

    ax1.legend(fontsize=9)

    # ── Bottom: Drawdown ─────────────────────────────────────
    ax2 = axes[1]
    drawdown = (equity / equity.cummax() - 1) * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    path = f"{REPORT_DIR}/equity_curve.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  Equity curve saved = {path}")


# =====================================================
# MAIN
# =====================================================
def main():

    # ── Load data ────────────────────────────────────────────
    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Total rows loaded : {len(df)}")

    # ── Check required columns ───────────────────────────────
    required_cols = FEATURES + ["Date", "Gap_pct", "S_Open", "S_High", "S_Low"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # ── Drop rows where features have NaN (rolling indicator warmup) ──
    before = len(df)
    df = df.dropna(subset=FEATURES)
    after  = len(df)
    print(f"Dropped {before - after} NaN rows (rolling indicator warmup)")
    print(f"Clean rows for backtest : {after}")

    # ── Also drop rows with Inf values in features ───────────
    df = df[~df[FEATURES].isin([np.inf, -np.inf]).any(axis=1)]
    print(f"Rows after Inf filter   : {len(df)}")

    # ── Run backtest ─────────────────────────────────────────
    trades, equity = run_backtest(df)

    # ── Evaluate ─────────────────────────────────────────────
    evaluate(trades, equity)

    # ── Plot ─────────────────────────────────────────────────
    plot_equity_curve(equity, trades)


if __name__ == "__main__":
    main()