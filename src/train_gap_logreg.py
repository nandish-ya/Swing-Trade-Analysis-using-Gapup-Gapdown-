import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from typing import Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================
DATA_FILE = "data/labeled.csv"
MODEL_DIR = "models"
PLOT_DIR = "plots"  # New directory for images

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Features must match the other ensemble models
FEATURES = [
    "Gap_pct", "Prev_Close",
    "S_ATR14", "S_RSI14", "S_MA20", "S_VOL20", "S_OBV",
    "BN_Return", "BN_ATR14", "BN_Direction"
]

# =====================================================
# LOAD DATA
# =====================================================
def load_data() -> pd.DataFrame:
    """Loads and prepares the labeled data."""
    try:
        df = pd.read_csv(DATA_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}. Please run gap_label_generator.py first.")
        return pd.DataFrame()


# =====================================================
# TRAIN FUNCTION (Logistic Regression)
# =====================================================
def train_logreg_directional(df: pd.DataFrame, direction: str):
    """Trains and evaluates a Logistic Regression model for a given trade direction."""

    print(f"\n================ TRAINING LOGREG ({direction}) ================")

    df_dir = df[df["Trade_Type"] == direction].copy()
    
    if df_dir.empty:
        print(f"Skipping training for {direction}: No data found.")
        return

    X = df_dir[FEATURES]
    y = df_dir["Gap_Success"]

    # Time-based split (80 / 20)
    split = int(0.8 * len(df_dir))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scaling (MANDATORY for Linear Models)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression Model
    # Note: 'liblinear' solver handles optimization iteratively and does not support n_jobs
    model = LogisticRegression(
        penalty='l2',                # L2 regularization
        C=1.0,                       # Regularization strength
        class_weight='balanced',     # Handles imbalance
        solver='liblinear',          # Good for small datasets
        random_state=42
    )
    
    model.fit(X_train_s, y_train)

    # Prediction
    preds = model.predict(X_test_s)
    
    # =====================================================
    # METRICS CALCULATION
    # =====================================================
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # =====================================================
    # OUTPUT PRINTING
    # =====================================================
    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1 Score : {f1}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    print("Confusion Matrix (Text):")
    print(cm)

    # =====================================================
    # PLOTTING CONFUSION MATRIX
    # =====================================================
    plt.figure(figsize=(6, 5))
    
    # Create Heatmap
    # fmt='d' ensures numbers are integers, not scientific notation
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Fail', 'Pred Success'],
                yticklabels=['Actual Fail', 'Actual Success'])
    
    plt.title(f'Confusion Matrix: {direction} (Logistic Regression)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save Plot
    plot_filename = f"cm_{direction.lower()}_logreg.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close() # Close figure to free memory
    
    print(f"Confusion Matrix Image saved = {plot_path}")

    # =====================================================
    # SAVE MODEL
    # =====================================================
    model_path = f"{MODEL_DIR}/gap_{direction.lower()}_logreg.pkl"
    scaler_path = f"{MODEL_DIR}/gap_{direction.lower()}_scaler_logreg.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved   = {model_path}")
    print(f"Scaler saved  = {scaler_path}")


# =====================================================
# RUN
# =====================================================
def main():
    df = load_data()
    if df.empty:
        return

    # Train Long and Short Models (Directional Modeling)
    train_logreg_directional(df, "Long")
    train_logreg_directional(df, "Short")

if __name__ == "__main__":
    main()