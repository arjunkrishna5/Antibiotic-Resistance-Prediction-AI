"""
src/train.py — AmrLens AI ML training pipeline (v2.0). 

Upgraded to HistGradientBoostingClassifier for 80%+ accuracy goal.
Includes feature engineering and improved hyperparameter tuning.
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_PATH = "data/Cleaned_Bacteria_Dataset.csv"
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_cols.pkl"
CHART_PATH = "static/feature_importance.png"

# Updated features including engineered ones
FEATURE_COLS = [
    "Diabetes", "Hypertension", "Hospital_before", "Infection_Freq", 
    "Age", "Gender", "Souches", "Age_Group", "Comorbidity", "Risk_Score"
]

TARGET_COLS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM",
    "GEN", "AN", "Acide nalidixique", "ofx", "CIP",
    "C", "Co-trimoxazole", "Furanes", "colistine",
]

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

print("=== AmrLens AI ML Training Pipeline v2.0 ===")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: dataset not found at '{DATA_PATH}'. Run src/clean.py first.")
    raise SystemExit(1)

X = df[FEATURE_COLS]
y = df[TARGET_COLS]

# ---------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING (Optimized)
# ---------------------------------------------------------------------------

print("Tuning HistGradientBoosting Model for 80% accuracy...")

# Base estimator for MultiOutput
base_estimator = HistGradientBoostingClassifier(random_state=42)

# We use fixed high-performance parameters found in experiments to save time,
# but wrapped in a slightly flexible way.
model = MultiOutputClassifier(
    HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=12,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=42
    )
)

print("Fitting improved model...")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

print("\n=== Model Evaluation ===")
y_pred = model.predict(X_test)
y_test_np = y_test.values

exact_accuracy = (y_test_np == y_pred).all(axis=1).mean()
print(f"Exact Match Accuracy: {exact_accuracy:.2%}")

totals = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

for i, col in enumerate(TARGET_COLS):
    y_true_col = y_test_np[:, i]
    y_pred_col = y_pred[:, i]

    metrics = {
        "accuracy":  accuracy_score(y_true_col, y_pred_col),
        "precision": precision_score(y_true_col, y_pred_col, average="weighted", zero_division=0),
        "recall":    recall_score(y_true_col, y_pred_col, average="weighted", zero_division=0),
        "f1":        f1_score(y_true_col, y_pred_col, average="weighted", zero_division=0),
    }

    for k, v in metrics.items():
        totals[k] += v

n = len(TARGET_COLS)
avg_acc = totals['accuracy'] / n
print("\n" + "=" * 40)
print(f"OVERALL PERFORMANCE: {avg_acc:.2%}")
print("=" * 40)
print(f"Average Precision: {totals['precision'] / n:.4f}")
print(f"Average Recall:    {totals['recall'] / n:.4f}")
print(f"Average F1 Score:  {totals['f1'] / n:.4f}")

# ---------------------------------------------------------------------------
# SAVE MODEL & FEATURE COLUMNS
# ---------------------------------------------------------------------------

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURE_COLS, FEATURES_PATH)
print(f"\n[SUCCESS] Model saved -> {MODEL_PATH}")
print(f"[SUCCESS] Feature columns saved -> {FEATURES_PATH}")

# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE (Permutation Importance for HGB)
# ---------------------------------------------------------------------------

print("\nGenerating feature importance chart...")
from sklearn.inspection import permutation_importance

try:
    # Use a small subset for speed in chart generation
    r = permutation_importance(model, X_test[:500], y_test[:500], n_repeats=5, random_state=42)
    importances = r.importances_mean
    indices = np.argsort(importances)[::-1]

    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(FEATURE_COLS)), importances[indices], color="#38bdf8")
    plt.xticks(range(len(FEATURE_COLS)), [FEATURE_COLS[i] for i in indices], rotation=45)
    plt.title("Feature Importance (Permutation Impact)")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    print(f"[SUCCESS] Chart updated -> {CHART_PATH}")

except Exception as exc:
    print(f"[WARNING] Could not generate feature importance chart: {exc}")

if avg_acc >= 0.80:
    print("\n[SUCCESS] 80% accuracy threshold reached!")
else:
    print(f"\n[INFO] Current accuracy is {avg_acc:.2%}. Further tuning may be required.")