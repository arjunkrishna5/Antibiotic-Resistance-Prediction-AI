"""
pattern_model.py — Train per-antibiotic cross-resistance pattern models.

For each antibiotic, a Random Forest is trained using all *other* antibiotics
as features. This captures cross-resistance patterns not encoded in patient
demographics alone.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/Cleaned_Dataset2.csv"
OUTPUT_PATH = "models/pattern_models.pkl"

TARGET_COLS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM",
    "GEN", "AN", "Acide nalidixique", "ofx", "CIP",
    "C", "Co-trimoxazole", "Furanes", "colistine",
]


def train_pattern_models() -> dict:
    """
    Train one RandomForestClassifier per antibiotic, using peer antibiotics as features.

    Returns a dict of {drug: fitted_model}.
    """
    df = pd.read_csv(DATA_PATH)
    models = {}

    print("=== Training Pattern Models (Dataset 2) ===")

    for target in TARGET_COLS:
        feature_cols = [col for col in TARGET_COLS if col != target]
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  {target:<25} accuracy: {acc:.4f}")

        models[target] = clf

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(models, f)

    print(f"\n✅ Pattern models saved → {OUTPUT_PATH}")
    return models


if __name__ == "__main__":
    train_pattern_models()