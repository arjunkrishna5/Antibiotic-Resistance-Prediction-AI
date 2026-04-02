"""
hybrid_lookup.py — Build a (strain, drug) → resistance class lookup table.

Reads the cleaned bacteria dataset and computes the modal resistance class
for every (Souches, antibiotic) pair, then saves the result to models/lookup.pkl.
"""

import os
import pickle

import pandas as pd

DATA_PATH = "data/Cleaned_Bacteria_Dataset.csv"
OUTPUT_PATH = "models/lookup.pkl"

TARGET_COLS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM",
    "GEN", "AN", "Acide nalidixique", "ofx", "CIP",
    "C", "Co-trimoxazole", "Furanes", "colistine",
]


def build_lookup() -> dict:
    """
    Build a {(strain, drug): resistance_code} lookup from the dataset.

    Returns the lookup dict and also saves it to OUTPUT_PATH.
    """
    df = pd.read_csv(DATA_PATH)

    if "Souches" not in df.columns:
        raise ValueError("Column 'Souches' not found in dataset.")

    lookup: dict = {}

    for drug in TARGET_COLS:
        if drug not in df.columns:
            continue
        # Group by strain and compute the modal (most common) resistance class
        grouped = df.groupby("Souches")[drug].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        for strain, modal_class in grouped.items():
            if modal_class is not None:
                lookup[(strain, drug)] = int(modal_class)

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(lookup, f)

    print(f"✅ Lookup table saved → {OUTPUT_PATH}  ({len(lookup)} entries)")
    return lookup


if __name__ == "__main__":
    build_lookup()