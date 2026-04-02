import pandas as pd
import numpy as np
import os

print("=== Cleaning Dataset 2 ===")

# Load raw dataset 2
df = pd.read_csv("data/d2.csv", low_memory=False)

# Antibiotic columns
target_cols = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM',
    'GEN', 'AN', 'Acide nalidixique', 'ofx', 'CIP',
    'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

# Keep only relevant columns
df = df[target_cols].copy()

# Standardize text
for col in target_cols:
    df[col] = df[col].astype(str).str.upper().str.strip()
    df[col] = df[col].replace('INTERMEDIATE', 'I')

# Replace garbage values
df = df.replace(['nan', 'None', '?', 'NA', '', 'error'], np.nan)

# Drop rows with too many missing values
df = df.dropna(thresh=len(target_cols) // 2)

# Convert S/I/R → numeric
mapping = {'S': 2, 'I': 1, 'R': 0}

for col in target_cols:
    df[col] = df[col].map(mapping)

# Fill remaining missing values safely
df = df.fillna(0)

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/Cleaned_Dataset2.csv", index=False)

print("✅ Dataset 2 cleaned and saved as Cleaned_Dataset2.csv")
print(f"Final shape: {df.shape}")