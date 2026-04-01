import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings

# Suppress minor formatting warnings
warnings.filterwarnings('ignore')

print("=== Codecure ML Training Pipeline ===")

# 1. Load Data
try:
    df = pd.read_csv("data/Cleaned_Bacteria_Dataset.csv")
except FileNotFoundError:
    print("Error: Cleaned dataset not found. Please run src/clean.py first.")
    exit(1)

# ---------------------------------------------------------
# REMOVED "SOUCHES" PERMANENTLY FOR CLINICAL USABILITY
# ---------------------------------------------------------
# Input Features (6) - Order strictly enforced!
feature_cols = ['Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq', 'Age', 'Gender']

# Target Columns (15)
target_cols = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
               'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']

X = df[feature_cols]
y = df[target_cols]

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training Multi-Output Random Forest Model...")
forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(forest)
model.fit(X_train, y_train)

# 4. Model Evaluation on Test Set
print("\n=== Model Evaluation ===")
y_pred = model.predict(X_test)

# Exact match strict accuracy across all 15 columns
# Note: Exact Match Accuracy is highly strict: it requires predicting all 15 drugs completely perfectly for a given patient.
exact_accuracy = (y_test.values == y_pred).all(axis=1).mean()
print(f"Exact Match Accuracy (All 15 drugs flawlessly predicted): {exact_accuracy:.2%}")

# Simple aggregate accuracy per column over the 15 drugs
# Note: Average accuracy is more practical for multi-output forecasting since it evaluates the per-drug hit rate.
col_accuracies = []
for i, col in enumerate(target_cols):
    col_accuracies.append(accuracy_score(y_test.iloc[:, i], y_pred[:, i]))
print(f"Average Accuracy across targets: {np.mean(col_accuracies):.2%}")

# 5. Safe Testing Example
print("\n=== Safe Testing Example ===")
# Testing a realistic baseline patient profile
# Order MUST EXACTLY match feature_cols: 
# ['Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq', 'Age', 'Gender']
# E.g., 45-year old Male (1), Diabetic (1), Hypertensive (1), No Hospital (0), 2 past infections.
sample_patient = [[1, 1, 0, 2, 45, 1]]
sample_pred = model.predict(sample_patient)[0]

print(f"Input features:   {sample_patient[0]}")
print(f"Prediction Array: {sample_pred.tolist()} (0=Resistant, 1=Intermediate, 2=Susceptible)")

# 6. Save Distributed Model and Feature Order
os.makedirs("models", exist_ok=True)
model_path = "models/model.pkl"
features_path = "models/feature_cols.pkl"

joblib.dump(model, model_path)
joblib.dump(feature_cols, features_path) # Save exact column order to prevent Flask input mismatch

print(f"\nModel successfully saved to {model_path}!")
print(f"Feature columns structurally secured at {features_path}!")
print("The AI is now ready to be deployed.")