import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
exact_accuracy = (y_test.values == y_pred).all(axis=1).mean()
print(f"Exact Match Accuracy (All 15 drugs flawlessly predicted): {exact_accuracy:.2%}")

# Convert y_test to a numpy array to easily select columns by index
y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

# Variables to store the sum of metrics to calculate the overall averages
total_accuracy = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0
num_cols = len(target_cols)

print("\n" + "="*40)
print("EVALUATION METRICS PER ANTIBIOTIC")
print("="*40)

# Loop through each target column and calculate evaluating metrics
for i, col in enumerate(target_cols):
    # Get the true labels and predictions for this specific antibiotic
    y_test_col = y_test_np[:, i]
    y_pred_col = y_pred[:, i]
    
    # Calculate metrics (using weighted average for multi-class classification)
    # zero_division=0 prevents warnings if a class is not present in predictions
    accuracy = accuracy_score(y_test_col, y_pred_col)
    precision = precision_score(y_test_col, y_pred_col, average='weighted', zero_division=0)
    recall = recall_score(y_test_col, y_pred_col, average='weighted', zero_division=0)
    f1 = f1_score(y_test_col, y_pred_col, average='weighted', zero_division=0)
    
    # Add to our running totals
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    
    # Print the results clearly in the terminal
    print(f"\nAntibiotic: {col}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

# Calculate the overall averages across all target columns
avg_accuracy = total_accuracy / num_cols
avg_precision = total_precision / num_cols
avg_recall = total_recall / num_cols
avg_f1 = total_f1 / num_cols

# Print the overall averages
print("\n" + "="*40)
print("OVERALL AVERAGE METRICS")
print("="*40)
print(f"Average Accuracy:  {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall:    {avg_recall:.4f}")
print(f"Average F1 Score:  {avg_f1:.4f}")
print("="*40 + "\n")

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