import pandas as pd
import numpy as np
import os
import joblib

# ---------------------------------------------------------
# Step 1: Load and Inspect Dataset
# ---------------------------------------------------------
print("=== Step 1: Loading Dataset ===")
file_path = "data/Bacteria_dataset_Multiresictance.csv"
df = pd.read_csv(file_path, low_memory=False)

# ---------------------------------------------------------
# Step 2 & 3: Full Data Cleaning & Dropping Columns
# ---------------------------------------------------------
print("\n=== Step 2 & 3: Cleaning & Dropping Columns ===")

# A. Remove exact duplicate rows
df = df.drop_duplicates()

# B. Remove irrelevant columns that cannot predict anything (Personal Identifiers & Notes)
cols_to_drop = ['ID', 'Name', 'Email', 'Address', 'Collection_Date', 'Notes']
existing_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_drop)

# C. Standardize inconsistent values BEFORE missing value imputation
# C1. The 'age/gender' column has combined data (e.g., '37/F'). We split it into two:
if 'age/gender' in df.columns:
    df[['Age', 'Gender']] = df['age/gender'].astype(str).str.split('/', n=1, expand=True)
    df = df.drop(columns=['age/gender'])
    
    # Clean Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Clean Gender
    df['Gender'] = df['Gender'].astype(str).str.upper().str.strip()

# C2. Standardize all the random text placeholders representing missing values across the table
df = df.replace(['nan', 'None', 'missing', '?', 'NA', 'error', 'NAN', ''], np.nan)

# C3. Standardize Antibiotic Columns (e.g., map 's' -> 'S', 'Intermediate' -> 'I', 'r' -> 'R')
antibiotic_cols = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
                   'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']

for col in antibiotic_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip() 
        df[col] = df[col].replace('INTERMEDIATE', 'I')

# C4. Convert Yes/No/True/False string indicators into integer binaries (1 or 0)
binary_cols = ['Diabetes', 'Hypertension', 'Hospital_before']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title().str.strip()
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Nan': np.nan})

# D. Handle all the missing values appropriately
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")

# ---------------------------------------------------------
# Step 4: Encode Categorical Columns (Strings to Numbers)
# ---------------------------------------------------------
print("\n=== Step 4: EXPLICIT Encoding Categorical Columns ===")

# B. Pre-convert binary clinical features to numeric
print("Standardizing clinical features...")
clinical_features = ['Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq', 'Age', 'Gender']
for col in clinical_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

# D1. Age Binning
def bin_age(age):
    if int(age) <= 18: return 0  # Child
    if int(age) <= 35: return 1  # Youth
    if int(age) <= 55: return 2  # Adult
    if int(age) <= 75: return 3  # Senior
    return 4  # Elderly

df['Age_Group'] = df['Age'].apply(bin_age)

# D2. Comorbidity Index
df['Comorbidity'] = (df['Diabetes'] + df['Hypertension']).astype(int)

# D3. Clinical Risk Score
# Weights: Hospitalization(2) + InfFreq(1) + Comorbidity(1)
df['Risk_Score'] = (df['Hospital_before'] * 2 + df['Infection_Freq'] + df['Comorbidity']).astype(float)

# D4. Persistent Strain Encoding
print("Encoding Strain IDs (Souches)...")
os.makedirs("models", exist_ok=True)

# Ensure Souches is string and fill missing
df['Souches'] = df['Souches'].astype(str).replace(['nan', 'None', 'NAN'], 'Unknown').str.strip()

# Build a mapping of unique strain names to integers
unique_strains = sorted(df['Souches'].unique().tolist())
if 'Unknown' not in unique_strains:
    unique_strains.append('Unknown')

strain_to_id = {name: i for i, name in enumerate(unique_strains)}
df['Souches'] = df['Souches'].map(strain_to_id).fillna(strain_to_id['Unknown']).astype(int)

# Save the mapping for the web app
joblib.dump(strain_to_id, "models/souches_mapping.pkl")
print(f"[SUCCESS] Saved strain mapping with {len(strain_to_id)} entries.")

# Explicit Antibiotic Mapping to prevent alphabetical risk
# 2: Susceptible (S), 1: Intermediate (I), 0: Resistant (R)
antibiotic_mapping = {'S': 2, 'I': 1, 'R': 0}

for col in antibiotic_cols:
    if col in df.columns:
        # We explicitly map known 'S', 'I', 'R' strings.
        mapped_col = df[col].map(antibiotic_mapping)
        # Clinical Safety Fallback: Any unmapped/missing strings are defaulted to 0 (Resistant).
        df[col] = mapped_col.fillna(0).astype(int)

# ---------------------------------------------------------
# Step 5: Final Quality Checks & Save
# ---------------------------------------------------------
print("\n=== Step 5: Ensuring Final Dataset Quality ===")

# Ensure all columns are numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

output_path = "data/Cleaned_Bacteria_Dataset.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset ready for training to: {output_path}")
