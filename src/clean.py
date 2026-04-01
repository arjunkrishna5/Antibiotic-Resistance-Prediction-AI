import pandas as pd
import numpy as np

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

# Gender to Numeric mapping
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1, 'UNKNOWN': 0})
    df['Gender'] = df['Gender'].fillna(0).astype(int) 

# Explicit Antibiotic Mapping to prevent alphabetical risk
# 2: Susceptible (S), 1: Intermediate (I), 0: Resistant (R)
antibiotic_mapping = {'S': 2, 'I': 1, 'R': 0}

for col in antibiotic_cols:
    if col in df.columns:
        # We explicitly map known 'S', 'I', 'R' strings.
        mapped_col = df[col].map(antibiotic_mapping)
        # Clinical Safety Fallback: Any unmapped/missing strings are defaulted to 0 (Resistant).
        # In healthcare, it is far safer to falsely assume a bacteria is resistant than to wrongfully claim it is susceptible.
        df[col] = mapped_col.fillna(0).astype(int)

# Use Pandas category codes for any remaining random text columns
string_columns = df.select_dtypes(include=['object']).columns
for col in string_columns:
    df[col] = df[col].astype('category').cat.codes

# ---------------------------------------------------------
# Step 5: Final Quality Checks & Save
# ---------------------------------------------------------
print("\n=== Step 5: Ensuring Final Dataset Quality ===")
print("Total missing values: ", df.isnull().sum().sum())
print("Total duplicate rows: ", df.duplicated().sum())
print("\nFinal Data Types:\n", df.dtypes)

output_path = "data/Cleaned_Bacteria_Dataset.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset ready for training to: {output_path}")
