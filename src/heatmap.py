"""
src/heatmap.py — AmrLens AI Scientific Heatmap Generator.
Generates a clinical resistance matrix showing which strains are most resistant.
"""

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/Cleaned_Bacteria_Dataset.csv"
OUTPUT_PATH = "static/resistance_heatmap.png"

TARGET_COLS = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM',
    'GEN', 'AN', 'Acide nalidixique', 'ofx', 'CIP',
    'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

def generate_heatmap():
    if not os.path.exists(DATA_PATH):
        print("Data not found for heatmap.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # We want a "Heat" map where Resistant (0) is Hot and Susceptible (2) is Cool.
    # We'll calculate the 'Resistance Rate' per strain.
    # Resistance Rate = 1 - (Average Status / 2) 
    # (So 0 becomes 1.0 [Full Heat], and 2 becomes 0.0 [No Heat])
    
    heatmap_data = df.groupby('Souches')[TARGET_COLS].mean()
    heatmap_data = 1 - (heatmap_data / 2.0)
    
    # Filter for top 20 most frequent strains to keep it clean
    top_strains = df['Souches'].value_counts().nlargest(20).index
    heatmap_data = heatmap_data.loc[top_strains]

    plt.figure(figsize=(14, 10))
    sns.set_theme(style="white")
    
    # Generate heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap="YlOrRd", 
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Resistance Intensity (1.0 = Critical)'}
    )
    
    plt.title("AmrLens AI: Clinical Resistance Heatmap (Top Strains)", fontsize=18, pad=20, weight='bold')
    plt.xlabel("Antibiotic Molecule", fontsize=12, labelpad=10)
    plt.ylabel("Bacterial Strain (Souches)", fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"[SUCCESS] Heatmap saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_heatmap()
