"""
network.py — Generate an antibiotic co-resistance correlation network graph.

Computes pairwise Pearson correlations between antibiotic resistance columns
and draws edges between pairs whose |correlation| exceeds a threshold.
Saves the graph to static/network_graph.png.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

DATA_PATH = "data/Cleaned_Bacteria_Dataset.csv"
OUTPUT_PATH = "static/network_graph.png"
CORRELATION_THRESHOLD = 0.2

TARGET_COLS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM",
    "GEN", "AN", "Acide nalidixique", "ofx", "CIP",
    "C", "Co-trimoxazole", "Furanes", "colistine",
]


def build_network_graph() -> None:
    """Build and save the antibiotic co-resistance network graph."""
    df = pd.read_csv(DATA_PATH)
    corr = df[TARGET_COLS].corr()

    G = nx.Graph()

    for i in range(len(TARGET_COLS)):
        for j in range(i + 1, len(TARGET_COLS)):
            weight = corr.iloc[i, j]
            if abs(weight) > CORRELATION_THRESHOLD:
                G.add_edge(TARGET_COLS[i], TARGET_COLS[j], weight=weight)

    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="#38bdf8",
        font_size=8,
        font_weight="bold",
        edge_color="#94a3b8",
    )
    plt.title("Antibiotic Resistance Co-resistance Network", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Network graph saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    build_network_graph()