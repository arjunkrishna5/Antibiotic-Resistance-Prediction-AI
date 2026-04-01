import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/Cleaned_Bacteria_Dataset.csv")

# Antibiotic columns (targets)
target_cols = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM',
    'GEN', 'AN', 'Acide nalidixique', 'ofx', 'CIP',
    'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

# Compute correlation
corr = df[target_cols].corr()

# Create graph
G = nx.Graph()

# Threshold for strong relationship
threshold = 0.2

# Add edges
for i in range(len(target_cols)):
    for j in range(i + 1, len(target_cols)):
        if abs(corr.iloc[i, j]) > threshold:
            G.add_edge(target_cols[i], target_cols[j], weight=corr.iloc[i, j])

# Create static folder if not exists
os.makedirs("static", exist_ok=True)

# Draw graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000,
    node_color="lightblue",
    font_size=8,
    font_weight="bold"
)

# Save image
output_path = "static/network_graph.png"
plt.title("Antibiotic Resistance Network")
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"\n✅ Network graph saved at: {output_path}")