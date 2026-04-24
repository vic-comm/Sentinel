# scripts/test_gnn_temporal.py
"""
Reloads Elliptic with temporal train/val/test split instead of random.
Elliptic has 49 time steps. Use steps 1-34 for train, 35-41 for val, 42-49 for test.
If PR-AUC drops significantly vs random split, temporal leakage was inflating results.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
classes  = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
edgelist = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

df      = features.merge(classes, left_on=0, right_on="txId", how="left")
labeled = df[df["class"].isin(["1", "2"])].copy()

# Time step is column 1
time_steps = labeled.iloc[:, 1].values
n_steps    = int(time_steps.max())
print(f"Time steps: 1 to {n_steps}")
print(f"Train: steps 1-{int(n_steps*0.70)}")
print(f"Val:   steps {int(n_steps*0.70)+1}-{int(n_steps*0.85)}")
print(f"Test:  steps {int(n_steps*0.85)+1}-{n_steps}")

train_mask = torch.tensor(time_steps <= n_steps * 0.70, dtype=torch.bool)
val_mask   = torch.tensor((time_steps > n_steps * 0.70) & (time_steps <= n_steps * 0.85), dtype=torch.bool)
test_mask  = torch.tensor(time_steps > n_steps * 0.85, dtype=torch.bool)

# Load existing pyg data and replace masks
data = torch.load("data/processed/elliptic_pyg.pt", weights_only=False)
data.train_mask = train_mask
data.val_mask   = val_mask
data.test_mask  = test_mask

torch.save(data, "data/processed/elliptic_pyg_temporal.pt")

print(f"\nTemporal split saved → data/processed/elliptic_pyg_temporal.pt")
print(f"  Train nodes: {train_mask.sum():,}")
print(f"  Val nodes:   {val_mask.sum():,}")
print(f"  Test nodes:  {test_mask.sum():,}")
print(f"\nTo test: temporarily rename files and retrain")
print(f"  cp data/processed/elliptic_pyg.pt data/processed/elliptic_pyg_random.pt")
print(f"  cp data/processed/elliptic_pyg_temporal.pt data/processed/elliptic_pyg.pt")
print(f"  python -m scripts.train_graphsage --epochs 50 --lr 0.0005 --batch-size 4096 --hidden 256")
print(f"  cp data/processed/elliptic_pyg_random.pt data/processed/elliptic_pyg.pt")