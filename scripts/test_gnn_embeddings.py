# scripts/test_gnn_embeddings.py
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from scripts.train_graphsage import (
    load_elliptic, load_neo4j_graph, combine_graphs, FraudGNN
)
from torch_geometric.loader import NeighborLoader

print("Loading graph and model...")
data   = combine_graphs(load_elliptic(), load_neo4j_graph()).cpu()

with open("models/graphsage_config.json") as f:
    config = json.load(f)

# model = FraudGNN(config["in_channels"], out_channels=config["out_channels"])
model = FraudGNN(
    config["in_channels"],
    hidden_channels=config.get("hidden_channels", 128),
    out_channels=config["out_channels"]
)
model.load_state_dict(torch.load("models/graphsage_weights.pt",
                                  map_location="cpu", weights_only=True))
model.eval()

# Extract embeddings for all labeled Elliptic nodes
print("Extracting embeddings...")
loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=2048,
                         input_nodes=data.test_mask, shuffle=False, num_workers=0)

all_emb, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        emb    = model(batch.x, batch.edge_index)[:batch.batch_size]
        labels = batch.y[:batch.batch_size]
        all_emb.append(emb.numpy())
        all_labels.append(labels.numpy())

emb_matrix = np.vstack(all_emb)
labels     = np.concatenate(all_labels)

print(f"Embeddings shape: {emb_matrix.shape}")
print(f"Fraud nodes: {labels.sum():,} / {len(labels):,}")

# Test 1: Can a simple logistic regression on embeddings alone detect fraud?
# This measures whether embeddings carry fraud signal
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(emb_matrix, labels)
pr_auc = average_precision_score(labels, lr.predict_proba(emb_matrix)[:, 1])
print(f"\nLogistic regression on embeddings alone: PR-AUC = {pr_auc:.4f}")
print(f"  (>0.60 = embeddings carry strong fraud signal)")
print(f"  (<0.30 = embeddings are not separating fraud from legit)")

# Test 2: Mean embedding distance between fraud and legit
fraud_emb = emb_matrix[labels == 1]
legit_emb = emb_matrix[labels == 0]
fraud_mean = fraud_emb.mean(axis=0)
legit_mean = legit_emb.mean(axis=0)
distance   = np.linalg.norm(fraud_mean - legit_mean)
print(f"\nMean embedding distance (fraud vs legit): {distance:.4f}")
print(f"  (>1.0 = good separation, <0.3 = poor separation)")

# Test 3: Intra-class vs inter-class variance
fraud_var  = np.var(fraud_emb, axis=0).mean()
legit_var  = np.var(legit_emb, axis=0).mean()
print(f"\nIntra-class variance — fraud: {fraud_var:.4f}, legit: {legit_var:.4f}")
print(f"  (fraud variance should be lower — fraud patterns are more consistent)")