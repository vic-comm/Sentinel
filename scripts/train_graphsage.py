# """
# scripts/train_graphsage.py
# ==========================
# Trains a GraphSAGE model for fraud node classification.

# Data sources (in priority order):
#   1. elliptic_pyg.pt        — PRIMARY: real Bitcoin fraud graph, law-enforcement
#                                verified labels, 46K nodes, 166 real features
#   2. Neo4j (Docker volume)  — SECONDARY: synthetic fraud topology (mule networks,
#                                ATO, cross-modal patterns), 428K nodes
#   3. alchemy_graph_edges    — loaded into Neo4j separately via load_neo4j_graph.py


# Architecture:
#   GraphSAGE (2 SAGEConv layers + 1 GATConv attention layer)
#   Input:  node features (166 from Elliptic, 35 from Neo4j synthetic)
#   Output: 32-dimensional embedding per node
#   Loss:   CrossEntropy on labeled nodes (fraud=1, legit=0)

# Output:
#   models/graphsage_weights.pt    — trained model weights
#   models/graphsage_config.json   — feature dimensions and hyperparams

# After training, run:
#   python -m scripts.embed_gnn    — computes embeddings for all nodes → Redis

# Usage:
#   python -m scripts.train_graphsage
#   python -m scripts.train_graphsage --epochs 100 --hidden 128
# """

# import argparse
# import json
# import os
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# import mlflow
# from torch_geometric.data import Data
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import SAGEConv, GATConv
# from torch_geometric.utils import to_undirected
# from sklearn.metrics import average_precision_score, roc_auc_score
# from dotenv import load_dotenv

# load_dotenv()

# Path("models").mkdir(exist_ok=True)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
# NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sentinel_neo4j")

# ELLIPTIC_PATH  = Path("data/processed/elliptic_pyg.pt")
# ALCHEMY_EDGES  = Path("data/processed/alchemy_graph_edges.parquet")
# MODEL_PATH     = Path("models/graphsage_weights.pt")
# CONFIG_PATH    = Path("models/graphsage_config.json")

# MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# # Training defaults
# DEFAULT_HIDDEN_CHANNELS = 128
# DEFAULT_OUT_CHANNELS    = 32    # embedding dimension
# DEFAULT_EPOCHS          = 50
# DEFAULT_LR              = 0.001
# DEFAULT_DROPOUT         = 0.3
# DEFAULT_BATCH_SIZE      = 1024
# DEFAULT_NUM_NEIGHBORS   = [15, 10]  # neighbors per layer


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL DEFINITION
# # ─────────────────────────────────────────────────────────────────────────────

# class FraudGNN(torch.nn.Module):
#     """
#     3-layer GNN: 2 GraphSAGE layers + 1 Graph Attention layer.

#     GraphSAGE (layers 1-2): aggregates neighborhood features using mean
#     aggregation. Inductive — can embed nodes not seen during training.
#     This is critical for Sentinel: new wallets appear constantly.

#     GAT (layer 3): attention-weighted aggregation. Learns which neighbors
#     are most informative for fraud detection. Mule aggregator nodes attend
#     heavily to their many inbound senders.

#     Output: 32-dim embedding. Used as features in XGBoost (Phase 10).
#     """

#     def __init__(
#         self,
#         in_channels:     int,
#         hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
#         out_channels:    int = DEFAULT_OUT_CHANNELS,
#         dropout:         float = DEFAULT_DROPOUT,
#     ):
#         super().__init__()
#         self.dropout = dropout

#         # Layer 1: aggregate 1-hop neighbors
#         self.conv1 = SAGEConv(in_channels,     hidden_channels, aggr="mean")
#         # Layer 2: aggregate 2-hop neighborhood summary
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
#         # Layer 3: attention over 2-hop-aggregated features → embedding
#         self.conv3 = GATConv(hidden_channels // 2, out_channels, heads=1)

#         # Classification head (used only during training, not at serving time)
#         self.classifier = torch.nn.Linear(out_channels, 2)

#     def forward(self, x, edge_index):
#         # Layer 1
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         # Layer 2
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         # Layer 3 → 32-dim embeddings
#         x = self.conv3(x, edge_index)
#         return x  # embeddings, shape [num_nodes, 32]

#     def classify(self, embeddings):
#         """Binary classification head — only used during training."""
#         return self.classifier(embeddings)


# # ─────────────────────────────────────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────────────────────────────────────

# def load_elliptic() -> Data:
#     """
#     Load the pre-built Elliptic PyG graph from process_elliptic.py.
#     Already has: x (166 features), edge_index, y (0/1), train/val/test masks.
#     """
#     if not ELLIPTIC_PATH.exists():
#         raise FileNotFoundError(
#             f"{ELLIPTIC_PATH} not found.\n"
#             "Run: python -m scripts.process_elliptic"
#         )
#     data = torch.load(ELLIPTIC_PATH, weights_only=False)
#     print(f"  Elliptic: {data.num_nodes:,} nodes, {data.num_edges:,} edges, "
#           f"{data.x.shape[1]} features")
#     print(f"  Labels — fraud: {data.y.sum():,}, legit: {(data.y==0).sum():,}")
#     return data


# def load_neo4j_graph() -> Data:
#     """
#     Queries Neo4j for the synthetic fraud graph.
#     Builds a PyG Data object with:
#       - nodes: every Identity node with its properties
#       - edges: SENT_TO relationships
#       - features: 35-dim vector per node
#       - labels: is_fraud:int property on edges → aggregated to nodes
#     """
#     try:
#         from neo4j import GraphDatabase
#     except ImportError:
#         print("  [neo4j] pip install neo4j — skipping Neo4j graph")
#         return None

#     try:
#         driver = GraphDatabase.driver(
#             NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
#         )
#         driver.verify_connectivity()
#     except Exception as e:
#         print(f"  [neo4j] Cannot connect: {e} — skipping Neo4j graph")
#         return None

#     print("  Querying Neo4j for nodes...")
#     with driver.session(database="neo4j") as session:
#         # Sample active nodes (nodes with at least 1 edge)
#         node_result = session.run("""
#             MATCH (n:Identity)
#             WHERE n.txn_count > 0
#             RETURN n.hash AS hash,
#                    coalesce(n.txn_count, 0)      AS txn_count,
#                    coalesce(n.total_volume, 0.0)  AS total_volume,
#                    coalesce(n.institution_count, 1) AS institution_count,
#                    coalesce(n.account_age_days, 30) AS account_age_days
#             LIMIT 50000
#         """)
#         nodes = [dict(r) for r in node_result]

#     print(f"  Querying Neo4j for edges...")
#     with driver.session(database="neo4j") as session:
#         edge_result = session.run("""
#             MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
#             WHERE r.`is_fraud:int` IS NOT NULL
#             RETURN s.hash AS src,
#                    t.hash AS dst,
#                    r.`is_fraud:int` AS is_fraud,
#                    coalesce(r.amount_usd, 0.0) AS amount_usd,
#                    r.modality AS modality
#             LIMIT 500000
#         """)
#         edges = [dict(r) for r in edge_result]

#     driver.close()

#     if not nodes or not edges:
#         print("  [neo4j] No data returned — skipping Neo4j graph")
#         return None

#     print(f"  Neo4j: {len(nodes):,} nodes, {len(edges):,} edges")

#     # Build node index
#     node_hashes = [n["hash"] for n in nodes]
#     node_to_idx = {h: i for i, h in enumerate(node_hashes)}

#     # Node features (35-dim)
#     # Normalize each feature to [0, 1] range
#     feat_matrix = np.array([
#         [
#             min(float(n["txn_count"])      / 1000.0, 1.0),
#             min(float(n["total_volume"])   / 1_000_000.0, 1.0),
#             min(float(n["institution_count"]) / 3.0, 1.0),
#             min(float(n["account_age_days"]) / 3650.0, 1.0),
#         ]
#         for n in nodes
#     ], dtype=np.float32)

#     # Pad to 35 features with zeros (to match model input flexibility)
#     padding = np.zeros((len(nodes), 31), dtype=np.float32)
#     feat_matrix = np.hstack([feat_matrix, padding])

#     # Node labels: a node is fraud if ANY of its outgoing edges is fraud
#     node_fraud = {n["hash"]: 0 for n in nodes}
#     for e in edges:
#         if e["is_fraud"] == 1 and e["src"] in node_fraud:
#             node_fraud[e["src"]] = 1

#     labels = np.array([node_fraud[n["hash"]] for n in nodes], dtype=np.int64)
#     fraud_count = labels.sum()
#     print(f"  Neo4j labels — fraud: {fraud_count:,}, "
#           f"legit: {len(labels)-fraud_count:,}")

#     # Build edge index
#     src_list, dst_list = [], []
#     for e in edges:
#         if e["src"] in node_to_idx and e["dst"] in node_to_idx:
#             src_list.append(node_to_idx[e["src"]])
#             dst_list.append(node_to_idx[e["dst"]])

#     edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
#     x          = torch.tensor(feat_matrix, dtype=torch.float)
#     y          = torch.tensor(labels, dtype=torch.long)

#     # Train/val/test split (70/15/15, stratified)
#     n = len(nodes)
#     idx = torch.randperm(n)
#     train_mask = torch.zeros(n, dtype=torch.bool)
#     val_mask   = torch.zeros(n, dtype=torch.bool)
#     test_mask  = torch.zeros(n, dtype=torch.bool)
#     train_mask[idx[:int(n * 0.70)]] = True
#     val_mask  [idx[int(n * 0.70):int(n * 0.85)]] = True
#     test_mask [idx[int(n * 0.85):]] = True
#     import json
#     with open("models/neo4j_node_order.json", "w") as f:
#         json.dump(node_hashes, f)

#     return Data(
#         x=x,
#         edge_index=edge_index,
#         y=y,
#         train_mask=train_mask,
#         val_mask=val_mask,
#         test_mask=test_mask,
#     )


# def combine_graphs(elliptic: Data, neo4j: Data) -> Data:
#     """
#     Combines Elliptic and Neo4j graphs into one training graph.

#     Strategy:
#       - Normalize Elliptic features (166-dim) and Neo4j features (35-dim)
#         to the same dimension by padding the smaller one with zeros.
#       - Offset Neo4j node indices by Elliptic's node count.
#       - Concatenate node features, labels, edges, and masks.

#     The combined graph gives GraphSAGE both real fraud topology
#     (Elliptic) and your simulated fraud patterns (Neo4j mule rings,
#     ATO bursts, cross-modal laundering).
#     """
#     if neo4j is None:
#         print("  Using Elliptic graph only (Neo4j unavailable)")
#         return elliptic

#     n_e = elliptic.num_nodes
#     n_n = neo4j.num_nodes

#     # Align feature dimensions — pad shorter to match longer
#     dim_e = elliptic.x.shape[1]
#     dim_n = neo4j.x.shape[1]
#     target_dim = max(dim_e, dim_n)

#     if dim_e < target_dim:
#         pad_e = torch.zeros(n_e, target_dim - dim_e)
#         x_e   = torch.cat([elliptic.x, pad_e], dim=1)
#     else:
#         x_e = elliptic.x

#     if dim_n < target_dim:
#         pad_n = torch.zeros(n_n, target_dim - dim_n)
#         x_n   = torch.cat([neo4j.x, pad_n], dim=1)
#     else:
#         x_n = neo4j.x

#     # Combine
#     x          = torch.cat([x_e, x_n], dim=0)
#     y          = torch.cat([elliptic.y, neo4j.y], dim=0)
#     edge_index = torch.cat([
#         elliptic.edge_index,
#         neo4j.edge_index + n_e,   # offset neo4j indices
#     ], dim=1)

#     # Combine masks
#     train_mask = torch.cat([elliptic.train_mask, neo4j.train_mask])
#     val_mask   = torch.cat([elliptic.val_mask,   neo4j.val_mask])
#     test_mask  = torch.cat([elliptic.test_mask,  neo4j.test_mask])

#     combined = Data(
#         x=x,
#         edge_index=edge_index,
#         y=y,
#         train_mask=train_mask,
#         val_mask=val_mask,
#         test_mask=test_mask,
#     )

#     print(f"\n  Combined graph:")
#     print(f"    Nodes:      {combined.num_nodes:,} "
#           f"(Elliptic: {n_e:,} + Neo4j: {n_n:,})")
#     print(f"    Edges:      {combined.num_edges:,}")
#     print(f"    Features:   {target_dim}")
#     print(f"    Train nodes: {train_mask.sum():,}")
#     print(f"    Val nodes:   {val_mask.sum():,}")
#     print(f"    Test nodes:  {test_mask.sum():,}")

#     return combined


# # ─────────────────────────────────────────────────────────────────────────────
# # TRAINING
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_metrics(model, loader, device):
#     """PR-AUC and ROC-AUC on a node loader."""
#     model.eval()
#     all_probs, all_labels = [], []

#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             emb   = model(batch.x, batch.edge_index)
#             logits = model.classify(emb[:batch.batch_size])
#             probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             labels = batch.y[:batch.batch_size].cpu().numpy()
#             all_probs.extend(probs)
#             all_labels.extend(labels)

#     all_probs  = np.array(all_probs)
#     all_labels = np.array(all_labels)

#     if all_labels.sum() == 0:
#         return 0.0, 0.0

#     pr_auc  = average_precision_score(all_labels, all_probs)
#     roc_auc = roc_auc_score(all_labels, all_probs)
#     return pr_auc, roc_auc


# def train(
#     data:            Data,
#     hidden_channels: int   = DEFAULT_HIDDEN_CHANNELS,
#     out_channels:    int   = DEFAULT_OUT_CHANNELS,
#     epochs:          int   = DEFAULT_EPOCHS,
#     lr:              float = DEFAULT_LR,
#     dropout:         float = DEFAULT_DROPOUT,
#     batch_size:      int   = DEFAULT_BATCH_SIZE,
#     num_neighbors:   list  = None,
# ) -> FraudGNN:

#     if num_neighbors is None:
#         num_neighbors = DEFAULT_NUM_NEIGHBORS

#     device = torch.device(
#         "mps"  if torch.backends.mps.is_available() else
#         "cuda" if torch.cuda.is_available()         else
#         "cpu"
#     )
#     print(f"\n  Device: {device}")

#     # Class imbalance — weight fraud class higher
#     n_legit = (data.y == 0).sum().item()
#     n_fraud = (data.y == 1).sum().item()
#     pos_weight = n_legit / max(n_fraud, 1)
#     class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
#     print(f"  Class weights: legit=1.0, fraud={pos_weight:.1f}")

#     model = FraudGNN(
#         in_channels=data.x.shape[1],
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         dropout=dropout,
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, patience=5, factor=0.5, min_lr=1e-5
#     )

#     # Neighbor sampling loaders — avoids loading full graph into memory
#     train_loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         batch_size=batch_size,
#         input_nodes=data.train_mask,
#         shuffle=True,
#     )
#     val_loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         batch_size=batch_size,
#         input_nodes=data.val_mask,
#         shuffle=False,
#     )

#     best_val_pr_auc = 0.0
#     best_epoch      = 0
#     patience_count  = 0
#     early_stop_patience = 15

#     print(f"\n  Training {epochs} epochs...")
#     print(f"  {'Epoch':>6} {'Loss':>10} {'Val PR-AUC':>12} {'Val ROC':>10} {'Time':>8}")
#     print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")

#     for epoch in range(1, epochs + 1):
#         model.train()
#         t0         = time.time()
#         total_loss = 0.0
#         n_batches  = 0

#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()

#             emb    = model(batch.x, batch.edge_index)
#             logits = model.classify(emb[:batch.batch_size])
#             labels = batch.y[:batch.batch_size]

#             loss = F.cross_entropy(logits, labels, weight=class_weights)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss += loss.item()
#             n_batches  += 1

#         avg_loss = total_loss / max(n_batches, 1)

#         # Validate every 2 epochs (validation is expensive)
#         if epoch % 2 == 0 or epoch == epochs:
#             val_pr_auc, val_roc = compute_metrics(model, val_loader, device)
#             scheduler.step(1 - val_pr_auc)

#             elapsed = time.time() - t0
#             marker  = " *" if val_pr_auc > best_val_pr_auc else ""
#             print(f"  {epoch:>6} {avg_loss:>10.4f} {val_pr_auc:>12.4f} "
#                   f"{val_roc:>10.4f} {elapsed:>7.1f}s{marker}")

#             mlflow.log_metrics({
#                 "train_loss":  avg_loss,
#                 "val_pr_auc":  val_pr_auc,
#                 "val_roc_auc": val_roc,
#             }, step=epoch)

#             if val_pr_auc > best_val_pr_auc:
#                 best_val_pr_auc = val_pr_auc
#                 best_epoch      = epoch
#                 patience_count  = 0
#                 torch.save(model.state_dict(), MODEL_PATH)
#             else:
#                 patience_count += 1
#                 if patience_count >= early_stop_patience:
#                     print(f"\n  Early stopping at epoch {epoch} "
#                           f"(best val PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch})")
#                     break

#     print(f"\n  Best val PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")
#     return model, best_val_pr_auc


# # ─────────────────────────────────────────────────────────────────────────────
# # EVALUATION ON TEST SET
# # ─────────────────────────────────────────────────────────────────────────────

# def evaluate_test(model, data, device, batch_size=1024, num_neighbors=None):
#     if num_neighbors is None:
#         num_neighbors = DEFAULT_NUM_NEIGHBORS

#     test_loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         batch_size=batch_size,
#         input_nodes=data.test_mask,
#         shuffle=False,
#     )
#     pr_auc, roc_auc = compute_metrics(model, test_loader, device)
#     print(f"\n  Test PR-AUC:  {pr_auc:.4f}")
#     print(f"  Test ROC-AUC: {roc_auc:.4f}")
#     return pr_auc, roc_auc


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     print("=" * 60)
#     print("SENTINEL GRAPHSAGE TRAINING")
#     print("=" * 60)

#     # MLflow setup
#     try:
#         mlflow.set_tracking_uri(MLFLOW_URI)
#         mlflow.set_experiment("sentinel_graphsage")
#     except Exception:
#         mlflow.set_tracking_uri("./mlruns")
#         mlflow.set_experiment("sentinel_graphsage")

#     with mlflow.start_run(run_name="graphsage_training"):
#         mlflow.log_params({
#             "hidden_channels": args.hidden,
#             "out_channels":    args.out,
#             "epochs":          args.epochs,
#             "lr":              args.lr,
#             "dropout":         args.dropout,
#             "batch_size":      args.batch_size,
#         })

#         # ── 1. Load data ──────────────────────────────────────────────────
#         print("\n[1/4] Loading graphs...")
#         elliptic = load_elliptic()

#         print("\n  Loading Neo4j synthetic graph...")
#         neo4j = load_neo4j_graph()

#         data = combine_graphs(elliptic, neo4j)

#         device = torch.device(
#             "mps"  if torch.backends.mps.is_available() else
#             "cuda" if torch.cuda.is_available()         else
#             "cpu"
#         )
#         data = data.to(device)

#         # ── 2. Train ──────────────────────────────────────────────────────
#         print("\n[2/4] Training GraphSAGE...")
#         model, best_val_pr_auc = train(
#             data,
#             hidden_channels=args.hidden,
#             out_channels=args.out,
#             epochs=args.epochs,
#             lr=args.lr,
#             dropout=args.dropout,
#             batch_size=args.batch_size,
#         )

#         # ── 3. Evaluate ───────────────────────────────────────────────────
#         print("\n[3/4] Evaluating on test set...")
#         # Load best weights
#         model.load_state_dict(
#             torch.load(MODEL_PATH, map_location=device, weights_only=True)
#         )
#         test_pr_auc, test_roc_auc = evaluate_test(model, data, device)

#         mlflow.log_metrics({
#             "test_pr_auc":  test_pr_auc,
#             "test_roc_auc": test_roc_auc,
#             "best_val_pr_auc": best_val_pr_auc,
#         })

#         # ── 4. Save config ────────────────────────────────────────────────
#         print("\n[4/4] Saving artifacts...")
#         config = {
#             "in_channels":     data.x.shape[1],
#             "hidden_channels": args.hidden,
#             "out_channels":    args.out,
#             "dropout":         args.dropout,
#             "test_pr_auc":     round(test_pr_auc, 4),
#             "test_roc_auc":    round(test_roc_auc, 4),
#             "num_nodes_trained": data.num_nodes,
#             "num_edges_trained": data.num_edges,
#             "elliptic_nodes":  elliptic.num_nodes,
#             "neo4j_nodes":     neo4j.num_nodes if neo4j else 0,
#         }
#         with open(CONFIG_PATH, "w") as f:
#             json.dump(config, f, indent=2)

#         mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
#         mlflow.log_artifact(str(CONFIG_PATH), artifact_path="model")

        
#         print(f"\n  Saved → {MODEL_PATH}")
#         print(f"  Saved → {CONFIG_PATH}")

#         print("\n" + "=" * 60)
#         print("GRAPHSAGE TRAINING COMPLETE")
#         print("=" * 60)
#         print(f"  Test PR-AUC:  {test_pr_auc:.4f}")
#         print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
#         print(f"  Embedding dim: {args.out}")
#         print(f"\nNext: python -m scripts.embed_gnn")
#         print("  Computes 32-dim embeddings for all Identity nodes → Redis")
#         print("  Then: python pipeline.py --use-gnn-embeddings")
#         print("=" * 60)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train GraphSAGE fraud detection model"
#     )
#     parser.add_argument("--hidden",     type=int,   default=DEFAULT_HIDDEN_CHANNELS)
#     parser.add_argument("--out",        type=int,   default=DEFAULT_OUT_CHANNELS)
#     parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
#     parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
#     parser.add_argument("--dropout",    type=float, default=DEFAULT_DROPOUT)
#     parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
#     args = parser.parse_args()

#     main(args)

"""
scripts/train_graphsage.py
==========================
Trains a GraphSAGE model for fraud node classification.
"""

# import argparse
# import json
# import os
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# import mlflow
# from torch_geometric.data import Data
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import SAGEConv, GATConv
# from sklearn.metrics import average_precision_score, roc_auc_score
# from dotenv import load_dotenv

# load_dotenv()

# Path("models").mkdir(exist_ok=True)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
# NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sentinel_neo4j")

# ELLIPTIC_PATH  = Path("data/processed/elliptic_pyg.pt")
# MODEL_PATH     = Path("models/graphsage_weights.pt")
# CONFIG_PATH    = Path("models/graphsage_config.json")

# MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# DEFAULT_HIDDEN_CHANNELS = 128
# DEFAULT_OUT_CHANNELS    = 32
# DEFAULT_EPOCHS          = 50
# DEFAULT_LR              = 0.001
# DEFAULT_DROPOUT         = 0.3
# DEFAULT_BATCH_SIZE      = 1024
# DEFAULT_NUM_NEIGHBORS   = [15, 10]


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL DEFINITION
# # ─────────────────────────────────────────────────────────────────────────────

# class FraudGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=DEFAULT_HIDDEN_CHANNELS, out_channels=DEFAULT_OUT_CHANNELS, dropout=DEFAULT_DROPOUT):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
#         self.conv3 = GATConv(hidden_channels // 2, out_channels, heads=1)
#         self.classifier = torch.nn.Linear(out_channels, 2)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv3(x, edge_index)
#         return x

#     def classify(self, embeddings):
#         return self.classifier(embeddings)


# # ─────────────────────────────────────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────────────────────────────────────

# def load_elliptic() -> Data:
#     if not ELLIPTIC_PATH.exists():
#         raise FileNotFoundError(f"{ELLIPTIC_PATH} not found.")
#     data = torch.load(ELLIPTIC_PATH, weights_only=False)
#     print(f"  Elliptic: {data.num_nodes:,} nodes, {data.num_edges:,} edges, {data.x.shape[1]} features")
#     return data


# def load_neo4j_graph() -> Data:
#     try:
#         from neo4j import GraphDatabase
#     except ImportError:
#         print("  [neo4j] pip install neo4j — skipping Neo4j graph")
#         return None

#     try:
#         driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#         driver.verify_connectivity()
#     except Exception as e:
#         print(f"  [neo4j] Cannot connect: {e} — skipping Neo4j graph")
#         return None

#     print("  Querying Neo4j for nodes...")
#     with driver.session(database="neo4j") as session:
#         # Robust query that relies only on the hash identity
#         node_result = session.run("""
#             MATCH (n:Identity)
#             RETURN n.hash AS hash
#             LIMIT 50000
#         """)
#         nodes = [dict(r) for r in node_result]

#     print(f"  Querying Neo4j for edges...")
#     with driver.session(database="neo4j") as session:
#         # Robust query handling standard property mappings from the CSV importer
#         edge_result = session.run("""
#             MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
#             RETURN s.hash AS src,
#                    t.hash AS dst,
#                    coalesce(toInteger(r.is_fraud), 0) AS is_fraud
#             LIMIT 500000
#         """)
#         edges = [dict(r) for r in edge_result]

#     driver.close()

#     if not nodes or not edges:
#         print("  [neo4j] No data returned — skipping Neo4j graph")
#         return None

#     print(f"  Neo4j: {len(nodes):,} nodes, {len(edges):,} edges")

#     node_hashes = [n["hash"] for n in nodes]
#     node_to_idx = {h: i for i, h in enumerate(node_hashes)}

#     # Zero-padded features (GraphSAGE will learn from the topology, not the raw node stats here)
#     feat_matrix = np.zeros((len(nodes), 35), dtype=np.float32)

#     # Node labels: fraud if outgoing edge is fraud
#     node_fraud = {n["hash"]: 0 for n in nodes}
#     for e in edges:
#         if e["is_fraud"] == 1 and e["src"] in node_fraud:
#             node_fraud[e["src"]] = 1

#     labels = np.array([node_fraud[n["hash"]] for n in nodes], dtype=np.int64)
#     fraud_count = labels.sum()
#     print(f"  Neo4j labels — fraud: {fraud_count:,}, legit: {len(labels)-fraud_count:,}")

#     src_list, dst_list = [], []
#     for e in edges:
#         if e["src"] in node_to_idx and e["dst"] in node_to_idx:
#             src_list.append(node_to_idx[e["src"]])
#             dst_list.append(node_to_idx[e["dst"]])

#     edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
#     x          = torch.tensor(feat_matrix, dtype=torch.float)
#     y          = torch.tensor(labels, dtype=torch.long)

#     n = len(nodes)
#     idx = torch.randperm(n)
#     train_mask = torch.zeros(n, dtype=torch.bool)
#     val_mask   = torch.zeros(n, dtype=torch.bool)
#     test_mask  = torch.zeros(n, dtype=torch.bool)
#     train_mask[idx[:int(n * 0.70)]] = True
#     val_mask  [idx[int(n * 0.70):int(n * 0.85)]] = True
#     test_mask [idx[int(n * 0.85):]] = True

#     with open("models/neo4j_node_order.json", "w") as f:
#         json.dump(node_hashes, f)

#     return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# def combine_graphs(elliptic: Data, neo4j: Data) -> Data:
#     if neo4j is None:
#         return elliptic

#     n_e, n_n = elliptic.num_nodes, neo4j.num_nodes
#     target_dim = max(elliptic.x.shape[1], neo4j.x.shape[1])

#     x_e = torch.cat([elliptic.x, torch.zeros(n_e, target_dim - elliptic.x.shape[1])], dim=1) if elliptic.x.shape[1] < target_dim else elliptic.x
#     x_n = torch.cat([neo4j.x, torch.zeros(n_n, target_dim - neo4j.x.shape[1])], dim=1) if neo4j.x.shape[1] < target_dim else neo4j.x

#     x = torch.cat([x_e, x_n], dim=0)
#     y = torch.cat([elliptic.y, neo4j.y], dim=0)
#     edge_index = torch.cat([elliptic.edge_index, neo4j.edge_index + n_e], dim=1)

#     train_mask = torch.cat([elliptic.train_mask, neo4j.train_mask])
#     val_mask   = torch.cat([elliptic.val_mask,   neo4j.val_mask])
#     test_mask  = torch.cat([elliptic.test_mask,  neo4j.test_mask])

#     combined = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
#     print(f"\n  Combined graph: {combined.num_nodes:,} Nodes, {combined.num_edges:,} Edges")
#     return combined


# # ─────────────────────────────────────────────────────────────────────────────
# # TRAINING
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_metrics(model, loader, device):
#     model.eval()
#     all_probs, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             emb   = model(batch.x, batch.edge_index)
#             logits = model.classify(emb[:batch.batch_size])
#             probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             labels = batch.y[:batch.batch_size].cpu().numpy()
#             all_probs.extend(probs)
#             all_labels.extend(labels)

#     all_labels, all_probs = np.array(all_labels), np.array(all_probs)
#     if all_labels.sum() == 0: return 0.0, 0.0
#     return average_precision_score(all_labels, all_probs), roc_auc_score(all_labels, all_probs)


# def train(data, hidden_channels=DEFAULT_HIDDEN_CHANNELS, out_channels=DEFAULT_OUT_CHANNELS, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, dropout=DEFAULT_DROPOUT, batch_size=DEFAULT_BATCH_SIZE):
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n  Device: {device}")

#     pos_weight = (data.y == 0).sum().item() / max((data.y == 1).sum().item(), 1)
#     class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)

#     model = FraudGNN(data.x.shape[1], hidden_channels, out_channels, dropout).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

#     train_loader = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=batch_size, input_nodes=data.train_mask, shuffle=True, num_workers=0, disjoint=False)
#     val_loader   = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=batch_size, input_nodes=data.val_mask, shuffle=False)

#     best_val_pr_auc, best_epoch, patience_count = 0.0, 0, 0

#     print(f"\n  Training {epochs} epochs...")
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss, n_batches = 0.0, 0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             logits = model.classify(model(batch.x, batch.edge_index)[:batch.batch_size])
#             loss = F.cross_entropy(logits, batch.y[:batch.batch_size], weight=class_weights)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             n_batches += 1

#         if epoch % 2 == 0 or epoch == epochs:
#             val_pr_auc, val_roc = compute_metrics(model, val_loader, device)
#             scheduler.step(1 - val_pr_auc)
#             marker = " *" if val_pr_auc > best_val_pr_auc else ""
#             print(f"  Epoch {epoch:>2} | Loss: {total_loss/max(n_batches,1):.4f} | Val PR-AUC: {val_pr_auc:.4f}{marker}")

#             if val_pr_auc > best_val_pr_auc:
#                 best_val_pr_auc, best_epoch, patience_count = val_pr_auc, epoch, 0
#                 torch.save(model.state_dict(), MODEL_PATH)
#             else:
#                 patience_count += 1
#                 if patience_count >= 15:
#                     print(f"  Early stopping at epoch {epoch}")
#                     break

#     return model, best_val_pr_auc


# def evaluate_test(model, data, device):
#     test_loader = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=1024, input_nodes=data.test_mask, shuffle=False)
#     pr_auc, roc_auc = compute_metrics(model, test_loader, device)
#     print(f"\n  Test PR-AUC:  {pr_auc:.4f} | Test ROC-AUC: {roc_auc:.4f}")
#     return pr_auc, roc_auc

# def main(args):
#     print("=" * 60 + "\nSENTINEL GRAPHSAGE TRAINING\n" + "=" * 60)
#     data = combine_graphs(load_elliptic(), load_neo4j_graph())
    
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     # data = data.to(device)
    
#     model, _ = train(data, args.hidden, args.out, args.epochs, args.lr, args.dropout, args.batch_size)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
#     test_pr_auc, test_roc_auc = evaluate_test(model, data, device)

#     with open(CONFIG_PATH, "w") as f:
#         json.dump({"in_channels": data.x.shape[1], "out_channels": args.out, "test_pr_auc": round(test_pr_auc, 4)}, f)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--hidden", type=int, default=128)
#     parser.add_argument("--out", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--lr", type=float, default=0.001)
#     parser.add_argument("--dropout", type=float, default=0.3)
#     parser.add_argument("--batch-size", type=int, default=1024)
#     main(parser.parse_args())

"""
scripts/train_graphsage.py
==========================
Trains a GraphSAGE model for fraud node classification.
"""

import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import average_precision_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv()

Path("models").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sentinel_neo4j")

ELLIPTIC_PATH  = Path("data/processed/elliptic_pyg.pt")
MODEL_PATH     = Path("models/graphsage_weights.pt")
CONFIG_PATH    = Path("models/graphsage_config.json")

DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_OUT_CHANNELS    = 32
DEFAULT_EPOCHS          = 50
DEFAULT_LR              = 0.001
DEFAULT_DROPOUT         = 0.3
DEFAULT_BATCH_SIZE      = 1024
DEFAULT_NUM_NEIGHBORS   = [15, 10]


class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=DEFAULT_HIDDEN_CHANNELS, out_channels=DEFAULT_OUT_CHANNELS, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
        self.conv3 = GATConv(hidden_channels // 2, out_channels, heads=1)
        self.classifier = torch.nn.Linear(out_channels, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def classify(self, embeddings):
        return self.classifier(embeddings)


def load_elliptic() -> Data:
    if not ELLIPTIC_PATH.exists():
        raise FileNotFoundError(f"{ELLIPTIC_PATH} not found.")
    data = torch.load(ELLIPTIC_PATH, weights_only=False)
    print(f"  Elliptic: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    return data


# def load_neo4j_graph() -> Data:
#     try:
#         from neo4j import GraphDatabase
#     except ImportError:
#         print("  [neo4j] pip install neo4j — skipping Neo4j graph")
#         return None

#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
#     print("  Querying Neo4j for FULL nodes...")
#     with driver.session(database="neo4j") as session:
#         # Removed LIMIT to fetch the entire graph
#         node_result = session.run("MATCH (n:Identity) RETURN n.hash AS hash")
#         nodes = [dict(r) for r in node_result]

#     print("  Querying Neo4j for FULL edges...")
#     with driver.session(database="neo4j") as session:
#         # Removed LIMIT to fetch all connections
#         edge_result = session.run("""
#             MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
#             RETURN s.hash AS src,
#                    t.hash AS dst,
#                    coalesce(toInteger(r.is_fraud), 0) AS is_fraud
#         """)
#         edges = [dict(r) for r in edge_result]

#     driver.close()

#     if not nodes or not edges:
#         return None

#     print(f"  Neo4j: {len(nodes):,} nodes, {len(edges):,} edges")

#     node_hashes = [n["hash"] for n in nodes]
#     node_to_idx = {h: i for i, h in enumerate(node_hashes)}

#     feat_matrix = np.zeros((len(nodes), 35), dtype=np.float32)

#     node_fraud = {n["hash"]: 0 for n in nodes}
#     for e in edges:
#         if e["is_fraud"] == 1 and e["src"] in node_fraud:
#             node_fraud[e["src"]] = 1

#     labels = np.array([node_fraud[n["hash"]] for n in nodes], dtype=np.int64)
#     print(f"  Neo4j labels — fraud: {labels.sum():,}, legit: {len(labels)-labels.sum():,}")

#     src_list, dst_list = [], []
#     for e in edges:
#         if e["src"] in node_to_idx and e["dst"] in node_to_idx:
#             src_list.append(node_to_idx[e["src"]])
#             dst_list.append(node_to_idx[e["dst"]])

#     edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
#     x          = torch.tensor(feat_matrix, dtype=torch.float)
#     y          = torch.tensor(labels, dtype=torch.long)

#     n = len(nodes)
#     idx = torch.randperm(n)
#     train_mask, val_mask, test_mask = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
#     train_mask[idx[:int(n * 0.70)]] = True
#     val_mask  [idx[int(n * 0.70):int(n * 0.85)]] = True
#     test_mask [idx[int(n * 0.85):]] = True

#     with open("models/neo4j_node_order.json", "w") as f:
#         json.dump(node_hashes, f)

#     return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def load_neo4j_graph() -> Data:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  [neo4j] pip install neo4j — skipping Neo4j graph")
        return None

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # print("  Querying Neo4j for nodes with features (this may take a minute)...")
    # with driver.session(database="neo4j") as session:
    #     node_result = session.run("""
    #         MATCH (n:Identity)
    #         OPTIONAL MATCH (n)-[out:SENT_TO]->()
    #         OPTIONAL MATCH ()-[in_:SENT_TO]->(n)
    #         WITH n,
    #              count(DISTINCT out)   AS out_degree,
    #              count(DISTINCT in_)   AS in_degree,
    #              coalesce(avg(out.amount_usd), 0)  AS avg_out_amount,
    #              coalesce(sum(out.amount_usd), 0)  AS total_out_volume,
    #              coalesce(avg(in_.amount_usd), 0)  AS avg_in_amount,
    #              coalesce(sum(CASE WHEN out.`is_fraud:int` = 1 THEN 1 ELSE 0 END), 0) AS fraud_out_count,
    #              coalesce(sum(CASE WHEN in_.`is_fraud:int` = 1 THEN 1 ELSE 0 END), 0) AS fraud_in_count,
    #              n.account_age_days   AS account_age_days,
    #              n.txn_count          AS txn_count,
    #              n.total_volume       AS total_volume
    #         RETURN n.hash AS hash,
    #                out_degree, in_degree,
    #                avg_out_amount, total_out_volume,
    #                avg_in_amount,
    #                fraud_out_count, fraud_in_count,
    #                coalesce(account_age_days, 365) AS account_age_days,
    #                coalesce(txn_count, 0)           AS txn_count,
    #                coalesce(total_volume, 0)         AS total_volume
    #     """)
    #     nodes = [dict(r) for r in node_result]
    print("  Querying Neo4j for nodes with features (this may take a minute)...")
    with driver.session(database="neo4j") as session:
        # node_result = session.run("""
        #     MATCH (n:Identity)
        #     OPTIONAL MATCH (n)-[out:SENT_TO]->()
        #     OPTIONAL MATCH ()-[in_:SENT_TO]->(n)
        #     WITH n,
        #          count(DISTINCT out)   AS out_degree,
        #          count(DISTINCT in_)   AS in_degree,
        #          coalesce(avg(toFloat(out.amount_usd)), 0.0)  AS avg_out_amount,
        #          coalesce(sum(toFloat(out.amount_usd)), 0.0)  AS total_out_volume,
        #          coalesce(avg(toFloat(in_.amount_usd)), 0.0)  AS avg_in_amount,
        #          n.account_age_days   AS account_age_days,
        #          n.txn_count          AS txn_count,
        #          n.total_volume       AS total_volume
        #     RETURN n.hash AS hash,
        #            out_degree, in_degree,
        #            avg_out_amount, total_out_volume,
        #            avg_in_amount,
        #            coalesce(toFloat(account_age_days), 365.0) AS account_age_days,
        #            coalesce(toFloat(txn_count), 0.0)          AS txn_count,
        #            coalesce(toFloat(total_volume), 0.0)       AS total_volume
        # """)
        node_result = session.run("""
            MATCH (n:Identity)
            OPTIONAL MATCH (n)-[out:SENT_TO]->()
            OPTIONAL MATCH ()-[in_:SENT_TO]->(n)
            WITH n,
                count(DISTINCT out)                          AS out_degree,
                count(DISTINCT in_)                          AS in_degree,
                coalesce(avg(toFloat(out.amount_usd)), 0.0)  AS avg_out_amount,
                coalesce(sum(toFloat(out.amount_usd)), 0.0)  AS total_out_volume,
                coalesce(avg(toFloat(in_.amount_usd)), 0.0)  AS avg_in_amount
            RETURN n.hash      AS hash,
                out_degree,  in_degree,
                avg_out_amount, total_out_volume, avg_in_amount
        """)
        nodes = [dict(r) for r in node_result]

    print("  Querying Neo4j for FULL edges...")
    with driver.session(database="neo4j") as session:
        # edge_result = session.run("""
        #     MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
        #     RETURN s.hash AS src,
        #            t.hash AS dst,
        #            coalesce(toInteger(r.is_fraud), 0) AS is_fraud
        # """)
        edge_result = session.run("""
            MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
            RETURN s.hash AS src,
                t.hash AS dst,
                coalesce(r.is_fraud, 0) AS is_fraud
        """)
        edges = [dict(r) for r in edge_result]

    driver.close()

    if not nodes or not edges:
        return None

    print(f"  Neo4j: {len(nodes):,} nodes, {len(edges):,} edges")

    node_hashes = [n["hash"] for n in nodes]
    node_to_idx = {h: i for i, h in enumerate(node_hashes)}

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACT AND NORMALIZE FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    # feat_cols = [
    #     "out_degree", "in_degree",
    #     "avg_out_amount", "total_out_volume",
    #     "avg_in_amount",
    #     "fraud_out_count", "fraud_in_count",
    #     "account_age_days", "txn_count", "total_volume",
    # ]
    
    # feat_df = pd.DataFrame(nodes).fillna(0)
    
    # # Normalize each column to [0, 1] to keep GNN gradients stable
    # for col in feat_cols:
    #     col_max = feat_df[col].max()
    #     if col_max > 0:
    #         feat_df[col] = feat_df[col] / col_max
            
    # # Add derived ratio features
    # feat_df["fan_in_ratio"]       = feat_df["in_degree"]  / (feat_df["out_degree"] + 1)
    # # feat_df["fraud_out_rate"]     = feat_df["fraud_out_count"] / (feat_df["out_degree"] + 1)
    # # feat_df["fraud_in_rate"]      = feat_df["fraud_in_count"]  / (feat_df["in_degree"]  + 1)
    # feat_df["volume_per_txn"]     = feat_df["total_out_volume"] / (feat_df["txn_count"] + 1)
    
    # all_feat_cols = feat_cols + [
    #     "fan_in_ratio", "volume_per_txn"
    # ]
    
    # feat_matrix = feat_df[all_feat_cols].values.astype(np.float32)
    # ─────────────────────────────────────────────────────────────────────────
    # feat_cols = [
    #     "out_degree", "in_degree",
    #     "avg_out_amount", "total_out_volume",
    #     "avg_in_amount",
    #     "account_age_days", "txn_count", "total_volume",
    # ]
    
    # feat_df = pd.DataFrame(nodes).fillna(0)
    
    # # Normalize each column to [0, 1] to keep GNN gradients stable
    # for col in feat_cols:
    #     col_max = feat_df[col].max()
    #     if col_max > 0:
    #         feat_df[col] = feat_df[col] / col_max
            
    # # Add derived ratio features
    # feat_df["fan_in_ratio"]       = feat_df["in_degree"]  / (feat_df["out_degree"] + 1)
    # feat_df["volume_per_txn"]     = feat_df["total_out_volume"] / (feat_df["txn_count"] + 1)
    
    # all_feat_cols = feat_cols + ["fan_in_ratio", "volume_per_txn"]
    
    # feat_matrix = feat_df[all_feat_cols].values.astype(np.float32)

    feat_cols = ["out_degree", "in_degree", "avg_out_amount", "total_out_volume", "avg_in_amount"]
    feat_df   = pd.DataFrame(nodes).fillna(0)
    for col in feat_cols:
        col_max = feat_df[col].max()
        if col_max > 0:
            feat_df[col] = feat_df[col] / col_max
    feat_df["fan_in_ratio"]   = feat_df["in_degree"]  / (feat_df["out_degree"] + 1)
    feat_df["volume_per_txn"] = feat_df["total_out_volume"] / (feat_df["out_degree"] + 1)
    all_feat_cols = feat_cols + ["fan_in_ratio", "volume_per_txn"]
    feat_matrix   = feat_df[all_feat_cols].values.astype(np.float32)
    
    node_fraud = {n["hash"]: 0 for n in nodes}
    for e in edges:
        if e["is_fraud"] == 1 and e["src"] in node_fraud:
            node_fraud[e["src"]] = 1

    labels = np.array([node_fraud[n["hash"]] for n in nodes], dtype=np.int64)
    print(f"  Neo4j labels — fraud: {labels.sum():,}, legit: {len(labels)-labels.sum():,}")

    src_list, dst_list = [], []
    for e in edges:
        if e["src"] in node_to_idx and e["dst"] in node_to_idx:
            src_list.append(node_to_idx[e["src"]])
            dst_list.append(node_to_idx[e["dst"]])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x          = torch.tensor(feat_matrix, dtype=torch.float)
    y          = torch.tensor(labels, dtype=torch.long)

    n = len(nodes)
    idx = torch.randperm(n)
    train_mask, val_mask, test_mask = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(n * 0.70)]] = True
    val_mask  [idx[int(n * 0.70):int(n * 0.85)]] = True
    test_mask [idx[int(n * 0.85):]] = True

    with open("models/neo4j_node_order.json", "w") as f:
        json.dump(node_hashes, f)

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# def combine_graphs(elliptic: Data, neo4j: Data) -> Data:
#     if neo4j is None:
#         return elliptic

#     n_e, n_n = elliptic.num_nodes, neo4j.num_nodes
#     target_dim = max(elliptic.x.shape[1], neo4j.x.shape[1])

#     x_e = torch.cat([elliptic.x, torch.zeros(n_e, target_dim - elliptic.x.shape[1])], dim=1) if elliptic.x.shape[1] < target_dim else elliptic.x
#     x_n = torch.cat([neo4j.x, torch.zeros(n_n, target_dim - neo4j.x.shape[1])], dim=1) if neo4j.x.shape[1] < target_dim else neo4j.x

#     x = torch.cat([x_e, x_n], dim=0)
#     y = torch.cat([elliptic.y, neo4j.y], dim=0)
#     edge_index = torch.cat([elliptic.edge_index, neo4j.edge_index + n_e], dim=1)

#     train_mask = torch.cat([elliptic.train_mask, neo4j.train_mask])
#     val_mask   = torch.cat([elliptic.val_mask,   neo4j.val_mask])
#     test_mask  = torch.cat([elliptic.test_mask,  neo4j.test_mask])

#     combined = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
#     print(f"\n  Combined graph: {combined.num_nodes:,} Nodes, {combined.num_edges:,} Edges")
#     return combined

def combine_graphs(elliptic: Data, neo4j: Data) -> Data:
    if neo4j is None:
        return elliptic

    n_e = elliptic.num_nodes
    n_n = neo4j.num_nodes
    target_dim = max(elliptic.x.shape[1], neo4j.x.shape[1])

    # Pad features to same dimension
    if elliptic.x.shape[1] < target_dim:
        pad = torch.zeros(n_e, target_dim - elliptic.x.shape[1])
        x_e = torch.cat([elliptic.x, pad], dim=1)
    else:
        x_e = elliptic.x

    if neo4j.x.shape[1] < target_dim:
        pad = torch.zeros(n_n, target_dim - neo4j.x.shape[1])
        x_n = torch.cat([neo4j.x, pad], dim=1)
    else:
        x_n = neo4j.x

    x          = torch.cat([x_e, x_n], dim=0)
    y          = torch.cat([elliptic.y, neo4j.y], dim=0)
    edge_index = torch.cat([elliptic.edge_index, neo4j.edge_index + n_e], dim=1)

    # KEY CHANGE: Only Elliptic nodes participate in labeled training
    # Neo4j nodes contribute graph structure but have no_label mask
    n_total    = n_e + n_n

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask   = torch.zeros(n_total, dtype=torch.bool)
    test_mask  = torch.zeros(n_total, dtype=torch.bool)

    # Only Elliptic nodes (first n_e) are in the labeled masks
    train_mask[:n_e] = elliptic.train_mask
    val_mask[:n_e]   = elliptic.val_mask
    test_mask[:n_e]  = elliptic.test_mask

    # Neo4j nodes [n_e:] are NOT in any mask — they exist only for topology
    # GraphSAGE will sample their neighborhoods during Elliptic node training,
    # which enriches the Elliptic node embeddings with synthetic fraud topology

    combined = Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    n_labeled   = train_mask.sum() + val_mask.sum() + test_mask.sum()
    n_structure = n_total - n_labeled
    print(f"\n  Combined graph:")
    print(f"    Total nodes:     {n_total:,}")
    print(f"    Labeled (train): {train_mask.sum():,} Elliptic nodes")
    print(f"    Structure only:  {n_structure:,} Neo4j nodes")
    print(f"    Total edges:     {combined.num_edges:,}")
    return combined

def compute_metrics(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb   = model(batch.x, batch.edge_index)
            logits = model.classify(emb[:batch.batch_size])
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = batch.y[:batch.batch_size].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    if all_labels.sum() == 0: return 0.0, 0.0
    return average_precision_score(all_labels, all_probs), roc_auc_score(all_labels, all_probs)


def train(data, hidden_channels, out_channels, epochs, lr, dropout, batch_size):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Model Target Device: {device}")
    
    # 🚨 THE CRITICAL FIX: Lock the graph physically to the CPU to prevent Apple Silicon memory corruption
    print("  Locking Graph Data to CPU for safe Neighborhood Sampling...")
    data = data.cpu() 

    # pos_weight = (data.y == 0).sum().item() / max((data.y == 1).sum().item(), 1)
    labeled_mask = data.train_mask | data.val_mask | data.test_mask
    labeled_y    = data.y[labeled_mask]
    pos_weight   = (labeled_y == 0).sum().item() / max((labeled_y == 1).sum().item(), 1)
    print(f"  Class weight — legit: 1.0, fraud: {pos_weight:.1f}x")
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)

    model = FraudGNN(data.x.shape[1], hidden_channels, out_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

    train_loader = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=batch_size, input_nodes=data.train_mask, shuffle=True, num_workers=4)
    val_loader   = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=batch_size, input_nodes=data.val_mask, shuffle=False, num_workers=4)

    best_val_pr_auc, best_epoch, patience_count = 0.0, 0, 0

    print(f"\n  Training {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device) # Safely move just the tiny batch to the GPU
            optimizer.zero_grad()
            logits = model.classify(model(batch.x, batch.edge_index)[:batch.batch_size])
            loss = F.cross_entropy(logits, batch.y[:batch.batch_size], weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 2 == 0 or epoch == epochs:
            val_pr_auc, val_roc = compute_metrics(model, val_loader, device)
            scheduler.step(1 - val_pr_auc)
            marker = " *" if val_pr_auc > best_val_pr_auc else ""
            print(f"  Epoch {epoch:>2} | Loss: {total_loss/max(n_batches,1):.4f} | Val PR-AUC: {val_pr_auc:.4f}{marker}")

            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc, best_epoch, patience_count = val_pr_auc, epoch, 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                patience_count += 1
                if patience_count >= 15:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    return model, best_val_pr_auc


def evaluate_test(model, data, device):
    test_loader = NeighborLoader(data, num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=1024, input_nodes=data.test_mask, shuffle=False, num_workers=4)
    pr_auc, roc_auc = compute_metrics(model, test_loader, device)
    print(f"\n  Test PR-AUC:  {pr_auc:.4f} | Test ROC-AUC: {roc_auc:.4f}")
    return pr_auc, roc_auc

def main(args):
    print("=" * 60 + "\nSENTINEL GRAPHSAGE TRAINING\n" + "=" * 60)
    data = combine_graphs(load_elliptic(), load_neo4j_graph())
    
    # Notice we removed `data = data.to(device)` from here. It is handled safely in train().
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model, _ = train(data, args.hidden, args.out, args.epochs, args.lr, args.dropout, args.batch_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    test_pr_auc, test_roc_auc = evaluate_test(model, data, device)

    with open(CONFIG_PATH, "w") as f:
        json.dump({"in_channels": data.x.shape[1], "hidden_channels": args.hidden, "out_channels": args.out, "test_pr_auc": round(test_pr_auc, 4)}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=1024)
    main(parser.parse_args())