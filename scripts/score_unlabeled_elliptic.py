"""
scripts/score_unlabeled_elliptic.py
=====================================
Runs inference on the 157K unlabeled Elliptic nodes.

Elliptic has 3 classes: 1=illicit, 2=licit, "unknown"=unlabeled (157K nodes).
These unknown nodes were real Bitcoin transactions from the same time period
but lacked human analyst verification. Running the model on them surfaces
likely fraud the ground truth didn't capture.

This script answers:
  1. How many unlabeled nodes does the model flag as high-confidence fraud?
  2. What is their time step distribution? (Are they clustered in known fraud periods?)
  3. Do they have high connectivity to confirmed illicit nodes? (graph proximity signal)

Output:
  - reports/unlabeled_predictions.csv       (all 157K scores)
  - reports/unlabeled_high_confidence.csv   (score > 0.8)
  - reports/unlabeled_scoring_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

Path("reports").mkdir(exist_ok=True)

SCORE_THRESHOLD = 0.70   # High-confidence fraud threshold
HIGH_CONF       = 0.85   # Very high confidence

print("=" * 60)
print("UNLABELED ELLIPTIC NODE SCORING")
print("=" * 60)

# ── Load champion model ──────────────────────────────────────────
print("\n[1/5] Loading champion model...")
mlflow.set_tracking_uri("./mlruns")

try:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='sentinel-fraud-detection'")
    champion = sorted(versions, key=lambda v: float(
        client.get_run(v.run_id).data.metrics.get("test_pr_auc", 0)
    ), reverse=True)[0]
    model = mlflow.sklearn.load_model(f"models:/sentinel-fraud-detection/{champion.version}")
    print(f"  Loaded MLflow champion v{champion.version}")
except Exception:
    model = joblib.load("models/champion_model.pkl")
    print("  Loaded from disk")

# ── Load full Elliptic dataset ───────────────────────────────────
print("\n[2/5] Loading Elliptic dataset...")

features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
classes  = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
edgelist = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

df = features.merge(classes, left_on=0, right_on="txId", how="left")
df["class"] = df["class"].fillna("unknown")

labeled   = df[df["class"].isin(["1", "2"])].copy()
unlabeled = df[df["class"] == "unknown"].copy()

print(f"  Labeled nodes:   {len(labeled):,}  (illicit={( labeled['class']=='1').sum():,}, licit={(labeled['class']=='2').sum():,})")
print(f"  Unlabeled nodes: {len(unlabeled):,}")

# ── Build connectivity features for unlabeled nodes ──────────────
print("\n[3/5] Computing graph proximity to confirmed illicit nodes...")

illicit_ids = set(labeled[labeled["class"] == "1"]["txId"])

# Count how many illicit neighbors each unlabeled node has
all_edges = pd.concat([
    edgelist.rename(columns={"txId1": "src", "txId2": "dst"}),
    edgelist.rename(columns={"txId2": "src", "txId1": "dst"}),
])
illicit_neighbor_count = (
    all_edges[all_edges["dst"].isin(illicit_ids)]
    .groupby("src")
    .size()
    .rename("n_illicit_neighbors")
)
unlabeled = unlabeled.merge(
    illicit_neighbor_count, left_on=0, right_index=True, how="left"
)
unlabeled["n_illicit_neighbors"] = unlabeled["n_illicit_neighbors"].fillna(0).astype(int)
unlabeled["has_illicit_neighbor"] = (unlabeled["n_illicit_neighbors"] > 0)

print(f"  Unlabeled nodes with ≥1 illicit neighbor: {unlabeled['has_illicit_neighbor'].sum():,}")

# ── Score unlabeled nodes ────────────────────────────────────────
print("\n[4/5] Scoring all unlabeled nodes...")

feat_cols = list(range(2, 168))
X = unlabeled.iloc[:, feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)

# Match model feature count
try:
    n_feat = model.n_features_in_
except AttributeError:
    n_feat = X.shape[1]

if X.shape[1] != n_feat:
    if X.shape[1] > n_feat:
        X = X[:, :n_feat]
    else:
        X = np.hstack([X, np.zeros((X.shape[0], n_feat - X.shape[1]))])

scaler = StandardScaler()
# Fit scaler on labeled data to avoid leakage
X_labeled = np.nan_to_num(labeled.iloc[:, feat_cols].values.astype(np.float32))
if X_labeled.shape[1] > n_feat:
    X_labeled = X_labeled[:, :n_feat]
elif X_labeled.shape[1] < n_feat:
    X_labeled = np.hstack([X_labeled, np.zeros((X_labeled.shape[0], n_feat - X_labeled.shape[1]))])
scaler.fit(X_labeled)
X_scaled = scaler.transform(X)

y_prob = model.predict_proba(X_scaled)[:, 1]

unlabeled = unlabeled.copy()
unlabeled["fraud_score"]      = y_prob
unlabeled["predicted_fraud"]  = (y_prob >= SCORE_THRESHOLD).astype(int)
unlabeled["high_confidence"]  = (y_prob >= HIGH_CONF).astype(int)
unlabeled["time_step"]        = unlabeled.iloc[:, 1].astype(int)

# ── Validation: PR-AUC on labeled as sanity check ───────────────
y_labeled = (labeled["class"] == "1").astype(int).values
X_lab_scaled = scaler.transform(X_labeled)
y_lab_prob = model.predict_proba(X_lab_scaled)[:, 1]
val_pr_auc = average_precision_score(y_labeled, y_lab_prob)
print(f"  Sanity check PR-AUC on labeled nodes: {val_pr_auc:.3f}")

# ── Results ──────────────────────────────────────────────────────
flagged     = unlabeled[unlabeled["predicted_fraud"] == 1]
high_conf   = unlabeled[unlabeled["high_confidence"] == 1]
illicit_adj = flagged[flagged["has_illicit_neighbor"]]

print(f"\n{'='*60}")
print("UNLABELED NODE SCORING RESULTS")
print(f"{'='*60}")
print(f"  Total unlabeled nodes scored:          {len(unlabeled):,}")
print(f"  Flagged as fraud (score ≥ {SCORE_THRESHOLD}):      {len(flagged):,}  ({len(flagged)/len(unlabeled)*100:.1f}%)")
print(f"  High confidence (score ≥ {HIGH_CONF}):      {len(high_conf):,}  ({len(high_conf)/len(unlabeled)*100:.1f}%)")
print(f"  Flagged AND near confirmed illicit:    {len(illicit_adj):,}  ({len(illicit_adj)/max(len(flagged),1)*100:.1f}% of flagged)")
print(f"\n  Score distribution:")
for thresh, label in [(0.9, "≥0.90"), (0.8, "≥0.80"), (0.7, "≥0.70"), (0.5, "≥0.50")]:
    n = (y_prob >= thresh).sum()
    print(f"    {label}: {n:,} nodes ({n/len(unlabeled)*100:.1f}%)")

# Time step distribution of flagged nodes
print(f"\n  Flagged nodes by time step (top 10):")
ts_dist = flagged["time_step"].value_counts().head(10)
for ts, cnt in ts_dist.items():
    print(f"    Step {ts:2d}: {cnt:4d} flagged nodes")

# ── Save outputs ─────────────────────────────────────────────────
print("\n[5/5] Saving outputs...")

save_cols = [0, 1, "fraud_score", "predicted_fraud", "high_confidence",
             "n_illicit_neighbors", "has_illicit_neighbor", "time_step"]
unlabeled[save_cols].rename(columns={0: "txId", 1: "raw_time_step"}) \
    .to_csv("reports/unlabeled_predictions.csv", index=False)

high_conf[save_cols].rename(columns={0: "txId", 1: "raw_time_step"}) \
    .sort_values("fraud_score", ascending=False) \
    .to_csv("reports/unlabeled_high_confidence.csv", index=False)

# Summary
with open("reports/unlabeled_scoring_summary.txt", "w") as f:
    f.write(f"SENTINEL — Unlabeled Elliptic Node Scoring Summary\n")
    f.write(f"{'='*50}\n")
    f.write(f"Total unlabeled:       {len(unlabeled):,}\n")
    f.write(f"Flagged (≥{SCORE_THRESHOLD}):       {len(flagged):,} ({len(flagged)/len(unlabeled)*100:.1f}%)\n")
    f.write(f"High confidence (≥{HIGH_CONF}): {len(high_conf):,} ({len(high_conf)/len(unlabeled)*100:.1f}%)\n")
    f.write(f"Near confirmed illicit:{len(illicit_adj):,}\n")
    f.write(f"Sanity PR-AUC:         {val_pr_auc:.4f}\n")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_prob, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
axes[0].axvline(SCORE_THRESHOLD, color="red",    ls="--", label=f"Threshold={SCORE_THRESHOLD}")
axes[0].axvline(HIGH_CONF,       color="orange", ls="--", label=f"High conf={HIGH_CONF}")
axes[0].set_xlabel("Fraud Score"); axes[0].set_ylabel("Count")
axes[0].set_title("Score Distribution — 157K Unlabeled Nodes")
axes[0].legend(); axes[0].set_yscale("log")

ts_flagged = flagged["time_step"].value_counts().sort_index()
ts_all     = unlabeled["time_step"].value_counts().sort_index()
flag_rate  = (ts_flagged / ts_all).fillna(0)
axes[1].bar(flag_rate.index, flag_rate.values, color="crimson", alpha=0.7)
axes[1].set_xlabel("Time Step"); axes[1].set_ylabel("Fraction Flagged")
axes[1].set_title("Flagged Rate by Time Step")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/unlabeled_scoring.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved → reports/unlabeled_predictions.csv      ({len(unlabeled):,} rows)")
print(f"  Saved → reports/unlabeled_high_confidence.csv  ({len(high_conf):,} rows)")
print(f"  Saved → reports/unlabeled_scoring_summary.txt")
print(f"  Saved → reports/unlabeled_scoring.png")
print(f"\n  Blog post headline: 'Model flagged {len(high_conf):,} high-confidence fraud")
print(f"  nodes not in Elliptic ground truth — {len(illicit_adj):,} are directly")
print(f"  connected to confirmed illicit transactions.'")