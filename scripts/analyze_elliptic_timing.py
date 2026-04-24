"""
scripts/analyze_elliptic_timing.py
===================================
Measures model performance degradation across the 49 Elliptic time steps.

The Elliptic dataset has a known temporal structure — Bitcoin transactions
from 49 consecutive time steps (each ~2 weeks). Fraud cluster composition
changes over time as new scam types emerge and old ones get busted.

This script answers:
  1. Does PR-AUC degrade on later time steps? (concept drift signal)
  2. Which time steps have the most performance degradation?
  3. At what drift rate should we trigger retraining?

Output:
  - reports/elliptic_timing_analysis.csv   (per-step metrics)
  - reports/elliptic_timing_analysis.png   (drift curve plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
import json
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

Path("reports").mkdir(exist_ok=True)

print("=" * 60)
print("ELLIPTIC TEMPORAL DRIFT ANALYSIS")
print("=" * 60)

# ── Load champion model ──────────────────────────────────────────
print("\n[1/4] Loading champion model from MLflow...")

mlflow.set_tracking_uri("./mlruns")
client = mlflow.tracking.MlflowClient()

try:
    # Get the champion model registered as 'sentinel-fraud-detection'
    versions = client.search_model_versions("name='sentinel-fraud-detection'")
    champion = sorted(versions, key=lambda v: float(
        client.get_run(v.run_id).data.metrics.get("test_pr_auc", 0)
    ), reverse=True)[0]
    model = mlflow.sklearn.load_model(f"models:/sentinel-fraud-detection/{champion.version}")
    print(f"  Loaded version {champion.version}")
except Exception:
    # Fallback: load from disk
    model_path = Path("models/champion_model.pkl")
    if not model_path.exists():
        print("  ERROR: No champion model found. Run ml.pipeline first.")
        exit(1)
    model = joblib.load(model_path)
    print(f"  Loaded from {model_path}")

# ── Load Elliptic data with time steps ──────────────────────────
print("\n[2/4] Loading Elliptic dataset with time steps...")

features_path = Path("data/raw/elliptic/elliptic_txs_features.csv")
classes_path  = Path("data/raw/elliptic/elliptic_txs_classes.csv")

if not features_path.exists():
    print("  ERROR: Elliptic raw data not found at data/raw/elliptic/")
    exit(1)

features = pd.read_csv(features_path, header=None)
classes  = pd.read_csv(classes_path)

df = features.merge(classes, left_on=0, right_on="txId", how="left")
labeled = df[df["class"].isin(["1", "2"])].copy()
labeled["is_fraud"] = (labeled["class"] == "1").astype(int)
labeled["time_step"] = labeled.iloc[:, 1].astype(int)

print(f"  Labeled nodes: {len(labeled):,}")
print(f"  Time steps:    {labeled['time_step'].nunique()} (steps {labeled['time_step'].min()}-{labeled['time_step'].max()})")
print(f"  Fraud rate:    {labeled['is_fraud'].mean()*100:.1f}%")

# ── Build feature matrix ─────────────────────────────────────────
print("\n[3/4] Building feature matrix...")

# Use Elliptic's 166 local + aggregated features
feat_cols = list(range(2, 168))
X = labeled.iloc[:, feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = labeled["is_fraud"].values
time_steps = labeled["time_step"].values

# Check model feature count
try:
    n_model_features = model.n_features_in_
except AttributeError:
    n_model_features = X_scaled.shape[1]

if X_scaled.shape[1] != n_model_features:
    print(f"  NOTE: Model expects {n_model_features} features, Elliptic has {X_scaled.shape[1]}.")
    print(f"  Using first {min(n_model_features, X_scaled.shape[1])} features for analysis.")
    X_scaled = X_scaled[:, :n_model_features]
    if X_scaled.shape[1] < n_model_features:
        pad = np.zeros((X_scaled.shape[0], n_model_features - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, pad])

# ── Per-time-step evaluation ─────────────────────────────────────
print("\n[4/4] Evaluating per time step...")

results = []
all_steps = sorted(labeled["time_step"].unique())

for step in all_steps:
    mask = time_steps == step
    X_step = X_scaled[mask]
    y_step = y[mask]

    n_total = mask.sum()
    n_fraud = y_step.sum()
    n_legit = n_total - n_fraud

    if n_fraud < 5 or n_legit < 5:
        continue

    try:
        y_prob = model.predict_proba(X_step)[:, 1]
        pr_auc  = average_precision_score(y_step, y_prob)
        roc_auc = roc_auc_score(y_step, y_prob)
    except Exception as e:
        pr_auc  = np.nan
        roc_auc = np.nan

    results.append({
        "time_step":  step,
        "n_total":    int(n_total),
        "n_fraud":    int(n_fraud),
        "fraud_rate": round(n_fraud / n_total, 4),
        "pr_auc":     round(pr_auc,  4),
        "roc_auc":    round(roc_auc, 4),
    })

    print(f"  Step {step:2d} | n={n_total:4d} | fraud={n_fraud:3d} ({n_fraud/n_total*100:.0f}%) | PR-AUC={pr_auc:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("reports/elliptic_timing_analysis.csv", index=False)

# ── Drift analysis ───────────────────────────────────────────────
valid = results_df.dropna(subset=["pr_auc"])
early_steps  = valid[valid["time_step"] <= 25]["pr_auc"].mean()
late_steps   = valid[valid["time_step"] >  25]["pr_auc"].mean()
overall      = valid["pr_auc"].mean()
min_step     = valid.loc[valid["pr_auc"].idxmin()]
max_step     = valid.loc[valid["pr_auc"].idxmax()]
drift_signal = early_steps - late_steps

print(f"\n{'='*60}")
print("DRIFT ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"  Overall mean PR-AUC:       {overall:.3f}")
print(f"  Early steps (1-25):        {early_steps:.3f}")
print(f"  Late steps  (26-49):       {late_steps:.3f}")
print(f"  Drift (early - late):      {drift_signal:+.3f}")
print(f"  Worst step:  #{int(min_step['time_step'])} (PR-AUC={min_step['pr_auc']:.3f})")
print(f"  Best step:   #{int(max_step['time_step'])} (PR-AUC={max_step['pr_auc']:.3f})")

if drift_signal > 0.05:
    print(f"\n  ⚠️  DRIFT DETECTED: {drift_signal:.3f} degradation on later steps.")
    print(f"     Recommendation: retrain every 2-4 weeks.")
elif drift_signal > 0.02:
    print(f"\n  ⚡ MILD DRIFT: {drift_signal:.3f}. Monitor weekly, retrain monthly.")
else:
    print(f"\n  ✅ STABLE: drift={drift_signal:.3f}. Model generalises across time.")

# ── Plot ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(valid["time_step"], valid["pr_auc"],  "b-o", ms=4, label="PR-AUC")
ax1.plot(valid["time_step"], valid["roc_auc"], "g-s", ms=4, label="ROC-AUC")
ax1.axvline(25, color="orange", ls="--", alpha=0.7, label="Early/Late split")
ax1.axhline(overall, color="blue", ls=":", alpha=0.5, label=f"Mean PR-AUC={overall:.3f}")
ax1.set_ylabel("Score")
ax1.set_title("Sentinel Model Performance Across Elliptic Time Steps")
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

ax2.bar(valid["time_step"], valid["fraud_rate"], color="red", alpha=0.6, label="Fraud rate")
ax2.set_xlabel("Time Step (each ~2 weeks of Bitcoin history)")
ax2.set_ylabel("Fraud Rate")
ax2.set_title("Fraud Rate by Time Step")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/elliptic_timing_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Saved → reports/elliptic_timing_analysis.csv")
print(f"  Saved → reports/elliptic_timing_analysis.png")
print(f"\n  Retraining trigger: if rolling 7-day PR-AUC drops >{drift_signal:.2f} below baseline")