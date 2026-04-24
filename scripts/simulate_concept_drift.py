"""
scripts/simulate_concept_drift.py
===================================
Injects new fraud patterns the model was NOT trained on.
Measures PR-AUC degradation as a function of:
  1. How novel the fraud pattern is (drift intensity)
  2. How much of the test set has drifted

This motivates:
  - The Prometheus alert threshold (at what drift level do we retrain?)
  - The retraining cadence recommendation
  - The monitoring dashboard design

Four drift scenarios:
  A. Micro-structuring: many sub-$1K transactions (was $2K-$9K in training)
  B. Dormant account reactivation: 3-year-old accounts suddenly active
  C. Synthetic identity v2: accounts aged 90-180 days (was <90 in training)
  D. Peer-to-peer carousel: circular payments between 6 accounts

Output:
  - reports/concept_drift_analysis.csv
  - reports/concept_drift_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from pathlib import Path
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings("ignore")

Path("reports").mkdir(exist_ok=True)

print("=" * 60)
print("CONCEPT DRIFT SIMULATION")
print("=" * 60)

# ── Load model and test data ─────────────────────────────────────
print("\n[1/4] Loading model and test data...")
mlflow.set_tracking_uri("./mlruns")
try:
    client   = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='sentinel-fraud-detection'")
    champion = sorted(versions, key=lambda v: float(
        client.get_run(v.run_id).data.metrics.get("test_pr_auc", 0)
    ), reverse=True)[0]
    model = mlflow.sklearn.load_model(f"models:/sentinel-fraud-detection/{champion.version}")
except Exception:
    model = joblib.load("models/champion_model.pkl")

test_df = pd.read_parquet("data/training/test.parquet")

DROP_COLS = [
    "transaction_id", "client_id", "timestamp", "modality",
    "identity_hash", "sender_hash", "receiver_hash",
    "sender_wallet_hash", "receiver_wallet_hash", "user_email_hash",
    "sender_device_hash", "sender_ip_hash", "transaction_hash",
    "fraud_type", "fraud_network_id", "cross_modality_fraud_id",
    "cross_modality_pattern", "linked_fiat_transaction",
    "source", "email", "is_fraud", "source_confidence",
    "elliptic_feature_2", "elliptic_feature_3", "elliptic_feature_4",
]

def prepare_X(df):
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X = X.fillna(0)
    try:
        expected = model.feature_names_in_
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X[expected]
    except AttributeError:
        pass
    return X.values.astype(np.float32)

# Baseline performance
legit_df = test_df[test_df["is_fraud"] == False].copy()
fraud_df = test_df[test_df["is_fraud"] == True].copy()
X_test = prepare_X(test_df)
y_test = test_df["is_fraud"].astype(int).values
baseline_scores = model.predict_proba(X_test)[:, 1]
baseline_pr_auc = average_precision_score(y_test, baseline_scores)
print(f"  Baseline test PR-AUC: {baseline_pr_auc:.4f}")

# ── Generate drifted fraud patterns ─────────────────────────────
print("\n[2/4] Generating drifted fraud patterns...")

def get_legit_template(n=500):
    """Sample legitimate rows to use as base for synthetic fraud."""
    return legit_df.sample(n=min(n, len(legit_df)), replace=True).copy()

# Scenario A: Micro-structuring (many tiny transfers)
def scenario_a_micro_structuring(n=500):
    df = get_legit_template(n)
    df["is_fraud"] = True
    # New pattern: sub-$1K amounts (model trained on $2K-$9K structuring)
    if "amount_usd" in df.columns:
        df["amount_usd"] = np.random.uniform(100, 999, n)
    if "count_1h"    in df.columns: df["count_1h"]    = np.random.randint(8, 20, n)
    if "count_6h"    in df.columns: df["count_6h"]    = np.random.randint(15, 40, n)
    if "velocity_ratio" in df.columns: df["velocity_ratio"] = np.random.uniform(5, 12, n)
    return df

# Scenario B: Dormant account reactivation (new mule recruitment)
def scenario_b_dormant(n=500):
    df = get_legit_template(n)
    df["is_fraud"] = True
    # Accounts that were dormant 3+ years, suddenly very active
    if "sender_account_age_days"    in df.columns: df["sender_account_age_days"]    = np.random.randint(1000, 3000, n)
    if "sender_lifetime_txn_count"  in df.columns: df["sender_lifetime_txn_count"]  = np.random.randint(1, 5, n)
    if "count_6h"                   in df.columns: df["count_6h"]                   = np.random.randint(10, 25, n)
    if "velocity_ratio"             in df.columns: df["velocity_ratio"]             = np.random.uniform(20, 50, n)
    # Old account but first-time receiver — dormant accounts being weaponised
    if "is_first_time_receiver"     in df.columns: df["is_first_time_receiver"]     = True
    return df

# Scenario C: Synthetic identity v2 (accounts 90-180 days — just outside training range)
def scenario_c_synthetic_v2(n=500):
    df = get_legit_template(n)
    df["is_fraud"] = True
    # Training data: synthetic identity = account_age < 90 days
    # Adversary uses 90-180 days to evade that signal
    if "sender_account_age_days" in df.columns:
        df["sender_account_age_days"] = np.random.randint(90, 180, n)
    if "balance_drain_ratio"     in df.columns:
        df["balance_drain_ratio"]     = np.random.uniform(0.6, 0.95, n)
    if "is_first_time_receiver"  in df.columns:
        df["is_first_time_receiver"]  = True
    if "sender_lifetime_txn_count" in df.columns:
        df["sender_lifetime_txn_count"] = np.random.randint(15, 40, n)
    return df

# Scenario D: P2P carousel (circular payments — no net flow but creates cover)
def scenario_d_carousel(n=500):
    df = get_legit_template(n)
    df["is_fraud"] = True
    # Carousel: circular payments. From graph: receiver_in_degree and fan_in_ratio
    # are lower than mule networks (balanced topology, not fan-in)
    if "fan_in_ratio"          in df.columns: df["fan_in_ratio"]          = np.random.uniform(0.8, 1.2, n)
    if "receiver_in_degree"    in df.columns: df["receiver_in_degree"]    = np.random.randint(3, 8, n)
    if "sender_out_degree"     in df.columns: df["sender_out_degree"]     = np.random.randint(3, 8, n)
    if "count_24h"             in df.columns: df["count_24h"]             = np.random.randint(6, 15, n)
    if "velocity_ratio"        in df.columns: df["velocity_ratio"]        = np.random.uniform(3, 8, n)
    if "is_first_time_receiver" in df.columns: df["is_first_time_receiver"] = False  # Known payees
    return df

scenarios = {
    "A: Micro-structuring (<$1K)":      scenario_a_micro_structuring,
    "B: Dormant reactivation (3yr acc)": scenario_b_dormant,
    "C: Synthetic identity v2 (90-180d)": scenario_c_synthetic_v2,
    "D: P2P carousel (circular)":        scenario_d_carousel,
}

# ── Measure PR-AUC at different drift intensities ────────────────
print("\n[3/4] Measuring PR-AUC vs drift intensity...")

drift_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
all_results  = []

for scenario_name, gen_fn in scenarios.items():
    print(f"\n  {scenario_name}")
    drifted_fraud = gen_fn(n=len(fraud_df))

    for drift_frac in drift_levels:
        # Mix original test data with drifted fraud
        n_drift = int(len(fraud_df) * drift_frac)
        n_orig  = len(fraud_df) - n_drift

        if n_orig > 0:
            orig_sample = fraud_df.sample(n=n_orig, replace=False)
        else:
            orig_sample = pd.DataFrame()

        if n_drift > 0:
            drift_sample = drifted_fraud.sample(n=n_drift, replace=True)
        else:
            drift_sample = pd.DataFrame()

        if len(orig_sample) > 0 and len(drift_sample) > 0:
            mixed_fraud = pd.concat([orig_sample, drift_sample], ignore_index=True)
        elif len(orig_sample) > 0:
            mixed_fraud = orig_sample
        else:
            mixed_fraud = drift_sample

        mixed_test = pd.concat([legit_df, mixed_fraud], ignore_index=True)
        mixed_test  = mixed_test.sample(frac=1, random_state=42).reset_index(drop=True)

        X_mix = prepare_X(mixed_test)
        y_mix = mixed_test["is_fraud"].astype(int).values

        if y_mix.sum() == 0:
            continue

        y_prob   = model.predict_proba(X_mix)[:, 1]
        pr_auc   = average_precision_score(y_mix, y_prob)
        pct_drop = (baseline_pr_auc - pr_auc) / baseline_pr_auc * 100

        all_results.append({
            "scenario":       scenario_name,
            "drift_fraction": drift_frac,
            "pr_auc":         round(pr_auc,  4),
            "pct_drop":       round(pct_drop, 2),
        })
        print(f"    Drift {drift_frac*100:.0f}%: PR-AUC={pr_auc:.3f} ({pct_drop:+.1f}%)")

results_df = pd.DataFrame(all_results)
results_df.to_csv("reports/concept_drift_analysis.csv", index=False)

# ── Alert threshold derivation ───────────────────────────────────
print(f"\n[4/4] Deriving alert thresholds...")
print(f"\n  Baseline PR-AUC: {baseline_pr_auc:.4f}")
print(f"  Alert at 5% drop: {baseline_pr_auc * 0.95:.4f}")
print(f"  Alert at 10% drop: {baseline_pr_auc * 0.90:.4f}")

# Find drift fraction that causes 5% and 10% PR-AUC drop per scenario
print(f"\n  Drift fraction triggering 5% PR-AUC drop by scenario:")
for scenario_name in scenarios:
    sc_df = results_df[results_df["scenario"] == scenario_name]
    trigger = sc_df[sc_df["pct_drop"] >= 5.0]["drift_fraction"].min()
    print(f"    {scenario_name[:40]}: {trigger*100:.0f}% of fraud drifted")

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ["steelblue", "crimson", "green", "orange"]
for i, (scenario_name, _) in enumerate(scenarios.items()):
    sc_df = results_df[results_df["scenario"] == scenario_name]
    axes[0].plot(sc_df["drift_fraction"]*100, sc_df["pr_auc"],
                 "-o", color=colors[i], label=scenario_name[:30], ms=5)

axes[0].axhline(baseline_pr_auc,           color="black", ls="--", alpha=0.5, label="Baseline")
axes[0].axhline(baseline_pr_auc * 0.95,    color="orange", ls=":",  alpha=0.7, label="Alert (5% drop)")
axes[0].axhline(baseline_pr_auc * 0.90,    color="red",    ls=":",  alpha=0.7, label="Critical (10% drop)")
axes[0].set_xlabel("% of Fraud Transactions Using New Pattern")
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("PR-AUC Degradation vs Drift Intensity")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Heatmap of drift impact
pivot = results_df.pivot_table(
    values="pct_drop",
    index="scenario",
    columns="drift_fraction"
)
im = axes[1].imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                     vmin=0, vmax=20)
axes[1].set_xticks(range(len(pivot.columns)))
axes[1].set_xticklabels([f"{int(c*100)}%" for c in pivot.columns])
axes[1].set_yticks(range(len(pivot.index)))
axes[1].set_yticklabels([s[:25] for s in pivot.index], fontsize=8)
axes[1].set_xlabel("Drift Fraction"); axes[1].set_title("PR-AUC % Drop Heatmap")
plt.colorbar(im, ax=axes[1], label="% PR-AUC drop")

plt.tight_layout()
plt.savefig("reports/concept_drift_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Monitoring recommendation:")
print(f"    - Alert threshold:    5% PR-AUC drop from {baseline_pr_auc:.3f}")
print(f"    - Alert fires at:     {baseline_pr_auc * 0.95:.3f}")
print(f"    - Critical threshold: 10% drop → immediate retrain")
print(f"    - Measurement cadence: hourly on last 1000 scored transactions")
print(f"\n  Saved → reports/concept_drift_analysis.csv")
print(f"  Saved → reports/concept_drift_analysis.png")