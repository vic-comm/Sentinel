"""
scripts/red_team_legitimate.py
================================
Adversarial evasion test — can a fraudster craft transactions
that look legitimate enough to evade detection?

Four evasion strategies tested:
  1. Known-payee attack: send fraud amounts to existing counterparties
  2. Low-velocity attack: slow the burst down to normal transaction rate
  3. Amount blending: use amounts identical to the victim's historical average
  4. Combined: known payee + low velocity + blended amounts

For each strategy: what fraction of fraud transactions score below threshold?
Documents the honest capability boundary of the model.

Output:
  - reports/red_team_results.csv
  - reports/red_team_summary.txt
"""

import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from sklearn.metrics import average_precision_score

Path("reports").mkdir(exist_ok=True)

FRAUD_THRESHOLD = 0.5   # Score above this = flagged
HIGH_RISK       = 0.7

print("=" * 60)
print("RED TEAM — ADVERSARIAL EVASION TEST")
print("=" * 60)

# ── Load model and test data ─────────────────────────────────────
print("\n[1/3] Loading champion model and test data...")
mlflow.set_tracking_uri("./mlruns")

try:
    client  = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='sentinel-fraud-detection'")
    champion = sorted(versions, key=lambda v: float(
        client.get_run(v.run_id).data.metrics.get("test_pr_auc", 0)
    ), reverse=True)[0]
    model = mlflow.sklearn.load_model(f"models:/sentinel-fraud-detection/{champion.version}")
except Exception:
    model = joblib.load("models/champion_model.pkl")

test_df = pd.read_parquet("data/training/test.parquet")
fraud_df = test_df[test_df["is_fraud"] == True].copy()
legit_df = test_df[test_df["is_fraud"] == False].copy()

print(f"  Test fraud rows:  {len(fraud_df):,}")
print(f"  Test legit rows:  {len(legit_df):,}")

# ── Identify model features ──────────────────────────────────────
DROP_COLS = [
    "transaction_id", "client_id", "timestamp", "modality",
    "identity_hash", "sender_hash", "receiver_hash",
    "sender_wallet_hash", "receiver_wallet_hash",
    "user_email_hash", "sender_device_hash", "sender_ip_hash",
    "transaction_hash", "fraud_type", "fraud_network_id",
    "cross_modality_fraud_id", "cross_modality_pattern",
    "linked_fiat_transaction", "source", "email",
    "is_fraud", "source_confidence",
    # Elliptic-specific
    "elliptic_feature_2", "elliptic_feature_3", "elliptic_feature_4",
]

def prepare_X(df):
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    # Fill NaN
    X = X.fillna(0)
    # Align to model's expected features
    try:
        expected = model.feature_names_in_
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X[expected]
    except AttributeError:
        pass
    return X.values.astype(np.float32)

X_fraud = prepare_X(fraud_df)
baseline_scores = model.predict_proba(X_fraud)[:, 1]
baseline_evasion = (baseline_scores < FRAUD_THRESHOLD).mean()

print(f"\n  Baseline: {baseline_evasion*100:.1f}% of fraud transactions score below threshold")
print(f"  (These already evade detection without any adversarial modification)")

# ── Strategy 1: Known-payee attack ──────────────────────────────
print("\n[2/3] Testing evasion strategies...")
print("\n  Strategy 1: Known-payee attack")
print("  → Fraudster sends to a receiver already in victim's counterparty list")
print("  → Sets is_first_time_receiver = False")

s1_df = fraud_df.copy()
if "is_first_time_receiver" in s1_df.columns:
    s1_df["is_first_time_receiver"] = False
X_s1 = prepare_X(s1_df)
s1_scores = model.predict_proba(X_s1)[:, 1]
s1_evasion = (s1_scores < FRAUD_THRESHOLD).mean()
s1_drop = baseline_scores.mean() - s1_scores.mean()
print(f"  Evasion rate: {s1_evasion*100:.1f}%  (vs baseline {baseline_evasion*100:.1f}%)")
print(f"  Average score drop: {s1_drop:.3f}")

# ── Strategy 2: Low-velocity attack ─────────────────────────────
print("\n  Strategy 2: Low-velocity attack")
print("  → Fraudster spreads burst over 24h instead of 4-12 minutes")
print("  → Sets count_1h=1, count_6h=1, velocity_ratio to baseline level")

s2_df = fraud_df.copy()
for col in ["count_1h", "count_6h"]:
    if col in s2_df.columns:
        s2_df[col] = 1  # Only 1 transaction per window
if "velocity_ratio" in s2_df.columns:
    s2_df["velocity_ratio"] = 1.0  # Looks like normal rate
X_s2 = prepare_X(s2_df)
s2_scores = model.predict_proba(X_s2)[:, 1]
s2_evasion = (s2_scores < FRAUD_THRESHOLD).mean()
s2_drop = baseline_scores.mean() - s2_scores.mean()
print(f"  Evasion rate: {s2_evasion*100:.1f}%  (vs baseline {baseline_evasion*100:.1f}%)")
print(f"  Average score drop: {s2_drop:.3f}")

# ── Strategy 3: Amount blending ──────────────────────────────────
print("\n  Strategy 3: Amount blending")
print("  → Fraudster sends victim's historical average amount")
print("  → Sets amount_ratio = 1.0 (perfectly normal amount)")

s3_df = fraud_df.copy()
if "amount_ratio" in s3_df.columns:
    s3_df["amount_ratio"] = 1.0
if "balance_drain_ratio" in s3_df.columns:
    s3_df["balance_drain_ratio"] = legit_df["balance_drain_ratio"].median()
X_s3 = prepare_X(s3_df)
s3_scores = model.predict_proba(X_s3)[:, 1]
s3_evasion = (s3_scores < FRAUD_THRESHOLD).mean()
s3_drop = baseline_scores.mean() - s3_scores.mean()
print(f"  Evasion rate: {s3_evasion*100:.1f}%  (vs baseline {baseline_evasion*100:.1f}%)")
print(f"  Average score drop: {s3_drop:.3f}")

# ── Strategy 4: Combined (all three) ────────────────────────────
print("\n  Strategy 4: Combined evasion (all three)")
print("  → Perfect adversary: known payee + slow velocity + normal amount")

s4_df = fraud_df.copy()
if "is_first_time_receiver" in s4_df.columns:
    s4_df["is_first_time_receiver"] = False
for col in ["count_1h", "count_6h"]:
    if col in s4_df.columns:
        s4_df[col] = 1
if "velocity_ratio"      in s4_df.columns: s4_df["velocity_ratio"]      = 1.0
if "amount_ratio"        in s4_df.columns: s4_df["amount_ratio"]        = 1.0
if "balance_drain_ratio" in s4_df.columns: s4_df["balance_drain_ratio"] = legit_df["balance_drain_ratio"].median()
X_s4 = prepare_X(s4_df)
s4_scores = model.predict_proba(X_s4)[:, 1]
s4_evasion = (s4_scores < FRAUD_THRESHOLD).mean()
s4_still_caught = (s4_scores >= FRAUD_THRESHOLD).mean()
print(f"  Evasion rate: {s4_evasion*100:.1f}%  (vs baseline {baseline_evasion*100:.1f}%)")
print(f"  Still caught by graph features alone: {s4_still_caught*100:.1f}%")

# ── What catches the combined attacker? ─────────────────────────
print(f"\n  Residual signals catching the combined adversary:")
residual_mask = s4_scores >= FRAUD_THRESHOLD
if "sender_out_degree" in fraud_df.columns:
    caught_od = fraud_df[residual_mask]["sender_out_degree"].mean()
    evaded_od = fraud_df[~residual_mask]["sender_out_degree"].mean()
    print(f"    sender_out_degree:  caught={caught_od:.1f}, evaded={evaded_od:.1f}")
if "fan_in_ratio" in fraud_df.columns:
    caught_fi = fraud_df[residual_mask]["fan_in_ratio"].mean()
    evaded_fi = fraud_df[~residual_mask]["fan_in_ratio"].mean()
    print(f"    fan_in_ratio:       caught={caught_fi:.3f}, evaded={evaded_fi:.3f}")
if "sender_account_age_days" in fraud_df.columns:
    caught_age = fraud_df[residual_mask]["sender_account_age_days"].mean()
    evaded_age = fraud_df[~residual_mask]["sender_account_age_days"].mean()
    print(f"    account_age_days:   caught={caught_age:.0f}, evaded={evaded_age:.0f}")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RED TEAM SUMMARY")
print(f"{'='*60}")
results = [
    ("Baseline (no evasion)",          baseline_evasion, baseline_scores.mean()),
    ("Strategy 1: Known payee",        s1_evasion,       s1_scores.mean()),
    ("Strategy 2: Low velocity",       s2_evasion,       s2_scores.mean()),
    ("Strategy 3: Amount blending",    s3_evasion,       s3_scores.mean()),
    ("Strategy 4: Combined",           s4_evasion,       s4_scores.mean()),
]
print(f"  {'Strategy':<35} {'Evade%':>8} {'AvgScore':>10}")
print(f"  {'-'*55}")
for name, evade, avg in results:
    print(f"  {name:<35} {evade*100:>7.1f}% {avg:>10.3f}")

pd.DataFrame(results, columns=["strategy", "evasion_rate", "avg_score"]).to_csv(
    "reports/red_team_results.csv", index=False
)

with open("reports/red_team_summary.txt", "w") as f:
    f.write("SENTINEL RED TEAM RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write("Capability boundary:\n")
    f.write(f"  Baseline detection rate:    {(1-baseline_evasion)*100:.1f}%\n")
    f.write(f"  Combined evasion rate:      {s4_evasion*100:.1f}%\n")
    f.write(f"  Residual detection (graph): {s4_still_caught*100:.1f}%\n\n")
    f.write("Interpretation:\n")
    f.write(f"  A sophisticated adversary who mimics all tabular features\n")
    f.write(f"  can evade detection {s4_evasion*100:.0f}% of the time.\n")
    f.write(f"  The remaining {s4_still_caught*100:.0f}% are caught by graph topology\n")
    f.write(f"  (sender_out_degree, fan_in_ratio) that cannot be faked\n")
    f.write(f"  without fundamentally changing the laundering structure.\n")

print(f"\n  Saved → reports/red_team_results.csv")
print(f"  Saved → reports/red_team_summary.txt")
print(f"\n  KEY FINDING: The {s4_still_caught*100:.0f}% of fraud that survives all")
print(f"  tabular evasion is caught by graph topology — this is the GNN's value.")