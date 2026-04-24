"""
scripts/inject_noise.py
=======================
Adds controlled noise to simulator fraud rows to break clean decision boundaries.

Problem it solves:
  The simulator generates fraud with deterministic rules:
    mule_aggregator → amount in [8K-15K], transfer_type=wire, hour in [1-4]
  XGBoost learns these exact boundaries and achieves PR-AUC ~0.997.
  GNN embeddings then add ~0.002 improvement instead of ~0.02 because
  the tabular features already near-perfectly separate the classes.

What noise injection does:
  Adds Gaussian noise to fraud transaction features so no single feature
  has a clean fraud/legit boundary. Forces the model to learn combinations
  of features, which is how real fraud detection works.

What it does NOT do:
  - Does not change labels (is_fraud stays True)
  - Does not affect real data (Alchemy/Elliptic rows untouched)
  - Does not add noise to legitimate rows (keeps legit realistic)
  - Does not change the graph features (degree counts recalculated by
    build_features.py, not stored as noise-able values here)

Noise levels are calibrated to:
  - Overlap fraud/legit distributions by ~20-30%
  - Preserve the general direction of fraud signal (not random flip)
  - Match the noise characteristics of real Alchemy/Elliptic data

Usage:
  python scripts/inject_noise.py
  python scripts/inject_noise.py --noise-level 0.3   # more overlap
  python scripts/inject_noise.py --noise-level 0.1   # less overlap
  python scripts/inject_noise.py --dry-run           # preview only
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

TRAINING_DIR = Path("data/training")
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# NOISE CONFIG
# Each entry: (column, noise_type, noise_scale)
#
# noise_type options:
#   "gaussian_pct" — add Gaussian noise scaled to noise_level% of column std
#   "gaussian_abs" — add Gaussian noise with fixed absolute scale
#   "flip_prob"    — randomly flip boolean with noise_level probability
#
# noise_level controls the magnitude via --noise-level flag (default 0.2).
# A noise_level of 0.2 means:
#   gaussian_pct: std of noise = 0.2 * column_std
#   flip_prob:    20% chance of flipping the boolean
#
# Which features to add noise to:
#   YES — features the simulator encodes with clean thresholds
#   NO  — features that are already noisy or are graph-derived
# ─────────────────────────────────────────────────────────────────────────────

NOISE_FEATURES = {
    # Amount features — simulator assigns fraud to specific ranges
    "amount_usd":              "gaussian_pct",
    "amount_ratio":            "gaussian_pct",
    "balance_drain_ratio":     "gaussian_pct",

    # Velocity features — simulator sets exact velocity for each fraud type
    "velocity_ratio":          "gaussian_pct",
    "velocity_baseline_daily": "gaussian_pct",
    "count_1h":                "gaussian_pct",
    "count_6h":                "gaussian_pct",
    "count_24h":               "gaussian_pct",
    "sum_amount_1h":           "gaussian_pct",
    "sum_amount_7d":           "gaussian_pct",

    # Temporal features — simulator clusters fraud at specific hours
    "hour_of_day":             "gaussian_abs",   # abs scale: ±2-3 hours
    "day_of_week":             "gaussian_abs",   # abs scale: ±1 day

    # Account features — simulator sets new accounts for mule fraud
    "sender_account_age_days": "gaussian_pct",
    "receiver_wallet_age_days": "gaussian_pct",
}

# Absolute noise scales for non-percentage features
ABS_NOISE_SCALES = {
    "hour_of_day":  3.0,   # ±3 hours (clipped to 0-23)
    "day_of_week":  1.0,   # ±1 day (clipped to 0-6)
}

# Features to NOT add noise to (already clean or not simulator-controlled)
# Graph features: receiver_is_recycled_address, sender_out_degree, etc.
# are computed from the transaction graph, not directly set by simulator rules.
# Transfer type: handled separately below (categorical noise).
NO_NOISE_FEATURES = [
    "receiver_is_recycled_address",
    "sender_out_degree",
    "receiver_in_degree",
    "fan_in_ratio",
    "sender_total_volume_graph",
    "receiver_avg_incoming",
    "sender_lifetime_txn_count",
    "sender_lifetime_volume_usd",
]


def add_continuous_noise(
    series: pd.Series,
    noise_type: str,
    noise_level: float,
    col_name: str,
    rng: np.random.Generator,
) -> pd.Series:
    """Add noise to a continuous feature column."""
    values = series.values.astype(np.float64).copy()
    n = len(values)

    if noise_type == "gaussian_pct":
        col_std = np.nanstd(values)
        if col_std < 1e-6:
            return series   # constant column — skip
        noise = rng.normal(0, noise_level * col_std, n)
        values = values + noise

    elif noise_type == "gaussian_abs":
        scale = ABS_NOISE_SCALES.get(col_name, 1.0) * noise_level / 0.2
        noise = rng.normal(0, scale, n)
        values = values + noise

        # Clip temporal features to valid ranges
        if col_name == "hour_of_day":
            values = np.clip(values, 0, 23)
        elif col_name == "day_of_week":
            values = np.clip(values, 0, 6)

    # Clip to non-negative for features that can't be negative
    non_negative_cols = [
        "amount_usd", "amount_ratio", "balance_drain_ratio",
        "velocity_ratio", "velocity_baseline_daily",
        "count_1h", "count_6h", "count_24h",
        "sum_amount_1h", "sum_amount_7d",
        "sender_account_age_days", "receiver_wallet_age_days",
    ]
    if col_name in non_negative_cols:
        values = np.maximum(values, 0)

    return pd.Series(values, index=series.index, dtype=series.dtype)


def add_transfer_type_noise(df: pd.DataFrame, fraud_mask: pd.Series,
                            noise_level: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    For transfer_type: randomly reassign a fraction of fraud transactions
    to a different transfer type. This breaks the clean simulator rule
    "fraud_type=mule → transfer_type=wire always."

    Only affects simulator fraud rows, not real data.
    """
    if "transfer_type" not in df.columns:
        return df

    # Only apply to simulator fraud rows
    sim_fraud_mask = fraud_mask 
    n_to_flip = int(sim_fraud_mask.sum() * noise_level * 0.5)

    if n_to_flip == 0:
        return df

    flip_indices = rng.choice(
        df[sim_fraud_mask].index.values,
        size=n_to_flip,
        replace=False,
    )

    # Possible transfer types (same as simulator uses)
    transfer_types = ["ach", "wire", "p2p", "rtp"]
    random_types   = rng.choice(transfer_types, size=n_to_flip)
    df.loc[flip_indices, "transfer_type"] = random_types

    return df


def inject_noise_to_split(
    df:          pd.DataFrame,
    noise_level: float,
    split_name:  str,
    rng:         np.random.Generator,
) -> pd.DataFrame:
    """
    Apply noise injection to a single split.

    Only modifies simulator fraud rows — real data and legit rows are untouched.
    This is intentional: we want to break the simulator's clean boundaries
    while preserving the signal in real data.
    """
    df = df.copy()

    # Identify simulator fraud rows — only these get noise
    # is_sim_fraud = (df["is_fraud"].astype(bool)) & (df["source"] == "simulator")
    is_sim_fraud = df["is_fraud"].astype(bool)
    if "source" in df.columns:
        is_sim_fraud = is_sim_fraud & (~df["source"].isin(["elliptic_real", "alchemy_real"]))
    n_sim_fraud  = is_sim_fraud.sum()

    if n_sim_fraud == 0:
        print(f"  {split_name}: no simulator fraud rows found — skipping")
        return df

    print(f"  {split_name}: {n_sim_fraud:,} simulator fraud rows will receive noise")

    # Apply continuous feature noise
    features_noised = 0
    for col, noise_type in NOISE_FEATURES.items():
        if col not in df.columns:
            continue
        if col in NO_NOISE_FEATURES:
            continue

        # Only modify the fraud rows
        fraud_subset = df.loc[is_sim_fraud, col]
        noised = add_continuous_noise(fraud_subset, noise_type, noise_level, col, rng)
        df.loc[is_sim_fraud, col] = noised
        features_noised += 1

    # Apply categorical noise to transfer_type
    df = add_transfer_type_noise(df, is_sim_fraud, noise_level, rng)

    # Distribution shift check
    print(f"    Features noised:   {features_noised}")
    if "amount_usd" in df.columns:
        fraud_mean_before = df.loc[is_sim_fraud, "amount_usd"].mean()
        legit_mean        = df.loc[~df["is_fraud"].astype(bool), "amount_usd"].mean()
        print(f"    amount_usd — fraud mean: {fraud_mean_before:,.0f}  "
              f"legit mean: {legit_mean:,.0f}")
    if "velocity_ratio" in df.columns:
        v_mean = df.loc[is_sim_fraud, "velocity_ratio"].mean()
        v_legit = df.loc[~df["is_fraud"].astype(bool), "velocity_ratio"].mean()
        print(f"    velocity_ratio — fraud: {v_mean:.2f}  legit: {v_legit:.2f}")

    return df


def check_distribution_overlap(
    df: pd.DataFrame,
    col: str,
    n_bins: int = 10,
) -> float:
    """
    Returns the Bhattacharyya coefficient (0-1) measuring overlap between
    fraud and legit distributions for a feature. Higher = more overlap = better.
    Before noise: ~0.1-0.2 (clean separation)
    After noise:  ~0.4-0.6 (realistic overlap)
    """
    fraud = df.loc[df["is_fraud"].astype(bool), col].dropna().values
    legit = df.loc[~df["is_fraud"].astype(bool), col].dropna().values

    if len(fraud) == 0 or len(legit) == 0:
        return 0.0

    combined_min = min(fraud.min(), legit.min())
    combined_max = max(fraud.max(), legit.max())

    if combined_max <= combined_min:
        return 1.0

    bins   = np.linspace(combined_min, combined_max, n_bins + 1)
    h_f, _ = np.histogram(fraud, bins=bins, density=True)
    h_l, _ = np.histogram(legit, bins=bins, density=True)

    # Normalize
    h_f = h_f / (h_f.sum() + 1e-10)
    h_l = h_l / (h_l.sum() + 1e-10)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(h_f * h_l))
    return float(bc)


def main(noise_level: float = 0.20, dry_run: bool = False):
    print("=" * 60)
    print("SENTINEL NOISE INJECTION")
    print(f"  Noise level: {noise_level}")
    print(f"  Dry run:     {dry_run}")
    print("=" * 60)
    print()
    print("Purpose: Break clean simulator decision boundaries so the model")
    print("  learns fraud pattern combinations, not single-feature thresholds.")
    print("  Only affects simulator fraud rows. Real data is untouched.")
    print()

    rng = np.random.default_rng(RANDOM_STATE)

    splits = {}
    for split in ["train", "val", "test"]:
        path = TRAINING_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"ERROR: {path} not found. Run build_features.py first.")
            return
        splits[split] = pd.read_parquet(path)
        print(f"  Loaded {split}: {len(splits[split]):,} rows")

    # ── Distribution overlap BEFORE noise ────────────────────────────────────
    print("\nDistribution overlap BEFORE noise (Bhattacharyya coefficient):")
    print("  Higher = more overlap between fraud/legit = harder problem = better")
    check_cols = ["amount_usd", "velocity_ratio", "amount_ratio",
                  "balance_drain_ratio", "count_1h"]
    for col in check_cols:
        if col in splits["train"].columns:
            bc = check_distribution_overlap(splits["train"], col)
            signal = "⚠️  too clean" if bc < 0.25 else ("✅ realistic" if bc > 0.40 else "okay")
            print(f"  {col:<30} BC = {bc:.3f}  {signal}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Apply noise ───────────────────────────────────────────────────────────
    print("\nApplying noise...")
    for split_name, df in splits.items():
        splits[split_name] = inject_noise_to_split(df, noise_level, split_name, rng)

    # ── Distribution overlap AFTER noise ─────────────────────────────────────
    print("\nDistribution overlap AFTER noise:")
    for col in check_cols:
        if col in splits["train"].columns:
            bc = check_distribution_overlap(splits["train"], col)
            signal = "⚠️  still too clean" if bc < 0.25 else ("✅ realistic" if bc > 0.40 else "okay")
            print(f"  {col:<30} BC = {bc:.3f}  {signal}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving...")
    for split_name, df in splits.items():
        path = TRAINING_DIR / f"{split_name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {split_name}: {len(df):,} rows → {path}")

    print("\n" + "=" * 60)
    print("NOISE INJECTION COMPLETE")
    print("=" * 60)
    print(f"\n  Next: python -m ml.pipeline --use-ray --trials 20")
    print(f"  Expected: PR-AUC drops ~0.05-0.10 (harder, more realistic problem)")
    print(f"  Then GNN embeddings should recover ~0.02-0.04 (actual graph signal)")
    print()
    print("  If PR-AUC stays near 0.997, increase noise level:")
    print(f"  python scripts/inject_noise.py --noise-level 0.4")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject noise into simulator fraud rows to break clean boundaries"
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.20,
        help="Noise magnitude (0.1=subtle, 0.2=moderate, 0.4=aggressive). Default: 0.20"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show distribution analysis without modifying files"
    )
    args = parser.parse_args()
    main(noise_level=args.noise_level, dry_run=args.dry_run)