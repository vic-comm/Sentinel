# scripts/merge_training_data.py
import pandas as pd
import numpy as np
from pathlib import Path

Path("data/training").mkdir(exist_ok=True)

print("Loading all data sources...")

dataframes = []

# SCHEMA CONTRACT

BOOL_COLS = [
    "is_fraud",
    "is_first_time_receiver",
    "known_mixer_interaction",
]

FLOAT_COLS = [
    "amount_usd",
    "amount_ratio",
    "velocity_ratio",
    "balance_drain_ratio",
    "count_1h",
    "count_24h",
    "sum_amount_1h",
    "sum_amount_24h",
]

CATEGORICAL_COLS = [
    "source",
    "fraud_type",
    "modality",
    "client_id",
]

ID_COLS = [
    "transaction_id",
    "identity_hash",
    "sender_hash",
    "receiver_hash",
    "transaction_hash",
]

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces strict schema across mixed data sources.
    Prevents Parquet crashes and silent dtype corruption.
    """

    # ── 1. Boolean normalization ────────────────────────────────────────────
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(bool)
            )

    # ── 2. Float normalization ──────────────────────────────────────────────
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )

    # ── 3. Categorical normalization ────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("category")

    # ── 4. ID columns → string (never object)
    for col in ID_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ── 5. Timestamp normalization ──────────────────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ── 6. Final safety: eliminate ALL object dtypes ─────────────────────────
    bad_cols = df.select_dtypes(include=["object"]).columns

    if len(bad_cols) > 0:
        print("\n[enforce_schema] ⚠️ Converting remaining object columns → string:")
        print(list(bad_cols))

        for col in bad_cols:
            df[col] = df[col].astype(str)

    return df

# ── 1. Simulator ──────────────────────────────────────────────────────────────
try:
    sim = pd.read_json("sentinel_training_data.jsonl", lines=True)
    print(f"  Simulator:         {len(sim):>9,} rows")
    dataframes.append(sim)
except:
    print("  Simulator:         FILE NOT FOUND")
    sim = pd.DataFrame()

# ── 2. Elliptic ───────────────────────────────────────────────────────────────
try:
    elliptic = pd.read_json("data/processed/elliptic_cross_modal.jsonl", lines=True)
    print(f"  Elliptic seeds:    {len(elliptic):>9,} rows")
    dataframes.append(elliptic)
except:
    print("  Elliptic seeds:    FILE NOT FOUND")
    elliptic = pd.DataFrame()

# ── 3. Alchemy ──────────────────────────────────────────────────────────────
try:
    alchemy = pd.read_parquet("data/processed/alchemy_transactions.parquet")
    print(f"  Alchemy (labeled): {len(alchemy):>9,} rows  "
          f"(fraud: {alchemy['is_fraud'].mean()*100:.2f}%)")

    # Ensure source column exists and is consistent
    if "source" not in alchemy.columns:
        alchemy["source"] = "alchemy_real"

    dataframes.append(alchemy)

except Exception as e:
    print("  Alchemy (labeled): FILE NOT FOUND")
    alchemy = pd.DataFrame()

# ── Guard: no data loaded ─────────────────────────────────────────────────────
if not dataframes:
    raise RuntimeError("No datasets were loaded. Check file paths.")

# ── Align schemas safely ──────────────────────────────────────────────────────
all_cols = sorted(set().union(*(df.columns for df in dataframes)))

dataframes = [df.reindex(columns=all_cols) for df in dataframes]

# ── Concatenate and shuffle ───────────────────────────────────────────────────
df = pd.concat(dataframes, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = enforce_schema(df)

# ── Final validation ──────────────────────────────────────────────────────────
if "is_fraud" not in df.columns:
    raise ValueError("Column 'is_fraud' is required but missing.")

total = len(df)
fraud = df["is_fraud"].sum()

print(f"\nFinal dataset:")
print(f"  Total rows:    {total:>9,}")
print(f"  Fraud rows:    {fraud:>9,}  ({fraud/total*100:.2f}%)")
print(f"  Legit rows:    {total-fraud:>9,}  ({(total-fraud)/total*100:.2f}%)")

# ── Split ─────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

train, temp = train_test_split(
    df, test_size=0.30, stratify=df["is_fraud"], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["is_fraud"], random_state=42
)

print(f"\nSplits:")
print(f"  Train: {len(train):,}  (fraud: {train['is_fraud'].sum():,})")
print(f"  Val:   {len(val):,}   (fraud: {val['is_fraud'].sum():,})")
print(f"  Test:  {len(test):,}   (fraud: {test['is_fraud'].sum():,})")

# ── Save ──────────────────────────────────────────────────────────────────────
train.to_parquet("data/training/train.parquet", index=False)
val.to_parquet("data/training/val.parquet",     index=False)
test.to_parquet("data/training/test.parquet",   index=False)

df.to_parquet("data/training/full_dataset.parquet", index=False)

print("\nSaved:")
print("  data/training/train.parquet")
print("  data/training/val.parquet")
print("  data/training/test.parquet")
print("  data/training/full_dataset.parquet")