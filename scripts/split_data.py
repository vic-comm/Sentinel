# import pandas as pd
# from pathlib import Path
# def main():
#     print("Loading enriched dataset...")
#     df = pd.read_parquet("data/training/enriched_dataset.parquet")

#     print("Sorting chronologically...")
#     df = df.sort_values("timestamp").reset_index(drop=True)

#     # Calculate split indices (80% Train, 10% Validation, 10% Test)
#     n = len(df)
#     train_end = int(n * 0.8)
#     val_end = int(n * 0.9)

#     print("Splitting data...")
#     train_df = df.iloc[:train_end]
#     val_df = df.iloc[train_end:val_end]
#     test_df = df.iloc[val_end:]

#     print("Saving final splits...")
#     train_df.to_parquet("data/training/train.parquet", index=False)
#     val_df.to_parquet("data/training/val.parquet", index=False)
#     test_df.to_parquet("data/training/test.parquet", index=False)

#     print(f"✅ Train: {len(train_df)} rows")
#     print(f"✅ Val:   {len(val_df)} rows")
#     print(f"✅ Test:  {len(test_df)} rows")
#     print("Ready for ML Pipeline!")

# if __name__ == '__main__':
#     main()

"""
scripts/split_data.py
=====================
Temporal train/val/test split on the enriched dataset.

Why temporal split instead of random:
  Random split leaks future information into training — the model sees
  transactions from the same users at both train and test time, inflating
  PR-AUC by ~3-8 points vs. a real deployment scenario.

  Temporal split simulates real deployment: train on history, test on future.
  PR-AUC will be lower but more honest.

Split: 80% train / 10% val / 10% test (by row count, sorted by timestamp)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("TEMPORAL SPLIT")
    print("=" * 60)

    enriched_path = Path("data/training/enriched_dataset.parquet")
    if not enriched_path.exists():
        print(f"ERROR: {enriched_path} not found.")
        print("Run: python -m scripts.build_features")
        return

    print("Loading enriched dataset...")
    df = pd.read_parquet(enriched_path)
    print(f"  Total rows: {len(df):,}")

    # Sort chronologically
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing — cannot do temporal split")

    print("Sorting chronologically...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time range
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"  Time range: {ts_min} → {ts_max}")

    # Split by index position (80/10/10)
    n         = len(df)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Validate fraud rates — should be roughly consistent across splits
    # Large discrepancy = fraud is temporally clustered (expected and fine)
    train_fraud = train_df["is_fraud"].mean() * 100
    val_fraud   = val_df["is_fraud"].mean()   * 100
    test_fraud  = test_df["is_fraud"].mean()  * 100

    print(f"\nSplit summary:")
    print(f"  {'Split':<8} {'Rows':>10} {'Fraud %':>10} {'Time start':<25}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*25}")
    print(f"  {'train':<8} {len(train_df):>10,} {train_fraud:>9.2f}% {str(train_df['timestamp'].min()):<25}")
    print(f"  {'val':<8} {len(val_df):>10,} {val_fraud:>9.2f}% {str(val_df['timestamp'].min()):<25}")
    print(f"  {'test':<8} {len(test_df):>10,} {test_fraud:>9.2f}% {str(test_df['timestamp'].min()):<25}")

    if abs(train_fraud - test_fraud) > 5.0:
        print(f"\n  NOTE: Fraud rate differs between train ({train_fraud:.1f}%) "
              f"and test ({test_fraud:.1f}%) by {abs(train_fraud-test_fraud):.1f} points.")
        print("  This is expected with temporal split — fraud patterns evolve over time.")
        print("  The model must generalise to this shift, which is the correct test.")

    # Source distribution check
    if "source" in df.columns:
        print(f"\n  Source distribution in test split:")
        for src, count in test_df["source"].value_counts().items():
            pct = count / len(test_df) * 100
            print(f"    {src:<25} {count:>8,} ({pct:.1f}%)")

    # GNN embedding coverage check
    gnn_cols = [c for c in df.columns if c.startswith("gnn_emb_")]
    if gnn_cols:
        zero_rows = (test_df[gnn_cols[0]] == 0.0).sum()
        print(f"\n  GNN embedding coverage in test: "
              f"{len(test_df) - zero_rows:,}/{len(test_df):,} "
              f"({(1 - zero_rows/len(test_df))*100:.1f}%)")

    # Save
    Path("data/training").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet("data/training/train.parquet", index=False)
    val_df.to_parquet(  "data/training/val.parquet",   index=False)
    test_df.to_parquet( "data/training/test.parquet",  index=False)

    print(f"\nSaved:")
    print(f"  data/training/train.parquet  ({len(train_df):,} rows)")
    print(f"  data/training/val.parquet    ({len(val_df):,} rows)")
    print(f"  data/training/test.parquet   ({len(test_df):,} rows)")
    print(f"\nNext: python -m ml.pipeline")


if __name__ == "__main__":
    main()