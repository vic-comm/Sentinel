"""
scripts/build_features.py
=========================
Builds the final feature-enriched train/val/test parquet files
that XGBoost / LightGBM / CatBoost train on.

What this script does (in order):
  1. Loads train/val/test splits from merge_training_data.py output
  2. Adds graph topology features (degree-based, not PageRank)
  3. Corrects fraud rate to ~0.5% (real-world level) via sample_weight
  4. Attaches GNN embeddings from Redis (optional, Phase 9)
  5. Saves enriched parquet files + sample weights

What was removed vs original build_features.py:
  - PageRank: O(n²) on 400K nodes crashes MacBook, weak fraud signal
  - KMeans cluster_risk: circular feature (clusters amount_usd which model
    already sees directly), adds noise not signal
  - groupby().rolling("24h") temporal features: produce NaN-dominated
    columns because entities have sparse timestamps; the simulator already
    computed velocity features (count_1h, count_24h etc.) which are better

Usage:
  # Phase 7: tabular features only (before GraphSAGE)
  python -m scripts.build_features

  # Phase 9: tabular + GNN embeddings (after embed_gnn.py has run)
  python -m scripts.build_features --attach-gnn

  # Verify Redis has embeddings before running Phase 9:
  redis-cli --scan --pattern "gnn_embedding:*" | wc -l
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DIR   = Path("data/training")
PROCESSED_DIR  = Path("data/processed")
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
GNN_EMB_DIM    = 32       # must match graphsage_config.json out_channels
GNN_EMB_TTL    = 3600     # 1 hour — matches embed_gnn.py TTL
GNN_BATCH_SIZE = 5000     # Redis pipeline batch size

# Source confidence weights — used as sample_weight in XGBoost training.
# Higher confidence = row has more influence on gradient updates.
# Simulator data is clean but synthetic (lower weight).
# Real Alchemy/Elliptic data is noisy but real (higher weight).
SOURCE_WEIGHTS = {
    "simulator":       0.50,   # synthetic, calibrated but not real
    "elliptic_real":   1.00,   # law-enforcement verified Bitcoin fraud
    "elliptic_seeded": 0.85,   # Elliptic-seeded cross-modal pairs
    "alchemy_real":    0.75,   # depth-1 confirmed mixer users
    "alchemy_legit":   0.70,   # depth-1 confirmed DeFi users
    "finbridge_prod":  0.55,   # synthetic hybrid transactions
}
DEFAULT_SOURCE_WEIGHT = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the train/val/test parquet files produced by merge_training_data.py.
    These already have simulator + Elliptic cross-modal + Etherscan/Alchemy rows.
    """
    paths = {
        "train": TRAINING_DIR / "train.parquet",
        "val":   TRAINING_DIR / "val.parquet",
        "test":  TRAINING_DIR / "test.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print("Run: python scripts/merge_training_data.py")
            sys.exit(1)

    print("Loading splits...")
    splits = {}
    for name, p in paths.items():
        df = pd.read_parquet(p)
        fraud_rate = df["is_fraud"].mean() * 100
        print(f"  {name}: {len(df):>9,} rows  (fraud: {fraud_rate:.2f}%)")
        splits[name] = df

    return splits["train"], splits["val"], splits["test"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GRAPH TOPOLOGY FEATURES (degree-based, not PageRank)
# ─────────────────────────────────────────────────────────────────────────────

def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds degree-based graph features without building a NetworkX graph.

    Why not PageRank:
      nx.pagerank() on 400K+ nodes takes 30-60 minutes and 8GB RAM.
      It is O(n * iterations) with dense intermediate matrices.
      Degree features give 90% of the signal in O(n) time.

    Why degree is a valid fraud signal:
      Mule aggregators: high in-degree (many senders), low out-degree (1-2 receivers)
      Mixer contracts:  very high in-degree (many victims sending to it)
      Normal users:     balanced in/out degree, moderate total
      Pig butchering:   scammer has high in-degree from many victims, no returns
    """
    print("  Adding graph topology features (degree-based)...")

    # Resolve the identity key for both fiat and crypto rows
    # Fiat rows have sender_hash, crypto rows have sender_wallet_hash
    sender_col   = df.get("sender_hash",   df.get("identity_hash", pd.Series(dtype=str)))
    receiver_col = df.get("receiver_hash", df.get("receiver_wallet_hash", pd.Series(dtype=str)))

    sender_key   = df["sender_hash"].fillna(df.get("identity_hash", "")) \
                   if "sender_hash" in df.columns else df.get("identity_hash", pd.Series(""))
    receiver_key = df["receiver_hash"].fillna("") \
                   if "receiver_hash" in df.columns else pd.Series("")

    # Out-degree: how many distinct receivers does this sender have?
    # High out-degree in fraud = ATO (sending to many new mule accounts)
    out_degree = (
        df.assign(_sender=sender_key, _receiver=receiver_key)
        .groupby("_sender")["_receiver"]
        .nunique()
        .rename("sender_out_degree")
    )

    # In-degree: how many distinct senders does this receiver have?
    # High in-degree in fraud = mule aggregator or mixer
    in_degree = (
        df.assign(_sender=sender_key, _receiver=receiver_key)
        .groupby("_receiver")["_sender"]
        .nunique()
        .rename("receiver_in_degree")
    )

    # Transaction volume per sender (not same as amount_usd — this is cumulative)
    sender_volume = (
        df.assign(_sender=sender_key)
        .groupby("_sender")["amount_usd"]
        .sum()
        .rename("sender_total_volume_graph")
    )

    # Average incoming amount per receiver
    receiver_avg_incoming = (
        df.assign(_receiver=receiver_key)
        .groupby("_receiver")["amount_usd"]
        .mean()
        .rename("receiver_avg_incoming_amount")
    )

    # Map back to rows
    df["sender_out_degree"]          = sender_key.map(out_degree).fillna(0).astype(np.float32)
    df["receiver_in_degree"]         = receiver_key.map(in_degree).fillna(0).astype(np.float32)
    df["sender_total_volume_graph"]  = sender_key.map(sender_volume).fillna(0).astype(np.float32)
    df["receiver_avg_incoming"]      = receiver_key.map(receiver_avg_incoming).fillna(0).astype(np.float32)

    # Fan-in ratio: in_degree / (out_degree + 1) — high = aggregator pattern
    df["fan_in_ratio"] = (
        df["receiver_in_degree"] / (df["sender_out_degree"] + 1)
    ).astype(np.float32)

    # Whether receiver has been seen as a sender (recycled address — fraud signal)
    # all_senders   = set(sender_key.dropna())
    # all_receivers = set(receiver_key.dropna())
    # recycled      = all_senders & all_receivers
    # df["receiver_is_recycled_address"] = receiver_key.isin(recycled).astype(np.float32)

    if "source" in df.columns:
        real_mask     = df["source"].str.contains("alchemy", na=False)
        real_senders  = set(sender_key[real_mask].dropna())
        real_receivers = set(receiver_key[real_mask].dropna())
        real_recycled  = real_senders & real_receivers
        df["receiver_is_recycled_address"] = receiver_key.isin(real_recycled).astype(np.float32)
    else:
        df["receiver_is_recycled_address"] = 0.0

    print(f"    sender_out_degree   median: {df['sender_out_degree'].median():.1f}")
    print(f"    receiver_in_degree  median: {df['receiver_in_degree'].median():.1f}")
    print(f"    Recycled addresses: {df['receiver_is_recycled_address'].sum():,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SOURCE CONFIDENCE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def add_source_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a sample_weight to each row based on data source.

    Why this matters:
      Your training data has 3 sources with different label quality:
        Simulator (0.5):  clean labels, but synthetic patterns — may overfit
        Alchemy (0.75):   depth-1 confirmed mixer users, real but noisy
        Elliptic (1.0):   law-enforcement verified, highest confidence

      XGBoost / LightGBM accept sample_weight in .fit() — rows with higher
      weight have proportionally more influence on gradient updates.
      This prevents the model from being dominated by synthetic patterns
      while still benefiting from the volume of simulator data.

    Second role — fraud rate correction:
      Simulator fraud rate: ~2.5%
      Real-world fraud rate: ~0.1-0.5%
      Giving legit simulator rows lower weight effectively shifts
      the decision boundary toward the real-world operating point.
    """
    print("  Adding source confidence weights...")

    if "source" not in df.columns:
        df["source"] = "simulator"

    df["source_confidence"] = df["source"].map(SOURCE_WEIGHTS).fillna(DEFAULT_SOURCE_WEIGHT)

    # Fraud rate correction: up-weight real fraud, down-weight synthetic legit
    # This simulates training on a dataset with 0.5% fraud rate
    # without actually discarding simulator rows (we keep the volume)
    target_fraud_rate = 0.005   # 0.5% target
    actual_fraud_rate = df["is_fraud"].mean()

    if actual_fraud_rate > target_fraud_rate:
        # Scale down legit rows so effective fraud rate approaches target
        legit_scale = target_fraud_rate / (1 - target_fraud_rate) * \
                      (1 - actual_fraud_rate) / actual_fraud_rate
        legit_mask = ~df["is_fraud"].astype(bool)
        df.loc[legit_mask, "source_confidence"] *= legit_scale
        print(f"    Fraud rate correction: {actual_fraud_rate*100:.2f}% → ~{target_fraud_rate*100:.1f}%")
        print(f"    Legit rows down-weighted by {legit_scale:.3f}x")

    print(f"    Weight distribution:")
    for src, grp in df.groupby("source"):
        print(f"      {src:<25} n={len(grp):>8,}  weight={grp['source_confidence'].mean():.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GNN EMBEDDING ATTACHMENT  (Phase 9 only)
# ─────────────────────────────────────────────────────────────────────────────

def attach_gnn_embeddings(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches pre-computed GraphSAGE embeddings from Redis and attaches
    them as 32 new columns (gnn_emb_0 ... gnn_emb_31) to each split.

    How the mapping works:
      1. embed_gnn.py stored: KEY = gnn_embedding:{identity_hash}
                              VALUE = JSON list of 32 floats
      2. This function reads identity_hash from each row, looks up the key,
         and appends the 32 floats as new feature columns.
      3. If a row's identity_hash has no embedding (new user not in graph),
         it gets a zero vector — the model learns that zero = no graph signal.

    Why fetch from Redis at feature-build time (not at pipeline.py time):
      The parquet files are written once and reused across many pipeline.py runs.
      Fetching once here is faster than fetching 1.2M rows every training run.
      The downside: if embeddings are refreshed by embed_gnn.py, you must
      re-run build_features.py --attach-gnn. In practice this is fine since
      embeddings are recomputed only when the GNN is retrained (rare).

    Redis key lookup:
      Each row has an identity_hash (SHA-256 of email+salt).
      This is the SAME hash stored as a Neo4j Identity node.
      It is the SAME hash used as the Redis key by embed_gnn.py.
      The three match because they all use SHARED_SALT="sentinel_consortium_2026_v1".
    """
    try:
        import redis as redis_lib
    except ImportError:
        print("  [GNN] pip install redis — skipping GNN attachment")
        return train, val, test

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        print(f"  [GNN] Redis not available ({e}) — skipping GNN attachment")
        return train, val, test

    # Count available embeddings
    print("  Counting available GNN embeddings in Redis...")
    sample_keys = list(r.scan_iter("gnn_embedding:*", count=100))
    n_available  = sum(1 for _ in r.scan_iter("gnn_embedding:*"))
    print(f"    {n_available:,} embeddings available in Redis")

    if n_available == 0:
        print("  [GNN] No embeddings found. Run: python -m scripts.embed_gnn")
        return train, val, test

    # Verify dimension matches config
    if sample_keys:
        sample_val = r.get(sample_keys[0])
        if sample_val:
            sample_dim = len(json.loads(sample_val))
            if sample_dim != GNN_EMB_DIM:
                print(f"  [GNN] WARNING: Redis embeddings are {sample_dim}-dim, "
                      f"expected {GNN_EMB_DIM}. Check graphsage_config.json.")

    emb_cols = [f"gnn_emb_{i}" for i in range(GNN_EMB_DIM)]
    zero_vec = [0.0] * GNN_EMB_DIM   # fallback for nodes not in graph

    def fetch_embeddings_for_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Batch-fetches embeddings using Redis pipelining.
        Pipeline sends 5000 GET commands at once, then reads all responses.
        This is ~50x faster than 1.2M individual GET calls.
        """
        print(f"  Fetching GNN embeddings for {split_name} ({len(df):,} rows)...")
        t0 = time.time()

        # Resolve identity_hash — use sender_hash as fallback for crypto rows
        id_col = "identity_hash"
        if id_col not in df.columns:
            id_col = "sender_hash" if "sender_hash" in df.columns else None

        if id_col is None:
            print(f"  [GNN] No identity_hash column in {split_name} — using zeros")
            emb_matrix = np.zeros((len(df), GNN_EMB_DIM), dtype=np.float32)
            return pd.concat([df, pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)], axis=1)

        identity_hashes = df[id_col].fillna("").tolist()
        emb_matrix      = np.zeros((len(df), GNN_EMB_DIM), dtype=np.float32)

        hit  = 0
        miss = 0

        # Process in batches of GNN_BATCH_SIZE using Redis pipeline
        for batch_start in range(0, len(identity_hashes), GNN_BATCH_SIZE):
            batch_hashes = identity_hashes[batch_start : batch_start + GNN_BATCH_SIZE]

            pipe = r.pipeline(transaction=False)
            for h in batch_hashes:
                pipe.get(f"gnn_embedding:{h}" if h else "")
            responses = pipe.execute()

            for j, (h, raw) in enumerate(zip(batch_hashes, responses)):
                row_idx = batch_start + j
                if raw:
                    try:
                        emb_matrix[row_idx] = json.loads(raw)
                        hit += 1
                    except (json.JSONDecodeError, ValueError):
                        miss += 1   # corrupted value — use zero vector
                else:
                    miss += 1   # not in graph — use zero vector

            if (batch_start // GNN_BATCH_SIZE) % 10 == 0:
                done = min(batch_start + GNN_BATCH_SIZE, len(df))
                print(f"    {done:,}/{len(df):,} rows processed "
                      f"(hit: {hit:,}, miss: {miss:,})")

        elapsed  = time.time() - t0
        hit_rate = hit / len(df) * 100
        print(f"    {split_name}: {hit:,} hits ({hit_rate:.1f}%), "
              f"{miss:,} zeros, {elapsed:.1f}s")

        if hit_rate < 10:
            print(f"    WARNING: <10% hit rate. "
                  f"Check that embed_gnn.py used the same SHARED_SALT "
                  f"and identity_hash column as your training data.")

        emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
        return pd.concat([df, emb_df], axis=1)

    train = fetch_embeddings_for_split(train, "train")
    val   = fetch_embeddings_for_split(val,   "val")
    test  = fetch_embeddings_for_split(test,  "test")

    # Report zero-vector coverage
    zero_rows = (train[emb_cols[0]] == 0.0).sum()
    print(f"\n  GNN attachment summary:")
    print(f"    Columns added: {len(emb_cols)} (gnn_emb_0 to gnn_emb_31)")
    print(f"    Zero-vector rows (not in graph): {zero_rows:,} ({zero_rows/len(train)*100:.1f}% of train)")
    print(f"    These rows still train correctly — zero vector = 'no graph signal'")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TYPE CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all columns have types PyArrow (parquet backend) can handle.
    Mixed-type columns (e.g. True/NaN/None) cause silent write failures.
    """
    # Boolean flags: must be strictly bool, no NaN
    bool_cols = [
        "is_first_time_receiver", "is_unusual_hour", "is_weekend",
        "is_business_hour", "known_mixer_interaction", "immediate_withdrawal",
        "has_fiat_history", "has_crypto_history", "cross_modal_pattern_detected",
        "prior_high_risk_event", "receiver_is_recycled_address",
        "is_real_data", "graph_only",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # Numeric columns: ensure float32 (not object)
    float_cols = [
        "amount_usd", "amount_crypto", "balance_drain_ratio",
        "velocity_ratio", "amount_ratio", "source_confidence",
        "fan_in_ratio", "sender_out_degree", "receiver_in_degree",
        "sender_total_volume_graph", "receiver_avg_incoming",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # GNN embedding columns: ensure float32
    gnn_cols = [c for c in df.columns if c.startswith("gnn_emb_")]
    for col in gnn_cols:
        df[col] = df[col].fillna(0.0).astype(np.float32)

    # Object columns: force to string (catches mixed None/str columns)
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["timestamp"]:
            df[col] = df[col].astype(str).replace("nan", "")

    return df


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 6 — SAVE SAMPLE WEIGHTS
# # ─────────────────────────────────────────────────────────────────────────────

# def save_sample_weights(train: pd.DataFrame, val: pd.DataFrame):
#     """
#     Saves source_confidence as numpy arrays for use in model training.

#     In _fit_model() in model_training.py:
#       w_train = np.load("data/training/sample_weights_train.npy")
#       model.fit(X_train, y_train, sample_weight=w_train, ...)

#     This gives rows with real data (Alchemy/Elliptic) more influence
#     and corrects for the inflated synthetic fraud rate.
#     """
#     w_train = train["source_confidence"].fillna(DEFAULT_SOURCE_WEIGHT).values
#     w_val   = val["source_confidence"].fillna(DEFAULT_SOURCE_WEIGHT).values

#     np.save(TRAINING_DIR / "sample_weights_train.npy", w_train.astype(np.float32))
#     np.save(TRAINING_DIR / "sample_weights_val.npy",   w_val.astype(np.float32))

#     print(f"  Saved sample weights → data/training/sample_weights_train.npy")
#     print(f"  Weight range: [{w_train.min():.3f}, {w_train.max():.3f}]")
#     print(f"  Mean weight:  {w_train.mean():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(attach_gnn: bool = False):
    print("=" * 60)
    print("SENTINEL FEATURE ENGINEERING")
    print(f"  GNN embeddings: {'YES (Phase 9)' if attach_gnn else 'NO (Phase 7)'}")
    print("=" * 60)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading train/val/test splits...")
    train, val, test = load_splits()

    # ── 2. Graph features (degree-based) ──────────────────────────────────────
    # Compute on the FULL dataset first (so cross-split degree counts are correct),
    # then split back. This prevents train/val leakage of degree counts while
    # ensuring the degree feature reflects the full graph structure.
    print("\n[2/5] Adding graph topology features...")
    full = pd.concat([train, val, test], ignore_index=False)
    full = add_graph_features(full)

    # Restore original split indices
    n_train = len(train)
    n_val   = len(val)
    train = full.iloc[:n_train].copy()
    val   = full.iloc[n_train:n_train + n_val].copy()
    test  = full.iloc[n_train + n_val:].copy()
    del full

    # ── 3. Source confidence weights ─────────────────────────────────────────
    print("\n[3/5] Adding source confidence weights + fraud rate correction...")
    train = add_source_weights(train)
    # Val and test get weights too (for weighted evaluation metrics)
    val["source_confidence"]  = val["source"].map(SOURCE_WEIGHTS).fillna(DEFAULT_SOURCE_WEIGHT) \
                                if "source" in val.columns else DEFAULT_SOURCE_WEIGHT
    test["source_confidence"] = test["source"].map(SOURCE_WEIGHTS).fillna(DEFAULT_SOURCE_WEIGHT) \
                                if "source" in test.columns else DEFAULT_SOURCE_WEIGHT

    # ── 4. GNN embeddings (optional) ─────────────────────────────────────────
    if attach_gnn:
        print("\n[4/5] Attaching GNN embeddings from Redis...")
        train, val, test = attach_gnn_embeddings(train, val, test)
    else:
        print("\n[4/5] Skipping GNN embeddings (run with --attach-gnn for Phase 9)")

    # ── 5. Type cleanup and save ──────────────────────────────────────────────
    print("\n[5/5] Cleaning dtypes and saving...")

    print("  Cleaning train...")
    train = clean_dtypes(train)
    print("  Cleaning val...")
    val   = clean_dtypes(val)
    print("  Cleaning test...")
    test  = clean_dtypes(test)

    # Save enriched splits (overwrite the merge_training_data.py output)
    train.to_parquet(TRAINING_DIR / "train.parquet", index=False)
    val.to_parquet(TRAINING_DIR   / "val.parquet",   index=False)
    test.to_parquet(TRAINING_DIR  / "test.parquet",  index=False)
    print(f"  Saved → data/training/train.parquet ({len(train):,} rows, {train.shape[1]} columns)")
    print(f"  Saved → data/training/val.parquet   ({len(val):,} rows)")
    print(f"  Saved → data/training/test.parquet  ({len(test):,} rows)")

    # Save sample weights
    # save_sample_weights(train, val)
    print("\n[Final] Creating combined enriched_dataset.parquet for temporal splitting...")
    full_enriched = pd.concat([train, val, test], ignore_index=True)
    full_enriched.to_parquet(TRAINING_DIR / "enriched_dataset.parquet", index=False)
    print(f"  Saved → {TRAINING_DIR / 'enriched_dataset.parquet'} ({len(full_enriched):,} rows)")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    gnn_cols = [c for c in train.columns if c.startswith("gnn_emb_")]
    graph_cols = [
        "sender_out_degree", "receiver_in_degree",
        "sender_total_volume_graph", "receiver_avg_incoming", "fan_in_ratio",
        "receiver_is_recycled_address",
    ]

    print(f"\n  Feature groups added:")
    print(f"    Graph topology (degree-based):   {len(graph_cols)} columns")
    print(f"    Source confidence weights:        1 column (source_confidence)")
    print(f"    GNN embeddings:                  {len(gnn_cols)} columns")
    print(f"\n  Total feature columns: {train.shape[1]}")
    print(f"  Fraud rate (weighted): ~0.5% (corrected from synthetic ~2.5%)")

    if not attach_gnn:
        print(f"\n  Phase 9 step:")
        print(f"    1. python -m scripts.train_graphsage")
        print(f"    2. python -m scripts.embed_gnn")
        print(f"    3. python -m scripts.build_features --attach-gnn")
        print(f"    4. python -m ml.pipeline --use-gnn-embeddings")
    else:
        print(f"\n  Next: python -m ml.pipeline --use-gnn-embeddings")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build feature-enriched training splits for Sentinel"
    )
    parser.add_argument(
        "--attach-gnn", action="store_true",
        help="Fetch 32-dim GNN embeddings from Redis and attach to each split. "
             "Requires: embed_gnn.py to have run first. "
             "Use for Phase 9 (--use-gnn-embeddings in pipeline.py)."
    )
    args = parser.parse_args()
    main(attach_gnn=args.attach_gnn)