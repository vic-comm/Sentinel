"""
scripts/process_elliptic.py — Path B
=====================================
Changes from original:
  - Fiat leg: realistic feature values (not hardcoded fraud signals)
  - transfer_type: drawn from realistic distribution
  - balance_drain_ratio: wide range overlapping legitimate
  - velocity_ratio: overlaps legitimate range
  - is_unusual_hour: 30% chance, not always True
  - is_first_time_receiver: 60% chance, not always True
  - sender_account_age_days: full realistic range (not just 30-180)

What still makes these detectable:
  - cross_modality_fraud_id: links fiat and crypto legs
  - The crypto side has real Elliptic features (confirmed illicit)
  - Graph proximity to known illicit transactions in the Elliptic graph
"""

import pandas as pd
import numpy as np
import hashlib
import uuid
import random
from datetime import datetime, timedelta
import torch
from torch_geometric.data import Data
from pathlib import Path

SHARED_SALT = "sentinel_consortium_2026_v1"
Path("data/processed").mkdir(exist_ok=True)

print("Loading Elliptic files...")
features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
classes  = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
edgelist = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

df       = features.merge(classes, left_on=0, right_on="txId", how="left")
labeled  = df[df["class"].isin(["1", "2"])].copy()
illicit  = labeled[labeled["class"] == "1"]
licit    = labeled[labeled["class"] == "2"]

print(f"Illicit: {len(illicit):,}")
print(f"Licit:   {len(licit):,}")

# ── Build PyG graph ───────────────────────────────────────────────────────────
print("\nBuilding PyG graph...")
all_node_ids = pd.concat([edgelist["txId1"], edgelist["txId2"]]).unique()
node_to_idx  = {nid: i for i, nid in enumerate(all_node_ids)}

edge_index = torch.tensor(
    [[node_to_idx.get(n, 0) for n in edgelist["txId1"]],
     [node_to_idx.get(n, 0) for n in edgelist["txId2"]]],
    dtype=torch.long
)

feat_matrix = labeled.iloc[:, 2:168].values.astype(np.float32)
feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)
col_rng     = np.where(
    feat_matrix.max(axis=0) - feat_matrix.min(axis=0) > 0,
    feat_matrix.max(axis=0) - feat_matrix.min(axis=0),
    1
)
feat_matrix = (feat_matrix - feat_matrix.min(axis=0)) / col_rng

x = torch.tensor(feat_matrix, dtype=torch.float)
y = torch.tensor((labeled["class"] == "1").astype(int).values, dtype=torch.long)

n   = len(labeled)
idx = torch.randperm(n)
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[idx[:int(n*0.70)]] = True
val_mask  [idx[int(n*0.70):int(n*0.85)]] = True
test_mask [idx[int(n*0.85):]] = True

pyg_data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
torch.save(pyg_data, "data/processed/elliptic_pyg.pt")
print(f"Saved PyG graph: {pyg_data.num_nodes} nodes, {pyg_data.num_edges} edges")

# ── Generate 500 cross-modal seeds ────────────────────────────────────────────
print("\nGenerating 500 cross-modal seeds from illicit transactions...")
seeds            = illicit.sample(500, random_state=42)
cross_modal_rows = []

# Compute realistic ranges from the licit population for reference
# so our fiat legs look like legitimate transactions with realistic variance
licit_amounts = licit.iloc[:, 2].abs().clip(upper=500000).values

TRANSFER_TYPES = ["ach", "wire", "p2p", "rtp"]
TRANSFER_WEIGHTS = [0.60, 0.20, 0.15, 0.05]

for _, row in seeds.iterrows():
    fake_email = f"elliptic_user_{int(row[0])}@example.com"
    identity_h = hashlib.sha256(f"{fake_email}{SHARED_SALT}".encode()).hexdigest()
    bridge_id  = str(uuid.uuid4())

    raw_val    = abs(float(row.iloc[2]))
    amount_usd = max(5_000, min(500_000, raw_val * 35_000))

    time_step = int(row.iloc[1])
    base_ts   = datetime(2026, 1, 1) + timedelta(weeks=time_step * 2)
    crypto_ts = base_ts + timedelta(hours=random.uniform(0, 48))
    fiat_ts   = crypto_ts - timedelta(minutes=random.randint(10, 60))

    # ── PATH B: Realistic fiat leg values ─────────────────────────────────────
    # These should NOT be obvious fraud flags. The real signal is:
    #   1. cross_modality_fraud_id linking to confirmed illicit crypto tx
    #   2. Graph proximity to illicit transactions in Elliptic graph
    #   3. The 500 fiat legs, when viewed as a group, have a temporal cluster

    # Account age: full realistic range — not fraud-specific short range
    sender_account_age = random.randint(30, 3650)

    # Balance drain: wide range overlapping legitimate large transfers
    # Real laundering often uses partial amounts to avoid triggers
    balance_drain = round(random.uniform(0.10, 0.75), 3)

    # Velocity ratio: overlaps with legitimate users (freelancers, small biz)
    # Real laundering activity looks like an active business user
    velocity_ratio_val = round(random.uniform(0.5, 4.0), 2)

    # Amount ratio: some are high (above their baseline), many are not
    # Laundering amounts are calibrated to look like normal business activity
    amount_ratio_val = round(random.uniform(0.8, 6.0), 2)

    # Unusual hour: 30% chance — real fraudsters work business hours too
    is_unusual = random.random() < 0.30

    # First time receiver: 60% chance — reasonable for a wire transfer
    is_first_time = random.random() < 0.60

    # Transfer type: realistic distribution
    transfer_type = random.choices(TRANSFER_TYPES, weights=TRANSFER_WEIGHTS)[0]

    # Transfer network: wires and ACH go international sometimes
    transfer_network = random.choices(
        ["same_bank", "domestic", "international"],
        weights=[0.30, 0.50, 0.20]
    )[0]

    cross_modal_rows.append({
        "transaction_id":          f"elliptic_fiat_{uuid.uuid4().hex[:10]}",
        "client_id":               "neobank_prod",
        "timestamp":               fiat_ts.isoformat() + "Z",
        "modality":                "fiat",
        "identity_hash":           identity_h,
        "sender_hash":             identity_h,
        "receiver_hash":           hashlib.sha256(f"mule_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              round(amount_usd, 2),
        "transfer_type":           transfer_type,
        "transfer_network":        transfer_network,
        "is_fraud":                True,
        "fraud_type":              "laundering_fiat_leg",
        "cross_modality_fraud_id": bridge_id,
        "source":                  "elliptic_seeded",
        # PATH B: realistic values — NOT hardcoded fraud signals
        "sender_balance_before":   round(amount_usd / max(balance_drain, 0.01), 2),
        "sender_balance_after":    round(amount_usd / max(balance_drain, 0.01) * (1 - balance_drain), 2),
        "balance_drain_ratio":     balance_drain,
        "sender_account_age_days": sender_account_age,
        "is_first_time_receiver":  is_first_time,
        "is_unusual_hour":         is_unusual,
        "velocity_ratio":          velocity_ratio_val,
        "amount_ratio":            amount_ratio_val,
        "hour_of_day":             fiat_ts.hour,
        "day_of_week":             fiat_ts.weekday(),
        "is_weekend":              fiat_ts.weekday() >= 5,
    })

    # Crypto leg: real Elliptic features (unchanged — this is the real signal)
    cross_modal_rows.append({
        "transaction_id":          f"elliptic_crypto_{uuid.uuid4().hex[:10]}",
        "client_id":               "cryptoex_prod",
        "timestamp":               crypto_ts.isoformat() + "Z",
        "modality":                "crypto",
        "identity_hash":           identity_h,
        "user_email_hash":         identity_h,
        "sender_wallet_hash":      hashlib.sha256(f"wallet_{row[0]}".encode()).hexdigest(),
        "receiver_wallet_hash":    hashlib.sha256(f"fresh_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              round(amount_usd * 0.97, 2),
        "amount_crypto":           round(amount_usd * 0.97 / 2800, 6),
        "cryptocurrency":          "BTC",
        "blockchain":              "bitcoin",
        "transaction_hash":        f"0x{uuid.uuid4().hex}",
        "known_mixer_interaction": True,
        "receiver_wallet_age_days": random.randint(0, 7),   # fresh wallet — legitimate signal
        "is_fraud":                True,
        "fraud_type":              "laundering_crypto_leg",
        "cross_modality_fraud_id": bridge_id,
        "source":                  "elliptic_real",
        "elliptic_feature_2":      float(row.iloc[2]),
        "elliptic_feature_3":      float(row.iloc[3]),
        "elliptic_feature_4":      float(row.iloc[4]),
    })

cross_df = pd.DataFrame(cross_modal_rows)
cross_df.to_json("data/processed/elliptic_cross_modal.jsonl",
                 orient="records", lines=True)

# Verify no perfectly separating values leaked in
fiat_rows = cross_df[cross_df["modality"] == "fiat"]
print(f"\nFiat leg feature distribution check (should overlap with legitimate):")
print(f"  balance_drain_ratio: min={fiat_rows['balance_drain_ratio'].min():.2f}, max={fiat_rows['balance_drain_ratio'].max():.2f}, mean={fiat_rows['balance_drain_ratio'].mean():.2f}")
print(f"  velocity_ratio:      min={fiat_rows['velocity_ratio'].min():.2f}, max={fiat_rows['velocity_ratio'].max():.2f}")
print(f"  is_unusual_hour:     {fiat_rows['is_unusual_hour'].mean()*100:.0f}% unusual (target: ~30%)")
print(f"  is_first_time:       {fiat_rows['is_first_time_receiver'].mean()*100:.0f}% first time (target: ~60%)")
print(f"\nSaved {len(cross_df):,} rows → data/processed/elliptic_cross_modal.jsonl")
print(f"  ({len(cross_df)//2} fiat legs + {len(cross_df)//2} crypto legs)")