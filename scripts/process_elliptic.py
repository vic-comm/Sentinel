# scripts/process_elliptic.py
import pandas as pd
import numpy as np
import hashlib, uuid, random
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

# ── Merge labels ─────────────────────────────────────────────────────────────
df = features.merge(classes, left_on=0, right_on="txId", how="left")
labeled   = df[df["class"].isin(["1", "2"])].copy()
illicit   = labeled[labeled["class"] == "1"]   # 4,545 confirmed fraud
licit     = labeled[labeled["class"] == "2"]   # 42,019 confirmed legit

print(f"Illicit: {len(illicit):,}")
print(f"Licit:   {len(licit):,}")

# ── Build PyTorch Geometric graph for GraphSAGE ───────────────────────────────
print("\nBuilding PyG graph...")
all_node_ids = pd.concat([edgelist["txId1"], edgelist["txId2"]]).unique()
node_to_idx  = {nid: i for i, nid in enumerate(all_node_ids)}

edges_src = [node_to_idx.get(n, 0) for n in edgelist["txId1"]]
edges_dst = [node_to_idx.get(n, 0) for n in edgelist["txId2"]]
edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

# Node features: columns 2–167 (166 features)
feat_matrix = labeled.iloc[:, 2:168].values.astype(np.float32)
feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

# Normalize each feature column to [0, 1]
col_min  = feat_matrix.min(axis=0)
col_max  = feat_matrix.max(axis=0)
col_rng  = np.where(col_max - col_min > 0, col_max - col_min, 1)
feat_matrix = (feat_matrix - col_min) / col_rng

x = torch.tensor(feat_matrix, dtype=torch.float)
y = torch.tensor((labeled["class"] == "1").astype(int).values, dtype=torch.long)

# Train/val/test masks (70/15/15)
n = len(labeled)
idx = torch.randperm(n)
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[idx[:int(n*0.70)]] = True
val_mask  [idx[int(n*0.70):int(n*0.85)]] = True
test_mask [idx[int(n*0.85):]] = True

pyg_data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)
torch.save(pyg_data, "data/processed/elliptic_pyg.pt")
print(f"Saved PyG graph: {pyg_data.num_nodes} nodes, {pyg_data.num_edges} edges")

# ── Generate 500 cross-modal seeds ───────────────────────────────────────────
print("\nGenerating 500 cross-modal seeds from illicit transactions...")
seeds = illicit.sample(500, random_state=42)
cross_modal_rows = []

for _, row in seeds.iterrows():
    fake_email   = f"elliptic_user_{int(row[0])}@example.com"
    identity_h   = hashlib.sha256(f"{fake_email}{SHARED_SALT}".encode()).hexdigest()
    bridge_id    = str(uuid.uuid4())

    # Scale amount from anonymized Elliptic feature[2]
    raw_val    = abs(float(row.iloc[2]))
    amount_usd = max(5_000, min(500_000, raw_val * 35_000))

    # Time step → timestamp (each step ≈ 2 weeks)
    time_step  = int(row.iloc[1])
    base_ts    = datetime(2026, 1, 1) + timedelta(weeks=time_step * 2)
    crypto_ts  = base_ts + timedelta(hours=random.uniform(0, 48))
    fiat_ts    = crypto_ts - timedelta(minutes=random.randint(10, 60))

    # Fiat leg (will be merged with NeoBank data)
    cross_modal_rows.append({
        "transaction_id":          f"elliptic_fiat_{uuid.uuid4().hex[:10]}",
        "client_id":               "neobank_prod",
        "timestamp":               fiat_ts.isoformat() + "Z",
        "modality":                "fiat",
        "identity_hash":           identity_h,
        "sender_hash":             identity_h,
        "receiver_hash":           hashlib.sha256(f"mule_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              round(amount_usd, 2),
        "transfer_type":           "wire",
        "transfer_network":        "international",
        "is_fraud":                True,
        "fraud_type":              "laundering_fiat_leg",
        "cross_modality_fraud_id": bridge_id,
        "source":                  "elliptic_seeded",
        # fill required numeric fields with fraud-consistent values
        "sender_balance_before":   round(amount_usd * 1.05, 2),
        "sender_balance_after":    round(amount_usd * 0.02, 2),
        "balance_drain_ratio":     0.97,
        "sender_account_age_days": random.randint(30, 180),
        "is_first_time_receiver":  True,
        "is_unusual_hour":         True,
        "velocity_ratio":          round(random.uniform(8, 25), 2),
        "amount_ratio":            round(random.uniform(4, 15), 2),
    })

    # Crypto leg (backed by real Elliptic features)
    cross_modal_rows.append({
        "transaction_id":          f"elliptic_crypto_{uuid.uuid4().hex[:10]}",
        "client_id":               "cryptoex_prod",
        "timestamp":               crypto_ts.isoformat() + "Z",
        "modality":                "crypto",
        "identity_hash":           identity_h,   # SAME HASH — the critical link
        "user_email_hash":         identity_h,
        "sender_wallet_hash":      hashlib.sha256(f"wallet_{row[0]}".encode()).hexdigest(),
        "receiver_wallet_hash":    hashlib.sha256(f"fresh_{uuid.uuid4()}".encode()).hexdigest(),
        "amount_usd":              round(amount_usd * 0.97, 2),
        "amount_crypto":           round(amount_usd * 0.97 / 2800, 6),
        "cryptocurrency":          "BTC",
        "blockchain":              "bitcoin",
        "transaction_hash":        f"0x{uuid.uuid4().hex}",
        "known_mixer_interaction": True,
        "receiver_wallet_age_days": 0,
        "is_fraud":                True,
        "fraud_type":              "laundering_crypto_leg",
        "cross_modality_fraud_id": bridge_id,
        "source":                  "elliptic_real",
        # Store real Elliptic features as additional signal
        "elliptic_feature_2":      float(row.iloc[2]),
        "elliptic_feature_3":      float(row.iloc[3]),
        "elliptic_feature_4":      float(row.iloc[4]),
    })

cross_df = pd.DataFrame(cross_modal_rows)
cross_df.to_json("data/processed/elliptic_cross_modal.jsonl",
                  orient="records", lines=True)
print(f"Saved {len(cross_df):,} rows → data/processed/elliptic_cross_modal.jsonl")
print(f"  ({len(cross_df)//2} fiat legs + {len(cross_df)//2} crypto legs)")