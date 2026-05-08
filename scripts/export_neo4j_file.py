# # scripts/export_for_neo4j_import.py
# import pandas as pd
# from pathlib import Path

# Path("data/neo4j_import").mkdir(exist_ok=True)

# print("Loading simulation data...")
# df = pd.read_json("sentinel_training_data.jsonl", lines=True)

# # Fill unified sender/receiver hashes
# df["sender_hash"]   = df["sender_hash"].fillna(df["sender_wallet_hash"])
# df["receiver_hash"] = df["receiver_hash"].fillna(df["receiver_wallet_hash"])
# df = df.dropna(subset=["sender_hash", "receiver_hash"])

# print("Available columns:", sorted(df.columns.tolist()))

# # ── 1. Nodes CSV ──────────────────────────────────────────────────
# # Neo4j import needs one row per unique node
# senders = df[["sender_hash", "sender_archetype"]].rename(
#     columns={"sender_hash": "hash:ID", "sender_archetype": "archetype"})
# receivers = df[["receiver_hash"]].rename(
#     columns={"receiver_hash": "hash:ID"})
# receivers["archetype"] = ""

# nodes = (
#     pd.concat([senders, receivers])
#     .drop_duplicates(subset=["hash:ID"])
#     .fillna("")
# )
# nodes[":LABEL"] = "Identity"
# nodes.to_csv("data/neo4j_import/nodes.csv", index=False)
# print(f"Nodes CSV: {len(nodes):,} rows")

# # ── 2. Relationships CSV ─────────────────────────────────────────
# edges = df[[
#     "sender_hash", "receiver_hash",
#     "transaction_id", "amount_usd", "timestamp",
#     "modality", "is_fraud", "fraud_type",
#     "fraud_network_id", "cross_modality_fraud_id",
# ]].copy()

# # Ensure is_fraud is a clean integer before renaming
# edges["is_fraud"] = edges["is_fraud"].fillna(False).astype(int)

# # Rename columns to match Neo4j bulk importer format
# edges = edges.rename(columns={
#     "sender_hash":   ":START_ID",
#     "receiver_hash": ":END_ID",
#     "is_fraud":      "is_fraud:int"
# })

# edges[":TYPE"]                   = "SENT_TO"
# edges["fraud_type"]              = edges["fraud_type"].fillna("")
# edges["fraud_network_id"]        = edges["fraud_network_id"].fillna("")
# edges["cross_modality_fraud_id"] = edges["cross_modality_fraud_id"].fillna("")

# edges.to_csv("data/neo4j_import/edges.csv", index=False)
# print(f"Edges CSV: {len(edges):,} rows")
# print("\nDone. Now run the neo4j-admin import command.")

# scripts/export_for_neo4j_import.py
import pandas as pd
from pathlib import Path

Path("data/neo4j_import").mkdir(exist_ok=True)

print("Loading simulation data...")
df = pd.read_json("sentinel_training_data.jsonl", lines=True)

# ── ENTITY RESOLUTION FIX ─────────────────────────────────────────
# Use client_id as the primary identity key to link cross-modal legs
if "client_id" in df.columns:
    df["client_id"] = df["client_id"].replace("", None)
    df["sender_hash"] = df["client_id"].combine_first(df["sender_hash"]).combine_first(df.get("sender_wallet_hash"))
else:
    df["sender_hash"] = df["sender_hash"].combine_first(df.get("sender_wallet_hash"))

df["receiver_hash"] = df["receiver_hash"].combine_first(df.get("receiver_wallet_hash"))
df = df.dropna(subset=["sender_hash", "receiver_hash"])

print("Available columns:", sorted(df.columns.tolist()))

# ── 1. Nodes CSV ──────────────────────────────────────────────────
# Neo4j import needs one row per unique node
senders = df[["sender_hash", "sender_archetype"]].rename(
    columns={"sender_hash": "hash:ID", "sender_archetype": "archetype"})
receivers = df[["receiver_hash"]].rename(
    columns={"receiver_hash": "hash:ID"})
receivers["archetype"] = ""

nodes = (
    pd.concat([senders, receivers])
    .drop_duplicates(subset=["hash:ID"])
    .fillna("")
)
nodes[":LABEL"] = "Identity"
nodes.to_csv("data/neo4j_import/nodes.csv", index=False)
print(f"Nodes CSV: {len(nodes):,} rows")

# ── 2. Relationships CSV ─────────────────────────────────────────
edges = df[[
    "sender_hash", "receiver_hash",
    "transaction_id", "amount_usd", "timestamp",
    "modality", "is_fraud", "fraud_type",
    "fraud_network_id", "cross_modality_fraud_id",
]].copy()

# ── DATA TYPE FIX ────────────────────────────────────────────────
# Ensure is_fraud is a clean boolean before renaming
edges["is_fraud"] = edges["is_fraud"].fillna(False).astype(bool).map({True: "true", False: "false"})
# Rename columns to match Neo4j bulk importer format
edges = edges.rename(columns={
    "sender_hash":   ":START_ID",
    "receiver_hash": ":END_ID",
    "is_fraud":      "is_fraud:boolean"
})

edges[":TYPE"]                   = "SENT_TO"
edges["fraud_type"]              = edges["fraud_type"].fillna("")
edges["fraud_network_id"]        = edges["fraud_network_id"].fillna("")
edges["cross_modality_fraud_id"] = edges["cross_modality_fraud_id"].fillna("")

edges.to_csv("data/neo4j_import/edges.csv", index=False)
print(f"Edges CSV: {len(edges):,} rows")
print("\nDone. Now run the neo4j-admin import command.")