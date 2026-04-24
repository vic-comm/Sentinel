# """
# scripts/load_neo4j.py
# =====================
# Reads sentinel_training_data.jsonl and populates Neo4j with:
#   - Identity nodes  (one per unique sender/receiver hash)
#   - SENT_TO edges   (one per transaction)

# Run AFTER:
#   1. docker compose up -d neo4j          (container running)
#   2. python scripts/setup_neo4j.py       (schema created)
#   3. python sentinel_simulator.py        (JSONL file exists)

# Usage:
#     python scripts/load_neo4j.py
#     python scripts/load_neo4j.py --input data/training/full_dataset.parquet
#     python scripts/load_neo4j.py --limit 100000   # load first 100K rows only

# Dependencies:
#     pip install neo4j pandas pyarrow tqdm
# """

# import argparse
# import sys
# import time
# import pandas as pd
# from pathlib import Path
# from neo4j import GraphDatabase
# from tqdm import tqdm

# NEO4J_URI      = "bolt://localhost:7687"
# NEO4J_USER     = "neo4j"
# NEO4J_PASSWORD = "sentinel_neo4j"
# BATCH_SIZE     = 500   # rows per Cypher UNWIND call

# # ─────────────────────────────────────────────────────────────────────────────
# # Cypher statement: MERGE nodes, CREATE edge
# # MERGE on sender/receiver means each unique hash → exactly one Identity node.
# # Properties are SET on CREATE (first time seen) or updated ON MATCH.
# # ─────────────────────────────────────────────────────────────────────────────
# LOAD_CYPHER = """
# UNWIND $rows AS row

# // Sender node
# MERGE (s:Identity {hash: row.sender_hash})
#   ON CREATE SET
#     s.first_seen       = row.timestamp,
#     s.last_seen        = row.timestamp,
#     s.txn_count        = 1,
#     s.total_volume_usd = row.amount_usd,
#     s.archetype        = row.archetype,
#     s.is_fraud         = row.is_fraud
#   ON MATCH SET
#     s.last_seen        = row.timestamp,
#     s.txn_count        = s.txn_count + 1,
#     s.total_volume_usd = s.total_volume_usd + row.amount_usd,
#     s.is_fraud         = CASE WHEN row.is_fraud THEN true ELSE s.is_fraud END

# // Receiver node
# MERGE (r:Identity {hash: row.receiver_hash})
#   ON CREATE SET
#     r.first_seen       = row.timestamp,
#     r.last_seen        = row.timestamp,
#     r.txn_count        = 0,
#     r.total_volume_usd = 0.0,
#     r.is_fraud         = false
#   ON MATCH SET
#     r.last_seen        = row.timestamp

# // Transaction edge
# CREATE (s)-[:SENT_TO {
#     transaction_id:           row.transaction_id,
#     amount_usd:               row.amount_usd,
#     timestamp:                row.timestamp,
#     modality:                 row.modality,
#     transfer_type:            row.transfer_type,
#     client_id:                row.client_id,
#     is_fraud:                 row.is_fraud,
#     fraud_type:               row.fraud_type,
#     fraud_network_id:         row.fraud_network_id,
#     cross_modality_fraud_id:  row.cross_modality_fraud_id,
#     balance_drain_ratio:      row.balance_drain_ratio,
#     velocity_ratio:           row.velocity_ratio,
#     amount_ratio:             row.amount_ratio,
#     is_unusual_hour:          row.is_unusual_hour,
#     known_mixer_interaction:  row.known_mixer_interaction,
#     receiver_wallet_type:     row.receiver_wallet_type
# }]->(r)
# """


# def load_data(input_path: str, limit: int | None) -> pd.DataFrame:
#     path = Path(input_path)
#     if not path.exists():
#         print(f"ERROR: {input_path} not found.")
#         print("Run: python sentinel_simulator.py   first.")
#         sys.exit(1)

#     print(f"Loading {input_path} ...")
#     if path.suffix == ".jsonl":
#         df = pd.read_json(path, lines=True)
#     elif path.suffix == ".parquet":
#         df = pd.read_parquet(path)
#     else:
#         print(f"ERROR: Unsupported format {path.suffix}. Use .jsonl or .parquet")
#         sys.exit(1)

#     if limit:
#         df = df.head(limit)
#         print(f"  Limited to first {limit:,} rows.")

#     print(f"  Loaded {len(df):,} rows.")
#     return df


# def prepare_batch(df_slice: pd.DataFrame) -> list[dict]:
#     """
#     Convert DataFrame slice to list of dicts for Cypher UNWIND.
#     Fills missing identity keys and coerces types so Neo4j doesn't choke.
#     """
#     rows = []
#     for _, row in df_slice.iterrows():
#         # Resolve sender/receiver hash — fiat uses sender_hash,
#         # crypto uses sender_wallet_hash. Both are stored as sender_hash
#         # in the schema but the column name varies by source.
#         sender   = (row.get("sender_hash")
#                     or row.get("sender_wallet_hash")
#                     or row.get("identity_hash", "unknown"))
#         receiver = (row.get("receiver_hash")
#                     or row.get("receiver_wallet_hash")
#                     or "unknown_receiver")

#         # Skip rows with no meaningful sender
#         if sender in ("unknown", None, ""):
#             continue

#         rows.append({
#             "transaction_id":          str(row.get("transaction_id", "")),
#             "sender_hash":             str(sender),
#             "receiver_hash":           str(receiver),
#             "timestamp":               str(row.get("timestamp", "")),
#             "amount_usd":              float(row.get("amount_usd") or 0),
#             "modality":                str(row.get("modality", "unknown")),
#             "transfer_type":           str(row.get("transfer_type") or ""),
#             "client_id":               str(row.get("client_id", "")),
#             "archetype":               str(row.get("sender_archetype") or ""),
#             "is_fraud":                bool(row.get("is_fraud", False)),
#             "fraud_type":              row.get("fraud_type"),
#             "fraud_network_id":        row.get("fraud_network_id"),
#             "cross_modality_fraud_id": row.get("cross_modality_fraud_id"),
#             "balance_drain_ratio":     float(row.get("balance_drain_ratio") or 0),
#             "velocity_ratio":          float(row.get("velocity_ratio") or 0),
#             "amount_ratio":            float(row.get("amount_ratio") or 0),
#             "is_unusual_hour":         bool(row.get("is_unusual_hour", False)),
#             "known_mixer_interaction": bool(row.get("known_mixer_interaction", False)),
#             "receiver_wallet_type":    row.get("receiver_wallet_type"),
#         })
#     return rows


# def load_into_neo4j(driver, df: pd.DataFrame):
#     total     = len(df)
#     n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
#     loaded    = 0
#     skipped   = 0

#     print(f"\nLoading {total:,} rows into Neo4j in {n_batches:,} batches of {BATCH_SIZE} ...")
#     start = time.time()

#     with driver.session(database="neo4j") as session:
#         for i in tqdm(range(0, total, BATCH_SIZE), desc="Loading", unit="batch"):
#             batch_df  = df.iloc[i : i + BATCH_SIZE]
#             batch     = prepare_batch(batch_df)
#             skipped  += (len(batch_df) - len(batch))
#             if not batch:
#                 continue
#             session.execute_write(lambda tx, b=batch: tx.run(LOAD_CYPHER, rows=b))
#             loaded += len(batch)

#     elapsed = time.time() - start
#     rate    = loaded / elapsed if elapsed > 0 else 0
#     print(f"\nDone. Loaded {loaded:,} rows in {elapsed:.1f}s ({rate:.0f} rows/s)")
#     if skipped:
#         print(f"Skipped {skipped:,} rows with missing sender_hash.")


# def print_graph_stats(driver):
#     print("\nGraph statistics:")
#     with driver.session(database="neo4j") as session:
#         r = session.run("MATCH (n:Identity) RETURN count(n) AS c").single()
#         print(f"  Identity nodes:        {r['c']:>10,}")

#         r = session.run("MATCH ()-[r:SENT_TO]->() RETURN count(r) AS c").single()
#         print(f"  SENT_TO edges:         {r['c']:>10,}")

#         r = session.run(
#             "MATCH ()-[r:SENT_TO {is_fraud: true}]->() RETURN count(r) AS c"
#         ).single()
#         print(f"  Fraud edges:           {r['c']:>10,}")

#         r = session.run("""
#             MATCH ()-[r:SENT_TO]->()
#             WHERE r.cross_modality_fraud_id IS NOT NULL
#             RETURN count(DISTINCT r.cross_modality_fraud_id) AS c
#         """).single()
#         print(f"  Cross-modal sequences: {r['c']:>10,}")

#         # Show mule ring count
#         r = session.run("""
#             MATCH ()-[r:SENT_TO {fraud_type: 'mule_network'}]->()
#             RETURN count(DISTINCT r.fraud_network_id) AS c
#         """).single()
#         print(f"  Mule networks:         {r['c']:>10,}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Load Sentinel transactions into Neo4j")
#     parser.add_argument(
#         "--input", default="sentinel_training_data.jsonl",
#         help="Path to JSONL or Parquet file (default: sentinel_training_data.jsonl)"
#     )
#     parser.add_argument(
#         "--limit", type=int, default=None,
#         help="Only load first N rows (useful for testing)"
#     )
#     args = parser.parse_args()

#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#     try:
#         driver.verify_connectivity()
#         print(f"Connected to Neo4j at {NEO4J_URI}")
#     except Exception as e:
#         print(f"ERROR: Cannot connect to Neo4j: {e}")
#         print("Make sure Neo4j is running:  docker compose up -d neo4j")
#         print("Then run setup first:        python scripts/setup_neo4j.py")
#         sys.exit(1)

#     try:
#         df = load_data(args.input, args.limit)
#         load_into_neo4j(driver, df)
#         print_graph_stats(driver)
#         print("\nNext step: python scripts/verify_neo4j.py")
#     finally:
#         driver.close()

"""
scripts/load_neo4j.py
=====================
Reads sentinel_training_data.jsonl and populates Neo4j with:
  - Identity nodes  (one per unique sender/receiver hash)
  - SENT_TO edges   (one per transaction)

OPTIMIZED VERSION: Separates Node creation from Edge creation to bypass 
the massive MERGE lock bottlenecks in Neo4j.
"""

import argparse
import sys
import time
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "sentinel_neo4j"
BATCH_SIZE     = 10_000   # Increased batch size for faster insertion

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED CYPHER QUERIES
# ─────────────────────────────────────────────────────────────────────────────

# Step 1: Create Nodes (Unique identities only)
LOAD_NODES_CYPHER = """
UNWIND $rows AS row
MERGE (n:Identity {hash: row.hash})
ON CREATE SET
    n.txn_count = 0,
    n.total_volume = 0.0,
    n.account_age_days = row.account_age_days,
    n.institution_count = 1
"""

# Step 2: Create Edges (Assuming nodes already exist)
LOAD_EDGES_CYPHER = """
UNWIND $rows AS row
MATCH (s:Identity {hash: row.sender_hash})
MATCH (r:Identity {hash: row.receiver_hash})
CREATE (s)-[e:SENT_TO {
    transaction_id: row.transaction_id,
    is_fraud:       CASE WHEN row.is_fraud THEN 1 ELSE 0 END,
    amount_usd:     row.amount_usd,
    modality:       row.modality
}]->(r)
SET s.txn_count = coalesce(s.txn_count, 0) + 1,
    s.total_volume = coalesce(s.total_volume, 0.0) + row.amount_usd
"""


def load_data(input_path: str, limit: int | None) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        print(f"ERROR: Unsupported format {path.suffix}.")
        sys.exit(1)

    if limit:
        df = df.head(limit)
        print(f"  Limited to first {limit:,} rows.")

    print(f"  Loaded {len(df):,} rows.")
    return df


def load_into_neo4j_optimized(driver, df: pd.DataFrame):
    # ── Normalize Data ──
    print("\nNormalizing dataset...")
    df['sender_hash'] = df['sender_hash'].combine_first(df.get('sender_wallet_hash'))
    df['receiver_hash'] = df['receiver_hash'].combine_first(df.get('receiver_wallet_hash'))
    
    # Drop rows missing critical routing data
    df = df.dropna(subset=['sender_hash', 'receiver_hash'])

    # ── Phase 1: Extract and Load Unique Nodes ──
    print("Extracting unique identities...")
    senders = df[['sender_hash', 'sender_account_age_days']].rename(columns={'sender_hash': 'hash', 'sender_account_age_days': 'account_age_days'})
    receivers = df[['receiver_hash', 'receiver_wallet_age_days']].rename(columns={'receiver_hash': 'hash', 'receiver_wallet_age_days': 'account_age_days'})
    
    unique_nodes = pd.concat([senders, receivers]).drop_duplicates(subset=['hash'])
    
    # Fill NaNs for ages
    unique_nodes['account_age_days'] = unique_nodes['account_age_days'].fillna(30).astype(int)
    
    nodes_list = unique_nodes.to_dict('records')
    total_nodes = len(nodes_list)

    print(f"\nPhase 1: Loading {total_nodes:,} Unique Nodes...")
    start_nodes = time.time()
    with driver.session(database="neo4j") as session:
        for i in tqdm(range(0, total_nodes, BATCH_SIZE), desc="Nodes"):
            batch = nodes_list[i : i + BATCH_SIZE]
            session.execute_write(lambda tx, b=batch: tx.run(LOAD_NODES_CYPHER, rows=b))
            
    print(f"Nodes loaded in {time.time() - start_nodes:.1f}s")

    # ── Phase 2: Load Edges ──
    edges_list = df[['transaction_id', 'sender_hash', 'receiver_hash', 'is_fraud', 'amount_usd', 'modality']].to_dict('records')
    total_edges = len(edges_list)

    print(f"\nPhase 2: Loading {total_edges:,} Edges...")
    start_edges = time.time()
    with driver.session(database="neo4j") as session:
        for i in tqdm(range(0, total_edges, BATCH_SIZE), desc="Edges"):
            batch = edges_list[i : i + BATCH_SIZE]
            session.execute_write(lambda tx, b=batch: tx.run(LOAD_EDGES_CYPHER, rows=b))

    print(f"Edges loaded in {time.time() - start_edges:.1f}s")


def print_graph_stats(driver):
    print("\nGraph statistics:")
    with driver.session(database="neo4j") as session:
        r = session.run("MATCH (n:Identity) RETURN count(n) AS c").single()
        print(f"  Identity nodes:        {r['c']:>10,}")

        r = session.run("MATCH ()-[r:SENT_TO]->() RETURN count(r) AS c").single()
        print(f"  SENT_TO edges:         {r['c']:>10,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Sentinel transactions into Neo4j")
    parser.add_argument("--input", default="sentinel_training_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"ERROR: Cannot connect to Neo4j: {e}")
        sys.exit(1)

    try:
        # Wipe database before loading to prevent duplicate edge clustering
        with driver.session(database="neo4j") as session:
            print("Wiping existing Neo4j database...")
            session.run("MATCH (n) DETACH DELETE n")
            
        df = load_data(args.input, args.limit)
        load_into_neo4j_optimized(driver, df)
        print_graph_stats(driver)
    finally:
        driver.close()