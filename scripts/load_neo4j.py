# scripts/load_neo4j.py
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

DRIVER = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "sentinel_neo4j")
)

def load_batch(tx, batch):
    tx.run("""
    UNWIND $rows AS row
    MERGE (s:Identity {hash: row.sender_hash})
      ON CREATE SET s.first_seen = row.timestamp,
                    s.archetype  = row.archetype,
                    s.txn_count  = 0,
                    s.total_volume = 0
      ON MATCH SET  s.last_seen  = row.timestamp,
                    s.txn_count  = s.txn_count + 1,
                    s.total_volume = s.total_volume + row.amount_usd

    MERGE (r:Identity {hash: row.receiver_hash})
      ON CREATE SET r.first_seen = row.timestamp,
                    r.txn_count  = 0,
                    r.total_volume = 0
      ON MATCH SET  r.last_seen  = row.timestamp

    CREATE (s)-[:SENT_TO {
        transaction_id:         row.transaction_id,
        amount_usd:             row.amount_usd,
        timestamp:              row.timestamp,
        modality:               row.modality,
        is_fraud:               row.is_fraud,
        fraud_type:             row.fraud_type,
        fraud_network_id:       row.fraud_network_id,
        cross_modality_fraud_id: row.cross_modality_fraud_id
    }]->(r)
    """, rows=batch)

df = pd.read_json("sentinel_training_data.jsonl", lines=True)

# Fill receiver_hash for crypto (uses receiver_wallet_hash)
df["receiver_hash"] = df["receiver_hash"].fillna(df["receiver_wallet_hash"])
df["sender_hash"]   = df["sender_hash"].fillna(df["sender_wallet_hash"])

# Drop rows still missing sender/receiver
df = df.dropna(subset=["sender_hash", "receiver_hash"])

print(f"Loading {len(df):,} transactions into Neo4j...")
batch_size = 2000
# for i in tqdm(range(0, len(df), batch_size)):
#     batch = df.iloc[i:i+batch_size].fillna("").to_dict("records")
#     with DRIVER.session() as session:
#         session.execute_write(load_batch, batch)

with DRIVER.session() as session:
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size].fillna("").to_dict("records")
        session.execute_write(load_batch, batch)
        
# Create indexes for fast lookup
with DRIVER.session() as session:
    session.run("CREATE INDEX identity_hash IF NOT EXISTS FOR (n:Identity) ON (n.hash)")
    session.run("CREATE INDEX fraud_type IF NOT EXISTS FOR ()-[r:SENT_TO]-() ON (r.fraud_type)")

print("Done. Open http://localhost:7474 to explore the graph.")