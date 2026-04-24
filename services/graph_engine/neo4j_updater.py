# """
# services/graph_engine/neo4j_updater.py
# =======================================
# Live Neo4j graph update service.

# Consumes from Redpanda topic "transactions.ingested" and appends
# new transactions to the Neo4j graph every 60 seconds.

# Why batch instead of per-transaction writes:
#   Neo4j MERGE operations acquire node-level locks. At 200 transactions/sec
#   that is 200 concurrent lock acquisitions. Batching to 60-second windows
#   reduces this to one batch write per minute, which Neo4j handles efficiently
#   with UNWIND + MERGE patterns.

# What it writes:
#   Nodes: Identity nodes (MERGE — create if not exists)
#   Edges: SENT_TO relationships (CREATE — always new, never merged)
#   Node properties updated: txn_count, total_volume (cumulative)

# This data feeds embed_gnn.py (embedding_job.py in production) which
# re-reads the graph every 60 seconds and updates Redis embeddings.

# Usage:
#   python -m services.graph_engine.neo4j_updater

#   Or via Docker:
#   docker compose up neo4j_updater
# """

# import json
# import logging
# import os
# import signal
# import time
# from collections import defaultdict
# from datetime import datetime, timezone
# from threading import Event

# from dotenv import load_dotenv
# from kafka import KafkaConsumer
# from neo4j import GraphDatabase

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [neo4j_updater] %(levelname)s %(message)s",
# )
# log = logging.getLogger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# REDPANDA_BROKERS = os.getenv("REDPANDA_BROKERS", "localhost:9092")
# NEO4J_URI        = os.getenv("NEO4J_URI",        "bolt://localhost:7687")
# NEO4J_USER       = os.getenv("NEO4J_USER",       "neo4j")
# NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD",   "sentinel_neo4j")

# INPUT_TOPIC      = "transactions.ingested"
# CONSUMER_GROUP   = "neo4j_updater_v1"
# BATCH_INTERVAL   = 15       # seconds between Neo4j writes
# MAX_BATCH_SIZE   = 10_000   # max transactions per batch (safety cap)

# # ─────────────────────────────────────────────────────────────────────────────
# # CYPHER QUERIES
# # ─────────────────────────────────────────────────────────────────────────────

# # MERGE nodes first — guarantees they exist before edge creation
# NODE_UPSERT_CYPHER = """
# UNWIND $rows AS row
# MERGE (n:Identity {hash: row.hash})
# ON CREATE SET
#     n.txn_count       = 0,
#     n.total_volume    = 0.0,
#     n.first_seen      = row.ts,
#     n.institution_count = 1
# ON MATCH SET
#     n.last_seen       = row.ts,
#     n.txn_count       = n.txn_count + 1,
#     n.total_volume    = n.total_volume + row.amount_usd
# """

# # CREATE edges after nodes — MATCH is safe because nodes are guaranteed to exist
# EDGE_CREATE_CYPHER = """
# UNWIND $rows AS row
# MATCH (s:Identity {hash: row.sender_hash})
# MATCH (r:Identity {hash: row.receiver_hash})
# CREATE (s)-[e:SENT_TO {
#     transaction_id:          row.transaction_id,
#     is_fraud:                row.is_fraud,
#     fraud_type:              row.fraud_type,
#     amount_usd:              row.amount_usd,
#     modality:                row.modality,
#     timestamp:               row.ts,
#     fraud_network_id:        row.fraud_network_id,
#     cross_modality_fraud_id: row.cross_modality_fraud_id
# }]->(r)
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # BATCH ACCUMULATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class TransactionBatch:
#     """
#     Accumulates transactions for batch write to Neo4j.

#     Maintains separate node and edge lists because:
#     - Nodes need deduplication (MERGE) — same hash from many transactions
#     - Edges never need deduplication (CREATE) — each transaction is unique

#     Deduplication strategy for nodes:
#     - Keep only the latest transaction data per hash
#     - Sum amount_usd across all transactions (for total_volume)
#     - Count occurrences (for txn_count increment)
#     """

#     def __init__(self):
#         self._node_totals:  dict[str, dict] = {}   # hash → accumulated stats
#         self._edges:        list[dict]       = []

#     def add(self, txn: dict):
#         """Add a transaction to the batch."""
#         # Resolve sender and receiver hashes
#         sender_hash   = (txn.get("identity_hash") or txn.get("sender_hash")
#                          or txn.get("user_email_hash") or txn.get("sender_wallet_hash") or "")
#         receiver_hash = (txn.get("receiver_hash") or txn.get("receiver_wallet_hash") or "")

#         if not sender_hash:
#             return  # Drop transactions without identity

#         amount_usd = float(txn.get("amount_usd") or 0)
#         ts         = txn.get("timestamp") or datetime.now(timezone.utc).isoformat()

#         # Accumulate sender node stats
#         if sender_hash not in self._node_totals:
#             self._node_totals[sender_hash] = {
#                 "hash":       sender_hash,
#                 "ts":         ts,
#                 "amount_usd": 0.0,
#             }
#         self._node_totals[sender_hash]["amount_usd"] += amount_usd
#         self._node_totals[sender_hash]["ts"] = ts   # update to latest

#         # Ensure receiver node exists (with zero amount, just to create the node)
#         if receiver_hash and receiver_hash not in self._node_totals:
#             self._node_totals[receiver_hash] = {
#                 "hash":       receiver_hash,
#                 "ts":         ts,
#                 "amount_usd": 0.0,
#             }

#         # Add edge if receiver exists
#         if receiver_hash:
#             self._edges.append({
#                 "transaction_id":          txn.get("transaction_id", ""),
#                 "sender_hash":             sender_hash,
#                 "receiver_hash":           receiver_hash,
#                 "amount_usd":              amount_usd,
#                 "modality":                txn.get("modality", "fiat"),
#                 "ts":                      ts,
#                 "is_fraud":                int(txn.get("is_fraud") or 0),
#                 "fraud_type":              txn.get("fraud_type") or "",
#                 "fraud_network_id":        txn.get("fraud_network_id") or "",
#                 "cross_modality_fraud_id": txn.get("cross_modality_fraud_id") or "",
#             })

#     @property
#     def size(self) -> int:
#         return len(self._edges)

#     @property
#     def node_rows(self) -> list:
#         return list(self._node_totals.values())

#     @property
#     def edge_rows(self) -> list:
#         return self._edges

#     def clear(self):
#         self._node_totals.clear()
#         self._edges.clear()


# # ─────────────────────────────────────────────────────────────────────────────
# # NEO4J WRITER
# # ─────────────────────────────────────────────────────────────────────────────

# def write_batch(driver, batch: TransactionBatch) -> dict:
#     """
#     Write one batch to Neo4j.

#     Returns stats dict with node_count, edge_count, elapsed_ms.
#     """
#     if batch.size == 0:
#         return {"node_count": 0, "edge_count": 0, "elapsed_ms": 0}

#     t0         = time.time()
#     node_rows  = batch.node_rows
#     edge_rows  = batch.edge_rows
#     chunk_size = 2_000   # process in sub-batches to avoid OOM

#     node_count = 0
#     edge_count = 0

#     with driver.session(database="neo4j") as session:
#         # Phase 1: upsert nodes
#         for i in range(0, len(node_rows), chunk_size):
#             chunk = node_rows[i : i + chunk_size]
#             # session.execute_write(
#             #     lambda tx: tx.run(NODE_UPSERT_CYPHER, rows=chunk)
#             # )
#             session.execute_write(lambda tx, c=chunk: tx.run(NODE_UPSERT_CYPHER, rows=c))
#             node_count += len(chunk)

#         # Phase 2: create edges (nodes guaranteed to exist now)
#         for i in range(0, len(edge_rows), chunk_size):
#             chunk = edge_rows[i : i + chunk_size]
#             # session.execute_write(
#             #     lambda tx: tx.run(EDGE_CREATE_CYPHER, rows=chunk)
#             # )
#             session.execute_write(lambda tx, c=chunk: tx.run(EDGE_CREATE_CYPHER, rows=c))
#             edge_count += len(chunk)

#     elapsed_ms = round((time.time() - t0) * 1000, 1)
#     return {
#         "node_count": node_count,
#         "edge_count": edge_count,
#         "elapsed_ms": elapsed_ms,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN LOOP
# # ─────────────────────────────────────────────────────────────────────────────

# def run():
#     log.info("=" * 60)
#     log.info("SENTINEL NEO4J UPDATER")
#     log.info("=" * 60)
#     log.info("  Redpanda:    %s → %s", REDPANDA_BROKERS, INPUT_TOPIC)
#     log.info("  Neo4j:       %s", NEO4J_URI)
#     log.info("  Batch interval: %ds", BATCH_INTERVAL)

#     # Connect to Neo4j
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#     try:
#         driver.verify_connectivity()
#         log.info("  Neo4j: connected")
#     except Exception as e:
#         log.error("  Neo4j: FAILED — %s", e)
#         log.error("  Run: docker compose up -d neo4j")
#         return

#     # Connect Kafka consumer
#     consumer = KafkaConsumer(
#         INPUT_TOPIC,
#         bootstrap_servers=REDPANDA_BROKERS,
#         group_id=CONSUMER_GROUP,
#         auto_offset_reset="latest",
#         enable_auto_commit=True,
#         value_deserializer=lambda v: json.loads(v.decode("utf-8")),
#         consumer_timeout_ms=5_000,   # 5s poll timeout — allows clean shutdown
#     )
#     log.info("  Kafka consumer: connected")

#     # Graceful shutdown handler
#     stop_event = Event()

#     def _shutdown(sig, frame):
#         log.info("Shutdown signal received — draining batch...")
#         stop_event.set()

#     signal.signal(signal.SIGINT,  _shutdown)
#     signal.signal(signal.SIGTERM, _shutdown)

#     # Main accumulation loop
#     batch      = TransactionBatch()
#     last_flush = time.time()
#     total_txns = 0

#     log.info("Consuming from %s... (flushing every %ds)", INPUT_TOPIC, BATCH_INTERVAL)

#     while not stop_event.is_set():
#         # Consume messages until timeout
#         try:
#             for message in consumer:
#                 txn = message.value
#                 batch.add(txn)
#                 total_txns += 1

#                 # Safety cap: flush early if batch gets huge
#                 if batch.size >= MAX_BATCH_SIZE:
#                     log.info("Max batch size reached (%d), flushing early", batch.size)
#                     break

#                 # Exit if shutdown requested
#                 if stop_event.is_set():
#                     break
#         except StopIteration:
#             pass   # consumer_timeout_ms elapsed, no new messages

#         # Time-based flush
#         elapsed = time.time() - last_flush
#         if elapsed >= BATCH_INTERVAL or stop_event.is_set():
#             if batch.size > 0:
#                 log.info(
#                     "Writing batch: %d nodes, %d edges (accumulated for %.0fs)",
#                     len(batch.node_rows), batch.size, elapsed
#                 )
#                 try:
#                     stats = write_batch(driver, batch)
#                     log.info(
#                         "  ✓ Neo4j: %d nodes, %d edges in %sms",
#                         stats["node_count"], stats["edge_count"], stats["elapsed_ms"]
#                     )
#                 except Exception as e:
#                     log.error("  ✗ Neo4j write failed: %s", e)
#                 finally:
#                     batch.clear()
#                     last_flush = time.time()
#             else:
#                 log.debug("  No transactions to flush")
#                 last_flush = time.time()

#     # Final flush on shutdown
#     if batch.size > 0:
#         log.info("Flushing final batch (%d transactions)...", batch.size)
#         try:
#             stats = write_batch(driver, batch)
#             log.info("  Final flush: %d nodes, %d edges", stats["node_count"], stats["edge_count"])
#         except Exception as e:
#             log.error("  Final flush failed: %s", e)

#     consumer.close()
#     driver.close()
#     log.info("Neo4j updater stopped. Total transactions processed: %d", total_txns)


# if __name__ == "__main__":
#     run()

"""
services/graph_engine/neo4j_updater.py
=======================================
Live Neo4j graph update service using Redis lists.

Consumes from Redis queue "transactions:neo4j" and appends
new transactions to the Neo4j graph every 15 seconds.
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from threading import Event

import redis
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [neo4j_updater] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", "6379"))
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "sentinel_neo4j")

QUEUE_KEY        = "transactions:neo4j"
BATCH_INTERVAL   = 15       # seconds between Neo4j writes
MAX_BATCH_SIZE   = 5_000    # max transactions per batch

# ─────────────────────────────────────────────────────────────────────────────
# CYPHER QUERIES
# ─────────────────────────────────────────────────────────────────────────────

NODE_UPSERT_CYPHER = """
UNWIND $rows AS row
MERGE (n:Identity {hash: row.hash})
ON CREATE SET
    n.txn_count       = 0,
    n.total_volume    = 0.0,
    n.first_seen      = row.ts,
    n.institution_count = 1
ON MATCH SET
    n.last_seen       = row.ts,
    n.txn_count       = n.txn_count + 1,
    n.total_volume    = n.total_volume + row.amount_usd
"""

EDGE_CREATE_CYPHER = """
UNWIND $rows AS row
MATCH (s:Identity {hash: row.sender_hash})
MATCH (r:Identity {hash: row.receiver_hash})
CREATE (s)-[e:SENT_TO {
    transaction_id:          row.transaction_id,
    is_fraud:                row.is_fraud,
    fraud_type:              row.fraud_type,
    amount_usd:              row.amount_usd,
    modality:                row.modality,
    timestamp:               row.ts,
    fraud_network_id:        row.fraud_network_id,
    cross_modality_fraud_id: row.cross_modality_fraud_id
}]->(r)
"""

# ─────────────────────────────────────────────────────────────────────────────
# BATCH ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class TransactionBatch:
    def __init__(self):
        self._node_totals: dict[str, dict] = {}
        self._edges: list[dict] = []

    def add(self, txn: dict):
        sender_hash = (txn.get("identity_hash") or txn.get("sender_hash") or 
                       txn.get("user_email_hash") or txn.get("sender_wallet_hash") or "")
        receiver_hash = txn.get("receiver_hash") or txn.get("receiver_wallet_hash") or ""

        if not sender_hash: return

        amount_usd = float(txn.get("amount_usd") or 0)
        ts = txn.get("timestamp") or datetime.now(timezone.utc).isoformat()

        if sender_hash not in self._node_totals:
            self._node_totals[sender_hash] = {"hash": sender_hash, "ts": ts, "amount_usd": 0.0}
        
        self._node_totals[sender_hash]["amount_usd"] += amount_usd
        self._node_totals[sender_hash]["ts"] = ts

        if receiver_hash and receiver_hash not in self._node_totals:
            self._node_totals[receiver_hash] = {"hash": receiver_hash, "ts": ts, "amount_usd": 0.0}

        if receiver_hash:
            self._edges.append({
                "transaction_id":          txn.get("transaction_id", ""),
                "sender_hash":             sender_hash,
                "receiver_hash":           receiver_hash,
                "amount_usd":              amount_usd,
                "modality":                txn.get("modality", "fiat"),
                "ts":                      ts,
                "is_fraud":                int(txn.get("is_fraud") or 0),
                "fraud_type":              txn.get("fraud_type", ""),
                "fraud_network_id":        txn.get("fraud_network_id", ""),
                "cross_modality_fraud_id": txn.get("cross_modality_fraud_id", ""),
            })

    @property
    def size(self) -> int: return len(self._edges)
    @property
    def node_rows(self) -> list: return list(self._node_totals.values())
    @property
    def edge_rows(self) -> list: return self._edges

    def clear(self):
        self._node_totals.clear()
        self._edges.clear()

def write_batch(driver, batch: TransactionBatch) -> dict:
    if batch.size == 0:
        return {"node_count": 0, "edge_count": 0, "elapsed_ms": 0}

    t0 = time.time()
    chunk_size = 2_000
    node_count = edge_count = 0

    with driver.session(database="neo4j") as session:
        for i in range(0, len(batch.node_rows), chunk_size):
            chunk = batch.node_rows[i : i + chunk_size]
            session.execute_write(lambda tx, c=chunk: tx.run(NODE_UPSERT_CYPHER, rows=c))
            node_count += len(chunk)

        for i in range(0, len(batch.edge_rows), chunk_size):
            chunk = batch.edge_rows[i : i + chunk_size]
            session.execute_write(lambda tx, c=chunk: tx.run(EDGE_CREATE_CYPHER, rows=c))
            edge_count += len(chunk)

    return {"node_count": node_count, "edge_count": edge_count, "elapsed_ms": round((time.time() - t0) * 1000, 1)}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("SENTINEL NEO4J UPDATER (REDIS VERSION)")
    log.info("=" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        log.info("  Neo4j: connected")
    except Exception as e:
        log.error("  Neo4j: FAILED — %s", e)
        return

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        log.info("  Redis: connected")
    except Exception as e:
        log.error("  Redis: FAILED — %s", e)
        driver.close()
        return

    stop_event = Event()
    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())

    batch = TransactionBatch()
    last_flush = time.time()
    total_txns = 0

    log.info("Consuming from Redis list '%s'...", QUEUE_KEY)

    while not stop_event.is_set():
        try:
            # Block for up to 1 second waiting for a new transaction
            item = r.blpop(QUEUE_KEY, timeout=1)
            if item:
                _, msg = item
                batch.add(json.loads(msg))
                total_txns += 1
        except Exception as e:
            log.error("Error reading from Redis: %s", e)
            time.sleep(1)

        elapsed = time.time() - last_flush
        if elapsed >= BATCH_INTERVAL or batch.size >= MAX_BATCH_SIZE or stop_event.is_set():
            if batch.size > 0:
                try:
                    stats = write_batch(driver, batch)
                    log.info("  ✓ Neo4j: %d nodes, %d edges in %sms", stats["node_count"], stats["edge_count"], stats["elapsed_ms"])
                except Exception as e:
                    log.error("  ✗ Neo4j write failed: %s", e)
                    # Note: In a true production system, you would push failed batches to a Dead Letter Queue (DLQ) here.
                finally:
                    batch.clear()
            last_flush = time.time()

    driver.close()
    log.info("Neo4j updater stopped. Total processed: %d", total_txns)

if __name__ == "__main__":
    run()