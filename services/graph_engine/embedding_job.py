"""
services/graph_engine/embedding_job.py
=======================================
Periodic GNN embedding refresh service.

Runs every 60 seconds (configurable). For each cycle:
  1. Queries Neo4j for nodes that were updated in the last 60 seconds
  2. Extracts their 2-hop subgraph (to capture neighborhood context)
  3. Runs FraudGNN inference on the subgraph
  4. Stores updated embeddings in Redis (TTL = 1 hour)

Why incremental (not full graph recomputation):
  Full graph has 500K+ nodes. Running inference on all of them takes
  ~3 minutes on CPU. At 60-second intervals, that would mean we're
  always running inference and never resting.

  Incremental: only nodes that received new transactions in the last 60s
  need embedding updates. That is typically 100-2K nodes, not 500K.
  Inference on 2K nodes takes ~2 seconds.

Why 2-hop subgraph (not just the node itself):
  GraphSAGE aggregates neighborhood information. A node's embedding
  depends on its 1-hop neighbors and their 1-hop neighbors.
  If node A received a new transaction, A's embedding changes.
  But A's neighbors' embeddings also change because A is now in their
  neighborhood. We update the full 2-hop subgraph to capture this.

Redis key format:
  gnn_embedding:{identity_hash} → JSON list of 32 floats
  TTL: 3600 seconds (refreshed every 60s, expires after 1 hour if missed)

Usage:
  python -m services.graph_engine.embedding_job

  Or via Docker:
  docker compose up embedding_job
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import numpy as np
import redis
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from neo4j import GraphDatabase


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [embedding_job] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NEO4J_URI        = os.getenv("NEO4J_URI",        "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER",       "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD",   "sentinel_neo4j")
REDIS_HOST       = os.getenv("REDIS_HOST",        "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT",   "6379"))

MODEL_PATH       = Path(os.getenv("MODEL_PATH",   "models/graphsage_weights.pt"))
CONFIG_PATH      = Path(os.getenv("CONFIG_PATH",  "models/graphsage_config.json"))

REFRESH_INTERVAL = int(os.getenv("EMBEDDING_REFRESH_INTERVAL", "60"))   # seconds
EMBEDDING_TTL    = 3600     # Redis TTL — 1 hour
BATCH_SIZE       = 1024     # nodes per inference batch
NUM_NEIGHBORS    = [15, 10] # matches training config
LOOKBACK_SECONDS = 120      # query nodes updated in last 2 * interval (safety buffer)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_config():
    """Load FraudGNN from disk. Returns (model, config, device)."""
    from scripts.train_graphsage import FraudGNN

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Run: python -m scripts.train_graphsage"
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run: python -m scripts.train_graphsage"
        )

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )

    model = FraudGNN(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        dropout=0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    log.info(
        "Model loaded: %d-dim embeddings, device=%s",
        config["out_channels"], device
    )
    return model, config, device


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def query_active_nodes(driver, lookback_seconds: int) -> list[dict]:
    """
    Query nodes that have been updated in the last lookback_seconds.

    Returns list of {hash, out_degree, in_degree, avg_out_amount,
    total_out_volume, avg_in_amount, fan_in_ratio, volume_per_txn}.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=lookback_seconds)).isoformat()

    with driver.session(database="neo4j") as session:
        # Get recently-updated nodes + their 2-hop neighbors
        result = session.run("""
            MATCH (n:Identity)
            WHERE n.last_seen >= $cutoff
            WITH collect(n.hash) AS active_hashes

            MATCH (n:Identity)
            WHERE n.hash IN active_hashes

            // Get 2-hop neighborhood for context-rich embeddings
            OPTIONAL MATCH (n)-[out:SENT_TO]->()
            OPTIONAL MATCH ()-[in_:SENT_TO]->(n)

            WITH n,
                 count(DISTINCT out)                         AS out_degree,
                 count(DISTINCT in_)                         AS in_degree,
                 coalesce(avg(toFloat(out.amount_usd)), 0.0) AS avg_out_amount,
                 coalesce(sum(toFloat(out.amount_usd)), 0.0) AS total_out_volume,
                 coalesce(avg(toFloat(in_.amount_usd)), 0.0) AS avg_in_amount

            RETURN n.hash        AS hash,
                   out_degree,   in_degree,
                   avg_out_amount, total_out_volume, avg_in_amount
            LIMIT 5000
        """, cutoff=cutoff)
        return [dict(r) for r in result]


def query_subgraph_edges(driver, node_hashes: list[str]) -> list[dict]:
    """
    Query edges within the 2-hop subgraph of the given nodes.
    Returns list of {src, dst, amount_usd, is_fraud}.
    """
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (s:Identity)-[r:SENT_TO]->(t:Identity)
            WHERE s.hash IN $hashes OR t.hash IN $hashes
            RETURN s.hash AS src,
                   t.hash AS dst,
                   coalesce(toFloat(r.amount_usd), 0.0) AS amount_usd,
                   coalesce(r.is_fraud, 0)               AS is_fraud
            LIMIT 100000
        """, hashes=node_hashes)
        return [dict(r) for r in result]


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_subgraph(nodes: list[dict], edges: list[dict]) -> tuple[Data, list[str]]:
    """
    Convert node/edge lists into a PyG Data object for inference.

    Node features (7 dims, matches Neo4j query in train_graphsage.py):
      [out_degree, in_degree, avg_out_amount, total_out_volume,
       avg_in_amount, fan_in_ratio, volume_per_txn]
    """
    # Build node index
    all_hashes = list({n["hash"] for n in nodes})
    # Also include hashes from edges (may not be in active nodes but needed for topology)
    for e in edges:
        if e["src"] not in all_hashes:
            all_hashes.append(e["src"])
        if e["dst"] not in all_hashes:
            all_hashes.append(e["dst"])

    node_to_idx = {h: i for i, h in enumerate(all_hashes)}
    n_nodes     = len(all_hashes)

    # Build feature matrix
    node_lookup = {n["hash"]: n for n in nodes}
    feat_matrix = []
    for h in all_hashes:
        n = node_lookup.get(h, {})
        out_deg    = float(n.get("out_degree",        0))
        in_deg     = float(n.get("in_degree",         0))
        avg_out    = float(n.get("avg_out_amount",    0))
        total_out  = float(n.get("total_out_volume",  0))
        avg_in     = float(n.get("avg_in_amount",     0))
        fan_in     = in_deg / (out_deg + 1)
        vol_per_txn = total_out / (out_deg + 1)
        feat_matrix.append([
            min(out_deg    / 1000.0,       1.0),
            min(in_deg     / 1000.0,       1.0),
            min(avg_out    / 100_000.0,    1.0),
            min(total_out  / 1_000_000.0,  1.0),
            min(avg_in     / 100_000.0,    1.0),
            min(fan_in     / 100.0,        1.0),
            min(vol_per_txn / 100_000.0,   1.0),
        ])

    x = torch.tensor(feat_matrix, dtype=torch.float)

    # Pad features to match model input dimension (166 from combined graph)
    target_dim = 166  # must match in_channels from graphsage_config.json
    if x.shape[1] < target_dim:
        pad = torch.zeros(n_nodes, target_dim - x.shape[1])
        x   = torch.cat([x, pad], dim=1)

    # Build edge index
    src_list, dst_list = [], []
    for e in edges:
        if e["src"] in node_to_idx and e["dst"] in node_to_idx:
            src_list.append(node_to_idx[e["src"]])
            dst_list.append(node_to_idx[e["dst"]])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Dummy labels (not used at inference time)
    y = torch.zeros(n_nodes, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), all_hashes


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_embeddings(
    model:   "FraudGNN",
    data:    Data,
    hashes:  list[str],
    device:  torch.device,
    out_dim: int,
) -> dict[str, list[float]]:
    """
    Run model inference and return {hash: embedding_list} dict.
    """
    data = data.to(device)
    all_nodes = torch.arange(data.num_nodes)

    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=all_nodes,
        shuffle=False,
        num_workers=0,
        disjoint=False,
    )

    # Accumulate embeddings by node index
    emb_accumulator = torch.zeros(data.num_nodes, out_dim)
    count_tracker   = torch.zeros(data.num_nodes)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch      = batch.to(device)
            embeddings = model(batch.x, batch.edge_index)[:batch.batch_size].cpu()
            seed_ids   = batch.n_id[:batch.batch_size]
            emb_accumulator[seed_ids] += embeddings
            count_tracker[seed_ids]   += 1

    # Average for nodes that appeared in multiple batches
    counts = count_tracker.clamp(min=1).unsqueeze(1)
    embeddings_final = (emb_accumulator / counts).numpy()

    return {h: embeddings_final[i].tolist() for i, h in enumerate(hashes)}


# ─────────────────────────────────────────────────────────────────────────────
# REDIS STORE
# ─────────────────────────────────────────────────────────────────────────────

def store_embeddings(r: redis.Redis, embeddings: dict[str, list[float]]) -> int:
    """Batch-store embeddings in Redis using pipeline. Returns count stored."""
    pipe    = r.pipeline(transaction=False)
    stored  = 0

    for h, emb in embeddings.items():
        key = f"gnn_embedding:{h}"
        pipe.set(key, json.dumps(emb), ex=EMBEDDING_TTL)
        stored += 1
        if stored % 1000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)

    pipe.execute()
    return stored


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("SENTINEL EMBEDDING JOB")
    log.info("=" * 60)
    log.info("  Neo4j:    %s", NEO4J_URI)
    log.info("  Redis:    %s:%s", REDIS_HOST, REDIS_PORT)
    log.info("  Model:    %s", MODEL_PATH)
    log.info("  Interval: %ds", REFRESH_INTERVAL)

    # Load model
    try:
        model, config, device = load_model_and_config()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    out_dim = config["out_channels"]
    in_dim  = config["in_channels"]

    # Connect Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        log.info("  Neo4j: connected")
    except Exception as e:
        log.error("  Neo4j: FAILED — %s", e)
        return

    # Connect Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        log.info("  Redis: connected")
    except Exception as e:
        log.error("  Redis: FAILED — %s", e)
        driver.close()
        return

    # Graceful shutdown
    stop_event = Event()
    signal.signal(signal.SIGINT,  lambda s, f: stop_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())

    cycle   = 0
    total   = 0

    log.info("Embedding refresh loop started...")

    while not stop_event.is_set():
        cycle_start = time.time()
        cycle      += 1

        try:
            # Step 1: Find recently-updated nodes
            active_nodes = query_active_nodes(driver, LOOKBACK_SECONDS)

            if not active_nodes:
                log.debug("  Cycle %d: no active nodes, skipping", cycle)
                stop_event.wait(timeout=REFRESH_INTERVAL)
                continue

            active_hashes = [n["hash"] for n in active_nodes]
            log.info(
                "  Cycle %d: %d active nodes to re-embed", cycle, len(active_nodes)
            )

            # Step 2: Fetch subgraph edges for neighborhood context
            edges = query_subgraph_edges(driver, active_hashes)

            # Step 3: Build PyG subgraph
            data, all_hashes = build_subgraph(active_nodes, edges)

            # Step 4: Inference
            embeddings = compute_embeddings(model, data, all_hashes, device, out_dim)

            # Step 5: Store in Redis
            stored = store_embeddings(r, embeddings)
            total += stored

            elapsed = round((time.time() - cycle_start) * 1000, 1)
            log.info(
                "  Cycle %d: stored %d embeddings in %sms (total: %d)",
                cycle, stored, elapsed, total
            )

        except Exception as e:
            log.error("  Cycle %d failed: %s", cycle, e, exc_info=True)

        # Wait for next cycle
        elapsed_s = time.time() - cycle_start
        sleep_s   = max(0, REFRESH_INTERVAL - elapsed_s)
        stop_event.wait(timeout=sleep_s)

    driver.close()
    log.info("Embedding job stopped. Total embeddings stored: %d", total)


if __name__ == "__main__":
    run()