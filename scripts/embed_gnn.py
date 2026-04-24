"""
scripts/embed_gnn.py
====================
Computes 32-dimensional GNN embeddings for all Identity nodes
and stores them in Redis for sub-millisecond lookup at serving time.

Runs once after train_graphsage.py, then again every 60 seconds
as a background job (services/graph_engine/embedding_job.py) to
incorporate new transactions into the embeddings.

Redis key format:
  gnn_embedding:{identity_hash} → JSON list of 32 floats
  TTL: 1 hour (refreshed by embedding_job.py)

Usage:
  python -m scripts.embed_gnn
"""

import json
import os
from pathlib import Path

import numpy as np
import redis
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from dotenv import load_dotenv

load_dotenv()

from scripts.train_graphsage import FraudGNN, load_elliptic, load_neo4j_graph, combine_graphs

REDIS_HOST   = os.getenv("REDIS_HOST",   "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))
MODEL_PATH   = Path("models/graphsage_weights.pt")
CONFIG_PATH  = Path("models/graphsage_config.json")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sentinel_neo4j")

EMBEDDING_TTL  = 3600   # 1 hour TTL — refreshed by embedding_job.py
BATCH_SIZE     = 2048
NUM_NEIGHBORS  = [15, 10]


def load_model(config: dict, device: torch.device) -> FraudGNN:
    model = FraudGNN(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def get_node_hashes_from_neo4j() -> list[str]:
    """Get all Identity node hashes from Neo4j."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        with driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (n:Identity) RETURN n.hash AS hash LIMIT 500000"
            )
            hashes = [r["hash"] for r in result]
        driver.close()
        return hashes
    except Exception as e:
        print(f"  [neo4j] {e}")
        return []


def compute_and_store_embeddings(
    model:     FraudGNN,
    data:      Data,
    node_hashes: list[str],
    r:         redis.Redis,
    device:    torch.device,
):
    """
    Run inference on all nodes, store embeddings in Redis.
    Uses NeighborLoader for memory efficiency.
    """
    # Full-graph inference loader
    data = data.cpu()
    all_nodes = torch.arange(data.num_nodes)
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=all_nodes,
        shuffle=False,
    )

    all_embeddings = torch.zeros(data.num_nodes, model.conv3.out_channels)
    node_counts    = torch.zeros(data.num_nodes)

    model.eval()
    processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb   = model(batch.x, batch.edge_index)
            # Only store embeddings for the "seed" nodes (first batch_size nodes)
            seed_emb   = emb[:batch.batch_size].cpu()
            seed_nodes = batch.n_id[:batch.batch_size]
            # all_embeddings[seed_nodes] += seed_emb
            all_embeddings[seed_nodes.cpu()] += seed_emb.cpu()
            # node_counts[seed_nodes]    += 1
            node_counts[seed_nodes.cpu()] += 1
            processed += batch.batch_size
            if processed % 10000 == 0:
                print(f"  Computed embeddings: {processed:,}/{data.num_nodes:,}")

    # Average embeddings for nodes that appeared in multiple batches
    node_counts = node_counts.clamp(min=1).unsqueeze(1)
    all_embeddings = all_embeddings / node_counts

    # Store in Redis
    print(f"\n  Storing {min(len(node_hashes), data.num_nodes):,} embeddings in Redis...")
    pipe = r.pipeline(transaction=False)
    stored = 0

    for i, h in enumerate(node_hashes):
        if i >= data.num_nodes:
            break
        emb_list = all_embeddings[i].tolist()
        key      = f"gnn_embedding:{h}"
        pipe.set(key, json.dumps(emb_list), ex=EMBEDDING_TTL)
        stored += 1

        if stored % 5000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            print(f"  Stored: {stored:,}")

    pipe.execute()
    print(f"  Total stored: {stored:,} embeddings in Redis")
    return stored


def main():
    print("=" * 60)
    print("GNN EMBEDDING JOB")
    print("=" * 60)

    # Verify model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Run: python -m scripts.train_graphsage")
        return

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"  Model: {config['out_channels']}-dim embeddings")
    num_nodes = config.get('num_nodes_trained', 'Unknown')
    print(f"  Trained on: {num_nodes} nodes")

    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        r.ping()
        print(f"  Redis: connected ({REDIS_HOST}:{REDIS_PORT})")
    except Exception as e:
        print(f"  Redis: FAILED — {e}")
        print("  Run: docker compose up -d redis")
        return

    # Device
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"  Device: {device}")

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(config, device)

    # Load graph (same as training)
    print("\n[2/4] Loading graphs...")
    elliptic = load_elliptic()
    neo4j    = load_neo4j_graph()
    data     = combine_graphs(elliptic, neo4j)
    data     = data.to(device)

    # Get node hashes for Redis key mapping
    print("\n[3/4] Getting node hashes from Neo4j...")
    # neo4j_hashes = get_node_hashes_from_neo4j()
    with open("models/neo4j_node_order.json") as f:
        neo4j_hashes = json.load(f)
    print(f"  Loaded {len(neo4j_hashes):,} hashes from saved node order")
    # Elliptic nodes use transaction IDs — not in Redis (no identity_hash)
    # Only store embeddings for Neo4j Identity nodes
    elliptic_count = elliptic.num_nodes
    # Neo4j nodes start at index elliptic_count in the combined graph
    neo4j_start = elliptic_count

    print(f"  Neo4j identity hashes: {len(neo4j_hashes):,}")

    # Compute and store
    print("\n[4/4] Computing and storing embeddings...")
    # Create a modified data object with only Neo4j nodes for embedding storage
    # (Elliptic transaction nodes don't have identity_hash keys in Redis)
    neo4j_data = Data(
        x=data.x[neo4j_start:],
        edge_index=data.edge_index,  # full graph for context
        y=data.y[neo4j_start:],
    )

    stored = compute_and_store_embeddings(
        model=model,
        data=data,
        node_hashes=neo4j_hashes,
        r=r,
        device=device,
    )

    print("\n" + "=" * 60)
    print("EMBEDDING JOB COMPLETE")
    print("=" * 60)
    print(f"  Embeddings stored: {stored:,}")
    print(f"  Redis key format:  gnn_embedding:{{identity_hash}}")
    print(f"  TTL:               {EMBEDDING_TTL}s (1 hour)")
    print(f"\n  Verify a sample embedding:")
    if neo4j_hashes:
        sample_key = f"gnn_embedding:{neo4j_hashes[0]}"
        sample     = r.get(sample_key)
        if sample:
            emb = json.loads(sample)
            print(f"    Key: {sample_key[:40]}...")
            print(f"    Dim: {len(emb)}")
            print(f"    Values: [{emb[0]:.4f}, {emb[1]:.4f}, ...]")
    print(f"\nNext: python pipeline.py --use-gnn-embeddings")
    print("=" * 60)


if __name__ == "__main__":
    main()