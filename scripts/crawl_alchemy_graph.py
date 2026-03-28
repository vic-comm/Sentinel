"""
scripts/crawl_alchemy_graph.py
==============================
Stage 2: Graph connectivity expansion (depth 2, NO fraud labels)

Takes the depth-1 confirmed wallets from Stage 1 and expands the
transaction graph one more hop outward. This builds richer graph
topology for GraphSAGE WITHOUT polluting training labels.

Why separation matters:
  Depth 1 wallets: CONFIRMED fraud/legit (used for XGBoost training)
  Depth 2 wallets: UNKNOWN — may be victims, exchanges, or more fraudsters
                   (used ONLY for graph structure, never for label training)

What GraphSAGE learns from depth-2 connectivity:
  - Flow patterns: money moves through intermediaries before reaching mixer
  - Hub detection: some depth-2 nodes aggregate from many depth-1 nodes
  - Temporal clustering: depth-2 transactions cluster temporally with depth-1

Output: data/processed/alchemy_graph_edges.parquet
        Columns: sender_hash, receiver_hash, amount_usd, timestamp,
                 depth, sender_wallet_raw, receiver_wallet_raw
        NO is_fraud column — this data is not used for supervised training.
        It feeds Neo4j graph construction and GraphSAGE only.

Usage:
    Run AFTER crawl_alchemy.py:
    python -m scripts.crawl_alchemy_graph
"""

import asyncio
import aiohttp
import hashlib
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ALCHEMY_KEY = os.getenv("ALCHEMY_API_KEY")
assert ALCHEMY_KEY, "Missing ALCHEMY_API_KEY"
URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
Path("data/processed").mkdir(exist_ok=True)

SHARED_SALT = "sentinel_consortium_2026_v1"

# ── TUNING KNOBS ─────────────────────────────────────────────────
MAX_DEPTH2_WALLETS = 3000   # how many depth-2 wallets to expand
TXNS_PER_WALLET    = 30     # fewer txns needed — just connectivity
CONCURRENCY        = 20
# ─────────────────────────────────────────────────────────────────


async def fetch_transfers(session, address, max_count=30):
    payload = {
        "jsonrpc": "2.0",
        "method":  "alchemy_getAssetTransfers",
        "params":  [{
            "fromBlock":    "0x0",
            "toBlock":      "latest",
            "fromAddress":  address,
            "category":     ["external", "internal"],
            "withMetadata": True,
            "maxCount":     hex(max_count),
        }],
        "id": 1,
    }
    try:
        async with session.post(
            URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            data = await r.json()
            return data.get("result", {}).get("transfers", [])
    except Exception as e:
        print(f"  Error fetching {address[:8]}: {e}")
        return []


def to_edge_row(tx, depth, source_wallet_is_fraud):
    """
    Build a graph edge row.
    Critically: NO is_fraud column.
    depth_1_fraud_connected indicates proximity to confirmed fraud
    but does NOT mean this transaction is fraudulent.
    """
    sender   = (tx.get("from") or "").lower()
    receiver = (tx.get("to")   or "").lower()
    amt      = float(tx.get("value") or 0)
    ts       = tx.get("metadata", {}).get("blockTimestamp")

    return {
        # Graph structure fields
        "sender_hash":              hashlib.sha256(sender.encode()).hexdigest(),
        "receiver_hash":            hashlib.sha256(receiver.encode()).hexdigest(),
        "sender_wallet_raw":        sender,
        "receiver_wallet_raw":      receiver,

        # Transaction details
        "amount_usd":               amt * 2800,
        "amount_crypto":            amt,
        "cryptocurrency":           tx.get("asset", "ETH"),
        "timestamp":                ts,
        "transaction_hash":         tx.get("hash", ""),
        "blockchain":               "ethereum",

        # Graph metadata — NOT a fraud label
        "depth":                    depth,
        "depth_1_fraud_connected":  source_wallet_is_fraud,

        # Explicitly NOT included: is_fraud, fraud_type
        # These rows go to Neo4j graph only, never to XGBoost training
        "graph_only":               True,
    }


async def expand_to_depth2(depth1_wallets):
    """
    For each depth-1 wallet, find the wallets it sent money TO.
    Those are depth-2 wallets. Then crawl their transaction histories.

    depth1_wallets: list of (wallet_address, is_fraud) tuples
    """
    print(f"Expanding {len(depth1_wallets)} depth-1 wallets to depth-2...")

    sem = asyncio.Semaphore(CONCURRENCY)
    depth2_candidates = {}  # wallet_addr → source_is_fraud

    # Find depth-2 wallets (receivers of depth-1 wallets)
    async def find_receivers(address, is_fraud):
        async with sem:
            txns = await fetch_transfers(session, address, max_count=20)
            receivers = {}
            for tx in txns:
                r = (tx.get("to") or "").lower()
                if r:
                    # Inherit fraud proximity from sender
                    receivers[r] = is_fraud
            return receivers

    async with aiohttp.ClientSession() as session:
        tasks = [find_receivers(addr, fraud) for addr, fraud in depth1_wallets]
        results = await asyncio.gather(*tasks)

    for result in results:
        for addr, fraud in result.items():
            if addr not in depth2_candidates:
                depth2_candidates[addr] = fraud

    # Remove depth-1 wallets (already crawled in Stage 1)
    depth1_addrs = {w[0] for w in depth1_wallets}
    depth2_new = {
        addr: fraud
        for addr, fraud in depth2_candidates.items()
        if addr not in depth1_addrs
    }

    # Cap and sort — prioritise fraud-connected wallets first
    fraud_connected    = [(a, f) for a, f in depth2_new.items() if f]
    non_fraud_connected = [(a, f) for a, f in depth2_new.items() if not f]

    # Take fraud-connected first, fill rest with non-fraud-connected
    n_fraud_take = min(len(fraud_connected),
                       int(MAX_DEPTH2_WALLETS * 0.7))
    n_legit_take = min(len(non_fraud_connected),
                       MAX_DEPTH2_WALLETS - n_fraud_take)

    selected = fraud_connected[:n_fraud_take] + non_fraud_connected[:n_legit_take]

    print(f"\nDepth-2 wallets selected: {len(selected)}")
    print(f"  Fraud-connected:  {n_fraud_take}")
    print(f"  Other connected:  {n_legit_take}")

    return selected


async def crawl_depth2_edges(depth2_wallets):
    """
    Fetch transaction histories for depth-2 wallets.
    Returns edge rows (graph connectivity only, no fraud labels).
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    rows = []
    completed = 0

    async def crawl_one(address, source_fraud):
        nonlocal completed
        async with sem:
            txns = await fetch_transfers(session, address, TXNS_PER_WALLET)
            edge_rows = [
                to_edge_row(t, depth=2, source_wallet_is_fraud=source_fraud)
                for t in txns
            ]
            completed += 1
            if completed % 200 == 0:
                pct = completed / len(depth2_wallets) * 100
                print(f"  Progress: {completed}/{len(depth2_wallets)} "
                      f"({pct:.0f}%) | {len(rows):,} edges")
            return edge_rows

    async with aiohttp.ClientSession() as session:
        tasks = [crawl_one(addr, fraud) for addr, fraud in depth2_wallets]
        results = await asyncio.gather(*tasks)

    for batch in results:
        rows.extend(batch)
    return rows


async def run():
    # Load depth-1 wallet seeds from Stage 1
    seeds_path = Path("data/processed/alchemy_wallet_seeds.parquet")
    if not seeds_path.exists():
        print("ERROR: data/processed/alchemy_wallet_seeds.parquet not found.")
        print("Run Stage 1 first: python -m scripts.crawl_alchemy")
        return []

    seeds_df = pd.read_parquet(seeds_path)
    depth1_wallets = list(zip(seeds_df["wallet"], seeds_df["is_fraud"]))
    print(f"Loaded {len(depth1_wallets)} depth-1 wallets from Stage 1")

    # Also load depth-1 transaction data to include as depth-1 edges
    d1_path = Path("data/processed/alchemy_transactions.parquet")
    depth1_edges = []
    if d1_path.exists():
        d1_df = pd.read_parquet(d1_path)
        # Convert to edge format (no is_fraud, just connectivity)
        for _, row in d1_df.iterrows():
            depth1_edges.append({
                "sender_hash":             row.get("sender_hash", ""),
                "receiver_hash":           row.get("receiver_hash", ""),
                "sender_wallet_raw":       row.get("sender_wallet_raw", ""),
                "receiver_wallet_raw":     row.get("receiver_wallet_raw", ""),
                "amount_usd":              row.get("amount_usd", 0),
                "amount_crypto":           row.get("amount_crypto", 0),
                "cryptocurrency":          row.get("cryptocurrency", "ETH"),
                "timestamp":               row.get("timestamp"),
                "transaction_hash":        row.get("transaction_hash", ""),
                "blockchain":              "ethereum",
                "depth":                   1,
                "depth_1_fraud_connected": row.get("is_fraud", False),
                "graph_only":              False,  # depth-1 also in training
            })
        print(f"Loaded {len(depth1_edges):,} depth-1 edges")

    # Expand to depth 2
    depth2_wallets = await expand_to_depth2(depth1_wallets)

    # Crawl depth-2 edges
    print(f"\nCrawling {len(depth2_wallets)} depth-2 wallets...")
    depth2_edges = await crawl_depth2_edges(depth2_wallets)

    return depth1_edges + depth2_edges


def main():
    print("=" * 60)
    print("ALCHEMY GRAPH CRAWLER — Stage 2 (connectivity only)")
    print(f"  Max depth-2 wallets: {MAX_DEPTH2_WALLETS}")
    print(f"  Txns per wallet:     {TXNS_PER_WALLET}")
    print(f"  Concurrency:         {CONCURRENCY}")
    print(f"  Fraud labels:        NONE (graph only)")
    print("=" * 60)

    rows = asyncio.run(run())

    if not rows:
        print("No edges collected.")
        return

    df = pd.DataFrame(rows)
    df = df[df["amount_usd"] > 0]
    df = df.drop_duplicates("transaction_hash")

    # Verify no is_fraud column leaked in
    assert "is_fraud" not in df.columns, \
        "is_fraud column should not exist in graph edges — check to_edge_row()"

    print(f"\nResults:")
    print(f"  Total edges:          {len(df):,}")
    print(f"  Depth-1 edges:        {(df['depth'] == 1).sum():,}")
    print(f"  Depth-2 edges:        {(df['depth'] == 2).sum():,}")
    print(f"  Fraud-connected:      {df['depth_1_fraud_connected'].sum():,}")
    print(f"  Unique sender hashes: {df['sender_hash'].nunique():,}")

    df.to_parquet("data/processed/alchemy_graph_edges.parquet")
    print(f"\nSaved → data/processed/alchemy_graph_edges.parquet")

    print("\nThis file feeds Neo4j graph construction and GraphSAGE.")
    print("It does NOT go into XGBoost training data.")
    print("\nNext: python -m scripts.load_neo4j_graph  (loads edges into Neo4j)")


if __name__ == "__main__":
    main()