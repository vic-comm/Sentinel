"""
scripts/crawl_alchemy.py
========================
Stage 1: Labeled training data (depth 1 only)

Crawls wallets that directly interacted with known mixer/fraud addresses.
These wallets are CONFIRMED bad actors — label quality is HIGH (0.75).

Also crawls known legitimate DeFi protocols for legitimate training rows.

Output: data/processed/alchemy_transactions.parquet
        ~25-100K rows, is_fraud labeled, ready for build_features.py

Stage 2 (graph expansion) is in crawl_alchemy_graph.py — that script
builds connectivity without assigning fraud labels.
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
MAX_WALLETS      = 2000   # unique wallets to crawl
TXNS_PER_WALLET  = 50     # transactions per wallet
CONCURRENCY      = 20     # parallel requests
# ─────────────────────────────────────────────────────────────────

# Known fraud seeds → wallets interacting with these are confirmed bad actors
FRAUD_SEEDS = {
    "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936": "tornado_cash",
    "0xA160cdAB225685dA1d56aa342Ad8841c3b53f291": "blender_io",
    "0xD691F27f38B395864Ea86b4007b2B75Fc8C8f4be": "tornado_cash_v2",
}

# Known legitimate seeds → wallets interacting with these are likely legit
LEGIT_SEEDS = {
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "uniswap_v3",
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "uniswap_v2",
    "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9": "aave_v2",
}


# async def fetch_transfers(session, address, max_count=50, direction="from"):
#     payload = {
#         "jsonrpc": "2.0",
#         "method":  "alchemy_getAssetTransfers",
#         "params":  [{
#             "fromBlock":    "0x0",
#             "toBlock":      "latest",
#             "fromAddress":  address,
#             "category":     ["external", "internal"],
#             "withMetadata": True,
#             "maxCount":     hex(max_count),
#         }],
#         "id": 1,
#     }

#     if direction == "to":
#         payload["params"][0]["toAddress"] = address
#     else:
#         payload["params"][0]["fromAddress"] = address

#     try:
#         async with session.post(
#             URL, json=payload,
#             timeout=aiohttp.ClientTimeout(total=15)
#         ) as r:
#             data = await r.json()
#             return data.get("result", {}).get("transfers", [])
#     except Exception as e:
#         print(f"  Error fetching {address[:8]}: {e}")
#         return []

async def fetch_transfers(session, address, max_count=50, direction="from"):
    params = {
        "fromBlock":    "0x0",
        "toBlock":      "latest",
        "category":     ["external", "internal"],
        "withMetadata": True,
        "maxCount":     hex(max_count),
    }

    # Set direction EXCLUSIVELY — never both at once
    if direction == "to":
        params["toAddress"] = address
    else:
        params["fromAddress"] = address

    payload = {
        "jsonrpc": "2.0",
        "method":  "alchemy_getAssetTransfers",
        "params":  [params],
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
    
def to_row(tx, is_fraud, source_label):
    sender   = (tx.get("from") or "").lower()
    receiver = (tx.get("to")   or "").lower()
    amt      = float(tx.get("value") or 0)
    ts       = tx.get("metadata", {}).get("blockTimestamp")
    return {
        "transaction_id":          f"eth_{(tx.get('hash') or '')[:16]}",
        "client_id":               "cryptoex_prod",
        "timestamp":               ts,
        "modality":                "crypto",
        "identity_hash":           hashlib.sha256(
                                       f"{sender}{SHARED_SALT}".encode()
                                   ).hexdigest(),
        "sender_hash":             hashlib.sha256(sender.encode()).hexdigest(),
        "receiver_hash":           hashlib.sha256(receiver.encode()).hexdigest(),
        "sender_wallet_raw":       sender,   # kept for graph expansion stage
        "receiver_wallet_raw":     receiver, # kept for graph expansion stage
        "amount_usd":              amt * 2800,
        "amount_crypto":           amt,
        "cryptocurrency":          tx.get("asset", "ETH"),
        "blockchain":              "ethereum",
        "transaction_hash":        tx.get("hash", ""),
        "is_fraud":                is_fraud,
        "fraud_type":              "mixer_usage" if is_fraud else None,
        "source":                  f"alchemy_{source_label}",
        "known_mixer_interaction": is_fraud,
        "label_depth":             1,        # explicit: this is a depth-1 label
    }


async def collect_wallets_from_seeds(seeds, is_fraud):
    """
    Hit each seed address, collect wallets that interacted with it.
    Returns list of (wallet_address, is_fraud, source_label).
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    wallets = []

    async def get_seed_wallets(seed_addr, source_label):
        async with sem:
            txns = await fetch_transfers(session, seed_addr, max_count=200, direction="to")
            found = set()
            for tx in txns:
                w = (tx.get("from") or "").lower()
                if w and w != seed_addr.lower():
                    found.add((w, is_fraud, source_label))
            print(f"  {source_label}: {len(found)} wallets found")
            return list(found)

    async with aiohttp.ClientSession() as session:   # ← session created HERE
        tasks = [get_seed_wallets(addr, label) for addr, label in seeds.items()]
        results = await asyncio.gather(*tasks)
    for batch in results:
        wallets.extend(batch)
    return wallets


# async def crawl_wallets(session, wallet_list):
#     """
#     Fetch transaction history for each wallet concurrently.
#     Returns flat list of row dicts.
#     """
#     sem = asyncio.Semaphore(CONCURRENCY)
#     rows = []
#     completed = 0

#     async def crawl_one(address, is_fraud, source_label):
#         nonlocal completed
#         async with sem:
#             txns = await fetch_transfers(session, address, TXNS_PER_WALLET, direction="from")
#             wallet_rows = [to_row(t, is_fraud, source_label) for t in txns]
#             completed += 1
#             if completed % 100 == 0:
#                 pct = completed / len(wallet_list) * 100
#                 print(f"  Progress: {completed}/{len(wallet_list)} "
#                       f"({pct:.0f}%) | {len(rows):,} rows")
#             return wallet_rows

#     tasks = [crawl_one(addr, fraud, label) for addr, fraud, label in wallet_list]
#     results = await asyncio.gather(*tasks)
#     for batch in results:
#         rows.extend(batch)
#     return rows


# async def run():
#     print("Phase A: Collecting wallets from seed addresses...")
#     async with aiohttp.ClientSession() as session:
#         fraud_wallets = await collect_wallets_from_seeds(
#             session, FRAUD_SEEDS, is_fraud=True)
#         legit_wallets = await collect_wallets_from_seeds(
#             session, LEGIT_SEEDS, is_fraud=False)

#     # Deduplicate — a wallet can't be both fraud and legit
#     fraud_addrs = {w[0] for w in fraud_wallets}
#     legit_wallets = [w for w in legit_wallets if w[0] not in fraud_addrs]

#     # Cap total
#     all_wallets = fraud_wallets + legit_wallets
#     seen = set()
#     unique = []
#     for item in all_wallets:
#         if item[0] not in seen:
#             seen.add(item[0])
#             unique.append(item)
#     unique = unique[:MAX_WALLETS]

#     n_fraud = sum(1 for w in unique if w[1])
#     n_legit = sum(1 for w in unique if not w[1])
#     print(f"\n  Total wallets: {len(unique)} "
#           f"({n_fraud} fraud, {n_legit} legit)")

#     print(f"\nPhase B: Crawling {len(unique)} wallets...")
#     async with aiohttp.ClientSession() as session:
#         rows = await crawl_wallets(session, unique)

#     # Save raw wallet list for Stage 2 graph expansion
#     wallet_df = pd.DataFrame([
#         {"wallet": w, "is_fraud": f, "source": s}
#         for w, f, s in unique
#     ])
#     wallet_df.to_parquet("data/processed/alchemy_wallet_seeds.parquet")
#     print(f"\n  Saved {len(wallet_df)} wallet seeds "
#           f"→ data/processed/alchemy_wallet_seeds.parquet")
#     print("  (Stage 2 graph expansion uses this file)")

#     return rows

async def crawl_wallets(wallet_list):
    sem = asyncio.Semaphore(CONCURRENCY)
    rows = []
    completed = 0

    async def crawl_one(address, is_fraud, source_label):
        nonlocal completed
        async with sem:
            txns = await fetch_transfers(
                session, address, TXNS_PER_WALLET, direction="from"
            )
            wallet_rows = [to_row(t, is_fraud, source_label) for t in txns]
            completed += 1
            if completed % 100 == 0:
                pct = completed / len(wallet_list) * 100
                print(f"  Progress: {completed}/{len(wallet_list)} ({pct:.0f}%)")
            return wallet_rows

    async with aiohttp.ClientSession() as session:   # ← session created HERE
        tasks = [crawl_one(addr, fraud, label) for addr, fraud, label in wallet_list]
        results = await asyncio.gather(*tasks)

    for batch in results:
        rows.extend(batch)
    return rows

async def run():
    print("Phase A: Collecting wallets from seed addresses...")

    # No session passed — each function manages its own
    fraud_wallets = await collect_wallets_from_seeds(FRAUD_SEEDS, is_fraud=True)
    legit_wallets = await collect_wallets_from_seeds(LEGIT_SEEDS, is_fraud=False)

    # Deduplicate
    fraud_addrs   = {w[0] for w in fraud_wallets}
    legit_wallets = [w for w in legit_wallets if w[0] not in fraud_addrs]

    all_wallets = fraud_wallets + legit_wallets
    seen, unique = set(), []
    for item in all_wallets:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    unique = unique[:MAX_WALLETS]

    n_fraud = sum(1 for w in unique if w[1])
    n_legit = sum(1 for w in unique if not w[1])
    print(f"\n  Total wallets: {len(unique)} ({n_fraud} fraud, {n_legit} legit)")

    print(f"\nPhase B: Crawling {len(unique)} wallets...")
    rows = await crawl_wallets(unique)   # ← no session passed here either

    wallet_df = pd.DataFrame([
        {"wallet": w, "is_fraud": f, "source": s}
        for w, f, s in unique
    ])
    wallet_df.to_parquet("data/processed/alchemy_wallet_seeds.parquet")
    print(f"\n  Saved {len(wallet_df)} wallet seeds → data/processed/alchemy_wallet_seeds.parquet")

    return rows

def main():
    print("=" * 60)
    print("ALCHEMY CRAWLER — Stage 1 (labeled training data)")
    print(f"  Max wallets:    {MAX_WALLETS}")
    print(f"  Txns/wallet:    {TXNS_PER_WALLET}")
    print(f"  Concurrency:    {CONCURRENCY}")
    print(f"  Depth:          1 (confirmed labels only)")
    print("=" * 60)

    rows = asyncio.run(run())

    if not rows:
        print("No data collected.")
        return

    df = pd.DataFrame(rows)
    df = df[df["amount_usd"] > 0]
    df = df.drop_duplicates("transaction_id")

    print(f"\nResults:")
    print(f"  Total rows:  {len(df):,}")
    print(f"  Fraud rows:  {df['is_fraud'].sum():,} "
          f"({df['is_fraud'].mean()*100:.1f}%)")
    print(f"  Sources:     {df['source'].value_counts().to_dict()}")

    df.to_parquet("data/processed/alchemy_transactions.parquet")
    print(f"\nSaved → data/processed/alchemy_transactions.parquet")
    print("\nNext: python -m scripts.crawl_alchemy_graph")


if __name__ == "__main__":
    main()