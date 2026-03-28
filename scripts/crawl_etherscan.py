# scripts/crawl_etherscan.py
import requests, time, hashlib, uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY    = os.getenv("ETHERSCAN_API_KEY")
SHARED_SALT = "sentinel_consortium_2026_v1"
Path("data/processed").mkdir(exist_ok=True)

assert API_KEY, "ETHERSCAN_API_KEY not set in .env"

# Known bad seed addresses (publicly documented)
SEEDS = {
    "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936": "tornado_cash",
    "0xA160cdAB225685dA1d56aa342Ad8841c3b53f291": "blender_io",
}

# Known legitimate DeFi protocols (for legit user crawl)
LEGIT_SEEDS = {
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "uniswap_v3",
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "uniswap_v2",
}

# def get_txns(address, max_rows=50):
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account&action=txlist"
#         f"&address={address}&sort=desc"
#         f"&offset={max_rows}&apikey={API_KEY}"
#     )
#     time.sleep(0.21)
#     try:
#         r = requests.get(url, timeout=15).json()
#         if r["status"] == "1" and isinstance(r["result"], list):
#             return r["result"]
#     except Exception as e:
#         print(f"  Error: {e}")
#     return []

def get_txns(address, max_rows=50):
    url = (
        f"https://api.etherscan.io/api"
        f"?module=account&action=txlist"
        f"&address={address}&sort=desc"
        f"&offset={max_rows}&apikey={API_KEY}"
    )
    time.sleep(0.25)

    try:
        r = requests.get(url, timeout=15).json()

        if r["status"] != "1":
            print(f"  API ERROR for {address}: {r}")
            return []

        return r["result"]

    except Exception as e:
        print(f"  Request failed: {e}")
        return []

test = get_txns("0x742d35Cc6634C0532925a3b844Bc454e4438f44e", 1)
print("API test result:", test[:1])

def to_row(txn, is_fraud, fraud_type, source_label):
    sender   = txn.get("from", "").lower()
    receiver = txn.get("to",   "").lower()
    amt_eth  = float(txn.get("value", 0)) / 1e18
    amt_usd  = amt_eth * 2800
    ts_epoch = int(txn.get("timeStamp", 0))
    ts       = datetime.utcfromtimestamp(ts_epoch).isoformat() + "Z"

    return {
        "transaction_id":           f"eth_{txn['hash'][:16]}",
        "client_id":                "cryptoex_prod",
        "timestamp":                ts,
        "modality":                 "crypto",
        "identity_hash":            hashlib.sha256(f"{sender}{SHARED_SALT}".encode()).hexdigest(),
        "sender_wallet_hash":       hashlib.sha256(sender.encode()).hexdigest(),
        "receiver_wallet_hash":     hashlib.sha256(receiver.encode()).hexdigest(),
        "amount_usd":               round(amt_usd, 2),
        "amount_crypto":            round(amt_eth, 8),
        "cryptocurrency":           "ETH",
        "blockchain":               "ethereum",
        "transaction_hash":         txn.get("hash", ""),
        "gas_price_gwei":           int(txn.get("gasPrice", 0)) // 10**9,
        "block_number":             int(txn.get("blockNumber", 0)),
        "known_mixer_interaction":  source_label in ("tornado_cash", "blender_io"),
        "receiver_wallet_age_days": None,
        "is_fraud":                 is_fraud,
        "fraud_type":               fraud_type,
        "source":                   f"etherscan_{source_label}",
    }

rows = []

# ── Phase A: Crawl mixer users (fraud) ───────────────────────────────────────
print("Phase A: Crawling mixer seed addresses...")
mixer_wallets = set()
for seed_addr, seed_label in SEEDS.items():
    print(f"  {seed_label}: {seed_addr}")
    txns = get_txns(seed_addr, max_rows=500)
    for t in txns:
        sender = t.get("from","").lower()
        if sender and sender != seed_addr.lower():
            mixer_wallets.add((sender, seed_label))
        rows.append(to_row(t, True, "mixer_usage", seed_label))
    print(f"    → {len(mixer_wallets)} mixer users found so far")

print(f"\nTotal mixer users found: {len(mixer_wallets)}")
mixer_list = list(mixer_wallets)[:1000]  # cap at 1000

print("\nPhase B: Crawling mixer user histories...")
for i, (wallet, label) in enumerate(mixer_list):
    if i % 100 == 0:
        print(f"  {i}/{len(mixer_list)} wallets processed, {len(rows):,} rows so far")
        # Save checkpoint every 100 wallets
        pd.DataFrame(rows).to_json(
            "data/processed/etherscan_checkpoint.jsonl",
            orient="records", lines=True)
    txns = get_txns(wallet, max_rows=50)
    for t in txns:
        rows.append(to_row(t, True, "mixer_usage", label))

# ── Phase B: Crawl legitimate DeFi users (legit) ─────────────────────────────
print("\nPhase C: Crawling legitimate DeFi users...")
legit_wallets = set()
for addr, label in LEGIT_SEEDS.items():
    txns = get_txns(addr, max_rows=200)
    for t in txns:
        legit_wallets.add((t.get("from","").lower(), label))
    print(f"  {label}: found {len(legit_wallets)} users")

for wallet, label in list(legit_wallets)[:300]:
    txns = get_txns(wallet, max_rows=30)
    for t in txns:
        rows.append(to_row(t, False, None, f"legit_{label}"))

# ── Save final output ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
print(df.columns)
df = df[df["amount_usd"] > 0]   # remove failed/zero-value txns
df = df.drop_duplicates(subset="transaction_id")

df.to_json("data/processed/etherscan_real_crypto.jsonl",
            orient="records", lines=True)

print(f"\nFinal: {len(df):,} real Ethereum transactions")
print(df["is_fraud"].value_counts())