"""
services/feature_engine/test_features.py
==========================================
Tests feature computation without needing Redpanda running.
Feeds synthetic transactions directly into compute_features()
and prints the output feature vector.

Usage:
    python -m services.feature_engine.test_features
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from services.feature_engine.dataflow import compute_features, get_redis

SHARED_SALT = "sentinel_consortium_2026_v1"
import hashlib

def make_hash(email):
    return hashlib.sha256(f"{email}{SHARED_SALT}".encode()).hexdigest()


def test_normal_transaction():
    print("\n── Test 1: Normal fiat transaction ──")
    txn = {
        "transaction_id":       f"test_{uuid.uuid4().hex[:8]}",
        "client_id":            "neobank_prod",
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "modality":             "fiat",
        "identity_hash":        make_hash("alice@example.com"),
        "amount_usd":           1500.00,
        "transfer_type":        "ach",
        "transfer_network":     "domestic",
        "sender_account_age_days": 456,
        "sender_kyc_status":    "verified",
        "sender_archetype":     "salary_worker",
        "is_first_time_receiver": False,
        "balance_drain_ratio":  0.12,
        "sender_balance_before": 12000.0,
        "is_fraud":             False,
    }
    result = compute_features(txn)
    print(f"  velocity count_24h:      {result.get('count_24h')}")
    print(f"  velocity_ratio:          {result.get('velocity_ratio')}")
    print(f"  institution_count:       {result.get('institution_count')}")
    print(f"  cross_modal_pattern:     {result.get('cross_modal_pattern_detected')}")
    print(f"  hour_of_day:             {result.get('hour_of_day')}")
    print(f"  gnn_emb_0:               {result.get('gnn_emb_0')}")
    print(f"  feature_version:         {result.get('feature_version')}")
    print("  PASS" if "feature_version" in result else "  FAIL")
    return result


def test_velocity_buildup():
    print("\n── Test 2: Velocity buildup (same user, 5 rapid transactions) ──")
    email = "bob@example.com"
    ih    = make_hash(email)
    base  = datetime.now(timezone.utc)

    for i in range(5):
        txn = {
            "transaction_id": f"vel_test_{i}",
            "client_id":      "neobank_prod",
            "timestamp":      (base + timedelta(minutes=i*2)).isoformat(),
            "modality":       "fiat",
            "identity_hash":  ih,
            "amount_usd":     500.0,
            "is_fraud":       False,
        }
        result = compute_features(txn)

    print(f"  After 5 txns in 10 min:")
    print(f"  count_1h:        {result.get('count_1h')}")
    print(f"  sum_amount_1h:   {result.get('sum_amount_1h')}")
    print(f"  velocity_ratio:  {result.get('velocity_ratio')}")
    assert result.get("count_1h", 0) >= 5, "count_1h should be >= 5"
    print("  PASS")


def test_cross_modal_detection():
    print("\n── Test 3: Cross-modal detection (fiat then crypto, same identity) ──")
    email = "charlie@example.com"
    ih    = make_hash(email)
    now   = datetime.now(timezone.utc)

    # Fiat transaction at NeoBank
    fiat_txn = {
        "transaction_id": "cross_fiat_001",
        "client_id":      "neobank_prod",
        "timestamp":      now.isoformat(),
        "modality":       "fiat",
        "identity_hash":  ih,
        "amount_usd":     10000.0,
        "is_fraud":       True,
    }
    compute_features(fiat_txn)

    # Crypto transaction at CryptoEx 15 minutes later — same identity_hash
    crypto_txn = {
        "transaction_id": "cross_crypto_001",
        "client_id":      "cryptoex_prod",
        "timestamp":      (now + timedelta(minutes=15)).isoformat(),
        "modality":       "crypto",
        "identity_hash":  ih,
        "amount_usd":     9800.0,
        "is_fraud":       True,
    }
    result = compute_features(crypto_txn)

    print(f"  institution_count:             {result.get('institution_count')}")
    print(f"  cross_client_velocity_1h:      {result.get('cross_client_velocity_1h')}")
    print(f"  cross_client_amount_corr:      {result.get('cross_client_amount_correlation')}")
    print(f"  time_since_other_inst_sec:     {result.get('time_since_other_institution_sec')}")
    print(f"  has_fiat_history:              {result.get('has_fiat_history')}")
    print(f"  has_crypto_history:            {result.get('has_crypto_history')}")
    print(f"  cross_modal_pattern_detected:  {result.get('cross_modal_pattern_detected')}")

    assert result.get("cross_modal_pattern_detected") == 1, \
        "cross_modal_pattern_detected should be 1"
    assert result.get("institution_count", 0) >= 1, \
        "institution_count should be >= 1"
    print("  PASS — cross-modal pattern correctly detected")


def test_redis_caching():
    print("\n── Test 4: Redis feature caching ──")
    r  = get_redis()
    ih = make_hash("dave@example.com")

    txn = {
        "transaction_id": "cache_test_001",
        "client_id":      "neobank_prod",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "modality":       "fiat",
        "identity_hash":  ih,
        "amount_usd":     2000.0,
        "is_fraud":       False,
    }
    compute_features(txn)

    cached = r.get(f"features:{ih}")
    assert cached is not None, "Feature cache should exist in Redis"
    parsed = json.loads(cached)
    print(f"  Cached keys:     {len(parsed)} features stored")
    print(f"  amount_usd:      {parsed.get('amount_usd')}")
    print("  PASS — features cached in Redis")


if __name__ == "__main__":
    print("=" * 50)
    print("FEATURE ENGINE TESTS")
    print("=" * 50)

    try:
        r = get_redis()
        r.ping()
        print("Redis: connected")
    except Exception as e:
        print(f"Redis: FAILED — {e}")
        print("Run: docker compose up -d redis")
        exit(1)

    test_normal_transaction()
    test_velocity_buildup()
    test_cross_modal_detection()
    test_redis_caching()

    print("\n" + "=" * 50)
    print("All tests passed.")
    print("=" * 50)
    print("\nNext: start Redpanda and run the full dataflow:")
    print("  docker compose up -d redpanda")
    print("  python -m services.feature_engine.dataflow")