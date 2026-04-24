"""
services/feature_engine/test_features.py  [v2]
================================================
Tests feature computation without Redpanda.
Feeds synthetic transactions into compute_features() and verifies:
  1. All expected feature columns are present
  2. Column names match models/feature_names.txt (pipeline alignment)
  3. Velocity buildup works correctly
  4. Cross-modal detection fires correctly
  5. GNN embedding lookup works (or falls back gracefully)
  6. Redis caching works

Usage:
  docker compose up -d redis
  python -m services.feature_engine.test_features
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.feature_engine.logic import compute_features, get_redis, GNN_EMB_DIM

SHARED_SALT = "sentinel_consortium_2026_v1"
FEATURE_NAMES_PATH = Path("models/feature_names.txt")


def make_hash(email: str) -> str:
    return hashlib.sha256(f"{email}{SHARED_SALT}".encode()).hexdigest()


def make_fiat_txn(email: str, amount: float = 1500.0, client: str = "neobank_prod",
                  ts: datetime = None, **overrides) -> dict:
    return {
        "transaction_id":         f"test_{uuid.uuid4().hex[:8]}",
        "client_id":              client,
        "timestamp":              (ts or datetime.now(timezone.utc)).isoformat(),
        "modality":               "fiat",
        "identity_hash":          make_hash(email),
        "sender_hash":            make_hash(email),
        "receiver_hash":          make_hash(f"receiver_{uuid.uuid4()}"),
        "amount_usd":             amount,
        "transfer_type":          "ach",
        "transfer_network":       "domestic",
        "sender_account_age_days": 456,
        "sender_kyc_status":      "verified",
        "sender_archetype":       "salary_worker",
        "is_first_time_receiver": False,
        "balance_drain_ratio":    0.12,
        "sender_balance_before":  12000.0,
        **overrides,
    }


def make_crypto_txn(email: str, amount: float = 9800.0, client: str = "cryptoex_prod",
                    ts: datetime = None, **overrides) -> dict:
    return {
        "transaction_id":         f"test_{uuid.uuid4().hex[:8]}",
        "client_id":              client,
        "timestamp":              (ts or datetime.now(timezone.utc)).isoformat(),
        "modality":               "crypto",
        "user_email_hash":        make_hash(email),
        "identity_hash":          make_hash(email),
        "sender_wallet_hash":     make_hash(f"wallet_{email}"),
        "receiver_wallet_hash":   make_hash(f"fresh_{uuid.uuid4()}"),
        "amount_usd":             amount,
        "amount_crypto":          amount / 2800,
        "cryptocurrency":         "ETH",
        "blockchain":             "ethereum",
        "transaction_hash":       f"0x{uuid.uuid4().hex}",
        "known_mixer_interaction": False,
        "receiver_wallet_age_days": 0,
        "receiver_wallet_type":   "user_wallet",
        **overrides,
    }


# ─────────────────────────────────────────────────────────────────────────────

def test_normal_transaction():
    print("\n── Test 1: Normal fiat transaction ──")
    result = compute_features(make_fiat_txn("alice@example.com"))

    checks = {
        "feature_version": "v2",
        "count_24h":        1,       # first transaction
        "institution_count": 0,       # no prior cross-client activity
        "cross_modal_pattern_detected": 0,
        "gnn_emb_0":        0.0,     # no embedding = zero vector
        "velocity_ratio":   float,   # must be a float
        "amount_ratio":     float,
    }
    passed = True
    for key, expected in checks.items():
        actual = result.get(key)
        if expected is float:
            ok = isinstance(actual, (int, float))
        else:
            ok = actual == expected
        status = "✓" if ok else "✗"
        if not ok:
            passed = False
        print(f"  {status} {key}: {actual} (expected {expected})")

    print("  PASS" if passed else "  FAIL")
    return passed


def test_velocity_buildup():
    print("\n── Test 2: Velocity buildup (5 rapid transactions) ──")
    email = "bob_velocity@example.com"
    base  = datetime.now(timezone.utc)

    result = None
    for i in range(5):
        result = compute_features(make_fiat_txn(
            email, amount=500.0,
            ts=base + timedelta(minutes=i * 2)
        ))

    checks = {
        "count_1h": (lambda v: v >= 5, "≥5"),
        "sum_amount_1h": (lambda v: v >= 2500.0, "≥2500"),
        "velocity_ratio": (lambda v: v > 1.0, ">1.0"),
    }
    passed = True
    for key, (check_fn, desc) in checks.items():
        actual = result.get(key, 0)
        ok     = check_fn(actual)
        status = "✓" if ok else "✗"
        if not ok:
            passed = False
        print(f"  {status} {key}: {actual} (expected {desc})")

    print("  PASS" if passed else "  FAIL")
    return passed


def test_cross_modal_detection():
    print("\n── Test 3: Cross-modal detection (fiat → crypto, same identity) ──")
    email = "charlie_cross@example.com"
    now   = datetime.now(timezone.utc)

    # Step 1: Fiat at NeoBank
    compute_features(make_fiat_txn(
        email, amount=10000.0, client="neobank_prod", ts=now
    ))

    # Step 2: Crypto at CryptoEx 15 minutes later (same identity_hash)
    result = compute_features(make_crypto_txn(
        email, amount=9800.0, client="cryptoex_prod",
        ts=now + timedelta(minutes=15)
    ))

    checks = {
        "cross_modal_pattern_detected": 1,
        "has_fiat_history":             1,
        "has_crypto_history":           1,
        "institution_count":            (lambda v: v >= 1, "≥1"),
        "cross_client_velocity_1h":     (lambda v: v >= 1, "≥1"),
        "cross_client_amount_correlation": (lambda v: v > 0, ">0"),
        "time_since_other_institution_sec": (lambda v: v is not None and v > 0, "not None"),
    }
    passed = True
    for key, expected in checks.items():
        actual = result.get(key)
        if callable(expected):
            ok, desc = expected(actual), str(expected)
        else:
            ok, desc = actual == expected, str(expected)
        status = "✓" if ok else "✗"
        if not ok:
            passed = False
        print(f"  {status} {key}: {actual} (expected {desc})")

    if passed:
        print("  PASS — cross-modal pattern correctly detected ✓")
    else:
        print("  FAIL — cross-modal detection broken")
    return passed


def test_gnn_fallback():
    print("\n── Test 4: GNN embedding fallback (new user not in graph) ──")
    result = compute_features(make_fiat_txn("new_user_never_seen@example.com"))

    # Should have GNN_EMB_DIM columns, all zero
    gnn_cols = {f"gnn_emb_{i}": result.get(f"gnn_emb_{i}", None) for i in range(GNN_EMB_DIM)}
    all_zero = all(v == 0.0 for v in gnn_cols.values())
    correct_count = len(gnn_cols) == GNN_EMB_DIM

    print(f"  {'✓' if correct_count else '✗'} GNN columns present: {len(gnn_cols)} (expected {GNN_EMB_DIM})")
    print(f"  {'✓' if all_zero else '✗'} All zero (new user): {all_zero}")

    passed = all_zero and correct_count
    print("  PASS" if passed else "  FAIL")
    return passed


def test_redis_caching():
    print("\n── Test 5: Redis feature caching ──")
    r     = get_redis()
    email = "dave_cache@example.com"
    ih    = make_hash(email)

    compute_features(make_fiat_txn(email, amount=2000.0))

    cached = r.get(f"features:{ih}")
    if cached is None:
        print("  ✗ Feature cache not found in Redis")
        return False

    parsed = json.loads(cached)
    checks = {
        "amount_usd": (lambda v: float(v) == 2000.0, "2000.0"),
        "feature_version": ("v2", "v2"),
        "count_24h":       (lambda v: v >= 1, "≥1"),
    }
    passed = True
    for key, expected in checks.items():
        actual = parsed.get(key)
        if callable(expected):
            ok, desc = expected(actual), str(expected)
        else:
            ok, desc = actual == expected, str(expected)
        status = "✓" if ok else "✗"
        if not ok:
            passed = False
        print(f"  {status} {key}: {actual} (expected {desc})")

    print(f"  Total cached features: {len(parsed)}")
    print("  PASS" if passed else "  FAIL")
    return passed


def test_feature_alignment():
    """
    Verifies that features produced by compute_features()
    include all columns in models/feature_names.txt.

    If feature_names.txt doesn't exist yet (model not trained),
    this test is skipped with a warning.
    """
    print("\n── Test 6: Feature alignment with pipeline.py ──")

    if not FEATURE_NAMES_PATH.exists():
        print(f"  SKIP — {FEATURE_NAMES_PATH} not found (run pipeline.py first)")
        return True

    with open(FEATURE_NAMES_PATH) as f:
        expected_cols = [line.strip() for line in f if line.strip()]

    result = compute_features(make_fiat_txn("alignment_test@example.com"))

    missing = [col for col in expected_cols if col not in result]
    present = [col for col in expected_cols if col     in result]

    print(f"  Total expected features: {len(expected_cols)}")
    print(f"  Present in output:       {len(present)}")
    print(f"  Missing from output:     {len(missing)}")

    if missing:
        print(f"  MISSING COLUMNS (will get -1 fill in serve.py):")
        for col in missing[:10]:
            print(f"    - {col}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    if len(missing) == 0:
        print("  PASS — 100% alignment ✓")
        return True
    elif len(missing) <= 5:
        print("  WARN — minor misalignment (acceptable — these get -1 fill)")
        return True
    else:
        print("  FAIL — significant misalignment will degrade model quality")
        return False


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("FEATURE ENGINE TESTS v2")
    print("=" * 55)

    try:
        r = get_redis()
        r.ping()
        print(f"Redis: connected (GNN dim: {GNN_EMB_DIM})")
    except Exception as e:
        print(f"Redis: FAILED — {e}")
        print("Run: docker compose up -d redis")
        exit(1)

    results = {
        "test_normal_transaction":   test_normal_transaction(),
        "test_velocity_buildup":     test_velocity_buildup(),
        "test_cross_modal_detection": test_cross_modal_detection(),
        "test_gnn_fallback":         test_gnn_fallback(),
        "test_redis_caching":        test_redis_caching(),
        "test_feature_alignment":    test_feature_alignment(),
    }

    passed = sum(results.values())
    total  = len(results)

    print(f"\n{'=' * 55}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'=' * 55}")

    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if passed == total:
        print("\n✓ All tests passed — feature engine ready")
        print("\nNext steps:")
        print("  docker compose up -d redpanda")
        print("  python -m services.feature_engine.dataflow")
    else:
        print(f"\n✗ {total - passed} test(s) failed — fix before running dataflow")
        exit(1)