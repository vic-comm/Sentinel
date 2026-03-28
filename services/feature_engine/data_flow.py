"""
services/feature_engine/dataflow.py
=====================================
Sentinel Feature Engineering Service — Bytewax streaming pipeline.

Sits between Redpanda and the model serving layer.

Flow:
  Redpanda (transactions.ingested)
      ↓
  Bytewax stateful operators
      ├── velocity features     (count/sum over 1h, 6h, 24h, 7d windows)
      ├── cross-client features (same identity_hash at 2+ institutions)
      ├── temporal features     (hour, day, unusual_hour flag)
      └── GNN embedding lookup  (Redis → 32-dim vector)
      ↓
  Redpanda (transactions.features_ready)
      +
  Redis  (features:{identity_hash} → latest feature vector)
      +
  PostgreSQL (offline feature store for training data)

Architecture note:
  Hopsworks replaced with Redis (online) + PostgreSQL (offline).
  Same semantics, zero license cost, easier local dev.

Dependencies:
  pip install bytewax==0.21 kafka-python redis psycopg2-binary

Usage:
  python -m services.feature_engine.dataflow

  Or via Docker:
  docker compose up feature_engine
"""

import json
import os
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

import redis as redis_lib
from bytewax.dataflow import Dataflow
from bytewax.inputs import KafkaSourceConfig, KafkaSinkConfig
import bytewax.operators as op
from bytewax.connectors.kafka import KafkaSource, KafkaSink, KafkaSourceMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [feature_engine] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REDPANDA_BROKERS  = os.getenv("REDPANDA_BROKERS",  "localhost:9092")
REDIS_HOST        = os.getenv("REDIS_HOST",         "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT",     "6379"))
PG_DSN            = os.getenv("PG_DSN",
    "postgresql://sentinel:sentinel@localhost:5432/sentinel")

INPUT_TOPIC       = "transactions.ingested"
OUTPUT_TOPIC      = "transactions.features_ready"
CONSUMER_GROUP    = "feature_engine_v1"

# Window sizes in hours
VELOCITY_WINDOWS  = [1, 6, 24, 168]   # 168 = 7 days
CROSS_CLIENT_TTL  = timedelta(days=7)

# Redis key prefixes
REDIS_FEATURES_PREFIX   = "features:"      # latest feature vector per identity
REDIS_VELOCITY_PREFIX   = "vel:"           # rolling window state
REDIS_CROSS_CLIENT_PREFIX = "xc:"          # cross-client history
REDIS_GNN_PREFIX        = "gnn_embedding:" # pre-computed GNN embeddings

# Feature TTL
FEATURE_TTL_SECONDS = 60 * 60 * 24 * 90   # 90 days

# ─────────────────────────────────────────────────────────────────────────────
# REDIS CLIENT (shared across operators)
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: Optional[redis_lib.Redis] = None

def get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY STATE HELPERS
# Rolling window aggregation stored in Redis as sorted sets.
# Key:   vel:{identity_hash}:{window_hours}
# Value: sorted set of (score=unix_ts, member=json(amount))
# ─────────────────────────────────────────────────────────────────────────────

def update_velocity(
    r: redis_lib.Redis,
    identity_hash: str,
    ts: datetime,
    amount_usd: float,
) -> dict[str, float]:
    """
    Adds the current transaction to all velocity windows.
    Prunes entries older than the largest window (7d).
    Returns count and sum features for each window.
    """
    unix_ts   = ts.timestamp()
    member    = json.dumps({"ts": unix_ts, "amt": amount_usd})
    pipe_key  = f"{REDIS_VELOCITY_PREFIX}{identity_hash}"

    r_pipe = r.pipeline()
    r_pipe.zadd(pipe_key, {member: unix_ts})
    # Prune anything older than 7 days
    cutoff_7d = (ts - timedelta(hours=168)).timestamp()
    r_pipe.zremrangebyscore(pipe_key, "-inf", cutoff_7d)
    r_pipe.expire(pipe_key, FEATURE_TTL_SECONDS)
    r_pipe.execute()

    # Fetch all remaining entries
    raw_entries = r.zrangebyscore(pipe_key, cutoff_7d, "+inf", withscores=True)

    feats = {}
    for window_hours in VELOCITY_WINDOWS:
        cutoff = (ts - timedelta(hours=window_hours)).timestamp()
        window_entries = [
            json.loads(m) for m, score in raw_entries
            if float(score) >= cutoff
        ]
        n   = len(window_entries)
        amt = sum(e["amt"] for e in window_entries)
        label = f"{window_hours}h" if window_hours < 168 else "7d"
        feats[f"count_{label}"]      = n
        feats[f"sum_amount_{label}"] = round(amt, 2)

    return feats


def get_velocity_baseline(
    r: redis_lib.Redis,
    identity_hash: str,
    ts: datetime,
) -> tuple[float, float]:
    """
    Returns (avg_txns_per_day, avg_amount) over the available history.
    Used to compute velocity_ratio and amount_ratio.
    """
    pipe_key = f"{REDIS_VELOCITY_PREFIX}{identity_hash}"
    cutoff   = (ts - timedelta(hours=168)).timestamp()
    entries  = r.zrangebyscore(pipe_key, cutoff, "+inf", withscores=True)

    if not entries:
        return 1.0, 1.0

    parsed   = [json.loads(m) for m, _ in entries]
    n        = len(parsed)
    total    = sum(e["amt"] for e in parsed)
    avg_amt  = total / n if n > 0 else 1.0

    # Days of history we actually have
    oldest_ts  = min(e["ts"] for e in parsed)
    days_span  = max(1.0, (ts.timestamp() - oldest_ts) / 86400)
    avg_per_day = n / days_span

    return round(avg_per_day, 4), round(avg_amt, 2)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CLIENT STATE HELPERS
# Tracks every time an identity_hash appears at any institution.
# Key:   xc:{identity_hash}
# Value: JSON list of {client_id, ts, amount_usd, modality, risk_score}
# ─────────────────────────────────────────────────────────────────────────────

def update_cross_client(
    r: redis_lib.Redis,
    identity_hash: str,
    client_id: str,
    ts: datetime,
    amount_usd: float,
    modality: str,
    risk_score: float = 0.0,
) -> dict[str, Any]:
    """
    Appends current transaction to cross-client history.
    Returns cross-client features for the current transaction.
    """
    xc_key  = f"{REDIS_CROSS_CLIENT_PREFIX}{identity_hash}"
    history = []

    raw = r.get(xc_key)
    if raw:
        try:
            history = json.loads(raw)
        except Exception:
            history = []

    # Prune entries older than 7 days
    cutoff_ts = (ts - CROSS_CLIENT_TTL).timestamp()
    history   = [h for h in history if h["ts"] >= cutoff_ts]

    # Compute cross-client features BEFORE appending current transaction
    institutions = list({h["client_id"] for h in history})
    n_institutions = len(institutions)

    # Cross-client velocity: txns at OTHER institutions in last 1h
    one_hour_ago  = (ts - timedelta(hours=1)).timestamp()
    xc_velocity_1h = sum(
        1 for h in history
        if h["client_id"] != client_id and h["ts"] >= one_hour_ago
    )

    # Amount correlation: compare current to prior transactions at other institutions
    other_amounts = [
        h["amount_usd"] for h in history
        if h["client_id"] != client_id
    ]
    amount_corr = 0.0
    if other_amounts:
        # Pearson-like: how close is current amount to mean of other-institution amounts?
        mean_other = sum(other_amounts) / len(other_amounts)
        amount_corr = round(
            1.0 - abs(amount_usd - mean_other) / max(mean_other, 1.0),
            4
        )
        amount_corr = max(0.0, min(1.0, amount_corr))

    # Time since last appearance at a DIFFERENT institution
    other_ts = [h["ts"] for h in history if h["client_id"] != client_id]
    time_since_other_sec = None
    if other_ts:
        time_since_other_sec = round(ts.timestamp() - max(other_ts), 1)

    # Detect the critical fiat→crypto pattern
    fiat_modalities   = {h["modality"] for h in history if h["modality"] == "fiat"}
    crypto_modalities = {h["modality"] for h in history if h["modality"] == "crypto"}
    has_fiat_history   = len(fiat_modalities) > 0
    has_crypto_history = len(crypto_modalities) > 0

    # Prior high-risk event at another institution (delayed laundering detection)
    prior_high_risk = any(
        h.get("risk_score", 0.0) >= 0.70 and h["client_id"] != client_id
        for h in history
    )

    feats = {
        "institution_count":              n_institutions,
        "cross_client_velocity_1h":       xc_velocity_1h,
        "cross_client_amount_correlation": amount_corr,
        "time_since_other_institution_sec": time_since_other_sec,
        "has_fiat_history":               int(has_fiat_history),
        "has_crypto_history":             int(has_crypto_history),
        "cross_modal_pattern_detected":   int(has_fiat_history and has_crypto_history),
        "prior_high_risk_event":          int(prior_high_risk),
        "institutions_seen":              institutions,  # list, not used as feature
    }

    # Append current transaction to history
    history.append({
        "client_id":  client_id,
        "ts":         ts.timestamp(),
        "amount_usd": amount_usd,
        "modality":   modality,
        "risk_score": risk_score,
    })

    r.set(xc_key, json.dumps(history), ex=FEATURE_TTL_SECONDS)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GNN EMBEDDING LOOKUP
# GraphSAGE pre-computes 32-dim embeddings every 60s via embed_gnn.py.
# We just do a Redis lookup here — <1ms.
# ─────────────────────────────────────────────────────────────────────────────

def get_gnn_embedding(
    r: redis_lib.Redis,
    identity_hash: str,
) -> list[float]:
    """
    Returns the pre-computed 32-dim GNN embedding for this identity.
    Falls back to zeros if embedding not yet computed (new user).
    """
    key = f"{REDIS_GNN_PREFIX}{identity_hash}"
    raw = r.get(key)
    if raw:
        try:
            emb = json.loads(raw)
            if isinstance(emb, list) and len(emb) == 32:
                return emb
        except Exception:
            pass
    return [0.0] * 32


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL FEATURES
# Pure computation — no state needed.
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_features(ts: datetime) -> dict[str, Any]:
    return {
        "hour_of_day":     ts.hour,
        "day_of_week":     ts.weekday(),
        "is_weekend":      int(ts.weekday() >= 5),
        "is_unusual_hour": int(ts.hour < 6 or ts.hour >= 23),
        "is_business_hour": int(9 <= ts.hour <= 17 and ts.weekday() < 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE COMPUTATION
# Called once per transaction. Assembles all feature groups.
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(txn: dict) -> dict:
    """
    Takes a raw transaction dict (from transactions.ingested topic).
    Returns the same dict enriched with ~100 features.
    """
    r = get_redis()

    identity_hash = (
        txn.get("identity_hash")
        or txn.get("sender_hash")
        or txn.get("user_email_hash")
        or ""
    )
    if not identity_hash:
        log.warning("Transaction missing identity_hash: %s",
                    txn.get("transaction_id", "?"))
        txn["feature_error"] = "missing_identity_hash"
        return txn

    # Parse timestamp
    ts_raw = txn.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)

    amount_usd  = float(txn.get("amount_usd", 0) or 0)
    client_id   = txn.get("client_id", "unknown")
    modality    = txn.get("modality", "fiat")

    # ── 1. Velocity features ─────────────────────────────────────────────────
    vel_feats = update_velocity(r, identity_hash, ts, amount_usd)
    baseline_rate, baseline_amt = get_velocity_baseline(r, identity_hash, ts)

    velocity_ratio = round(
        vel_feats.get("count_24h", 0) / max(baseline_rate, 0.1), 3
    )
    amount_ratio = round(
        amount_usd / max(baseline_amt, 1.0), 3
    )

    # ── 2. Cross-client features ──────────────────────────────────────────────
    xc_feats = update_cross_client(
        r, identity_hash, client_id, ts, amount_usd, modality
    )

    # ── 3. Temporal features ──────────────────────────────────────────────────
    temp_feats = compute_temporal_features(ts)

    # ── 4. GNN embedding ─────────────────────────────────────────────────────
    gnn_embedding = get_gnn_embedding(r, identity_hash)
    gnn_feats = {f"gnn_emb_{i}": v for i, v in enumerate(gnn_embedding)}

    # ── 5. Passthrough features from ingestion enrichment ────────────────────
    passthrough = {
        "amount_usd":                   amount_usd,
        "sender_account_age_days":      txn.get("sender_account_age_days", -1),
        "sender_kyc_status":            txn.get("sender_kyc_status", "unknown"),
        "sender_archetype":             txn.get("sender_archetype", "unknown"),
        "is_first_time_receiver":       int(txn.get("is_first_time_receiver", False) or False),
        "balance_drain_ratio":          float(txn.get("balance_drain_ratio", 0) or 0),
        "sender_balance_before":        float(txn.get("sender_balance_before", 0) or 0),
        "modality":                     modality,
        "transfer_type":                txn.get("transfer_type", "unknown"),
        "transfer_network":             txn.get("transfer_network", "unknown"),
        "client_id":                    client_id,
        # Crypto-specific
        "known_mixer_interaction":      int(txn.get("known_mixer_interaction", False) or False),
        "receiver_wallet_age_days":     txn.get("receiver_wallet_age_days", -1),
        "receiver_wallet_type":         txn.get("receiver_wallet_type", "unknown"),
        "cryptocurrency":               txn.get("cryptocurrency", ""),
        "blockchain":                   txn.get("blockchain", ""),
        "gas_price_gwei":               txn.get("gas_price_gwei", -1),
        # Source quality (real data flag)
        "source_confidence":            float(txn.get("source_confidence", 0.5) or 0.5),
        "is_real_data":                 int(txn.get("is_real_data", 0) or 0),
    }

    # ── Assemble complete enriched transaction ────────────────────────────────
    enriched = {
        **txn,
        # velocity
        **vel_feats,
        "velocity_ratio":               velocity_ratio,
        "amount_ratio":                 amount_ratio,
        "velocity_baseline_daily":      baseline_rate,
        # cross-client (drop institutions_seen list — not a model feature)
        "institution_count":            xc_feats["institution_count"],
        "cross_client_velocity_1h":     xc_feats["cross_client_velocity_1h"],
        "cross_client_amount_correlation": xc_feats["cross_client_amount_correlation"],
        "time_since_other_institution_sec": xc_feats["time_since_other_institution_sec"],
        "has_fiat_history":             xc_feats["has_fiat_history"],
        "has_crypto_history":           xc_feats["has_crypto_history"],
        "cross_modal_pattern_detected": xc_feats["cross_modal_pattern_detected"],
        "prior_high_risk_event":        xc_feats["prior_high_risk_event"],
        # temporal
        **temp_feats,
        # passthrough
        **passthrough,
        # gnn embeddings (32 dims)
        **gnn_feats,
        # metadata
        "features_computed_at": ts.isoformat(),
        "feature_version":      "v1",
    }

    # ── Cache latest feature vector in Redis ─────────────────────────────────
    cache_key = f"{REDIS_FEATURES_PREFIX}{identity_hash}"
    # Store only the numeric/categorical features (not the full txn)
    feature_cache = {
        k: v for k, v in enriched.items()
        if k not in ("sender_hash", "receiver_hash", "identity_hash",
                     "transaction_hash", "email", "institutions_seen")
        and not isinstance(v, (dict, list))
    }
    try:
        r.set(cache_key, json.dumps(feature_cache), ex=FEATURE_TTL_SECONDS)
    except Exception as e:
        log.warning("Failed to cache features for %s: %s", identity_hash[:8], e)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# BYTEWAX DATAFLOW
# ─────────────────────────────────────────────────────────────────────────────

def deserialize(msg: KafkaSourceMessage) -> Optional[dict]:
    """Parse incoming Kafka message. Return None to drop malformed messages."""
    try:
        return json.loads(msg.value)
    except Exception as e:
        log.error("Failed to deserialize message: %s", e)
        return None


def serialize(txn: dict) -> bytes:
    """Serialize enriched transaction for output topic."""
    return json.dumps(txn, default=str).encode("utf-8")


def build_dataflow() -> Dataflow:
    """
    Constructs the Bytewax dataflow:
      Redpanda input
        → deserialize
        → compute_features  (stateful via Redis)
        → serialize
        → Redpanda output
    """
    flow = Dataflow("sentinel_feature_engine")

    # Input: consume from transactions.ingested
    kafka_input = KafkaSource(
        brokers=[REDPANDA_BROKERS],
        topics=[INPUT_TOPIC],
        group_id=CONSUMER_GROUP,
        starting_offset="end",   # process live transactions only
        add_config={
            "auto.offset.reset":        "latest",
            "enable.auto.commit":       "true",
            "session.timeout.ms":       "30000",
            "max.poll.interval.ms":     "300000",
        },
    )

    stream = op.input("kafka_in", flow, kafka_input)

    # Deserialize JSON → dict
    parsed = op.filter_map(
        "deserialize",
        stream,
        deserialize,
    )

    # Compute all features (stateful — reads/writes Redis)
    enriched = op.map(
        "compute_features",
        parsed,
        compute_features,
    )

    # Serialize dict → bytes
    serialized = op.map(
        "serialize",
        enriched,
        lambda txn: (
            txn.get("transaction_id", "").encode(),  # Kafka key
            serialize(txn),                           # Kafka value
        ),
    )

    # Output: publish to transactions.features_ready
    kafka_output = KafkaSink(
        brokers=[REDPANDA_BROKERS],
        topic=OUTPUT_TOPIC,
    )

    op.output("kafka_out", serialized, kafka_output)

    return flow


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import bytewax.run

    log.info("Starting Sentinel Feature Engine")
    log.info("  Input:  %s → %s", REDPANDA_BROKERS, INPUT_TOPIC)
    log.info("  Output: %s → %s", REDPANDA_BROKERS, OUTPUT_TOPIC)
    log.info("  Redis:  %s:%s", REDIS_HOST, REDIS_PORT)

    # Verify Redis connection before starting
    try:
        r = get_redis()
        r.ping()
        log.info("  Redis connection: OK")
    except Exception as e:
        log.error("Cannot connect to Redis: %s", e)
        raise

    flow = build_dataflow()
    bytewax.run.cli_main(flow)