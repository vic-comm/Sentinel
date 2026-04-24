"""
services/feature_engine/logic.py
==========================================
Upgrades from v1:
  1. count_7d / sum_amount_7d (not count_168h) — matches pipeline.py column names
  2. GNN embedding dimension loaded from graphsage_config.json at startup
  3. Missing features fill with -1 (not 0) — matches pipeline.py fillna(-1)
  4. cross_modality_fraud_id and fraud_network_id passed through to serve.py

"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import redis as redis_lib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [feature_engine] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REDPANDA_BROKERS          = os.getenv("REDPANDA_BROKERS",  "localhost:9092")
REDIS_HOST                = os.getenv("REDIS_HOST",         "localhost")
REDIS_PORT                = int(os.getenv("REDIS_PORT",     "6379"))
CONFIG_PATH               = Path(os.getenv("CONFIG_PATH",   "models/graphsage_config.json"))

INPUT_TOPIC               = "transactions.ingested"
OUTPUT_TOPIC              = "transactions.features_ready"
CONSUMER_GROUP            = "feature_engine_v1"

VELOCITY_WINDOWS          = [1, 6, 24, 168]
CROSS_CLIENT_TTL          = timedelta(days=7)
FEATURE_TTL_SECONDS       = 60 * 60 * 24 * 90

REDIS_FEATURES_PREFIX     = "features:"
REDIS_VELOCITY_PREFIX     = "vel:"
REDIS_CROSS_CLIENT_PREFIX = "xc:"
REDIS_GNN_PREFIX          = "gnn_embedding:"

# Load GNN dim from config
def _load_gnn_dim() -> int:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return int(json.load(f).get("out_channels", 32))
        except Exception:
            pass
    return 32

GNN_EMB_DIM = _load_gnn_dim()

# ─────────────────────────────────────────────────────────────────────────────
_redis_client: Optional[redis_lib.Redis] = None

def get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client

# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY

def update_velocity(r, identity_hash, ts, amount_usd):
    unix_ts  = ts.timestamp()
    member   = json.dumps({"ts": unix_ts, "amt": amount_usd})
    key      = f"{REDIS_VELOCITY_PREFIX}{identity_hash}"
    cutoff7d = (ts - timedelta(hours=168)).timestamp()

    pipe = r.pipeline()
    pipe.zadd(key, {member: unix_ts})
    pipe.zremrangebyscore(key, "-inf", cutoff7d)
    pipe.expire(key, FEATURE_TTL_SECONDS)
    pipe.execute()

    raw_entries = r.zrangebyscore(key, cutoff7d, "+inf", withscores=True)
    feats       = {}
    # FIXED: use count_7d / sum_amount_7d (not count_168h) to match pipeline.py
    labels      = {1: "1h", 6: "6h", 24: "24h", 168: "7d"}

    for wh in VELOCITY_WINDOWS:
        cutoff = (ts - timedelta(hours=wh)).timestamp()
        window = [json.loads(m) for m, s in raw_entries if float(s) >= cutoff]
        label  = labels[wh]
        feats[f"count_{label}"]      = len(window)
        feats[f"sum_amount_{label}"] = round(sum(e["amt"] for e in window), 2)
    return feats


def get_velocity_baseline(r, identity_hash, ts):
    key     = f"{REDIS_VELOCITY_PREFIX}{identity_hash}"
    cutoff  = (ts - timedelta(hours=168)).timestamp()
    entries = r.zrangebyscore(key, cutoff, "+inf", withscores=True)
    if not entries:
        return 1.0, 1.0
    parsed     = [json.loads(m) for m, _ in entries]
    n          = len(parsed)
    total      = sum(e["amt"] for e in parsed)
    oldest_ts  = min(e["ts"] for e in parsed)
    days_span  = max(1.0, (ts.timestamp() - oldest_ts) / 86400)
    return round(n / days_span, 4), round(total / n if n > 0 else 1.0, 2)

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CLIENT

def update_cross_client(r, identity_hash, client_id, ts, amount_usd, modality, risk_score=0.0):
    key     = f"{REDIS_CROSS_CLIENT_PREFIX}{identity_hash}"
    history = []
    raw     = r.get(key)
    if raw:
        try:
            history = json.loads(raw)
        except Exception:
            pass

    cutoff_ts = (ts - CROSS_CLIENT_TTL).timestamp()
    history   = [h for h in history if h["ts"] >= cutoff_ts]

    institutions   = list({h["client_id"] for h in history})
    n_institutions = len(institutions)
    one_hour_ago   = (ts - timedelta(hours=1)).timestamp()

    xc_velocity_1h = sum(1 for h in history if h["client_id"] != client_id and h["ts"] >= one_hour_ago)

    other_amounts = [h["amount_usd"] for h in history if h["client_id"] != client_id]
    amount_corr   = 0.0
    if other_amounts:
        mean_other  = sum(other_amounts) / len(other_amounts)
        amount_corr = max(0.0, min(1.0, 1.0 - abs(amount_usd - mean_other) / max(mean_other, 1.0)))
        amount_corr = round(amount_corr, 4)

    other_ts             = [h["ts"] for h in history if h["client_id"] != client_id]
    time_since_other_sec = round(ts.timestamp() - max(other_ts), 1) if other_ts else None

    has_fiat   = any(h["modality"] == "fiat"   for h in history)
    has_crypto = any(h["modality"] == "crypto" for h in history)
    prior_risk = any(h.get("risk_score", 0.0) >= 0.70 and h["client_id"] != client_id for h in history)

    feats = {
        "institution_count":               n_institutions,
        "cross_client_velocity_1h":        xc_velocity_1h,
        "cross_client_amount_correlation": amount_corr,
        "time_since_other_institution_sec": time_since_other_sec,
        "has_fiat_history":                int(has_fiat),
        "has_crypto_history":              int(has_crypto),
        "cross_modal_pattern_detected":    int(has_fiat and has_crypto),
        "prior_high_risk_event":           int(prior_risk),
    }

    history.append({"client_id": client_id, "ts": ts.timestamp(),
                    "amount_usd": amount_usd, "modality": modality, "risk_score": risk_score})
    r.set(key, json.dumps(history), ex=FEATURE_TTL_SECONDS)
    return feats

# ─────────────────────────────────────────────────────────────────────────────
# GNN EMBEDDING

def get_gnn_embedding(r, identity_hash):
    raw = r.get(f"{REDIS_GNN_PREFIX}{identity_hash}")
    if raw:
        try:
            emb = json.loads(raw)
            if isinstance(emb, list) and len(emb) == GNN_EMB_DIM:
                return emb
        except Exception:
            pass
    return [0.0] * GNN_EMB_DIM

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL

def compute_temporal_features(ts):
    return {
        "hour_of_day":      ts.hour,
        "day_of_week":      ts.weekday(),
        "is_weekend":       int(ts.weekday() >= 5),
        "is_unusual_hour":  int(ts.hour < 6 or ts.hour >= 23),
        "is_business_hour": int(9 <= ts.hour <= 17 and ts.weekday() < 5),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE COMPUTATION

def compute_features(txn: dict) -> dict:
    """Assembles all 137 features. Column names match models/feature_names.txt."""
    r = get_redis()

    identity_hash = (txn.get("identity_hash") or txn.get("sender_hash")
                     or txn.get("user_email_hash") or txn.get("sender_wallet_hash") or "")
    if not identity_hash:
        log.warning("Missing identity_hash: %s", txn.get("transaction_id", "?"))
        txn["feature_error"] = "missing_identity_hash"
        return txn

    ts_raw = txn.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)

    amount_usd = float(txn.get("amount_usd", 0) or 0)
    client_id  = txn.get("client_id", "unknown")
    modality   = txn.get("modality", "fiat")

    vel_feats                    = update_velocity(r, identity_hash, ts, amount_usd)
    baseline_rate, baseline_amt  = get_velocity_baseline(r, identity_hash, ts)
    velocity_ratio               = round(vel_feats.get("count_24h", 0) / max(baseline_rate, 0.1), 3)
    amount_ratio                 = round(amount_usd / max(baseline_amt, 1.0), 3)
    xc_feats                     = update_cross_client(r, identity_hash, client_id, ts, amount_usd, modality)
    temp_feats                   = compute_temporal_features(ts)
    gnn_embedding                = get_gnn_embedding(r, identity_hash)
    gnn_feats                    = {f"gnn_emb_{i}": v for i, v in enumerate(gnn_embedding)}

    # Passthrough — fill with -1 (matches pipeline.py fillna(-1))
    passthrough = {
        "amount_usd":                float(txn.get("amount_usd", -1) or -1),
        "sender_account_age_days":   txn.get("sender_account_age_days", -1),
        "sender_kyc_status":         txn.get("sender_kyc_status", "unknown"),
        "sender_archetype":          txn.get("sender_archetype", "unknown"),
        "is_first_time_receiver":    int(txn.get("is_first_time_receiver") or 0),
        "balance_drain_ratio":       float(txn.get("balance_drain_ratio", -1) or -1),
        "sender_balance_before":     float(txn.get("sender_balance_before", -1) or -1),
        "modality":                  modality,
        "transfer_type":             txn.get("transfer_type", "unknown"),
        "transfer_network":          txn.get("transfer_network", "unknown"),
        "client_id":                 client_id,
        "known_mixer_interaction":   int(txn.get("known_mixer_interaction") or 0),
        "receiver_wallet_age_days":  txn.get("receiver_wallet_age_days", -1),
        "receiver_wallet_type":      txn.get("receiver_wallet_type", "unknown"),
        "cryptocurrency":            txn.get("cryptocurrency", ""),
        "blockchain":                txn.get("blockchain", ""),
        "gas_price_gwei":            txn.get("gas_price_gwei", -1),
        "source_confidence":         float(txn.get("source_confidence", 0.5) or 0.5),
        "is_real_data":              int(txn.get("is_real_data", 0) or 0),
    }

    enriched = {
        **txn,
        **vel_feats,
        "velocity_ratio":              velocity_ratio,
        "amount_ratio":                amount_ratio,
        "velocity_baseline_daily":     baseline_rate,
        "institution_count":           xc_feats["institution_count"],
        "cross_client_velocity_1h":    xc_feats["cross_client_velocity_1h"],
        "cross_client_amount_correlation": xc_feats["cross_client_amount_correlation"],
        "time_since_other_institution_sec": xc_feats["time_since_other_institution_sec"],
        "has_fiat_history":            xc_feats["has_fiat_history"],
        "has_crypto_history":          xc_feats["has_crypto_history"],
        "cross_modal_pattern_detected": xc_feats["cross_modal_pattern_detected"],
        "prior_high_risk_event":       xc_feats["prior_high_risk_event"],
        **temp_feats,
        **passthrough,
        **gnn_feats,
        "features_computed_at": ts.isoformat(),
        "feature_version":      "v2",
    }

    # Cache in Redis
    try:
        cache = {k: v for k, v in enriched.items()
                 if not isinstance(v, (dict, list)) and k not in
                 ("sender_hash", "receiver_hash", "identity_hash", "transaction_hash")}
        r.set(f"{REDIS_FEATURES_PREFIX}{identity_hash}",
              json.dumps(cache, default=str), ex=FEATURE_TTL_SECONDS)
    except Exception:
        pass

    return enriched

