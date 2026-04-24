# """
# services/ingestion/main.py
# ==========================
# FastAPI ingestion service — the front door of Sentinel.

# What it does:
#   1. Receives raw transaction JSON from NeoBank, CryptoEx, FinBridge
#   2. Validates schema (required fields, hash format, amount sanity)
#   3. Normalises (UTC timestamps, USD amounts, canonical transfer type)
#   4. Publishes to Redpanda topic "transactions.ingested"
#   5. Returns transaction_id immediately (async mode)
#       OR waits for a score from model_serving and returns risk score (sync mode)

# Redpanda integration:
#   - Producer uses kafka-python with acks="all" for durability
#   - Partition key = client_id (ensures all NeoBank events land on the same
#     partition in order, important for cross-client correlation)
#   - Topic "transactions.ingested" is consumed by:
#       * services/feature_engine/dataflow.py   (feature computation)
#       * services/graph_engine/neo4j_updater.py (graph update)

# Endpoints:
#   POST /v1/transactions/evaluate  → publish + optional sync score
#   GET  /v1/health                 → liveness check
#   GET  /v1/results/{txn_id}       → poll for async result (uses Redis)

# Usage:
#   uvicorn services.ingestion.main:app --host 0.0.0.0 --port 8000 --reload

#   Or via Docker:
#   docker compose up ingestion
# """

# import hashlib
# import json
# import logging
# import os
# import re
# import time
# from datetime import datetime, timezone
# from typing import Optional
# import asyncio
# import redis as redis_lib
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException, Request, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from kafka import KafkaProducer
# from pydantic import BaseModel, Field, field_validator, model_validator

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [ingestion] %(levelname)s %(message)s",
# )
# log = logging.getLogger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# REDPANDA_BROKERS = os.getenv("REDPANDA_BROKERS", "localhost:9092")
# REDIS_HOST       = os.getenv("REDIS_HOST",       "localhost")
# REDIS_PORT       = int(os.getenv("REDIS_PORT",   "6379"))

# INPUT_TOPIC      = "transactions.ingested"
# RESULT_TTL       = 300   # seconds — how long async results live in Redis
# SYNC_TIMEOUT     = 5.0   # seconds — max wait for sync scoring

# VALID_CLIENTS    = {"neobank_prod", "cryptoex_prod", "finbridge_prod"}
# VALID_MODALITIES = {"fiat", "crypto"}
# VALID_TRANSFERS  = {"ach", "wire", "p2p", "rtp", "bank_deposit",
#                     "withdrawal", "deposit", "transfer", "crypto_purchase",
#                     "contract_approval", "on_chain_transfer"}

# HASH_RE = re.compile(r"^[0-9a-f]{64}$")  # SHA-256 hex digest

# # ─────────────────────────────────────────────────────────────────────────────
# # PYDANTIC SCHEMA
# # ─────────────────────────────────────────────────────────────────────────────

# class TransactionInput(BaseModel):
#     # ── Required fields ───────────────────────────────────────────────────────
#     transaction_id: str = Field(..., min_length=5, max_length=128)
#     client_id:      str
#     timestamp:      str
#     modality:       str
#     amount_usd:     float = Field(..., gt=0, lt=50_000_000)

#     # ── Identity hashes (at least one required) ───────────────────────────────
#     # Fiat: identity_hash or sender_hash
#     # Crypto: user_email_hash or sender_wallet_hash
#     identity_hash:       Optional[str] = None
#     sender_hash:         Optional[str] = None
#     receiver_hash:       Optional[str] = None
#     user_email_hash:     Optional[str] = None
#     sender_wallet_hash:  Optional[str] = None
#     receiver_wallet_hash: Optional[str] = None

#     # ── Fiat fields ───────────────────────────────────────────────────────────
#     transfer_type:          Optional[str]  = None
#     transfer_network:       Optional[str]  = None
#     sender_account_age_days: Optional[int] = None
#     sender_kyc_status:      Optional[str]  = None
#     sender_archetype:       Optional[str]  = None
#     is_first_time_receiver: Optional[bool] = None
#     balance_drain_ratio:    Optional[float] = None
#     sender_balance_before:  Optional[float] = None

#     # ── Crypto fields ─────────────────────────────────────────────────────────
#     amount_crypto:              Optional[float] = None
#     cryptocurrency:             Optional[str]   = None
#     blockchain:                 Optional[str]   = None
#     transaction_hash:           Optional[str]   = None
#     known_mixer_interaction:    Optional[bool]  = None
#     receiver_wallet_age_days:   Optional[int]   = None
#     receiver_wallet_type:       Optional[str]   = None
#     gas_price_gwei:             Optional[int]   = None

#     # ── Cross-modal ───────────────────────────────────────────────────────────
#     cross_modality_fraud_id:    Optional[str]  = None
#     linked_fiat_transaction:    Optional[str]  = None

#     # ── Source metadata ───────────────────────────────────────────────────────
#     source:             Optional[str]   = None
#     source_confidence:  Optional[float] = None
#     is_real_data:       Optional[int]   = None

#     # ── Sync mode flag ────────────────────────────────────────────────────────
#     sync_score: bool = Field(
#         default=False,
#         description="If True, wait up to 5 seconds for a risk score before responding."
#     )

#     @field_validator("client_id")
#     @classmethod
#     def validate_client(cls, v):
#         if v not in VALID_CLIENTS:
#             raise ValueError(f"Unknown client_id '{v}'. Valid: {VALID_CLIENTS}")
#         return v

#     @field_validator("modality")
#     @classmethod
#     def validate_modality(cls, v):
#         if v not in VALID_MODALITIES:
#             raise ValueError(f"Unknown modality '{v}'. Valid: {VALID_MODALITIES}")
#         return v

#     @field_validator("identity_hash", "sender_hash", "receiver_hash",
#                      "user_email_hash", "sender_wallet_hash",
#                      "receiver_wallet_hash", mode="before")
#     @classmethod
#     def validate_hash(cls, v):
#         if v is None:
#             return v
#         if not HASH_RE.match(str(v)):
#             raise ValueError(
#                 f"Hash must be a 64-char hex string (SHA-256). Got: {str(v)[:16]}..."
#             )
#         return v

#     @field_validator("timestamp", mode="before")
#     @classmethod
#     def validate_timestamp(cls, v):
#         if not v:
#             raise ValueError("timestamp is required")
#         try:
#             ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
#             # Reject timestamps more than 30 minutes in the future
#             now = datetime.now(timezone.utc)
#             if ts.tzinfo is None:
#                 ts = ts.replace(tzinfo=timezone.utc)
#             if (ts - now).total_seconds() > 1800:
#                 raise ValueError("Timestamp is more than 30 minutes in the future")
#         except (ValueError, TypeError) as e:
#             raise ValueError(f"Invalid timestamp format: {v}. Use ISO 8601.")
#         return v

#     @model_validator(mode="after")
#     def require_identity(self):
#         """At least one identity hash must be present."""
#         has_identity = any([
#             self.identity_hash,
#             self.sender_hash,
#             self.user_email_hash,
#             self.sender_wallet_hash,
#         ])
#         if not has_identity:
#             raise ValueError(
#                 "At least one identity hash is required: "
#                 "identity_hash, sender_hash, user_email_hash, or sender_wallet_hash"
#             )
#         return self


# # ─────────────────────────────────────────────────────────────────────────────
# # KAFKA PRODUCER (lazy init)
# # ─────────────────────────────────────────────────────────────────────────────

# _producer: Optional[KafkaProducer] = None


# def get_producer() -> KafkaProducer:
#     global _producer
#     if _producer is None:
#         _producer = KafkaProducer(
#             bootstrap_servers=REDPANDA_BROKERS,
#             value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
#             key_serializer=lambda k: k.encode("utf-8") if k else None,
#             acks="all",             # wait for all replicas to confirm write
#             retries=3,
#             linger_ms=5,            # 5ms batching window — reduces round-trips
#             compression_type="snappy",
#         )
#         log.info("Kafka producer connected: %s", REDPANDA_BROKERS)
#     return _producer


# # ─────────────────────────────────────────────────────────────────────────────
# # REDIS CLIENT (for async result polling and cross-client detection)
# # ─────────────────────────────────────────────────────────────────────────────

# _redis_client: Optional[redis_lib.Redis] = None


# def get_redis() -> redis_lib.Redis:
#     global _redis_client
#     if _redis_client is None:
#         _redis_client = redis_lib.Redis(
#             host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
#         )
#     return _redis_client


# # ─────────────────────────────────────────────────────────────────────────────
# # NORMALISATION
# # ─────────────────────────────────────────────────────────────────────────────

# def normalise_transaction(txn: dict) -> dict:
#     """
#     Applies normalisation rules before publishing to Redpanda.

#     1. Timestamp → UTC ISO 8601 with Z suffix
#     2. transfer_type → lowercase canonical form
#     3. Resolve primary identity_hash (for cross-client lookup)
#     4. Flag missing optional fields with None (not absent)
#     """
#     # Normalise timestamp to UTC
#     ts_raw = txn.get("timestamp", "")
#     try:
#         ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
#         if ts.tzinfo is None:
#             ts = ts.replace(tzinfo=timezone.utc)
#         txn["timestamp"] = ts.isoformat().replace("+00:00", "Z")
#     except Exception:
#         txn["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

#     # Canonical transfer type
#     if txn.get("transfer_type"):
#         txn["transfer_type"] = txn["transfer_type"].lower().strip()

#     # Resolve canonical identity_hash — the cross-client key
#     # This is used by feature_engine to detect cross-modal fraud
#     canonical_hash = (
#         txn.get("identity_hash")
#         or txn.get("sender_hash")
#         or txn.get("user_email_hash")
#         or txn.get("sender_wallet_hash")
#         or ""
#     )
#     txn["_canonical_identity_hash"] = canonical_hash

#     # Enrich with ingestion metadata
#     txn["_ingested_at"] = datetime.now(timezone.utc).isoformat()
#     txn["_ingestion_version"] = "v1"

#     return txn


# def check_pii(txn: dict) -> list[str]:
#     """
#     Scans transaction fields for potential raw PII.
#     Returns list of suspicious field names (should be empty for valid requests).

#     Real PII patterns: email addresses, phone numbers, SSNs, card numbers.
#     All of these should have been hashed before sending to Sentinel.
#     """
#     email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
#     phone_re = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
#     ssn_re   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

#     violations = []
#     for key, val in txn.items():
#         if key.startswith("_") or not isinstance(val, str):
#             continue
#         if email_re.search(val):
#             violations.append(f"{key} (looks like email)")
#         if phone_re.search(val):
#             violations.append(f"{key} (looks like phone)")
#         if ssn_re.search(val):
#             violations.append(f"{key} (looks like SSN)")
#     return violations


# # ─────────────────────────────────────────────────────────────────────────────
# # FASTAPI APP
# # ─────────────────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title="Sentinel Ingestion API",
#     description="Real-time fraud detection — transaction ingestion endpoint",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["POST", "GET"],
#     allow_headers=["*"],
# )


# @app.on_event("startup")
# async def startup():
#     """Verify Redpanda and Redis are reachable on startup."""
#     log.info("Sentinel Ingestion API starting...")

#     # Test Kafka
#     try:
#         p = get_producer()
#         log.info("  Redpanda: connected (%s)", REDPANDA_BROKERS)
#     except Exception as e:
#         log.warning("  Redpanda: NOT connected (%s) — will retry on first request", e)

#     # Test Redis
#     try:
#         r = get_redis()
#         r.ping()
#         log.info("  Redis: connected (%s:%s)", REDIS_HOST, REDIS_PORT)
#     except Exception as e:
#         log.warning("  Redis: NOT connected (%s) — async results disabled", e)

#     log.info("Sentinel Ingestion API ready on port 8000")


# @app.on_event("shutdown")
# async def shutdown():
#     """Flush Kafka producer on graceful shutdown."""
#     global _producer
#     if _producer:
#         _producer.flush(timeout=10)
#         _producer.close()
#         log.info("Kafka producer flushed and closed")


# # ── Health check ──────────────────────────────────────────────────────────────

# @app.get("/v1/health")
# async def health():
#     """Liveness probe — returns 200 if the service is running."""
#     status_info = {
#         "status": "ok",
#         "service": "sentinel-ingestion",
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#     }

#     # Check Redpanda
#     try:
#         get_producer()
#         status_info["redpanda"] = "connected"
#     except Exception as e:
#         status_info["redpanda"] = f"error: {e}"
#         status_info["status"] = "degraded"

#     # Check Redis
#     try:
#         get_redis().ping()
#         status_info["redis"] = "connected"
#     except Exception as e:
#         status_info["redis"] = f"error: {e}"
#         status_info["status"] = "degraded"

#     return status_info


# # ── Main ingestion endpoint ───────────────────────────────────────────────────

# @app.post("/v1/transactions/evaluate", status_code=202)
# async def evaluate_transaction(txn_input: TransactionInput, request: Request):
#     """
#     Main ingestion endpoint.

#     Flow:
#       1. Validate (Pydantic schema already checked at this point)
#       2. PII check — reject if raw email/phone found
#       3. Normalise timestamp, transfer_type, canonical hash
#       4. Publish to Redpanda "transactions.ingested"
#       5a. Async mode (default): return 202 Accepted with transaction_id
#       5b. Sync mode (sync_score=True): poll Redis for up to 5 seconds for result

#     Partition strategy:
#       Key = client_id (NeoBank events always land on the same partition,
#       preserving ordering for velocity feature computation in Bytewax)
#     """
#     t0  = time.time()
#     txn = txn_input.model_dump(exclude={"sync_score"})

#     # PII guard — reject if raw PII detected
#     pii_violations = check_pii(txn)
#     if pii_violations:
#         log.warning(
#             "PII detected in transaction %s: %s",
#             txn.get("transaction_id"), pii_violations
#         )
#         raise HTTPException(
#             status_code=400,
#             detail={
#                 "error": "PII detected — hash all identifiers before sending to Sentinel",
#                 "fields": pii_violations,
#             },
#         )

#     # Normalise
#     txn = normalise_transaction(txn)

#     # Publish to Redpanda
#     try:
#         producer = get_producer()
#         future   = producer.send(
#             topic=INPUT_TOPIC,
#             key=txn["client_id"],    # partition key
#             value=txn,
#         )
#         # Block for ack — ensures durability before responding
#         record_metadata = future.get(timeout=5)
#         log.info(
#             "Published %s (client=%s, partition=%d, offset=%d, %.1fms)",
#             txn["transaction_id"],
#             txn["client_id"],
#             record_metadata.partition,
#             record_metadata.offset,
#             (time.time() - t0) * 1000,
#         )
#     except Exception as e:
#         log.error("Failed to publish transaction %s: %s", txn.get("transaction_id"), e)
#         raise HTTPException(
#             status_code=503,
#             detail=f"Failed to publish to Redpanda: {e}. Is Redpanda running?",
#         )

#     # Async mode (default): return 202 immediately
#     if not txn_input.sync_score:
#         return {
#             "transaction_id": txn["transaction_id"],
#             "status": "processing",
#             "message": "Transaction received. Score will be available via polling.",
#             "poll_url": f"/v1/results/{txn['transaction_id']}",
#             "latency_ms": round((time.time() - t0) * 1000, 1),
#         }

#     # Sync mode: poll Redis for result (written by model_serving/serve.py)
#     # The result key is set by decision_engine.py after model inference
#     result_key    = f"result:{txn['transaction_id']}"
#     poll_start    = time.time()
#     poll_interval = 0.05   # 50ms polling interval
#     result        = None

#     while (time.time() - poll_start) < SYNC_TIMEOUT:
#         try:
#             raw = get_redis().get(result_key)
#             if raw:
#                 result = json.loads(raw)
#                 break
#         except Exception:
#             pass
#         await asyncio.sleep(poll_interval)

#     if result is None:
#         # Timeout — return 202 with partial info
#         return JSONResponse(
#             status_code=202,
#             content={
#                 "transaction_id": txn["transaction_id"],
#                 "status": "scoring_timeout",
#                 "message": "Score not available within 5 seconds. Use poll_url.",
#                 "poll_url": f"/v1/results/{txn['transaction_id']}",
#                 "latency_ms": round((time.time() - t0) * 1000, 1),
#             },
#         )

#     return {
#         "transaction_id":   txn["transaction_id"],
#         "status":           "scored",
#         "risk_score":       result.get("risk_score"),
#         "decision":         result.get("decision"),
#         "reasons":          result.get("reasons", []),
#         "shap_top_features": result.get("shap_top_features", []),
#         "latency_ms":       round((time.time() - t0) * 1000, 1),
#     }


# # ── Async result polling ──────────────────────────────────────────────────────

# @app.get("/v1/results/{transaction_id}")
# async def get_result(transaction_id: str):
#     """
#     Poll for the result of an asynchronously scored transaction.

#     The result is written to Redis by decision_engine.py after model inference.
#     Key format: result:{transaction_id}
#     TTL: 300 seconds
#     """
#     result_key = f"result:{transaction_id}"
#     try:
#         raw = get_redis().get(result_key)
#     except Exception as e:
#         raise HTTPException(
#             status_code=503,
#             detail=f"Redis unavailable: {e}"
#         )

#     if raw is None:
#         return JSONResponse(
#             status_code=202,
#             content={
#                 "transaction_id": transaction_id,
#                 "status": "pending",
#                 "message": "Score not yet available. Retry in 1 second.",
#             },
#         )

#     result = json.loads(raw)
#     return {
#         "transaction_id":    transaction_id,
#         "status":            "scored",
#         "risk_score":        result.get("risk_score"),
#         "decision":          result.get("decision"),
#         "reasons":           result.get("reasons", []),
#         "shap_top_features": result.get("shap_top_features", []),
#     }


# # ── Demo endpoint — simulates a transaction for testing ──────────────────────

# @app.post("/v1/demo/simulate")
# async def simulate_transaction(scenario: str = "normal"):
#     """
#     Generates a synthetic transaction for demo purposes.
#     Scenarios: normal | ato | cross_modal | mixer

#     Useful for testing the pipeline end-to-end without a simulator running.
#     """
#     import hashlib, uuid

#     SALT = "sentinel_consortium_2026_v1"

#     def make_hash(email: str) -> str:
#         return hashlib.sha256(f"{email}{SALT}".encode()).hexdigest()

#     now = datetime.now(timezone.utc).isoformat()

#     scenarios = {
#         "normal": {
#             "transaction_id":         f"demo_normal_{uuid.uuid4().hex[:8]}",
#             "client_id":              "neobank_prod",
#             "timestamp":              now,
#             "modality":               "fiat",
#             "identity_hash":          make_hash("alice@example.com"),
#             "sender_hash":            make_hash("alice@example.com"),
#             "receiver_hash":          make_hash("landlord@example.com"),
#             "amount_usd":             1500.0,
#             "transfer_type":          "ach",
#             "transfer_network":       "domestic",
#             "sender_account_age_days": 730,
#             "sender_kyc_status":      "verified",
#             "sender_archetype":       "salary_worker",
#             "is_first_time_receiver": False,
#             "balance_drain_ratio":    0.12,
#         },
#         "ato": {
#             "transaction_id":         f"demo_ato_{uuid.uuid4().hex[:8]}",
#             "client_id":              "neobank_prod",
#             "timestamp":              now,
#             "modality":               "fiat",
#             "identity_hash":          make_hash("victim@example.com"),
#             "sender_hash":            make_hash("victim@example.com"),
#             "receiver_hash":          make_hash(f"mule_{uuid.uuid4()}@throwaway.com"),
#             "amount_usd":             9800.0,
#             "transfer_type":          "wire",
#             "transfer_network":       "international",
#             "sender_account_age_days": 400,
#             "sender_kyc_status":      "verified",
#             "sender_archetype":       "salary_worker",
#             "is_first_time_receiver": True,
#             "balance_drain_ratio":    0.82,
#         },
#         "mixer": {
#             "transaction_id":         f"demo_mixer_{uuid.uuid4().hex[:8]}",
#             "client_id":              "cryptoex_prod",
#             "timestamp":              now,
#             "modality":               "crypto",
#             "user_email_hash":        make_hash("launcher@darkweb.com"),
#             "sender_wallet_hash":     make_hash("wallet_source"),
#             "receiver_wallet_hash":   hashlib.sha256(
#                                           b"0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936"
#                                       ).hexdigest(),
#             "amount_usd":             45000.0,
#             "amount_crypto":          16.07,
#             "cryptocurrency":         "ETH",
#             "blockchain":             "ethereum",
#             "transaction_hash":       f"0x{uuid.uuid4().hex}",
#             "known_mixer_interaction": True,
#             "receiver_wallet_age_days": 9999,
#             "receiver_wallet_type":   "mixer",
#         },
#     }

#     txn_data = scenarios.get(scenario, scenarios["normal"])
#     log.info("Demo transaction generated: scenario=%s", scenario)
#     return {"scenario": scenario, "transaction": txn_data, "tip": "POST this to /v1/transactions/evaluate"}


# # ─────────────────────────────────────────────────────────────────────────────
# # RUN
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "services.ingestion.main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info",
#     )

"""
services/ingestion/main.py
==========================
FastAPI ingestion service — the front door of Sentinel.

Architecture Upgrades:
  - Removed Kafka/Redpanda completely.
  - Synchronous scoring: Direct HTTP POST to the Feature Engine.
  - Asynchronous scoring: Handled via FastAPI BackgroundTasks.
  - Graph Updates: Raw transactions are pushed to a Redis list (transactions:neo4j).
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import redis as redis_lib
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingestion] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST         = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT         = int(os.getenv("REDIS_PORT", "6379"))
FEATURE_ENGINE_URL = os.getenv("FEATURE_ENGINE_URL", "http://localhost:8002/compute")

NEO4J_QUEUE_KEY    = "transactions:neo4j"
RESULT_TTL         = 300

VALID_CLIENTS      = {"neobank_prod", "cryptoex_prod", "finbridge_prod"}
VALID_MODALITIES   = {"fiat", "crypto"}
HASH_RE            = re.compile(r"^[0-9a-f]{64}$")

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    transaction_id: str = Field(..., min_length=5, max_length=128)
    client_id:      str
    timestamp:      str
    modality:       str
    amount_usd:     float = Field(..., gt=0, lt=50_000_000)

    identity_hash:        Optional[str] = None
    sender_hash:          Optional[str] = None
    receiver_hash:        Optional[str] = None
    user_email_hash:      Optional[str] = None
    sender_wallet_hash:   Optional[str] = None
    receiver_wallet_hash: Optional[str] = None

    transfer_type:          Optional[str]   = None
    transfer_network:       Optional[str]   = None
    sender_account_age_days: Optional[int]  = None
    sender_kyc_status:      Optional[str]   = None
    sender_archetype:       Optional[str]   = None
    is_first_time_receiver: Optional[bool]  = None
    balance_drain_ratio:    Optional[float] = None
    sender_balance_before:  Optional[float] = None

    amount_crypto:            Optional[float] = None
    cryptocurrency:           Optional[str]   = None
    blockchain:               Optional[str]   = None
    transaction_hash:         Optional[str]   = None
    known_mixer_interaction:  Optional[bool]  = None
    receiver_wallet_age_days: Optional[int]   = None
    receiver_wallet_type:     Optional[str]   = None
    gas_price_gwei:           Optional[int]   = None

    cross_modality_fraud_id:  Optional[str] = None
    linked_fiat_transaction:  Optional[str] = None

    source:             Optional[str]   = None
    source_confidence:  Optional[float] = None
    is_real_data:       Optional[int]   = None

    sync_score: bool = Field(default=False)

    @field_validator("client_id")
    @classmethod
    def validate_client(cls, v):
        if v not in VALID_CLIENTS:
            raise ValueError(f"Unknown client_id '{v}'.")
        return v

    @field_validator("modality")
    @classmethod
    def validate_modality(cls, v):
        if v not in VALID_MODALITIES:
            raise ValueError(f"Unknown modality '{v}'.")
        return v

    @field_validator("identity_hash", "sender_hash", "receiver_hash",
                     "user_email_hash", "sender_wallet_hash",
                     "receiver_wallet_hash", mode="before")
    @classmethod
    def validate_hash(cls, v):
        if v is None: return v
        if not HASH_RE.match(str(v)):
            raise ValueError(f"Hash must be 64-char hex string (SHA-256).")
        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v):
        if not v: raise ValueError("timestamp is required")
        try:
            ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
            if (ts - now).total_seconds() > 1800:
                raise ValueError("Timestamp is >30m in the future")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid timestamp format.")
        return v

    @model_validator(mode="after")
    def require_identity(self):
        if not any([self.identity_hash, self.sender_hash, self.user_email_hash, self.sender_wallet_hash]):
            raise ValueError("At least one primary identity hash is required.")
        return self

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: Optional[redis_lib.Redis] = None
_http_client: Optional[httpx.AsyncClient] = None

def get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=50),
        )
    return _http_client

def normalise_transaction(txn: dict) -> dict:
    ts_raw = txn.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        txn["timestamp"] = ts.isoformat().replace("+00:00", "Z")
    except Exception:
        txn["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if txn.get("transfer_type"):
        txn["transfer_type"] = txn["transfer_type"].lower().strip()

    txn["_canonical_identity_hash"] = (
        txn.get("identity_hash") or txn.get("sender_hash") or 
        txn.get("user_email_hash") or txn.get("sender_wallet_hash") or ""
    )
    txn["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    return txn

def check_pii(txn: dict) -> list[str]:
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    phone_re = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
    ssn_re   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    
    violations = []
    for key, val in txn.items():
        if key.startswith("_") or not isinstance(val, str): continue
        if email_re.search(val): violations.append(key)
        if phone_re.search(val): violations.append(key)
        if ssn_re.search(val): violations.append(key)
    return violations

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Sentinel Ingestion API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    log.info("Ingestion API starting...")
    get_http_client()
    try:
        get_redis().ping()
        log.info("  Redis: connected")
    except Exception as e:
        log.warning("  Redis: NOT connected (%s)", e)

@app.on_event("shutdown")
async def shutdown():
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()

@app.get("/v1/health")
async def health():
    status_info = {"status": "ok", "service": "sentinel-ingestion"}
    try:
        get_redis().ping()
        status_info["redis"] = "connected"
    except Exception as e:
        status_info["redis"] = f"error: {e}"
        status_info["status"] = "degraded"
    return status_info

async def send_to_feature_engine_bg(txn: dict):
    """Background task for async scoring requests."""
    try:
        client = get_http_client()
        await client.post(FEATURE_ENGINE_URL, json=txn)
    except Exception as e:
        log.error("Async feature engine call failed for %s: %s", txn.get('transaction_id'), e)

@app.post("/v1/transactions/evaluate", status_code=202)
async def evaluate_transaction(txn_input: TransactionInput, background_tasks: BackgroundTasks):
    t0 = time.time()
    txn = txn_input.model_dump(exclude={"sync_score"})

    pii_violations = check_pii(txn)
    if pii_violations:
        raise HTTPException(status_code=400, detail="PII detected. Hash identifiers before sending.")

    txn = normalise_transaction(txn)

    # 1. Push to Redis for Graph Engine
    try:
        get_redis().rpush(NEO4J_QUEUE_KEY, json.dumps(txn))
    except Exception as e:
        log.error("Failed to push to Neo4j queue: %s", e)
        # We do not block the critical path for a graph queuing failure

    # 2. Async Mode: Return immediately, process in background
    if not txn_input.sync_score:
        background_tasks.add_task(send_to_feature_engine_bg, txn)
        return {
            "transaction_id": txn["transaction_id"],
            "status": "processing",
            "poll_url": f"/v1/results/{txn['transaction_id']}",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    # 3. Sync Mode: Await HTTP response from Feature Engine directly
    try:
        client = get_http_client()
        response = await client.post(FEATURE_ENGINE_URL, json=txn)
        response.raise_for_status()
        result = response.json()
        
        # Inject API latency
        result["ingestion_latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Feature Engine timeout")
    except Exception as e:
        log.error("Feature Engine failed: %s", e)
        raise HTTPException(status_code=502, detail="Feature Engine unavailable")

@app.get("/v1/results/{transaction_id}")
async def get_result(transaction_id: str):
    try:
        raw = get_redis().get(f"result:{transaction_id}")
    except Exception as e:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    if raw is None:
        return JSONResponse(status_code=202, content={"status": "pending"})

    return {"status": "scored", **json.loads(raw)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.ingestion.main:app", host="0.0.0.0", port=8000)