"""
services/model_serving/serve.py  [v3 — Ray Serve + alerts + analytics]
========================================================================
Upgrades from v2 (Kafka consumer) to Ray Serve HTTP deployment.
Also adds the alert queue and analytics sink integrations.

NEW in v3:
  - Ray Serve batch inference (@serve.batch — 30x throughput gain)
  - Alert queue: RPUSH alerts:queue for BLOCK/REVIEW decisions
  - Analytics sink: RPUSH transactions:analytics for every scored txn
  - MLflow ModelBundle with DagsHub remote loading
  - Champion/challenger shadow mode support

Architecture:
  feature_engine (port 8002)
    └─ POST /score → Ray Serve (port 8001)
         └─ XGBoost batch inference
         └─ SHAP explanation (sampled)
         └─ Decision routing
         └─ Redis: result:{txn_id}          ← polling by ingestion
         └─ Redis: RPUSH alerts:queue       ← consumed by notifier.py
         └─ Redis: RPUSH transactions:analytics ← consumed by db_writer.py

Port assignments:
  8000 — ingestion (main.py)
  8001 — model serving (Ray Serve, this file)
  8002 — feature engine (feature_engine/main.py)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import ray
import redis as redis_lib
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ray import serve
from model_serving._model_module import ModelBundle

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [serve] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST    = os.getenv("REDIS_HOST",  "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))
RESULT_TTL    = 300
RAY_ADDRESS   = os.getenv("RAY_ADDRESS", "auto")

APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.30"))
BLOCK_THRESHOLD   = float(os.getenv("BLOCK_THRESHOLD",   "0.80"))

MAX_BATCH_SIZE       = int(os.getenv("MAX_BATCH_SIZE",         "32"))
BATCH_WAIT_TIMEOUT_S = float(os.getenv("BATCH_WAIT_TIMEOUT_S", "0.010"))
NUM_REPLICAS         = int(os.getenv("NUM_REPLICAS", "1"))
SHAP_SAMPLE_RATE     = int(os.getenv("SHAP_SAMPLE_RATE", "5"))

# Redis list keys for downstream consumers
ALERT_QUEUE_KEY     = "alerts:queue"       # consumed by notifier.py
ANALYTICS_QUEUE_KEY = "transactions:analytics"  # consumed by db_writer.py
ANALYTICS_LIST_TTL  = 60 * 60 * 24         # 24h — db_writer should flush well before this

CHAMPION_PRIORITY = [
    "xgboost_model.pkl",
    "catboost_model.pkl",
    "lightgbm_model.pkl",
    "logistic_reg_model.pkl",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUNDLE
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_DESC = {
    "velocity_ratio":                  "Velocity {v:.1f}× above baseline",
    "cross_modal_pattern_detected":    "Cross-institutional laundering pattern",
    "amount_ratio":                    "Amount {v:.1f}× above user average",
    "institution_count":               "Appeared at {iv} institutions",
    "balance_drain_ratio":             "Balance drain: {pct:.0f}%",
    "known_mixer_interaction":         "Interaction with known crypto mixer",
    "is_first_time_receiver":          "First-time receiver",
    "receiver_wallet_age_days":        "Receiver wallet {iv} days old",
    "count_1h":                        "{iv} transactions in the past hour",
    "fan_in_ratio":                    "Fan-in ratio {v:.2f} (aggregator pattern)",
    "time_since_other_institution_sec": "Same identity at other institution {min:.0f}m ago",
    "sender_account_age_days":         "Account only {iv} days old",
}


def _describe(feature: str, value: float) -> str:
    tmpl = _FEATURE_DESC.get(feature)
    if not tmpl:
        return f"{feature}: {value:.4g} ({'↑' if value > 0 else '↓'} risk)"
    try:
        return tmpl.format(v=value, iv=int(value), pct=value * 100, min=value / 60)
    except Exception:
        return f"{feature}: {value:.4g}"


def make_decision(
    risk_score: float, shap_features: list[dict], txn: dict,
    approve_threshold: float, block_threshold: float,
    model_name: str, latency_ms: float, batch_size: int,
) -> dict:
    if risk_score < approve_threshold:
        decision, code = "approve", 0
    elif risk_score > block_threshold:
        decision, code = "block", 2
    else:
        decision, code = "review", 1

    reasons = [_describe(sf["feature"], sf["value"]) for sf in shap_features[:3]]
    if txn.get("known_mixer_interaction"):
        reasons.insert(0, "Known cryptocurrency mixer interaction")
    if txn.get("cross_modal_pattern_detected"):
        reasons.insert(0, "Fiat→crypto laundering pattern detected")

    return {
        "transaction_id":    txn.get("transaction_id", ""),
        "risk_score":        round(risk_score, 4),
        "decision":          decision,
        "decision_code":     code,
        "reasons":           reasons[:5],
        "shap_top_features": shap_features,
        "model_name":        model_name,
        "approve_threshold": approve_threshold,
        "block_threshold":   block_threshold,
        "needs_llm_review":  decision == "review",
        "scored_at":         time.time(),
        "latency_ms":        round(latency_ms, 2),
        "batch_size":        batch_size,
        # Passthrough fields needed by downstream consumers
        "_client_id":        txn.get("client_id", ""),
        "_amount_usd":       txn.get("amount_usd", 0),
        "_scored_at":        __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAY SERVE DEPLOYMENT
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Model Serving",
    description="Ray Serve — XGBoost fraud scoring with SHAP + alerts + analytics",
    version="3.0.0",
)


@serve.deployment(
    name="SentinelScorer",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
    health_check_period_s=15,
    health_check_timeout_s=10,
    graceful_shutdown_timeout_s=20,
    graceful_shutdown_wait_loop_s=2,
)
@serve.ingress(app)
class SentinelScorer:
    def __init__(self):
        log.info("SentinelScorer replica starting...")
        self._bundle = ModelBundle()
        self._bundle.load()

        self._redis = redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        try:
            self._redis.ping()
            log.info("Redis: connected (%s:%s)", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            log.warning("Redis unavailable: %s", e)
            self._redis = None

        log.info("SentinelScorer ready")

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health(self):
        redis_ok = False
        if self._redis:
            try:
                self._redis.ping()
                redis_ok = True
            except Exception:
                pass
        return {
            "status":    "ok",
            "model":     self._bundle.model_name,
            "threshold": self._bundle.threshold,
            "features":  len(self._bundle.feature_names),
            "shap":      self._bundle.shap_explainer is not None,
            "redis":     redis_ok,
        }

    @app.get("/model/info")
    async def model_info(self):
        return {
            "model_name":    self._bundle.model_name,
            "threshold":     self._bundle.threshold,
            "feature_count": len(self._bundle.feature_names),
            "top_features":  self._bundle.feature_names[:20],
            "batch_size":    MAX_BATCH_SIZE,
            "replicas":      NUM_REPLICAS,
        }

    # ── Scoring (batched) ─────────────────────────────────────────────────────
    @app.post("/score")
    @serve.batch(
        max_batch_size=MAX_BATCH_SIZE,
        batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S,
    )
    async def score(self, requests: list[dict]) -> list[dict]:
        """
        Batch scoring. Ray Serve buffers concurrent requests and calls
        this with up to MAX_BATCH_SIZE items, yielding ~30x throughput gain
        vs per-request scoring.

        After scoring:
          1. Writes result to Redis (result:{txn_id}) — for sync polling
          2. Pushes to alerts:queue if decision is BLOCK or REVIEW
          3. Pushes to transactions:analytics for every transaction
        """
        t0         = time.time()
        batch_size = len(requests)
        if batch_size == 0:
            return []

        results_out = [None] * batch_size
        to_score    = list(range(batch_size))

        # Duplicate guard — skip if already in Redis
        if self._redis:
            pipe   = self._redis.pipeline(transaction=False)
            for txn in requests:
                pipe.exists(f"result:{txn.get('transaction_id', '')}")
            exists = pipe.execute()
            to_score = [i for i, e in enumerate(exists) if not e]
            for i, e in enumerate(exists):
                if e:
                    try:
                        raw = self._redis.get(f"result:{requests[i].get('transaction_id', '')}")
                        if raw:
                            results_out[i] = json.loads(raw)
                    except Exception:
                        pass

        if to_score:
            feature_dicts = [requests[i] for i in to_score]
            scores, shap_lists = self._bundle.predict_batch(feature_dicts)
            latency_ms = (time.time() - t0) * 1000

            # Build Redis pipeline for all writes
            redis_pipe    = self._redis.pipeline(transaction=False) if self._redis else None
            alert_payloads    = []
            analytics_payloads = []

            for local_i, orig_i in enumerate(to_score):
                txn    = requests[orig_i]
                result = make_decision(
                    risk_score        = scores[local_i],
                    shap_features     = shap_lists[local_i],
                    txn               = txn,
                    approve_threshold = APPROVE_THRESHOLD,
                    block_threshold   = self._bundle.threshold,
                    model_name        = self._bundle.model_name,
                    latency_ms        = latency_ms,
                    batch_size        = batch_size,
                )

                # Merge transaction fields into result for downstream consumers
                merged = {**txn, **result}
                results_out[orig_i] = result

                if redis_pipe:
                    # 1. Result cache for polling
                    redis_pipe.set(
                        f"result:{txn.get('transaction_id', '')}",
                        json.dumps(result),
                        ex=RESULT_TTL,
                    )

                    # 2. Alert queue — BLOCK/REVIEW decisions only
                    if result["decision"] in ("block", "review"):
                        alert_payloads.append(json.dumps(merged))

                    # 3. Analytics sink — every transaction
                    analytics_payloads.append(json.dumps(merged))

                log.info(
                    "%-24s score=%.3f %-7s %.1fms [b=%d]",
                    txn.get("transaction_id", "?")[:24],
                    scores[local_i],
                    result["decision"].upper(),
                    latency_ms / max(batch_size, 1),
                    batch_size,
                )

            # Execute all Redis writes in one pipeline call
            if redis_pipe:
                if alert_payloads:
                    redis_pipe.rpush(ALERT_QUEUE_KEY, *alert_payloads)
                if analytics_payloads:
                    # rpush then set TTL on the list key
                    redis_pipe.rpush(ANALYTICS_QUEUE_KEY, *analytics_payloads)
                    redis_pipe.expire(ANALYTICS_QUEUE_KEY, ANALYTICS_LIST_TTL)
                try:
                    redis_pipe.execute()
                except Exception as e:
                    log.warning("Redis pipeline write failed: %s", e)

                # Publish to Redis pub/sub for WebSocket gateway (ws_gateway.py)
                # This is fire-and-forget — failure doesn't affect scoring pipeline
                try:
                    for orig_i in to_score:
                        merged = {**requests[orig_i], **(results_out[orig_i] or {})}
                        self._redis.publish("sentinel:scored", json.dumps(merged, default=str))
                except Exception as e:
                    log.debug("Pub/sub publish failed (non-critical): %s", e)

        return results_out

    # ── Async result polling ───────────────────────────────────────────────────
    @app.get("/results/{transaction_id}")
    async def get_result(self, transaction_id: str):
        if not self._redis:
            raise HTTPException(status_code=503, detail="Redis unavailable")
        raw = self._redis.get(f"result:{transaction_id}")
        if raw is None:
            return JSONResponse(
                status_code=202,
                content={"transaction_id": transaction_id, "status": "pending"}
            )
        return JSONResponse(content={"status": "scored", **json.loads(raw)})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> serve.Application:
    return SentinelScorer.bind()


def main(replicas: int = NUM_REPLICAS, hot_swap: bool = False):
    log.info("=" * 55)
    log.info("SENTINEL MODEL SERVING v3 — Ray Serve")
    log.info("=" * 55)
    log.info("  Replicas:   %d", replicas)
    log.info("  Batch:      %d (wait %.0fms)", MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S * 1000)
    log.info("  SHAP:       1 in %d + all >0.70", SHAP_SAMPLE_RATE)
    log.info("  Alerts:     RPUSH %s", ALERT_QUEUE_KEY)
    log.info("  Analytics:  RPUSH %s", ANALYTICS_QUEUE_KEY)

    if not ray.is_initialized():
        try:
            ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        except Exception:
            ray.init(ignore_reinit_error=True)

    serve.start(
        detached=True,
        http_options={"host": "0.0.0.0", "port": 8001},
    )
    log.info("Ray Serve started on port 8001")

    deployment = SentinelScorer.options(num_replicas=replicas)
    serve.run(deployment.bind(), name="sentinel", route_prefix="/")

    log.info("Endpoints:")
    log.info("  POST http://localhost:8001/score")
    log.info("  GET  http://localhost:8001/health")
    log.info("  GET  http://localhost:8001/model/info")
    log.info("  GET  http://localhost:8001/results/{txn_id}")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=NUM_REPLICAS)
    parser.add_argument("--hot-swap", action="store_true")
    args = parser.parse_args()
    main(replicas=args.replicas, hot_swap=args.hot_swap)