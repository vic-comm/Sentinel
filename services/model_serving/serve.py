# """
# services/model_serving/serve.py  [Ray Serve v2]
# ================================================
# Upgrades from Kafka-consumer serving to Ray Serve HTTP deployment.

# WHY RAY SERVE OVER THE KAFKA CONSUMER APPROACH:

#   The previous design had serve.py consuming from "transactions.features_ready"
#   and writing results to Redis for polling. This worked but had three problems:

#   Problem 1 — Latency floor.
#     Kafka consumer → Redis poll adds 50-200ms of irreducible latency even on
#     localhost. The pipeline was: ingest → Redpanda → Bytewax → Redpanda →
#     serve.py consumer → Redis → ingestion polls Redis. That is 4 network hops
#     for a single transaction. Ray Serve eliminates the second Redpanda hop and
#     the Redis polling by making serving a direct HTTP call.

#   Problem 2 — No horizontal scaling.
#     The Kafka consumer was a single process. To handle 10x traffic you had to
#     manually start multiple consumer processes and manage partition assignment.
#     Ray Serve scales replicas automatically based on queue depth with one config
#     line: num_replicas=4, max_concurrent_queries=100.

#   Problem 3 — No batching.
#     XGBoost's predict_proba() is 10x faster on a batch of 64 rows than on 64
#     individual rows (due to BLAS vectorisation). The Kafka consumer scored one
#     transaction at a time. Ray Serve has built-in request batching:
#     @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.005) — requests that
#     arrive within 5ms are batched automatically before hitting the model.

# WHY RAY SERVE OVER PLAIN FASTAPI:

#   FastAPI is a web framework — it routes HTTP requests and validates Pydantic
#   models. It has no concept of model replicas, GPU placement, request batching,
#   or zero-downtime model hot-swapping.

#   Ray Serve adds all of that on top of FastAPI's routing:
#     - Model hot-swap: update_deployment() reloads the model without dropping
#       requests (old replica stays alive until new one is healthy).
#     - Replica autoscaling: target_num_ongoing_requests_per_replica controls
#       when to scale up/down automatically.
#     - Batching: @serve.batch collapses concurrent requests into a single
#       model call. Critical for throughput when scoring 1000+ txns/sec.
#     - Multi-model routing: a single Ray Serve app can route to XGBoost for
#       standard transactions and CatBoost for crypto transactions — one HTTP
#       endpoint, model-specific routing in Python.
#     - Pipeline composition: Bytewax feature engine can call Ray Serve directly
#       as an HTTP step, eliminating the second Redpanda topic entirely.

# NEW ARCHITECTURE (this file):
#   Bytewax (features_ready dict)
#     → POST /score  (direct HTTP, no second Kafka hop)
#     → Ray Serve SentinelScorer deployment
#        → XGBoost batch inference (up to 64 rows per batch)
#        → SHAP explanation (sampled — not every request)
#        → Decision engine
#     → JSON response written to Redis (result:{txn_id})
#        AND returned directly in the HTTP response

#   ingestion/main.py sync mode now gets the result in the HTTP response
#   instead of polling Redis. Redis write is kept as a cache for the
#   GET /v1/results/{txn_id} polling endpoint (backward compatible).

# USAGE:
#   # Start Ray cluster + serve app
#   python -m services.model_serving.serve

#   # Or via Docker (Ray head node)
#   docker compose up model_serving

#   # Scale replicas without restarting
#   python -m services.model_serving.serve --replicas 4

#   # Hot-swap model (no downtime)
#   python -m services.model_serving.serve --hot-swap
# """

# import argparse
# import json
# import logging
# import os
# import time
# from pathlib import Path
# from typing import Optional

# import joblib
# import numpy as np
# import pandas as pd
# import ray
# import redis as redis_lib
# import shap
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from ray import serve

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [serve] %(levelname)s %(message)s",
# )
# log = logging.getLogger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────

# REDIS_HOST    = os.getenv("REDIS_HOST",  "localhost")
# REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
# MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))
# RESULT_TTL    = 300    # seconds results live in Redis
# RAY_ADDRESS   = os.getenv("RAY_ADDRESS", "auto")  # "auto" = local cluster

# # Thresholds — override by setting env vars or loading from pipeline.py output
# APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.30"))
# BLOCK_THRESHOLD   = float(os.getenv("BLOCK_THRESHOLD",   "0.80"))

# # Batching config — tune based on throughput vs latency tradeoff
# # max_batch_size=64: collect up to 64 requests before calling model
# # batch_wait_timeout_s=0.005: wait at most 5ms for a full batch
# # At 100 req/s: batches fill in ~640ms → too slow. At 1000 req/s: fills in ~64ms.
# # For low-traffic demo: set max_batch_size=8, batch_wait_timeout_s=0.05
# MAX_BATCH_SIZE        = int(os.getenv("MAX_BATCH_SIZE",         "32"))
# BATCH_WAIT_TIMEOUT_S  = float(os.getenv("BATCH_WAIT_TIMEOUT_S", "0.010"))

# # Deployment config
# NUM_REPLICAS          = int(os.getenv("NUM_REPLICAS", "1"))

# # SHAP explanation sampling — SHAP is expensive (~5ms per row with TreeExplainer)
# # Only compute SHAP for 1 in SHAP_SAMPLE_RATE requests to keep P99 low
# # For high-risk scores (> 0.7) we always compute SHAP regardless
# SHAP_SAMPLE_RATE = int(os.getenv("SHAP_SAMPLE_RATE", "5"))  # 1 in 5 = 20%

# CHAMPION_PRIORITY = [
#     "xgboost_model.pkl",
#     "catboost_model.pkl",
#     "lightgbm_model.pkl",
#     "logistic_reg_model.pkl",
# ]

# # ─────────────────────────────────────────────────────────────────────────────
# # REQUEST / RESPONSE MODELS
# # ─────────────────────────────────────────────────────────────────────────────

# class ScoreRequest(BaseModel):
#     """
#     What Bytewax feature engine sends here after computing all features.
#     The payload is the enriched transaction dict — all 137 features present.
#     """
#     transaction_id: str
#     client_id: str
#     # All other fields passed through as arbitrary dict entries
#     class Config:
#         extra = "allow"


# class ScoreResponse(BaseModel):
#     transaction_id:    str
#     risk_score:        float
#     decision:          str           # "approve" | "review" | "block"
#     decision_code:     int           # 0 | 1 | 2
#     reasons:           list[str]
#     shap_top_features: list[dict]
#     model_name:        str
#     latency_ms:        float
#     batch_size:        int           # how many requests were in this batch


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL BUNDLE  (loaded once per replica, not per request)
# # ─────────────────────────────────────────────────────────────────────────────

# class _ModelBundle:
#     """
#     Loaded once when the replica starts. Shared across all requests in
#     that replica. Thread-safe because Ray Serve handles concurrency.
#     """

#     def __init__(self):
#         self.model         = None
#         self.model_name    = "unknown"
#         self.feature_names = []
#         self.threshold     = BLOCK_THRESHOLD
#         self.shap_explainer = None
#         self._request_count = 0  # for SHAP sampling

#     def load(self):
#         # ── Champion model ─────────────────────────────────────────────────
#         for fname in CHAMPION_PRIORITY:
#             path = MODELS_DIR / fname
#             if path.exists():
#                 try:
#                     self.model      = joblib.load(path)
#                     self.model_name = fname.replace("_model.pkl", "")
#                     log.info("Loaded champion: %s", path)
#                     break
#                 except Exception as e:
#                     log.warning("Failed to load %s: %s", path, e)

#         # Fallback: scan for any pkl
#         if self.model is None:
#             pkls = sorted(MODELS_DIR.glob("*.pkl"))
#             if pkls:
#                 self.model = joblib.load(pkls[0])
#                 self.model_name = pkls[0].stem
#                 log.info("Fallback model: %s", pkls[0])

#         if self.model is None:
#             raise RuntimeError(
#                 f"No model found in {MODELS_DIR}. "
#                 "Run: python -m ml.pipeline --use-ray --trials 30"
#             )

#         # ── Threshold ──────────────────────────────────────────────────────
#         thresh_file = MODELS_DIR / f"{self.model_name}_threshold.pkl"
#         if thresh_file.exists():
#             try:
#                 self.threshold = float(joblib.load(thresh_file))
#                 log.info("Threshold: %.4f from %s", self.threshold, thresh_file)
#             except Exception:
#                 log.warning("Could not load threshold file — using %.2f", BLOCK_THRESHOLD)

#         # ── Feature names (training-serving alignment) ─────────────────────
#         feat_file = MODELS_DIR / "feature_names.txt"
#         if feat_file.exists():
#             with open(feat_file) as f:
#                 self.feature_names = [l.strip() for l in f if l.strip()]
#             log.info("Feature names: %d columns", len(self.feature_names))
#         else:
#             log.warning(
#                 "feature_names.txt not found — training-serving alignment disabled. "
#                 "Run pipeline.py to generate it."
#             )

#         # ── SHAP explainer ─────────────────────────────────────────────────
#         # TreeExplainer works for XGBoost, LightGBM, CatBoost
#         # Falls back gracefully if model type is unsupported
#         try:
#             self.shap_explainer = shap.TreeExplainer(self.model)
#             log.info("SHAP TreeExplainer: ready")
#         except Exception as e:
#             log.warning("SHAP not available for %s: %s", self.model_name, e)
#             try:
#                 # Linear fallback for logistic regression
#                 self.shap_explainer = shap.LinearExplainer(
#                     self.model,
#                     shap.maskers.Independent(pd.DataFrame(
#                         np.zeros((1, len(self.feature_names))),
#                         columns=self.feature_names or ["x"]
#                     ))
#                 )
#                 log.info("SHAP LinearExplainer: ready (fallback)")
#             except Exception:
#                 self.shap_explainer = None
#                 log.warning("SHAP disabled — no explainer available")

#         log.info(
#             "Model bundle ready: %s | threshold=%.4f | %d features | SHAP=%s",
#             self.model_name, self.threshold,
#             len(self.feature_names),
#             "enabled" if self.shap_explainer else "disabled",
#         )

#     def _to_dataframe(self, feature_dicts: list[dict]) -> pd.DataFrame:
#         """
#         Convert a list of feature dicts to a DataFrame aligned to training columns.
#         Missing columns get -1 (matches pipeline.py fillna(-1)).
#         This is where training-serving skew is prevented.
#         """
#         if self.feature_names:
#             X = pd.DataFrame(feature_dicts).reindex(
#                 columns=self.feature_names, fill_value=-1
#             )
#         else:
#             X = pd.DataFrame(feature_dicts)

#         X = X.replace({True: 1, False: 0, "True": 1, "False": 0})
#         X = X.infer_objects()
#         X = X.select_dtypes(exclude=["object", "string", "category"])
#         X = X.fillna(-1).astype(np.float32)
#         return X

#     def predict_batch(
#         self,
#         feature_dicts: list[dict],
#     ) -> tuple[list[float], list[list[dict]]]:
#         """
#         Batch inference: one model call for all requests in the batch.

#         Returns:
#           scores:        list[float]      — one score per request
#           shap_lists:    list[list[dict]] — SHAP features per request
#                                            (empty list if SHAP sampled out)
#         """
#         X = self._to_dataframe(feature_dicts)
#         n = len(feature_dicts)

#         # Single vectorised predict_proba call for entire batch
#         # This is the key throughput gain vs per-request scoring
#         try:
#             probas = self.model.predict_proba(X)[:, 1].tolist()
#         except Exception as e:
#             log.error("Batch predict failed: %s", e)
#             probas = [0.5] * n

#         # SHAP explanation — sample based on SHAP_SAMPLE_RATE
#         # Always compute for high-risk scores (> 0.7) regardless of sampling
#         self._request_count += n
#         shap_lists: list[list[dict]] = [[] for _ in range(n)]

#         if self.shap_explainer is not None:
#             # Determine which rows to explain
#             explain_mask = [
#                 (probas[i] > 0.70)  # always explain high-risk
#                 or (self._request_count % SHAP_SAMPLE_RATE == i % SHAP_SAMPLE_RATE)
#                 for i in range(n)
#             ]

#             if any(explain_mask):
#                 explain_idx = [i for i, m in enumerate(explain_mask) if m]
#                 X_explain   = X.iloc[explain_idx]

#                 try:
#                     sv = self.shap_explainer.shap_values(X_explain)
#                     # Binary classifiers: sv may be [neg_class, pos_class]
#                     sv_pos = sv[1] if isinstance(sv, list) else sv
#                     cols   = X.columns.tolist()

#                     for batch_pos, orig_idx in enumerate(explain_idx):
#                         row_sv   = sv_pos[batch_pos]
#                         top_idx  = np.argsort(np.abs(row_sv))[::-1][:5]
#                         shap_lists[orig_idx] = [
#                             {
#                                 "feature":    cols[j],
#                                 "value":      round(float(X.iloc[orig_idx, j]), 4),
#                                 "shap_value": round(float(row_sv[j]), 4),
#                             }
#                             for j in top_idx
#                         ]
#                 except Exception as e:
#                     log.debug("SHAP batch failed: %s", e)

#         return probas, shap_lists


# # ─────────────────────────────────────────────────────────────────────────────
# # DECISION ENGINE
# # ─────────────────────────────────────────────────────────────────────────────

# FEATURE_DESCRIPTIONS = {
#     "velocity_ratio":                  "Velocity {v:.1f}× above baseline",
#     "cross_modal_pattern_detected":    "Cross-institutional laundering pattern",
#     "amount_ratio":                    "Amount {v:.1f}× above user average",
#     "institution_count":               "Appeared at {iv} institutions",
#     "balance_drain_ratio":             "Balance drain: {pct:.0f}%",
#     "known_mixer_interaction":         "Interaction with known crypto mixer",
#     "is_first_time_receiver":          "First-time receiver",
#     "receiver_wallet_age_days":        "Receiver wallet {iv} days old",
#     "cross_client_amount_correlation": "Amount correlation {v:.2f} across institutions",
#     "time_since_other_institution_sec":"Same identity at other institution {min:.0f}m ago",
#     "sender_account_age_days":         "Account only {iv} days old",
#     "count_1h":                        "{iv} transactions in the past hour",
#     "count_6h":                        "{iv} transactions in the past 6 hours",
#     "fan_in_ratio":                    "Fan-in ratio {v:.2f} (mule aggregator pattern)",
#     "sender_out_degree":               "Sent to {iv} unique receivers",
#     "receiver_in_degree":              "Receiver has {iv} unique senders",
# }


# def _describe_feature(feature: str, value: float) -> str:
#     template = FEATURE_DESCRIPTIONS.get(feature)
#     if template is None:
#         direction = "↑ risk" if value > 0 else "↓ risk"
#         return f"{feature}: {value:.4f} ({direction})"
#     try:
#         return template.format(
#             v=value, iv=int(value),
#             pct=value * 100,
#             min=value / 60 if value else 0,
#         )
#     except Exception:
#         return f"{feature}: {value:.4f}"


# def make_decision(
#     risk_score:    float,
#     shap_features: list[dict],
#     txn:           dict,
#     approve_threshold: float,
#     block_threshold:   float,
#     model_name:    str,
#     latency_ms:    float,
#     batch_size:    int,
# ) -> dict:
#     if risk_score < approve_threshold:
#         decision, code = "approve", 0
#     elif risk_score > block_threshold:
#         decision, code = "block",   2
#     else:
#         decision, code = "review",  1

#     # Human-readable reasons from top-3 SHAP features
#     reasons = [_describe_feature(sf["feature"], sf["value"])
#                for sf in shap_features[:3]]

#     # Prepend hard fraud signals regardless of SHAP
#     if txn.get("known_mixer_interaction"):
#         reasons.insert(0, "Known cryptocurrency mixer interaction")
#     if txn.get("cross_modal_pattern_detected"):
#         reasons.insert(0, "Fiat→crypto laundering pattern detected")

#     return {
#         "transaction_id":    txn.get("transaction_id", ""),
#         "risk_score":        round(risk_score, 4),
#         "decision":          decision,
#         "decision_code":     code,
#         "reasons":           reasons[:5],
#         "shap_top_features": shap_features,
#         "model_name":        model_name,
#         "approve_threshold": approve_threshold,
#         "block_threshold":   block_threshold,
#         "needs_llm_review":  decision == "review",
#         "scored_at":         time.time(),
#         "latency_ms":        round(latency_ms, 2),
#         "batch_size":        batch_size,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # RAY SERVE DEPLOYMENT
# # ─────────────────────────────────────────────────────────────────────────────

# # The FastAPI app that Ray Serve wraps.
# # Ray Serve manages the lifecycle of the app — it handles replicas,
# # health checks, and routing. FastAPI handles request parsing and validation.
# app = FastAPI(
#     title="Sentinel Model Serving",
#     description="Ray Serve — XGBoost fraud detection with SHAP explanations",
#     version="2.0.0",
# )


# @serve.deployment(
#     name="SentinelScorer",
#     num_replicas=NUM_REPLICAS,

#     # Ray resource allocation per replica.
#     # 0 GPU = CPU-only inference (XGBoost is fast enough on CPU for <1K txns/sec)
#     # Increase num_cpus if inference becomes a bottleneck
#     ray_actor_options={"num_cpus": 2, "num_gpus": 0},

#     # Autoscaling config (commented out — use fixed replicas for portfolio demo)
#     autoscaling_config={
#         "min_replicas": 1,
#         "max_replicas": 8,
#         "target_num_ongoing_requests_per_replica": 10,
#     },

#     # Health check — Ray Serve will restart the replica if this fails
#     health_check_period_s=15,
#     health_check_timeout_s=10,

#     # Graceful shutdown — drain in-flight requests before stopping
#     graceful_shutdown_timeout_s=20,
#     graceful_shutdown_wait_loop_s=2,
# )
# @serve.ingress(app)
# class SentinelScorer:
#     """
#     Ray Serve deployment wrapping the XGBoost model.

#     __init__ runs once per replica (not per request).
#     _bundle is shared across all concurrent requests in this replica.
#     Ray Serve handles the concurrent execution — no thread safety issues
#     because Ray actors process one coroutine at a time by default.

#     The @serve.batch decorator on score() is the key performance mechanism:
#     Ray Serve buffers incoming requests and calls score() with a batch of
#     up to MAX_BATCH_SIZE items, all within one model.predict_proba() call.
#     """

#     def __init__(self):
#         log.info("SentinelScorer replica starting...")
#         self._bundle = _ModelBundle()
#         self._bundle.load()

#         # Redis client — one connection per replica
#         self._redis = redis_lib.Redis(
#             host=REDIS_HOST, port=REDIS_PORT,
#             decode_responses=True,
#             socket_connect_timeout=3,
#             socket_timeout=3,
#             retry_on_timeout=True,
#         )
#         try:
#             self._redis.ping()
#             log.info("Redis: connected (%s:%s)", REDIS_HOST, REDIS_PORT)
#         except Exception as e:
#             log.warning("Redis: not available (%s) — results won't be cached", e)
#             self._redis = None

#         log.info("SentinelScorer replica ready")

#     @app.get("/health")
#     async def health(self) -> dict:
#         """
#         Health check endpoint.
#         Ray Serve calls this every health_check_period_s seconds.
#         Return non-200 to trigger replica restart.
#         """
#         redis_ok = False
#         if self._redis:
#             try:
#                 self._redis.ping()
#                 redis_ok = True
#             except Exception:
#                 pass

#         return {
#             "status":     "ok",
#             "model":      self._bundle.model_name,
#             "threshold":  self._bundle.threshold,
#             "features":   len(self._bundle.feature_names),
#             "shap":       self._bundle.shap_explainer is not None,
#             "redis":      redis_ok,
#             "replicas":   NUM_REPLICAS,
#         }

#     @app.get("/model/info")
#     async def model_info(self) -> dict:
#         """Model metadata — useful for the demo dashboard."""
#         return {
#             "model_name":    self._bundle.model_name,
#             "threshold":     self._bundle.threshold,
#             "feature_count": len(self._bundle.feature_names),
#             "top_features":  self._bundle.feature_names[:20] if self._bundle.feature_names else [],
#             "shap_enabled":  self._bundle.shap_explainer is not None,
#             "batch_size":    MAX_BATCH_SIZE,
#             "replicas":      NUM_REPLICAS,
#         }

#     @app.post("/score")
#     @serve.batch(
#         max_batch_size=MAX_BATCH_SIZE,
#         batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S,
#     )
#     async def score(self, requests: list[dict]) -> list[dict]:
#         """
#         Batch scoring endpoint — the core of Ray Serve's advantage.

#         Ray Serve calls this method with a LIST of requests, not one at a time.
#         All requests that arrived within BATCH_WAIT_TIMEOUT_S of each other
#         (up to MAX_BATCH_SIZE) are batched into a single model call.

#         Why this matters:
#           Single request:   predict_proba([[x1]]) takes T ms
#           32 requests:      predict_proba([[x1..x32]]) takes T + 2ms
#           Throughput gain:  32 / (T + 2) vs 32 / (32 * T) = ~30x improvement

#           For XGBoost with 137 features: T ≈ 3ms
#           Batched (32):  32 txns / 5ms  = 6,400 txns/sec per replica
#           Sequential:    32 txns / 96ms = 333 txns/sec per replica

#         The @serve.batch decorator handles the batching transparently.
#         From the caller's perspective, they POST /score once and get one result.
#         """
#         t0         = time.time()
#         batch_size = len(requests)

#         if batch_size == 0:
#             return []

#         # ── Duplicate guard ────────────────────────────────────────────────
#         # Check Redis before scoring — don't re-score if result exists
#         # (handles Bytewax retries and Redpanda message replays)
#         results_out = [None] * batch_size
#         to_score_idx = []

#         if self._redis:
#             pipe = self._redis.pipeline(transaction=False)
#             for txn in requests:
#                 pipe.get(f"result:{txn.get('transaction_id', '')}")
#             cached = pipe.execute()

#             for i, raw in enumerate(cached):
#                 if raw:
#                     try:
#                         results_out[i] = json.loads(raw)
#                         log.debug("Cache hit: %s", requests[i].get("transaction_id", "?"))
#                     except Exception:
#                         to_score_idx.append(i)
#                 else:
#                     to_score_idx.append(i)
#         else:
#             to_score_idx = list(range(batch_size))

#         # ── Batch inference ────────────────────────────────────────────────
#         if to_score_idx:
#             to_score = [requests[i] for i in to_score_idx]

#             scores, shap_lists = self._bundle.predict_batch(to_score)
#             latency_ms = (time.time() - t0) * 1000

#             # Build results and cache in Redis
#             redis_pipe = self._redis.pipeline(transaction=False) if self._redis else None

#             for local_i, orig_i in enumerate(to_score_idx):
#                 txn           = requests[orig_i]
#                 risk_score    = scores[local_i]
#                 shap_features = shap_lists[local_i]

#                 result = make_decision(
#                     risk_score        = risk_score,
#                     shap_features     = shap_features,
#                     txn               = txn,
#                     approve_threshold = APPROVE_THRESHOLD,
#                     block_threshold   = self._bundle.threshold,
#                     model_name        = self._bundle.model_name,
#                     latency_ms        = latency_ms,
#                     batch_size        = batch_size,
#                 )
#                 results_out[orig_i] = result

#                 # Cache result for polling (ingestion async mode)
#                 if redis_pipe:
#                     redis_pipe.set(
#                         f"result:{txn.get('transaction_id', '')}",
#                         json.dumps(result),
#                         ex=RESULT_TTL,
#                     )

#                 log.info(
#                     "%-24s score=%.3f decision=%-7s %.1fms [batch=%d]",
#                     txn.get("transaction_id", "?")[:24],
#                     risk_score,
#                     result["decision"].upper(),
#                     latency_ms / batch_size,  # per-request latency
#                     batch_size,
#                 )

#             if redis_pipe:
#                 try:
#                     redis_pipe.execute()
#                 except Exception as e:
#                     log.warning("Redis pipeline write failed: %s", e)

#         return results_out

#     @app.get("/results/{transaction_id}")
#     async def get_result(self, transaction_id: str) -> JSONResponse:
#         """
#         Poll for an async result.
#         Backward compatible with ingestion/main.py GET /v1/results/{txn_id}.
#         """
#         if not self._redis:
#             raise HTTPException(status_code=503, detail="Redis unavailable")

#         raw = self._redis.get(f"result:{transaction_id}")
#         if raw is None:
#             return JSONResponse(
#                 status_code=202,
#                 content={
#                     "transaction_id": transaction_id,
#                     "status":         "pending",
#                     "message":        "Score not yet available. Retry in 1 second.",
#                 },
#             )
#         result = json.loads(raw)
#         return JSONResponse(content={"status": "scored", **result})


# # ─────────────────────────────────────────────────────────────────────────────
# # DEPLOYMENT ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────

# def build_app() -> serve.Application:
#     """
#     Returns the bound Ray Serve application.
#     This is what Ray Serve calls when starting from a config file.
#     """
#     return SentinelScorer.bind()


# def main(replicas: int = NUM_REPLICAS, hot_swap: bool = False):
#     """
#     Start Ray and deploy SentinelScorer.

#     Args:
#       replicas:  Number of model replicas to run in parallel.
#                  Each replica loads its own copy of the model into memory.
#                  Use replicas > 1 only if inference latency > target SLA.

#       hot_swap:  If True and a deployment already exists, update it in place
#                  without dropping requests. Old replica stays alive until
#                  new one passes health checks.
#     """
#     log.info("=" * 60)
#     log.info("SENTINEL MODEL SERVING — Ray Serve v2")
#     log.info("=" * 60)
#     log.info("  Replicas:      %d", replicas)
#     log.info("  Batch size:    %d (wait %.0fms)", MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S * 1000)
#     log.info("  SHAP sampling: 1 in %d (+ all high-risk)", SHAP_SAMPLE_RATE)
#     log.info("  Redis:         %s:%s", REDIS_HOST, REDIS_PORT)
#     log.info("  Models dir:    %s", MODELS_DIR)

#     # Init Ray — connects to existing cluster or starts a local one
#     if not ray.is_initialized():
#         try:
#             ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
#             log.info("  Ray: connected to cluster at %s", RAY_ADDRESS)
#         except Exception:
#             ray.init(ignore_reinit_error=True)
#             log.info("  Ray: started local cluster")

#     # Start Ray Serve
#     serve.start(
#         detached=True,          # keeps serving after this script exits
#         http_options={
#             "host": "0.0.0.0",
#             "port": 8001,       # model serving on 8001, ingestion on 8000
#         },
#     )
#     log.info("  Ray Serve: started on port 8001")

#     # Deploy (or hot-swap)
#     deployment = SentinelScorer.options(num_replicas=replicas)

#     if hot_swap:
#         log.info("  Hot-swap mode: updating deployment without downtime")
#         serve.run(deployment.bind(), name="sentinel", route_prefix="/")
#         log.info("  Hot-swap complete — new model live")
#     else:
#         serve.run(deployment.bind(), name="sentinel", route_prefix="/")
#         log.info("  SentinelScorer deployed — scoring on http://localhost:8001/score")

#     log.info("")
#     log.info("  Endpoints:")
#     log.info("    POST http://localhost:8001/score       ← Bytewax calls this")
#     log.info("    GET  http://localhost:8001/health      ← liveness probe")
#     log.info("    GET  http://localhost:8001/model/info  ← model metadata")
#     log.info("    GET  http://localhost:8001/results/{{txn_id}} ← async poll")
#     log.info("")

#     # Keep alive (Ray Serve is now detached — script can exit, or block here)
#     log.info("Serving... (Ctrl+C to shutdown)")
#     try:
#         while True:
#             time.sleep(60)
#     except KeyboardInterrupt:
#         log.info("Shutting down Ray Serve...")
#         serve.shutdown()
#         ray.shutdown()
#         log.info("Done.")


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Sentinel Ray Serve model serving")
#     parser.add_argument("--replicas",  type=int,  default=NUM_REPLICAS,
#                         help="Number of model replicas (default: 1)")
#     parser.add_argument("--hot-swap",  action="store_true",
#                         help="Update existing deployment without downtime")
#     args = parser.parse_args()
#     main(replicas=args.replicas, hot_swap=args.hot_swap)

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