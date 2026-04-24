"""
services/feature_engine/main.py
================================
Feature engineering as a plain HTTP service — no Kafka, no Bytewax dataflow.

Replaces the Bytewax Kafka consumer pattern with a direct FastAPI endpoint.
The feature LOGIC (velocity windows, cross-modal detection, GNN lookup) is
identical to dataflow.py — only the I/O wrapper changes.

What changed from Bytewax dataflow:
  BEFORE: ingestion → Redpanda → Bytewax consumer → Redpanda → serve consumer
  AFTER:  ingestion → POST /compute → feature_engine → POST /score → Ray Serve

Why this is better for Sentinel:
  Bytewax's streaming primitives add value when you have a genuine unbounded
  event stream that needs windowed aggregation on the fly. For Sentinel, the
  windowed aggregation (count_1h, count_6h etc.) is already handled by Redis
  sorted sets — Bytewax was acting as a Kafka consumer with extra steps.

  Direct HTTP gives you:
  - ~8ms latency vs ~120ms with Kafka round-trips
  - No broker to operate
  - Full request tracing (single trace spans the whole pipeline)
  - Simpler error handling (HTTP status codes, not Kafka offset management)

The Redis-backed state (velocity windows, cross-client history, GNN embeddings)
is unchanged — that is the actual streaming state engine.

Usage:
  uvicorn services.feature_engine.main:app --host 0.0.0.0 --port 8002 --workers 2
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Import the feature computation logic — unchanged from dataflow.py
from services.feature_engine.logic import compute_features, get_redis, GNN_EMB_DIM

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [feature_engine] %(levelname)s %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RAY_SERVE_URL    = os.getenv("RAY_SERVE_URL",    "http://localhost:8001/score")
SCORE_TIMEOUT_S  = float(os.getenv("SCORE_TIMEOUT_S", "8.0"))

# httpx async client — shared across requests (connection pooling)
# limits: max 200 concurrent connections, 20 per host
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(SCORE_TIMEOUT_S),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=20),
        )
    return _http_client


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Feature Engine",
    description="Computes 137 features and forwards to Ray Serve for scoring",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    log.info("Feature engine starting...")
    try:
        get_redis().ping()
        log.info("  Redis: connected (GNN dim: %d)", GNN_EMB_DIM)
    except Exception as e:
        log.warning("  Redis: NOT connected (%s) — features will degrade", e)

    # Warm up httpx client
    get_http_client()
    log.info("  Ray Serve target: %s", RAY_SERVE_URL)
    log.info("Feature engine ready on port 8002")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


@app.get("/health")
async def health():
    redis_ok = False
    ray_ok   = False

    try:
        get_redis().ping()
        redis_ok = True
    except Exception:
        pass

    try:
        client = get_http_client()
        resp   = await client.get(RAY_SERVE_URL.replace("/score", "/health"), timeout=2.0)
        ray_ok = resp.status_code == 200
    except Exception:
        pass

    status = "ok" if (redis_ok and ray_ok) else "degraded"
    return {
        "status":    status,
        "redis":     redis_ok,
        "ray_serve": ray_ok,
        "gnn_dim":   GNN_EMB_DIM,
    }


@app.post("/compute")
async def compute_and_score(request: Request):
    """
    Main endpoint called by ingestion/main.py.

    Flow:
      1. Receive raw transaction dict
      2. Run compute_features() — velocity windows, cross-modal, GNN lookup
      3. POST enriched dict to Ray Serve /score
      4. Return the score result directly

    This is a synchronous pipeline — the caller gets the full score
    in the same HTTP response. No Redis polling needed.
    """
    t0 = time.perf_counter()

    try:
        txn = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    txn_id = txn.get("transaction_id", "?")

    # ── Step 1: Feature computation ─────────────────────────────────────────
    # compute_features() is CPU-bound + Redis I/O.
    # Run in thread pool to avoid blocking the event loop.
    try:
        loop     = asyncio.get_event_loop()
        enriched = await loop.run_in_executor(None, compute_features, txn)
    except Exception as e:
        log.error("Feature computation failed for %s: %s", txn_id, e)
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    feature_ms = (time.perf_counter() - t0) * 1000

    # ── Step 2: Score via Ray Serve ─────────────────────────────────────────
    try:
        client   = get_http_client()
        response = await client.post(RAY_SERVE_URL, json=enriched)
        response.raise_for_status()
        result   = response.json()
    except httpx.TimeoutException:
        log.error("Ray Serve timeout for %s after %.0fms", txn_id, SCORE_TIMEOUT_S * 1000)
        raise HTTPException(status_code=504, detail="Scoring timeout")
    except httpx.HTTPStatusError as e:
        log.error("Ray Serve returned %d for %s", e.response.status_code, txn_id)
        raise HTTPException(status_code=502, detail=f"Scoring service error: {e}")
    except Exception as e:
        log.error("Ray Serve call failed for %s: %s", txn_id, e)
        raise HTTPException(status_code=502, detail=f"Scoring service unavailable: {e}")

    total_ms = (time.perf_counter() - t0) * 1000

    # Augment result with pipeline timing breakdown
    if isinstance(result, dict):
        result["pipeline_ms"]  = round(total_ms, 2)
        result["feature_ms"]   = round(feature_ms, 2)
        result["scoring_ms"]   = round(total_ms - feature_ms, 2)

    log.info(
        "%-24s score=%.3f %-7s total=%.1fms (feat=%.1fms score=%.1fms)",
        txn_id[:24],
        result.get("risk_score", 0),
        result.get("decision", "?").upper(),
        total_ms, feature_ms, total_ms - feature_ms,
    )

    return result


@app.post("/compute/batch")
async def compute_and_score_batch(request: Request):
    """
    Batch endpoint — accepts a list of transactions.
    Computes features concurrently (asyncio.gather), then sends all to Ray Serve.
    Useful for replaying historical transactions or bulk testing.
    """
    t0 = time.perf_counter()

    try:
        transactions = await request.json()
        if not isinstance(transactions, list):
            raise ValueError("Expected a JSON array")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit: 500 transactions")

    # Compute features concurrently in thread pool
    loop = asyncio.get_event_loop()
    enriched_list = await asyncio.gather(*[
        loop.run_in_executor(None, compute_features, txn)
        for txn in transactions
    ])

    feature_ms = (time.perf_counter() - t0) * 1000

    # Send batch to Ray Serve
    # Ray Serve's @serve.batch will handle batching these requests together
    try:
        client = get_http_client()
        results = await asyncio.gather(*[
            client.post(RAY_SERVE_URL, json=enriched)
            for enriched in enriched_list
        ])
        scored = [r.json() for r in results if r.status_code == 200]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Batch scoring failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "Batch: %d transactions | total=%.1fms feat=%.1fms score=%.1fms",
        len(transactions), total_ms, feature_ms, total_ms - feature_ms,
    )

    return {
        "count":      len(scored),
        "results":    scored,
        "total_ms":   round(total_ms, 2),
        "feature_ms": round(feature_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.feature_engine.main:app",
        host="0.0.0.0",
        port=8002,
        workers=2,    # 2 workers = 2 concurrent feature computation threads
        log_level="info",
    )