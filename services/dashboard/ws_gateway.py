"""
services/dashboard/ws_gateway.py
==================================
WebSocket gateway — streams live scored transactions to the Next.js dashboard.

This FastAPI service exposes a WebSocket endpoint (/ws/dashboard) that:
  1. Subscribes to Redis Pub/Sub channel "sentinel:scored" (published by serve.py)
  2. Pushes scored transaction events to all connected browser clients in real-time
  3. Also exposes REST endpoints that the dashboard queries for historical charts

Why a separate WebSocket service (not inside ingestion/main.py):
  - WebSocket clients are long-lived connections. Keeping them in the ingestion
    service would cause resource contention with high-throughput transaction POST.
  - This service is stateless and can be horizontally scaled independently.
  - serve.py publishes to Redis pub/sub — any number of WS gateway replicas
    can subscribe and fan-out to their connected clients.

Integration with serve.py:
  serve.py adds one line after scoring:
    r.publish("sentinel:scored", json.dumps(result_with_txn))
  This gateway receives it instantly and pushes to connected browsers.

Dashboard events pushed via WebSocket:
  { type: "transaction_scored", data: { ...result } }
  { type: "stats_update", data: { block_rate, review_rate, avg_score, txn_count } }
  { type: "alert", data: { ...high_risk_result } }

REST endpoints (for dashboard charts):
  GET /api/stats/live      — last 5 minutes metrics
  GET /api/stats/hourly    — last 24 hours aggregated
  GET /api/decisions       — recent decisions list (last 100)
  GET /api/fraud/timeline  — fraud rate timeline for chart

Usage:
  uvicorn services.dashboard.ws_gateway:app --host 0.0.0.0 --port 8003 --reload
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ws_gateway] %(levelname)s %(message)s",
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages all active WebSocket connections."""

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        log.info("WS client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)
        log.info("WS client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, message: dict):
        """Send to all connected clients. Remove stale connections."""
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        dead    = set()
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATS ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class LiveStats:
    """
    In-memory rolling stats for the last 5 minutes.
    Pushed to all WS clients every 5 seconds as a stats_update event.
    """

    def __init__(self, window_minutes: int = 5):
        self._window = timedelta(minutes=window_minutes)
        self._events: list[dict] = []  # {ts, decision, risk_score, amount, client_id}

    def record(self, result: dict):
        self._events.append({
            "ts":         time.time(),
            "decision":   result.get("decision", ""),
            "risk_score": result.get("risk_score", 0),
            "amount":     result.get("amount_usd", result.get("_amount_usd", 0)),
            "client_id":  result.get("client_id", result.get("_client_id", "")),
            "latency_ms": result.get("latency_ms", 0),
        })
        # Prune old events
        cutoff = time.time() - self._window.total_seconds()
        self._events = [e for e in self._events if e["ts"] >= cutoff]

    def snapshot(self) -> dict:
        if not self._events:
            return {
                "txn_count": 0, "block_count": 0, "review_count": 0,
                "approve_count": 0, "block_rate": 0, "review_rate": 0,
                "avg_risk_score": 0, "avg_latency_ms": 0,
                "total_volume_usd": 0,
                "by_client": {},
                "window_minutes": self._window.total_seconds() / 60,
            }

        n       = len(self._events)
        blocks  = sum(1 for e in self._events if e["decision"] == "block")
        reviews = sum(1 for e in self._events if e["decision"] == "review")

        by_client: dict[str, dict] = {}
        for e in self._events:
            cid = e.get("client_id", "unknown")
            if cid not in by_client:
                by_client[cid] = {"count": 0, "block_count": 0}
            by_client[cid]["count"] += 1
            if e["decision"] == "block":
                by_client[cid]["block_count"] += 1

        return {
            "txn_count":       n,
            "block_count":     blocks,
            "review_count":    reviews,
            "approve_count":   n - blocks - reviews,
            "block_rate":      round(blocks / n, 4),
            "review_rate":     round(reviews / n, 4),
            "avg_risk_score":  round(sum(e["risk_score"] for e in self._events) / n, 4),
            "avg_latency_ms":  round(sum(e["latency_ms"] for e in self._events) / n, 2),
            "total_volume_usd": round(sum(e["amount"] for e in self._events), 2),
            "by_client":       by_client,
            "window_minutes":  self._window.total_seconds() / 60,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }


live_stats = LiveStats()
recent_decisions: list[dict] = []   # last 100 scored transactions for dashboard table
MAX_RECENT = 100


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel WebSocket Gateway",
    description="Real-time event stream for the Sentinel dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Startup / shutdown ─────────────────────────────────────────────────────

_redis_subscriber: Optional[aioredis.Redis] = None
_subscriber_task: Optional[asyncio.Task]    = None
_stats_task:      Optional[asyncio.Task]    = None


@app.on_event("startup")
async def startup():
    global _redis_subscriber, _subscriber_task, _stats_task

    log.info("WS Gateway starting...")
    _redis_subscriber = aioredis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )

    # Background task: subscribe to Redis pub/sub and broadcast to WS clients
    _subscriber_task = asyncio.create_task(_redis_listener())
    # Background task: push rolling stats every 5 seconds
    _stats_task      = asyncio.create_task(_stats_broadcaster())

    log.info("  Redis: %s:%s", REDIS_HOST, REDIS_PORT)
    log.info("WS Gateway ready on port 8003")


@app.on_event("shutdown")
async def shutdown():
    if _subscriber_task:
        _subscriber_task.cancel()
    if _stats_task:
        _stats_task.cancel()
    if _redis_subscriber:
        await _redis_subscriber.close()


# ── Background tasks ──────────────────────────────────────────────────────

async def _redis_listener():
    """
    Subscribe to 'sentinel:scored' Redis pub/sub channel.
    serve.py publishes here after every scored transaction.
    Broadcast each event to all connected WS clients.
    """
    global recent_decisions
    pubsub = _redis_subscriber.pubsub()
    await pubsub.subscribe("sentinel:scored")
    log.info("Redis pub/sub: subscribed to sentinel:scored")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            result = json.loads(message["data"])
        except json.JSONDecodeError:
            continue

        # Update in-memory stats
        live_stats.record(result)
        recent_decisions = ([result] + recent_decisions)[:MAX_RECENT]

        # Build WS event
        event = {"type": "transaction_scored", "data": result}
        await manager.broadcast(event)

        # Also fire an alert event for high-risk transactions
        if result.get("decision") in ("block", "review") and result.get("risk_score", 0) > 0.5:
            await manager.broadcast({"type": "alert", "data": result})


async def _stats_broadcaster():
    """Push rolling stats to all WS clients every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        if manager.count > 0:
            snapshot = live_stats.snapshot()
            await manager.broadcast({"type": "stats_update", "data": snapshot})


# ── WebSocket endpoint ────────────────────────────────────────────────────

@app.websocket("/ws/dashboard")
async def websocket_dashboard(ws: WebSocket):
    """
    Main WebSocket endpoint for the Next.js dashboard.

    Events pushed to client:
      { type: "transaction_scored", data: {...} }   — every scored transaction
      { type: "stats_update",       data: {...} }   — rolling stats every 5s
      { type: "alert",              data: {...} }   — BLOCK/high-REVIEW alerts
      { type: "connected",          data: {...} }   — welcome message on connect

    Client can send:
      { type: "ping" }  → server responds { type: "pong" }
    """
    await manager.connect(ws)
    try:
        # Send welcome + initial state
        await ws.send_text(json.dumps({
            "type": "connected",
            "data": {
                "message":     "Connected to Sentinel live feed",
                "stats":       live_stats.snapshot(),
                "recent":      recent_decisions[:10],
                "ws_clients":  manager.count,
            }
        }))

        # Keep connection alive, handle client messages
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await ws.send_text(json.dumps({
                        "type": "pong",
                        "ts":   time.time(),
                    }))
            except asyncio.TimeoutError:
                # Send keepalive ping to prevent proxy timeouts
                await ws.send_text(json.dumps({"type": "keepalive", "ts": time.time()}))
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)


# ── REST endpoints (for chart queries) ────────────────────────────────────

@app.get("/api/stats/live")
async def get_live_stats():
    """Rolling 5-minute stats for the dashboard header."""
    return live_stats.snapshot()


@app.get("/api/decisions")
async def get_recent_decisions(limit: int = 50):
    """Last N scored transactions for the decisions table."""
    return {
        "count":   min(limit, len(recent_decisions)),
        "results": recent_decisions[:limit],
    }


@app.get("/api/stats/hourly")
async def get_hourly_stats():
    """
    Hourly aggregated stats from ClickHouse (last 24 hours).
    Falls back to in-memory data if ClickHouse unavailable.
    """
    try:
        from clickhouse_driver import Client
        ch = Client(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
            database=os.getenv("CLICKHOUSE_DB", "sentinel"),
        )
        rows = ch.execute("""
            SELECT
                toStartOfHour(scored_at) AS hour,
                client_id,
                count() AS txn_count,
                countIf(decision='block') AS block_count,
                countIf(decision='review') AS review_count,
                avg(risk_score) AS avg_risk,
                sum(amount_usd) AS total_volume
            FROM sentinel.transactions
            WHERE scored_at >= now() - INTERVAL 24 HOUR
            GROUP BY hour, client_id
            ORDER BY hour DESC
        """)
        return {
            "source": "clickhouse",
            "rows": [
                {
                    "hour":         r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "client_id":    r[1],
                    "txn_count":    r[2],
                    "block_count":  r[3],
                    "review_count": r[4],
                    "avg_risk":     round(float(r[5]), 4),
                    "total_volume": round(float(r[6]), 2),
                }
                for r in rows
            ]
        }
    except Exception as e:
        log.debug("ClickHouse query failed: %s — falling back to in-memory", e)
        # Fallback: aggregate in-memory recent_decisions
        buckets: dict[str, dict] = {}
        for d in recent_decisions:
            hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00")
            cid      = d.get("_client_id", d.get("client_id", "unknown"))
            key      = f"{hour_key}_{cid}"
            if key not in buckets:
                buckets[key] = {"hour": hour_key, "client_id": cid,
                                "txn_count": 0, "block_count": 0,
                                "review_count": 0, "score_sum": 0.0, "vol_sum": 0.0}
            b = buckets[key]
            b["txn_count"]   += 1
            b["score_sum"]   += d.get("risk_score", 0)
            b["vol_sum"]     += d.get("_amount_usd", d.get("amount_usd", 0))
            if d.get("decision") == "block":
                b["block_count"] += 1
            elif d.get("decision") == "review":
                b["review_count"] += 1

        return {
            "source": "memory",
            "rows": [
                {**b, "avg_risk": round(b["score_sum"] / max(b["txn_count"], 1), 4),
                 "total_volume": round(b["vol_sum"], 2)}
                for b in buckets.values()
            ]
        }


@app.get("/health")
async def health():
    redis_ok = False
    try:
        await _redis_subscriber.ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status":     "ok" if redis_ok else "degraded",
        "redis":      redis_ok,
        "ws_clients": manager.count,
        "recent_txns": len(recent_decisions),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.dashboard.ws_gateway:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info",
    )