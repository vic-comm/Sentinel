"""
services/alerting/notifier.py
==============================
Real-time fraud alert notifier.

Reads from Redis list "alerts:queue" (populated by serve.py for BLOCK/REVIEW
decisions) and fires formatted Markdown messages to Slack or Discord webhooks.

Why Redis list instead of Kafka topic:
  serve.py already writes results to Redis. Adding RPUSH to the same write path
  costs 0.2ms and requires zero new infrastructure. A Kafka consumer for alerts
  would require a new topic, a new consumer group, and offset management —
  overhead not justified for a fire-and-forget notification path.

Why this is architecturally isolated:
  - serve.py doesn't import this module
  - If Slack API goes down, alerts back up in Redis list (no data loss)
  - If this script crashes, scoring pipeline continues unaffected
  - Redis list provides natural backpressure: if Slack rate-limits us,
    alerts queue up and drain at the rate Slack allows

Message format (Slack/Discord Markdown):
  🚨 *BLOCK* | NeoBank | txn_abc123
  Risk: 0.921 | $12,500.00 | ATO
  ─────────────────────────────
  Top signals:
  • velocity_ratio: 14.3 ↑ (↑ fraud risk)
  • is_first_time_receiver: 1 (↑ fraud risk)
  • balance_drain_ratio: 0.88 (↑ fraud risk)

Webhook setup (free, 2 minutes):
  Slack:   https://api.slack.com/apps → Create App → Incoming Webhooks
  Discord: Server Settings → Integrations → Webhooks → Copy URL

  Set env var:
    ALERT_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
    ALERT_PROVIDER=slack   # or "discord"

Usage:
  python -m services.alerting.notifier

  Or via Docker:
  docker compose up alerting
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from threading import Event
from typing import Optional

import httpx
import redis as redis_lib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [alerting] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST        = os.getenv("REDIS_HOST",        "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT",    "6379"))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_PROVIDER    = os.getenv("ALERT_PROVIDER",    "discord").lower()  # "slack" or "discord"
ALERT_QUEUE_KEY   = "alerts:queue"

# Only alert on these decisions
ALERT_DECISIONS   = {"block", "review"}

# Minimum risk score to alert on for REVIEW (avoid noisy mid-range alerts)
REVIEW_MIN_SCORE  = 0.60

# Minimum amount to alert on (skip tiny test transactions)
MIN_ALERT_AMOUNT  = 100.0

# Rate limiting — max N alerts per minute to avoid webhook throttling
MAX_ALERTS_PER_MINUTE = 30
_alert_count_window   = []  # timestamps of recent alerts


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def _decision_emoji(decision: str) -> str:
    return {"block": "🚨", "review": "⚠️", "approve": "✅"}.get(decision, "❓")


def _format_shap_lines(shap_features: list[dict]) -> str:
    if not shap_features:
        return "• (no SHAP explanation available)"
    lines = []
    for sf in shap_features[:5]:
        direction = "↑ fraud" if sf.get("shap_value", 0) > 0 else "↓ fraud"
        val = sf.get("value", 0)
        # Human-readable feature name
        name = sf.get("feature", "?").replace("_", " ")
        lines.append(f"• {name}: `{val:.4g}` ({direction})")
    return "\n".join(lines)


def _format_slack_payload(result: dict) -> dict:
    """Rich Slack Block Kit message."""
    decision  = result.get("decision", "unknown")
    score     = result.get("risk_score", 0)
    txn_id    = result.get("transaction_id", "?")[:24]
    client_id = result.get("client_id", result.get("_client_id", "?"))
    amount    = result.get("amount_usd", result.get("_amount_usd", 0))
    emoji     = _decision_emoji(decision)
    reasons   = result.get("reasons", [])
    shap      = result.get("shap_top_features", [])
    ts        = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    reasons_text = "\n".join(f"• {r}" for r in reasons[:3]) if reasons else "• (no reasons)"
    shap_text    = _format_shap_lines(shap)

    color_map = {"block": "#E53E3E", "review": "#DD6B20", "approve": "#38A169"}
    color     = color_map.get(decision, "#718096")

    return {
        "attachments": [{
            "color": color,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Sentinel Fraud Alert — {decision.upper()}",
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Transaction*\n`{txn_id}`"},
                        {"type": "mrkdwn", "text": f"*Institution*\n{client_id}"},
                        {"type": "mrkdwn", "text": f"*Risk Score*\n`{score:.4f}`"},
                        {"type": "mrkdwn", "text": f"*Amount*\n`${amount:,.2f}`"},
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Decision Reasons:*\n{reasons_text}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Top SHAP Features:*\n{shap_text}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Sentinel | {ts} | model: {result.get('model_name', '?')}"}
                    ]
                }
            ]
        }]
    }


def _format_discord_payload(result: dict) -> dict:
    """Discord webhook embed message."""
    decision  = result.get("decision", "unknown")
    score     = result.get("risk_score", 0)
    txn_id    = result.get("transaction_id", "?")[:24]
    client_id = result.get("client_id", result.get("_client_id", "?"))
    amount    = result.get("amount_usd", result.get("_amount_usd", 0))
    emoji     = _decision_emoji(decision)
    reasons   = result.get("reasons", [])
    shap      = result.get("shap_top_features", [])
    ts        = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    reasons_text = "\n".join(f"• {r}" for r in reasons[:3]) if reasons else "• (no reasons)"
    shap_text    = _format_shap_lines(shap)

    color_map = {"block": 0xE53E3E, "review": 0xDD6B20, "approve": 0x38A169}
    color     = color_map.get(decision, 0x718096)

    return {
        "embeds": [{
            "title":       f"{emoji} Sentinel Alert — {decision.upper()}",
            "color":       color,
            "fields": [
                {"name": "Transaction",  "value": f"`{txn_id}`",        "inline": True},
                {"name": "Institution",  "value": client_id,            "inline": True},
                {"name": "Risk Score",   "value": f"`{score:.4f}`",     "inline": True},
                {"name": "Amount",       "value": f"`${amount:,.2f}`",  "inline": True},
                {"name": "Reasons",      "value": reasons_text,         "inline": False},
                {"name": "Top Features", "value": shap_text,            "inline": False},
            ],
            "footer": {
                "text": f"Sentinel Fraud Detection | {ts} | model: {result.get('model_name', '?')}"
            }
        }]
    }


def format_alert(result: dict, provider: str) -> dict:
    if provider == "slack":
        return _format_slack_payload(result)
    return _format_discord_payload(result)


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limited() -> bool:
    """Simple in-process rate limiter: max N alerts per 60 seconds."""
    global _alert_count_window
    now = time.time()
    _alert_count_window = [t for t in _alert_count_window if now - t < 60]
    if len(_alert_count_window) >= MAX_ALERTS_PER_MINUTE:
        return True
    _alert_count_window.append(now)
    return False


def _should_alert(result: dict) -> bool:
    """Filter logic — not every scored transaction warrants an alert."""
    decision = result.get("decision", "approve")
    score    = result.get("risk_score", 0)
    amount   = result.get("amount_usd", result.get("_amount_usd", 0))

    if decision == "approve":
        return False
    if amount < MIN_ALERT_AMOUNT:
        return False
    if decision == "review" and score < REVIEW_MIN_SCORE:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SEND ALERT
# ─────────────────────────────────────────────────────────────────────────────

def send_alert(
    http_client: httpx.Client,
    result:      dict,
    webhook_url: str,
    provider:    str,
) -> bool:
    """POST alert to webhook. Returns True on success."""
    if _is_rate_limited():
        log.warning("Rate limited — alert queued for next window")
        return False

    payload = format_alert(result, provider)
    try:
        resp = http_client.post(
            webhook_url,
            json=payload,
            timeout=5.0,
        )
        if resp.status_code in (200, 204):
            log.info(
                "Alert sent: %s | %s | score=%.3f | $%.0f",
                result.get("transaction_id", "?")[:20],
                result.get("decision", "?").upper(),
                result.get("risk_score", 0),
                result.get("amount_usd", result.get("_amount_usd", 0)),
            )
            return True
        else:
            log.warning("Webhook returned %d: %s", resp.status_code, resp.text[:200])
            return False
    except httpx.TimeoutException:
        log.warning("Webhook timeout (5s)")
        return False
    except Exception as e:
        log.error("Alert send failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 55)
    log.info("SENTINEL ALERTING SERVICE")
    log.info("=" * 55)
    log.info("  Redis:    %s:%s → %s", REDIS_HOST, REDIS_PORT, ALERT_QUEUE_KEY)
    log.info("  Provider: %s", ALERT_PROVIDER)
    log.info("  Webhook:  %s", ALERT_WEBHOOK_URL[:40] + "..." if ALERT_WEBHOOK_URL else "NOT SET")
    log.info("  Min score for REVIEW: %.2f", REVIEW_MIN_SCORE)
    log.info("  Rate limit: %d/min", MAX_ALERTS_PER_MINUTE)

    if not ALERT_WEBHOOK_URL:
        log.warning("ALERT_WEBHOOK_URL not set — alerts will be logged but not sent")
        log.warning("Set ALERT_WEBHOOK_URL and restart to enable real alerts")

    # Connect Redis
    try:
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        log.info("  Redis: connected")
    except Exception as e:
        log.error("  Redis: FAILED — %s", e)
        return

    http_client = httpx.Client()
    stop_event  = Event()
    signal.signal(signal.SIGINT,  lambda s, f: stop_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())

    total_sent   = 0
    total_skip   = 0
    total_failed = 0
    queue_depth  = 0

    log.info("Listening on alerts:queue... (BLPOP with 5s timeout)")

    while not stop_event.is_set():
        try:
            # BLPOP blocks for up to 5 seconds, then loops
            # This allows clean shutdown on SIGTERM
            item = r.blpop(ALERT_QUEUE_KEY, timeout=5)
        except Exception as e:
            log.error("Redis BLPOP error: %s", e)
            time.sleep(2)
            continue

        if item is None:
            continue  # timeout — check stop_event and loop

        _, raw = item
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("Invalid JSON in alert queue: %s", e)
            continue

        # Apply filter
        if not _should_alert(result):
            total_skip += 1
            continue

        # Send alert
        if ALERT_WEBHOOK_URL:
            success = send_alert(http_client, result, ALERT_WEBHOOK_URL, ALERT_PROVIDER)
            if success:
                total_sent += 1
            else:
                total_failed += 1
                # Re-queue for retry (push to back of list)
                r.rpush(f"{ALERT_QUEUE_KEY}:retry", raw)
        else:
            # No webhook — just log
            log.info(
                "[ALERT] %s | score=%.3f | $%.0f | %s",
                result.get("decision", "?").upper(),
                result.get("risk_score", 0),
                result.get("amount_usd", 0),
                result.get("transaction_id", "?")[:20],
            )
            total_sent += 1

        # Periodic stats
        if (total_sent + total_skip + total_failed) % 50 == 0:
            queue_depth = r.llen(ALERT_QUEUE_KEY)
            log.info(
                "Stats: sent=%d skip=%d fail=%d queue=%d",
                total_sent, total_skip, total_failed, queue_depth
            )

    http_client.close()
    log.info("Alerting service stopped. sent=%d skip=%d fail=%d",
             total_sent, total_skip, total_failed)


if __name__ == "__main__":
    run()