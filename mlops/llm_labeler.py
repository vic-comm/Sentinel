"""
mlops/llm_labeler.py
=====================
LLM-powered pseudo-labeler for REVIEW-zone transactions.

Pulls REVIEW-zone transactions from PostgreSQL, calls Claude claude-sonnet-4-20250514
to label them, and stores labels in labeled_transactions table.

Label quality → sample_weight mapping:
  confidence ≥ 0.85 → weight 1.0
  confidence ≥ 0.70 → weight 0.7
  confidence < 0.70 → discarded

Usage:
  python -m mlops.llm_labeler              # label last 24h
  python -m mlops.llm_labeler --hours 72
  python -m mlops.llm_labeler --dry-run
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [llm_labeler] %(levelname)s %(message)s",
)

ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
POSTGRES_DSN            = os.getenv("DATABASE_URL",
    "postgresql://sentinel:sentinel@localhost:5432/sentinel")

LLM_MODEL               = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS          = 512
LLM_TEMPERATURE         = 0.0
MIN_CONFIDENCE_INCLUDE  = 0.70
BATCH_SIZE              = 20
RATE_LIMIT_SLEEP_S      = 0.5
HUMAN_REVIEW_THRESHOLD  = 50_000.0

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS labeled_transactions (
    id                      SERIAL PRIMARY KEY,
    transaction_id          TEXT NOT NULL UNIQUE,
    client_id               TEXT,
    model_score             FLOAT,
    model_decision          TEXT,
    llm_label               INT,
    llm_confidence          FLOAT,
    llm_reasoning           TEXT,
    llm_fraud_type          TEXT,
    llm_fraud_indicators    JSONB,
    chargeback_label        INT,
    chargeback_received_at  TIMESTAMPTZ,
    human_label             INT,
    human_reviewed_at       TIMESTAMPTZ,
    human_reviewer_id       TEXT,
    label_source            TEXT,
    sample_weight           FLOAT,
    included_in_training    BOOLEAN DEFAULT FALSE,
    training_run_id         TEXT,
    features_json           JSONB,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    labeled_at              TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lt_transaction_id ON labeled_transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_lt_label_source   ON labeled_transactions(label_source);
CREATE INDEX IF NOT EXISTS idx_lt_included       ON labeled_transactions(included_in_training);
CREATE INDEX IF NOT EXISTS idx_lt_created_at     ON labeled_transactions(created_at);
"""

SYSTEM_PROMPT = """You are a senior financial fraud analyst at a multi-institutional
fraud detection platform. You review transactions that an ML model scored between
0.30 and 0.80 — the uncertain zone where the model cannot decide with confidence.

OUTPUT FORMAT (strict JSON, no other text):
{
  "label": 0 or 1,
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentences explaining your decision",
  "fraud_indicators": ["feature1", "feature2"],
  "legitimate_indicators": ["feature3"],
  "fraud_type": "ato|mule_network|structuring|mixer|legitimate|ambiguous"
}

FRAUD PATTERNS:
- ATO: high velocity + new receiver + balance drain
- Mule network: fan-in topology + cross-institutional + same-bank
- Structuring: many sub-threshold amounts + elevated velocity
- Crypto mixer: known_mixer_interaction + large cross-modal amount
- Cross-modal laundering: fiat→crypto within 15 minutes

If confidence < 0.60, still output best guess with that confidence."""


def build_user_prompt(txn: dict, shap_features: list[dict], model_score: float) -> str:
    interpretable = {k: txn.get(k) for k in [
        "amount_usd", "modality", "transfer_network", "sender_account_age_days",
        "sender_archetype", "sender_kyc_status", "is_first_time_receiver",
        "balance_drain_ratio", "count_1h", "count_6h", "velocity_ratio",
        "velocity_baseline_daily", "amount_ratio", "institution_count",
        "cross_modal_pattern_detected", "known_mixer_interaction",
        "receiver_wallet_age_days", "receiver_wallet_type",
        "sender_out_degree", "receiver_in_degree", "fan_in_ratio",
        "hour_of_day", "prior_high_risk_event", "time_since_other_institution_sec",
    ] if txn.get(k) is not None}

    shap_lines = []
    for sf in shap_features[:5]:
        direction = "↑ fraud" if sf.get("shap_value", 0) > 0 else "↓ fraud"
        shap_lines.append(
            f"  {sf['feature']}: value={sf['value']:.4f}, "
            f"SHAP={sf['shap_value']:+.4f} ({direction})"
        )

    return f"""TRANSACTION REVIEW REQUEST

Transaction ID: {txn.get('transaction_id', '?')}
Institution:    {txn.get('client_id', '?')}
ML Model Score: {model_score:.4f} (REVIEW zone: 0.30-0.80)

TRANSACTION FEATURES:
{json.dumps(interpretable, indent=2, default=str)}

ML MODEL EXPLANATION (top SHAP features):
{chr(10).join(shap_lines) or '  (no SHAP available)'}

Is this FRAUD (1) or LEGITIMATE (0)?"""


def call_llm(client, txn, shap_features, model_score, dry_run=False) -> Optional[dict]:
    prompt = build_user_prompt(txn, shap_features, model_score)
    if dry_run:
        log.info("DRY RUN — %s", txn.get("transaction_id", "?")[:20])
        return {"label": 0, "confidence": 0.99, "reasoning": "DRY RUN",
                "fraud_indicators": [], "legitimate_indicators": [], "fraud_type": "ambiguous"}
    try:
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        assert "label" in result and result["label"] in (0, 1)
        assert 0.0 <= result.get("confidence", 0) <= 1.0
        return result
    except Exception as e:
        log.error("LLM call failed for %s: %s", txn.get("transaction_id", "?"), e)
        return None


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def fetch_review_transactions(conn, hours: int = 24) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT t.transaction_id, t.client_id, t.scored_at,
                   t.risk_score AS model_score, t.decision AS model_decision,
                   t.shap_features AS shap_features_json, t.features_json
            FROM scored_transactions t
            LEFT JOIN labeled_transactions l ON t.transaction_id = l.transaction_id
            WHERE t.decision = 'review'
              AND t.scored_at >= %s
              AND l.transaction_id IS NULL
              AND t.amount_usd < %s
            ORDER BY t.risk_score DESC
            LIMIT 500
        """, (cutoff, HUMAN_REVIEW_THRESHOLD))
        return [dict(r) for r in cur.fetchall()]


def save_label(conn, txn_id, client_id, model_score, model_decision, llm_result, features_json):
    confidence = llm_result.get("confidence", 0.5)
    weight     = 1.0 if confidence >= 0.85 else (0.7 if confidence >= 0.70 else 0.4)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO labeled_transactions (
                transaction_id, client_id, model_score, model_decision,
                llm_label, llm_confidence, llm_reasoning, llm_fraud_type,
                llm_fraud_indicators, label_source, sample_weight, features_json
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'llm',%s,%s)
            ON CONFLICT (transaction_id) DO NOTHING
        """, (
            txn_id, client_id, model_score, model_decision,
            llm_result["label"], confidence,
            llm_result.get("reasoning", ""),
            llm_result.get("fraud_type", "unknown"),
            json.dumps(llm_result.get("fraud_indicators", [])),
            weight, json.dumps(features_json or {}),
        ))
    conn.commit()


def ingest_chargebacks(conn, chargeback_file: Optional[str] = None) -> int:
    if not chargeback_file:
        return 0
    import pandas as pd
    df = pd.read_csv(chargeback_file)
    updated = 0
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute("""
                UPDATE labeled_transactions
                SET chargeback_label=%s, chargeback_received_at=NOW(),
                    label_source='chargeback', sample_weight=1.0
                WHERE transaction_id=%s AND chargeback_label IS NULL
            """, (int(row["is_fraud"]), row["transaction_id"]))
            updated += cur.rowcount
    conn.commit()
    log.info("Chargebacks ingested: %d", updated)
    return updated


def run_labeling(hours: int = 24, dry_run: bool = False) -> dict:
    log.info("LLM LABELER | window=%dh dry_run=%s", hours, dry_run)

    if not ANTHROPIC_API_KEY and not dry_run:
        log.error("ANTHROPIC_API_KEY not set. Use --dry-run or set the key.")
        return {"status": "failed", "reason": "no_api_key"}

    conn       = psycopg2.connect(POSTGRES_DSN)
    ensure_schema(conn)
    llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if not dry_run else None

    candidates = fetch_review_transactions(conn, hours=hours)
    log.info("Found %d unlabeled REVIEW transactions", len(candidates))

    if not candidates:
        conn.close()
        return {"status": "ok", "labeled": 0, "skipped": 0, "failed": 0}

    stats = {"labeled": 0, "skipped": 0, "failed": 0, "high_conf": 0}

    for i, row in enumerate(candidates):
        txn_id      = row["transaction_id"]
        model_score = float(row["model_score"])

        try:
            shap_features = json.loads(row.get("shap_features_json") or "[]")
        except Exception:
            shap_features = []

        try:
            features_json = json.loads(row.get("features_json") or "{}")
        except Exception:
            features_json = {}

        log.info("[%3d/%3d] %s | score=%.3f", i+1, len(candidates), txn_id[:20], model_score)

        result = call_llm(
            client        = llm_client,
            txn           = {**features_json, "transaction_id": txn_id,
                             "client_id": row.get("client_id", "?")},
            shap_features = shap_features,
            model_score   = model_score,
            dry_run       = dry_run,
        )

        if result is None:
            stats["failed"] += 1
            continue

        confidence = result.get("confidence", 0)
        if confidence < MIN_CONFIDENCE_INCLUDE:
            log.info("  Discarded (conf=%.2f): %s", confidence, result.get("reasoning", "")[:60])
            stats["skipped"] += 1
            continue

        if not dry_run:
            save_label(conn, txn_id, row.get("client_id", "?"), model_score,
                       row.get("model_decision", "review"), result, features_json)

        stats["labeled"] += 1
        if confidence >= 0.85:
            stats["high_conf"] += 1

        log.info("  label=%d conf=%.2f type=%s | %s",
                 result["label"], confidence,
                 result.get("fraud_type", "?"),
                 result.get("reasoning", "")[:70])

        if not dry_run:
            time.sleep(RATE_LIMIT_SLEEP_S)

    conn.close()
    log.info("SUMMARY: labeled=%d skip=%d fail=%d high_conf=%d",
             stats["labeled"], stats["skipped"], stats["failed"], stats["high_conf"])
    return {"status": "ok", **stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",       type=int,  default=24)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--chargebacks", type=str,  default=None)
    args = parser.parse_args()

    if args.chargebacks:
        conn = psycopg2.connect(POSTGRES_DSN)
        ingest_chargebacks(conn, args.chargebacks)
        conn.close()

    run_labeling(hours=args.hours, dry_run=args.dry_run)