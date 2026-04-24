"""
mlops/retrain_pipeline.py
==========================
Sentinel MLOps orchestrator — Prefect-based retraining pipeline.

Stages:
  1. LLM labeling    — label REVIEW-zone txns from last 24h
  2. Chargeback      — ingest delayed gold labels
  3. Drift detection — 4 signals: PSI, PR-AUC, fraud rate, model age
  4. Data ingestion  — merge labeled data into training set
  5. Retrain gate    — decide: retrain / schedule / skip
  6. Training        — run ml.pipeline with new data
  7. Challenger eval — compare challenger vs champion PR-AUC
  8. Deployment      — promote challenger if improved

Schedule: daily at 2am WAT
  prefect deployment build mlops/retrain_pipeline.py:sentinel_mlops_pipeline \\
    --name sentinel-daily --cron "0 1 * * *" --apply
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

from mlops.llm_labeler import run_labeling, ingest_chargebacks

load_dotenv()

POSTGRES_DSN  = os.getenv("DATABASE_URL",
    "postgresql://sentinel:sentinel@localhost:5432/sentinel")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))

# Drift thresholds
PSI_WARN             = 0.10
PSI_ALERT            = 0.25
PR_AUC_DROP_HIGH     = 0.05
PR_AUC_DROP_CRIT     = 0.10
FRAUD_RATE_MULT      = 3.0
MANDATORY_RETRAIN_DAYS = 14


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@task(name="compute_psi", retries=2, retry_delay_seconds=30)
def compute_psi_drift(conn) -> dict:
    """PSI on top-10 SHAP features vs 30-day baseline."""
    logger = get_run_logger()
    top_features = [
        "velocity_ratio", "amount_ratio", "count_6h", "count_1h",
        "balance_drain_ratio", "sender_account_age_days",
        "is_first_time_receiver", "velocity_baseline_daily",
        "institution_count", "fan_in_ratio",
    ]
    cutoff_ref_end   = datetime.now(timezone.utc) - timedelta(days=7)
    cutoff_ref_start = cutoff_ref_end - timedelta(days=30)
    cutoff_curr      = datetime.now(timezone.utc) - timedelta(days=7)

    psi_results = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for feat in top_features:
            try:
                cur.execute("""
                    SELECT AVG((features_json->%s)::float) AS mean_val,
                           STDDEV((features_json->%s)::float) AS std_val
                    FROM scored_transactions
                    WHERE scored_at BETWEEN %s AND %s
                      AND features_json->%s IS NOT NULL
                """, (feat, feat, cutoff_ref_start, cutoff_ref_end, feat))
                ref = cur.fetchone()

                cur.execute("""
                    SELECT AVG((features_json->%s)::float) AS mean_val,
                           STDDEV((features_json->%s)::float) AS std_val
                    FROM scored_transactions
                    WHERE scored_at >= %s
                      AND features_json->%s IS NOT NULL
                """, (feat, feat, cutoff_curr, feat))
                curr = cur.fetchone()

                if not (ref and curr and ref["mean_val"] is not None):
                    continue

                psi_approx = abs(float(curr["mean_val"] or 0) - float(ref["mean_val"] or 0))
                psi_approx /= max(float(ref["std_val"] or 1), 1e-9)
                psi_approx  = min(psi_approx / 3.0, 1.0)
                psi_results[feat] = round(psi_approx, 4)
            except Exception as e:
                logger.warning(f"PSI failed for {feat}: {e}")

    max_psi       = max(psi_results.values(), default=0.0)
    drifted       = [f for f, p in psi_results.items() if p > PSI_WARN]
    severity      = "high" if max_psi > PSI_ALERT else ("medium" if max_psi > PSI_WARN else "none")

    logger.info(f"PSI: max={max_psi:.4f} drifted={drifted} → {severity}")
    return {"max_psi": max_psi, "psi_by_feature": psi_results,
            "drifted_features": drifted, "severity": severity}


@task(name="compute_pr_auc_drift", retries=2)
def compute_pr_auc_drift(conn) -> dict:
    """Rolling PR-AUC on labeled transactions vs MLflow baseline."""
    logger = get_run_logger()
    from sklearn.metrics import average_precision_score

    # Load baseline from MLflow
    baseline_pr_auc = 0.770
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client   = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='sentinel-fraud-detection'")
        if versions:
            best = max(versions, key=lambda v: float(
                client.get_run(v.run_id).data.metrics.get("test_pr_auc", 0)
            ))
            baseline_pr_auc = float(
                client.get_run(best.run_id).data.metrics.get("test_pr_auc", baseline_pr_auc)
            )
    except Exception as e:
        logger.warning(f"Could not load baseline PR-AUC: {e}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT t.risk_score AS model_score,
                   COALESCE(l.chargeback_label, l.llm_label) AS true_label
            FROM scored_transactions t
            JOIN labeled_transactions l ON t.transaction_id = l.transaction_id
            WHERE t.scored_at >= %s
              AND COALESCE(l.chargeback_label, l.llm_label) IS NOT NULL
              AND (l.label_source='chargeback' OR l.llm_confidence >= 0.80)
        """, (cutoff,))
        rows = cur.fetchall()

    if len(rows) < 50:
        logger.warning(f"Too few labeled rows ({len(rows)}) for PR-AUC — skipping")
        return {"rolling_pr_auc": None, "baseline_pr_auc": baseline_pr_auc,
                "drop": 0.0, "severity": "none", "n_labeled": len(rows)}

    y_true  = [int(r["true_label"]) for r in rows]
    y_score = [float(r["model_score"]) for r in rows]
    rolling = average_precision_score(y_true, y_score)
    drop    = baseline_pr_auc - rolling
    pct     = (drop / baseline_pr_auc) * 100 if baseline_pr_auc > 0 else 0

    severity = ("critical" if drop > PR_AUC_DROP_CRIT else
                "high"     if drop > PR_AUC_DROP_HIGH else "none")

    logger.info(f"PR-AUC: rolling={rolling:.4f} baseline={baseline_pr_auc:.4f} "
                f"drop={drop:+.4f} ({pct:.1f}%) → {severity}")
    return {"rolling_pr_auc": round(rolling, 4), "baseline_pr_auc": round(baseline_pr_auc, 4),
            "drop": round(drop, 4), "pct_drop": round(pct, 1),
            "severity": severity, "n_labeled": len(rows)}


@task(name="compute_fraud_rate_drift")
def compute_fraud_rate_drift(conn) -> dict:
    """Compare current 7d block rate vs 30-day baseline."""
    logger = get_run_logger()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE decision='block') * 1.0 / NULLIF(COUNT(*),0)
            FROM scored_transactions
            WHERE scored_at >= NOW() - INTERVAL '37 days'
              AND scored_at <  NOW() - INTERVAL '7 days'
        """)
        baseline = float((cur.fetchone() or [0.005])[0] or 0.005)

        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE decision='block') * 1.0 / NULLIF(COUNT(*),0)
            FROM scored_transactions WHERE scored_at >= NOW() - INTERVAL '7 days'
        """)
        current = float((cur.fetchone() or [0.005])[0] or 0.005)

    ratio    = current / max(baseline, 0.001)
    severity = ("critical" if ratio >= FRAUD_RATE_MULT else
                "high"     if ratio >= 2.0 else "none")

    logger.info(f"Fraud rate: baseline={baseline:.3%} current={current:.3%} "
                f"ratio={ratio:.1f}x → {severity}")
    return {"baseline_rate": round(baseline, 5), "current_rate": round(current, 5),
            "ratio": round(ratio, 2), "severity": severity}


@task(name="check_mandatory_retrain")
def check_mandatory_retrain() -> dict:
    logger = get_run_logger()
    path = MODELS_DIR / "last_trained_at.txt"
    if not path.exists():
        logger.warning("No last_trained_at.txt — assuming stale model")
        return {"days_since_train": 999, "mandatory": True}
    last     = datetime.fromisoformat(path.read_text().strip())
    days_old = (datetime.now(timezone.utc) - last).days
    logger.info(f"Model age: {days_old}d (threshold: {MANDATORY_RETRAIN_DAYS})")
    return {"days_since_train": days_old, "mandatory": days_old >= MANDATORY_RETRAIN_DAYS}


# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

@task(name="ingest_labeled_data", retries=1)
def ingest_labeled_data(conn, min_new_labels: int = 100) -> dict:
    logger = get_run_logger()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT transaction_id,
                   COALESCE(chargeback_label, human_label, llm_label) AS label,
                   label_source, sample_weight, features_json
            FROM labeled_transactions
            WHERE included_in_training = FALSE
              AND COALESCE(chargeback_label, human_label, llm_label) IS NOT NULL
              AND sample_weight >= 0.40
            ORDER BY CASE label_source
                       WHEN 'chargeback' THEN 1
                       WHEN 'human'      THEN 2
                       ELSE 3 END,
                     sample_weight DESC
        """)
        rows = cur.fetchall()

    logger.info(f"Candidate labels: {len(rows)}")
    if not rows:
        return {"status": "skipped", "reason": "no_new_labels", "n": 0}
    if len(rows) < min_new_labels:
        logger.warning(f"Only {len(rows)}/{min_new_labels} labels — accumulating")
        return {"status": "insufficient", "n": len(rows), "needed": min_new_labels - len(rows)}

    records = []
    for row in rows:
        try:
            features = json.loads(row["features_json"] or "{}")
            features["is_fraud"]      = int(row["label"])
            features["sample_weight"] = float(row["sample_weight"])
            features["label_source"]  = row["label_source"]
            records.append(features)
        except Exception as e:
            logger.warning(f"Skipping {row['transaction_id']}: {e}")

    new_df     = pd.DataFrame(records)
    train_path = Path("data/training/train.parquet")

    if train_path.exists():
        existing = pd.read_parquet(train_path)
        merged   = pd.concat([existing, new_df], ignore_index=True)
        if "transaction_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["transaction_id"], keep="last")
        merged.to_parquet(train_path, index=False)
        logger.info(f"Merged: {len(existing)}+{len(new_df)}={len(merged)} rows")
    else:
        new_df.to_parquet(train_path, index=False)

    txn_ids = [row["transaction_id"] for row in rows]
    with conn.cursor() as cur:
        cur.execute("UPDATE labeled_transactions SET included_in_training=TRUE WHERE transaction_id=ANY(%s)", (txn_ids,))
    conn.commit()

    breakdown = {}
    for row in rows:
        s = row["label_source"]
        breakdown[s] = breakdown.get(s, 0) + 1

    return {"status": "ok", "n": len(rows), "label_breakdown": breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING + DEPLOYMENT
# ─────────────────────────────────────────────────────────────────────────────

@task(name="run_training", retries=1, retry_delay_seconds=60)
def run_training(use_gnn: bool = True) -> dict:
    logger = get_run_logger()
    import subprocess, sys

    steps = [[sys.executable, "-m", "scripts.build_features"]]
    if use_gnn:
        steps += [
            [sys.executable, "-m", "scripts.embed_gnn"],
            [sys.executable, "-m", "scripts.build_features", "--attach-gnn"],
        ]
    steps += [
        [sys.executable, "-m", "scripts.split_data"],
        [sys.executable, "-m", "ml.pipeline", "--use-ray", "--trials", "30",
         *(["--use-gnn-embeddings"] if use_gnn else [])],
    ]

    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"Step failed: {' '.join(cmd)}\n{r.stderr}")
            return {"status": "failed", "step": ' '.join(cmd)}
        logger.info(f"  ✓ {' '.join(cmd)}")

    (MODELS_DIR / "last_trained_at.txt").write_text(datetime.now(timezone.utc).isoformat())

    new_pr_auc = None
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client   = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='sentinel-fraud-detection'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            new_pr_auc = float(
                client.get_run(latest.run_id).data.metrics.get("test_pr_auc", 0)
            )
            logger.info(f"New model PR-AUC: {new_pr_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not read new PR-AUC: {e}")

    return {"status": "ok", "new_pr_auc": new_pr_auc}


@task(name="champion_challenger")
def evaluate_challenger(champion_pr_auc: float, challenger_pr_auc: Optional[float]) -> dict:
    logger = get_run_logger()
    if challenger_pr_auc is None:
        return {"promote": False, "reason": "unknown_challenger_pr_auc"}
    delta   = challenger_pr_auc - champion_pr_auc
    promote = delta > 0.005
    logger.info(f"Champion={champion_pr_auc:.4f} Challenger={challenger_pr_auc:.4f} "
                f"delta={delta:+.4f} → {'PROMOTE' if promote else 'KEEP'}")
    return {"promote": promote, "champion_pr_auc": champion_pr_auc,
            "challenger_pr_auc": challenger_pr_auc, "delta": round(delta, 4),
            "reason": "improvement" if promote else "no_improvement"}


@task(name="hot_swap_model")
def hot_swap_model() -> dict:
    logger = get_run_logger()
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "services.model_serving.serve", "--hot-swap"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode == 0:
        logger.info("Hot-swap successful")
        return {"status": "ok"}
    logger.error(f"Hot-swap failed: {r.stderr}")
    return {"status": "failed", "error": r.stderr}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FLOW
# ─────────────────────────────────────────────────────────────────────────────

@flow(
    name="sentinel_mlops_pipeline",
    description="Daily Sentinel MLOps: LLM labeling + drift detection + conditional retraining",
    log_prints=True,
)
def sentinel_mlops_pipeline(
    llm_label_hours: int  = 24,
    min_new_labels:  int  = 100,
    force_retrain:   bool = False,
    dry_run:         bool = False,
):
    logger = get_run_logger()
    logger.info("=" * 55)
    logger.info("SENTINEL MLOPS PIPELINE")
    logger.info(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"  force_retrain={force_retrain} dry_run={dry_run}")
    logger.info("=" * 55)

    conn = psycopg2.connect(POSTGRES_DSN)

    # Stage 1: LLM labeling
    logger.info("\n[1] LLM labeling...")
    llm_stats = run_labeling(hours=llm_label_hours, dry_run=dry_run)
    logger.info(f"  labeled={llm_stats.get('labeled')} skip={llm_stats.get('skipped')} fail={llm_stats.get('failed')}")

    # Stage 2: Chargebacks
    logger.info("\n[2] Chargeback ingestion...")
    cb_file = Path("data/chargebacks/latest.csv")
    if cb_file.exists():
        n_cb = ingest_chargebacks(conn, str(cb_file))
        logger.info(f"  ingested {n_cb} chargebacks")

    # Stage 3: Drift detection
    logger.info("\n[3] Drift detection...")
    psi_r   = compute_psi_drift(conn)
    prauc_r = compute_pr_auc_drift(conn)
    fraud_r = compute_fraud_rate_drift(conn)
    time_r  = check_mandatory_retrain()

    severity_rank = {"none": 0, "medium": 1, "high": 2, "critical": 3}
    severities    = [psi_r["severity"], prauc_r["severity"], fraud_r["severity"],
                     "critical" if time_r["mandatory"] else "none"]
    overall       = max(severities, key=lambda s: severity_rank.get(s, 0))

    logger.info(f"  PSI={psi_r['max_psi']:.4f}({psi_r['severity']}) "
                f"PR-AUC drop={prauc_r.get('pct_drop','?')}%({prauc_r['severity']}) "
                f"fraud_rate={fraud_r['ratio']:.1f}x({fraud_r['severity']}) "
                f"age={time_r['days_since_train']}d")
    logger.info(f"  Overall: {overall.upper()}")

    # Stage 4: Data ingestion
    logger.info("\n[4] Ingesting labels...")
    ingest_r = ingest_labeled_data(conn, min_new_labels=min_new_labels)

    if ingest_r["status"] == "insufficient" and not force_retrain:
        conn.close()
        return {"status": "waiting_for_labels", "drift": overall, **ingest_r}

    # Stage 5: Retrain gate
    if not force_retrain and overall not in ("high", "critical"):
        if overall == "medium":
            logger.info("[5] Medium drift — scheduling for next cycle")
            conn.close()
            return {"status": "scheduled", "drift_severity": overall}
        logger.info("[5] Model stable — no action")
        conn.close()
        return {"status": "stable", "drift_severity": overall}

    if dry_run:
        logger.info("[5] DRY RUN — would retrain")
        conn.close()
        return {"status": "dry_run", "would_retrain": True}

    # Stage 6: Training
    logger.info(f"\n[6] Retraining ({overall.upper()})...")
    champion_pr_auc = prauc_r.get("baseline_pr_auc", 0.770)
    train_r         = run_training(use_gnn=True)

    if train_r["status"] == "failed":
        conn.close()
        return {"status": "training_failed"}

    # Stage 7: Challenger evaluation
    eval_r = evaluate_challenger(champion_pr_auc, train_r.get("new_pr_auc"))
    if not eval_r["promote"]:
        logger.warning(f"[7] Challenger rejected ({eval_r['reason']}) — champion retained")
        conn.close()
        return {"status": "challenger_rejected", **eval_r}

    # Stage 8: Deployment
    logger.info(f"\n[8] Promoting challenger PR-AUC "
                f"{champion_pr_auc:.4f} → {eval_r['challenger_pr_auc']:.4f}")
    swap_r = hot_swap_model()
    conn.close()

    return {
        "status":            "retrained_and_deployed" if swap_r["status"] == "ok" else "deploy_failed",
        "drift_severity":    overall,
        "champion_pr_auc":   champion_pr_auc,
        "challenger_pr_auc": eval_r["challenger_pr_auc"],
        "improvement":       eval_r["delta"],
        "new_labels_used":   ingest_r.get("n", 0),
        "label_breakdown":   ingest_r.get("label_breakdown", {}),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hours",   type=int, default=24)
    args = parser.parse_args()
    sentinel_mlops_pipeline(
        llm_label_hours=args.hours,
        force_retrain=args.force,
        dry_run=args.dry_run,
    )