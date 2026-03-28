"""
pipeline.py
===========
Sentinel Fraud Detection — Training Pipeline Entry Point

Usage:
    # Sequential Optuna (default — works everywhere)
    python pipeline.py

    # Parallel Ray Tune (recommended — ~4x faster on 8-core machine)
    python pipeline.py --use-ray

    # Ray Tune with custom CPU allocation per trial
    # 2 CPUs/trial on 8-core = 4 concurrent trials
    # 1 CPU/trial on 8-core  = 8 concurrent trials (more concurrency, less per-trial speed)
    python pipeline.py --use-ray --n-cpus-per-trial 2

    # Quick smoke test (5 trials, no Ray)
    python pipeline.py --trials 5

    # Specific models only
    python pipeline.py --models xgboost,catboost

    # Phase 9: add 32 GNN embedding features
    python pipeline.py --use-gnn-embeddings

    # Prefect deployment mode (listens for runs)
    python pipeline.py --mode serve

Ray Tune notes:
    - Requires: pip install "ray[tune]" optuna-integration
    - Trials run in parallel Ray actors (separate processes)
    - ASHA scheduler kills bad trials early (~40% compute saved)
    - OptunaSearch (TPE) guides the search — same algorithm as Optuna
    - 100K-row subsample per trial, full data for final fit
    - MLflow logging happens from the driver, not inside actors
    - macOS: uses 'spawn' multiprocessing to avoid fork issues
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from model_training import (
    Config,
    ModelType,
    load_and_prepare_data,
    run_all_experiments,
    promote_best_model,
)

# Prevent OpenMP thread contention between scikit-learn, XGBoost, LightGBM
# Same fix used in AntiBully pipeline
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

try:
    from prefect import flow
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(**kwargs):
        def decorator(func):
            return func
        return decorator


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_environment(config: Config, use_ray: bool) -> bool:
    print("\n" + "=" * 60)
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)

    issues = []

    required = [
        ("xgboost",  "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        ("sklearn",  "scikit-learn"),
        ("mlflow",   "MLflow"),
        ("optuna",   "Optuna"),
        ("shap",     "SHAP"),
        ("pandas",   "Pandas"),
        ("pyarrow",  "PyArrow"),
    ]

    if use_ray:
        required += [
            ("ray",                        "Ray"),
            ("ray.tune",                   "Ray Tune"),
            ("ray.tune.search.optuna",     "Ray Tune / OptunaSearch"),
        ]

    for pkg, name in required:
        try:
            __import__(pkg)
            print(f"  ✅ {name}")
        except ImportError:
            pkg_install = "ray[tune] optuna-integration" if "ray" in pkg else pkg
            print(f"  ❌ {name}  →  pip install {pkg_install}")
            issues.append(pkg)

    for path_attr, label in [
        ("TRAIN_PATH", "train.parquet"),
        ("VAL_PATH",   "val.parquet"),
        ("TEST_PATH",  "test.parquet"),
    ]:
        p = getattr(config, path_attr)
        if p.exists():
            print(f"  ✅ {label} found ({p})")
        else:
            print(f"  ❌ {label} not found at {p}")
            issues.append(f"Run: python scripts/merge_training_data.py")

    try:
        config.setup_directories()
        print(f"  ✅ Output directories OK")
    except Exception as e:
        print(f"  ❌ Cannot create directories: {e}")
        issues.append("filesystem")

    if use_ray:
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            n_cpus = int(ray.available_resources().get("CPU", 0))
            print(f"  ✅ Ray cluster: {n_cpus} CPUs available")
        except Exception as e:
            print(f"  ❌ Ray init failed: {e}")
            issues.append("ray")

    print("=" * 60)
    if issues:
        print("\n  Issues found:")
        for issue in set(issues):
            print(f"    - {issue}")
        return False

    print("\n  ✅ All checks passed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(config: Config) -> MlflowClient:
    print("\n" + "=" * 60)
    print("MLFLOW SETUP")
    print("=" * 60)

    dagshub_repo  = os.getenv("DAGSHUB_REPO")
    dagshub_owner = os.getenv("DAGSHUB_OWNER")

    if dagshub_repo and dagshub_owner:
        try:
            import dagshub
            dagshub.init(repo_owner=dagshub_owner,
                         repo_name=dagshub_repo, mlflow=True)
            print(f"  ✅ DagsHub: {dagshub_owner}/{dagshub_repo}")
        except Exception as e:
            print(f"  ⚠️  DagsHub failed ({e}) — using local tracking")
            _set_local_mlflow()
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"  Tracking URI: {tracking_uri}")

    client = MlflowClient()

    try:
        exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
        if not exp:
            mlflow.create_experiment(
                config.EXPERIMENT_NAME,
                artifact_location=os.getenv("MLFLOW_ARTIFACT_LOCATION", "/mlflow/artifacts"),
            )
            print(f"  ✅ Created experiment: {config.EXPERIMENT_NAME}")
        else:
            print(f"  ✅ Using experiment:   {config.EXPERIMENT_NAME}")
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    except Exception as e:
        print(f"  ⚠️  MLflow server unreachable ({e}) — falling back to ./mlruns")
        _set_local_mlflow()
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        client = MlflowClient()

    print("=" * 60)
    return client


def _set_local_mlflow():
    mlflow.set_tracking_uri("./mlruns")
    print("  Tracking URI: ./mlruns  (local fallback)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────────────────────────────────────────

@flow(name="sentinel-fraud-detection-training", log_prints=True)
def main_flow(
    n_trials:           int  = 50,
    models:             Optional[str]  = None,
    use_gnn_embeddings: bool = False,
    use_ray:            bool = False,
    n_cpus_per_trial:   int  = 2,
) -> dict:

    print("\n" + "=" * 80)
    print(" " * 18 + "SENTINEL FRAUD DETECTION")
    print(" " * 20 + "TRAINING PIPELINE")
    print("=" * 80)

    config = Config()
    config.setup_directories()

    # ── 1. Validate ────────────────────────────────────────────────────────────
    if not validate_environment(config, use_ray=use_ray):
        print("\n❌ Environment validation failed.")
        sys.exit(1)

    # ── 2. Initialise Ray (once, before any workers are spawned) ───────────────
    if use_ray:
        import ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,      # suppress per-actor logs
                # Ray uses 'fork' on Linux and 'spawn' on macOS by default.
                # Force spawn on all platforms to avoid deadlocks with
                # OpenMP/MKL libraries used by XGBoost / LightGBM.
                runtime_env={"env_vars": {
                    "OMP_NUM_THREADS":      "1",
                    "MKL_NUM_THREADS":      "1",
                    "OPENBLAS_NUM_THREADS": "1",
                }},
            )
        n_cpus = int(ray.available_resources().get("CPU", 0))
        n_concurrent = max(1, n_cpus // n_cpus_per_trial)
        print(f"\n  Ray cluster: {n_cpus} CPUs  →  "
              f"{n_concurrent} concurrent trials "
              f"({n_cpus_per_trial} CPUs each)")

    # ── 3. Resolve models ──────────────────────────────────────────────────────
    model_map = {
        "xgboost":  ModelType.XGBOOST,
        "lightgbm": ModelType.LIGHTGBM,
        "catboost": ModelType.CATBOOST,
        "logistic": ModelType.LOGISTIC,
    }
    models_to_train = None
    if models:
        requested       = [m.strip().lower() for m in models.split(",")]
        models_to_train = [model_map[m] for m in requested if m in model_map]
        unknown         = [m for m in requested if m not in model_map]
        if unknown:
            print(f"  ⚠️  Unknown models ignored: {unknown}")
        if not models_to_train:
            print("  No valid models — training all.")
            models_to_train = None

    print(f"\n  Models:         {[m.value for m in (models_to_train or list(ModelType))]}")
    print(f"  Optuna trials:  {n_trials} per model")
    print(f"  Search method:  {'Ray Tune (parallel)' if use_ray else 'Optuna (sequential)'}")
    print(f"  GNN embeddings: {'yes (Phase 9)' if use_gnn_embeddings else 'no (tabular baseline)'}")

    # ── 4. MLflow ─────────────────────────────────────────────────────────────
    client = setup_mlflow(config)

    # ── 5. Load data ──────────────────────────────────────────────────────────
    data = load_and_prepare_data(use_gnn_embeddings=use_gnn_embeddings)

    # ── 6. Tournament ─────────────────────────────────────────────────────────
    try:
        winner_id, winner_uuid, winner_score, winner_name = run_all_experiments(
            data=data,
            n_trials=n_trials,
            models_to_train=models_to_train,
            use_ray=use_ray,
            n_cpus_per_trial=n_cpus_per_trial,
        )
    except Exception as e:
        print(f"\n❌ Tournament failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── 7. Promote champion ────────────────────────────────────────────────────
    version = promote_best_model(
        winner_id=winner_id,
        winner_uuid=winner_uuid,
        winner_score=winner_score,
        winner_name=winner_name,
        client=client,
    )

    # ── 8. Shutdown Ray ────────────────────────────────────────────────────────
    if use_ray:
        import ray
        ray.shutdown()

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print("\n" + "=" * 80)
    print(" " * 30 + "PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  🏆 Best Model:     {winner_name}")
    print(f"  📊 Test PR-AUC:    {winner_score:.4f}")
    print(f"  📦 MLflow version: {version}")
    print(f"  🔗 Run ID:         {winner_id}")
    print(f"\n  View results:      {tracking_uri}")
    print("\n" + "=" * 80 + "\n")

    return {
        "model_name": winner_name,
        "pr_auc":     winner_score,
        "run_id":     winner_id,
        "version":    version,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # macOS: use 'spawn' to avoid fork-related deadlocks with OpenMP/MKL.
    # Must be called before any multiprocessing starts.
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass   # already set — safe to ignore

    parser = argparse.ArgumentParser(
        description="Sentinel Fraud Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                            # sequential, all models, 50 trials
  python pipeline.py --use-ray                  # parallel Ray Tune, all models
  python pipeline.py --use-ray --trials 20      # parallel, 20 trials (faster)
  python pipeline.py --use-ray --n-cpus-per-trial 1   # max concurrency
  python pipeline.py --models xgboost,catboost  # specific models only
  python pipeline.py --use-gnn-embeddings       # Phase 9: add GNN features
  python pipeline.py --trials 5                 # smoke test
        """,
    )
    parser.add_argument(
        "--mode", choices=["serve", "run"], default="run",
        help="'run' = execute once and exit (default). "
             "'serve' = Prefect deployment mode (listens for runs).",
    )
    parser.add_argument(
        "--trials", type=int,
        default=int(os.getenv("OPTUNA_TRIALS", "50")),
        help="Optuna/Ray Tune trials per model (default: env.OPTUNA_TRIALS or 50). "
             "Recommended: 20 for Ray Tune, 50 for sequential.",
    )
    parser.add_argument(
        "--models", type=str,
        default=os.getenv("TRAIN_MODELS", None),
        help="Comma-separated model names: xgboost,lightgbm,catboost,logistic "
             "(default: all four)",
    )
    parser.add_argument(
        "--use-gnn-embeddings", action="store_true",
        default=os.getenv("USE_GNN_EMBEDDINGS", "false").lower() == "true",
        help="Include 32-dim GNN embedding columns (Phase 9 — after train_graphsage.py)",
    )
    parser.add_argument(
        "--use-ray", action="store_true",
        default=os.getenv("USE_RAY", "false").lower() == "true",
        help="Use Ray Tune for parallel hyperparameter search. "
             "Requires: pip install 'ray[tune]' optuna-integration. "
             "Typical speedup: 3-4x on 8-core machine with --n-cpus-per-trial 2.",
    )
    parser.add_argument(
        "--n-cpus-per-trial", type=int,
        default=int(os.getenv("N_CPUS_PER_TRIAL", "2")),
        help="CPU cores allocated to each Ray trial (default: 2). "
             "Concurrency = total_cpus / n_cpus_per_trial. "
             "Lower = more parallel trials but each trial trains slower. "
             "Use 1 for maximum concurrency, 4 for maximum per-trial speed.",
    )

    args = parser.parse_args()

    if args.mode == "serve":
        if not PREFECT_AVAILABLE:
            print("❌ Prefect not installed. pip install prefect")
            sys.exit(1)
        print("=" * 80)
        print("  PREFECT DEPLOYMENT MODE")
        print("=" * 80)
        main_flow.serve(
            name="sentinel-fraud-training",
            parameters={
                "n_trials":           args.trials,
                "models":             args.models,
                "use_gnn_embeddings": args.use_gnn_embeddings,
                "use_ray":            args.use_ray,
                "n_cpus_per_trial":   args.n_cpus_per_trial,
            },
        )

    else:
        print(f"🚀 Starting pipeline")
        print(f"   Trials:         {args.trials}")
        print(f"   Models:         {args.models or 'all'}")
        print(f"   GNN embeddings: {args.use_gnn_embeddings}")
        print(f"   Ray Tune:       {args.use_ray}")
        if args.use_ray:
            print(f"   CPUs/trial:     {args.n_cpus_per_trial}")

        try:
            result = main_flow(
                n_trials=args.trials,
                models=args.models,
                use_gnn_embeddings=args.use_gnn_embeddings,
                use_ray=args.use_ray,
                n_cpus_per_trial=args.n_cpus_per_trial,
            )
            print(f"\n✅ Training complete — PR-AUC: {result['pr_auc']:.4f}")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            sys.exit(1)