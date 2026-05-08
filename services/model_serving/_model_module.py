"""
services/model_serving/_model_bundle.py
=========================================
ModelBundle with MLflow + DagsHub loading.

Replaces the joblib.load() fallback chain with proper MLflow model registry
integration. Supports three loading strategies in priority order:

  1. MLflow Model Registry (local server or DagsHub remote)
  2. Local MLflow run artifacts (./mlruns — offline fallback)
  3. Local .pkl files (last resort — matches old behaviour)

DagsHub integration:
  DagsHub hosts MLflow tracking servers for free. When you push your
  Sentinel runs to DagsHub, every model version, metric, and artifact
  is accessible remotely. This means:
    - serve.py on Railway/Render loads the champion model from DagsHub
    - No model files committed to git
    - Model registry shows version history and PR-AUC progression

  Setup (one-time):
    pip install dagshub mlflow
    dagshub.init(repo_owner="your_username", repo_name="Sentinel", mlflow=True)
    # OR set env var: MLFLOW_TRACKING_URI=https://dagshub.com/user/Sentinel.mlflow

Loading strategy:
  1. Check MLFLOW_TRACKING_URI env var (DagsHub or local server)
  2. Search model registry for 'sentinel-fraud-detection'
  3. Find version with highest test_pr_auc metric
  4. Load via mlflow.sklearn.load_model() OR mlflow.pyfunc.load_model()
  5. Fall back to local ./mlruns if registry unreachable
  6. Fall back to .pkl files if no MLflow at all
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Set ONE of these — they are tried in order:
#   DAGSHUB_URI:   "https://dagshub.com/your_username/Sentinel.mlflow"
#   LOCAL_URI:     "http://localhost:5000"  (mlflow server running)
#   MLRUNS_URI:    "./mlruns"               (file-based, always works)
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
DAGSHUB_TOKEN         = os.getenv("DAGSHUB_TOKEN", "")          # Personal access token
DAGSHUB_USERNAME      = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_REPO          = os.getenv("DAGSHUB_REPO", "Sentinel")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentinel-fraud-detection")
MODELS_DIR            = Path(os.getenv("MODELS_DIR", "models"))

# Fallback: .pkl files if MLflow entirely unavailable
CHAMPION_PRIORITY = [
    "xgboost_model.pkl",
    "catboost_model.pkl",
    "lightgbm_model.pkl",
    "logistic_reg_model.pkl",
]

# SHAP sampling — 1 in N requests (always compute for risk_score > 0.70)
SHAP_SAMPLE_RATE = int(os.getenv("SHAP_SAMPLE_RATE", "5"))

APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.30"))
BLOCK_THRESHOLD   = float(os.getenv("BLOCK_THRESHOLD",   "0.80"))


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _setup_mlflow_client() -> Optional[MlflowClient]:
    """
    Configure MLflow tracking URI and return a client.

    DagsHub setup:
      Option A (env vars — recommended for deployment):
        MLFLOW_TRACKING_URI=https://dagshub.com/user/Sentinel.mlflow
        MLFLOW_TRACKING_USERNAME=your_username
        MLFLOW_TRACKING_PASSWORD=your_dagshub_token

      Option B (dagshub SDK — requires dagshub package):
        import dagshub
        dagshub.init(repo_owner="user", repo_name="Sentinel", mlflow=True)

    The env var approach is preferred for production because it doesn't
    require the dagshub package — just set the three vars in your deployment.
    """
    uri = MLFLOW_TRACKING_URI

    # DagsHub: set auth if token provided
    if "dagshub.com" in uri and DAGSHUB_TOKEN:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        log.info("MLflow → DagsHub (%s)", uri)
    elif uri == "./mlruns" or uri.startswith("/"):
        log.info("MLflow → local file store (%s)", uri)
    else:
        log.info("MLflow → server (%s)", uri)

    try:
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        # Quick connectivity test
        client.search_experiments(max_results=1)
        return client
    except Exception as e:
        log.warning("MLflow client setup failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUNDLE
# ─────────────────────────────────────────────────────────────────────────────

class ModelBundle:
    """
    Loads the champion fraud detection model with three fallback strategies.

    Strategy 1 — MLflow Model Registry:
      Searches for registered model 'sentinel-fraud-detection', finds the
      version with highest test_pr_auc, loads via mlflow.sklearn.load_model().
      Works with DagsHub, local MLflow server, and ./mlruns.

    Strategy 2 — MLflow run artifacts (no registry):
      Searches all experiments for runs with the highest test_pr_auc,
      loads the model artifact directly from the run.
      Falls back to this when no model is registered.

    Strategy 3 — Local .pkl files:
      Loads the first matching file from CHAMPION_PRIORITY list.
      Legacy fallback — works even with no MLflow at all.
    """

    def __init__(self):
        self.model           = None
        self.model_name      = "unknown"
        self.model_version   = None
        self.model_pr_auc    = None
        self.feature_names:  list[str]  = []
        self.threshold:      float      = BLOCK_THRESHOLD
        self.shap_explainer             = None
        self._request_count: int        = 0
        self._loaded_from:   str        = "none"

    def load(self) -> "ModelBundle":
        """
        Load model using best available strategy.
        Raises RuntimeError if all strategies fail.
        """
        loaded = (
            self._load_from_registry()
            or self._load_from_runs()
            or self._load_from_pkl()
        )

        if not loaded:
            raise RuntimeError(
                f"No model found. Tried:\n"
                f"  1. MLflow registry ({MLFLOW_TRACKING_URI})\n"
                f"  2. MLflow run artifacts\n"
                f"  3. Local .pkl in {MODELS_DIR}\n"
                f"Run: python -m ml.pipeline --use-ray --trials 30"
            )

        self._load_threshold()
        self._load_feature_names()
        self._build_shap_explainer()

        log.info(
            "ModelBundle loaded: %s v%s | PR-AUC=%s | threshold=%.4f | "
            "%d features | SHAP=%s | source=%s",
            self.model_name,
            self.model_version or "?",
            f"{self.model_pr_auc:.4f}" if self.model_pr_auc else "?",
            self.threshold,
            len(self.feature_names),
            "enabled" if self.shap_explainer else "disabled",
            self._loaded_from,
        )
        return self

    def _load_from_registry(self) -> bool:
        """Strategy 1: MLflow Model Registry."""
        client = _setup_mlflow_client()
        if client is None:
            return False

        try:
            versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
        except Exception as e:
            log.debug("Registry search failed: %s", e)
            return False

        if not versions:
            log.info("No registered model '%s' found", REGISTERED_MODEL_NAME)
            return False

        # Find version with highest test_pr_auc
        best_version   = None
        best_pr_auc    = -1.0

        for v in versions:
            try:
                run    = client.get_run(v.run_id)
                pr_auc = float(run.data.metrics.get("test_pr_auc", 0))
                if pr_auc > best_pr_auc:
                    best_pr_auc  = pr_auc
                    best_version = v
            except Exception:
                continue

        if best_version is None:
            return False

        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{best_version.version}"
        log.info(
            "Loading from registry: %s (version %s, PR-AUC=%.4f)",
            model_uri, best_version.version, best_pr_auc
        )

        try:
            # Try sklearn flavour first (XGBoost/LightGBM/CatBoost registered this way)
            self.model = mlflow.sklearn.load_model(model_uri)
            log.info("  Loaded sklearn flavour")
        except Exception:
            try:
                # Fallback: pyfunc flavour (works for any model type)
                pyfunc_model  = mlflow.pyfunc.load_model(model_uri)
                self.model    = pyfunc_model._model_impl.sklearn_model
                log.info("  Loaded pyfunc flavour")
            except Exception as e:
                log.warning("  Both flavours failed: %s", e)
                return False

        self.model_version = str(best_version.version)
        self.model_pr_auc  = best_pr_auc
        self.model_name    = best_version.tags.get("model_type", REGISTERED_MODEL_NAME)
        self._loaded_from  = f"mlflow_registry:{MLFLOW_TRACKING_URI}"
        return True

    def _load_from_runs(self) -> bool:
        """Strategy 2: Best MLflow run artifact (no registry required)."""
        client = _setup_mlflow_client()
        if client is None:
            return False

        try:
            experiments = client.search_experiments()
        except Exception:
            return False

        best_run    = None
        best_pr_auc = -1.0

        for exp in experiments:
            try:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="metrics.test_pr_auc > 0",
                    order_by=["metrics.test_pr_auc DESC"],
                    max_results=5,
                )
                for run in runs:
                    pr_auc = float(run.data.metrics.get("test_pr_auc", 0))
                    if pr_auc > best_pr_auc:
                        best_pr_auc = pr_auc
                        best_run    = run
            except Exception:
                continue

        if best_run is None:
            log.info("No MLflow runs with test_pr_auc found")
            return False

        log.info(
            "Loading from run %s (PR-AUC=%.4f, experiment=%s)",
            best_run.info.run_id[:8], best_pr_auc,
            best_run.info.experiment_id,
        )

        # Try loading model artifact from the run
        for artifact_path in ["model", "xgboost_model", "catboost_model",
                               "lightgbm_model", "logistic_model"]:
            try:
                model_uri  = f"runs:/{best_run.info.run_id}/{artifact_path}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_name   = artifact_path
                self.model_pr_auc = best_pr_auc
                self._loaded_from = f"mlflow_run:{best_run.info.run_id[:8]}"
                log.info("  Loaded artifact: %s", artifact_path)
                return True
            except Exception:
                continue

        return False

    def _load_from_pkl(self) -> bool:
        """Strategy 3: Local .pkl files — legacy fallback."""
        for fname in CHAMPION_PRIORITY:
            path = MODELS_DIR / fname
            if path.exists():
                try:
                    self.model        = joblib.load(path)
                    self.model_name   = fname.replace("_model.pkl", "")
                    self._loaded_from = f"pkl:{path}"
                    log.info("Loaded from pkl: %s", path)
                    return True
                except Exception as e:
                    log.warning("Failed to load %s: %s", path, e)

        # Last resort: scan for any pkl
        pkls = sorted(MODELS_DIR.glob("*.pkl"))
        if pkls:
            try:
                self.model        = joblib.load(pkls[0])
                self.model_name   = pkls[0].stem
                self._loaded_from = f"pkl_scan:{pkls[0]}"
                log.info("Loaded fallback pkl: %s", pkls[0])
                return True
            except Exception as e:
                log.warning("Fallback pkl failed: %s", e)

        return False

    def _load_threshold(self):
        """Load calibrated decision threshold from disk or MLflow tags."""
        # Try pkl file first
        thresh_file = MODELS_DIR / f"{self.model_name}_threshold.pkl"
        if thresh_file.exists():
            try:
                self.threshold = float(joblib.load(thresh_file))
                log.info("Threshold: %.4f from %s", self.threshold, thresh_file)
                return
            except Exception:
                pass

        # Try MLflow run tag
        if self._loaded_from.startswith("mlflow"):
            client = _setup_mlflow_client()
            if client:
                try:
                    # Search for threshold metric in the best run
                    runs = client.search_runs(
                        experiment_ids=["0"],
                        filter_string="metrics.optimal_threshold > 0",
                        order_by=["metrics.test_pr_auc DESC"],
                        max_results=1,
                    )
                    if runs:
                        t = runs[0].data.metrics.get("optimal_threshold")
                        if t:
                            self.threshold = float(t)
                            log.info("Threshold: %.4f from MLflow run", self.threshold)
                            return
                except Exception:
                    pass

        log.info("Threshold: %.4f (default)", self.threshold)

    def _load_feature_names(self):
        """Load feature column order from feature_names.txt."""
        feat_file = MODELS_DIR / "feature_names.txt"
        if feat_file.exists():
            with open(feat_file) as f:
                self.feature_names = [l.strip() for l in f if l.strip()]
            log.info("Feature names: %d columns", len(self.feature_names))
        else:
            # Try to get from model itself
            try:
                self.feature_names = list(self.model.feature_names_in_)
                log.info("Feature names from model: %d", len(self.feature_names))
            except AttributeError:
                log.warning(
                    "feature_names.txt not found and model has no feature_names_in_. "
                    "Training-serving alignment is disabled. Run pipeline.py."
                )

    def _build_shap_explainer(self):
        """Build SHAP explainer — TreeExplainer for trees, LinearExplainer fallback."""
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            log.info("SHAP: TreeExplainer ready")
            return
        except Exception as e:
            log.debug("TreeExplainer failed: %s", e)

        try:
            bg = pd.DataFrame(
                np.zeros((1, len(self.feature_names))),
                columns=self.feature_names or ["x"],
            )
            self.shap_explainer = shap.LinearExplainer(
                self.model, shap.maskers.Independent(bg)
            )
            log.info("SHAP: LinearExplainer ready (fallback)")
        except Exception as e:
            log.debug("LinearExplainer failed: %s", e)
            self.shap_explainer = None
            log.warning("SHAP disabled")

    def _to_dataframe(self, feature_dicts: list[dict]) -> pd.DataFrame:
        """Align incoming features to training column order. Missing cols → -1."""
        if self.feature_names:
            X = pd.DataFrame(feature_dicts).reindex(
                columns=self.feature_names, fill_value=-1
            )
        else:
            X = pd.DataFrame(feature_dicts)

        X = X.replace({True: 1, False: 0, "True": 1, "False": 0})
        X = X.infer_objects()
        X = X.select_dtypes(exclude=["object", "string", "category"])
        return X.fillna(-1).astype(np.float32)

    def predict_batch(
        self,
        feature_dicts: list[dict],
    ) -> tuple[list[float], list[list[dict]]]:
        """
        Batch inference: one model call for all items.

        Returns:
          scores:     list of float — one per request
          shap_lists: list of list[dict] — top-5 SHAP per request
                      (empty for sampled-out requests unless high risk)
        """
        X      = self._to_dataframe(feature_dicts)
        n      = len(feature_dicts)

        try:
            probas = self.model.predict_proba(X)[:, 1].tolist()
        except Exception as e:
            log.error("Batch predict_proba failed: %s", e)
            probas = [0.5] * n

        # SHAP — selective computation
        self._request_count += n
        shap_lists: list[list[dict]] = [[] for _ in range(n)]

        if self.shap_explainer is not None:
            explain_idx = [
                i for i in range(n)
                if probas[i] > 0.70                              # always high-risk
                or (i % SHAP_SAMPLE_RATE == self._request_count % SHAP_SAMPLE_RATE)
            ]

            if explain_idx:
                X_exp = X.iloc[explain_idx]
                try:
                    sv     = self.shap_explainer.shap_values(X_exp)
                    sv_pos = sv[1] if isinstance(sv, list) else sv
                    cols   = X.columns.tolist()

                    for batch_pos, orig_i in enumerate(explain_idx):
                        row_sv  = sv_pos[batch_pos]
                        top_idx = np.argsort(np.abs(row_sv))[::-1][:5]
                        shap_lists[orig_i] = [
                            {
                                "feature":    cols[j],
                                "value":      round(float(X.iloc[orig_i, j]), 4),
                                "shap_value": round(float(row_sv[j]),         4),
                            }
                            for j in top_idx
                        ]
                except Exception as e:
                    log.debug("SHAP batch failed: %s", e)

        return probas, shap_lists

    def info(self) -> dict:
        """Metadata for /model/info endpoint."""
        return {
            "model_name":    self.model_name,
            "model_version": self.model_version,
            "pr_auc":        self.model_pr_auc,
            "threshold":     self.threshold,
            "feature_count": len(self.feature_names),
            "shap_enabled":  self.shap_explainer is not None,
            "loaded_from":   self._loaded_from,
            "mlflow_uri":    MLFLOW_TRACKING_URI,
        }