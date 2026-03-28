"""
model_training.py
=================
Sentinel Fraud Detection — Core Training Module

  - Metric:    PR-AUC  — correct for 2.5% fraud rate imbalance
  - Models:    XGBoost | LightGBM | CatBoost | Logistic Regression
  - Features:  Tabular only (Phase 4) → Tabular + GNN embeddings (Phase 9)
  - Imbalance: scale_pos_weight / is_unbalance (not SMOTE)
  - CatBoost:  receives raw string categoricals (no one-hot encoding)
  - Tuning:    Optuna (sequential) or Ray Tune (parallel, --use-ray)

Ray Tune integration:
  - OptunaSearch inside Ray Tune: keeps TPE smart search, adds parallelism
  - ASHAScheduler: kills bad trials after 20% of iterations (early stopping)
  - Each trial is an isolated Ray actor: safe for CatBoost Pool, XGBoost
  - Subsample (100K rows) used during search; full data for final fit
  - MLflow logged per-run from driver, not from actors (serialisation safe)
"""

import os
import json
import joblib
import warnings
import multiprocessing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from prefect import task
except ImportError:
    def task(log_prints=True):
        def decorator(func):
            return func
        return decorator


# CONFIGURATION

class Config:
    SCRIPT_DIR    = Path(__file__).parent.absolute()
    TRAIN_PATH    = SCRIPT_DIR / "data/training/train.parquet"
    VAL_PATH      = SCRIPT_DIR / "data/training/val.parquet"
    TEST_PATH     = SCRIPT_DIR / "data/training/test.parquet"
    ARTIFACTS_DIR = SCRIPT_DIR / "models"
    CACHE_DIR     = SCRIPT_DIR / "cache"
    RAY_RESULTS   = SCRIPT_DIR / "ray_results"

    RANDOM_STATE    = 42
    EXPERIMENT_NAME = "sentinel_fraud_detection"
    REGISTRY_NAME   = "sentinel_fraud_classifier"

    S3_BUCKET = os.getenv("S3_BUCKET", "s3://sentinel-artifacts/mlflow")

    # Primary metric: PR-AUC (integrates over all thresholds — better than F2
    # for model *selection* when operating threshold is not yet decided)
    PRIMARY_METRIC = "pr_auc"

    # F2 used only for threshold optimisation at serving time:
    # at 2.5% fraud rate, missing fraud (FN) costs more than a false alarm (FP)
    BETA_SCORE = 2

    DROP_COLS: List[str] = [
        "transaction_id", "sender_hash", "receiver_hash", "identity_hash",
        "user_email_hash", "sender_wallet_hash", "receiver_wallet_hash",
        "sender_device_hash", "sender_ip_hash", "transaction_hash",
        "fraud_network_id", "cross_modality_fraud_id", "linked_fiat_transaction",
        "timestamp",
        "fraud_type",
        "source", "source_address", "source_label",
        "elliptic_feature_2", "elliptic_feature_3", "elliptic_feature_4",
        "cross_modality_pattern",
    ]

    CAT_COLS: List[str] = [
        "modality", "transfer_type", "transfer_network",
        "sender_archetype", "sender_kyc_status",
        "cryptocurrency", "blockchain", "receiver_wallet_type", "client_id",
    ]

    @classmethod
    def setup_directories(cls):
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAY_RESULTS.mkdir(parents=True, exist_ok=True)


# ENUMS & DATACLASSES

class ModelType(Enum):
    XGBOOST  = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC = "logistic_reg"


@dataclass
class ModelMetrics:
    pr_auc:                 float
    roc_auc:                float
    precision_at_90_recall: float
    threshold_at_90_recall: float
    f2_score:               float
    f1_score:               float
    precision:              float
    recall:                 float
    accuracy:               float
    false_positive_rate:    float
    false_negative_rate:    float

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (
            f"PR-AUC: {self.pr_auc:.4f} | ROC-AUC: {self.roc_auc:.4f} | "
            f"P@90%R: {self.precision_at_90_recall:.4f} | "
            f"F2: {self.f2_score:.4f} | FPR: {self.false_positive_rate:.4f}"
        )


@dataclass
class TrainingData:
    X_train:     pd.DataFrame
    X_val:       pd.DataFrame
    X_test:      pd.DataFrame
    X_train_cat: pd.DataFrame
    X_val_cat:   pd.DataFrame
    X_test_cat:  pd.DataFrame
    y_train:     pd.Series
    y_val:       pd.Series
    y_test:      pd.Series
    scale_pos_weight:  float
    cat_feature_names: List[str]
    feature_names:     List[str]

    def get_shapes_summary(self) -> Dict[str, Any]:
        return {
            "train_size":         len(self.X_train),
            "val_size":           len(self.X_val),
            "test_size":          len(self.X_test),
            "n_features_encoded": len(self.feature_names),
            "n_features_raw":     self.X_train_cat.shape[1],
            "n_cat_features":     len(self.cat_feature_names),
            "scale_pos_weight":   self.scale_pos_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATOR
# ─────────────────────────────────────────────────────────────────────────────

class DataPreparator:

    def __init__(self, config: Config, use_gnn_embeddings: bool = False):
        self.config = config
        self.use_gnn_embeddings = use_gnn_embeddings

    @task(log_prints=True)
    def prepare_data(self) -> TrainingData:
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        X_train_raw, y_train = self._load_split(self.config.TRAIN_PATH)
        X_val_raw,   y_val   = self._load_split(self.config.VAL_PATH)
        X_test_raw,  y_test  = self._load_split(self.config.TEST_PATH)

        (X_train_enc, X_val_enc, X_test_enc,
         X_train_cat, X_val_cat, X_test_cat,
         cat_present) = self._prepare_features(X_train_raw, X_val_raw, X_test_raw)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        data = TrainingData(
            X_train=X_train_enc, X_val=X_val_enc, X_test=X_test_enc,
            X_train_cat=X_train_cat, X_val_cat=X_val_cat, X_test_cat=X_test_cat,
            y_train=y_train, y_val=y_val, y_test=y_test,
            scale_pos_weight=n_neg / n_pos,
            cat_feature_names=cat_present,
            feature_names=X_train_enc.columns.tolist(),
        )
        self._print_summary(data)
        return data

    def _load_split(self, path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/merge_training_data.py"
            )
        df = pd.read_parquet(path)
        print(f"  Loaded {path.name}: {len(df):,} rows  "
              f"(fraud: {df['is_fraud'].sum():,} = {df['is_fraud'].mean()*100:.2f}%)")
        y = df["is_fraud"].astype(int)
        X = df.drop(
            columns=["is_fraud"] + [c for c in self.config.DROP_COLS if c in df.columns],
            errors="ignore",
        )
        return X, y

    def _prepare_features(self, X_train, X_val, X_test):
        gnn_cols = [c for c in X_train.columns if c.startswith("gnn_emb_")]
        if not self.use_gnn_embeddings and gnn_cols:
            print(f"  [features] Dropping {len(gnn_cols)} GNN cols")
            for df in [X_train, X_val, X_test]:
                df.drop(columns=gnn_cols, inplace=True, errors="ignore")

        cat_present = [c for c in self.config.CAT_COLS if c in X_train.columns]

        def _make_cat_version(df):
            out = df.copy()
            for col in cat_present:
                if col in out.columns:
                    out[col] = out[col].fillna("missing").astype(str)
            return out.fillna(-1)

        X_train_cat = _make_cat_version(X_train)
        X_val_cat   = _make_cat_version(X_val)
        X_test_cat  = _make_cat_version(X_test)

        if cat_present:
            X_train = pd.get_dummies(X_train, columns=cat_present, dtype=float)
            X_val   = pd.get_dummies(X_val,   columns=[c for c in cat_present if c in X_val.columns],  dtype=float)
            X_test  = pd.get_dummies(X_test,  columns=[c for c in cat_present if c in X_test.columns], dtype=float)

        X_val   = X_val.reindex(columns=X_train.columns,  fill_value=0)
        X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train = X_train.fillna(-1)
        X_val   = X_val.fillna(-1)
        X_test  = X_test.fillna(-1)

        print(f"  [features] Encoded: {X_train.shape[1]} cols | "
              f"Raw: {X_train_cat.shape[1]} cols | "
              f"Categoricals: {len(cat_present)}")

        return X_train, X_val, X_test, X_train_cat, X_val_cat, X_test_cat, cat_present

    def _print_summary(self, data: TrainingData):
        s = data.get_shapes_summary()
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE")
        print("=" * 60)
        print(f"  Train: {s['train_size']:>9,}  |  Val: {s['val_size']:>7,}  |  Test: {s['test_size']:>7,}")
        print(f"  Features (encoded): {s['n_features_encoded']}")
        print(f"  scale_pos_weight:   {s['scale_pos_weight']:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:

    @staticmethod
    def calculate(y_true, y_prob, threshold=0.5) -> ModelMetrics:
        y_pred = (y_prob >= threshold).astype(int)
        pr_auc  = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
        idx_90 = np.searchsorted(recalls[::-1], 0.90)
        p_at_90 = float(precisions[::-1][idx_90]) if idx_90 < len(precisions) else 0.0
        t_at_90 = float(thresholds_pr[::-1][idx_90]) if idx_90 < len(thresholds_pr) else threshold

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        return ModelMetrics(
            pr_auc=round(float(pr_auc), 4),
            roc_auc=round(float(roc_auc), 4),
            precision_at_90_recall=round(p_at_90, 4),
            threshold_at_90_recall=round(t_at_90, 4),
            f2_score=round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
            f1_score=round(float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)), 4),
            precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            false_positive_rate=round(fp / (fp + tn + 1e-9), 4),
            false_negative_rate=round(fn / (fn + tp + 1e-9), 4),
        )

    @staticmethod
    def optimize_threshold(model, X_val, y_val, beta=2, model_type=None):
        y_prob     = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.05, 0.95, 0.01)
        scores     = [fbeta_score(y_val, (y_prob >= t).astype(int),
                                  beta=beta, zero_division=0) for t in thresholds]
        best_idx   = int(np.argmax(scores))
        return float(thresholds[best_idx]), float(scores[best_idx])

    @staticmethod
    def log_confusion_matrix(y_true, y_pred, model_name, save_dir):
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title(f"{model_name} — Confusion Matrix")
        path = save_dir / f"confusion_matrix_{model_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def log_feature_importance(model, model_name, feature_names, save_dir, top_n=20):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
        else:
            return None
        fi = (pd.DataFrame({"feature": feature_names, "importance": importances})
              .sort_values("importance", ascending=False).head(top_n))
        plt.figure(figsize=(10, 6))
        plt.barh(fi["feature"][::-1], fi["importance"][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Features — {model_name}")
        plt.tight_layout()
        plot_path = save_dir / f"feature_importance_{model_name}.png"
        csv_path  = save_dir / f"feature_importance_{model_name}.csv"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        fi.to_csv(csv_path, index=False)
        return plot_path, csv_path

    @staticmethod
    def run_shap(model, X_val, model_type, save_dir, n_samples=2000):
        try:
            sample = X_val.sample(min(n_samples, len(X_val)), random_state=42)
            if model_type == ModelType.LOGISTIC:
                clf    = model.named_steps["clf"]
                scaler = model.named_steps["scaler"]
                X_s    = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
                sv     = shap.LinearExplainer(clf, X_s).shap_values(X_s)
            else:
                sv = shap.TreeExplainer(model).shap_values(sample)
                if isinstance(sv, list):
                    sv = sv[1]
            fig, _ = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv, sample, max_display=20, show=False, plot_type="bar")
            plt.title(f"SHAP Feature Importance — {model_type.value}")
            plt.tight_layout()
            path = save_dir / f"shap_{model_type.value}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            return path
        except Exception as e:
            print(f"  [SHAP] Skipped: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZER — OPTUNA (sequential)
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterOptimizer:
    """Sequential Optuna search. Used when --use-ray is not set."""

    SAMPLE_SIZE = 100_000   # rows used per trial — full data used only for final fit

    def __init__(self, config: Config):
        self.config = config
        self._best: Dict[str, float] = {}
        self._global_best       = 0.0
        self._global_best_model = ""

    def _log_trial(self, model_name: str, trial_num: int, pr_auc: float):
        prev_best      = self._best.get(model_name, 0.0)
        is_model_best  = pr_auc > prev_best
        is_global_best = pr_auc > self._global_best
        if is_model_best:
            self._best[model_name] = pr_auc
        if is_global_best:
            self._global_best       = pr_auc
            self._global_best_model = model_name
        print(
            f"  [{model_name:<12}] trial {trial_num:>3} | "
            f"PR-AUC: {pr_auc:.4f} | "
            f"best({model_name}): {self._best[model_name]:.4f} | "
            f"global: {self._global_best:.4f} [{self._global_best_model}]"
            + (" ★" if is_model_best else "")
            + (" 🏆" if is_global_best else "")
        )

    def _subsample(self, X, y):
        n = min(self.SAMPLE_SIZE, len(X))
        idx = np.random.RandomState(self.config.RANDOM_STATE).choice(len(X), n, replace=False)
        X_s = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        y_s = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        return X_s, y_s

    # ── one objective per model ───────────────────────────────────────────────

    def xgboost_objective(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        X_s, y_s = self._subsample(X_train, y_train)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":          trial.suggest_int("n_estimators",       200, 1200),
                "max_depth":             trial.suggest_int("max_depth",           3, 10),
                "learning_rate":         trial.suggest_float("learning_rate",     0.01, 0.3,  log=True),
                "subsample":             trial.suggest_float("subsample",         0.5, 1.0),
                "colsample_bytree":      trial.suggest_float("colsample_bytree",  0.5, 1.0),
                "min_child_weight":      trial.suggest_int("min_child_weight",    1, 10),
                "gamma":                 trial.suggest_float("gamma",             0.0, 5.0),
                "reg_alpha":             trial.suggest_float("reg_alpha",         1e-8, 10.0, log=True),
                "reg_lambda":            trial.suggest_float("reg_lambda",        1e-8, 10.0, log=True),
                "scale_pos_weight":      scale_pos_weight,
                "eval_metric":           "aucpr",
                "early_stopping_rounds": 30,
                "use_label_encoder":     False,
                "random_state":          self.config.RANDOM_STATE,
                "n_jobs":                -1,
                "verbosity":             0,
            }
            model = XGBClassifier(**params)
            model.fit(X_s.values, y_s.values,
                      eval_set=[(X_val.values, y_val.values)], verbose=False)
            pr_auc = average_precision_score(y_val, model.predict_proba(X_val.values)[:, 1])
            mlflow.log_metric("trial_pr_auc", pr_auc)
            self._log_trial("xgboost", trial.number, pr_auc)
            return -pr_auc
        return objective

    def lightgbm_objective(self, X_train, y_train, X_val, y_val):
        X_s, y_s = self._subsample(X_train, y_train)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators",      200, 1200),
                "max_depth":         trial.suggest_int("max_depth",          3, 12),
                "learning_rate":     trial.suggest_float("learning_rate",    0.01, 0.3,  log=True),
                "num_leaves":        trial.suggest_int("num_leaves",         20, 300),
                "subsample":         trial.suggest_float("subsample",        0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples",  5, 100),
                "reg_alpha":         trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
                "is_unbalance":      True,
                "metric":            "average_precision",
                "random_state":      self.config.RANDOM_STATE,
                "n_jobs":            -1,
                "verbosity":         -1,
            }
            model = LGBMClassifier(**params)
            model.fit(X_s, y_s, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)])
            pr_auc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
            mlflow.log_metric("trial_pr_auc", pr_auc)
            self._log_trial("lightgbm", trial.number, pr_auc)
            return -pr_auc
        return objective

    def catboost_objective(self, X_train, y_train, X_val, y_val, cat_features):
        cat_indices = [X_train.columns.tolist().index(c)
                       for c in cat_features if c in X_train.columns]
        X_s, y_s = self._subsample(X_train, y_train)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations":           trial.suggest_int("iterations",            200, 1200),
                "depth":                trial.suggest_int("depth",                 4, 10),
                "learning_rate":        trial.suggest_float("learning_rate",       0.01, 0.3,  log=True),
                "l2_leaf_reg":          trial.suggest_float("l2_leaf_reg",         1e-8, 10.0, log=True),
                "border_count":         trial.suggest_int("border_count",          32, 255),
                "bagging_temperature":  trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "auto_class_weights":   "Balanced",
                "eval_metric":          "AUC",
                "cat_features":         cat_indices,
                "random_seed":          self.config.RANDOM_STATE,
                "verbose":              False,
                "allow_writing_files":  False,
            }
            model = CatBoostClassifier(**params)
            model.fit(Pool(X_s, y_s, cat_features=cat_indices),
                      eval_set=Pool(X_val, y_val, cat_features=cat_indices),
                      early_stopping_rounds=30, verbose=False)
            pr_auc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
            mlflow.log_metric("trial_pr_auc", pr_auc)
            self._log_trial("catboost", trial.number, pr_auc)
            return -pr_auc
        return objective

    def logistic_objective(self, X_train, y_train, X_val, y_val):
        X_s, y_s = self._subsample(X_train, y_train)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "C":            trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty":      trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver":       "saga",
                "class_weight": "balanced",
                "max_iter":     1000,
                "random_state": self.config.RANDOM_STATE,
                "n_jobs":       -1,
            }
            model = Pipeline([("scaler", StandardScaler()),
                               ("clf",   LogisticRegression(**params))])
            model.fit(X_s.values, y_s.values)
            pr_auc = average_precision_score(y_val, model.predict_proba(X_val.values)[:, 1])
            mlflow.log_metric("trial_pr_auc", pr_auc)
            self._log_trial("logistic", trial.number, pr_auc)
            return -pr_auc
        return objective


# ─────────────────────────────────────────────────────────────────────────────
# RAY TUNE OPTIMIZER  (parallel — used when --use-ray is set)
# ─────────────────────────────────────────────────────────────────────────────

# Search spaces defined at module level so Ray actors can import them
# without pickling closures over large DataFrames.
_RAY_SEARCH_SPACES = {
    ModelType.XGBOOST: {
        "n_estimators":     ("randint",    200, 1200),
        "max_depth":        ("randint",    3,   10),
        "learning_rate":    ("loguniform", 0.01, 0.3),
        "subsample":        ("uniform",    0.5,  1.0),
        "colsample_bytree": ("uniform",    0.5,  1.0),
        "min_child_weight": ("randint",    1,    10),
        "gamma":            ("uniform",    0.0,  5.0),
        "reg_alpha":        ("loguniform", 1e-8, 10.0),
        "reg_lambda":       ("loguniform", 1e-8, 10.0),
    },
    ModelType.LIGHTGBM: {
        "n_estimators":      ("randint",    200, 1200),
        "max_depth":         ("randint",    3,   12),
        "learning_rate":     ("loguniform", 0.01, 0.3),
        "num_leaves":        ("randint",    20,   300),
        "subsample":         ("uniform",    0.5,  1.0),
        "colsample_bytree":  ("uniform",    0.5,  1.0),
        "min_child_samples": ("randint",    5,    100),
        "reg_alpha":         ("loguniform", 1e-8, 10.0),
        "reg_lambda":        ("loguniform", 1e-8, 10.0),
    },
    ModelType.CATBOOST: {
        "iterations":          ("randint",  200, 1200),
        "depth":               ("randint",  4,   10),
        "learning_rate":       ("loguniform", 0.01, 0.3),
        "l2_leaf_reg":         ("loguniform", 1e-8, 10.0),
        "border_count":        ("randint",  32,  255),
        "bagging_temperature": ("uniform",  0.0, 1.0),
    },
    ModelType.LOGISTIC: {
        "C":       ("loguniform", 1e-4, 100.0),
        "penalty": ("choice",     ["l1", "l2"]),
    },
}


def _build_ray_search_space(model_type: ModelType) -> dict:
    """Convert the abstract search space spec into Ray Tune samplers."""
    from ray import tune
    space = {}
    for name, spec in _RAY_SEARCH_SPACES[model_type].items():
        kind = spec[0]
        if kind == "randint":
            space[name] = tune.randint(spec[1], spec[2])
        elif kind == "uniform":
            space[name] = tune.uniform(spec[1], spec[2])
        elif kind == "loguniform":
            space[name] = tune.loguniform(spec[1], spec[2])
        elif kind == "choice":
            space[name] = tune.choice(spec[1])
    return space


def _make_ray_trainable(
    model_type: ModelType,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np:   np.ndarray,
    y_val_np:   np.ndarray,
    cat_indices: List[int],
    scale_pos_weight: float,
    random_state: int,
    sample_size: int,
    X_train_cols: List[str],  # column names for CatBoost DataFrame reconstruction
):
    """
    Returns a Ray Tune trainable function.

    Design decisions:
    - Numpy arrays (not DataFrames) are passed to avoid pickling column metadata
      across Ray actors repeatedly. DataFrames are rebuilt inside the actor when
      CatBoost needs them.
    - Subsample is done inside the actor so each trial trains on a random 100K
      subset, not always the same rows.
    - No MLflow calls inside actors — MLflow tracking is not reliable across
      separate processes. We log per-trial results from the driver after the study.
    - early_stopping_rounds kept low (20) inside tuning — final fit uses 50.
    """
    from ray import tune as ray_tune

    def trainable(config):
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score

        # Subsample for speed — each trial sees a different 100K-row slice
        rng = np.random.default_rng(abs(hash(str(config))) % (2**32))
        n   = min(sample_size, len(X_train_np))
        idx = rng.choice(len(X_train_np), n, replace=False)
        X_s = X_train_np[idx]
        y_s = y_train_np[idx]

        try:
            if model_type == ModelType.XGBOOST:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    **config,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    early_stopping_rounds=20,
                    use_label_encoder=False,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(X_s, y_s, eval_set=[(X_val_np, y_val_np)], verbose=False)
                y_prob = model.predict_proba(X_val_np)[:, 1]

            elif model_type == ModelType.LIGHTGBM:
                from lightgbm import LGBMClassifier, early_stopping, log_evaluation
                model = LGBMClassifier(
                    **config,
                    is_unbalance=True,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(
                    X_s, y_s,
                    eval_set=[(X_val_np, y_val_np)],
                    callbacks=[early_stopping(20, verbose=False), log_evaluation(-1)],
                )
                y_prob = model.predict_proba(X_val_np)[:, 1]

            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostClassifier, Pool
                import pandas as pd
                # Rebuild DataFrames — CatBoost needs string categoricals
                X_s_df  = pd.DataFrame(X_s,      columns=X_train_cols)
                X_v_df  = pd.DataFrame(X_val_np, columns=X_train_cols)
                model = CatBoostClassifier(
                    **config,
                    auto_class_weights="Balanced",
                    cat_features=cat_indices,
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    Pool(X_s_df,  y_s,      cat_features=cat_indices),
                    eval_set=Pool(X_v_df, y_val_np, cat_features=cat_indices),
                    early_stopping_rounds=20,
                    verbose=False,
                )
                y_prob = model.predict_proba(X_v_df)[:, 1]

            else:  # LOGISTIC
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    LogisticRegression(
                        C=config["C"],
                        penalty=config["penalty"],
                        solver="saga",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                        n_jobs=-1,
                    )),
                ])
                model.fit(X_s, y_s)
                y_prob = model.predict_proba(X_val_np)[:, 1]

            pr_auc = float(average_precision_score(y_val_np, y_prob))

        except Exception as e:
            pr_auc = 0.0   # treat failed trials as worst-case

        ray_tune.report({"pr_auc": pr_auc})

    return trainable


class RayTuneOptimizer:
    """
    Parallel hyperparameter search via Ray Tune.
    Internally uses OptunaSearch (TPE sampler) for smart search
    plus ASHAScheduler for early killing of bad trials.

    Parallelism: n_cpus_total / n_cpus_per_trial concurrent trials.
    Example: 8-core MacBook, n_cpus_per_trial=2 → 4 concurrent trials.
    """

    SAMPLE_SIZE = 100_000

    def __init__(self, config: Config, n_cpus_per_trial: int = 2):
        self.config           = config
        self.n_cpus_per_trial = n_cpus_per_trial

    def tune(
        self,
        model_type:       ModelType,
        data:             "TrainingData",
        n_trials:         int,
    ) -> Tuple[dict, float]:
        """
        Returns (best_params, best_val_pr_auc).
        best_params is a plain dict ready to pass to _build_model().
        """
        from ray import tune, available_resources
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.schedulers import ASHAScheduler

        # Select correct feature set
        if model_type == ModelType.CATBOOST:
            X_train = data.X_train_cat
            X_val   = data.X_val_cat
            cat_indices = [data.X_train_cat.columns.tolist().index(c)
                           for c in data.cat_feature_names
                           if c in data.X_train_cat.columns]
            col_names = data.X_train_cat.columns.tolist()
        else:
            X_train = data.X_train
            X_val   = data.X_val
            cat_indices = []
            col_names   = []

        # Convert to numpy once — passed to every Ray actor
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = data.y_train.values.astype(np.int32)
        X_val_np   = X_val.values.astype(np.float32)
        y_val_np   = data.y_val.values.astype(np.int32)

        search_space = _build_ray_search_space(model_type)
        trainable_fn = _make_ray_trainable(
            model_type       = model_type,
            X_train_np       = X_train_np,
            y_train_np       = y_train_np,
            X_val_np         = X_val_np,
            y_val_np         = y_val_np,
            cat_indices      = cat_indices,
            scale_pos_weight = data.scale_pos_weight,
            random_state     = self.config.RANDOM_STATE,
            sample_size      = self.SAMPLE_SIZE,
            X_train_cols     = col_names,
        )

        n_cpus_available = int(available_resources().get("CPU", 4))
        n_concurrent     = max(1, n_cpus_available // self.n_cpus_per_trial)

        print(f"  [Ray Tune] {model_type.value}: {n_trials} trials, "
              f"{n_concurrent} concurrent "
              f"({self.n_cpus_per_trial} CPUs/trial, "
              f"{n_cpus_available} total CPUs)")

        # ASHA kills bad trials after grace_period iterations,
        # halving the survivor set each reduction_factor rounds.
        # This saves ~40-60% of compute vs running all trials to completion.
        scheduler = ASHAScheduler(
            metric="pr_auc",
            mode="max",
            max_t=10,           # 10 "time units" — one report per trial
            grace_period=2,     # keep alive for at least 2 units
            reduction_factor=2,
        )

        search_alg = OptunaSearch(
            metric="pr_auc",
            mode="max",
            seed=self.config.RANDOM_STATE,
        )

        analysis = tune.run(
            trainable_fn,
            config=search_space,
            num_samples=n_trials,
            search_alg=search_alg,
            scheduler=scheduler,
            resources_per_trial={"cpu": self.n_cpus_per_trial},
            verbose=1,
            storage_path=str(self.config.RAY_RESULTS.absolute()),
            name=f"sentinel_{model_type.value}",
            raise_on_failed_trial=False,
        )

        best_trial   = analysis.get_best_trial(metric="pr_auc", mode="max")
        best_params  = best_trial.config
        best_pr_auc  = best_trial.last_result.get("pr_auc", 0.0)

        # Print top-5 trials
        df = analysis.results_df.sort_values("pr_auc", ascending=False).head(5)
        print(f"\n  [Ray Tune] Top-5 trials for {model_type.value}:")
        print(f"  {'Trial':>6}  {'PR-AUC':>8}")
        for _, row in df.iterrows():
            print(f"  {str(row.get('trial_id','?'))[:6]:>6}  {row.get('pr_auc', 0):.4f}")

        print(f"\n  Best val PR-AUC: {best_pr_auc:.4f}")
        print(f"  Best params:     {best_params}")

        return best_params, best_pr_auc


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:

    def __init__(
        self,
        config:           Config,
        use_ray:          bool = False,
        n_cpus_per_trial: int  = 2,
    ):
        self.config           = config
        self.use_ray          = use_ray
        self.optuna_optimizer = HyperparameterOptimizer(config)
        self.ray_optimizer    = RayTuneOptimizer(config, n_cpus_per_trial) if use_ray else None
        self.metrics          = MetricsCalculator()

    @task(log_prints=True)
    def train_model(
        self,
        model_type: ModelType,
        data:       TrainingData,
        n_trials:   int = 50,
    ) -> Tuple[str, str, float]:

        print(f"\n{'='*60}")
        print(f"  {model_type.value.upper()}  —  {n_trials} trials  "
              f"[{'Ray Tune (parallel)' if self.use_ray else 'Optuna (sequential)'}]")
        print(f"{'='*60}")

        # ── Hyperparameter search ─────────────────────────────────────────────
        if self.use_ray:
            best_params, best_val_pr_auc = self.ray_optimizer.tune(
                model_type, data, n_trials
            )
        else:
            best_params, best_val_pr_auc = self._tune_optuna(model_type, data, n_trials)

        # ── Final fit on full train+val ────────────────────────────────────────
        run_id, run_uuid, test_pr_auc = self._train_final_model(
            model_type, best_params, data
        )
        return run_id, run_uuid, test_pr_auc

    # ── Sequential Optuna search ──────────────────────────────────────────────

    def _tune_optuna(
        self,
        model_type: ModelType,
        data:       TrainingData,
        n_trials:   int,
    ) -> Tuple[dict, float]:
        from optuna.integration.mlflow import MLflowCallback

        if model_type == ModelType.CATBOOST:
            X_tr, X_v = data.X_train_cat, data.X_val_cat
        else:
            X_tr, X_v = data.X_train, data.X_val

        self.optuna_optimizer._best[model_type.value] = 0.0
        objective_fn = self._get_objective(model_type, X_tr, data.y_train,
                                           X_v, data.y_val, data)

        study = optuna.create_study(
            direction="minimize",
            study_name=f"sentinel_{model_type.value}",
            sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

        mlflow_cb = MLflowCallback(
            metric_name="trial_pr_auc",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )

        study.optimize(
            objective_fn,
            n_trials=n_trials,
            callbacks=[mlflow_cb],
            show_progress_bar=False,
        )

        if mlflow.active_run():
            mlflow.end_run()

        best_val_pr_auc = -study.best_value
        print(f"\n  Best val PR-AUC: {best_val_pr_auc:.4f}")
        return study.best_params, best_val_pr_auc

    def _get_objective(self, model_type, X_tr, y_tr, X_v, y_v, data):
        opt = self.optuna_optimizer
        if model_type == ModelType.XGBOOST:
            return opt.xgboost_objective(X_tr, y_tr, X_v, y_v, data.scale_pos_weight)
        elif model_type == ModelType.LIGHTGBM:
            return opt.lightgbm_objective(X_tr, y_tr, X_v, y_v)
        elif model_type == ModelType.CATBOOST:
            return opt.catboost_objective(X_tr, y_tr, X_v, y_v, data.cat_feature_names)
        else:
            return opt.logistic_objective(X_tr, y_tr, X_v, y_v)

    # ── Final model: retrain on train+val, evaluate on test ──────────────────

    def _train_final_model(
        self,
        model_type:  ModelType,
        best_params: dict,
        data:        TrainingData,
    ) -> Tuple[str, str, float]:

        print(f"  Retraining {model_type.value} on full train+val...")

        if model_type == ModelType.CATBOOST:
            X_tr_full = pd.concat([data.X_train_cat, data.X_val_cat])
            X_v       = data.X_val_cat
            X_te      = data.X_test_cat
        else:
            X_tr_full = pd.concat([data.X_train, data.X_val])
            X_v       = data.X_val
            X_te      = data.X_test

        y_tr_full = pd.concat([data.y_train, data.y_val])

        with mlflow.start_run(run_name=f"{model_type.value}_best") as run:
            model = self._build_model(model_type, best_params, data)
            self._fit_model(model_type, model, X_tr_full, y_tr_full,
                            X_v, data.y_val, data)

            opt_threshold, _ = self.metrics.optimize_threshold(
                model, X_v, data.y_val.values,
                beta=self.config.BETA_SCORE, model_type=model_type,
            )

            y_prob  = model.predict_proba(X_te)[:, 1]
            test_m  = self.metrics.calculate(data.y_test.values, y_prob, opt_threshold)
            print(f"  Test: {test_m}")

            self._log_run(model_type, model, best_params, opt_threshold,
                          test_m, data, X_v, X_te)

            run_id   = run.info.run_id
            run_uuid = run.info.artifact_uri.split("/")[-2]

        return run_id, run_uuid, test_m.pr_auc

    def _build_model(self, model_type: ModelType, params: dict, data: TrainingData):
        p = params.copy()

        if model_type == ModelType.XGBOOST:
            p.update({
                "scale_pos_weight":  data.scale_pos_weight,
                "use_label_encoder": False,
                "random_state":      self.config.RANDOM_STATE,
                "n_jobs":            -1,
                "verbosity":         0,
            })
            p.pop("early_stopping_rounds", None)
            return XGBClassifier(**p)

        elif model_type == ModelType.LIGHTGBM:
            p.update({
                "is_unbalance": True,
                "random_state": self.config.RANDOM_STATE,
                "n_jobs":       -1,
                "verbosity":    -1,
            })
            return LGBMClassifier(**p)

        elif model_type == ModelType.CATBOOST:
            cat_indices = [data.X_train_cat.columns.tolist().index(c)
                           for c in data.cat_feature_names
                           if c in data.X_train_cat.columns]
            p.update({
                "auto_class_weights": "Balanced",
                "cat_features":       cat_indices,
                "random_seed":        self.config.RANDOM_STATE,
                "verbose":            False,
                "allow_writing_files": False,
            })
            return CatBoostClassifier(**p)

        else:
            p.update({
                "solver":       "saga",
                "class_weight": "balanced",
                "max_iter":     1000,
                "random_state": self.config.RANDOM_STATE,
                "n_jobs":       -1,
            })
            return Pipeline([("scaler", StandardScaler()),
                              ("clf",   LogisticRegression(**p))])

    def _fit_model(self, model_type, model, X_tr, y_tr, X_v, y_v, data):
        if model_type == ModelType.XGBOOST:
            model.fit(X_tr.values, y_tr.values,
                      eval_set=[(X_v.values, y_v.values)], verbose=False)
        elif model_type == ModelType.LIGHTGBM:
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
        elif model_type == ModelType.CATBOOST:
            cat_indices = [data.X_train_cat.columns.tolist().index(c)
                           for c in data.cat_feature_names
                           if c in data.X_train_cat.columns]
            model.fit(Pool(X_tr, y_tr, cat_features=cat_indices),
                      eval_set=Pool(X_v, y_v, cat_features=cat_indices),
                      early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_tr.values, y_tr.values)

    def _log_run(self, model_type, model, params, threshold, metrics, data, X_v, X_te):
        mlflow.log_params({**params,
                           "model_type":       model_type.value,
                           "optimal_threshold": threshold,
                           "use_ray":           str(self.use_ray)})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.to_dict().items()})

        y_pred = (model.predict_proba(X_te)[:, 1] >= threshold).astype(int)
        cm_path = self.metrics.log_confusion_matrix(
            data.y_test.values, y_pred, model_type.value, self.config.ARTIFACTS_DIR)
        mlflow.log_artifact(str(cm_path))

        feat_names = (data.feature_names if model_type != ModelType.CATBOOST
                      else data.X_train_cat.columns.tolist())
        fi = self.metrics.log_feature_importance(
            model, model_type.value, feat_names, self.config.ARTIFACTS_DIR)
        if fi:
            mlflow.log_artifact(str(fi[0]))
            mlflow.log_artifact(str(fi[1]))

        X_for_shap = X_v if model_type != ModelType.CATBOOST else data.X_val_cat
        shap_path  = self.metrics.run_shap(model, X_for_shap, model_type,
                                           self.config.ARTIFACTS_DIR)
        if shap_path:
            mlflow.log_artifact(str(shap_path), artifact_path="shap")

        model_path     = self.config.ARTIFACTS_DIR / f"{model_type.value}_model.pkl"
        threshold_path = self.config.ARTIFACTS_DIR / f"{model_type.value}_threshold.pkl"
        joblib.dump(model,     model_path)
        joblib.dump(threshold, threshold_path)

        if model_type == ModelType.XGBOOST:
            mlflow.xgboost.log_model(model, "model",
                                     registered_model_name=self.config.REGISTRY_NAME)
        elif model_type == ModelType.LIGHTGBM:
            mlflow.lightgbm.log_model(model, "model",
                                      registered_model_name=self.config.REGISTRY_NAME)
        else:
            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name=self.config.REGISTRY_NAME)

        mlflow.set_tag("model_type", model_type.value)
        mlflow.set_tag("pr_auc",     str(metrics.pr_auc))
        mlflow.set_tag("use_ray",    str(self.use_ray))

        feat_path = self.config.ARTIFACTS_DIR / "feature_names.txt"
        with open(feat_path, "w") as f:
            f.write("\n".join(feat_names))


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW SERVING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SentinelFraudWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model     = joblib.load(context.artifacts["model_path"])
        self.threshold = joblib.load(context.artifacts["threshold_path"])
        with open(context.artifacts["feature_names_path"]) as f:
            self.feature_names = [l.strip() for l in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame):
        X       = model_input.reindex(columns=self.feature_names, fill_value=-1)
        y_prob  = self.model.predict_proba(X)[:, 1]
        y_pred  = (y_prob >= self.threshold).astype(int)
        return pd.DataFrame({
            "fraud_probability": y_prob,
            "decision":          ["block" if p else "approve" for p in y_pred],
        })


# ─────────────────────────────────────────────────────────────────────────────
# TOURNAMENT
# ─────────────────────────────────────────────────────────────────────────────

@task(log_prints=True)
def run_all_experiments(
    data:             TrainingData,
    n_trials:         int = 50,
    models_to_train:  Optional[List[ModelType]] = None,
    use_ray:          bool = False,
    n_cpus_per_trial: int  = 2,
) -> Tuple[str, str, float, str]:

    if models_to_train is None:
        models_to_train = list(ModelType)

    print("\n" + "=" * 60)
    print("  SENTINEL MODEL TOURNAMENT")
    print(f"  Models:  {[m.value for m in models_to_train]}")
    print(f"  Trials:  {n_trials} per model")
    print(f"  Search:  {'Ray Tune (parallel)' if use_ray else 'Optuna (sequential)'}")
    if use_ray:
        print(f"  CPUs/trial: {n_cpus_per_trial}")
    print("=" * 60)

    config  = Config()
    trainer = ModelTrainer(config, use_ray=use_ray, n_cpus_per_trial=n_cpus_per_trial)
    results = []

    for model_type in models_to_train:
        try:
            run_id, run_uuid, pr_auc = trainer.train_model(
                model_type, data, n_trials)
            results.append((model_type.value, run_id, run_uuid, pr_auc))

            print(f"\n  ── LEADERBOARD (after {model_type.value}) ──")
            print(f"  {'Model':<15} {'Test PR-AUC':>12}")
            print(f"  {'─'*15} {'─'*12}")
            for name, _, _, score in sorted(results, key=lambda x: x[3], reverse=True):
                leader = " ← best" if name == sorted(results, key=lambda x: x[3], reverse=True)[0][0] else ""
                print(f"  {name:<15} {score:>12.4f}{leader}")
            print()

        except Exception as e:
            print(f"\n  ERROR training {model_type.value}: {e}")
            import traceback; traceback.print_exc()

    if not results:
        raise RuntimeError("All model training attempts failed.")

    results.sort(key=lambda x: x[3], reverse=True)
    winner_name, winner_id, winner_uuid, winner_pr_auc = results[0]

    print("\n" + "=" * 60)
    print("  FINAL TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"  {'Model':<15} {'Test PR-AUC':>12}")
    print(f"  {'─'*15} {'─'*12}")
    for name, _, _, score in results:
        tag = "  🏆 WINNER" if name == winner_name else ""
        print(f"  {name:<15} {score:>12.4f}{tag}")
    print("=" * 60)

    return winner_id, winner_uuid, winner_pr_auc, winner_name


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PROMOTION
# ─────────────────────────────────────────────────────────────────────────────

def promote_best_model(winner_id, winner_uuid, winner_score, winner_name, client):
    print("\n" + "=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)

    registry_name = Config.REGISTRY_NAME
    model_uri     = f"runs:/{winner_id}/model"

    current_score = 0.0
    try:
        versions = client.search_model_versions(f"name='{registry_name}'")
        for v in versions:
            if v.aliases and "champion" in v.aliases:
                tag_score = v.tags.get("pr_auc")
                if tag_score:
                    current_score = float(tag_score)
                print(f"  Current champion PR-AUC: {current_score:.4f}")
                break
    except Exception:
        pass

    mv = mlflow.register_model(model_uri, registry_name)
    print(f"  Registered as version {mv.version}")

    if winner_score >= current_score:
        try:
            client.delete_registered_model_alias(registry_name, "champion")
            client.set_registered_model_alias(registry_name, "challenger",
                                              str(int(mv.version) - 1))
        except Exception:
            pass
        client.set_registered_model_alias(registry_name, "champion", mv.version)
        client.set_model_version_tag(registry_name, mv.version, "pr_auc",    str(winner_score))
        client.set_model_version_tag(registry_name, mv.version, "model_type", winner_name)
        client.set_model_version_tag(registry_name, mv.version, "run_id",     winner_id)
        improvement = ((winner_score - current_score) / current_score * 100
                       if current_score > 0 else 100.0)
        print(f"  ✅ Promoted v{mv.version} as champion  "
              f"(PR-AUC={winner_score:.4f}, +{improvement:.1f}%)")
    else:
        client.set_registered_model_alias(registry_name, "archived", mv.version)
        print(f"  ⚠️  Archived v{mv.version} — did not beat champion")

    print("=" * 60)
    return mv.version


def load_and_prepare_data(use_gnn_embeddings: bool = False) -> TrainingData:
    config = Config()
    config.setup_directories()
    return DataPreparator(config, use_gnn_embeddings=use_gnn_embeddings).prepare_data()