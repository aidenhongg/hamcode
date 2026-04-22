"""Train the point-task random forest classifier.

Inputs (parquet): combined feature table with one row per PointRecord, columns =
    id, source, problem_id, label, split (train/val/test), and the 18 features.

Outputs:
    models/point_rf.joblib — the trained ensemble + feature names + label map
    models/metrics.json — train/val/test metrics
    models/confusion_test.png — confusion matrix on test set
    models/feature_importance.png — average importance across ensemble
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV

from .features import FEATURE_NAMES
from .labels import IDX_TO_LABEL, LABEL_TO_IDX, POINT_LABELS


@dataclass
class TrainedModel:
    estimators: list[RandomForestClassifier]
    feature_names: tuple[str, ...]
    labels: tuple[str, ...]
    best_params: dict[str, Any] | None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Average probabilities across the ensemble.
        probs = np.zeros((X.shape[0], len(self.labels)), dtype=np.float64)
        for est in self.estimators:
            est_probs = est.predict_proba(X)
            # Align columns: est.classes_ may be a subset of label indices.
            aligned = np.zeros_like(probs)
            for col_idx, cls in enumerate(est.classes_):
                aligned[:, cls] = est_probs[:, col_idx]
            probs += aligned
        probs /= len(self.estimators)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def load_features_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")
    if "label" not in df.columns:
        raise ValueError("feature table missing 'label' column")
    if "split" not in df.columns:
        raise ValueError("feature table missing 'split' column — run 02_extract_features first")
    return df


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"] == "train"].reset_index(drop=True),
        df[df["split"] == "val"].reset_index(drop=True),
        df[df["split"] == "test"].reset_index(drop=True),
    )


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = df["label"].map(LABEL_TO_IDX).to_numpy(dtype=np.int64)
    if np.any(y < 0):
        raise ValueError("encountered labels not in POINT_LABELS")
    return X, y


def _best_rf_params(X_train: np.ndarray, y_train: np.ndarray,
                    cfg: dict) -> dict[str, Any]:
    """Optionally run RandomizedSearchCV to pick hyperparameters."""
    search_cfg = cfg.get("search", {})
    if not search_cfg.get("enabled", False):
        return {}
    base = RandomForestClassifier(
        class_weight=cfg["rf"]["class_weight"],
        bootstrap=cfg["rf"]["bootstrap"],
        n_jobs=cfg["rf"]["n_jobs"],
        random_state=cfg["rf"]["random_state"],
    )
    # Convert YAML nulls to None
    distributions = {
        k: [None if v == "null" else v for v in vals]
        for k, vals in search_cfg["distributions"].items()
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=distributions,
        n_iter=search_cfg["n_iter"],
        cv=search_cfg["cv_folds"],
        scoring=search_cfg["scoring"],
        n_jobs=cfg["rf"]["n_jobs"],
        random_state=cfg["rf"]["random_state"],
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_params_


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   cfg: dict, best_params: dict[str, Any]) -> TrainedModel:
    rf_cfg = dict(cfg["rf"])
    rf_cfg.update(best_params)  # override with search results if any
    # Remove None-valued max_depth from YAML null
    if rf_cfg.get("max_depth") == "null":
        rf_cfg["max_depth"] = None

    ens_cfg = cfg.get("ensemble", {})
    if ens_cfg.get("enabled", False):
        seeds = ens_cfg.get("seeds", [42])
    else:
        seeds = [rf_cfg.get("random_state", 42)]

    estimators: list[RandomForestClassifier] = []
    for seed in seeds:
        params = dict(rf_cfg)
        params["random_state"] = seed
        est = RandomForestClassifier(**params)
        est.fit(X_train, y_train)
        estimators.append(est)

    return TrainedModel(
        estimators=estimators,
        feature_names=FEATURE_NAMES,
        labels=POINT_LABELS,
        best_params=best_params or None,
    )


def evaluate(model: TrainedModel, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y, y_pred,
        labels=list(range(len(POINT_LABELS))),
        target_names=list(POINT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=list(range(len(POINT_LABELS))))
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _mean_feature_importance(model: TrainedModel) -> np.ndarray:
    return np.mean(
        [est.feature_importances_ for est in model.estimators],
        axis=0,
    )


def save_model(model: TrainedModel, metrics: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "estimators": model.estimators,
        "feature_names": list(model.feature_names),
        "labels": list(model.labels),
        "best_params": model.best_params,
    }, out_dir / "point_rf.joblib")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Feature importance plot (optional; matplotlib is a dev-time dep)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        imp = _mean_feature_importance(model)
        order = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            [FEATURE_NAMES[i] for i in order[::-1]],
            imp[order[::-1]],
        )
        ax.set_xlabel("mean importance across ensemble")
        ax.set_title("Random forest — feature importance")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[train] feature importance plot skipped: {e}")

    # Confusion matrix plot (uses test-set cm if present)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm_test = metrics.get("test", {}).get("confusion_matrix")
        if cm_test is not None:
            cm = np.array(cm_test)
            fig, ax = plt.subplots(figsize=(10, 8))
            # Row-normalize for readability
            row_sums = cm.sum(axis=1, keepdims=True)
            norm = np.divide(cm, np.maximum(row_sums, 1))
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(POINT_LABELS)))
            ax.set_yticks(range(len(POINT_LABELS)))
            ax.set_xticklabels(POINT_LABELS, rotation=45, ha="right")
            ax.set_yticklabels(POINT_LABELS)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title("Confusion matrix (test, row-normalized)")
            # annotate
            for i in range(len(POINT_LABELS)):
                for j in range(len(POINT_LABELS)):
                    val = cm[i, j]
                    if val:
                        ax.text(j, i, str(val), ha="center", va="center",
                                color="white" if norm[i, j] > 0.5 else "black",
                                fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / "confusion_test.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"[train] confusion matrix plot skipped: {e}")


def run(features_path: Path, config_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    df = load_features_table(features_path)
    train_df, val_df, test_df = split_frame(df)
    print(f"[train] sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df) if len(val_df) else (np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int64))
    X_test, y_test = to_xy(test_df) if len(test_df) else (np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int64))

    best_params = _best_rf_params(X_train, y_train, cfg)
    if best_params:
        print(f"[train] best params from search: {best_params}")

    model = train_ensemble(X_train, y_train, cfg, best_params)

    metrics: dict[str, Any] = {"best_params": best_params}
    metrics["train"] = evaluate(model, X_train, y_train)
    if len(val_df):
        metrics["val"] = evaluate(model, X_val, y_val)
    if len(test_df):
        metrics["test"] = evaluate(model, X_test, y_test)

    save_model(model, metrics, out_dir)
    print(f"[train] saved model + metrics to {out_dir}")
    print(f"[train] test f1_macro = {metrics.get('test', {}).get('f1_macro')}")
    return metrics


def load_model(path: Path) -> TrainedModel:
    obj = joblib.load(path)
    return TrainedModel(
        estimators=obj["estimators"],
        feature_names=tuple(obj["feature_names"]),
        labels=tuple(obj["labels"]),
        best_params=obj.get("best_params"),
    )
