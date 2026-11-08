"""Module B: train and evaluate the Random Forest on hostname features.

Run:  python -m src.train            (add --fast to skip hyper-parameter search)
Out:  models/rf_pipeline.joblib, models/model_meta.json,
      results/metrics.json, results/*.png
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_prep import OUT_CSV
from src.features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODEL_PATH = MODELS / "rf_pipeline.joblib"
META_PATH = MODELS / "model_meta.json"
SEED = 42
TARGET_RECALL = 0.85
MAX_MODEL_MB = 95

# Best values found by `tune` (RandomizedSearchCV, 10 candidates, GroupKFold(3), PR-AUC).
DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 8, "max_features": "sqrt"}


def build_pipeline(**rf_params) -> Pipeline:
    pre = ColumnTransformer(
        [("tld", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED, **rf_params)
    return Pipeline([("pre", pre), ("rf", rf)])


def load_data():
    df = pd.read_csv(OUT_CSV, keep_default_na=False)
    X = df[FEATURE_COLUMNS].copy()
    X[NUMERIC] = X[NUMERIC].astype(float)
    return df, X, df["is_phishing"].astype(int), df["reg_domain"]


def grouped_split(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    return next(gss.split(X, y, groups))


def scores(y_true, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision_phishing": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall_phishing": round(recall_score(y_true, pred), 4),
        "f1_phishing": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def choose_threshold(y, proba) -> float:
    """Max-F1 threshold, moved down if needed so phishing recall reaches TARGET_RECALL,
    but never so far that precision drops below 0.70."""
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = int(np.argmax(f1))
    if rec[best] >= TARGET_RECALL:
        return float(thr[best])
    ok = np.where((rec >= TARGET_RECALL) & (prec >= 0.70))[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[best])


def tune(X, y, groups, n_iter=10, sample=60_000) -> dict:
    """Light random search on a grouped subsample (full data is too slow for a mini project)."""
    rng = np.random.default_rng(SEED)
    uniq = groups.unique()
    keep = set(rng.permutation(uniq)[: int(len(uniq) * min(1.0, sample / len(X)))])
    m = groups.isin(keep).values
    space = {
        "rf__n_estimators": [150, 200],
        "rf__max_depth": [16, 24, 32, None],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", 0.3, 0.5],
    }
    search = RandomizedSearchCV(
        build_pipeline(), space, n_iter=n_iter, scoring="average_precision",
        cv=GroupKFold(n_splits=3), random_state=SEED, n_jobs=1, verbose=1,
    )
    t0 = time.time()
    search.fit(X[m], y[m], groups=groups[m])
    print(f"search done in {time.time() - t0:.0f}s on {m.sum():,} rows; best PR-AUC "
          f"{search.best_score_:.4f}: {search.best_params_}")
    best = {k.replace("rf__", ""): v for k, v in search.best_params_.items()}
    if best["min_samples_leaf"] == 1:  # keep the model file small enough for GitHub
        best["min_samples_leaf"] = 2
    best["n_estimators"] = 300
    return best


def plot_all(y_te, proba, thr, pipe):
    RESULTS.mkdir(parents=True, exist_ok=True)
    pred = (proba >= thr).astype(int)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Legit", "Phishing"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=",")
    ax.set_title(f"Random Forest (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrix.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fpr, tpr, _ = roc_curve(y_te, proba)
    axes[0].plot(fpr, tpr, color="#264653"); axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title=f"ROC (AUC {roc_auc_score(y_te, proba):.3f})", xlabel="FPR", ylabel="TPR")
    p, r, _ = precision_recall_curve(y_te, proba)
    axes[1].plot(r, p, color="#e76f51")
    axes[1].set(title=f"Precision-Recall (AP {average_precision_score(y_te, proba):.3f})",
                xlabel="Recall (phishing)", ylabel="Precision")
    fig.tight_layout(); fig.savefig(RESULTS / "roc_pr.png", dpi=120); plt.close(fig)

    imp = feature_importance(pipe)
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(top.index, top.values, color="#2a9d8f")
    ax.set_title("Random Forest feature importance (impurity)")
    fig.tight_layout(); fig.savefig(RESULTS / "feature_importance.png", dpi=120); plt.close(fig)
    return imp


def feature_importance(pipe) -> pd.Series:
    names = pipe.named_steps["pre"].get_feature_names_out()
    raw = pd.Series(pipe.named_steps["rf"].feature_importances_, index=names)
    merged = raw.groupby(lambda n: "tld_group" if n.startswith("tld_group_") else n).sum()
    return merged.sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip hyper-parameter search")
    args = ap.parse_args()

    df, X, y, groups = load_data()
    tr, te = grouped_split(X, y, groups)
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]
    print(f"train {len(tr):,} rows / test {len(te):,} rows; shared reg domains: "
          f"{len(set(groups.iloc[tr]) & set(groups.iloc[te]))}")
    print("test class balance:", y_te.value_counts().to_dict())

    params = DEFAULT_PARAMS if args.fast else tune(X_tr, y_tr, g_tr)
    print("final params:", params)

    # Threshold from out-of-fold predictions on the TRAINING split only.
    t0 = time.time()
    oof_params = {**params, "n_estimators": 100}
    oof = cross_val_predict(build_pipeline(**oof_params), X_tr, y_tr, groups=g_tr,
                            cv=GroupKFold(n_splits=3), method="predict_proba")[:, 1]
    thr = choose_threshold(y_tr.values, oof)
    print(f"chosen threshold {thr:.3f} from OOF predictions ({time.time() - t0:.0f}s)")

    t0 = time.time()
    pipe = build_pipeline(**params).fit(X_tr, y_tr)
    print(f"fit final model in {time.time() - t0:.0f}s")
    proba = pipe.predict_proba(X_te)[:, 1]

    metrics = {
        "model": "RandomForestClassifier (hostname features)",
        "params": params,
        "split": "GroupShuffleSplit 80/20 grouped by registered domain, seed 42",
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_class_balance": {"legitimate": int((y_te == 0).sum()), "phishing": int((y_te == 1).sum())},
        "at_0.5": scores(y_te, proba, 0.5),
        "at_chosen_threshold": scores(y_te, proba, thr),
    }
    # Where does the model miss? Phishing on hosting platforms vs on its own domain.
    seg = np.where(X_te["is_free_hosting"].values == 1, "free_hosting_platform", "own_domain")
    metrics["phishing_recall_by_segment"] = {
        s: {"n_phishing": int(((seg == s) & (y_te.values == 1)).sum()),
            "recall": round(float(((proba >= thr) & (seg == s) & (y_te.values == 1)).sum()
                                  / max(((seg == s) & (y_te.values == 1)).sum(), 1)), 4)}
        for s in ("free_hosting_platform", "own_domain")
    }
    print("phishing recall by segment:", metrics["phishing_recall_by_segment"])
    imp = plot_all(y_te, proba, thr, pipe)
    metrics["feature_importance"] = imp.round(4).to_dict()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    while size_mb > MAX_MODEL_MB and len(pipe.named_steps["rf"].estimators_) > 50:
        # Trim trees (a forest is an average, so a subset is still a valid forest).
        rf = pipe.named_steps["rf"]
        rf.estimators_ = rf.estimators_[: int(len(rf.estimators_) * 0.75)]
        rf.n_estimators = len(rf.estimators_)
        joblib.dump(pipe, MODEL_PATH, compress=3)
        size_mb = MODEL_PATH.stat().st_size / 1e6
        print(f"  trimmed to {rf.n_estimators} trees -> {size_mb:.1f} MB")

    num = X_tr[y_tr == 0][[c for c in FEATURE_COLUMNS if c not in CATEGORICAL]]
    meta = {
        "trained_on": str(date.today()),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": round(float(thr), 4),
        "n_trees": int(pipe.named_steps["rf"].n_estimators),
        "model_size_mb": round(size_mb, 1),
        "test_metrics": metrics["at_chosen_threshold"],
        "legit_median": num.median().round(4).to_dict(),
        "legit_p95": num.quantile(0.95).round(4).to_dict(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(json.dumps({k: metrics[k] for k in ("at_0.5", "at_chosen_threshold")}, indent=2))
    print("top features:", imp.head(8).round(3).to_dict())
    print(f"saved {MODEL_PATH} ({size_mb:.1f} MB) and {META_PATH}")


if __name__ == "__main__":
    main()
