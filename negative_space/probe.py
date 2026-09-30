"""Probes over frozen activations (N1) and a regressor for distance-to-boundary (N3).

Includes the confound controls required by the design:

* grouped cross-validation (``group`` = entity) so an entity never spans train/test;
* Hewitt & Liang (2019)-style **control task**: each surface *type* (template) gets a
  random label with the task's marginal; a probe that merely memorises surface type
  scores well on it too. **Selectivity** = task AUROC - control AUROC;
* **surface-feature baselines** (length, domain id, template id) that never look at
  activations.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_probe(kind: str = "logreg", C: float = 0.5, seed: int = 0):
    if kind == "logreg":
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000,
                            early_stopping=False, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def group_splits(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int):
    """Stratified grouped folds; falls back to plain GroupKFold if stratification fails."""
    n_groups = len(np.unique(groups))
    k = max(2, min(n_folds, n_groups))
    try:
        if min(np.bincount(y.astype(int))) < k:
            raise ValueError
        return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(y, y, groups))
    except ValueError:
        return list(GroupKFold(n_splits=k).split(y, y, groups))


def cv_predict_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_folds: int = 5,
                     seed: int = 0, kind: str = "logreg", C: float = 0.5) -> np.ndarray:
    """Out-of-fold P(y=1). Folds whose training split is single-class predict the prior."""
    oof = np.full(len(y), np.nan)
    for tr, te in group_splits(y, groups, n_folds, seed):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        m = make_probe(kind, C, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def control_labels(types: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Random label per surface type, matching the positive rate of ``y``."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(types)
    rate = y.mean()
    lab = {t: int(rng.random() < rate) for t in uniq}
    out = np.array([lab[t] for t in types])
    if len(np.unique(out)) < 2 and len(uniq) > 1:  # force both classes to exist
        out[types == uniq[0]] = 1 - out[types == uniq[0]]
    return out


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def selectivity(X: np.ndarray, y: np.ndarray, groups: np.ndarray, types: np.ndarray, *,
                n_folds: int = 5, seed: int = 0, kind: str = "logreg", C: float = 0.5) -> dict:
    task = safe_auroc(y, cv_predict_proba(X, y, groups, n_folds=n_folds, seed=seed, kind=kind, C=C))
    yc = control_labels(types, y, seed)
    ctrl = safe_auroc(yc, cv_predict_proba(X, yc, groups, n_folds=n_folds, seed=seed, kind=kind, C=C))
    return dict(task_auroc=task, control_auroc=ctrl, selectivity=task - ctrl)


def surface_features(records: list[dict]) -> np.ndarray:
    """Surface-only features: length in chars / words / prompt tokens + domain and template one-hot."""
    num = np.array([[r["question_chars"], len(r["question"].split()), r.get("prompt_tokens", 0)]
                    for r in records], dtype=float)
    cat = np.array([[r["domain"], r["template"]] for r in records])
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(cat)
    return np.hstack([num, oh])


# --------------------------------------------------------------------------- #
# Distance-to-boundary regression (predicting the noise flip level)
# --------------------------------------------------------------------------- #
def make_flip_regressor(alpha: float = 10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def cv_predict_flip(X: np.ndarray, flip: np.ndarray, groups: np.ndarray, *, n_folds: int = 5,
                    seed: int = 0) -> np.ndarray:
    oof = np.full(len(flip), np.nan)
    k = max(2, min(n_folds, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits=k).split(X, flip, groups):
        oof[te] = make_flip_regressor().fit(X[tr], flip[tr]).predict(X[te])
    return oof
