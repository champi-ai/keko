"""The negative model: a per-promoter ``NegativeSpace`` over frozen activations.

Components (all fitted on the training split only):

* ``probe``     - P(not positive) from activations (N1);
* ``knn``/``maha`` - distance to the positive region, fitted on positives only (N2);
* ``flip_reg``  - regression of the measured noise flip level from activations, so the
                  boundary distance can be *predicted* at inference time without running
                  noise (N3);
* ``topology``  - persistent H1 holes of the positive cloud vs null models (N4, optional);
* ``combiner``  - logistic regression over the component scores -> the **negative
                  score** P(error). It is trained on *cross-fitted* component scores
                  (inner grouped folds) so it never sees scores produced by a component
                  that was trained on the same item.

``score()`` returns the negative score, the components and a ``Region`` per item
(``promoters.assign_region``), from which negative-promoter events are emitted.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig
from .geometry import KNNDistance, MahalanobisLW, OCSVMScore
from .probe import group_splits, make_flip_regressor, make_probe
from .promoters import Promoter, Region, RegionThresholds, assign_region
from . import topology as topo_mod

FEATURES = ("p_not_positive", "knn", "maha", "neg_pred_flip")


@dataclass
class _Components:
    probe: object | None
    prior: float
    knn: KNNDistance
    maha: MahalanobisLW
    ocsvm: OCSVMScore | None
    flip_reg: object
    topo: topo_mod.TopologyResult | None = None
    cloud: object | None = None


class NegativeSpace:
    def __init__(self, promoter: Promoter, cfg: AnalysisConfig, seed: int = 0):
        self.promoter, self.cfg, self.seed = promoter, cfg, seed

    # ------------------------------------------------------------------ #
    def _fit_components(self, X, pos, flip, with_topology: bool) -> _Components:
        y = (~pos).astype(int)
        probe = None
        if len(np.unique(y)) == 2:
            probe = make_probe(self.cfg.probe, self.cfg.probe_C, self.seed).fit(X, y)
        Xp = X[pos]
        if len(Xp) < 3:
            raise ValueError(f"need >= 3 positive samples to build a negative space, got {len(Xp)}")
        comps = _Components(
            probe=probe, prior=float(y.mean()),
            knn=KNNDistance(self.cfg.knn_k).fit(Xp),
            maha=MahalanobisLW(self.cfg.pca_dim).fit(Xp),
            ocsvm=OCSVMScore(pca_dim=self.cfg.pca_dim).fit(Xp) if self.cfg.ocsvm else None,
            flip_reg=make_flip_regressor().fit(X, flip),
        )
        tc = self.cfg.topology
        if with_topology and tc.enabled:
            comps.topo, comps.cloud = topo_mod.find_holes(
                Xp, tc.primary_pca_dim, n_null=tc.n_null, n_boot=tc.n_boot,
                null_quantile=tc.null_quantile, max_points=tc.max_points, seed=self.seed)
        return comps

    @staticmethod
    def _features(c: _Components, X) -> np.ndarray:
        p = c.probe.predict_proba(X)[:, 1] if c.probe is not None else np.full(len(X), c.prior)
        return np.column_stack([p, c.knn.score(X), c.maha.score(X), -c.flip_reg.predict(X)])

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, records: list[dict], flip: np.ndarray, groups: np.ndarray) -> "NegativeSpace":
        pos = np.array(self.promoter.positives(records))
        err = np.array([bool(r["error"]) for r in records]).astype(int)
        F = np.full((len(X), len(FEATURES)), np.nan)
        for tr, va in group_splits(err, groups, self.cfg.n_folds, self.seed):
            if pos[tr].sum() < 3:
                continue
            c = self._fit_components(X[tr], pos[tr], flip[tr], with_topology=False)
            F[va] = self._features(c, X[va])
        ok = ~np.isnan(F).any(1)
        self.combiner = None
        if ok.sum() >= 4 and len(np.unique(err[ok])) == 2:
            self.combiner = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                                              max_iter=5000))
            self.combiner.fit(F[ok], err[ok])
        self.components = self._fit_components(X, pos, flip, with_topology=True)
        knn_pos = F[ok & pos, 1] if (ok & pos).sum() else self.components.knn.score(X[pos], exclude_self=True)
        self.thresholds = RegionThresholds(
            tau_outside=float(np.quantile(knn_pos, self.cfg.outside_quantile)),
            tau_flip=float(np.quantile(flip[pos], self.cfg.boundary_flip_quantile)),
            confidence_high=self.cfg.confidence_high,
        )
        self.oof_features_ = F
        return self

    def score(self, X: np.ndarray, confidence: np.ndarray) -> dict:
        c = self.components
        F = self._features(c, X)
        if self.combiner is not None:
            neg = self.combiner.predict_proba(F)[:, 1]
        else:  # fallback: mean percentile rank of components
            neg = np.mean([np.argsort(np.argsort(F[:, j])) / max(1, len(F) - 1) for j in range(F.shape[1])], 0)
        hole_flag, hole_pers = (topo_mod.in_hole(X, c.topo, c.cloud, self.cfg.topology.hole_core_frac)
                                if c.topo is not None else (np.zeros(len(X), bool), np.zeros(len(X))))
        regions = [assign_region(F[i, 1], -F[i, 3], F[i, 0], bool(hole_flag[i]), float(confidence[i]),
                                 self.thresholds) for i in range(len(X))]
        out = dict(negative_score=neg, p_not_positive=F[:, 0], knn=F[:, 1], maha=F[:, 2],
                   pred_flip=-F[:, 3], in_hole=hole_flag, hole_persistence=hole_pers, region=regions)
        if c.ocsvm is not None:
            out["ocsvm"] = c.ocsvm.score(X)
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        label_fn, self.promoter.label_fn = self.promoter.label_fn, None  # lambdas do not pickle
        try:
            with path.open("wb") as f:
                pickle.dump(self, f)
        finally:
            self.promoter.label_fn = label_fn

    @staticmethod
    def load(path: str | Path, promoter: Promoter | None = None) -> "NegativeSpace":
        with Path(path).open("rb") as f:
            obj: NegativeSpace = pickle.load(f)
        if promoter is not None:
            obj.promoter = promoter
        else:
            from .promoters import KNOWLEDGE_PROMOTER
            if obj.promoter.name == KNOWLEDGE_PROMOTER.name:
                obj.promoter = KNOWLEDGE_PROMOTER
        return obj


def region_counts(regions: list[Region]) -> dict[str, int]:
    return {r.value: int(sum(x == r for x in regions)) for r in Region}
