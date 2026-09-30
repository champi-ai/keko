"""Persistent homology of the known region (N4, exploratory). Optional dependency: ripser.

What is computed
----------------
1. The known-item activations are standardised, PCA-reduced to ``pca_dim`` and
   rescaled so the median pairwise distance is 1 (makes persistences comparable).
2. Vietoris-Rips persistence (H0, H1) with ``ripser``.
3. **Null models.** The same computation on (a) Gaussian clouds with the same mean
   and covariance and (b) per-coordinate shuffled clouds (same marginals, dependence
   destroyed), same n and dimension. The ``null_quantile`` of the *maximum* H1
   persistence across null clouds is the significance threshold.
4. **Bootstrap.** The fraction of bootstrap resamples of the known cloud whose max
   H1 persistence still exceeds the null threshold.
5. A feature is called a **hole** only if its persistence exceeds the null
   threshold. For each hole we take the vertices of its representative cocycle,
   use their centroid as the hole centre and their mean distance to it as the
   radius. A point is flagged as "in the hole" when it lies within
   ``core_frac * radius`` of the centre *and* farther from every known point than
   the hole's birth scale (i.e. inside the empty region). This is a heuristic
   localisation, not an exact one; see README "Threats to validity".

If ``ripser`` is missing everything degrades to ``available=False`` with no holes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def have_ripser() -> bool:
    try:
        import ripser  # noqa: F401
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


@dataclass
class Hole:
    birth: float
    death: float
    persistence: float
    center: np.ndarray
    radius: float


@dataclass
class TopologyResult:
    available: bool
    pca_dim: int = 0
    n_points: int = 0
    h0: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    h1: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    null_threshold: float = float("nan")
    null_max_pers: dict = field(default_factory=dict)
    boot_frac_above: float = float("nan")
    holes: list[Hole] = field(default_factory=list)
    note: str = ""

    def summary(self) -> dict:
        h1p = (self.h1[:, 1] - self.h1[:, 0]) if len(self.h1) else np.zeros(0)
        return dict(available=self.available, pca_dim=self.pca_dim, n_points=self.n_points,
                    n_h1=int(len(self.h1)), max_h1_persistence=float(h1p.max()) if len(h1p) else 0.0,
                    null_threshold=self.null_threshold, n_significant_holes=len(self.holes),
                    bootstrap_frac_above_null=self.boot_frac_above, note=self.note)


class KnownCloud:
    """Standardise -> PCA -> scale-normalise, fitted on known activations."""

    def __init__(self, pca_dim: int):
        self.pca_dim = pca_dim

    def fit(self, X: np.ndarray) -> "KnownCloud":
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        dim = max(1, min(self.pca_dim, Z.shape[0] - 1, Z.shape[1]))
        self.pca = PCA(n_components=dim, random_state=0).fit(Z)
        P = self.pca.transform(Z)
        D = pairwise_distances(P)
        self.scale = float(np.median(D[np.triu_indices_from(D, 1)])) or 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X)) / self.scale


def _max_h1(P: np.ndarray) -> float:
    from ripser import ripser

    dg = ripser(P, maxdim=1)["dgms"][1]
    return float((dg[:, 1] - dg[:, 0]).max()) if len(dg) else 0.0


def null_cloud(P: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        cov = np.cov(P, rowvar=False) if P.shape[1] > 1 else np.var(P, keepdims=True)
        return rng.multivariate_normal(P.mean(0), np.atleast_2d(cov), size=len(P))
    if kind == "shuffle":
        return np.column_stack([rng.permutation(P[:, j]) for j in range(P.shape[1])])
    raise ValueError(kind)


def find_holes(X_known: np.ndarray, pca_dim: int, *, n_null: int = 20, n_boot: int = 20,
               null_quantile: float = 0.95, max_points: int = 300, seed: int = 0) -> tuple[TopologyResult, KnownCloud | None]:
    if not have_ripser():
        return TopologyResult(available=False, note="ripser not installed (uv sync --extra topology)"), None
    if len(X_known) < 8:
        return TopologyResult(available=True, note=f"too few known points ({len(X_known)}) for H1"), None
    from ripser import ripser

    rng = np.random.default_rng(seed)
    if len(X_known) > max_points:
        X_known = X_known[rng.choice(len(X_known), max_points, replace=False)]
    cloud = KnownCloud(pca_dim).fit(X_known)
    P = cloud.transform(X_known)
    res = ripser(P, maxdim=1, do_cocycles=True)
    h0, h1 = res["dgms"][0], res["dgms"][1]
    nulls = {k: [_max_h1(null_cloud(P, k, rng)) for _ in range(n_null)] for k in ("gaussian", "shuffle")}
    thr = float(np.quantile(np.concatenate([nulls["gaussian"], nulls["shuffle"]]), null_quantile))
    boots = [_max_h1(P[rng.choice(len(P), len(P), replace=True)]) for _ in range(n_boot)]
    holes = []
    for (b, d), cocycle in zip(h1, res["cocycles"][1]):
        if not np.isfinite(d) or (d - b) <= thr:
            continue
        verts = np.unique(cocycle[:, :2].astype(int))
        c = P[verts].mean(0)
        r = float(np.linalg.norm(P[verts] - c, axis=1).mean())
        holes.append(Hole(birth=float(b), death=float(d), persistence=float(d - b), center=c, radius=r))
    out = TopologyResult(available=True, pca_dim=P.shape[1], n_points=len(P), h0=h0, h1=h1,
                         null_threshold=thr, null_max_pers={k: list(map(float, v)) for k, v in nulls.items()},
                         boot_frac_above=float(np.mean(np.array(boots) > thr)), holes=holes)
    cloud.P_known = P  # type: ignore[attr-defined]
    return out, cloud


def in_hole(X: np.ndarray, topo: TopologyResult, cloud: KnownCloud | None,
            core_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Return (flag, persistence-of-containing-hole) for each row of X."""
    n = len(X)
    flag, pers = np.zeros(n, dtype=bool), np.zeros(n, dtype=np.float32)
    if cloud is None or not topo.holes:
        return flag, pers
    P = cloud.transform(X)
    nn = pairwise_distances(P, cloud.P_known).min(1)  # type: ignore[attr-defined]
    for h in topo.holes:
        inside = (np.linalg.norm(P - h.center, axis=1) < core_frac * h.radius) & (nn > h.birth)
        better = inside & (h.persistence > pers)
        flag |= inside
        pers[better] = h.persistence
    return flag, pers


def sensitivity(X_known: np.ndarray, pca_dims: list[int], **kw) -> list[dict]:
    """Hole counts / persistence vs null across PCA dimensions."""
    rows = []
    for dim in pca_dims:
        topo, _ = find_holes(X_known, dim, **kw)
        rows.append(topo.summary())
    return rows
