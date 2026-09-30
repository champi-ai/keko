"""Distance-to-the-known-region models fitted on *positive* (known) activations only (N2).

All models expose ``fit(X_pos)`` and ``score(X) -> larger = farther from the known region``.
Standardisation is fitted on the positive set only, so nothing about the negatives
leaks into the geometry.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class KNNDistance:
    """Mean Euclidean distance to the k nearest known points (standardised space)."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X_pos: np.ndarray) -> "KNNDistance":
        self.scaler = StandardScaler().fit(X_pos)
        Z = self.scaler.transform(X_pos)
        self.k_eff = max(1, min(self.k, len(Z) - 1))
        self.nn = NearestNeighbors(n_neighbors=self.k_eff + 1).fit(Z)
        return self

    def score(self, X: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        Z = self.scaler.transform(X)
        d, _ = self.nn.kneighbors(Z, n_neighbors=self.k_eff + 1)
        d = d[:, 1:] if exclude_self else d[:, : self.k_eff]
        return d.mean(1)


class MahalanobisLW:
    """Mahalanobis distance with Ledoit-Wolf shrinkage covariance after PCA.

    With n_known << d (e.g. ~50 known items, d = 960), the sample covariance is
    singular; PCA to ``pca_dim`` (capped at n-1) followed by Ledoit-Wolf shrinkage
    gives a well-conditioned estimate. The chosen layer and dimension are reported.
    """

    def __init__(self, pca_dim: int = 32):
        self.pca_dim = pca_dim

    def fit(self, X_pos: np.ndarray) -> "MahalanobisLW":
        self.scaler = StandardScaler().fit(X_pos)
        Z = self.scaler.transform(X_pos)
        dim = max(1, min(self.pca_dim, Z.shape[0] - 1, Z.shape[1]))
        self.pca = PCA(n_components=dim, random_state=0).fit(Z)
        self.lw = LedoitWolf().fit(self.pca.transform(Z))
        self.shrinkage_ = float(self.lw.shrinkage_)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        P = self.pca.transform(self.scaler.transform(X))
        return np.sqrt(np.maximum(self.lw.mahalanobis(P), 0.0))


class OCSVMScore:
    """Optional one-class SVM support estimate (Schölkopf et al., 2001)."""

    def __init__(self, nu: float = 0.1, pca_dim: int = 32):
        self.nu, self.pca_dim = nu, pca_dim

    def fit(self, X_pos: np.ndarray) -> "OCSVMScore":
        self.scaler = StandardScaler().fit(X_pos)
        Z = self.scaler.transform(X_pos)
        dim = max(1, min(self.pca_dim, Z.shape[0] - 1, Z.shape[1]))
        self.pca = PCA(n_components=dim, random_state=0).fit(Z)
        self.svm = OneClassSVM(nu=self.nu, gamma="scale").fit(self.pca.transform(Z))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.svm.decision_function(self.pca.transform(self.scaler.transform(X)))
