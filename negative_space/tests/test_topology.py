import numpy as np
import pytest

from negative_space import topology as T


def _embed(P, d, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(d, P.shape[1])))
    return P @ Q.T


@pytest.mark.skipif(not T.have_ripser(), reason="topology extra not installed")
def test_circle_has_significant_hole_and_center_is_inside(rng):
    th = rng.uniform(0, 2 * np.pi, 80)
    ring = np.column_stack([np.cos(th), np.sin(th)]) + rng.normal(scale=0.03, size=(80, 2))
    X = _embed(ring, 20, rng)
    topo, cloud = T.find_holes(X, pca_dim=2, n_null=10, n_boot=10, seed=0)
    assert topo.available and len(topo.holes) >= 1
    center = _embed(np.zeros((1, 2)), 20, np.random.default_rng(0))  # same basis needed
    flag, pers = T.in_hole(X.mean(0, keepdims=True), topo, cloud)
    assert flag[0] and pers[0] > 0
    assert topo.boot_frac_above > 0.5


@pytest.mark.skipif(not T.have_ripser(), reason="topology extra not installed")
def test_gaussian_blob_has_no_significant_hole(rng):
    X = rng.normal(size=(80, 20))
    topo, _ = T.find_holes(X, pca_dim=3, n_null=10, n_boot=5, seed=0)
    assert len(topo.holes) == 0


def test_graceful_without_ripser(monkeypatch, rng):
    monkeypatch.setattr(T, "have_ripser", lambda: False)
    topo, cloud = T.find_holes(rng.normal(size=(30, 5)), pca_dim=3)
    assert not topo.available and cloud is None and "ripser" in topo.note
    flag, pers = T.in_hole(rng.normal(size=(4, 5)), topo, cloud)
    assert not flag.any()
    assert T.sensitivity(rng.normal(size=(30, 5)), [2, 3])[0]["available"] is False
