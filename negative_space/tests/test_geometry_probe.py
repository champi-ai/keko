import numpy as np

from negative_space.geometry import KNNDistance, MahalanobisLW, OCSVMScore
from negative_space.probe import (control_labels, cv_predict_flip, cv_predict_proba, safe_auroc,
                                  selectivity)


def test_distances_grow_away_from_known(rng):
    Xk = rng.normal(size=(40, 100))           # n < d: needs PCA + shrinkage
    near = rng.normal(size=(20, 100))
    far = rng.normal(size=(20, 100)) + 4.0
    for m in (KNNDistance(5), MahalanobisLW(16), OCSVMScore(pca_dim=16)):
        m.fit(Xk)
        assert m.score(far).mean() > m.score(near).mean()
    maha = MahalanobisLW(16).fit(Xk)
    assert 0.0 <= maha.shrinkage_ <= 1.0
    assert KNNDistance(5).fit(Xk).score(Xk, exclude_self=True).min() > 0


def test_probe_on_separable_data(rng):
    n = 120
    y = np.arange(n) % 2
    X = rng.normal(size=(n, 20))
    X[:, 0] += 3 * y
    groups = np.arange(n) // 2
    assert safe_auroc(y, cv_predict_proba(X, y, groups, n_folds=5)) > 0.9
    flip = X[:, 1] * 2 + rng.normal(scale=0.1, size=n)
    pf = cv_predict_flip(X, flip, groups, n_folds=5)
    assert np.corrcoef(pf, flip)[0, 1] > 0.9


def test_control_task_and_selectivity(rng):
    n = 200
    types = np.array([f"t{i % 10}" for i in range(n)])
    y = (rng.random(n) < 0.4).astype(int)
    yc = control_labels(types, y, seed=0)
    for t in np.unique(types):                # constant within a type
        assert len(set(yc[types == t])) == 1
    X = rng.normal(size=(n, 10))
    X[:, 0] += 3 * y                          # signal about y, not about type
    s = selectivity(X, y, np.arange(n), types, n_folds=5)
    assert s["task_auroc"] > 0.9 and s["selectivity"] > 0.2
