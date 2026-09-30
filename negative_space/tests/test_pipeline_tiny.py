"""End-to-end on a tiny random model: shapes, stage wiring, negative-space save/load."""

import numpy as np

from negative_space.config import load_config
from negative_space.evaluate import bootstrap_ci, ece, paired_delta
from negative_space.extract import run_extract
from negative_space.modeling import generate, hidden_states
from negative_space.negative_model import NegativeSpace
from negative_space.noise import run_noise
from negative_space.promoters import KNOWLEDGE_PROMOTER, Region

ITEMS = [
    dict(id="a", question="what is the capital of france ?", answers=["paris"], kind="answerable",
         domain="geo", template="capital", group="g1", answer_type="text"),
    dict(id="b", question="what is 2 + 3 ?", answers=["5"], kind="answerable", domain="arith",
         template="add", group="g2", answer_type="number"),
    dict(id="c", question="who wrote a b c ?", answers=[], kind="fictional", domain="geo",
         template="who", group="g3", answer_type="text"),
]


def _cfg():
    return load_config(None, model=dict(use_chat_template=False, max_new_tokens=4, batch_size=2),
                       extract=dict(layers=[1, 2], k_samples=2),
                       noise=dict(levels=[0.0, 0.5, 1.0], hidden_layer=1))


def test_generate_and_hidden_shapes(tiny_model):
    model, tok = tiny_model
    out = generate(model, tok, ["what is", "the capital of france"], max_new_tokens=3, batch_size=2)
    assert len(out.texts) == 2 and out.first_max_prob.shape == (2,)
    assert np.all((out.first_max_prob > 0) & (out.first_max_prob <= 1))
    H = hidden_states(model, tok, ["what is", "a b c d"], [1, 2, -1], ["last", "mean"], 2)
    assert set(H) == {"L1_last", "L1_mean", "L2_last", "L2_mean"}
    assert H["L1_last"].shape == (2, 32)


def test_extract_and_noise_stages(tiny_model):
    model, tok = tiny_model
    cfg = _cfg()
    recs, H = run_extract(cfg, ITEMS, model, tok)
    assert [r["label"] for r in recs][2] == "unknown"
    assert all(len(r["samples"]) == 2 for r in recs)
    assert H["L2_last"].shape == (3, 32)
    res = run_noise(cfg, recs, model, tok)
    for site in ("embedding", "hidden"):
        st = res["sites"][site]["stability"]
        assert st.shape == (3, 3) and np.all(st[:, 0] == 1)
        assert res["sites"][site]["flip_level"].shape == (3,)


def test_negative_space_fit_score_save_load(tmp_path, rng):
    n = 90
    y_known = np.arange(n) % 3 == 0
    X = rng.normal(size=(n, 16))
    X[~y_known] += rng.normal(size=16) * 2
    recs = [dict(label="known" if k else "unknown", error=not k) for k in y_known]
    flip = np.where(y_known, 1.0, 0.2) + rng.normal(scale=0.05, size=n)
    cfg = load_config(None).analysis
    cfg.pca_dim = 8
    cfg.topology.n_null = 5
    cfg.topology.n_boot = 5
    ns = NegativeSpace(KNOWLEDGE_PROMOTER, cfg).fit(X, recs, flip, np.arange(n))
    out = ns.score(X, confidence=np.full(n, 0.9))
    assert out["negative_score"].shape == (n,)
    assert out["negative_score"][~y_known].mean() > out["negative_score"][y_known].mean()
    assert all(isinstance(r, Region) for r in out["region"])
    ns.save(tmp_path / "ns.pkl")
    ns2 = NegativeSpace.load(tmp_path / "ns.pkl")
    assert ns2.promoter.label_fn is not None
    np.testing.assert_allclose(ns2.score(X, np.full(n, 0.9))["negative_score"], out["negative_score"])


def test_metrics(rng):
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    s = y + rng.normal(scale=0.5, size=100)
    a, lo, hi = bootstrap_ci(y, s, "auroc", 200, 0)
    assert lo <= a <= hi and a > 0.8
    d, lo, hi, p = paired_delta(y, s, rng.normal(size=100), 200, 0)
    assert d > 0 and lo > 0 and p < 0.05
    assert ece(y, y.astype(float)) == 0.0
    assert ece(y, np.full(100, 0.5)) == 0.0
    assert ece(y, 1 - y.astype(float)) == 1.0
