"""Stage 3: fit negative spaces out-of-fold and evaluate N1-N5 against baselines.

Everything reported is out-of-fold under grouped splits (``group`` = entity),
repeated over ``analysis.n_seeds`` split seeds; per-item scores are averaged over
seeds. 95% CIs are percentile bootstraps over items; comparisons against baselines are
paired bootstraps of the AUROC difference on the same resamples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .geometry import KNNDistance, MahalanobisLW
from .negative_model import NegativeSpace, region_counts
from .probe import (cv_predict_flip, cv_predict_proba, group_splits, safe_auroc, selectivity,
                    surface_features)
from .promoters import KNOWLEDGE_PROMOTER, Region, keko_window_behaviour
from . import topology as topo_mod

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of P(y=1) with equal-width bins."""
    p = np.clip(p, 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def _metric(name):
    return {"auroc": roc_auc_score, "auprc": average_precision_score, "ece": ece}[name]


def bootstrap_ci(y, s, metric="auroc", n_boot=1000, seed=0) -> tuple[float, float, float]:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    f = _metric(metric)
    point = float(f(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(f(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, float(lo), float(hi)


def paired_delta(y, s1, s2, n_boot=1000, seed=0) -> tuple[float, float, float, float]:
    """AUROC(s1) - AUROC(s2) with paired bootstrap CI and a two-sided bootstrap p-value."""
    y = np.asarray(y)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    m = ~(np.isnan(s1) | np.isnan(s2))
    y, s1, s2 = y[m], s1[m], s2[m]
    if len(np.unique(y)) < 2:
        return (float("nan"),) * 4
    point = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return float(point), float(lo), float(hi), p


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def flip_target(noise: dict, idx: np.ndarray) -> np.ndarray:
    """Mean flip level across noise sites (per item)."""
    return np.mean([d["flip_level"][idx] for d in noise["sites"].values()], axis=0)


def cv_distance(X, pos, groups, strat, n_folds, seed, pca_dim, k):
    """Out-of-fold kNN and Mahalanobis distances, fitted on training positives only."""
    knn, maha = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for tr, te in group_splits(strat, groups, n_folds, seed):
        p = tr[pos[tr]]
        if len(p) < 3:
            continue
        knn[te] = KNNDistance(k).fit(X[p]).score(X[te])
        maha[te] = MahalanobisLW(pca_dim).fit(X[p]).score(X[te])
    return knn, maha


def _ci_row(name, y, s, n_boot, seed, kind):
    a = bootstrap_ci(y, s, "auroc", n_boot, seed)
    p = bootstrap_ci(y, s, "auprc", n_boot, seed)
    return dict(score=name, kind=kind, auroc=a[0], auroc_lo=a[1], auroc_hi=a[2],
                auprc=p[0], auprc_lo=p[1], auprc_hi=p[2], n=int((~np.isnan(np.asarray(s, float))).sum()))


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def evaluate(cfg: Config, records_all: list[dict], H: dict[str, np.ndarray], noise: dict) -> dict:
    A = cfg.analysis
    keep = np.array([r["label"] != "ambiguous" for r in records_all])
    idx = np.where(keep)[0]
    recs = [records_all[i] for i in idx]
    key = f"L{A.layer}_{A.pooling}"
    X = H[key][idx]
    groups = np.array([r["group"] for r in recs])
    types = np.array([r["template"] for r in recs])
    err = np.array([r["error"] for r in recs]).astype(int)
    known = np.array([r["label"] == "known" for r in recs])
    not_known = (~known).astype(int)
    conf = np.array([r["first_max_prob"] for r in recs])
    flip = flip_target(noise, idx)
    n_boot, S = A.n_boot, A.n_seeds
    res: dict = dict(layer_key=key, n_items=len(records_all), n_analysed=len(recs),
                     label_counts={l: int(sum(r["label"] == l for r in records_all))
                                   for l in ("known", "fragile", "unknown", "ambiguous")},
                     kind_counts={k: int(sum(r["kind"] == k for r in records_all))
                                  for k in ("answerable", "fictional", "unanswerable", "ambiguous")},
                     error_rate=float(err.mean()),
                     hallucination_rate=float(np.mean([r["hallucination"] for r in recs])))

    # ---------------- outer CV: negative space per fold, averaged over seeds ------------
    comp_names = ["negative_score", "p_not_positive", "knn", "maha", "pred_flip"]
    acc = {c: np.zeros(len(recs)) for c in comp_names}
    cnt = np.zeros(len(recs))
    regions0: list = [None] * len(recs)
    in_hole0 = np.zeros(len(recs), bool)
    surf = np.zeros(len(recs))
    Xs = surface_features(recs)
    for seed in range(S):
        for tr, te in group_splits(err, groups, A.n_folds, seed):
            trr = [recs[i] for i in tr]
            if sum(KNOWLEDGE_PROMOTER.positives(trr)) < 3:
                continue
            ns = NegativeSpace(KNOWLEDGE_PROMOTER, A, seed=seed)
            if seed > 0:  # topology only needed once (it is slow and seed-independent in spirit)
                ns.cfg = _no_topo(A)
            ns.fit(X[tr], trr, flip[tr], groups[tr])
            out = ns.score(X[te], conf[te])
            for c in comp_names:
                acc[c][te] += out[c]
            cnt[te] += 1
            if seed == 0:
                for j, t in enumerate(te):
                    regions0[t] = out["region"][j]
                in_hole0[te] = out["in_hole"]
        surf += cv_predict_proba(Xs, err, groups, n_folds=A.n_folds, seed=seed, C=1.0) / S
    scores = {c: np.where(cnt > 0, acc[c] / np.maximum(cnt, 1), np.nan) for c in comp_names}
    res["folds_scored_frac"] = float((cnt > 0).mean())

    # ---------------- N1: probe decodability + selectivity + surface baseline ----------
    probe_oof = np.mean([cv_predict_proba(X, not_known, groups, n_folds=A.n_folds, seed=s,
                                          kind=A.probe, C=A.probe_C) for s in range(S)], 0)
    surf_nk = np.mean([cv_predict_proba(Xs, not_known, groups, n_folds=A.n_folds, seed=s, C=1.0)
                       for s in range(S)], 0)
    sel = [selectivity(X, not_known, groups, types, n_folds=A.n_folds, seed=s, kind=A.probe, C=A.probe_C)
           for s in range(S)]
    res["N1"] = dict(
        target="known vs not-known (fragile or unknown)",
        probe=_ci_row("activation probe", not_known, probe_oof, n_boot, 0, "ours"),
        surface=_ci_row("surface-only (length, domain, template)", not_known, surf_nk, n_boot, 0, "baseline"),
        delta_vs_surface=paired_delta(not_known, probe_oof, surf_nk, n_boot, 0),
        selectivity=dict(task_auroc=float(np.nanmean([s["task_auroc"] for s in sel])),
                         control_auroc=float(np.nanmean([s["control_auroc"] for s in sel])),
                         selectivity=float(np.nanmean([s["selectivity"] for s in sel]))),
    )
    # answerable-only variant (removes fictional/unanswerable controls)
    ans = np.array([r["kind"] == "answerable" for r in recs])
    if ans.sum() > 10 and len(np.unique(not_known[ans])) == 2:
        po = np.mean([cv_predict_proba(X[ans], not_known[ans], groups[ans], n_folds=A.n_folds, seed=s,
                                       kind=A.probe, C=A.probe_C) for s in range(S)], 0)
        res["N1"]["probe_answerable_only"] = _ci_row("probe (answerable only)", not_known[ans], po, n_boot, 0, "ours")

    # layer-wise
    lw = []
    for k in sorted(H, key=lambda s: (s.split("_")[1], int(s.split("_")[0][1:]))):
        Xl = H[k][idx]
        p = cv_predict_proba(Xl, not_known, groups, n_folds=A.n_folds, seed=0, kind=A.probe, C=A.probe_C)
        kn, mh = cv_distance(Xl, known, groups, err, A.n_folds, 0, A.pca_dim, A.knn_k)
        lw.append(dict(layer_key=k, probe_auroc_notknown=safe_auroc(not_known, p),
                       maha_auroc_error=safe_auroc(err, mh), knn_auroc_error=safe_auroc(err, kn)))
    res["layerwise"] = lw

    # ---------------- N2 + baselines: error detection ----------------------------------
    base = {
        "1 - max prob (first token)": 1 - conf,
        "1 - max prob (mean over answer)": 1 - np.array([r["mean_max_prob"] for r in recs]),
        "token entropy (mean)": np.array([r["mean_entropy"] for r in recs]),
        "keko U score (v4.2 §8, mean)": np.array([r["mean_U"] for r in recs]),
        "sampling inconsistency (1 - consistency)": 1 - np.array([r["consistency"] for r in recs]),
        "semantic-entropy proxy (answer clusters)": np.array([r["semantic_entropy_proxy"] for r in recs]),
        "surface-only classifier": surf,
    }
    ours = {
        "negative score (combined)": scores["negative_score"],
        "probe P(not known)": scores["p_not_positive"],
        "kNN distance to known": scores["knn"],
        "Mahalanobis-LW distance to known": scores["maha"],
        "predicted flip level (negated)": -scores["pred_flip"],
        "measured flip level (negated; needs noise at test time)": -flip,
    }
    rows = [_ci_row(n, err, s, n_boot, 0, "baseline") for n, s in base.items()]
    rows += [_ci_row(n, err, s, n_boot, 0, "ours") for n, s in ours.items()]
    res["N2_error_detection"] = rows
    res["N2_paired"] = [dict(ours=o, baseline=b, **dict(zip(("delta", "lo", "hi", "p"),
                                                          paired_delta(err, ours[o], base[b], n_boot, 0))))
                        for o in ("negative score (combined)", "kNN distance to known",
                                  "Mahalanobis-LW distance to known")
                        for b in ("token entropy (mean)", "1 - max prob (first token)",
                                  "sampling inconsistency (1 - consistency)")]
    ns_score = scores["negative_score"]
    res["calibration"] = dict(
        negative_score_ece=bootstrap_ci(err, ns_score, "ece", n_boot, 0),
        one_minus_maxprob_ece=bootstrap_ci(err, 1 - conf, "ece", n_boot, 0),
    )

    # ---------------- held-out domain -------------------------------------------------
    hod = []
    doms = np.array([r["domain"] for r in recs])
    for d in sorted(set(doms)):
        te = np.where(doms == d)[0]
        tr = np.where(doms != d)[0]
        trr = [recs[i] for i in tr]
        if sum(KNOWLEDGE_PROMOTER.positives(trr)) < 3 or len(te) < 4:
            continue
        ns = NegativeSpace(KNOWLEDGE_PROMOTER, _no_topo(A), seed=0).fit(X[tr], trr, flip[tr], groups[tr])
        out = ns.score(X[te], conf[te])
        hod.append(dict(domain=d, n=len(te), error_rate=float(err[te].mean()),
                        negative_score_auroc=safe_auroc(err[te], out["negative_score"]),
                        entropy_auroc=safe_auroc(err[te], base["token entropy (mean)"][te]),
                        maxprob_auroc=safe_auroc(err[te], (1 - conf)[te])))
    res["held_out_domain"] = hod

    # ---------------- N3: noise boundary ----------------------------------------------
    labels = np.array([r["label"] for r in recs])
    n3 = dict(levels=noise["levels"], sites={})
    ans_tok = np.array([r["answer_tokens"] for r in recs])
    p_tok = np.array([r["prompt_tokens"] for r in recs])
    for site, d in noise["sites"].items():
        fl = d["flip_level"][idx]
        st = d["stability"][idx]
        curves = {l: st[labels == l].mean(0).tolist() for l in ("known", "fragile", "unknown") if (labels == l).any()}
        mw = (mannwhitneyu(fl[known], fl[~known], alternative="greater")
              if known.any() and (~known).any() else None)
        n3["sites"][site] = dict(
            curves=curves, counts={l: int((labels == l).sum()) for l in curves},
            flip_auroc_notknown=bootstrap_ci(not_known, -fl, "auroc", n_boot, 0),
            flip_auroc_error=bootstrap_ci(err, -fl, "auroc", n_boot, 0),
            mannwhitney_known_gt_notknown_p=float(mw.pvalue) if mw else float("nan"),
            censored_frac=float(d["censored"][idx].mean()),
            spearman_flip_vs_answer_tokens=_spear(fl, ans_tok),
            spearman_flip_vs_prompt_tokens=_spear(fl, p_tok),
            median_flip_by_label={l: float(np.median(fl[labels == l])) for l in curves},
        )
    pf = np.mean([cv_predict_flip(X, flip, groups, n_folds=A.n_folds) for _ in range(1)], 0)
    n3["flip_regression_spearman_oof"] = _spear(pf, flip)
    res["N3"] = n3

    # ---------------- N4: topology (exploratory) ------------------------------------
    tc = A.topology
    Xk = X[known]
    if tc.enabled:
        res["N4_sensitivity"] = topo_mod.sensitivity(Xk, tc.pca_dims, n_null=tc.n_null, n_boot=tc.n_boot,
                                                     null_quantile=tc.null_quantile, max_points=tc.max_points,
                                                     seed=cfg.seed)
        topo, cloud = topo_mod.find_holes(Xk, tc.primary_pca_dim, n_null=tc.n_null, n_boot=tc.n_boot,
                                          null_quantile=tc.null_quantile, max_points=tc.max_points, seed=cfg.seed)
        flag, _ = topo_mod.in_hole(X[~known], topo, cloud, tc.hole_core_frac)
        conf_nk = conf[~known]
        err_nk = err[~known]
        res["N4"] = dict(primary=topo.summary(),
                         n_notknown_in_holes=int(flag.sum()),
                         n_confident_in_holes=int((flag & (conf_nk >= A.confidence_high)).sum()),
                         error_rate_in_holes=float(err_nk[flag].mean()) if flag.any() else float("nan"),
                         mean_conf_in_holes=float(conf_nk[flag].mean()) if flag.any() else float("nan"))
        res["_topo_plot"] = dict(h1=topo.h1.tolist(), thr=topo.null_threshold,
                                 null=topo.null_max_pers, available=topo.available)
    # ---------------- N5 (illustrative): promoter regions vs keko window ------------
    regions = [r if r is not None else Region.INTERIOR for r in regions0]
    keko = [keko_window_behaviour(c, tuple(A.keko_window)) for c in conf]
    beh = [KNOWLEDGE_PROMOTER.behaviours[r] for r in regions]
    cross: dict[str, dict[str, int]] = {}
    for b, k in zip(beh, keko):
        cross.setdefault(b.value, {}).setdefault(k.value, 0)
        cross[b.value][k.value] += 1
    by_region = {}
    for rg in Region:
        m = np.array([r == rg for r in regions])
        by_region[rg.value] = dict(n=int(m.sum()), error_rate=float(err[m].mean()) if m.any() else float("nan"),
                                   labels={l: int(((labels == l) & m).sum()) for l in ("known", "fragile", "unknown")})
    keko_int = np.array([k.value != "frozen_core_answers" for k in keko])
    matched = []
    for rate in sorted({round(float(keko_int.mean()), 3), 0.1, 0.2, 0.3}):
        if rate <= 0:
            continue
        kk = max(1, int(round(rate * len(err))))
        top_neg = np.argsort(-np.nan_to_num(ns_score, nan=-1))[:kk]
        top_msp = np.argsort(-(1 - conf))[:kk]
        matched.append(dict(rate=rate, n_intervene=kk, n_errors=int(err.sum()),
                            errors_caught_negative_score=int(err[top_neg].sum()),
                            errors_caught_low_maxprob=int(err[top_msp].sum())))
    ne = np.array([r != Region.INTERIOR for r in regions])
    res["N5_illustrative"] = dict(
        region_counts=region_counts(regions), by_region=by_region, behaviour_vs_keko_window=cross,
        keko_window_intervention_rate=float(keko_int.mean()),
        keko_window_errors_caught=int(err[keko_int].sum()),
        promoter_intervention_rate=float(ne.mean()), promoter_errors_caught=int(err[ne].sum()),
        matched_rate=matched,
    )
    # ambiguous items (aleatoric): where do they land? (descriptive only)
    amb = np.where(~keep)[0]
    res["ambiguous_n"] = int(len(amb))

    res["_curves_for_plot"] = {s: v["curves"] for s, v in n3["sites"].items()}
    res["_per_item"] = [dict(id=r["id"], label=r["label"], error=bool(r["error"]), kind=r["kind"],
                             domain=r["domain"], negative_score=_f(ns_score[i]), region=regions[i].value,
                             keko_window=keko[i].value, first_max_prob=float(conf[i]),
                             flip_level=float(flip[i]))
                        for i, r in enumerate(recs)]
    return res


def _spear(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _f(x) -> float | None:
    return None if x is None or np.isnan(x) else float(x)


def _no_topo(A):
    import copy

    B = copy.deepcopy(A)
    B.topology.enabled = False
    return B


def write_manual_check(path: Path, records: list[dict], n: int = 30, seed: int = 0) -> None:
    """Sample answerable items (balanced correct/incorrect) for manual label checking."""
    rng = np.random.default_rng(seed)
    ans = [r for r in records if r["kind"] == "answerable"]
    ok = [r for r in ans if r["greedy_correct"]]
    bad = [r for r in ans if not r["greedy_correct"]]
    pick = []
    for pool in (ok, bad):
        if pool:
            pick += [pool[i] for i in rng.choice(len(pool), min(n // 2, len(pool)), replace=False)]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold", "greedy_answer", "auto_correct", "manual_correct"])
        for r in sorted(pick, key=lambda r: r["id"]):
            w.writerow([r["id"], r["question"], " | ".join(r["answers"]), r["greedy"], r["greedy_correct"], ""])


def dump_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "value"):
            return o.value
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))
