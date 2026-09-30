"""Plots (matplotlib) and the results markdown table."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Categorical order validated for CVD separation (dataviz reference palette, light mode).
C_KNOWN, C_FRAGILE, C_UNKNOWN, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LABEL_COLORS = {"known": C_KNOWN, "fragile": C_FRAGILE, "unknown": C_UNKNOWN}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_all(res: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = []
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "text.color": INK,
                         "axes.labelcolor": INK2})

    # 1. stability vs noise curves (N3)
    curves = res["_curves_for_plot"]
    levels = res["N3"]["levels"]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.2), squeeze=False)
    for ax, (site, cv) in zip(axes[0], curves.items()):
        _style(ax)
        counts = res["N3"]["sites"][site]["counts"]
        for lab, ys in cv.items():
            ax.plot(levels, ys, color=LABEL_COLORS[lab], lw=2, marker="o", ms=5,
                    label=f"{lab} (n={counts[lab]})")
            ax.annotate(lab, (levels[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.set_title(f"noise site: {site}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("relative noise scale σ (× activation RMS)")
        ax.set_ylabel("answer stability")
        ax.set_ylim(-0.03, 1.05)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out / "stability_vs_noise.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    # 2. AUROC for error detection with 95% CI (N2 + baselines)
    rows = [r for r in res["N2_error_detection"] if not np.isnan(r["auroc"])]
    rows = sorted(rows, key=lambda r: r["auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.2))
    _style(ax)
    for i, r in enumerate(rows):
        col = C_KNOWN if r["kind"] == "ours" else INK2
        ax.plot([r["auroc_lo"], r["auroc_hi"]], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["auroc"], i, "o", color=col, ms=6)
        ax.annotate(f"{r['auroc']:.2f}", (r["auroc_hi"], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["score"] for r in rows], fontsize=7)
    ax.set_xlabel("AUROC for detecting errors (95% bootstrap CI)   blue = negative-space scores, grey = baselines")
    ax.set_xlim(0, 1.08)
    fig.tight_layout()
    p = out / "auroc_error_detection.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)

    # 3. layer-wise
    lw = [r for r in res["layerwise"] if r["layer_key"].endswith("_last")]
    if lw:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        _style(ax)
        xs = [int(r["layer_key"].split("_")[0][1:]) for r in lw]
        for key, col, lab in (("probe_auroc_notknown", C_KNOWN, "probe: known vs not (N1)"),
                              ("maha_auroc_error", C_FRAGILE, "Mahalanobis: error (N2)"),
                              ("knn_auroc_error", C_UNKNOWN, "kNN: error (N2)")):
            ax.plot(xs, [r[key] for r in lw], color=col, lw=2, marker="o", ms=5, label=lab)
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("layer (hidden_states index, last prompt token)")
        ax.set_ylabel("AUROC (out-of-fold)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / "layerwise_auroc.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p.name)

    # 4. persistence diagram vs null threshold (N4)
    tp = res.get("_topo_plot")
    if tp and tp["available"] and tp["h1"]:
        h1 = np.array(tp["h1"])
        fig, ax = plt.subplots(figsize=(3.8, 3.6))
        _style(ax)
        ax.plot(h1[:, 0], h1[:, 1], "o", color=C_KNOWN, ms=6, label="H1 features (known cloud)")
        hi = float(np.nanmax(h1)) * 1.1 if len(h1) else 1.0
        ax.plot([0, hi], [0, hi], color=INK2, lw=0.8)
        if np.isfinite(tp["thr"]):
            ax.plot([0, hi], [tp["thr"], hi + tp["thr"]], color=C_FRAGILE, lw=1.5, ls="--",
                    label=f"null {int(95)}% threshold ({tp['thr']:.3f})")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        p = out / "persistence_h1_vs_null.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p.name)

    # 5. region counts by label (N5 illustrative)
    br = res["N5_illustrative"]["by_region"]
    regs = list(br)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _style(ax)
    bottom = np.zeros(len(regs))
    for lab, col in LABEL_COLORS.items():
        vals = np.array([br[r]["labels"][lab] for r in regs], dtype=float)
        ax.bar(regs, vals, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=2, label=lab, width=0.6)
        bottom += vals
    for i, r in enumerate(regs):
        ax.annotate(str(br[r]["n"]), (i, bottom[i]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_ylabel("items")
    ax.set_title("negative-promoter region assignment (illustrative)", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out / "promoter_regions.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p.name)
    return files


def _fmt(x, nd=2):
    try:
        return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def _ci(r, k="auroc"):
    return f"{_fmt(r[k])} [{_fmt(r[k + '_lo'])}, {_fmt(r[k + '_hi'])}]"


def results_markdown(res: dict, cfg, files: list[str], title: str, preamble: str = "") -> str:
    L = [f"# {title}", ""]
    if preamble:
        L += [preamble, ""]
    L += ["## Setup", "",
          f"- model: `{cfg.model.id}` ({cfg.model.dtype}, device_map={cfg.model.device_map})",
          f"- items: {res['n_items']} total, {res['n_analysed']} analysed (ambiguous excluded: {res['ambiguous_n']})",
          f"- kinds: {res['kind_counts']}",
          f"- labels: {res['label_counts']}",
          f"- error rate (analysed): {_fmt(res['error_rate'])}; hallucination rate: {_fmt(res['hallucination_rate'])}",
          f"- primary representation: `{res['layer_key']}`; k samples = {cfg.extract.k_samples} "
          f"(T={cfg.extract.temperature}); noise levels = {cfg.noise.levels}, sites = {cfg.noise.sites}, "
          f"draws = {cfg.noise.n_draws}",
          f"- grouped {cfg.analysis.n_folds}-fold CV × {cfg.analysis.n_seeds} seeds; bootstrap n = {cfg.analysis.n_boot}",
          ""]
    n1 = res["N1"]
    L += ["## N1: known-vs-not decodability", "", "| score | AUROC [95% CI] | AUPRC [95% CI] | n |", "|---|---|---|---|"]
    for key in ("probe", "probe_answerable_only", "surface"):
        if key in n1:
            r = n1[key]
            L.append(f"| {r['score']} | {_ci(r)} | {_ci(r, 'auprc')} | {r['n']} |")
    d = n1["delta_vs_surface"]
    s = n1["selectivity"]
    L += ["", f"- probe - surface: ΔAUROC = {_fmt(d[0])} [{_fmt(d[1])}, {_fmt(d[2])}], bootstrap p = {_fmt(d[3], 3)}",
          f"- control task (Hewitt & Liang style, random label per template): task AUROC {_fmt(s['task_auroc'])}, "
          f"control AUROC {_fmt(s['control_auroc'])}, selectivity {_fmt(s['selectivity'])}", ""]
    L += ["### Layer-wise (out-of-fold, seed 0)", "", "| representation | probe AUROC (not known) | Mahalanobis AUROC (error) | kNN AUROC (error) |",
          "|---|---|---|---|"]
    for r in res["layerwise"]:
        L.append(f"| `{r['layer_key']}` | {_fmt(r['probe_auroc_notknown'])} | {_fmt(r['maha_auroc_error'])} | {_fmt(r['knn_auroc_error'])} |")
    L += ["", "## N2: error detection vs baselines", "", "| score | type | AUROC [95% CI] | AUPRC [95% CI] |", "|---|---|---|---|"]
    for r in res["N2_error_detection"]:
        L.append(f"| {r['score']} | {r['kind']} | {_ci(r)} | {_ci(r, 'auprc')} |")
    L += ["", "Paired bootstrap ΔAUROC (ours - baseline):", "", "| ours | baseline | ΔAUROC [95% CI] | p |", "|---|---|---|---|"]
    for r in res["N2_paired"]:
        L.append(f"| {r['ours']} | {r['baseline']} | {_fmt(r['delta'])} [{_fmt(r['lo'])}, {_fmt(r['hi'])}] | {_fmt(r['p'], 3)} |")
    c = res["calibration"]
    L += ["", f"Calibration (ECE, 10 bins): negative score {_fmt(c['negative_score_ece'][0])} "
          f"[{_fmt(c['negative_score_ece'][1])}, {_fmt(c['negative_score_ece'][2])}]; "
          f"1 - max prob {_fmt(c['one_minus_maxprob_ece'][0])} "
          f"[{_fmt(c['one_minus_maxprob_ece'][1])}, {_fmt(c['one_minus_maxprob_ece'][2])}]", ""]
    L += ["### Held-out domain (train on the other domains)", "", "| held-out domain | n | error rate | negative score AUROC | entropy AUROC | 1 - max prob AUROC |",
          "|---|---|---|---|---|---|"]
    for r in res["held_out_domain"]:
        L.append(f"| {r['domain']} | {r['n']} | {_fmt(r['error_rate'])} | {_fmt(r['negative_score_auroc'])} | "
                 f"{_fmt(r['entropy_auroc'])} | {_fmt(r['maxprob_auroc'])} |")
    L += ["", "## N3: progressive noise boundary", "", "| site | median flip σ by label | AUROC flip→not known [CI] | AUROC flip→error [CI] | MWU p (known > not) | censored | ρ(flip, answer tokens) | ρ(flip, prompt tokens) |",
          "|---|---|---|---|---|---|---|---|"]
    for site, v in res["N3"]["sites"].items():
        med = ", ".join(f"{k}: {_fmt(x, 3)}" for k, x in v["median_flip_by_label"].items())
        a, b = v["flip_auroc_notknown"], v["flip_auroc_error"]
        L.append(f"| {site} | {med} | {_fmt(a[0])} [{_fmt(a[1])}, {_fmt(a[2])}] | {_fmt(b[0])} [{_fmt(b[1])}, {_fmt(b[2])}] | "
                 f"{_fmt(v['mannwhitney_known_gt_notknown_p'], 3)} | {_fmt(v['censored_frac'])} | "
                 f"{_fmt(v['spearman_flip_vs_answer_tokens'])} | {_fmt(v['spearman_flip_vs_prompt_tokens'])} |")
    L += ["", f"Flip-level regression from activations (out-of-fold Spearman ρ): {_fmt(res['N3']['flip_regression_spearman_oof'])}", ""]
    if "N4" in res:
        L += ["## N4: topology of the known region (exploratory)", "", "| PCA dim | points | H1 features | max H1 persistence | null threshold | significant holes | bootstrap frac > null |",
              "|---|---|---|---|---|---|---|"]
        for r in res["N4_sensitivity"]:
            L.append(f"| {r['pca_dim']} | {r['n_points']} | {r['n_h1']} | {_fmt(r['max_h1_persistence'], 3)} | "
                     f"{_fmt(r['null_threshold'], 3)} | {r['n_significant_holes']} | {_fmt(r['bootstrap_frac_above_null'])} |"
                     + (f" {r['note']}" if r.get("note") else ""))
        n4 = res["N4"]
        L += ["", f"Primary (PCA {n4['primary']['pca_dim']}): not-known items inside significant holes: {n4['n_notknown_in_holes']} "
              f"(confident: {n4['n_confident_in_holes']}); error rate in holes: {_fmt(n4['error_rate_in_holes'])}.", ""]
    n5 = res["N5_illustrative"]
    L += ["## N5 (illustrative only): negative-promoter regions vs the keko 0.3–0.7 window", "",
          "| region | promoted behaviour | n | error rate | known / fragile / unknown |", "|---|---|---|---|---|"]
    from .promoters import KNOWLEDGE_PROMOTER, Region
    for rg in Region:
        b = n5["by_region"][rg.value]
        lab = b["labels"]
        L.append(f"| {rg.value} | `{KNOWLEDGE_PROMOTER.behaviours[rg].value}` | {b['n']} | {_fmt(b['error_rate'])} | "
                 f"{lab['known']} / {lab['fragile']} / {lab['unknown']} |")
    L += ["", "Behaviour promoted by the negative space (rows) vs what the keko window would trigger (columns):", ""]
    cols = sorted({k for v in n5["behaviour_vs_keko_window"].values() for k in v})
    L += ["| negative promoter \\ keko window | " + " | ".join(f"`{c}`" for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for rb, v in n5["behaviour_vs_keko_window"].items():
        L.append(f"| `{rb}` | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |")
    L += ["", f"- keko window intervenes on {_fmt(100 * n5['keko_window_intervention_rate'], 1)}% of items and covers "
          f"{n5['keko_window_errors_caught']} errors; the negative promoter intervenes on "
          f"{_fmt(100 * n5['promoter_intervention_rate'], 1)}% and covers {n5['promoter_errors_caught']} errors "
          "(different intervention rates, so not directly comparable; see matched rates below).", "",
          "| intervention rate | items | errors (total) | caught by top negative score | caught by lowest max prob |", "|---|---|---|---|---|"]
    for m in n5["matched_rate"]:
        L.append(f"| {_fmt(m['rate'])} | {m['n_intervene']} | {m['n_errors']} | {m['errors_caught_negative_score']} | {m['errors_caught_low_maxprob']} |")
    L += ["", "## Figures", ""] + [f"![{f}]({f})" for f in files] + [""]
    return "\n".join(L)
