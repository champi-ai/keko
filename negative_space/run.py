"""CLI: run the negative-space pipeline in cached stages.

    uv run python -m negative_space.run --config negative_space/configs/smoke_cpu.yaml
    uv run python -m negative_space.run --config ... --stages evaluate,report   # reuse cached stages
    uv run python -m negative_space.run --config ... --force                    # recompute everything

Stages: ``extract`` (answers, samples, hidden states, labels) -> ``noise`` (progressive
noise) -> ``evaluate`` (N1-N5, fits negative spaces out-of-fold, saves a final
``NegativeSpace`` trained on all items) -> ``report`` (plots + results.md).
Model-dependent stages are cached under ``cache_dir/<name>/``; nothing large is
written to ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config

STAGES = ("extract", "noise", "evaluate", "report")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", action="store_true", help="ignore cached stages")
    ap.add_argument("--title", default=None)
    ap.add_argument("--preamble-file", default=None, help="markdown inserted at the top of results.md")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    cache, out = cfg.run_cache, Path(cfg.out_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (cache / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    timings: dict[str, float] = {}
    model = tok = None

    def ensure_model():
        nonlocal model, tok
        if model is None:
            from .modeling import load_model
            t = time.time()
            model, tok = load_model(cfg.model)
            timings["load_model_s"] = time.time() - t
        return model, tok

    from .extract import load_extract, run_extract, save_extract
    from .noise import load_noise, run_noise, save_noise

    if "extract" in stages and (args.force or not (cache / "records.jsonl").exists()):
        from .data import load_items
        items = load_items(cfg.data, cfg.seed)
        print(f"[extract] {len(items)} items")
        t = time.time()
        records, H = run_extract(cfg, items, *ensure_model())
        save_extract(cache, records, H)
        timings["extract_s"] = time.time() - t
    records, H = load_extract(cache)

    if "noise" in stages and (args.force or not (cache / "noise.npz").exists()):
        print(f"[noise] sites={cfg.noise.sites} levels={cfg.noise.levels}")
        t = time.time()
        save_noise(cache, run_noise(cfg, records, *ensure_model()))
        timings["noise_s"] = time.time() - t
    noise = load_noise(cache)

    from .evaluate import dump_json, evaluate, write_manual_check
    if "evaluate" in stages or not (cache / "results.json").exists():
        t = time.time()
        res = evaluate(cfg, records, H, noise)
        timings["evaluate_s"] = time.time() - t
        dump_json(cache / "results.json", res)
        _fit_final(cfg, records, H, noise, cache)
    res = json.loads((cache / "results.json").read_text())

    if "report" in stages:
        from .report import plot_all, results_markdown
        files = plot_all(res, out)
        pre = Path(args.preamble_file).read_text() if args.preamble_file else ""
        md = results_markdown(res, cfg, files, args.title or f"Negative-space results: {cfg.name}", pre)
        (out / "results.md").write_text(md)
        slim = {k: v for k, v in res.items() if not k.startswith("_")}
        dump_json(out / "results.json", slim)
        write_manual_check(out / "manual_check_sample.csv", records)
        _write_items(out / "items.jsonl", records, res)
        prev = cache / "timings.json"
        allt = json.loads(prev.read_text()) if prev.exists() else {}
        allt.update(timings)
        prev.write_text(json.dumps(allt, indent=2))
        dump_json(out / "timings.json", allt)
        print(f"[report] wrote {out / 'results.md'}")
    elif timings:
        prev = cache / "timings.json"
        allt = json.loads(prev.read_text()) if prev.exists() else {}
        allt.update(timings)
        prev.write_text(json.dumps(allt, indent=2))
    print("timings:", {k: round(v, 1) for k, v in timings.items()})


def _fit_final(cfg, records, H, noise, cache: Path) -> None:
    """Fit one NegativeSpace on all non-ambiguous items and save it (for downstream use)."""
    from .evaluate import flip_target
    from .negative_model import NegativeSpace
    from .promoters import KNOWLEDGE_PROMOTER

    idx = np.array([i for i, r in enumerate(records) if r["label"] != "ambiguous"])
    recs = [records[i] for i in idx]
    if sum(KNOWLEDGE_PROMOTER.positives(recs)) < 3:
        return
    X = H[f"L{cfg.analysis.layer}_{cfg.analysis.pooling}"][idx]
    ns = NegativeSpace(KNOWLEDGE_PROMOTER, cfg.analysis, seed=cfg.seed)
    ns.fit(X, recs, flip_target(noise, idx), np.array([r["group"] for r in recs]))
    ns.save(cache / "negative_space_knowledge.pkl")


def _write_items(path: Path, records: list[dict], res: dict) -> None:
    per = {p["id"]: p for p in res.get("_per_item", [])}
    with path.open("w") as f:
        for r in records:
            row = dict(id=r["id"], question=r["question"], kind=r["kind"], domain=r["domain"],
                       gold=r["answers"], greedy=r["greedy"], samples=r["samples"], label=r["label"],
                       error=r["error"], first_max_prob=round(r["first_max_prob"], 4))
            if r["id"] in per:
                row.update(region=per[r["id"]]["region"], keko_window=per[r["id"]]["keko_window"],
                           negative_score=per[r["id"]]["negative_score"],
                           flip_level=per[r["id"]]["flip_level"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
