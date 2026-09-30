"""Stage 1: answers, k samples, hidden states and known/fragile/unknown labels.

Operational definitions (see README "Operational definitions"):

* answerable items (gold aliases exist, not ambiguous)
    - **known**:   greedy answer correct AND all k temperature samples correct;
    - **fragile**: greedy answer correct AND at least one of the k samples incorrect;
    - **unknown**: greedy answer incorrect (whether or not the samples agree).
* fictional / unanswerable items: **unknown** by construction. The model *errs* on
  them iff it gives a substantive answer instead of abstaining.
* ambiguous items (aleatoric): label ``ambiguous``; excluded from the primary
  known/unknown analysis.

``error`` (the target for error detection, N2/N5) = greedy answer is not correct for
answerable items (abstention counts as not correct), or is not an abstention for
fictional/unanswerable items. ``hallucination`` = error AND not an abstention.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .data import answer_key, is_abstention, is_correct

LABELS = ("known", "fragile", "unknown", "ambiguous")


def label_item(item: dict, greedy: str, samples: list[str]) -> dict:
    kind = item["kind"]
    abst = is_abstention(greedy)
    keys = [answer_key(s, item) for s in samples]
    gkey = answer_key(greedy, item)
    consistency = float(np.mean([k == gkey for k in keys])) if keys else 1.0
    counts = Counter(keys + [gkey])
    tot = sum(counts.values())
    sem_entropy = -sum(c / tot * math.log(c / tot) for c in counts.values())
    rec = dict(greedy=greedy, samples=samples, abstained=abst, consistency=consistency,
               semantic_entropy_proxy=sem_entropy, n_clusters=len(counts))
    if kind == "answerable":
        g_ok = is_correct(greedy, item)
        s_ok = [is_correct(s, item) for s in samples]
        label = "known" if g_ok and all(s_ok) else "fragile" if g_ok else "unknown"
        error = not g_ok
        rec.update(greedy_correct=g_ok, sample_correct_frac=float(np.mean(s_ok)) if s_ok else float(g_ok))
    elif kind in ("fictional", "unanswerable"):
        label, error = "unknown", not abst
        rec.update(greedy_correct=None, sample_correct_frac=None)
    else:  # ambiguous
        label, error = "ambiguous", False
        rec.update(greedy_correct=None, sample_correct_frac=None)
    rec.update(label=label, error=bool(error), hallucination=bool(error and not abst))
    return rec


def run_extract(cfg: Config, items: list[dict], model, tok) -> tuple[list[dict], dict[str, np.ndarray]]:
    from .modeling import build_prompts, generate, hidden_states

    prompts = build_prompts(tok, [it["question"] for it in items], cfg.model)
    g = generate(model, tok, prompts, max_new_tokens=cfg.model.max_new_tokens,
                 batch_size=cfg.model.batch_size, do_sample=False, seed=cfg.seed)
    k = cfg.extract.k_samples
    s = generate(model, tok, prompts, max_new_tokens=cfg.model.max_new_tokens,
                 batch_size=max(1, cfg.model.batch_size // max(1, k)), do_sample=True,
                 temperature=cfg.extract.temperature, top_p=cfg.extract.top_p,
                 num_return_sequences=k, seed=cfg.seed + 1)
    H = hidden_states(model, tok, prompts, cfg.extract.layers, cfg.extract.poolings,
                      cfg.model.batch_size)
    prompt_tokens = [len(tok(p).input_ids) for p in prompts]

    records = []
    for i, it in enumerate(items):
        rec = dict(it)
        rec.update(label_item(it, g.texts[i], s.texts[i * k:(i + 1) * k]))
        rec.update(first_max_prob=float(g.first_max_prob[i]), mean_max_prob=float(g.mean_max_prob[i]),
                   first_entropy=float(g.first_entropy[i]), mean_entropy=float(g.mean_entropy[i]),
                   mean_U=float(g.mean_U[i]), answer_tokens=int(g.n_tokens[i]),
                   prompt_tokens=int(prompt_tokens[i]), question_chars=len(it["question"]))
        records.append(rec)
    return records, H


def save_extract(path: Path, records: list[dict], H: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "hidden.npz", **H)
    with (path / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_extract(path: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    with (path / "records.jsonl").open() as f:
        records = [json.loads(l) for l in f]
    npz = np.load(path / "hidden.npz")
    return records, {k: npz[k] for k in npz.files}
