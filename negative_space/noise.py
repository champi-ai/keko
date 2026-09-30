"""Stage 2: progressive noise injection - the empirical boundary probe (N3).

Two sites:

* ``embedding`` - output of the input-embedding layer;
* ``hidden``    - output of decoder layer ``noise.hidden_layer``.

At relative scale sigma, each token's activation h (dimension d) receives
``sigma * rms(h) * eps`` with ``eps ~ N(0, I_d)`` and ``rms(h) = ||h|| / sqrt(d)``, so the
expected noise norm is ``sigma * ||h||`` whatever the layer's natural scale is. The
noise is applied per token at every forward call during greedy decoding, so it is
independent of prompt length and answer length by construction (the residual
dependence on length is measured and reported, not assumed away).

Per item and level we record *stability* (fraction of draws whose answer key equals
the clean greedy answer key, see ``data.answer_key``) and first-token entropy. The
**flip level** is the smallest sigma at which stability drops below
``flip_threshold``; items that never flip are right-censored at ``2 * max(levels)``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import answer_key


def validate_schedule(levels: list[float]) -> list[float]:
    lv = [float(x) for x in levels]
    if not lv or lv[0] != 0.0:
        raise ValueError("noise schedule must start at 0.0 (the clean reference)")
    if any(b <= a for a, b in zip(lv, lv[1:])):
        raise ValueError("noise schedule must be strictly increasing")
    return lv


def add_relative_noise(h: torch.Tensor, sigma: float, generator: torch.Generator | None) -> torch.Tensor:
    if sigma <= 0:
        return h
    rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
    eps = torch.randn(h.shape, generator=generator, device=h.device, dtype=torch.float32)
    return (h.float() + sigma * rms * eps).to(h.dtype)


@contextmanager
def inject_noise(model, site: str, sigma: float, seed: int, hidden_layer: int = 16):
    """Context manager registering a forward hook that perturbs the chosen site."""
    from .modeling import decoder_layers, model_device

    gen = torch.Generator(device=model_device(model))
    gen.manual_seed(seed)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (add_relative_noise(output[0], sigma, gen),) + tuple(output[1:])
        return add_relative_noise(output, sigma, gen)

    if site == "embedding":
        module = model.get_input_embeddings()
    elif site == "hidden":
        module = decoder_layers(model)[hidden_layer - 1]  # hidden_states[L] = output of block L-1
    else:
        raise ValueError(f"unknown noise site {site!r}")
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def flip_level(levels: list[float], stability: np.ndarray, threshold: float) -> tuple[float, bool]:
    """Smallest level with stability < threshold; (2*max_level, True) if never."""
    for lv, st in zip(levels, stability):
        if lv > 0 and st < threshold:
            return float(lv), False
    return 2.0 * float(levels[-1]), True


def stability_auc(levels: list[float], stability: np.ndarray) -> float:
    """Normalised area under the stability-vs-sigma curve (1 = never moves)."""
    x = np.asarray(levels, dtype=float)
    y = np.asarray(stability, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def run_noise(cfg: Config, records: list[dict], model, tok) -> dict:
    from .modeling import build_prompts, generate

    levels = validate_schedule(cfg.noise.levels)
    prompts = build_prompts(tok, [r["question"] for r in records], cfg.model)
    clean_keys = [answer_key(r["greedy"], r) for r in records]
    n, L, D = len(records), len(levels), cfg.noise.n_draws
    out: dict = {"levels": levels, "sites": {}}
    for site in cfg.noise.sites:
        stab = np.ones((n, L), dtype=np.float32)
        ent = np.zeros((n, L), dtype=np.float32)
        ent[:, 0] = [r["first_entropy"] for r in records]
        answers: list[list[list[str]]] = [[[] for _ in range(L)] for _ in range(n)]
        for li, sigma in enumerate(levels):
            if sigma == 0:
                continue
            same = np.zeros((n, D), dtype=np.float32)
            ents = np.zeros((n, D), dtype=np.float32)
            for d in range(D):
                seed = cfg.seed * 100_003 + li * 1_009 + d * 17 + (0 if site == "embedding" else 7)
                with inject_noise(model, site, sigma, seed, cfg.noise.hidden_layer):
                    g = generate(model, tok, prompts, max_new_tokens=cfg.model.max_new_tokens,
                                 batch_size=cfg.model.batch_size, do_sample=False, seed=seed)
                for i, r in enumerate(records):
                    same[i, d] = float(answer_key(g.texts[i], r) == clean_keys[i])
                    answers[i][li].append(g.texts[i])
                ents[:, d] = g.first_entropy
            stab[:, li] = same.mean(1)
            ent[:, li] = ents.mean(1)
        flips = [flip_level(levels, stab[i], cfg.noise.flip_threshold) for i in range(n)]
        out["sites"][site] = dict(
            stability=stab, entropy=ent,
            flip_level=np.array([f for f, _ in flips], dtype=np.float32),
            censored=np.array([c for _, c in flips]),
            stability_auc=np.array([stability_auc(levels, stab[i]) for i in range(n)], dtype=np.float32),
            answers=answers,
        )
    return out


def save_noise(path: Path, res: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"levels": np.asarray(res["levels"], dtype=np.float32)}
    answers = {}
    for site, d in res["sites"].items():
        for k in ("stability", "entropy", "flip_level", "censored", "stability_auc"):
            arrays[f"{site}__{k}"] = d[k]
        answers[site] = d["answers"]
    np.savez_compressed(path / "noise.npz", **arrays)
    (path / "noise_answers.json").write_text(json.dumps(answers, ensure_ascii=False))


def load_noise(path: Path) -> dict:
    npz = np.load(path / "noise.npz")
    res: dict = {"levels": npz["levels"].tolist(), "sites": {}}
    for key in npz.files:
        if "__" in key:
            site, k = key.split("__", 1)
            res["sites"].setdefault(site, {})[k] = npz[key]
    return res
