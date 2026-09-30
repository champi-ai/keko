"""Model loading, prompting, generation with logit statistics, hidden-state capture.

Loading follows the conventions of ``uncertainty_probe.py`` / ``uncertain.py``
(frozen base, ``pad_token = eos_token``, left padding for batched generation).
Uncertainty statistics follow KEKO v4.2 §8 (normalised entropy, max confidence,
top-2 margin and the combined score U).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig


def load_model(cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.id,
        dtype=getattr(torch, cfg.dtype),
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.eval()
    for p in model.parameters():  # frozen base, as in keko
        p.requires_grad_(False)
    return model, tok


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_prompts(tok, questions: list[str], cfg: ModelConfig) -> list[str]:
    if cfg.use_chat_template and getattr(tok, "chat_template", None):
        def render(q: str) -> str:
            try:
                return tok.apply_chat_template(
                    [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:  # templates without a system role (e.g. Mistral)
                return tok.apply_chat_template(
                    [{"role": "user", "content": f"{cfg.system_prompt}\n\n{q}"}],
                    tokenize=False, add_generation_prompt=True)
        return [render(q) for q in questions]
    return [f"{cfg.system_prompt}\nQuestion: {q}\nAnswer:" for q in questions]


def decoder_layers(model):
    """Return the list of decoder blocks for Llama/Qwen/Mistral/SmolLM-style models."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers on this model")


# --------------------------------------------------------------------------- #
# Logit statistics (KEKO v4.2 §8)
# --------------------------------------------------------------------------- #
def token_stats(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """logits [..., V] -> dict of max_prob, norm_entropy, margin, U (same leading shape)."""
    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(probs * logp).sum(-1) / math.log(probs.shape[-1])
    top2 = probs.topk(2, dim=-1).values
    maxp, margin = top2[..., 0], top2[..., 0] - top2[..., 1]
    U = 0.50 * ent + 0.35 * (1 - maxp) + 0.15 * (1 - margin)
    return {"max_prob": maxp, "norm_entropy": ent, "margin": margin, "U": U}


@dataclass
class GenOutput:
    texts: list[str]
    first_max_prob: np.ndarray
    mean_max_prob: np.ndarray
    first_entropy: np.ndarray
    mean_entropy: np.ndarray
    mean_U: np.ndarray
    n_tokens: np.ndarray


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, max_new_tokens: int, batch_size: int,
             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
             num_return_sequences: int = 1, seed: int = 0) -> GenOutput:
    """Batched generation. Statistics use the *raw* logits of each generated token."""
    dev = model_device(model)
    texts, fmp, mmp, fen, men, mU, ntok = [], [], [], [], [], [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, return_token_type_ids=False).to(dev)
        torch.manual_seed(seed + start)
        kw = dict(do_sample=do_sample, max_new_tokens=max_new_tokens,
                  num_return_sequences=num_return_sequences, pad_token_id=tok.pad_token_id,
                  return_dict_in_generate=True, output_logits=True)
        if do_sample:
            kw.update(temperature=temperature, top_p=top_p)
        else:
            kw.update(temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, **kw)
        gen = out.sequences[:, enc.input_ids.shape[1]:]
        logits = torch.stack(out.logits, dim=1)            # [B*R, T, V]
        stats = token_stats(logits)
        eos_ids = {tok.eos_token_id, tok.pad_token_id}
        for i in range(gen.shape[0]):
            ids = gen[i].tolist()
            n = next((j for j, t in enumerate(ids) if t in eos_ids), len(ids))
            n_eff = max(n, 1)
            texts.append(tok.decode(ids[:n], skip_special_tokens=True).strip())
            fmp.append(stats["max_prob"][i, 0].item())
            mmp.append(stats["max_prob"][i, :n_eff].mean().item())
            fen.append(stats["norm_entropy"][i, 0].item())
            men.append(stats["norm_entropy"][i, :n_eff].mean().item())
            mU.append(stats["U"][i, :n_eff].mean().item())
            ntok.append(n)
    arr = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    return GenOutput(texts, arr(fmp), arr(mmp), arr(fen), arr(men), arr(mU), np.asarray(ntok))


@torch.no_grad()
def hidden_states(model, tok, prompts: list[str], layers: list[int], poolings: list[str],
                  batch_size: int) -> dict[str, np.ndarray]:
    """Capture prompt representations (before any answer token) at chosen layers.

    Returns ``{"L{layer}_{pooling}": array [n, d]}``; ``last`` is the final prompt
    token (the position that emits the first answer token), ``mean`` is the
    attention-masked mean over prompt tokens.
    """
    dev = model_device(model)
    n_layers = model.config.num_hidden_layers
    layers = [l if l >= 0 else n_layers + 1 + l for l in layers]
    out: dict[str, list[np.ndarray]] = {f"L{l}_{p}": [] for l in layers for p in poolings}
    for start in range(0, len(prompts), batch_size):
        enc = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True, return_token_type_ids=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc.attention_mask.unsqueeze(-1).float()
        for l in layers:
            h = hs[l].float()
            if "last" in poolings:
                out[f"L{l}_last"].append(h[:, -1, :].cpu().numpy())   # left padding
            if "mean" in poolings:
                out[f"L{l}_mean"].append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
