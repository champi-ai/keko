"""Configuration for the negative-space experiment.

A config is a YAML file whose top-level keys mirror the dataclasses below. Any key
left out keeps its default, so a YAML file only needs to state what differs.
Defaults describe the CPU smoke configuration on the keko base model
(``HuggingFaceTB/SmolLM-360M-Instruct``, the same frozen base used by
``main.py`` / ``uncertain.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question with only the short answer. "
    "If you do not know the answer, or the question has no answer, reply exactly: I don't know."
)


@dataclass
class ModelConfig:
    id: str = "HuggingFaceTB/SmolLM-360M-Instruct"
    dtype: str = "float32"            # float32 on CPU; bfloat16 / float16 on GPU
    device_map: str | None = "cpu"    # "auto" for GPU / multi-GPU
    max_new_tokens: int = 16
    batch_size: int = 16
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat_template: bool = True
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    bundled_path: str = "negative_space/qa_data/qa_bundled.jsonl"
    # Extra unanswerable controls reused from keko's own honesty dataset.
    keko_honest_path: str = "datasets/keko_honest_v1/dataset.jsonl"
    n_keko_honest: int = 12
    # Stratified subset size (None = everything). Stratified by (domain, kind).
    n_items: int | None = None
    # Optional HF open-domain QA (requires network). One of: None, "trivia_qa", "nq_open".
    hf_dataset: str | None = None
    hf_n: int = 0
    dedup_jaccard: float = 0.9


@dataclass
class ExtractConfig:
    # hidden_states indices: 0 = embeddings, i = output of decoder layer i.
    layers: list[int] = field(default_factory=lambda: [4, 8, 16, 24, 32])
    poolings: list[str] = field(default_factory=lambda: ["last", "mean"])
    k_samples: int = 3
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class NoiseConfig:
    sites: list[str] = field(default_factory=lambda: ["embedding", "hidden"])
    hidden_layer: int = 16            # decoder layer whose output is perturbed
    # Relative noise scale sigma: noise = sigma * rms(h) * N(0, I), per token.
    levels: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4])
    n_draws: int = 1
    flip_threshold: float = 0.5       # item flips at a level when stability < threshold


@dataclass
class TopologyConfig:
    enabled: bool = True
    pca_dims: list[int] = field(default_factory=lambda: [3, 5, 8])
    primary_pca_dim: int = 5
    max_points: int = 300
    n_null: int = 20                  # null clouds per null model
    n_boot: int = 20                  # bootstrap resamples of the known cloud
    null_quantile: float = 0.95
    hole_core_frac: float = 0.6       # a point is "in" a hole if within this frac of its radius


@dataclass
class AnalysisConfig:
    layer: int = 16                   # primary layer for the negative model
    pooling: str = "last"
    pca_dim: int = 32                 # PCA before Mahalanobis (shrinkage on top)
    knn_k: int = 5
    ocsvm: bool = False
    probe: str = "logreg"             # "logreg" | "mlp"
    probe_C: float = 0.5
    n_folds: int = 5
    n_seeds: int = 3
    n_boot: int = 1000
    confidence_high: float = 0.7      # "confident" for hole detection (first-token max prob)
    outside_quantile: float = 0.95    # kNN-distance quantile of known items that bounds "inside"
    boundary_flip_quantile: float = 0.25
    keko_window: list[float] = field(default_factory=lambda: [0.3, 0.7])  # uncertain.py
    topology: TopologyConfig = field(default_factory=TopologyConfig)


@dataclass
class Config:
    name: str = "smollm_cpu"
    out_dir: str = "negative_space/results/smollm_cpu"
    cache_dir: str = "negative_space/.cache"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_cache(self) -> Path:
        return Path(self.cache_dir) / self.name


def _merge(dc_type, values: dict[str, Any] | None):
    obj = dc_type()
    if not values:
        return obj
    known = {f.name: f for f in fields(dc_type)}
    for key, val in values.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {dc_type.__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            val = _merge(type(cur), val)
        setattr(obj, key, val)
    return obj


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config (or defaults when ``path`` is None) and apply overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides)
    return _merge(Config, values)
