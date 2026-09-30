"""QA items with ground truth, answer normalisation and matching, dedup and subsetting.

Item schema (dict): ``id, question, answers (aliases), domain, kind, template, group,
answer_type, source``. ``kind`` is one of

* ``answerable``   - has a ground-truth answer;
* ``fictional``    - asks about an invented entity using a real template (negative control);
* ``unanswerable`` - no answer exists or can exist yet (negative control);
* ``ambiguous``    - aleatoric: several defensible answers or missing context.

Matching is deliberately simple and is a known source of label noise; see
``negative_space/README.md`` ("Operational definitions") and the manual-check sample
written by ``evaluate.py``.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .config import DataConfig

KINDS = ("answerable", "fictional", "unanswerable", "ambiguous")

# --------------------------------------------------------------------------- #
# Normalisation and matching
# --------------------------------------------------------------------------- #
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "hundred": "100",
}
_ARTICLES = {"a", "an", "the"}

ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|dont|don’t) know\b", r"\bi'?m not sure\b", r"\bnot sure\b",
    r"\bunknown\b", r"\bno (?:such|known|real)\b", r"\bdoes(?:n't| not) exist\b",
    r"\bis not a real\b", r"\bisn't a real\b", r"\bfictional\b", r"\bcannot (?:be )?(?:know|determin|predict|answer)",
    r"\bcan't (?:be )?(?:know|determin|predict|answer)", r"\bimpossible to (?:know|determine|predict)\b",
    r"\bno (?:way|one) (?:to|can) know\b", r"\bnot (?:possible|able) to\b", r"\bi (?:have no|lack) (?:information|data)\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents/punctuation/articles, map small number words to digits."""
    s = _strip_accents(text).lower()
    s = s.replace("’", "'").replace("×", "x")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # 1,000 -> 1000
    s = re.sub(r"[^\w\s\-+.']", " ", s)
    s = re.sub(r"(?<!\d)\.(?!\d)", " ", s)              # drop sentence dots, keep decimals
    toks = [_NUMBER_WORDS.get(t, t) for t in s.replace("-", " ").split()]
    toks = [t.strip("'") for t in toks if t not in _ARTICLES]
    return " ".join(t for t in toks if t)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", normalize_answer(text))


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text)) or not normalize_answer(text)


def is_correct(prediction: str, item: dict) -> bool:
    """Ground-truth match. Only defined for items with at least one gold alias."""
    golds = item.get("answers") or []
    if not golds or is_abstention(prediction):
        return False
    atype = item.get("answer_type", "text")
    if atype == "number":
        nums = extract_numbers(prediction)
        if not nums:
            return False
        if item.get("domain") == "arithmetic":
            # Use the number after '=' if present, else the last number; operands are echoed often.
            after_eq = re.findall(r"=\s*(-?[\d,]+(?:\.\d+)?)", prediction)
            cand = after_eq[-1].replace(",", "") if after_eq else nums[-1]
            return any(_num_eq(cand, g) for g in golds)
        return any(_num_eq(n, g) for n in nums for g in golds)
    if atype == "symbol":
        # Case-sensitive token match on the raw text (chemical symbols).
        raw_toks = set(re.findall(r"[A-Za-z]+", prediction))
        return any(g in raw_toks for g in golds)
    pred = f" {normalize_answer(prediction)} "
    return any(f" {normalize_answer(g)} " in pred for g in golds)


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def answer_key(prediction: str, item: dict) -> str:
    """Equivalence key used for consistency and noise-stability.

    Correct answers share one key (so rephrasings are not counted as flips), abstentions
    share one key, and other answers are compared by their first 6 normalised tokens.
    """
    if is_abstention(prediction):
        return "<ABSTAIN>"
    if item.get("answers") and is_correct(prediction, item):
        return "<CORRECT>"
    return " ".join(normalize_answer(prediction).split()[:6])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bundled(path: str | Path) -> list[dict]:
    return load_jsonl(path)


def load_keko_honest(path: str | Path, n: int, seed: int) -> list[dict]:
    """Reuse keko's ``datasets/keko_honest_v1`` prompts as unanswerable controls.

    These prompts ("What is the exact weight of love?") have a different surface form
    from the factual templates, so they are *easy* controls; they are tagged with their
    own template so their effect can be separated in the analysis.
    """
    p = Path(path)
    if n <= 0 or not p.exists():
        return []
    rows = load_jsonl(p)
    prompts = sorted({r["prompt"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    out = []
    for i, q in enumerate(prompts[:n]):
        out.append(dict(id=f"kh{i:04d}", question=q, answers=[], domain="unanswerable_keko",
                        kind="unanswerable", template="keko_honest", group=f"keko_honest:{i}",
                        answer_type="text", source="keko_honest_v1"))
    return out


def load_hf_open_qa(name: str, n: int, seed: int) -> list[dict]:  # pragma: no cover - network
    """Optional: TriviaQA (rc.nocontext) or NQ-open validation items via ``datasets``."""
    from datasets import load_dataset

    if name == "trivia_qa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        rows = [dict(question=r["question"], answers=r["answer"]["aliases"]) for r in ds]
    elif name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        rows = [dict(question=r["question"], answers=list(r["answer"])) for r in ds]
    else:
        raise ValueError(f"unknown hf_dataset {name!r}")
    return [dict(id=f"{name}{i:05d}", domain=name, kind="answerable", template=name,
                 group=f"{name}:{i}", answer_type="text", source=name, **r)
            for i, r in enumerate(rows)]


def _tokens(q: str) -> set[str]:
    return set(normalize_answer(q).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def dedup(items: list[dict], threshold: float = 0.9) -> list[dict]:
    """Drop near-duplicate questions (token Jaccard >= threshold), keeping the first."""
    kept: list[dict] = []
    for it in items:
        if any(jaccard(it["question"], k["question"]) >= threshold for k in kept):
            continue
        kept.append(it)
    return kept


def stratified_subset(items: list[dict], n: int | None, seed: int) -> list[dict]:
    """Stratified by (domain, kind); proportional allocation, at least 1 per stratum."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        strata[(it["domain"], it["kind"])].append(it)
    total = len(items)
    chosen: list[dict] = []
    for key in sorted(strata):
        pool = strata[key][:]
        rng.shuffle(pool)
        k = max(1, round(n * len(pool) / total))
        chosen.extend(pool[:k])
    rng.shuffle(chosen)
    return sorted(chosen[:n], key=lambda it: it["id"])


def load_items(cfg: DataConfig, seed: int = 0) -> list[dict]:
    items = load_bundled(cfg.bundled_path)
    items += load_keko_honest(cfg.keko_honest_path, cfg.n_keko_honest, seed)
    if cfg.hf_dataset:
        items += load_hf_open_qa(cfg.hf_dataset, cfg.hf_n, seed)
    items = dedup(items, cfg.dedup_jaccard)
    return stratified_subset(items, cfg.n_items, seed)
