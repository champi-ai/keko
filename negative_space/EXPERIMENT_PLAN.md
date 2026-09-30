# Experiment plan and pre-registration: negative space / behavioural promoters

This file fixes the hypotheses, primary metrics, success and failure criteria, and
the confirmatory/exploratory split **before** the confirmatory (GPU) run is analysed.
It was committed before the smoke run was executed. It is not to be edited after
confirmatory results exist, except for an appended, dated "Deviations" section.
Null or negative results are reported as such.

Theory, definitions and threats to validity are in [README.md](README.md).

---

## 0. What the smoke run is and is not

The CPU smoke run (SmolLM-360M-Instruct, ~120 items, k = 3, 4 noise levels) checks that
every stage runs and that outputs have the right shape. Its numbers are **not**
evidence for or against N1–N6. The confirmatory analysis is the GPU run below. The
smoke run is also not used to tune any threshold listed in §3.

## 1. Models

| role | model | why |
|---|---|---|
| primary | **Qwen2.5-7B-Instruct** | strong open 7B, permissive licence, 28 layers |
| replication | **Llama-3.1-8B-Instruct** | different family and tokenizer (gated licence) |
| replication | **Mistral-7B-Instruct-v0.3** | third family; no system role (handled) |
| scale point (48 GB) | Qwen2.5-14B-Instruct | same family, 2× parameters |
| keko base (sanity) | SmolLM-360M-Instruct | keko's frozen base; CPU |

A hypothesis is **confirmed** only if it meets its criterion on the primary model
**and** on at least one replication model.

## 2. Data and sizes

* **Bundled benchmark** (`qa_data/qa_bundled.jsonl`, ~296 self-authored items) plus
  40 keko_honest_v1 prompts as easy unanswerable controls.
* **TriviaQA** (rc.nocontext, validation): 2,000 items (`hf_dataset: trivia_qa`). NQ-open
  2,000 items as a second open-domain set.
* **Expanded matched controls.** Before the run we will generate 300 additional
  fictional-entity items with the bundled templates and 100 more ambiguous items. They
  will be frozen in a new JSONL, committed with a hash, and never edited afterwards.
* k = 10 samples (T = 0.7, top-p 0.95); noise levels σ ∈ {0, 0.05, 0.1, 0.2, 0.35, 0.5,
  0.75, 1.0} at 2 sites × 3 draws.
* Layers: every 3–4 layers, including embeddings-adjacent and final. **Primary layer
  fixed in advance: the middle layer** (Qwen 14, Llama/Mistral 16), last prompt token.
* **Manual label audit.** 200 items per model are checked by hand. We report agreement
  with the automatic matcher, and results with and without disagreeing items.

## 3. Hypotheses, primary metrics and criteria

The error target is defined in README §5.1. All CIs are 95% percentile bootstrap (2,000
resamples) over items, and splits are grouped by entity, 5 folds × 5 seeds. "Beats"
means the lower CI bound of the paired bootstrap ΔAUROC is above 0. The
primary-comparison family is Holm-corrected over the 6 confirmatory tests below.

| # | hypothesis | primary metric | success criterion | failure criterion | status |
|---|---|---|---|---|---|
| N1 | known vs not-known is probe-decodable | out-of-fold AUROC of a logistic probe at the primary layer, target = not-known, **answerable items only** | AUROC ≥ 0.75, beats the surface-only baseline, **and** control-task selectivity ≥ 0.10 | ΔAUROC vs surface CI includes 0, or selectivity < 0.05 | confirmatory |
| N2 | distance to known predicts errors, beyond logits | AUROC for error of Mahalanobis-LW distance (primary layer, PCA 64) | beats mean token entropy **and** first-token max-prob | CI of Δ vs entropy includes 0 or is negative | confirmatory |
| N3 | noise traces the boundary | (a) one-sided Mann–Whitney, flip level known > not-known; (b) AUROC of −flip level for error; (c) length control | (a) p < 0.01 at **both** sites; (b) AUROC ≥ 0.65; (c) effect survives a logistic regression with log answer length and prompt length as covariates (flip coefficient CI excludes 0) | any of (a)–(c) fails | confirmatory |
| N4 | holes in the known region; items in holes are confident errors | (a) number of H1 features above the null 95% threshold with bootstrap reproducibility ≥ 0.8, in ≥ 2 of PCA dims {3, 5, 8, 12}; (b) error rate of confident items in holes vs confidence-matched items outside holes | (a) ≥ 1 such feature **and** (b) higher error rate, Fisher exact p < 0.05 | no feature survives nulls, or (b) not higher | **exploratory** (no confirmatory claim regardless of outcome) |
| N5a | region routing beats the keko window at intercepting errors | errors caught at an intervention rate **matched** to the keko 0.3–0.7 window's own rate; paired bootstrap over items | negative-score routing catches more errors (CI of the difference > 0) | CI includes 0 | confirmatory |
| N5b | targeted hole replay beats random replay | error rate on held-out paraphrases of hole items after training keko columns on UB samples weighted by hole persistence, vs the same budget of randomly chosen UB samples | lower error, paired over 5 seeds, CI > 0 | otherwise | exploratory (depends on N4) |
| N6 | the pipeline generalises to a second promoter | see §5 | see §5 | see §5 | planned; confirmatory in a follow-up |

**Secondary metrics** (reported, not tested): AUPRC, ECE of the negative score,
layer-wise curves, held-out-domain AUROC, per-site and per-scale sensitivity, and
results without keko_honest controls.

## 4. Ablations

1. Noise site: embedding only vs hidden only vs both; hidden-layer depth (¼, ½, ¾).
2. Noise scale: flip-level definitions with thresholds 0.5 and 0.67; draws 1 vs 3.
3. Representation: last token vs mean-pooled; each captured layer.
4. Geometry: kNN k ∈ {3, 5, 10}; Mahalanobis PCA dims {16, 32, 64, 128}; shrinkage on vs off;
   per-coordinate standardisation vs a single global scale (restores rotation invariance).
5. Topology: PCA dims {3, 5, 8, 12}; max points; null model (Gaussian vs shuffle);
   filtration (Vietoris–Rips vs a witness complex on landmarks).
6. Controls: with and without keko_honest; answerable-only; **template-held-out** splits
   (train on some templates, test on others).
7. Combiner: learned logistic combiner vs equal-weight rank average.
8. Baselines: add the NLI-clustered semantic entropy of Farquhar et al. (2024) on the
   primary model, in place of the string-cluster proxy.

## 5. N6 design (planned, not run): a second promoter, clarification-seeking

* **Positives:** prompts with a single clear interpretation (unambiguous paraphrases
  of the bundled items, plus instruction-following prompts with complete context).
* **Negatives for evaluation only** (never used to build the region): ambiguous prompts
  with ≥ 2 defensible interpretations. These are the bundled `ambiguous` items plus 300
  new ones, balanced across lexical ambiguity, missing referent and under-specified
  quantity.
* **Construction:** the same `NegativeSpace` pipeline with `Promoter(name="clarify", …)`.
  Noise flip here means that the *interpretation* changes; it is measured by clustering
  sampled answers.
* **Comparison:** a single steering or probing *direction* computed from the same
  positives and a matched set of negatives, i.e. a difference of means (ActAdd/RepE
  style), thresholded to the same intervention rate.
* **Metric:** ambiguous prompts caught (clarification triggered) at matched
  intervention rates of 10%, 20% and 30%. Paired bootstrap.
* **Success:** the region beats the direction at ≥ 2 of 3 rates, on both the primary
  and one replication model.

## 6. N5 behavioural protocol (planned for the GPU run)

1. Fit `NegativeSpace` on training folds and assign each test item a region.
2. **Interception at matched rates.** For each method (negative-score ranking,
   keko window, low max-prob ranking), intervene on the same number of items and count
   errors intercepted. Abstention on an error counts as an avoided error; abstention on
   a correct answer counts as a cost.
3. **Replay (N5b).** Items in holes become UB rows (`to_ub_row`) weighted by
   persistence. We then train keko columns (`uncertain.py` background-training loop,
   base frozen) for a fixed step budget, and evaluate on held-out paraphrases
   against a random-replay control with the same budget.

## 7. Compute estimate

Setup: HF `generate` with hooks (noise injection needs forward hooks, so no vLLM),
bf16, batch 32, ≤ 24 new tokens.

| item | per model (7–8B, 1× 24 GB GPU) |
|---|---|
| items | ~2,300 bundled + TriviaQA (+ 2,000 NQ-open in the replication pass) |
| generations per item | 1 greedy + 10 samples + 7 levels × 2 sites × 3 draws = 53 |
| total generations | ~120k (~3,800 batches) |
| wall time for extract + noise | ~1.5–3 GPU-hours (≈ 0.6–1.5 s per batch of 32 at 24 tokens) |
| evaluation (CPU) | < 30 min, topology included |

Totals, including headroom for re-runs:

* primary + 2 replications ≈ **6–10 GPU-hours**;
* ablations ≈ **8–12 GPU-hours**;
* N5b replay training (keko columns, frozen base) ≈ **4–8 GPU-hours**;
* the 14B scale point (48 GB GPU) ≈ **4–6 GPU-hours**.

**Total ≈ 25–35 GPU-hours** on a single 24/48 GB card. These are estimates to be
replaced by measured timings (`timings.json` is written by every run).

## 8. Paper outline

**Title candidates**

* *Drawing the Negative Space: Behavioural Promoters from the Shape of What a Model Knows*
* *Holes in Known Territory: Topological Negative Spaces for LLM Self-Knowledge*
* *Negative Promoters: Region-Triggered Developmental Behaviour from Positive Samples*

**Sections**

1. Introduction: positive vs negative space; behavioural promoters; contributions.
2. Background: self-knowledge in LLMs; OOD distance; TDA; region-triggered
   interventions (CAST, circuit breakers); keko and grounding promoters.
3. Method: promoters; constructing a negative space (geometry, noise boundary,
   topology with nulls); the negative model; region → behaviour mapping; invariances.
4. Experimental setup: models, data, operational labels, controls, pre-registration.
5. Results: N1–N3 (confirmatory), N5a (confirmatory), N4/N5b (exploratory).
6. Generalisation to a second promoter (N6).
7. Threats to validity, limitations, negative results.
8. Related work and positioning.

**Figures**

1. Concept figure: positive cloud with boundary, hole and outside; the four regions and their behaviours.
2. Stability-vs-noise curves by label and site (N3), with the flip-level distribution.
3. Layer-wise AUROC (probe, kNN, Mahalanobis) vs logit baselines (N1/N2).
4. Persistence diagrams vs null threshold bands; bootstrap reproducibility (N4).
5. Errors intercepted vs intervention rate: negative score vs keko window vs max-prob (N5a).
6. Replay: error reduction for targeted vs random replay (N5b).
7. N6: region vs direction at matched rates for the clarification promoter.

## 9. Deviations

*(None yet. Any deviation from §1–§6 is appended here, dated, with its reason, before
the corresponding results are reported.)*
