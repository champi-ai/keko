# Negative space: behavioural promoters defined by constructing their negative space

*An experiment track of [keko](../README.md). The idea, "AI behavioural promoters by
creating the negative space for the promoter desired", is due to the keko author
(champi-ai / keko). This directory states it precisely, implements its first
instance, and sets up tests that could falsify it.*

> Status: **pipeline implemented, hypotheses untested.** The only run so far is a CPU
> smoke run on a 360M model with ~120 items ([results/smoke](results/smoke/results.md)).
> It checks that the code runs end to end. It is not evidence for or against any hypothesis below.

---

## 1. The general mechanism

Pick a desired behaviour: the **promoter**. Collect *positive samples*, meaning
model states (hidden activations) where the behaviour is not needed or already
holds. From those samples, **construct the negative space of that promoter**:

* its **boundary**: the transition from inside to fragile to outside;
* its **holes**: pockets of "outside" enclosed by "inside";
* its **outside**: states far from every positive sample.

Then attach the behaviour to those regions, so that it fires when the model's state
enters them. The attached behaviour can be an activation-steering intervention, a keko
routing decision (columns, clarification, retrieval), or a training signal.

In keko's vocabulary the resulting signals are **negative promoters**. They are the
mirror image of the **grounding promoters** of the KEKO v4.2 spec
([Executive Summary, §1](../KEKO_Uncertainty_Mechanism_Spec_v4_2.md)). Grounding
promoters are low-uncertainty sensorial evidence that *reduces* uncertainty and builds
grounded beliefs. Negative promoters are signals derived from the model's negative
space that *promote developmental behaviour*: clarification, column activation and
growth, adaptive-k retrieval, buffering for background and sleep training (§11, §13,
§18). This keeps the spec's core thesis: *uncertainty is a developmental signal, not
merely an error signal*.

Uncertainty and abstention is the **first instance**, not the whole idea. Other
promoters the same machinery should support (not implemented here):

| promoter | positive samples | negative space means | behaviour when entered |
|---|---|---|---|
| **knowledge** *(this PR)* | questions the model reliably answers correctly | boundary of competence, unknown pockets, the unknown | abstain / clarify / retrieve / grow columns |
| clarification (N6, planned) | prompts with one clear interpretation | where interpretations diverge | ask a clarifying question |
| grounding / tool use | claims supported by internal knowledge | unsupported claims | call a tool, request sensor grounding |
| caution / deferral | reversible, low-stakes actions | consequential or irreversible actions | defer to a human, add a confirmation step |
| trait / tone | on-persona responses | the drift region | steer back |

## 2. Why "negative space": the topological intuition (motivation, not a theorem)

Models are trained on the *positive space*: what they know. Nobody draws the
*negative space*, the shape of what they don't know. Topology suggests that once
the ambient space is fixed, a subset determines a lot about its complement:

* The **boundary** of a set A is ∂A = cl(A) \ int(A). It is where "known" turns into "not known".
* **Alexander duality**: for a compact, locally contractible K ⊂ Sⁿ, the reduced
  homology of the complement is determined by the cohomology of K:
  H̃_q(Sⁿ \ K) ≅ H̃^{n−q−1}(K). In particular, holes and voids in K correspond to
  components and cycles of the complement (Hatcher 2002, §3.3).

**What carries over.** Once the ambient space and its topology are fixed, the
complement and its boundary structure *follow from the subset*. So a sample of
the positive space constrains the negative space and does not have to be explored
separately. Most of the complement is uninformative (it is "everything else"). The
parts that should matter for behaviour are:

* the **boundary**, which is where answers become fragile;
* the **holes**, which are unknown pockets *surrounded by* known territory. In a hole
  the model is in familiar surroundings and so can be **confident but wrong**. This
  is the distinctive, testable claim of this track (N4/N5).

**What does not carry over.**

1. The activation space is not Sⁿ, and the "known region" is not a known compact
   set. Alexander duality does not apply literally.
2. The ambient "knowledge space" is unknown. We see activations of a finite, biased
   sample of prompts.
3. We only sample A (the positive set), with noise, and estimate its shape from a
   finite point cloud. Persistent homology of a point cloud is a *statistical
   estimate* whose features can be sampling artefacts. That is why every topological
   claim here is tested against null models (§5.4).

So "negative space" in this work means **the boundary plus the holes (plus the
outside), estimated empirically** from positive samples, and nothing more.

## 3. Hypotheses

N1–N4 are about the representation; N5–N6 are behavioural. Primary metrics,
success/failure criteria and the confirmatory/exploratory split are **pre-registered in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)** and are not to be changed after analysis.

* **N1 (decodability).** Known vs not-known status is linearly (probe-) decodable
  from hidden states, beyond what surface features (length, domain, template) give,
  with positive selectivity against a control task.
* **N2 (distance).** Distance to the known region in activation space (kNN,
  Mahalanobis with shrinkage) predicts errors, and improves over logit entropy and
  max-probability.
* **N3 (noise traces the boundary).** Under progressive noise, stability-vs-noise
  curves differ systematically between known and not-known items, and the noise level
  at which the answer flips estimates distance to the boundary. The difference must
  not be explained by prompt or answer length.
* **N4 (holes; the most exploratory).** Persistent homology of known-region
  activations reveals H1 features that survive null models and resampling, and items
  falling inside them are errors despite high model confidence.
* **N5 (behavioural routing).** Routing by negative-space region (outside, boundary,
  hole, interior) decides when to abstain, clarify or retrieve, and grow or replay
  better than keko's scalar confidence window (0.3 < p < 0.7; H7). "Better" means
  more errors intercepted at a matched intervention rate, and, for holes, larger error
  reduction after targeted replay than after random replay.
* **N6 (generalisation; planned, not run).** The same pipeline, given positives for a
  second behaviour (clarification-seeking on ambiguous prompts), yields a
  promoter-specific negative space that outperforms a single steering direction at a
  matched intervention rate.

**Relation to keko's H1/H3/H7** ([v4.2 §20](../KEKO_Uncertainty_Mechanism_Spec_v4_2.md)).
H1 (uncertainty drives column activation), H3 (adaptive-k retrieval) and H7 (layered
thresholds) all act on a *scalar* uncertainty score U. N5 replaces the scalar trigger
with a *region* trigger and keeps the same downstream behaviours. It also respects
the cortex distinction and the v4.2 stream invariants (§10): negative promoters are
computed only for inferential/epistemic streams (`LANGUAGE_INFERENCE`,
`PERCEPTUAL_INFERENCE`), never for raw sensorial observations or telemetry. The code
enforces this with `StreamRoutingError`.

## 4. Region → behaviour mapping (the knowledge promoter)

| region | operational rule (`promoters.assign_region`, precedence top-down) | keko behaviour promoted |
|---|---|---|
| **hole** | inside a significant H1 hole of the known cloud **and** confident (first-token p ≥ 0.7) | targeted column activation/growth, and enqueue to the uncertainty buffer (UB) for background and sleep training, with replay weight scaled by the hole's persistence |
| **outside** | kNN distance to known > 95th percentile of held-out known items | abstain / say "I don't know" / request grounding or tools |
| **boundary** | probe P(not known) ≥ 0.5, **or** predicted flip level ≤ 25th percentile of known items | clarification path (`ClarificationEngine` meta-query) + adaptive-k retrieval (k ∈ [1, 5], v4.2 §11) |
| **interior** | otherwise | none; the frozen core answers |

Events are `NegativePromoterEvent`s. `to_ub_row()` emits exactly the columns of the
v4.2 §15 `uncertainty_buffer` table (stream type, uncertainty score, type, severity
and region in `metadata_json`), with the §10 UB write guard applied.
`to_clarification_record()` builds keko's `uncertain.ClarificationRecord`. The
scalar baseline is `uncertain.py`'s window (`uncertainty_lower = 0.3`,
`uncertainty_upper = 0.7`).

## 5. Method

### 5.1 Operational definitions

For each question the model answers greedily and is sampled k times (T = 0.7).
Answers are matched to gold aliases after normalisation (lowercase, accents,
punctuation, articles, small number words to digits). Numbers are compared
numerically, and arithmetic uses the number after "=". Chemical symbols use a
case-sensitive token match.

* **known**: greedy answer correct **and** all k samples correct.
* **fragile**: greedy answer correct **and** ≥ 1 sample incorrect.
* **unknown**: greedy answer incorrect (consistently or not). Fictional-entity and
  unanswerable questions are unknown by construction.
* **error** (detection target): answerable → greedy answer not correct (abstention
  counts as not correct). Fictional/unanswerable → the model answered instead of
  abstaining.
* **Aleatoric vs epistemic.** Ambiguous questions ("What is the capital of Georgia?")
  are tagged `ambiguous` and **excluded** from the knowledge analysis: their difficulty
  is ambiguity, not missing knowledge. They are the natural positives and negatives for
  the clarification promoter (N6).

**Label noise.** String matching mislabels some answers. Examples: correct answers
phrased unexpectedly, aliases missing, operands echoed in arithmetic, an abstention
phrased in an unrecognised way. Each run writes `manual_check_sample.csv` (a sample
balanced between auto-correct and auto-incorrect answers) for manual checking, and the
agreement rate is reported with the results.

### 5.2 Representations

Prompt representations are taken before any answer token is produced: the last
prompt token (`last`) and the masked mean over prompt tokens (`mean`), at several
layers. Every analysis is reported **layer-wise**, and the primary layer is fixed in
the config beforehand.

### 5.3 Progressive noise (the boundary probe)

Keko's progressive noise injection is used as an empirical probe of the boundary.
There are two sites: the **input embeddings** and a **mid-depth decoder layer
output**. At relative scale σ each token's activation receives σ · rms(h) · ε, with
ε ~ N(0, I). The perturbation is therefore normalised to the layer's own activation
norm, and applied per token at every decoding step. For each σ in a fixed schedule we
record answer stability (does the answer key change?) and first-token entropy. The
**flip level** is the smallest σ at which the answer changes; items that never flip
are right-censored. The flip level is regressed from clean activations, so at
inference time the boundary distance is *predicted* without running noise.

### 5.4 Geometry and topology

* **kNN distance** to the known set and **Mahalanobis distance** with Ledoit–Wolf
  shrinkage after PCA (n_known ≪ d), fitted on **positive samples only**. An optional
  one-class SVM gives a support estimate.
* **Persistent homology** (Vietoris–Rips, H0/H1, `ripser`) of the PCA-reduced known
  cloud. A feature counts as a **hole** only if its persistence exceeds the 95th
  percentile of the maximum H1 persistence in **null clouds**: Gaussian clouds with the
  same mean and covariance, and per-coordinate shuffled clouds, both with the same n
  and dimension. We also report bootstrap reproducibility and sensitivity to the PCA
  dimension. Points are assigned to a hole heuristically: they must lie within 0.6 ×
  the radius of the hole's representative cycle around its centroid, and be farther
  from every known point than the hole's birth scale.

**Invariance.** The neuron order and the residual-stream basis of a transformer are
largely arbitrary. Hidden units can be permuted without changing the function
(e.g. Ainsworth et al. 2023), and features are stored in superposition across
non-privileged directions (Elhage et al. 2022). The quantities used here are chosen
to depend on the *shape* of the point cloud rather than on individual coordinates:

* Pairwise Euclidean distances (kNN) and persistent homology are invariant to
  permutations and orthogonal changes of basis.
* Mahalanobis distance is affine-invariant (exactly so at full rank, approximately so
  after PCA truncation and shrinkage).
* The per-coordinate standardisation applied before them keeps permutation invariance
  but not full rotation invariance. This is a deliberate trade-off for scale
  robustness. A single global scale factor would restore rotation invariance; that
  variant is listed as an ablation in the plan.

Comparing negative spaces **across models or checkpoints** needs an explicit alignment
step (e.g. Procrustes/CCA on shared prompts). Without it, coordinates are not
comparable, although distances and persistence diagrams within one model are.

### 5.5 The negative model

`NegativeSpace` (one per promoter) combines the following into one **negative score**,
P(error):

* probe P(not known);
* kNN and Mahalanobis distance;
* predicted flip level;
* hole membership.

The combiner is a logistic regression trained on **cross-fitted** component scores,
so it never sees a score from a component trained on the same item. The model
emits a region per item, and from the region a negative-promoter event. It can be
saved and loaded (`NegativeSpace.save/load`).

### 5.6 Evaluation and controls

* **Splits.** Grouped K-fold with group = entity, so the same country, element or
  event never spans train and test. Splits are repeated over seeds. There is also a
  **leave-one-domain-out** evaluation. Near-duplicate questions are removed before
  splitting (token Jaccard ≥ 0.9).
* **Matched controls.** Fictional-entity questions use *the same templates* as real
  ones ("What is the capital of Veltoria?" vs "... of Portugal?"), so surface form does
  not give them away. Keko's own `datasets/keko_honest_v1` prompts are included as
  additional unanswerable controls. They are known to be *easy* (different surface
  form) and are tagged separately.
* **Surface baselines and selectivity.** A classifier on length, domain and template
  only, plus a Hewitt & Liang (2019)-style control task (a random label per template)
  to compute probe selectivity.
* **Logit and sampling baselines.**
  * max-softmax probability (first token, mean);
  * mean token entropy;
  * keko's U score (v4.2 §8);
  * sampling inconsistency;
  * a semantic-entropy proxy: entropy over clusters of normalised answers. This is a
    string-level proxy, not the NLI-based semantic entropy of Farquhar et al.
* **Statistics.**
  * AUROC and AUPRC with 95% percentile-bootstrap CIs;
  * paired bootstrap ΔAUROC against each baseline;
  * ECE for calibration;
  * Mann–Whitney tests and Spearman correlations, including flip level vs answer and
    prompt length (the length confound).

## 6. Running it

```bash
uv sync --extra topology                 # ripser is optional; everything else degrades gracefully
uv run pytest negative_space/tests -q    # ~5 s, tiny random-weight model, no downloads

# CPU smoke run (SmolLM-360M-Instruct, 120 items, k=3, 4 noise levels × 2 sites): ~19 min measured on a laptop CPU
uv run python -m negative_space.run --config negative_space/configs/smoke_cpu.yaml

# Full bundled benchmark on CPU (~310 items, k=5, 2 noise draws)
uv run python -m negative_space.run --config negative_space/configs/smollm_cpu.yaml

# GPU (bf16, device_map=auto); see EXPERIMENT_PLAN.md for budgets
uv run python -m negative_space.run --config negative_space/configs/qwen2.5-7b-instruct.yaml
uv run python -m negative_space.run --config negative_space/configs/llama3.1-8b-instruct.yaml
uv run python -m negative_space.run --config negative_space/configs/mistral-7b-instruct.yaml

# re-run analysis only, reusing cached model outputs
uv run python -m negative_space.run --config ... --stages evaluate,report
```

The stages are `extract → noise → evaluate → report`. Model outputs are cached under
`negative_space/.cache/<name>/`, which is git-ignored. `out_dir` receives
`results.md`, `results.json`, plots, `items.jsonl` (per-item answers, labels,
regions) and `manual_check_sample.csv`. The fitted `NegativeSpace` is saved to the
cache as `negative_space_knowledge.pkl`.

Layout:

| file | role |
|---|---|
| `config.py` | dataclass config + YAML loader (`configs/*.yaml`) |
| `qa_data/` | bundled self-authored QA benchmark (`build_qa.py` → `qa_bundled.jsonl`) |
| `data.py` | loaders (bundled, keko_honest_v1, optional TriviaQA/NQ-open), normalisation, matching, dedup |
| `modeling.py` | model loading (keko conventions), prompts, generation with v4.2 §8 logit stats, hidden states |
| `extract.py` | stage 1: answers, k samples, hidden states, known/fragile/unknown labels |
| `noise.py` | stage 2: progressive noise at two sites, stability curves, flip levels |
| `geometry.py` | kNN, Mahalanobis-LW, one-class SVM |
| `topology.py` | persistent homology, null models, bootstrap, hole localisation (optional) |
| `probe.py` | probes, grouped CV, control tasks, surface baselines, flip regression |
| `promoters.py` | `Promoter`, `Region`, `Behaviour`, `NegativePromoterEvent`, keko stream invariants |
| `negative_model.py` | `NegativeSpace`: per-promoter combination, regions, save/load |
| `evaluate.py` | N1–N5 analyses, metrics, bootstrap CIs |
| `report.py` | plots and `results.md` |
| `run.py` | staged CLI |

## 7. Relation to prior work, and what is new

**Knowing what you know.**
* Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*: models can
  be trained to predict whether they know an answer (P(IK)).
* Semantic entropy (Kuhn, Gal & Farquhar 2023; Farquhar et al. 2024, *Nature*)
  detects confabulations by clustering sampled answers by meaning. Semantic entropy
  probes (Kossen et al. 2024) predict it from hidden states.
* Burns et al. (2022, CCS) find truth-like directions without labels. Azaria &
  Mitchell (2023), *The Internal State of an LLM Knows When It's Lying*, probe
  truthfulness from hidden states.
* Orgad et al. (2024), *LLMs Know More Than They Show*, study error information
  encoded in hidden states.

N1 and N2 are close to this body of work and are not claimed as new. They are the
baselines the new parts must beat.

**Density, distance and support.**
* Mahalanobis-distance OOD detection (Lee et al. 2018).
* One-class SVM support estimation (Schölkopf et al. 2001).
* Ledoit–Wolf shrinkage (Ledoit & Wolf 2004).
* Selective prediction / classification with a reject option (Geifman & El-Yaniv
  2017) is the decision-theoretic frame for "abstain when outside".

**Topology.**
* Persistent homology and TDA (Edelsbrunner & Harer 2010; Carlsson 2009).
* Topology of representations in deep networks (Naitzat, Zhitnikov & Lim 2020).
* Implementation: `ripser` (Tralie, Saul & Bar-On 2018).

**Behaviour from representations.**
* Activation addition / activation steering (Turner et al. 2023) and representation
  engineering (Zou et al. 2023) add or read *directions*.
* DPO (Rafailov et al. 2023) and unlikelihood training (Welleck et al. 2019) shape
  behaviour from *positive/negative example pairs or sets*.
* The two nearest neighbours of the promoter idea:
  * **Conditional Activation Steering** (Lee et al. 2024, *Programming Refusal with
    Conditional Activation Steering*) applies a steering behaviour only when
    activations match a *condition* vector above a threshold. This is a
    region-triggered intervention where the region is a half-space or cone around a
    direction.
  * **Circuit Breakers** (Zou et al. 2024, *Improving Alignment and Robustness with
    Circuit Breakers*) train the model to reroute representations that enter a harmful
    region, learned from examples.
* Latent adversarial training (e.g. Casper et al. 2024) perturbs latents during
  training for robustness. Our noise is used for *measurement*, not training.

**Positioning (to our knowledge).** This track proposes a *topological,
per-behaviour generalisation of region-triggered interventions*:

1. The trigger region is **constructed from positive samples** as a *shape*, with a
   boundary, holes and an outside. It is not a direction, a single threshold, or a
   learned classifier boundary.
2. The boundary is **traced empirically by progressive noise**. The flip level is a
   measured distance to the boundary, then predicted from activations.
3. The construction uses **basis- and permutation-invariant** quantities (distances,
   shrinkage Mahalanobis, persistent homology).
4. The regions drive **developmental behaviour** (keko columns, clarification,
   adaptive retrieval, sleep replay), not only blocking or steering.

The distinctive, testable claim is **holes inside known territory**: states where the
model is confident *because* it is surrounded by known territory, yet wrong (N4, N5).
We are not aware of prior work that treats a model's negative space as a
topological object with a boundary and holes derived from its positive subset and
traced by progressive noise, but we make no stronger priority claim than that.

## 8. Threats to validity

* **The probe may detect question type, domain, length or difficulty rather than
  knowledge.**
  * Mitigations: grouped and held-out-domain splits, template-matched fictional
    controls, surface-only baselines, and control-task selectivity.
  * Residual risk: fictional names may still be distributionally unusual tokens.
    Detecting "unusual entity" is not the same as detecting "not known".
* **Leakage.** Grouping by entity prevents the same entity spanning folds, but
  templates are shared across folds by design. A template-held-out analysis is part
  of the GPU plan.
* **Label noise from string matching.** It biases every AUROC towards 0.5 and can
  create spurious "fragile" labels. We report manual-check agreement.
* **Flip level confounds.** The noise flip level may track answer length, token
  count or the choice of site and scale rather than knowledge.
  * Noise is normalised per token to the activation RMS, and we report Spearman
    correlations with prompt and answer length and results per site.
  * The answer-key comparison treats correct rephrasings as stable, but
    wrong-answer flips are compared as strings. That makes long wrong answers look
    less stable.
* **Mahalanobis in high dimension.** With few known items relative to d, the
  estimate depends on PCA dimension and shrinkage. The layer and dimension are
  reported, and layer-wise results are shown.
* **Topological artefacts.** Small, noisy samples produce spurious H1 features. Hole
  claims require exceeding both null models and reproducing under bootstrap; the
  in-hole assignment is heuristic.
* **Aleatoric leakage.** Imperfect ambiguity tagging leaves some ambiguous items in
  the knowledge analysis.
* **Controls too easy.** keko_honest prompts differ in surface form. Results should be
  read with and without them (`kind`/`template` are in `items.jsonl`).
* **Multiple comparisons.** Many scores × layers × sites are reported. Only the
  pre-registered primary comparisons in EXPERIMENT_PLAN.md are confirmatory.

## 9. Limitations

* Short-form factual QA only, in English, with small self-authored benchmarks. This
  is not long-form generation or reasoning.
* The "known region" is defined relative to this benchmark and prompt format. A
  different prompt distribution gives a different positive cloud.
* Point-cloud topology with tens to hundreds of known points is statistically weak.
  N4 needs the GPU-scale run to say anything.
* N5 as run here is only a *matched-rate error-interception* comparison. The
  developmental part (targeted replay into keko columns) is planned, not run.
* Only one promoter (knowledge) is implemented. N6 is a design, not a result.
* Comparisons across models or checkpoints need an alignment step (§5.4).

## References

* Ainsworth, Hayase & Srinivasa (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR*.
* Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP*.
* Burns, Ye, Klein & Steinhardt (2022). Discovering Latent Knowledge in Language Models Without Supervision. arXiv:2212.03827 (ICLR 2023).
* Carlsson (2009). Topology and Data. *Bulletin of the AMS* 46(2).
* Casper et al. (2024). Defending Against Unforeseen Failure Modes with Latent Adversarial Training. arXiv.
* Edelsbrunner & Harer (2010). *Computational Topology: An Introduction*. AMS.
* Elhage et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*.
* Farquhar, Kossen, Kuhn & Gal (2024). Detecting hallucinations in large language models using semantic entropy. *Nature* 630.
* Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks. *NeurIPS*.
* Hatcher (2002). *Algebraic Topology*. Cambridge University Press.
* Hewitt & Liang (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP*.
* Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
* Kossen et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927.
* Kuhn, Gal & Farquhar (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR*.
* Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Analysis* 88(2).
* Lee, Lee, Lee & Shin (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS*.
* Lee et al. (2024). Programming Refusal with Conditional Activation Steering. arXiv:2409.05907.
* Naitzat, Zhitnikov & Lim (2020). Topology of Deep Neural Networks. *JMLR* 21.
* Orgad et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707.
* Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.
* Schölkopf, Platt, Shawe-Taylor, Smola & Williamson (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation* 13(7).
* Tralie, Saul & Bar-On (2018). Ripser.py: A Lean Persistent Homology Library for Python. *JOSS* 3(29).
* Turner et al. (2023). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248.
* Welleck et al. (2019). Neural Text Generation with Unlikelihood Training. arXiv:1908.04319 (ICLR 2020).
* Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
* Zou et al. (2024). Improving Alignment and Robustness with Circuit Breakers. arXiv:2406.04313 (NeurIPS 2024).
