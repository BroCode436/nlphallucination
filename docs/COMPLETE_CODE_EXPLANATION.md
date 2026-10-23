# Complete code explanation

Read the data pipeline first, then generation and evidence evaluation, then supervision and selection. Features describe the state before intervention. Labels describe outcomes after intervention. Keeping those timelines separate is the main leakage safeguard.

## V2 extension: file and function contracts

### chc\v2\__init__.py

CHC-Propagate: observable traces and measured intervention effects.

This extension reuses CHC data, graph, correction and policy components. All
automatic factual judgments carry their origin; no NLI label is human gold.

### chc\v2\alignment.py

Explicit citation alignment for already generated observable factual claims.

This is a separately audited processing stage, not silent JSON repair. Invalid
JSON and bad dependencies are rejected. Missing source references are supplied
by documented retrieval, which does not declare a statement true. Clean-prefix
factual acceptance is a separate gate. Claim text and
dependency declarations remain byte-for-byte the model's actual content.

#### align_uncited_artifact

Build a new validated citation-aligned artifact or preserve the rejection.

Args:
    generation: Real raw model output with its original validation outcome.
    question/evidence: Serving-visible data only; no answer/reference labels.
    judge: Automatic source alignment model; thresholds fixed before testing.
    prefix_ids: Prior claims allowed in dependencies.
Returns:
    Original generation with explicit raw-validation and alignment provenance.
    A processed artifact gets a fresh trace with explicit retrieved-source
    provenance; contradiction and insufficient evidence remain possible.
    The raw output is
    retained unchanged. This artifact is not a raw schema-compliant response.

#### AlignedTraceBackend

Compose real raw generation with explicit, reference-free source alignment.

Args:
    backend: Existing TraceBackend instance, configured without grammar.
    judge: Shared automatic NLI model.
Attributes delegate execution counters/model identity to the actual backend;
alignment changes metadata only, never model text or measured answer strings.

**__init__**

Retain the real inference objects; no second language model is loaded.

**__getattr__**

Delegate resource metadata and counters to the underlying real model.

**generate**

Generate raw content then validate separately aligned citations.

The output retains raw_validation_error and citation_alignment, making
strict raw-JSON reliability distinguishable from processed-artifact yield.

### chc\v2\analysis.py

Derive propagation diagnostics, review sheets and figures from actual episodes.

No favorable values are hardcoded. Missing/unknown labels remain visible and
figures distinguish automatic evidence judgments from human-reviewed truth.

#### summarize_collection

Summarize all attempts, benefit signs, labels, effects and review status.

Downstream contradiction rates exclude unsupported/unreviewed claims. Error
amplification is a descriptive count by model-declared graph depth, not a
causal edge effect. A changed final answer is not necessarily a worse answer.
Human review files leave decisions blank and keep a separate machine key.

#### save_figure

Save a measured figure in PNG/PDF with a separate interpretation caption.

#### plot_collection

Plot benefits, labeled error rates and declared-depth diagnostics.

Missing plots are skipped when no measured rows exist. No empty graph is
filled with invented data. The before/after graph is a labeled real example,
with status colors from the automatic source judgment only.

#### plot_detection

Plot every detector variant's actual PR and calibration curves.

The selected confusion matrix is chosen by validation F1, not test scores.
Test curves are descriptive and retain the natural sampled positive rate.

#### plot_policies

Visualize completed policy tables, ablations, budgets, harm and prediction error.

### chc\v2\atomic.py

Conservative factual-unit extraction and recorded-output sensitivity scoring.

Only explicit sentence boundaries and shared-subject verb conjunctions are split.
Uncertain relative clauses/lists are flagged rather than inventing paraphrases.
These units are correlated pieces of an observable model artifact, not independent
examples or access to private reasoning. Human atomicity review remains necessary.

#### factual_units

Extract conservative units while preserving original text and edit provenance.

Args:
    text: One actual generated factual statement, possibly compound.
Returns:
    Units with text, original text, extraction rule and a residual-compound
    flag. Shared subjects are copied only when an explicit verb follows
    “and”; noun lists, relative clauses and ambiguous conjunctions remain
    unsplit. The parser does not claim complete semantic atomicity.

#### score_atomic_outcome

Judge units from saved real output without generating or inventing outcomes.

Invalid structured outputs remain invalid. Existing whole-claim judgments
are preserved, and unit judgments are stored separately. The factual summary
is recomputed over evaluable units; insufficient units remain unknown.

#### atomic_records

Compute the predeclared factual-unit objective from actual recorded outputs.

Raw sentence-level results remain intact in their original JSONL. Each new
record preserves its sentence benefits and adds factual-unit benefits. Clean
prefix eligibility is audited separately; this operation never discards
neutral/negative interventions based on their effects.

### chc\v2\backend.py

Memory-bounded, cached real generation for structured observable claims.

Extends the frozen-model approach used by CHC generation.py. A configurable
instruction model uses its official chat template; historical FLAN uses a plain
prompt. Invalid JSON is never repaired with hand-written benchmark content.

#### TraceBackend

Frozen local model with exact prompt/config caching and explicit OOM logs.

Args:
    cache: Shared weight and response cache root.
    model_id: Model to assess; no automatic download of a larger replacement.
    max_new_tokens: Output cap for compact structured artifacts.
    device: CUDA if available, otherwise CPU. CUDA uses FP16 to bound memory.
Attributes:
    fallbacks: Actual OOM recovery events, not assumed hardware failures.
    actual_generations/cache_hits: Process-local execution counters.

**__init__**

Load a cached snapshot, with a single explicit CPU fallback on OOM.

A 1.5B FP16 model needs roughly 3.1 GB for weights, plus activations/KV
cache. Actual CUDA availability is checked, and the run records allocator
peaks. Four-bit quantization is not required when FP16 fits; callers can
choose another model without changing benchmark or scoring code.

**generate**

Generate an actual structured trace or continuation without gold labels.

Args:
    question/evidence: Fixed serving inputs, identical across interventions.
    prefix: Visible prior claims; corruption/correction changes only these.
    seed: Paired sampling seed. Greedy development mode is not a replicate.
Returns:
    Raw output, validated trace or validation error, tokens, latency,
    truncation, model identity and cache key. Invalid outputs stay invalid.
Failure cases:
    Truncated inputs are rejected rather than silently losing citations.
    CUDA OOM retries once on CPU and records the change of execution path.

### chc\v2\boosting.py

Validation-tuned benefit boosting with question-balanced regression weights.

Extends the original CHC regressor rather than introducing a large architecture.
The central richer/final-only comparison uses this identical tuning procedure.
Neutral targets remain; weights balance questions, not desired benefit signs.

#### flatten_targets

Extract masked actual labels after label-free features, retaining question IDs.

Args:
    records: One predeclared train or validation partition with named targets.
    columns: Numerical feature columns for a matched retrained ablation.
Returns:
    [A,F] inputs, [A] signed utility targets, and [A] question grouping IDs.
    Missing targets are excluded by the existing graph mask, never filled.

#### fit_boosting

Fit a small predeclared grid and choose mean question-validation MSE.

Training weights give each question equal total weight regardless of missing
labels or repeated seed rows. The grid varies leaf count, minimum leaf size
and L2 regularization. No final-test features/labels enter fitting or tuning.
Best model and feature columns use the existing TabularPolicy file contract.
Failure cases include empty measurable partitions or incompatible state data.

### chc\v2\collection.py

Resumable CHC-Propagate collection using actual frozen-model continuations.

The collector retains every attempted question and candidate mutation. Selection
of error families occurs before observing continuation outcomes. Clean, corrupt,
restored and verified continuations share question, evidence and sampling seed.

#### source_evidence

Build bounded real source passages from annotated supporting sentences.

Returns passages with document titles, original sentence indices, dataset
question ID and official dataset link. This controlled evidence condition is
easier than retrieval over full distractor documents and is reported as such.
Nothing is synthesized from the reference answer.

#### annotate_state

Attach pre-action predicted risk, source relevance and tokenizer costs.

Only visible claims and their evidence judgments are consumed. Reference
answers, injected locations and observed action benefits never enter features.
Uses the same source-aligned judge as the other V2 verification policies.

#### collect

Collect actual paired outcomes without filtering on benefit or correctness.

Args:
    splits: Predeclared disjoint development/train/validation/test rows.
    backend/judge/repair: Shared real inference objects with persistent caches.
    directory: New experiment directory; prior V1 outputs are never overwritten.
    seeds: Paired model-generation seeds, not additional independent questions.
    conditions: Maximum eligible distinct error families per question.
Returns:
    Recorded complete episodes; attempted and rejected records are separate.
Algorithm:
    Validate a generated trace, keep its first two supported factual claims as
    a prefix, select a valid mutation without observing effects, and continue
    after each real verifier action. An exact original restoration is an
    analysis-only sensitivity diagnostic, never an available serving action.
Missing data:
    Malformed continuations remain missing and are not reinterpreted as
    correct/neutral outcomes. Complete-case policy evaluation reports coverage.

### chc\v2\corruptions.py

Auditable factual perturbation candidates with conservative validity gates.

Ten error families are implemented where their prerequisites exist. Generated
mutations are candidates, not automatically true error labels. Primary inclusion
requires a supported original and a contradicted mutation from the automatic NLI
proxy. All provenance/disambiguation/composition cases still need human review.

#### corruption_candidates

Generate eligible mutations without references or intervention outcomes.

Args:
    claims: Validated clean prefix; only explicit textual anchors are changed.
    evidence: Supplies alternative real entity names and citation IDs.
Returns:
    Candidates containing original/modified text, source claim ID, error
    family, exact transformation and its proposed justification. Unsupported
    assertions are not assumed false. Missing prerequisites yield no example
    for that family instead of inventing a label or an entity relationship.

#### validate_candidates

Check original/mutated statements and retain uncertainty explicitly.

Every candidate survives in the audit. Only a high-confidence automatically
supported original and contradicted mutation can enter the primary automatic
challenge set. This does not imply human validation or natural prevalence.

### chc\v2\detection.py

Matched natural-distribution detector evaluation with validation-only tuning.

Separates evidence alignment, token length, and model changes. Binary RAGTruth
labels include unsupported hallucinations, so NLI contradiction alone may not
cover them; both contradiction and non-entailment scores are reported honestly.

#### reliability

Return ten-bin expected calibration error and empirical reliability points.

Every score enters exactly one half-open bin, including p=1 in the last bin.
Empty bins are omitted. ECE depends on binning and is not sufficient alone
to establish useful or trustworthy probability calibration.

#### binary_metrics

Calculate detection metrics on the original sampled class distribution.

PR-AUC is average precision (step-weighted PR area), not trapezoidal ROC AUC.
The threshold must be supplied from a validation decision, never test labels.

#### evaluate_detection

Evaluate fixed variants and fit Platt calibration on validation rows only.

Samples are uniform with seed 42; positives are not enriched. Test IDs/source
IDs are audited against validation. Original 256 sample is reproduced as a
separately marked comparison, not conflated with the larger sample.
No detector is fine-tuned without a separate supervised experiment.

#### CalibratedDetector

Validation-selected existing detector with explicit domain-limited calibration.

Args:
    project: Root holding the unchanged DistilBERT checkpoint.
    report: Completed detector report with validation scores and Platt weights.
Attributes:
    selected: Fixed choice made by validation F1, never by test performance.
    detector: The retained local detector, with the selected input condition.

**__init__**

Select an eligible DistilBERT variant from validation measurements only.

The model is not fine-tuned here. Missing calibration metadata fails
explicitly. RAGTruth calibration is reported as out-of-domain on HotpotQA.

**predict**

Return stable binary suspicion scores and actual evidence references.

Classification means suspected hallucination, not certified contradiction.
Empty evidence abstains. A calibrated number is only exposed as calibrated
for RAGTruth; transferred calibrated scores remain marked domain-shifted.

### chc\v2\evidence.py

Aligned NLI evidence checks with explicit abstention and judgment provenance.

Sentence selection reduces the original detector's truncation problem. Three-way
NLI is a fallible automatic measurement, not an oracle. Neutral evidence remains
insufficient rather than being converted into a contradiction label.

#### align_evidence

Select relevant whole sentences using label-free lexical overlap.

Args:
    claim: Visible statement; reference answers and labels are not used.
    evidence: Full supplied passage, never evidence retrieved from the future.
    sentences: Maximum number of top ranked sentences, restored to source order.
Returns:
    A compact passage. Empty evidence stays empty, causing later abstention.
Assumptions:
    English punctuation is a heuristic sentence boundary. This selector can
    miss paraphrases; aligned length and truncation are recorded separately.

#### EvidenceJudge

Batched three-way NLI with cache and a stable four-state claim interface.

Args:
    cache: Shared model cache and independent evidence-judgment JSON cache.
    threshold: Conservative automatic status threshold, fixed before test.
    device: Requested single execution device; CPU is supported.
Attributes:
    labels: Model-card order is checked against the downloaded configuration.
    calibration: Optional validation-fitted binary contradiction calibration;
        it does not calibrate three-way truth or turn neutral into false.

**__init__**

Load the small NLI checkpoint and verify label semantics.

Missing weights trigger ordinary Hugging Face loading, never fake scores.
A wrong label mapping raises an error rather than reversing factual labels.

**probabilities**

Infer [contradiction, entailment, neutral] for aligned evidence/claim pairs.

Inputs are (claim, full evidence) tuples. Outputs retain selected evidence,
actual token lengths and cache identity. Token budget is 384 with
truncation on the evidence side only so the claim is preserved. On CUDA
OOM the complete small batch is retried on CPU and the event is recorded.

**assess**

Attach fallible supported/contradicted/insufficient evidence judgments.

Missing citations return unreviewed. High-confidence entailment or
contradiction returns the corresponding automatic status; all other
cases abstain. Every result contains exact cited source material and
human_review='pending'. No benchmark reference answer is consulted.

**embeddings**

Return frozen mean-pooled contextual embeddings for semantic regression.

Input strings combine claim/question/evidence selected before actions.
Padding is masked; [B,T,H] last hidden states become [B,H] vectors. These
are generic NLI-backbone representations, not tuned sentence embeddings.
No outcomes or reference answers are embedded. CPU numpy output supports
small tabular regressors without keeping an autograd graph in GPU memory.

### chc\v2\grammar.py

Small Transformers-5 adapter to LM Format Enforcer's public token API.

LMFE 0.11.3's optional Transformers adapter imports a tokenizer class from an old
module path. The core token/JSON parser works independently. This adapter uses
that core, avoiding global monkey patches or modifying installed packages.

#### tokenizer_data

Build a reusable decoded-token trie, preserving leading-space behavior.

Each ordinary vocabulary token is decoded after a sentinel, matching LMFE's
official integration. EOS is handled by the grammar. Unicode replacement
fragments are removed only from incremental decoding, not generated facts.
Cost is paid once per backend, not once per continuation.

#### prefix_filter

Return Transformers' public prefix_allowed_tokens_fn callable.

A fresh parser is used for every generation so paired runs do not share
mutable parser state. The resulting token filter constrains format only.

### chc\v2\graph_features.py

Pre-action source-bridge hypotheses, separate from measured propagation.

Model-declared dependencies can be empty even on multi-hop questions. A source
document mentioning another cited document's entity supplies an additional
uncertain relevance feature. It does not certify a causal reasoning edge.

#### enrich_source_bridges

Add ordered cross-document entity links using serving-visible evidence only.

Args:
    state: Current claims and retrieved passages; contains no outcome labels.
Returns:
    A copied validated state. Existing declared edges are retained. An extra
    earlier-to-later edge is added only if a source passage mentions the
    other claim's document entity, with distinct source IDs. Parenthetical
    and comma suffixes in titles are removed for matching. Minimum four
    letters avoids links on short ambiguous titles. Confidence .35 is a
    heuristic indicator, not a calibrated causal probability.
Limitations:
    Document mentions do not prove claim dependence. Direction follows
    observable order. This fixed recipe must be ablated and human-reviewed.

#### prepare_policy_records

Derive fixed source-bridge states without changing any collected outcome.

All action subsets were already generated, so label-free graph enrichment
can be applied before model fitting without regenerating language-model
outputs. The graph recipe is identical across train, validation and test.
This is feature extraction, never post-hoc editing of experimental benefits.

### chc\v2\learning.py

Matched utility learning, zero controls and paired question-level evaluation.

Reuses the original CHC tabular/GNN implementations. The central comparison is
identical architecture/features trained on final-answer-only versus downstream
factual loss targets. Semantic regression is a simple frozen-embedding control,
not an automatic justification for a hybrid network.

#### supervised

Join chosen targets without mutating raw collected episodes.

A complete pre-action baseline is required; candidate masks exclude missing
utility labels. The original generator outputs and reference scores remain
intact. The label scope is explicit rather than overloading correctness.

#### visible_features

Extract structured and frozen semantic features from visible state only.

Returns tabular [A,14], semantic [A,H], and (episode,node) indexing. Embeddings
concatenate the question, current claim and aligned source passages. They do
not contain injection identity, gold answer, corrected text, or future claims.

#### train_models

Fit tabular, directed graph and semantic models using validation only.

Retains all neutral outcomes and trains the same model classes for both
objectives. The zero predictor and training-frequency mean are saved controls.
Small pilot models are exploratory; no final test performance selects them.
Hybrid semantic-graph training is intentionally gated on simpler-model results.

#### question_interval

Bootstrap question means, preserving dependence of conditions and seeds.

The caller groups all dependent rows within each question first. Returns a
descriptive percentile interval from 2,000 resamples. One question cannot
support a useful population uncertainty estimate and returns a point interval.

#### evaluate_models

Evaluate identical complete test episodes under budgets 0/1/2.

Every policy uses the same backend, evidence, corrections and outcomes. Budget
two uses a measured joint continuation, never an additive singleton fiction.
Direct V1 transfer is labeled separately from V2 retraining. Richer/final-only
objective comparisons are predeclared, with matched question intervals.

### chc\v2\propagation.py

Transparent propagation denominators and signed downstream intervention loss.

Unsupported claims are not false. Automatic judgments are explicitly distinguished
from human labels. Metrics aggregate by trajectory/question, never by pretending
that dependent claims are independent experimental observations.

#### factual_summary

Count statuses and compute contradicted/(supported+contradicted).

Returns null rate when no downstream claim can be evaluated. Unknown and
unreviewed claims are counted separately and excluded from the denominator.
This is an observed automatic-proxy error rate, not human factual accuracy.

#### score_continuation

Score only after generation; the evaluator alone receives the reference.

Invalid generations have missing correctness and propagation scores. Error
repetition is a conservative exact normalized-text check; it can miss a
paraphrase. Reported propagation concerns newly generated claims, not the
injected prefix itself. Graph declarations are kept apart from these scores.

#### observed_loss

Calculate L=.5 downstream contradiction rate + .5 final error + .25 harm.

Weights are a predeclared equal-weight diagnostic choice, not optimized test
parameters. Final-only ablation uses 1-final EM. If the required downstream
denominator is missing, richer loss is unknown rather than assuming zero.
Harm means a repair of a previously supported prefix becomes contradicted;
ordinary final-answer deterioration is separately measured at evaluation.

#### utility

Return signed observed loss reduction; unknown labels remain missing.

Verification cost is subtracted during selection, not inside supervision, so
changing lambda does not require relabeling benchmark outcomes. Beneficial,
neutral and harmful labels follow the actual sign, without forced balancing.

### chc\v2\repair.py

Counterfactual-aware repair checks layered over the retained lexical verifier.

Source entailment alone cannot establish a valid repair: a different supported
sentence may describe the wrong entity or discard valid information. This module
adds conservative anchor/preservation diagnostics and semantic evidence checks.
It is an untrained safety extension, not a claimed newly trained CAHR model.

#### RepairVerifier

Propose existing lexical repairs and apply an additional semantic gate.

Args:
    project: Existing root containing correction_validity.joblib.
    judge: Shared evidence-NLI checker; its judgments remain automatic proxies.
Attributes:
    lexical: Preserved trained support/preservation model.

**__init__**

Reuse saved correction weights without synthetic relabeling or refitting.

**verify**

Return a sourced action, with explicit reasons for acceptance/abstention.

Question and original claim are supplied alongside retrieved evidence and
the actual proposed correction. Capitalized anchors in the original that
appear in source evidence must remain in the proposal. This conservative
heuristic may reject valid paraphrases. The lexical proposal mechanism is
shared across policies; an ablation can remove only the extra gate.

### chc\v2\review.py

Import explicitly supplied review labels without overwriting automatic results.

Blank worksheets are not completed reviews. A reviewer identifier and valid label
are required. This tool records supplied annotations; it cannot certify that a
human performed them. Review-sample metrics are not population prevalence.

#### import_review

Validate supplied annotations and save them separately with provenance.

Args:
    worksheet: Filled CSV derived from human_review_blank.csv; empty labels
        remain unreviewed and are not converted to factual judgments.
    machine_key: Matching IDs/episode/variant mapping from the same collection.
    output: New annotation JSONL; never replaces raw machine judgments.
    reviewer: Explicit identifier for annotation accountability.
Returns:
    Imported and pending counts with automatic-versus-reviewed agreement.
    Agreement describes the reviewed workload only, not benchmark accuracy.
Raises:
    ValueError: Missing reviewer, duplicate/unknown IDs, invalid labels or
        required columns. Existing conflicting annotation files are preserved.

### chc\v2\robustness.py

All-input answer scoring and denominator audits for recorded policy outcomes.

Primary propagation analyses require evaluable structured claims. This separate
analysis also scores answers from every eligible input, exposing format failures
and complete-case selection. It never marks an invalid trace valid or an unknown
claim false. No model is retrained or chosen using these held-out results.

#### recorded_answer

Extract an actual final-answer field even if a trace failed validation.

Only a nonempty string from an existing answer or parseable raw JSON object
is accepted. No regex guess, reference-based repair or invented answer is
allowed. A rejected trace remains rejected regardless of answer availability.

#### selected_outcome

Retrieve the actual measured subset outcome without adding singleton gains.

#### operational_evaluation

Score every primary test input, without post-output complete-case filtering.

Generation/claim validity and final-answer availability are separate fields.
Missing answers count as operational answer failures, not factual falsehoods.
Malformed traces with parseable answer fields can still have measured answer
EM, but have no claimed downstream factual label. All selectors see only
serving-visible states; reference strings enter scoring after selection.

#### augment_denominators

Add per-unit coverage and logical NLI token accounting to policy CSVs.

Unknown units remain separate from contradicted units. Logical NLI input
tokens reflect bounded original/proposal checks, not all upfront detector or
embedding computation and not uncached physical executions or API billing.

### chc\v2\traces.py

Validate observable reasoning artifacts before graph building or evaluation.

Only explicit short factual statements are requested. These are public model
outputs, not measurements of private model reasoning. Malformed structures raise
errors and are retained in the rejection log by the collection pipeline.

#### parse_trace

Parse strict JSON and validate ordered, source-linked factual claims.

Args:
    raw: Actual generated text; only a surrounding Markdown fence is removed.
    evidence: Available source IDs. Unknown citations invalidate the trace.
    question: Serving-visible question, attached by the caller, not a label.
    prefix_ids: Earlier visible claim IDs that a continuation may reference.
Returns:
    Validated JSON object with the question attached and an atomicity warning.
Raises:
    ValueError: Invalid JSON, extra/missing fields, empty/duplicate claims,
        forward dependencies, missing evidence, or an oversized artifact.
Algorithm:
    Topological order is enforced as claims are read. This also prevents
    cycles without guessing dependency relationships from final outcomes.

#### trace_state

Convert validated visible claims into the existing label-free CHC state.

Inputs contain text, source references and model-declared dependencies only.
The final placeholder contains the question, never a reference answer. Edges
carry explicit uncertain provenance; downstream effects are stored elsewhere.
Returns a validated DecisionState or raises ValueError for inconsistent input.

### chc\v2\validity.py

Independent pre-outcome validity veto for selected factual corruptions.

A source-specific NLI contradiction is insufficient if another available source
supports the modified statement. Entity swaps can remain true. This guard never
reads downstream outcomes and never forces beneficial/harmful label balance.

#### selected_corruption_validity

Veto mutations that duplicate known support or have cross-source support.

Args:
    injection: Previously selected candidate, before continuation generation.
    clean_prefix: Initial automatically supported claims, not future outputs.
    evidence: All available verifier passages, including alternative entities.
    judge: Frozen NLI model; a high entailment score conservatively vetoes
        an asserted contradiction. No human-gold validity is claimed.
Returns:
    Eligibility flag, exact matching prefix IDs, per-source entailment scores
    and a reason. A rejected selection is logged; no favorable alternative
    is substituted after observing an outcome.

### chc\v2\web.py

Expose recorded propagation experiments and optional real local live inference.

Recorded examples require no model loading. Live inference uses supplied or
locally retrieved evidence and never invents a reference answer for evaluation.
The old application and routes remain available.

#### PropagationService

Serve measured examples and label-free recommendations through one API.

Args:
    project: Existing project containing source, checkpoints and V2 outputs.
Attributes:
    lock: Serializes RNG/model use across live requests.
    models: Lazy inference resources; recorded requests do not allocate GPU.

**__init__**

Locate configured results and defer heavyweight imports until needed.

**examples**

List actual stored benchmark questions without exposing gold to a policy.

**analyze**

Return a recorded example or generate actual live factual continuations.

Recorded mode exposes reference scoring after returning the already
collected output. Live mode receives no reference and leaves final-answer
accuracy unknown. Submitted text and evidence have explicit size limits.

## Existing project modules retained

# File-by-file code walkthrough

Read ARCHITECTURE.md first, then TRAINING_AND_METRICS.md for the training equations and tensor shapes. The source docstrings explain parameters and failure cases. Any annotations in legacy functions preserve dynamic interfaces rather than promising an invented concrete type.

## New CHC package

### `chc/__init__.py`

CHC: controlled intervention experiments, not a certified causal graph system.

The package separates observable policy state from benchmark reference labels.
See docs/METHODOLOGY.md before interpreting an intervention as a causal effect.


### `chc/cli.py`

Reproducible staged commands.

- **main**: Parse reproducible run settings and execute the selected experiment stage.

### `chc/data.py`

Real bounded dataset loading, supporting facts, and split audits.

- **read_jsonl**: Read nonempty UTF-8 JSONL records with ordinary parse errors preserved.
- **write_jsonl**: Save reproducible Unicode JSONL without ASCII-escaping Hindi characters.
- **fetch_hotpot**: Fetch a bounded, reproducibly selected prefix of a real official split.
- **supporting_sentences**: Extract explicitly annotated supporting sentences, not invented reasoning.
- **split_questions**: Partition official train into fit/tuning and reserve official validation.
- **audit_splits**: Audit question/evidence overlap and report shared entities as limitations.
- **load_ragtruth**: Load an actual held-out detector sample without using it for CHC training.

### `chc/detector.py`

Actual saved DistilBERT and safe correction adapter.

- **ExistingDetector**: Load the user's saved cross-encoder once and batch actual inference.
  - `__init__`: Initialize local-only weights; missing artifacts raise an explicit error.
  - `predict_pairs`: Return actual hallucination scores for equally sized text lists.
  - `predict_hallucination`: Expose the common detector interface for one claim.
  - `annotate`: Attach predicted risk and lexical evidence quality to visible nodes.
  - `verify`: Retrieve supplied sources and accept only justified candidate repairs.

### `chc/experiment.py`

Paired real-model trajectories, validation, fitting, and evaluation.

- **apply_actions**: Apply only recorded accepted/unchanged replacements to a copied trace.
- **outcome**: Generate and externally score one continuation; never expose reference to prompt.
- **collect_episodes**: Record controlled paired outcomes on real questions, with crash-safe resume.
- **validate_episodes**: Check schema, graph integrity, paired settings, and train/test separation.
- **train_policies**: Train both intervention approaches and retrained feature ablations.
- **evaluate_policies**: Replay matched held-out outcomes under each policy and actual budget.
- **evaluate_detector**: Evaluate the actual local detector on a held-out natural RAGTruth sample.

### `chc/generation.py`

Frozen small-model continuations with explicit cache provenance.

- **SmallReasoner**: Frozen FLAN-T5 continuation backend with bounded memory and actual outputs.
  - `__init__`: Load a bounded real model after the caller has estimated memory.
  - `prompt`: Build a label-free prompt with notes first, then fixed source evidence.
  - `generate`: Generate or reuse one actual continuation with a fully specified cache key.

### `chc/graph.py`

Uncertain textual DAG validation and pre-action feature construction.

- **build_state**: Build a small trace graph from annotated sentences and an unfilled answer.
- **validate_graph**: Validate a DAG and all evidence references before computing features.
- **node_features**: Create graph/node tensors exclusively from serving-visible state.

### `chc/injection.py`

Eligible primary perturbations and review-only auxiliary types.

- **candidates**: Construct perturbations at explicitly identified supporting-sentence nodes.
- **select_conditions**: Choose up to three distinct primary error types without inventing eligibility.

### `chc/metrics.py`

Mathematical outcomes and question-cluster uncertainty.

- **normalize_answer**: Apply standard lowercasing, punctuation/article removal, and whitespace cleanup.
- **exact_match**: Compare a generated answer to a held-out reference after normalization.
- **prediction_metrics**: Measure benefit-regression error without interpreting correlation as causality.
- **summarize_policies**: Aggregate matched policy outcomes and question-cluster bootstrap intervals.

### `chc/models.py`

Outcome regression, directed message passing, and resumable training.

- **BenefitGNN**: Two rounds of directed mean message passing followed by scalar regression.
  - `__init__`: Create standard learnable linear layers without pretrained embeddings.
  - `forward`: Propagate directed messages and return one predicted benefit per node.
- **training_graph**: Join a visible state to training targets after features have been extracted.
- **fit_tabular**: Fit a small regressor to measured singleton intervention benefits.
- **pack_graphs**: Create a disconnected graph batch with correct node-index offsets.
- **fit_gnn**: Train graph benefit regression with validation, AMP, and resumable state.
- **GraphPolicy**: Load a trained directed graph regressor for label-free action scoring.
  - `__init__`: Restore trained weights and training-only normalization constants.
  - `scores`: Return predicted singleton benefit for each nonfinal visible claim.

### `chc/plots.py`

PNG/PDF figures from saved measured tables.

- **save_figure**: Save one actual-results figure in raster/vector formats with a caption file.
- **make_plots**: Render ten families of plots without hardcoded performance values.

### `chc/policy.py`

Matched baseline priorities and budget-feasible greedy selection.

- **select_claims**: Greedily select affordable nodes while tracking overlapping descendants.
- **baseline_scores**: Compute transparent priorities without seeing outcomes or reference labels.
- **TabularPolicy**: Inference adapter for a fitted intervention-benefit regression checkpoint.
  - `__init__`: Load the saved estimator and feature mask.
  - `scores`: Predict single-action benefit from pre-action features only.

### `chc/resources.py`

Seeds, per-device memory, and CUDA/CPU selection.

- **seed_everything**: Seed Python, NumPy, and PyTorch for matched controlled runs.
- **hardware_report**: Measure usable NVIDIA devices and save a machine-readable report.
- **choose_device**: Choose one CUDA device or CPU without assuming pooled GPU memory.

### `chc/schema.py`

Typed boundary that keeps observed outcomes out of serving-time inputs.

- **Claim**: One visible reasoning node; its risk is a prediction, not a gold label.
- **DecisionState**: Serving-visible inputs, deliberately excluding reference outcomes.
  - `to_dict`: Return a JSON-serializable state without attaching evaluation labels.
  - `from_dict`: Parse a saved pre-action state, rejecting unexpected top-level keys.
- **Budget**: Hard selection limits, independent of observed future outcomes.
  - `__post_init__`: Reject negative limits immediately instead of violating budgets later.

### `chc/web.py`

Original-framework CHC frontend/service with no hidden reference features.

- **CHCService**: Lazy real-model service sharing the original project's saved components.
  - `__init__`: Load actual inference models and locate the strongest available run.
  - `retrieve`: Rank actual cached corpus documents using transparent lexical matching.
  - `analyze`: Execute a live intervention without leaking optional evaluation labels.
- **serve**: Run an optional standalone CHC HTTP view using the existing server framework.

## Existing project modules

The original detectors were not retrained for CHC. Its correction checkpoint is reused. Documentation-only changes were compared against the executable AST of the backup; app.py additionally received intentional CHC route changes.

### `analyze_claim.py`

Inspect a claim locally and emit an auditable JSON explanation.

- `inspect_claim`: Inspect claim.
- `main`: Main.

### `app.py`

Run: python app.py --port 8765. Local UI, no API key or web calls.

- `verify_payload`: Verify payload.
- `main`: Main.

### `audit_dataset.py`

Audit saved claim splits, including repeated evidence under different IDs.

- `audit_splits`: Audit splits.
- `main`: Main.

### `build_updated_mse2_notebook.py`

build_updated_mse2_notebook.py

- `img_to_base64`: Img to base64.
- `make_md_cell`: Make md cell.
- `make_code_cell`: Make code cell.
- `build_notebook`: Build notebook.

### `evaluate_challenges.py`

Small author-written challenge suite. Not a human-reviewed benchmark.

- `main`: Main.

### `ragtruth_mse1.py`

ragtruth_mse1.py

- `load_and_inspect_ragtruth`: Load and inspect ragtruth.
- `preprocess_data`: Preprocess data.
- `run_eda`: Run eda.
- `explain_models`: Explain models.

### `review_corrections.py`

Export a human-review worksheet or score a completed worksheet.

- `score_reviews`: Score reviews.
- `main`: Main.

### `run_pipeline.py`

run_pipeline.py

- `run_unit_tests`: Run unit tests.
- `run_preprocessing`: Run preprocessing.
- `run_training`: Run training.
- `run_evaluation`: Run evaluation.
- `check_prerequisites_exist`: Checks if processed data and checkpoints (including actual model weights) are available.
- `main`: Main.

### `src/correction.py`

Evidence-guided correction with proposal rollback and post-verification.

- `EvidenceGuidedCorrector`: Evidence-Guided Targeted Claim Corrector with Post-Correction Verification.
- `evaluate_correction_pipeline`: Evaluates correction and post-verification under exact mathematical definitions:

### `src/correction_validity.py`

Local, trained correction verifier. Lexical features are not semantic proof.

- `tokens`: Tokens.
- `words`: Words.
- `numbers`: Numbers.
- `clauses`: Clauses.
- `recall`: Recall.
- `same_clause_order`: Allow conjunct reordering, but do not silently reverse relation arguments.
- `features`: Features.
- `CorrectionValidityModel`: Group related CorrectionValidityModel operations.

### `src/evaluate.py`

evaluate.py

- `load_eval_resources`: Load eval resources.
- `run_model_benchmark`: Run model benchmark.
- `run_hallucination_type_analysis`: Run hallucination type analysis.
- `run_evidence_quality_experiment`: Run evidence quality experiment.
- `run_fixed_vs_adaptive_benchmark`: Run fixed vs adaptive benchmark.
- `run_correction_evaluation`: Run correction evaluation.
- `run_unseen_question_experiment`: Run unseen question experiment.
- `run_ablation_study`: Run ablation study.
- `generate_master_comparison_table`: Generate master comparison table.
- `run_full_evaluation`: Run full evaluation.

### `src/grounding.py`

Explainable lexical and local numerical grounding (heuristics, not entailment).

- `clean_text`: Clean text.
- `content_tokens`: Content tokens.
- `numerical_conflicts`: Check each number against the most lexically relevant evidence clause.

### `src/preprocessing.py`

preprocessing.py

- `extract_evidence`: Normalize evidence field according to RAGTruth format.
- `split_sentences_with_spans`: Split text into sentences while recording character start and end spans.
- `load_raw_ragtruth`: Load and merge source_info.jsonl and response.jsonl.
- `create_leak_free_splits`: Split train into train and val by source_id so no claims from the same
- `run_pipeline`: Execute preprocessing and persist processed splits.

### `src/reliability.py`

Probability calibration and source-cluster uncertainty for detector evaluation.

- `probability_metrics`: Probability metrics.
- `source_bootstrap_f1`: Resample sources, keeping correlated claims together; percentile 95% CI.

### `src/settings.py`

Shared configuration and reproducibility helpers; no model downloads on import.

- `load_config`: Load config.
- `seed_everything`: Seed everything.

### `src/train.py`

train.py

- `ClaimDataset`: PyTorch Dataset for Premise (Evidence) and Hypothesis (Claim) pairs.
- `load_claim_data`: Load train and val claim datasets.
- `tune_baseline`: Tune baseline.
- `evaluate_transformer`: Evaluate transformer.
- `tune_transformer`: Tune transformer.
- `run_training_pipeline`: Run training pipeline.

### `src/verification.py`

verification.py

- `compute_evidence_quality`: Evaluates evidence quality based on grounding, length, and content token overlap.
- `VerificationEngine`: Multi-verifier engine executing:
- `run_no_verification`: Policy A: Baseline Single-Detector.
- `run_fixed_verification`: Policy B: Fixed Multi-Check Verification.
- `run_adaptive_verification`: Policy C: Adaptive Evidence-Guided Verification.
- `compare_verification_policies`: Compares No Verification, Fixed Verification, and Adaptive Verification

### `test_integration.py`

Run from installed project root: python -m unittest test_integration -v.

- `IntegrationTests`: Group related IntegrationTests operations.

### `test_model.py`

Local NLP project module: test model.

- `ModelTests`: Group related ModelTests operations.

### `tests/test_correction.py`

test_correction.py

- `TestCorrection`: Group related TestCorrection operations.

### `tests/test_evaluation.py`

test_evaluation.py

- `TestEvaluationIntegrity`: Group related TestEvaluationIntegrity operations.

### `tests/test_improvements.py`

Local NLP project module: test improvements.

- `ImprovementTests`: Group related ImprovementTests operations.

### `tests/test_pipeline.py`

test_pipeline.py

- `TestIntegrationPipeline`: Group related TestIntegrationPipeline operations.

### `tests/test_preprocessing.py`

test_preprocessing.py

- `TestPreprocessing`: Group related TestPreprocessing operations.

### `tests/test_verification.py`

test_verification.py

- `TestVerification`: Group related TestVerification operations.

### `train_model.py`

Reproducible training with disjoint relation templates and entity names.

- `generate`: Generate.
- `metrics`: Metrics.
- `main`: Main.

### `verify_correction.py`

CLI: python verify_correction.py --original ... --evidence ... [--candidate ...]

- `main`: Main.

## Notebooks, configuration, and artifacts

`notebooks/ragtruth_mse1.ipynb` and `notebooks/ragtruth_mse2.ipynb` are retained historical workflows. Their saved outputs are not regenerated CHC measurements. The former contains baseline-oriented experiment code; the latter includes orchestration and older reported tables. Do not copy those numbers into a CHC paper.

`configs/config.json` controls the original pipeline. `configs/chc.json` controls the new small-model experiment. Keep their seeds and results distinct. Saved detector weights remain in `results/checkpoints`; intervention weights are in `outputs/chc/<stage>/models`.

## Suggested reading order

1. schema.py and data.py: understand records and what labels are forbidden.
2. graph.py and injection.py: understand hypotheses versus controlled transformations.
3. detector.py and generation.py: see actual model calls and cache handling.
4. experiment.py: follow one no-action/action pair and where reference scoring occurs.
5. models.py and policy.py: follow target construction, learning, and selection.
6. metrics.py and plots.py: inspect denominators, missing labels, and uncertainty.
7. cli.py and web.py: execute the experiment and inspect live predictions.

## Common debugging questions

If all action benefits are zero, inspect accepted corrections and actual outputs before training a larger model. If the oracle also cannot help, better selection alone cannot solve the environment. If train results improve but test does not, inspect source/topic overlap and data scarcity. If the detector is badly calibrated, do not reinterpret its raw confidence as a reliable probability. If a source-linked correction damages meaning, preserve its rejected status and add human-reviewed cases rather than assuming the checker is authoritative.

## Reading the central code

`utility(before, after)` first calls `observed_loss` on actual recorded continuations. If the downstream factual denominator is unknown, it returns `None`; the training mask excludes that candidate. It does not silently teach the regressor that unknown means zero gain. A final-only target can still be known when a richer target is missing, so evaluation must disclose cohort/target coverage.

`trace_state` copies only current text and supplied citations into the old typed graph. `node_features` runs before `training_graph` joins targets. The V2 change in `models.py` checks an explicit `supervision_scope`, reads `training_benefits`, masks missing labels, and preserves V1's default final-correctness difference. The saved raw `correct` values are not overwritten with utility numbers.

`collect` calls the backend without the reference answer. Only `score_continuation` receives the reference afterward. It actually regenerates each different prefix, reusing a cache only for exact matches of prompt/config/seed. Joint intervention calls use the joint prefix, so their measured outcome can differ from the sum of singleton effects.

`align_uncited_artifact` performs structural validation before source inference. Temporary source IDs serve only to test the schema; they cannot survive without the NLI entailment gate. It preserves the raw output and records every assigned citation. An original model citation is distinguishable from an automatically assigned one. Bad dependency order is never repaired.

`RepairVerifier.verify` asks the existing trained lexical verifier for a proposal and checks the proposal against the source. It also checks original-aware entity anchors and whether a supported original is being unnecessarily replaced. These are conservative diagnostics, not a trained universal semantic preservation classifier. The gate-off ablation keeps exactly the same proposal.

`evaluate_models` receives recorded outcomes only for scoring. Each policy sees `DecisionState`, predicts priorities, and the strict budget selector chooses IDs. The evaluator then looks up the corresponding actual outcome. This replay is efficient because actions were exhaustively collected for the tiny prefix, but it must never be counted as a fresh model generation per policy row.
