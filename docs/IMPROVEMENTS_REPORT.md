# NLP Hallucination Project — Implementation Report

Date: 22 October 2026

Project updated in place: `C:\Users\BroCode\Documents\nlphallucination\nlphallucination`

## 1. What the project does

The project studies whether a language-model response contains claims unsupported by supplied evidence. Its input is an evidence passage and a claim, usually a sentence extracted from a RAGTruth response. It predicts a binary hallucination label, can abstain when evidence or verifier agreement is inadequate, and can propose a replacement for a flagged claim.

The system combines three different signals:

1. A trained detector: either TF-IDF with Logistic Regression or a fine-tuned DistilBERT cross-encoder.
2. A lexical coverage heuristic: content from the claim is checked against words in the evidence.
3. A numerical grounding heuristic: numbers in the claim are checked against relevant evidence clauses.

Adaptive routing chooses how many stages to run. A confident, well-grounded nonnumerical claim may take the fast path. Uncertain cases reach additional checks. Numerical claims now reach the numerical stage before acceptance, unless that stage is explicitly disabled for an ablation. Missing evidence produces abstention.

This is a research prototype. Its heuristic scores and detector decisions are not independent proof of factual truth.

## 2. Main improvements

### Local numerical grounding

Previously, any matching number anywhere in the evidence could support a claim. For example:

```text
Evidence: Alice scored 10 points; Bob scored 20 points.
Claim:    Alice scored 20 points.
```

The old numerical rule saw “20” in the evidence and missed the mismatch. The new `src/grounding.py` splits evidence into clauses, scores each clause by content-word overlap with the claim context, and checks numbers within the best matching clauses. Here, Alice’s clause has more relevant tokens than Bob’s clause, so the mismatch is flagged.

Numerical strings use Decimal normalization, allowing `1,000` and `1000.0` to match. When equally relevant clauses disagree about support, the result is marked ambiguous instead of asserting a definite numerical conflict. Each check exposes the number, character offsets, relevant clauses, supported flag, ambiguous flag, and conflict flag.

This is lexical context matching, not full entity linking. Coreference, negation, units, complex dates, and table interpretation remain limitations.

### Safer correction behavior

Correction remains evidence-guided: it attempts contextual number replacement or substitutes a relevant evidence sentence. Improvements include:

- Replacing every occurrence of a matched numerical token, rather than only its first occurrence.
- Applying replacements in one regex pass to prevent replacement cascades.
- Running the numerical check during post-correction verification.
- Rejecting proposals with ambiguous numerical grounding.
- Returning the original claim when a proposal fails verification.
- Retaining the proposed text and rejection flag for audit.

An important return value now follows this pattern:

```python
'corrected_claim': corrected_claim if is_still_hal == 0 else claim,
'proposed_claim': corrected_claim,
'proposal_rejected': bool(is_still_hal),
'was_edited': is_still_hal == 0,
```

`corrected_claim` is the final output. `proposed_claim` records what the system tried. A rejected proposal therefore remains inspectable without replacing the original output. An unchanged detector decision now has an unknown post-factuality value instead of claiming that it is independently factual.

### Honest correction evaluation and independent review

The same detector and lexical checks cannot independently establish whether their own accepted correction is true. Automatic acceptance is now explicitly identified as a proxy. The legacy CSR key remains for compatibility and carries the same proxy value; a new clearly named verifier-accepted correction rate is also available.

Automatic CIHR is now null. The system no longer reports an automatic zero as independent evidence of safety. Charts show rejected proposals instead, and the comparison table states that CIHR requires independent review.

The new `review_corrections.py` exports evidence and final claims to a review CSV. Reviewers fill in:

- `independent_post_label`: 0 supported, 1 hallucinated, 2 uncertain.
- `reviewer`: reviewer identifier.
- Optional notes.

Hide the original gold-label column during review to reduce bias. The scorer refuses incomplete reviews and computes independently labeled CSR and CIHR. Uncertain outcomes do not count as correction successes, and their proportion is reported separately. Human annotation quality and reviewer agreement still need their own research protocol.

### Comparable policy evaluation

Selective F1 excludes abstained claims, which can make comparisons misleading when policies cover different populations. The policy summary now adds:

- F1 on the exact same claims covered by every compared policy.
- The number of those jointly covered claims.
- Hallucinations detected divided by all gold hallucinations, retaining abstained positives in the denominator.

Existing selective F1 and coverage remain available. These columns answer different questions: how accurate the answered subset is, how comparable the policies are on the same items, and how much of the overall hallucination burden was detected.

Policy execution order is randomized with a fixed seed after warm-up. Benchmark timers wrap complete policy calls, including routing overhead. Adaptive results also include complete internal elapsed time. Full-system latency no longer adds detection twice, and full-system call counts include post-correction checks.

### Shared ablation implementation

The earlier ablation code duplicated routing and changed unrelated thresholds, weights, and available stages. The variants now call the same `run_adaptive_verification` implementation with an explicit component disabled:

```python
variant(use_evidence_quality=False)
variant(disabled_stage=2)
variant(disabled_stage=3)
```

Removing a stage necessarily changes the available fusion signals; remaining fusion weights are normalized. The numerical stage’s safety gate is also absent in the numerical-stage ablation. These effects are explicit, and other thresholds remain shared. The fixed policy remains a policy comparison because its fusion differs from the adaptive policy.

### Calibration and uncertainty metrics

The new `src/reliability.py` adds Brier score, expected calibration error, average precision, ROC AUC, and a 95% source-cluster bootstrap interval for detector F1.

Brier score measures squared probability error; lower is better. Calibration error compares predicted hallucination probabilities with observed hallucination frequency within probability bins. Average precision is useful under class imbalance. ROC AUC evaluates ranking across thresholds and is unavailable when only one class is present.

Bootstrap resampling keeps claims from the same source together:

```python
groups = [np.flatnonzero(source_ids == source) for source in sources]
indices = np.concatenate([
    groups[i] for i in rng.integers(len(groups), size=len(groups))
])
```

Each sampled source contributes all of its claims. This respects within-source dependence more closely than independently resampling sentences. The default is 200 replicates; the interval is a percentile estimate, and it can be unstable for small or unrepresentative samples.

Both detector probabilities and predictions are now logged for natural and balanced evaluation subsets. Previously, the persistent benchmark log retained only balanced baseline predictions.

### Configuration and reproducibility

`src/settings.py` provides cached configuration loading and shared seeding. The implementation now reads routing and evidence-quality thresholds, model architecture/context length, training batch size, epochs, weight decay, selected hyperparameter grid values, and the random seed from configuration.

Training seeds Python, NumPy, and PyTorch. Each learning-rate trial starts with the same seed to reduce initialization differences across trials. Deterministic cuDNN options reduce variability; they do not guarantee identical results across environments. Some execution budgets remain explicit defaults, as documented in the README.

Preprocessing preserves the original response text when projecting character annotations. Trimming a response before projection could shift annotation offsets. Multi-subtype selection is now deterministic instead of depending on set iteration order. Prerequisite checks now include validation data.

## 3. New project tools

### Claim inspector: `analyze_claim.py`

This command turns the research components into a usable local inspection tool. It produces JSON containing model identity, evidence-quality score, missing content words, numerical evidence clauses, route, prediction, latency, correction details, and a review recommendation.

```powershell
python analyze_claim.py --claim "Alice scored 20 points." --evidence "Alice scored 10 points; Bob scored 20 points."
python analyze_claim.py --model baseline --claim "Revenue was 90 million." --evidence "Revenue was 50 million."
```

The default is explicitly labeled heuristic mode. Baseline and transformer modes use saved local checkpoints. Transformer inspection uses `local_files_only=True` and does not download model weights. An example JSON output is supplied with this report.

### Dataset audit: `audit_dataset.py`

The audit checks required columns, binary labels, blank claims/evidence, source-ID overlap, and repeated evidence under different IDs. Evidence is lowercased and whitespace-normalized before hashing with SHA-256. Repeated evidence triggers a review warning because generic passages can legitimately repeat; source overlap or invalid labels cause a failing exit status.

```powershell
python audit_dataset.py --output results/tables/dataset_audit.json
```

### Independent correction review

```powershell
python review_corrections.py prepare --input results/predictions/correction_predictions.csv --output results/review/review.csv
python review_corrections.py score --input results/review/review.csv --output results/review/independent_metrics.json
```

The supplied review worksheet contains four smoke-test outputs. It is a demonstration worksheet, not a completed independent evaluation.

## 4. Validation actually performed

All 40 unit and integration tests passed. Fourteen new tests cover numerical entity confusion, numerical formatting, ambiguous evidence, numerical routing, proposal rollback, repeated-number repair, missing values, whole-word coverage, calibration, seeded bootstrap behavior, incomplete review rejection, independent review calculations, common-subset metrics, evidence-fingerprint overlap, and configurable routing. One existing test was updated because unchanged output no longer implies known factuality.

A small smoke run loaded both existing TF-IDF/Logistic Regression and DistilBERT checkpoints. It completed detector benchmarking on eight natural and eight balanced claims, policy comparison on four claims, correction on four claims, ablations on four claims, and master-table generation. These small samples validate execution and output schemas. Their F1 values and timings are not research findings or evidence of improved accuracy.

The saved data audit found:

| Split | Claims | Sources | Hallucinated claims |
|---|---:|---:|---:|
| Train | 96,804 | 2,138 | 10,977 |
| Validation | 16,972 | 377 | 2,020 |
| Test | 18,930 | 450 | 1,641 |

There were zero cross-split source-ID overlaps and zero identical normalized-evidence fingerprint overlaps. No invalid binary labels, blank claims, or blank evidence fields were found. Test prevalence is approximately 8.67%. This audit does not rule out paraphrased evidence, shared entities, or other semantic dependencies.

## 5. Files added or changed

| File | Purpose |
|---|---|
| `src/grounding.py` — new | Missing-value normalization, content tokens, local numerical explanations |
| `src/settings.py` — new | Shared configuration and reproducible seeding |
| `src/reliability.py` — new | Calibration and source-bootstrap uncertainty |
| `analyze_claim.py` — new | Local JSON claim inspector |
| `audit_dataset.py` — new | Saved-data integrity audit |
| `review_corrections.py` — new | Human-review worksheet and independent scorer |
| `tests/test_improvements.py` — new | Fourteen regression tests |
| `src/verification.py` | Configured routing, numerical gate, shared ablations, timing and paired metrics |
| `src/correction.py` | Safer replacement, post-checks, rollback, metric integrity and audit fields |
| `src/evaluate.py` | Calibration, uncertainty, complete prediction logs, corrected aggregate reporting |
| `src/train.py` | Configuration use and controlled random seeds |
| `src/preprocessing.py` | Annotation coordinate preservation and deterministic subtype selection |
| `run_pipeline.py` | Validation-data prerequisite check |
| `tests/test_correction.py` | Updated unknown-factuality expectation |
| `README.md` | Current commands, audited counts, metric interpretation and limitations |
| `docs/IMPROVEMENTS_REPORT.md` — new | A copy of this report in the project |

## 6. What remains to establish scientifically

Existing model weights were not retrained. Existing processed datasets, historical experiment CSVs, plots, notebooks, and paper results were not regenerated. The new code changes routing, metrics, and correction behavior; historical numbers must not be presented as results of this revision.

The next research run should use the full intended evaluation budgets, report both natural and balanced distributions, compare common-coverage and overall detection metrics, and complete independent correction review. A full rerun is available through `python run_pipeline.py --all`; retraining can be requested with `--all --force-train`.

DistilBERT still has the configured 128-token pair budget, so long evidence can be truncated. Lexical overlap does not establish entailment. Corrections can substitute a whole evidence sentence and change the original claim’s scope. A separate entailment model, retrieval of supporting evidence windows, reviewed corrections, and agreement between multiple reviewers would be valuable future extensions.

The implemented improvements make the prototype more inspectable, reproducible, and cautious. They provide the machinery needed to measure progress without treating verifier agreement as independent factual truth.
