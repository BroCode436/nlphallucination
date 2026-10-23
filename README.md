# Causal Hallucination Containment Research Prototype

Derived from [Prateekbaliyan/nlphallucination](https://github.com/Prateekbaliyan/nlphallucination). See [upstream attribution and rights](UPSTREAM_NOTICE.md) and [data notices](docs/DATA_NOTICE.md). No new software license is assigned to inherited code.

## Executed CHC V2 study

College research prototype investigating whether signed downstream factual-error targets help choose verification actions. The release includes actual model continuations, 31 small policy checkpoints, matched baselines, uncertainty estimates, a six-page paper and 109 previously passing tests. External transformer weights and the processed RAGTruth corpus are not bundled.

On the same 1,024 RAGTruth examples, detector accuracy changes from 64.26% to 84.18% and F1 from 25.91% to 34.15%, mainly through validation-selected thresholds/calibration of unchanged neural weights. Recall decreases from 68.09% to 44.68%. The main gated containment policies do not improve final-answer accuracy: 2/16 eligible test questions remain correct. These are separate tasks. Broad causal, safety or state-of-the-art superiority is not established.

Read the [accuracy comparison](docs/ACCURACY_AND_CHANGES.md), [paper](docs/CHC_COLLEGE_RESEARCH_PAPER.pdf), [experimental report](docs/EXPERIMENTAL_RESULTS.md) and [code explanation](docs/COMPLETE_CODE_EXPLANATION.md).

## Quick start

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements_chc_v2.txt
./run_chc_v2.ps1
```

Open the printed local address with `/chc/propagate`. Recorded mode uses shipped experiment outputs. Live mode downloads frozen external models on first use unless `CHC_V2_CACHE` points to an existing cache; it requires enough memory and remains error-prone. Optional `chc.local.json` stores private workstation cache/dependency paths and is ignored by Git. The launcher itself contains no developer-specific path.

A no-download recorded demonstration is available at [docs/CHC_V2_DEMO.html](docs/CHC_V2_DEMO.html). Download/open the HTML file locally; GitHub's code view does not execute it. The [reproduction guide](docs/GITHUB_REPRODUCTION.md) explains data/model requirements, controls and checks.

## Original project and earlier extensions

# Adaptive Evidence-Guided Hallucination Detection and Correction

Research project using RAGTruth to classify sentence-level claims against supplied evidence, route uncertain cases through additional checks, propose corrections, and verify those proposals.

## Current workflow

1. Extract claims while preserving raw annotation coordinates.
2. Split by source ID; audit both source IDs and normalized evidence fingerprints.
3. Train TF-IDF + Logistic Regression and a DistilBERT evidence/claim cross-encoder.
4. Route claims using model confidence, evidence coverage, and configurable thresholds.
5. Check numerical claims against relevant evidence clauses before acceptance.
6. Correct flagged claims, reject unsafe proposals, and preserve originals on failure.
7. Evaluate detectors, calibration, source-bootstrap uncertainty, and policy coverage.
8. Export corrections for independent human review.

The lexical and numerical stages are heuristics. Verifier acceptance does not establish factual correctness. Independent correction safety needs reviewed final outputs.

## Run

Use a Python environment with requirements.txt installed. The active Windows installation used for validation was Python 3.12; invoke `python` if the `py -3.12` launcher is unavailable.

```powershell
python -m unittest discover tests -v
python audit_dataset.py --output results/tables/dataset_audit.json
python analyze_claim.py --claim "Alice scored 20 points." --evidence "Alice scored 10 points; Bob scored 20 points."
python analyze_claim.py --model baseline --claim "Revenue was 90 million." --evidence "Revenue was 50 million."
python run_pipeline.py --all
```

`analyze_claim.py` defaults to explicitly identified heuristic mode. `--model baseline` and `--model transformer` load saved local checkpoints. Transformer inspection never downloads weights.

The full pipeline can train missing checkpoints and takes substantially longer than the tests. `--force-train` with `--all` forces retraining. Training may download the configured pretrained base model.

## Independent correction evaluation

```powershell
python review_corrections.py prepare --input results/predictions/correction_predictions.csv --output results/review/review.csv
# Review every final claim against its evidence. Fill independent_post_label and reviewer.
python review_corrections.py score --input results/review/review.csv --output results/review/independent_metrics.json
```

Use 0 for supported, 1 for hallucinated, and 2 for uncertain. Hide the original gold-label column during review. Incomplete worksheets cannot produce independent safety metrics. The automatic CIHR field is null; the legacy automatic CSR field is a verifier-acceptance proxy only.

## Configuration and reproducibility

`src/settings.py` loads `configs/config.json`. Routing thresholds, evidence-quality thresholds, model architecture/context length, training batch size, epochs, weight decay, and the random seed are consumed by the implementation. Hyperparameter tuning searches several C values, n-gram ranges, and learning rates; values in the config are included in those grids. Some execution budgets, including the default 3,000-claim training subset and 2,000-claim natural evaluation subset, remain explicit code defaults.

Random, NumPy, and PyTorch RNGs are seeded before transformer training; each learning-rate trial starts from the same seed. Deterministic cuDNN settings reduce variability but do not promise identical results across hardware and library versions. The existing checkpoints were not retrained as part of the improvements.

## Audited saved data (22 October 2026)

| Split | Claims | Sources | Hallucinated claims |
|---|---:|---:|---:|
| Train | 96,804 | 2,138 | 10,977 |
| Validation | 16,972 | 377 | 2,020 |
| Test | 18,930 | 450 | 1,641 |

No overlapping source IDs or identical normalized evidence fingerprints were found across these splits. There were no invalid binary labels, blank claims, or blank evidence fields. Test hallucination prevalence is 8.67%.

## Evaluation notes

- Detector benchmarks report F1, MCC, Brier score, calibration error, average precision, ROC AUC, and a source-cluster bootstrap F1 interval.
- Policy benchmarks retain selective F1 and coverage, and add F1 on the same jointly covered claims plus detection recall over all gold hallucinations.
- Policy order is randomized reproducibly after warm-up; timings cover complete policy calls.
- Full-system timings include detection once and post-correction checks; call counts include post-correction verification.
- Automatic correction acceptance is a proxy. Use the independent review workflow for factual CSR and CIHR.
- Historical CSVs, figures, notebooks, and paper claims have not been regenerated. They must not be presented as measurements of this updated implementation.

## Limits

Numerical grounding is lexical and clause-based, not entity linking or logical entailment. Units, negation, coreference, and complex tables remain challenging. DistilBERT still uses the configured 128-token pair budget; evidence can be truncated. Full evidence retrieval, an independent NLI model, and a reviewed correction dataset are future research extensions.

See `docs/IMPROVEMENTS_REPORT.md` for architecture, code explanations, validation, and the change inventory.


## Local correction-validity model

Run `python app.py --port 8765` and open http://127.0.0.1:8765. The saved model in `models/correction_validity.joblib` checks evidence support and preservation of supported original details. It is also loaded automatically by `EvidenceGuidedCorrector`.

CLI: `python verify_correction.py --original "Alice scored 30 points." --evidence "Alice scored 10 points."`

Additional tests: `python -m unittest test_model test_integration -v`. See `docs/CORRECTION_MODEL_REPORT.md` for measured results and limitations. This compact lexical prototype does not establish general semantic factuality or cross-language consistency.

## CHC Research V2

Executed local research release: 300 attempted questions, 90 primary episodes, 31 policy checkpoints, 109 passing tests. Primary held-out answer accuracy remains 12.5%; robust containment superiority is unproven. Read `docs/EXPERIMENTAL_RESULTS.md` and `docs/COMPLETE_CODE_EXPLANATION.md`.

Set `CHC_V2_CACHE` to the existing external-model cache and `CHC_V2_RESULTS` to `outputs/chc_v2/pilot`; run `python app.py` and open `/chc/propagate`. See the experimental report for exact collection/training commands, provenance and limitations.
