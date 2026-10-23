# CHC V2 accuracy improvement and GitHub readiness report

Report date: 23 October 2026. All performance numbers below come from saved predictions and recorded real experiments. No models were retrained for this report.

## Direct answer

The hallucination detector improves on matched data. End-to-end answer accuracy does not improve for the main gated containment system. These are different tasks and must not be combined into one accuracy claim. The college research release is complete as a prototype, but the current working repository is not ready for an indiscriminate GitHub push.

## 1 What base model and improved model mean

There are two separate comparisons. The detector classifies whether generated text contains hallucinations. The containment system chooses which reasoning claim to verify and then asks a language model to continue. Detector F1 or accuracy is not the same as answer correctness.

For the detector, the base is the preserved DistilBERT checkpoint with threshold 0.5. The selected improved version uses the same neural weights with Platt calibration and a threshold selected only on validation data. This is a decision-rule/calibration improvement; it is not new transformer finetuning.

## 2 Detector comparison on identical test examples

This comparison uses the same 1,024 natural-distribution RAGTruth test rows, including 94 hallucination-positive rows and 930 negative rows. Accuracy, precision, recall and F1 are recomputed from the actual prediction CSV.

| Variant | Accuracy | Precision | Recall | F1 | True positive | False positive | False negative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original detector fixed threshold | 64.26% | 16.00% | 68.09% | 25.91% | 64 | 336 | 30 |
| Original detector threshold tuned on validation | 83.50% | 26.42% | 44.68% | 33.20% | 42 | 117 | 52 |
| Validation selected calibrated detector | 84.18% | 27.63% | 44.68% | 34.15% | 42 | 110 | 52 |

**Accuracy improvement: 64.26% to 84.18%, a gain of 19.92 percentage points.** Correct classifications increase from 658 to 862 out of 1,024.

**F1 improvement: 25.91% to 34.15%, a gain of 8.24 percentage points, or 31.8% relative improvement.** F1 is a precision/recall balance; it must not be called answer accuracy.

The tradeoff is material: false positives fall from 336 to 110, but detected hallucinations fall from 64 to 42, and missed hallucinations rise from 30 to 52. The selected operating point is more conservative. It improves F1 and overall classification accuracy while reducing recall.

Because 90.82% of this sample is negative, predicting “no hallucination” for every row would achieve 90.82% accuracy while detecting none. Therefore accuracy alone is misleading here. Report precision, recall, F1 and the class distribution with it.

Most of the gain over the old default comes from choosing an operating threshold on validation data. Threshold tuning alone gives 83.50% accuracy and F1 33.20%. Calibration plus its selected threshold gives 84.18% accuracy and F1 34.15%: approximately +0.68 accuracy points and +0.94 F1 points beyond threshold tuning alone. The unchanged full-128 PR average precision (.3686) indicates no learned improvement in ranking from monotonic calibration.

Confidence calibration improves too: ECE falls from .36037 to .04866 and Brier score from .23573 to .07595 on the same 1,024 rows. Lower values are better. These quantities evaluate probability reliability rather than final-answer accuracy.

Paired source-cluster bootstrap gives F1 difference +.08235 with descriptive 95% interval [.02474, .13440], using 397 source clusters and 2,000 resamples. This is not a multiplicity-adjusted confirmatory significance claim.

On the historical matched 256-row sample, F1 is .1263 to .1633 (+3.69 points). Comparing .1263 on 256 rows directly with .3415 on 1,024 rows is invalid.

## 3 Actual answer accuracy of the containment system

The fair V2 comparison uses the same 16 eligible held-out questions and the same Qwen backend. “Base” below means the corrupted continuation without a verification intervention, not the historical FLAN experiment.

| Condition | Correct answers | Exact-match accuracy | Gain over no action |
| --- | --- | --- | --- |
| No verification action | 2/16 | 12.50% | Reference |
| Richer tabular policy with one action | 2/16 | 12.50% | 0.00 points |
| Richer graph policy with one action | 2/16 | 12.50% | 0.00 points |
| Richer semantic policy with one action | 2/16 | 12.50% | 0.00 points |
| Main gated policies with two actions | 2/16 | 12.50% | 0.00 points |
| Gate-off ablation with two actions | 3/16 | 18.75% | +6.25 points |

The gate-off result is one extra correct answer on 16 questions. It is an exploratory ablation, not the validated primary model and not evidence for disabling the gate in deployment. The main containment model has **zero demonstrated answer-accuracy improvement**.

Of the 60 originally requested test questions, only 16 pass eligibility, or 26.7%. Conditional accuracy on those questions is not accuracy on all 60 requested inputs. All 16 primary test episodes have complete action outcomes; this is completion coverage within an already filtered cohort.

The previous FLAN answer-only experiment has 44.822% no-action exact match across 3,708 seeded episodes. Its backend, task, samples, seeds and eligibility rules differ from V2. The move from 44.822% to 12.5% cannot be interpreted as a measured improvement or regression. The language-model weights were not finetuned, and a matched generator-only FLAN versus Qwen accuracy improvement has not been established.

## 4 What improved besides final-answer accuracy

The measurement can now distinguish a factual continuation improving, staying unchanged or getting worse even when the final answer stays the same. Across 90 primary episodes, richer singleton effects include 15 positive, 149 neutral, 10 negative and six missing targets. Final-only targets include nine positive, 170 neutral and one negative effect. Final answers remain unchanged for 141 of 180 actions.

At one action, the richer tabular policy has mean signed utility +.0167, interval [0, .05], and downstream error-rate reduction .0333, interval [0, .10]. The paired utility comparison with matched final-only and full final-only tabular controls uses 15 evaluable questions. These are small automatic-proxy effects; zero is at the lower confidence bound. Neither interval establishes robust superiority. Graph utility is -.0208 and semantic utility -.0313, so those added models do not improve this run.

The harmful-repair proxy is zero in the primary cohort, but prefiltering and a shared automatic judge prevent interpreting this as proven repair safety. No independent human labels were fabricated. A real live example still returns “Yes” to a date question and confuses launch with landing evidence. The system is a research prototype, not a reliable factual assistant.

## 5 What changed in the project

| Area | Implemented change | Why it matters |
| --- | --- | --- |
| Local reasoner | Frozen Qwen structured claim traces and cached CUDA generation | Makes intermediate continuations observable |
| Validation | Strict claim IDs, dependencies, citations and raw failure logging | Separates parse failure from factual error |
| Evidence | Citation attachment preserving model text and contradictory outputs | Prevents hiding errors during retrieval |
| Fact units | Conservative splitting and four-way NLI evidence status | Separates unsupported from contradicted claims |
| Corruptions | Source-local mutations with cross-source validity veto | Rejects still-true controlled replacements |
| Interventions | Actual clean/corrupt/singleton/joint/gate-off generations | Measures effects rather than assuming them |
| Learning targets | Signed downstream factual/answer loss and missing-label masks | Retains neutral and negative outcomes |
| Models | Tabular, GNN, frozen-embedding Ridge and stopping variants | Tests whether extra complexity helps |
| Controls | Matched and full final-only controls, historical controls, ablations | Makes comparisons more defensible |
| Calibration | Validation-only Platt fitting and threshold choice | Improves the detector operating point and probability reliability |
| Evaluation | Question/source bootstrap, coverage and prediction exports | Exposes uncertainty and selection effects |
| Application | Preserved old routes plus recorded/live propagation UI | Supports a working demonstration |
| Release | 109 passing tests, paper, detailed docs, viva guide and launcher | Makes the project inspectable and reproducible |

The V2 implementation is mainly in `chc/v2/`; `chc/models.py` joins V2 labels after pre-action features are built. `app.py` adds `/chc/propagate` and `/api/chc-v2`. Scripts implement collection, training, finalization, detector evaluation and review import. Existing DistilBERT, correction-validity and historical reasoner weights passed hash-preservation checks. Thirty-one policy checkpoints were fitted; this is not 31 independent architectures.

The broader working tree also contains modifications from earlier project phases. Git does not identify which person or session made each uncommitted change, so this report does not attribute every dirty file to V2. These pre-existing edits must be reviewed rather than blindly staged.

## 6 Is it ready to push to GitHub

**Not as-is with `git add .`.** The report/PDF is ready as a college research draft, but repository publication needs a curated commit. No files were staged, committed or pushed during this audit.

Audit branch: `codex/chc-research-v2`. Git reports 70 changed/untracked status entries and 470 candidate files. These include source, research outputs, backup files and earlier-phase edits.

There are 0 Git candidates at least 50 MiB, of which 0 exceed 100 MiB. See the JSON audit for actual paths. Existing `.gitignore` excludes `.pt` GNN checkpoints as well as `.safetensors`/`.bin`; a normal source push therefore does not publish all trained models. Preserve them in a release asset or model hosting service and document retrieval.

There are 14 backup candidates and 6 inspected text files containing machine-specific paths. For example, the launcher has a development-machine cache fallback, and the reproduction command references a local historical checkpoint directory. Environment/configuration overrides should be documented or made portable before publishing as a reusable project.

A bounded token/private-key pattern scan flagged 0 candidate files. A zero count does not prove the repository is secret-free. The scan does not cover every credential format, binary, notebook, large text file or Git history.

No top-level LICENSE file was found. Choose a software license you are entitled to grant and retain dataset/model notices; do not apply your software license to upstream model weights or data indiscriminately. Large processed RAGTruth data and frozen weights are intentionally not a self-contained part of the archive.

Before a push: review the diff; exclude backup directories and machine-local caches; choose a license; verify secrets and data redistribution; document model downloads; stage the intended source/docs/tests/configs and selected results explicitly; inspect the staged diff; then commit. If you want a publicly reproducible repository, test installation from a clean checkout as a separate release check.

GitHub readiness here refers to publication packaging. It does not mean the research findings need to be positive before publication. The negative answer-accuracy result and automatic-judge limitations should remain in the README and paper.

## 7 What to say in a college presentation

“We built an evidence-grounded framework that learns which verification action reduces downstream factual errors. On a matched 1,024-example detector evaluation, accuracy increased from 64.26% to 84.18% and F1 from 25.91% to 34.15%, mainly by changing the operating threshold and calibrating the existing detector. Recall decreased, so this is a tradeoff. The main containment model did not improve answer accuracy on 16 eligible test questions, although the tabular policy showed a small factual-error proxy benefit. We report those limitations and provide code, predictions, controls and a reproducible pilot.”

## Evidence files

- `chc_v2/detector/predictions.csv` and `metrics.csv`: detector scores and matched metrics.

- `chc_v2/detector/paired_source_bootstrap.json`: descriptive detector uncertainty.

- `chc_v2/pilot/evaluation/policy_summary.csv` and `paired_comparisons.csv`: action-budget results.

- `chc_v2/pilot/collection_summary.json` and `cohort_audit.json`: counts and eligibility.

- `chc_v2/final_release_checks.json`: frozen and legacy-weight checks.

- `CHC_GITHUB_READINESS_AUDIT.json`: actual Git candidate findings and scope.

## Publication preparation update 23 October 2026

The earlier GitHub-readiness section records the pre-cleanup audit. Publication preparation now adds upstream attribution, ignores backups/private local configuration, makes the launcher portable, marks media as binary, and explicitly includes all 31 small policy checkpoints. No new license is assigned over inherited code because upstream has no declared license. A curated branch is prepared; pushing from the desktop execution environment is blocked because Windows denies Git Credential Manager access to saved credentials.
