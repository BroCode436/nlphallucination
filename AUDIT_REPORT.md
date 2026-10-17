# Scientific & Technical Audit Report
**Project:** Adaptive Evidence-Guided Hallucination Detection and Correction with Post-Correction Verification for Large Language Models  
**Dataset:** RAGTruth (ACL 2024 Benchmark)  
**Date:** October 2026  
**Auditor:** Senior ML Engineer & Research Auditor  
**Status:** Audit Completed & All Defects Remediated and Verified

---

## Executive Summary
This audit report documents the comprehensive inspection, debugging, leakage elimination, and verification of the research codebase. All critical issues raised in auditing—most notably ground-truth leakage in the correction pipeline, metric coercion on abstention, and split/confusion matrix alignment—have been resolved directly in source code, validated with 21 unit/regression tests, and benchmarked on live dataset executions with zero fabricated or mock data.

---

## Detailed Remediation of Audited Issues

### 1. Separation of Abstention from Predictions (`prediction=None`)
- **Root Cause:** In earlier evaluation loops (`run_evidence_quality_experiment`, `run_unseen_question_experiment`, `run_ablation_study`), `prediction=None` was coerced into `0` (factual).
- **Engineering Fix:** Removed every conversion of `prediction=None` into `0`. Evaluated metrics (Accuracy, Precision, Recall, F1) are computed strictly over **covered claims** where `prediction is not None`.
- **Reporting:** Separate metrics for **`Coverage (%)`** and **`Abstention Rate (%)`** are now explicitly reported across all experimental tables.

### 2. Confusion-Matrix Subset Alignment Bug
- **Root Cause:** In `run_model_benchmark()`, loop variable shadowing caused `base_preds` to contain the 2,000-claim natural test subset while `confusion_matrix` evaluated the 1,000-claim balanced subset.
- **Engineering Fix:** Implemented dedicated variables `balanced_base_preds` and `natural_base_preds`. The confusion matrix now strictly pairs `df_test_balanced['label'].values` with `balanced_base_preds`, matching exact dimensions and subsets.

### 3. Adaptive Routing Conflict Resolution
- **Root Cause:** When Primary Detector (Stage 1) and Lexical Coverage (Stage 2) yielded conflicting probabilities with high spread, the system defaulted without deep verification or explicit abstention.
- **Engineering Fix:** In `src/verification.py`, `prob2` is computed prior to agreement checks. Conflicting verifiers (`abs(prob1 - prob2) >= 0.50`) now trigger Stage 3 (Numerical/Date Rule Verifier). If Stage 3 does not resolve the conflict decisively, the system outputs an explicit abstention (`status: Abstain-Conflicting-Verifiers`, `prediction: None`), preserving scientific precision. Added `VerificationEngine.adaptive_verify` method for consistent object-oriented invocation.
- **Empirical Validation:** On the 600-claim test subset, Adaptive Verification achieves **0.6571 F1**, **0.8050 Precision**, and **0.6968 Accuracy** across covered claims (**92.33% coverage**, 46 abstentions), reducing average verification calls from 3.00 down to **2.45 calls/claim**.

### 4. Ground-Truth Leakage Remediation & Context-Aligned Numerical Correction
- **Critical Leakage Root Cause:** In `src/correction.py`, line 279 previously contained:
  ```python
  detected_hal = item.get('prediction', lbl)
  ```
  When test records omitted `'prediction'`, the system fell back to the gold ground-truth label `lbl`. This shielded factual claims from being targeted by false-positive detector alarms, artificially creating a 100% CCPR and 0% UER.
- **Engineering Fix:**
  - Completely eradicated fallback to `lbl`. Correction decisions at inference time depend solely on genuine model predictions (`item['prediction']` or dynamic inference via `corrector.engine.adaptive_verify(ev, cl)`).
  - If a record is missing `'prediction'` and no verification engine is attached, the system raises an explicit `KeyError` instead of falling back to ground truth.
  - If the detector abstains (`prediction is None`), the corrector explicitly abstains from modifying the claim.
  - Gold labels `lbl` are strictly sequestered for post-hoc calculation of CSR, RHR, CIHR, CCPR, and UER.
  - Implemented `align_numerical_entities()` with stopword-filtered entity context windows to prevent ambiguous number substitutions.
- **Empirical Impact (Before vs. After Leakage Fix):**
  - *Leaked (Invalid):* CSR = 78.85%, CCPR = 100.00%, UER = 0.00%.
  - *Un-leaked (Authentic):* CSR = 37.98%, CCPR = 87.50%, UER = 12.50%, CIHR = 0.00%, AR = 14.75%.
  - The authentic metrics honestly reflect detector false positives (12.50% UER on factual claims), while proving that post-correction verification successfully prevented any factual claims from being corrupted into hallucinations (**0.00% CIHR**).

### 5. Optimal DistilBERT Checkpoint Saving
- **Root Cause:** `tune_transformer()` in `src/train.py` saved weights after the training loop completed (saving the final epoch), regardless of whether validation F1 peaked at an earlier epoch.
- **Engineering Fix:** Added `best_val_f1` tracking inside the epoch loop. Whenever validation F1 improves, `model.save_pretrained()` and `tokenizer.save_pretrained()` are called immediately, and `training_meta.json` records `best_epoch`, `best_val_f1`, and hyperparameter configs.

### 6. Safe Split Creation & Small-Dataset Handling
- **Root Cause:** `create_leak_free_splits()` in `src/preprocessing.py` failed with `FileNotFoundError` if the manifest directory did not exist, and produced an empty validation set on tiny datasets ($N \le 3$).
- **Engineering Fix:** Added `os.makedirs(os.path.dirname(manifest_path), exist_ok=True)` and safe split allocation (`n_val = max(1, int(len * 0.15))` when unique sources $\ge 2$), guaranteeing non-empty validation splits.

### 7. CLI Pipeline Alignment & Training Flags
- **Engineering Fix:** Updated `run_pipeline.py` with accurate documented flags: `--all`, `--test`, `--preprocess`, `--train`, `--force-train`, `--evaluate`. Added `check_prerequisites_exist()` so models are not needlessly retrained if checkpoints already exist, while allowing manual re-training when desired.

### 8. Comprehensive Regression Test Suite
- **Added Regression Tests (21 Tests Total):**
  - `tests/test_preprocessing.py`: `test_tiny_split_sizes_creates_non_empty_val`, `test_single_source_handling`.
  - `tests/test_verification.py`: `test_conflicting_verifiers_trigger_deep_verification_and_abstention`.
  - `tests/test_correction.py`:
    - `test_context_aligned_numerical_correction_with_multiple_evidence_numbers`
    - `test_ambiguous_numerical_correction_abstains`
    - `test_omitted_prediction_without_engine_raises_keyerror` (Regression: Gold label leakage prevention)
    - `test_gold_label_cannot_control_correction_decision` (Regression: Prediction governs correction, not label)
  - `tests/test_evaluation.py`: `test_confusion_matrix_subset_alignment`, `test_no_prediction_none_to_zero_conversion`.
- **Outcome:** `Ran 21 tests in 0.052s. OK`.

---

## Exact Files Modified in this Debugging Pass

| File | Changes Made |
| :--- | :--- |
| `src/correction.py` | Eradicated `item.get('prediction', lbl)` gold label leakage, added out-of-sample detector prediction binding, stopword-filtered numerical entity alignment. |
| `src/verification.py` | Added `adaptive_verify` convenience method on `VerificationEngine`, fixed stage 2 evaluation order, deep Stage 3 routing on conflicting verifiers, explicit `Abstain-Conflicting-Verifiers`. |
| `src/preprocessing.py` | Added safe manifest directory creation, tiny split size handling, non-empty validation guarantee. |
| `src/train.py` | Added best validation epoch checkpoint tracking and optimal weight saving. |
| `src/evaluate.py` | Fixed confusion matrix subset alignment, removed all `prediction=None` to 0 conversions, added separate Coverage/Abstention reporting. |
| `run_pipeline.py` | Added `--train`, `--force-train`, prerequisite checking before evaluation runs. |
| `tests/test_correction.py` | Added regression tests verifying zero gold-label leakage and prediction-controlled editing. |
| `tests/test_preprocessing.py` | Added tiny split and single-source safety regression tests. |
| `tests/test_verification.py` | Added conflicting verifier deep verification and abstention regression test. |
| `tests/test_evaluation.py` | Added confusion matrix alignment and non-coercion regression tests. |
| `build_updated_mse2_notebook.py` | Updated with live un-leaked execution outputs and re-generated `notebooks/ragtruth_mse2.ipynb`. |
| `RESULTS_SUMMARY.md` | Updated with live verified benchmark results and un-leaked correction metrics. |
| `paper/ieee_research_paper.md` | Updated empirical tables and analytical findings. |
| `README.md` | Updated CLI usage documentation and verified findings. |

---

## Remaining Blockers
- **None.** All user requirements have been resolved, regression-tested, and verified through live pipeline execution.
