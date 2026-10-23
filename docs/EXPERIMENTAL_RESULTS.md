# Causal Hallucination Containment V2 — executed research report

Release date: 23 October 2026. This report describes actual code changes and local experiments, not a proposed experiment or a claim of state-of-the-art performance.

## Executive assessment

The system now measures whether a verification intervention changes downstream evidence-grounded factual errors, rather than learning only whether the final answer becomes correct. It records actual clean, corrupted, corrected, joint-action and gate-off model continuations. Neutral and harmful outcomes remain in the data. Unknown evidence is represented separately from false statements.

The 300-question collection produced 97 raw eligible episodes and 90 primary episodes after clean-prefix factual-unit and pre-outcome mutation-validity audits: 61 training, 13 validation and 16 held-out test questions. This 30% end-to-end eligibility is a major limitation. All 16 primary test questions have complete action outcomes; 100% completeness within this filtered cohort must not be confused with coverage of the original 60 test inputs.

The previous saved FLAN answer-only experiment has 44.822% no-action exact match over 3,708 seeded held-out episodes. V2 has 12.5% over 16 filtered Qwen continuation questions. Different models, tasks, cohorts, seeds and eligibility rules make these numbers incomparable as a before/after improvement or regression. The valid V2 comparison is within its matched held-out cohort, where one-action answer accuracy does not improve. Validated mutation candidates in this pilot occur in four families: negation, entity substitution, numbers and dates. Ten planned families are not ten empirically validated error categories.

At one verification action, every evaluated policy has 12.5% final-answer exact match (2/16). The richer tabular model has positive proxy utility, while graph and semantic models do not beat it. The evidence supports a working measurement and comparison framework with a preliminary tabular signal. It does not establish robust superiority, graph necessity, general causal understanding, or reduced end-to-end serving cost.

## What was implemented

1. A strict structured trace backend using frozen Qwen2.5-1.5B-Instruct, local CUDA inference and exact prompt/configuration caching. Claim IDs, dependencies and citations are validated; raw failures are retained.
2. Explicit evidence attachment that can attach supporting, contradicting or insufficient evidence without editing model claim text. An earlier support-only attachment bug was corrected before the main pilot, because it censored false downstream outputs.
3. Frozen DeBERTa NLI evidence comparison, conservative factual-unit splitting, residual-compound flags, four evidence states, and unit-to-parent trace mapping.
4. Controlled corruptions with source-local contradiction checks and a cross-source veto for still-true entity substitutions. Review-only corruption families are kept separate. Eligibility never uses measured intervention gain.
5. Actual singleton and joint verification interventions, semantic repair gating and gate-off controls. Correcting a prefix triggers real model continuation; effects are not calculated by adding singleton predictions.
6. Signed utility targets, matched final-answer-only controls, a stronger full-coverage final-only control, retrained historical feature controls, graph policies, frozen semantic embeddings, ablations and cost-aware stopping.
7. Validation-only detector calibration and threshold selection, source-cluster bootstrap uncertainty, paired question-level policy intervals, transparent denominators and logical token accounting.
8. Recorded and live application routes, a portable recorded demo, review import tools, dataset notices, code explanations and 109 passing tests. Existing application routes and saved legacy weights are preserved.

## Architecture and important code

`chc/v2/backend.py` produces a short JSON trace. `traces.py` rejects duplicate IDs, invalid dependency ordering and missing citations. These structural checks do not prove that text contains one fact per node.

`alignment.py` retrieves source passages for missing citation metadata. It preserves the actual generated words, including errors. `evidence.py` applies the NLI judge. `atomic.py` splits only conservative sentence/shared-subject structures and flags unresolved compounds; this is an approximation to atomic claim extraction.

`collection.py` records a clean prefix and continuation, then replaces one prefix statement with a prevalidated corruption. It collects the corrupted continuation and the real outcomes of each verification action. The reference boundary stays fixed, so a correction cannot move the scoring boundary in its own favor.

`validity.py` checks the selected mutation before intervention outcomes are used. A mutation that is still supported elsewhere, or duplicates an already supported fact, is vetoed. All rejected attempts and raw records remain available for audit.

`propagation.py` computes:

    observed_error_rate = contradicted / (supported + contradicted)
    loss = 0.5 * observed_error_rate + 0.5 * final_answer_error + 0.25 * harmful_repair_proxy
    intervention_utility = loss_before - loss_after

If no units have support/contradiction judgments, the factual rate is undefined and its richer target is missing. Insufficient evidence is never silently counted as contradiction. Positive utility means improvement; zero means no measured change; negative utility means measured damage. These are automatic proxies, not human causal gold labels. The harmful-repair term was zero throughout the primary cohort, so its coefficient is not empirically validated here.

`graph_features.py` builds pre-action state features and possible source-entity bridges. Bridge confidence is heuristic and dependencies may be model-declared. Edges represent possible information flow, not proven causal relations. `chc/models.py` joins measured labels only after state features are built, preventing outcome labels from entering model input.

`boosting.py` searches 18 histogram-gradient-boosting configurations: leaves 3/7/15, minimum leaf size 3/10, L2 1/5/20, 80 iterations; selection uses question-balanced validation MSE. `learning.py` trains policies, masks missing targets, freezes models and evaluates held-out actions. The final-only matched control uses identical measurable candidates; the full control additionally uses all available final-answer targets.

The GNN uses 14 input features, width 32, two rounds of incoming/outgoing mean aggregation, AdamW learning rate .002, weight decay .01, 25 epochs, batch size 16 and gradient accumulation 2 with AMP. Frozen mean-pooled NLI encoder embeddings feed Ridge regression with alpha selected from .1/1/10/100. Transformer weights were not finetuned.

`web.py` lazily loads models for a live question with supplied evidence. It leaves reference correctness unknown. Recorded examples expose already measured references and outcomes explicitly. The primary tabular checkpoint is selected by the predefined method, not by browsing test wins.

## Data and executed protocol

HotpotQA supplies questions and annotated supporting documents. This is an oracle-evidence condition, not a full open-web retrieval evaluation. Train rows 200:400, tuning rows 400:440, backend development 440:460, debug 460:480, and official validation rows 510:570 for final testing are separate bounded slices. Previous V1 final evaluation used official validation rows 10:510. Question IDs are checked for overlap; repeated entities, titles and templates can still occur.

One continuation seed (101) was used. Training seeds 42, 43 and 44 were used for stochastic tabular/GNN comparisons; deterministic semantic Ridge fits are not three independent stochastic repetitions. There are 31 saved checkpoints: 21 tabular, 6 graph and 4 semantic/control fits. This does not mean 31 independent architectures.

The primary training targets contain 14 positive, 99 neutral and 6 negative actions; three targets are missing. This passes the original count gate of at least 50 training questions and 20 nonzero actions, but only just. Validation has 0 positive, 22 neutral and 2 negative measurable actions. The lack of positive validation effects severely limits model selection.

The original collector crashed on a still-true entity replacement/duplicate graph. A pre-outcome validity fix was applied while all existing episodes were training-only; the audit preserves the earlier source and protocol hashes. A later graph orchestration logger error was fixed without changing the fitting recipe; the saved checkpoint and migration audit were retained. Model weights were frozen before final test collection completed. Test records were excluded before factual-unit processing for fitting; the shared JSONL file itself was read and filtered, so this is not a claim that test bytes were never read.

## Measured containment results

The following table uses one verification-action budget and 16 held-out questions. Intervals are 2,000-resample question bootstrap descriptions. Seed averages do not increase the independent question count. Factual denominators can change when outputs become insufficiently evidenced; coverage must be considered with the rate.

| method | accuracy | mean_utility | utility_ci_low | utility_ci_high | downstream_reduction | mean_calls | evaluable_unit_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_richer | 0.1250 | -0.0208 | -0.0729 | 0.0208 | -0.0417 | 1.0000 | 0.8021 |
| no_intervention | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7812 |
| original_tabular_retrained | 0.1250 | 0.0000 | -0.0469 | 0.0469 | 0.0000 | 1.0000 | 0.8750 |
| random | 0.1250 | -0.0104 | -0.0312 | 0.0000 | -0.0208 | 1.0000 | 0.7604 |
| semantic_richer | 0.1250 | -0.0312 | -0.1094 | 0.0312 | -0.0625 | 1.0000 | 0.8438 |
| tabular_final_only_all | 0.1250 | 0.0000 | -0.0469 | 0.0469 | 0.0000 | 1.0000 | 0.8750 |
| tabular_final_only_full_all | 0.1250 | 0.0000 | -0.0469 | 0.0469 | 0.0000 | 1.0000 | 0.8750 |
| tabular_richer_all | 0.1250 | 0.0167 | 0.0000 | 0.0500 | 0.0333 | 1.0000 | 0.8125 |
| tabular_richer_all_cost_aware | 0.1250 | 0.0156 | 0.0000 | 0.0469 | 0.0312 | 0.6875 | 0.8438 |
| tabular_richer_all_without_semantic_gate | 0.1250 | 0.0000 | -0.0833 | 0.0667 | 0.0000 | 1.0000 | 0.8125 |
| tabular_richer_without_graph | 0.1250 | -0.0333 | -0.1167 | 0.0333 | -0.0667 | 1.0000 | 0.7812 |
| tabular_richer_without_risk | 0.1250 | 0.0000 | -0.0500 | 0.0500 | 0.0000 | 1.0000 | 0.7812 |
| tabular_richer_without_source_bridges | 0.1250 | -0.0333 | -0.1167 | 0.0333 | -0.0667 | 1.0000 | 0.8125 |
| uncertainty | 0.1250 | -0.0333 | -0.1167 | 0.0333 | -0.0667 | 1.0000 | 0.7500 |

The richer tabular policy's one-action mean utility is +0.0167, interval [0, .05], versus 0 for the matched and full final-only tabular controls. Its downstream reduction is small and judged automatically. The lower interval endpoint at zero, only 16 questions, multiple comparisons, no positive validation targets, and shared NLI generation of features/targets prevent a confirmatory superiority claim. Paired policy contrasts are provided in `pilot/evaluation/paired_comparisons.csv` and prediction-level outputs accompany all summaries.

Graph richer utility is −0.0208 and semantic richer utility is −0.0313 at the same budget. Added complexity is not justified by this run. Removing graph/source-bridge features worsens the tabular point estimate, but those ablations do not isolate a generalizable graph benefit on this sample. At budget two the gated policies still show no answer-accuracy gain; the gate-off control reaches 18.75% (3/16). That is one extra correct answer on a tiny sample, not a deployable recommendation to disable safeguards. It suggests the gate can block a useful correction and needs a separate reviewed study.

Across all 90 primary episodes and both singleton actions, richer utility has 15 positive, 149 neutral and 10 negative targets, with 6 undefined; final-only utility has 9 positive, 170 neutral and 1 negative targets. There are 74 accepted corrections and 106 rejected/unchanged actions. In 141 actions the final answer is unchanged. This is why measuring only final exact match can hide consequential differences. Negative proxy effects do not automatically prove human-verified harm.

The harmful-repair proxy rate is zero in the primary data. This does not imply all accepted repairs are safe: prefiltering, a shared automatic judge and absent human review constrain that result. A separate uncensored debug run records a provisional negative propagation case while final-answer EM stays unchanged; its entity-binding interpretation requires review and is not a primary held-out success claim.

The all-input answer analysis scores every eligible primary input independently of trace parse validity and reports missing answers as operational failures. It does not restore the 44 originally requested test questions rejected upstream. Report `16/60` test eligibility alongside conditional accuracy.

## Detector evaluation

The preserved original DistilBERT detector was recalibrated, not finetuned. On the same natural 1,024-row RAGTruth test sample (94 positives, 397 source clusters), original fixed-threshold F1 is .2591; validation-selected full-128-token Platt calibration gives precision .2763, recall .4468, F1 .3415, PR average precision .3686, Brier .07595 and ECE .04866. The validation-selected threshold is .24.

Paired source-cluster bootstrap F1 difference is +.08235, 95% descriptive interval [.02474, .13440]. This is the clearest measured improvement, although it is calibration on one dataset rather than proof of better propagation containment. On the identical historical 256-row sample, F1 changes .1263 → .1633 and ECE .3846 → .07994. Comparing historical .1263 directly with new .3415 would mix cohorts and is invalid.

Aligned-128, aligned-384 and NLI alternatives were evaluated too; test ranking was not used to replace the validation-selected variant. NLI binary detection is weak on RAGTruth, which raises judge-validity concerns. Calibrated RAGTruth probabilities are not assumed calibrated on HotpotQA; live/domain-shift outputs identify that limitation.

## Novelty and related work

The defensible candidate contribution is a reproducible intervention-target construction and evaluation protocol: learning signed changes in downstream factual errors under a fixed continuation boundary, retaining neutral/harmful effects, distinguishing missing evidence, and comparing richer targets with matched final-only targets at identical budgets. It addresses the limitation that a final answer may stay unchanged while the factual continuation improves or deteriorates.

This mechanism is implemented and executed. Broad novelty remains unestablished. Step attribution, evidence graphs, budgeted verification, recoverability and error propagation already have close precedents. No claim of being the first causal hallucination system, first graph verifier or first automatic repair method is supported. Cross-language consistency checking is not implemented in this V2 release.

- [HotpotQA](https://aclanthology.org/D18-1259/) provides multi-hop QA and supporting evidence.
- [RAGTruth](https://aclanthology.org/2024.acl-long.585/) provides hallucination annotations for grounded generation.
- [AgentHallu](https://arxiv.org/abs/2601.06818) addresses step-level hallucination attribution.
- [AgentPropBench](https://arxiv.org/abs/2604.16706) studies agent error propagation and mitigation.
- [SEVRA](https://arxiv.org/abs/2606.19808) is close prior work on verification/recoverability under budgets.
- [FlowScore/SRM](https://github.com/soncheinbok/FlowScore) is relevant graph/process verification work; its faithful baseline has not been executed here.
- [CausalGaze](https://aclanthology.org/2026.findings-acl.1943/) studies internal counterfactual graphs.
- [ZeroFEC](https://aclanthology.org/2023.acl-long.311/) is prior factual correction work.

The related-work comparison document gives a more detailed positioning matrix. A stronger paper requires independently reviewed unit/error/repair labels, more held-out questions and continuation seeds, faithful close-prior baselines, alternative judges, and a preregistered primary comparison. These are remaining work, not completed experiments.

## Hardware, runtime and reproducibility

Executed on a single RTX 4050 Laptop GPU, about 6 GB VRAM, Python 3.12.10, PyTorch 2.11.0+cu128 and Transformers 5.19.0. Qwen FP16 plus NLI fits without quantization; 4-bit loading was not implemented or needed. Recorded generation allocator peak is 3,798,550,528 bytes (about 3.54 GiB). No real CUDA OOM fallback occurred; fallback behavior is covered by a simulated software test.

The resumed collection invocation records 3,563.56 seconds, 304 new generations and 206 cache hits. It excludes the earlier interrupted invocation and is not total experiment runtime. Do not infer a whole-run generation count by adding incomplete log counters. Cached verification timings and logical token counts exclude upfront detector/embedding work, model loading and many shared costs; they are not a full serving efficiency or API billing benchmark.

Qwen snapshot: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. NLI snapshot: `a150876415327c80daeff35ca6f68f5ed8cf5c24`. Model cards: [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [NLI](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall). Exact environment, protocol, hashes, migration logs, raw generations and measured outcomes are saved separately. Frozen external weights are cached locally, not bundled in the source archive.

## Installation and execution

Run these commands in PowerShell from the existing project directory. Use the appropriate CUDA-enabled PyTorch build for your machine. The delivered machine uses a private dependency directory, not a global package upgrade.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements_chc_v2.txt
# On this machine, reuse existing external model cache:
$env:CHC_V2_CACHE='C:/Users/BroCode/Documents/Codex/2026-10-22/sea/work/chc_v2_cache'
$env:CHC_V2_RESULTS='outputs/chc_v2/pilot'
python scripts/run_chc_v2.py --cache $env:CHC_V2_CACHE --hotpot data/chc_v2 --output outputs/chc_v2/reproduction --stage pilot --seeds 101 --detector-report outputs/chc_v2/detector/detector_report.json
python scripts/train_chc_v2.py --cache $env:CHC_V2_CACHE --stage outputs/chc_v2/reproduction --judge-device cpu
python scripts/finalize_chc_v2.py --cache $env:CHC_V2_CACHE --stage outputs/chc_v2/reproduction --historical C:/Users/BroCode/Documents/Codex/2026-10-22/sea/work/chc_run/pilot/models
python -m pytest tests_chc tests/test_model.py tests/test_integration.py
python app.py
```

Open `/chc/propagate` on the address printed by the app. The old root verifier and `/chc` remain available. The portable HTML demo contains ten held-out cases selected in stored order, never selected for favorable outcomes; it cannot run new inference. External weights need downloading on a new machine. The source/results archive is not a self-contained offline distribution of all historical or external weights.

## Verification and limitations

The full release test run passed 109 tests in 71.55 seconds. Tests cover schemas, leakage boundaries, signed utility, unknown evidence, source vetoes, missing targets, atomic rules, reviewer import and software fallback paths. They are not evidence of model factual accuracy.

Actual HTTP handlers were exercised using in-memory request/response streams; this is a transport-contract check, not a TCP or browser test. A separate live service test records its actual result and latency in `live_service_test.json`; supplied evidence is unauthenticated and no gold correctness is invented. Browser visual QA is unverified because localhost access failed and automatic browser approval review rejected file-protocol navigation. No alternate browser workaround was attempted.

The real live Chandrayaan-3 request completed in 134.54 seconds including cold resource loading and returned `evaluation_available=false` and null reference correctness. However, both before and after answers were merely “Yes” to a date/organization question, and the generator invented different launch dates. The NLI judge marked launch-date statements as contradicted by a passage that gives the landing date, which is not a logically adequate contradiction. This is a concrete semantic/judge failure, not a successful factual correction. The supplied passage had no external URL and is explicitly unauthenticated. No source URL was fabricated. Live inference is operational, but reliable open-question answer repair is not established.

Human review has not been performed or fabricated. Blank review sheets and machine-label keys are separate. Other limits include low generator eligibility, small filtered test size, one generation seed, imperfect atomic splitting, proxy uncertainty, possible judge circularity, heuristic graph edges, oracle supporting documents, missing faithful close-prior baselines, no multilingual evaluation, no neural finetuning and no confirmed end-to-end cost advantage.

## Files and reading order

Read this report first, then `docs/ARCHITECTURE.md`, `docs/DATASET_METHODOLOGY.md`, `docs/MODEL_TRAINING.md`, `docs/COMPLETE_CODE_EXPLANATION.md` and `docs/VIVA_GUIDE.md`. Raw/primary episode JSONL files, exact policy predictions, detector predictions, paired contrasts, fitted calibration coefficients and model freeze manifests make the summary auditable. Data licensing/provenance notices accompany the package.
