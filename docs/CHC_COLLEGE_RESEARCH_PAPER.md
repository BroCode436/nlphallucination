# Learning Verification Interventions from Downstream Factual Error Propagation

Author name, department, institution and supervisor to be supplied by the student. This manuscript reports an executed local study; it has not been peer reviewed or accepted for publication.


## Abstract

Verification systems often measure success through final-answer correctness, although an intervention can change intermediate factual errors without changing the final answer. We implement Causal Hallucination Containment V2, a local experimental framework that learns the signed change in downstream evidence-grounded factual error after a verification action. It preserves neutral and negative outcomes, distinguishes insufficient evidence from contradiction, and compares tabular, graph and semantic policies under matched action budgets. A 300-question HotpotQA collection yields 90 eligible episodes after automatic source and factual-unit audits, with 61 training, 13 validation and 16 held-out test questions. At one action, a richer tabular policy achieves mean proxy utility .0167 with a descriptive bootstrap interval [0, .05], while final-answer exact match remains 12.5% for all tested policies. Graph and semantic models show no advantage. Separately, validation-only calibration improves a preserved RAGTruth detector from F1 .2591 to .3415 on the same 1,024 test examples. The results demonstrate an executable measurement protocol and a limited tabular signal, rather than reliable general containment superiority. Low eligibility, automatic-judge errors and the small test cohort remain central limitations.

Keywords: hallucination verification; factual error propagation; intervention utility; evidence grounding; reproducible evaluation.


## 1 Introduction

A corrected reasoning statement can prevent an error from spreading, leave the continuation unchanged, or trigger another error. A binary final-answer improvement target cannot distinguish these cases when the final answer remains the same. This motivates learning the observed downstream consequence of a verification action rather than assuming that a high-risk or highly connected statement is always the best intervention target.

Our research question is whether signed downstream factual-error targets improve verification-action selection over final-answer-only targets at an identical budget. The implemented contribution is a controlled intervention dataset construction protocol with a fixed scoring boundary, explicit unknown-evidence handling, paired policy comparisons and retained harmful or neutral outcomes. It is evaluated using real local model continuations. We do not claim to introduce graph verification or causal analysis of language models as a field, and model-declared dependencies are not treated as proven causal relations.


## 2 Related Work and Research Position

HotpotQA provides multi-hop questions and supporting documents [1], while RAGTruth supplies hallucination annotations for grounded generation [2]. These resources support different parts of this study: controlled continuation experiments and detector calibration respectively. Their metrics and populations are reported separately.

AgentHallu studies hallucination attribution within agent execution [3]; AgentPropBench investigates propagation and mitigation [4]. SEVRA is relevant to budgeted verification and recoverability [5], while FlowScore/SRM provides graph or process verification machinery [6]. These precedents rule out broad first-system claims. Our candidate contribution is narrower: constructing signed factual propagation targets, measuring them under a fixed reference boundary, and comparing them with matched final-answer targets. A faithful implementation of the closest prior methods has not been evaluated, so field-level methodological novelty and superiority remain unestablished.


## 3 Method

The pipeline receives a question and its supporting passages. A frozen Qwen2.5-1.5B-Instruct model produces short JSON claims, dependency IDs, citations and a final answer [7]. Structural validation requires unique IDs, earlier-only dependencies and available evidence identifiers. Explicit retrieval may fill missing citation metadata without changing generated text. Evidence attachment accepts contradictory or insufficient passages too; otherwise false continuations would be censored before scoring.

A frozen DeBERTa NLI cross-encoder compares claims with evidence [8]. Conservative sentence and shared-subject splitting yields factual units, with unresolved compound statements flagged. Each unit receives supported, contradicted, insufficient-evidence or unreviewed status. These are fallible automatic labels, not human gold annotations. The observed error rate is the number of contradicted units divided by supported plus contradicted units. It is undefined when that denominator is zero; unknown evidence is not counted as false.

The collector first obtains a supported clean prefix. It replaces one prefix statement with a controlled source-contradicted mutation, then generates actual clean, corrupted, corrected and joint-action continuations. A cross-source veto rejects replacements that remain supported elsewhere or duplicate a known supported fact. This eligibility decision occurs without using intervention gains. The scoring boundary is fixed before an action so a correction cannot improve its score by moving the boundary.

Loss combines .5 times observed downstream error, .5 times final-answer error, and .25 times a harmful-repair proxy. Utility equals loss before minus loss after. Positive, zero and negative values represent measured improvement, no change and damage. Missing factual rates produce missing richer targets. A semantic repair gate requires evidence support and preservation checks; a gate-off condition reuses the same candidate for an ablation. Joint outcomes are generated directly rather than approximated by adding singleton effects.


## 4 Experimental Design

The main study requests 200 training, 40 tuning and 60 final test questions, using one continuation seed, 101. Bounded HotpotQA train rows 200–399 supply training, 400–439 tuning, 440–459 backend development and 460–479 debug questions. Official validation rows 510–569 supply final test inputs, separate from the previous project evaluation. Question identifiers are audited for overlap. These bounded slices are not representative random samples, and entity or template overlap remains possible.

The evidence condition uses annotated supporting documents. This removes much open-web retrieval uncertainty, making the experiment an oracle-evidence controlled challenge rather than an unrestricted factual assistant benchmark. Mutations are attempted only when their prerequisites exist. Source-attribution, unsupported-appending and other uncertain transformations are retained as review-only rather than silently accepted as validated factual corruptions.

Of 300 attempted inputs, 97 initially yield eligible records. Factual-unit and source-validity audits retain 90 primary episodes: 61 training, 13 tuning and 16 test. Test eligibility is therefore 16/60, or 26.7%; collection-wide eligibility is 30%. All 16 primary test episodes have complete action outcomes. Completeness within this selected cohort does not remove upstream selection bias.

The richer training labels contain 14 positive, 99 neutral and 6 negative actions, with three missing targets. The original complexity count gate of 50 training questions and 20 nonzero actions is passed exactly at 20 nonzero labels. Tuning contains no positive richer effects, 22 neutral and two negative effects. This limits the reliability of hyperparameter and stopping selection.

Policies include random, highest-risk, dependency-based and historical-feature baselines; histogram gradient boosting; a two-round width-32 GNN; and Ridge regression over frozen semantic embeddings and state features. Matched final-only controls use identical measurable candidates. A stronger full final-only tabular control uses all available final-answer labels. Ablations remove risk, graph, source bridges, semantic embeddings or repair gating. Budgets are zero, one or two actions; cost-aware variants can stop when expected benefit is insufficient.

Tabular and graph fits use seeds 42, 43 and 44; deterministic Ridge fits are not counted as independent training repetitions. The release contains 31 checkpoints. Frozen transformers are not finetuned. Models are frozen before final test collection completes, and test records are excluded from fitting. Descriptive uncertainty uses 2,000 question bootstrap samples for policy outcomes and source-cluster samples for detector comparisons. Multiple comparisons are exploratory and not multiplicity-adjusted confirmatory tests.

A single RTX 4050 Laptop GPU with approximately 6 GB VRAM executes Qwen in FP16. The recorded generation allocator peak is approximately 3.54 GiB. Training, generation and cached verification timings have different scopes; no whole-system cost advantage is inferred from cached timings.


## 5 Results

Table 1 compares one-action policies on the same 16 primary held-out questions. Accuracy is final-answer exact match; utility is the signed automatic factual/answer loss reduction. Values are seed-averaged where applicable; seeds do not increase the independent question count. The no-action utility is zero by definition.

| Policy | Accuracy | Mean utility | Utility interval |
| --- | --- | --- | --- |
| No intervention | 12.5% | 0 | [0, 0] |
| Highest risk | 12.5% | −.0333 | [−.1167, .0333] |
| Final only tabular matched | 12.5% | 0 | [−.0469, .0469] |
| Final only tabular full | 12.5% | 0 | [−.0469, .0469] |
| Richer tabular | 12.5% | +.0167 | [0, .05] |
| Richer graph | 12.5% | −.0208 | [−.0729, .0208] |
| Richer semantic | 12.5% | −.0313 | [−.1094, .0313] |

Table 1  One verification-action budget with descriptive 95% question-bootstrap intervals. Undefined richer outcomes are masked; the paired tabular utility contrast uses 15 questions, not 16.

The richer tabular point estimate exceeds matched and full final-only tabular controls by .0167, with paired interval [0, .05]. Its downstream error-rate reduction is .0333, interval [0, .10]. This effect is small, includes zero at the interval boundary, and depends on automatic evidence judgments. There is no final-answer accuracy improvement: all methods answer two of 16 questions correctly at one action. Graph and semantic complexity is not justified by these results.

Removing graph or source-bridge features worsens the tabular point estimate, but this small exploratory ablation does not establish a general graph contribution. At budget two, gated policies still show no final-answer gain. The gate-off control reaches three correct answers, 18.75%, suggesting that gating can block a helpful change. One extra success does not establish that disabling safeguards is preferable.

Across all 90 primary episodes, singleton richer targets contain 15 positive, 149 neutral, 10 negative and six missing effects. Final-only targets contain nine positive, 170 neutral and one negative effect. The final answer is unchanged for 141 of 180 actions. These different target distributions demonstrate the measurement distinction; they do not alone demonstrate better intervention selection. The harmful-repair proxy is zero throughout the primary data and therefore does not validate the proxy coefficient or imply universal repair safety.

Separately, on the same 1,024 RAGTruth test rows with 94 positives, validation-selected Platt calibration raises preserved detector F1 from .2591 to .3415. Precision is .2763, recall .4468, PR average precision .3686 and ECE .04866. Paired source-cluster bootstrap gives F1 difference .08235 with interval [.02474, .13440]. The matched historical 256-row comparison is .1263 to .1633; comparisons across these sample sizes would be invalid. Calibration is not assumed transferable to HotpotQA.


## 6 Error Analysis and Threats to Validity

A real live request asks when Chandrayaan-3 landed and which organization developed it, using a supplied passage containing those facts. The service completes local inference in 134.54 seconds including cold loading, but both before and after final answers are “Yes.” The model invents different launch dates in its continuations. The judge incorrectly treats a passage about landing as adequate contradiction evidence for a launch date. This exposes both answer-format failure and semantic event confusion. It is not counted as a successful repair, and live reference accuracy remains null.

This example matters because shared NLI features and target labels can make a policy learn the judge’s errors. Human review has not been performed. Reviewer sheets remain blank and machine keys are separate. Without independent annotation and an alternative judge, automatic utility should be interpreted as a provisional proxy. Insufficient evidence, event distinctions and entity-binding errors require particular attention.

Eligibility is the largest external-validity concern. Requiring a supported initial prefix and a validated corruption selects easy-to-ground starting points while discarding many generator failures. The reported 12.5% accuracy is conditional on 16 selected questions and is not the answer accuracy for all 60 requested test inputs. An additional answer analysis ignores downstream trace validity within the primary cohort, but it does not restore upstream rejected inputs.

The experiment uses one small frozen reasoner, one continuation seed, short traces, bounded source slices and annotated evidence. Dependency edges are partly generated by the model; source bridges have heuristic confidence. Conservative splitting cannot guarantee atomic factual units. The loss weights are fixed design choices, and primary data provide no nonzero harmful-repair proxy observations. A two-action budget is too small to establish long-horizon agent behavior or deployment efficiency.

Development corrections were necessary. A pre-outcome source-validity fix was applied while existing records were training-only and is logged with source hashes. An orchestration logger failure was repaired while preserving checkpoint state and history. Raw rejected records, migration audits, full prediction tables and freeze manifests remain available. No records are removed because their intervention utility is zero or negative.


## 7 Conclusion and Future Work

CHC V2 provides a functioning college-level research system for studying the consequences of verification interventions. Its contribution is an implemented, auditable signed-target construction and comparison protocol. The executed study provides a small tabular proxy signal and a clearer calibration improvement, but no held-out answer-accuracy gain and no evidence that graph or semantic models outperform simpler controls. Accordingly, the current result is a preliminary empirical study with informative negative findings, not a demonstrated state-of-the-art containment method.

The next research steps are independent human factual-unit and repair review, a larger preregistered test set, several continuation seeds, faithful close-prior baselines, and alternative evidence judges. A publication claim should follow those experiments. For college submission, this manuscript, the detailed report, code explanations, reproducible outputs and demo provide a coherent project; authorship, institutional details and the college’s required formatting must be supplied before formal submission.


## References

[1] Yang et al. HotpotQA A Dataset for Diverse Explainable Multi hop Question Answering. EMNLP 2018. https://aclanthology.org/D18-1259/

[2] Niu et al. RAGTruth A Hallucination Corpus for Developing Trustworthy Retrieval Augmented Language Models. ACL 2024. https://aclanthology.org/2024.acl-long.585/

[3] AgentHallu. Research preprint, 2026. https://arxiv.org/abs/2601.06818

[4] AgentPropBench. Research preprint, 2026, revised version consulted. https://arxiv.org/abs/2604.16706

[5] SEVRA. Research preprint, 2026. https://arxiv.org/abs/2606.19808

[6] FlowScore and SRM. Official research implementation, 2026. https://github.com/soncheinbok/FlowScore

[7] Qwen2.5 1.5B Instruct. Official model card. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

[8] NLI DeBERTa v3 xsmall. Official cross encoder model card. https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall


## Appendix A Reproducibility and Release Contents

The release includes source, tests, 31 fitted policies, calibration coefficients, raw and audited episode JSONL, policy and detector predictions, bootstrap contrasts, protocol/freeze hashes, environment records and a portable ten-case recorded demo. The report supplies installation and execution commands. External transformer weights and the original processed RAGTruth corpus are not redistributed in the source/results archive. Existing local caches and original checkpoints remain available on the development machine.

Qwen snapshot 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and NLI snapshot a150876415327c80daeff35ca6f68f5ed8cf5c24 identify the frozen backends. Software includes Python 3.12.10, PyTorch 2.11.0+cu128 and Transformers 5.19.0. The resumed collector logs 3,563.56 seconds, excluding an earlier interrupted invocation; this is not total experiment runtime.

All 109 release tests pass. Real HTTP handlers are checked through in-memory streams and a real live service request is recorded separately. Browser visual verification is unavailable because localhost access failed and file URL navigation was rejected by automatic browser approval review. The manuscript does not claim network or browser end-to-end verification. HotpotQA attribution and upstream data/model licensing notices accompany the package.

This manuscript contains no fabricated reviewer labels, publication status, author credentials or favorable experiments. It should be presented as a student research draft based on the measured pilot. The title term “causal” refers to controlled prefix interventions in a fixed experimental setting; it does not imply identification of a model’s internal causal mechanism.
