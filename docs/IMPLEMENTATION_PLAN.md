# CHC implementation plan

## Initial inspection (23 October 2026)

The existing Git checkout had substantial tracked and untracked changes before this task. They are preserved. The requested branch `feature/causal-hallucination-containment` was created without resetting those changes. An inventory with source hashes is in `outputs/initial_repository_audit.json`. Existing Python, JSON, Markdown, and notebook text was loaded for structural inspection; critical detector, correction, training, application, and evaluation paths were also reviewed directly. This is not a claim that every historical notebook result was reproduced. No `nlp.pdf` was found under the project parent.

Reusable artifacts are the existing DistilBERT weights/tokenizer, TF-IDF weights, numerical helpers, correction-validity artifact, and supplied-source application. There is no live retrieval system or trained causal predictor. Historical results are not evidence for CHC performance. RAGTruth is detector data, not intervention-outcome data.

Detected NVIDIA hardware: RTX 4050 Laptop GPU, approximately 6 GB dedicated VRAM. Query PyTorch availability and record actual memory separately. Do not pool this with another display adapter. Missing pytest and sentencepiece can be installed into a workspace dependency directory. No large language model will be downloaded. FLAN-T5-small is the initial real continuation model (roughly 80 million parameters; ~320 MB FP32 weights and a conservative ~1.8 GB inference estimate).

## Staged implementation

1. Preserve source/configuration/documentation backups and run the original 53 tests with a writable temporary directory.
2. Add a separate typed `chc` package, leaving saved models and datasets intact. Document every new public/internal function, its arguments, outputs, assumptions, and failure modes. Add educational documentation to legacy functions without changing executable AST behavior.
3. Add hardware/resource utilities, strict schemas, NetworkX graph validation, local RAGTruth loading, and HotpotQA integration. Cache an actual bounded HotpotQA subset; never substitute synthetic performance data if retrieval fails.
4. Extract annotated supporting sentences as intermediate facts. Mark their validity as annotation-supported, with human review pending. Infer only uncertain textual graph edges. Keep all benchmark reference answers and outcomes outside policy feature builders.
5. Generate matched continuations using a real small encoder-decoder model. Hold question, evidence, generation settings, and paired seed fixed. Inject controlled errors only into eligible intermediate facts; store proposed corruptions separately when validity cannot be established. Verify by supplying evidence-backed corrections; preserve original facts on rejected corrections. Record ignored injections and harmful interventions.
6. Start with 50 real questions. Store every actual generation, pre-intervention state, intervention action, outcome, token count, latency, and provenance. Continue toward 500 only if the smoke run validates and execution is practical. Cached deterministic outcomes can support repeated policy evaluation, but cache hits are not new generations or stochastic replications.
7. Train a tabular benefit regressor and a small PyTorch message-passing regressor on observed paired outcomes. Train/validation use disjoint questions from official training; held-out evaluation uses official validation. Fit preprocessing on training only. Support AMP, accumulation, checkpoints/resume, and CPU/OOM fallback. Do not train on test outcomes.
8. Compare no-intervention, random, highest predicted error risk, dependency reach, risk-times-reach, learned tabular, and learned graph policies with matched budgets. An outcome oracle is analysis-only. Multiple-action outcomes must be regenerated or reported unavailable; never sum single-action benefits as observed multi-action effects.
9. Measure exact-match final answers, observed final-node factual loss, prediction error, harmful intervention, actual costs, and paired question-cluster intervals. Broader downstream factual counts stay unknown if intermediate generated claims cannot be independently labeled. Produce plots from measured CSV/JSON only.
10. Extend the existing HTTP application with a CHC route and API. Show pre-intervention graph/risk, policy selection, evidence, regeneration, and evaluation only when a reference is explicitly supplied by the user. Preserve the correction interface.
11. Add pytest fixtures and offline tests, then run unit/integration tests and the application. Produce architecture, methodology, data, walkthrough, commands, results, novelty assessment, and a 30-question viva guide.

## Scientific acceptance conditions

- No invented benchmark accuracy or causal effect. A textual dependency edge is not a verified causal edge.
- Interventions are evaluated as paired controlled changes in this particular model/prompt environment; no claim of general real-world causality.
- The policy input excludes reference answers, corruption labels, intervention gains, and post-action outputs.
- Recorded counts distinguish source questions, valid injections, conditions, generated continuations, policy evaluations, and cache reuse.
- Source/question leakage is audited; entity overlap is reported rather than silently dismissed. Hyperparameter selection uses validation only.
- No learning advantage or novelty is claimed unless supported by measured comparisons and appropriate uncertainty.
- Human review is prepared for, but not falsely marked complete by an automated process.

## Feasible fallback

If a model or dataset download fails, retain working loaders and tests and explicitly mark the real experiment incomplete. Deterministic generated fixtures are for tests only. If GPU inference fails, retry bounded CPU inference and log the change. If only the 50-question smoke run completes, do not call it the 500-question pilot.
