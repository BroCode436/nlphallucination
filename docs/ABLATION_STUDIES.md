# Ablation design and current execution status

The central experiment is tabular richer-loss targets versus tabular final-answer-only targets, with identical features, real episodes, correction mechanism and budgets. The corresponding GNN and semantic objective comparison runs only after the signal gate passes. This tests whether measuring downstream factual damage changes intervention selection, rather than crediting a larger backend for architectural novelty.

The tabular without-risk model is retrained excluding risk/uncertainty features. Without-graph is retrained excluding structural columns. A tabular model also acts as the no-semantic baseline against frozen-embedding regression. These are standard controls, not proof that feature groups contribute.

No-outcome-training uses zero-benefit and training-frequency mean controls plus risk/reach/random heuristics. Direct transfer of original V1 checkpoints is labeled transfer across a changed feature/risk distribution, not a matched V2 retrained baseline. V2 final-only models are retrained on the same backend and corrections as the richer method.

The semantic validity gate ablation uses the exact same lexical candidate proposal and ranking, changing only the extra acceptance filter. The original lexical verifier itself remains active, so this is not a complete removal of all correction-validity safeguards. Its singleton and budget-two joint continuations are generated separately when their prefixes differ. It must not be mislabeled a full lexical-validity-off ablation.

Cost-aware selection subtracts a fixed .01 expected loss per call and can stop. Quota variants use their allowed call count. Report actual mean calls and tokens; do not claim identical cost when a stopping policy performs fewer actions. Budget-two outcomes come from actual joint continuation, not a sum of singleton benefits.

Execution is established only by the generated checkpoint files and measured evaluation tables. If a model or ablation is absent because the pilot fails the signal gate, the results report must call it unexecuted rather than infer its performance.

Matched-mask final-only supervision and full-coverage final-only supervision are separate controls. The declaration-only source-bridge ablation holds feature columns/model class fixed while removing the added source links. Semantic without-embedding comparison uses the same Ridge class and validation alpha grid; it does not attribute a regressor-class change to embeddings. The original regressor is also retrained on V2 final-only labels.

The all-input answer robustness table uses the same selectors and actual recorded outcomes but includes every eligible test input, with answer availability and trace validity shown separately. Propagation-rate comparisons retain unknown labels and per-unit coverage. Extra detector/semantic feature costs are not included in a logical verification-call budget, so a reduction in selected actions is not a demonstrated reduction in total serving computation.
