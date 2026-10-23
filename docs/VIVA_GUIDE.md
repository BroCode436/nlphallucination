# CHC viva guide

## Explain the project in one minute

Our project asks which intermediate factual claim is worth checking when we cannot check every claim. We extract a small set of working notes from real multi-hop question-answering data, introduce a controlled error, and generate an actual final answer. We repeat after verifying selected notes. The difference in final-answer correctness is an observed training target. A small regression model and a graph neural network learn to predict useful checks from information available before intervention. We compare them against random, risk-based, and graph-based selection under the same budget. The graph is a textual hypothesis, and our current factual outcome labels cover the final answer only. We report negative results and do not claim novelty merely because the system runs.

## Questions and answers

1. **What is a hallucination in this project?** An unsupported or conflicting claim relative to supplied evidence. The reference evidence can itself be incomplete or wrong, so this is not universal truth detection.

2. **What does CHC stand for?** Causal Hallucination Containment. It is the project name; it does not certify that the inferred graph is a causal graph.

3. **What is the research question?** Whether observed effects of verification can train a policy that selects more useful intermediate checks than equally budgeted heuristics.

4. **Why not verify everything?** Verification consumes resources and a correction can be harmful. We measure accuracy and cost rather than assuming more checks are always better.

5. **Why is uncertainty insufficient?** A doubtful claim may not influence the final answer. Conversely, a confident but consequential error can be worth checking. This is a hypothesis tested against risk-based selection.

6. **What is HotpotQA used for?** Real multi-hop questions, reference answers, documents, and supporting-fact annotations.

7. **What is RAGTruth used for?** Evaluating the existing hallucination detector. It is not treated as a causal propagation benchmark.

8. **Are the intermediate notes naturally generated chain-of-thought?** No. They are extracted supporting sentences used as externally controlled working notes. That is a significant experimental limitation.

9. **Are all nodes atomic claims?** No. Sentence extraction is a lightweight starting point and can retain multiple propositions in one node.

10. **What does a graph edge mean?** An uncertain textual dependency or annotated relevance relationship. It does not prove that changing the parent changes the child.

11. **Why a directed graph?** It distinguishes a potential upstream fact from its downstream use. Reversing the direction changes reach and message-passing features.

12. **Why reject cycles?** The implemented trace assumes a directed acyclic order of intermediate steps. Cycles violate its interpretation and can invalidate path/depth features.

13. **What is downstream reach?** The number of unique nodes reachable from a claim. It measures potential textual influence, not observed causal benefit.

14. **How is hallucination risk obtained?** From actual inference with the saved DistilBERT evidence/claim classifier, using its raw positive-class softmax score.

15. **Is a score of 0.8 an 80% factual error guarantee?** No. Calibration must be measured on representative data; our HotpotQA scores are explicitly uncalibrated.

16. **What is the benefit label?** Error before checking minus error after checking. With binary final-answer loss it is -1, 0, or +1.

17. **Why keep negative benefit labels?** They represent harmful interventions. Dropping them would bias the model toward believing correction is always helpful.

18. **What can a zero benefit mean?** The injected error was ignored, verification failed or abstained, the corrected fact was irrelevant, or the final answer remained equally correct/incorrect.

19. **Which information is forbidden to the policy?** Reference answers, injection identity/location, measured future gains, and post-intervention outputs.

20. **How is leakage reduced in code?** The feature builder accepts a strict DecisionState that lacks outcome fields. Question splits precede condition/seed expansion, and audits reject shared question IDs.

21. **Does that guarantee no leakage whatsoever?** No. Natural entities/documents can overlap, pretrained models can have seen benchmark material, and code review is still necessary.

22. **What is the feature-based predictor?** A histogram gradient-boosted regressor trained on structured risk, graph, relevance, length, and cost features.

23. **What is the graph predictor?** A small PyTorch network that combines each node's hidden vector with separate mean incoming and outgoing neighbor messages for two rounds.

24. **What are its tensor shapes?** Raw features are [N,F], hidden states [N,H], edges [2,E], and predicted benefits [N]. Final placeholders are excluded by a loss mask.

25. **How are multiple graphs batched?** Their node features are concatenated and edge indices offset. No edges connect unrelated questions.

26. **What is the training loss?** Mean squared error between predicted and observed signed benefit. It is standard regression, not a new loss contribution.

27. **What happens during backpropagation?** The loss computes gradients of model parameters. The optimizer then applies updates; gradient computation itself does not change the weights.

28. **What is gradient accumulation?** Several minibatches contribute gradients before an optimizer update, approximating a larger effective batch without storing all activations at once.

29. **What is automatic mixed precision?** Supported CUDA operations use lower-precision arithmetic, with loss scaling to reduce underflow. It saves resources but does not improve factual reliability by itself.

30. **Why not combine both laptop GPUs?** Different adapters do not automatically pool VRAM. CHC selects one CUDA device and reports actual memory.

31. **Why use a small reasoner?** It makes real repeated interventions feasible on roughly 6 GB VRAM. Its weaker reasoning quality also limits how broadly the findings generalize.

32. **Is FLAN-T5 fine-tuned in this work?** No. It is a frozen continuation model. The tabular and graph intervention predictors are trained.

33. **How does the budget selector work?** It greedily chooses the highest affordable priority, checking call, estimated-token, and optional estimated-latency caps before each selection.

34. **Is greedy selection globally optimal?** No. It is an efficient approximation suitable for the small baseline graphs.

35. **Why not add singleton benefits for multiple checks?** Checks can overlap, interact, or interfere. We use actually generated outcomes for each evaluated subset.

36. **What does the overlap discount do?** It reduces priority when descendants are already covered. It is a transparent heuristic, not an observed causal interaction model.

37. **What is the oracle?** An analysis-only selector using observed held-out subset outcomes. It is not deployable and never supplies live features or test training labels.

38. **What is a matched comparison?** Policies share the same state, detector, evidence, correction mechanism, continuation model, generation settings, and maximum budget.

39. **Why three generation seeds?** They expose stochastic variation. They are grouped within questions for uncertainty rather than counted as unrelated questions.

40. **What is the difference between generation and training seeds?** Generation seeds control sampled continuations; training seeds control initialization and model fitting. Both are recorded separately.

41. **Why use caching?** Exact repeated prompts/settings can reuse an actual saved output. Cache hits are not new generations or independent replications.

42. **What is harmful intervention rate?** The fraction of selected episodes whose final-answer accuracy is worse after the action than with no action. It does not measure unannotated intermediate harms.

43. **What is bootstrap uncertainty?** We repeatedly resample whole questions and compute the paired mean gain, preserving conditions/seeds within each question.

44. **Does an interval prove significance?** Not automatically. The protocol must specify its estimand and valid comparison; we report intervals without inventing a significance test.

45. **What is exact match's weakness?** Correct aliases and paraphrases can be marked wrong. A representative human review should check these cases.

46. **Do we measure every downstream hallucination?** No. Current observed labels cover the final answer; broader downstream factual counts remain unknown.

47. **What is safe correction here?** An evidence-based proposal that passes separate support and preservation checks plus conservative vetoes. Passing is still not independent proof of truth.

48. **Why can correction abstain?** Evidence may be insufficient, numeric alignment ambiguous, or a proposed edit may damage valid information. A rejected correction leaves the original visible with its status.

49. **Which injected error types require review?** Entity, relationship, unsupported-conclusion, and evidence-misattribution candidates are retained as unvalidated auxiliary cases. They are not automatically given false gold labels.

50. **What is the novelty today?** A candidate experimental contribution, not an established new method. Related work already learns verification from continuation outcomes and studies budget-aware recovery. Stronger data, comparisons, and positive or informative negative evidence are required.

51. **What did the smoke experiment show?** It executed actual data collection/training/evaluation, but its small held-out subset did not show a learned-policy accuracy advantage. Exact measured counts and pilot findings are in RESULTS.md.

52. **What should be improved before a paper?** Human-validated natural traces/injections, richer dependency validation, stronger detector calibration, meaningful nonzero intervention outcomes, unseen-model/topic transfer, and measured intermediate factual effects.


## V2 questions

53. **What is the new primary hypothesis?** Downstream factual-loss targets may improve verification allocation beyond final-answer-only supervision at matched cost.
54. **Does valid JSON mean valid reasoning?** No. Schema, evidence support, atomicity and dependencies are distinct checks.
55. **Why preserve raw citation failures?** Otherwise source alignment could inflate claims about instruction following.
56. **Is citation alignment a model-generated citation?** No. It is explicitly recorded automatic metadata assignment to unchanged model text.
57. **Why is neutral NLI not false?** It means the source does not establish the statement; the statement may still be true.
58. **What is OHPR's denominator?** Supported plus contradicted newly generated claims; unknown and unreviewed claims are excluded and counted separately.
59. **What if no claims are evaluable?** The rate and richer target are missing, not zero.
60. **What establishes propagation?** Observed differences between matched real continuations, with fallible labels and separate uncertain dependency annotations.
61. **Why retain zero-gain interventions?** They are part of the actual eligible distribution and reveal whether learning has useful signal.
62. **Can the exact restoration oracle be deployed?** No. It uses the known original clean prefix and is only a sensitivity diagnostic.
63. **What is the most important ablation?** Identical models trained on final-answer-only versus richer downstream loss.
64. **Why gate graph/semantic training?** Complex models cannot compensate for almost no informative outcome targets.
65. **Is the NLI judge independent of the policy?** Not fully: NLI supplies risk and outcome proxies, a judge-circularity limitation requiring independent validation.
66. **Does a new reasoner prove a new intervention method?** No. Only matched within-backend comparisons isolate intervention learning.
67. **Are all ten corruption families validated?** No. Implemented candidates require prerequisites and evidence checks; many still need human review.
68. **Why can entity replacement be true?** The new entity may share the same property; a textual change is not automatically a factual error.
69. **How is test leakage prevented?** Split IDs are fixed beforehand; references enter post-generation scoring; normalization/model selection use train/validation.
70. **What is harmful correction?** A previously correct outcome becoming wrong, or a supported prefix changed into contradiction; both proxies need clear denominators.
71. **Why compare with zero benefit?** Sparse targets can reward a trivial predictor; MAE alone can mislead.
72. **What would justify a paper claim?** Reviewed reliable labels, strong matched baselines, the central objective ablation, useful effect sizes with uncertainty, and generalization.
