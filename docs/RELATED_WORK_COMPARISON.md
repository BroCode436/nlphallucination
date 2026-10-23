# Closest related work checked for V2

This is a focused comparison of primary sources, not an exhaustive novelty clearance. Published and preprint findings are attributed to their authors; none are claimed as this project's results.

| Source | Relevant overlap | Difference and remaining comparison |
|---|---|---|
| [SEVRA: Think Again or Think Longer?](https://arxiv.org/abs/2606.19808) | Outcome/recoverability-informed selective verification under cost constraints. | Budgeted outcome-trained checking is already occupied. A faithful strong-gate comparison remains necessary. |
| [AgentProp-Bench, revised August 2026](https://arxiv.org/abs/2604.16706) | Audits judge reliability, injected-error propagation and runtime mitigation in tool agents. | CHC uses external factual working notes and an action-selection loss. It cannot claim the first propagation benchmark; human validation remains a major gap. |
| [AgentHallu](https://arxiv.org/abs/2601.06818) | Human-annotated step attribution and hallucination trajectories. | CHC's automatic proxy annotations do not establish a superior reviewed benchmark. |
| [Structural Reasoning Model / FlowScore official code](https://github.com/soncheinbok/FlowScore) | Structured reasoning DAGs, local step verification and graph-based signal aggregation. | Source-bridge features are ordinary hypotheses. A faithful FlowScore baseline has not been executed here. |
| [CausalGaze, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1943/) | Counterfactual graph intervention for hallucination detection using internal representations. | CHC intervenes on observable external text, with no certified internal causal graph. |
| [Hallucination Across the Reasoning Lifecycle](https://arxiv.org/abs/2610.05472) | Separates observable evidence, causal study design and justified corrective action. | CHC's paired automatic-proxy study still requires independent labels and stronger validation. |
| [More Claims, Less Evidence](https://arxiv.org/abs/2610.05547) | Studies what bounded claim verification implies about artifact reliability. | CHC's logical call budget does not supply a Bayesian reliability guarantee. |
| [MiPS](https://aclanthology.org/2024.findings-emnlp.429/) | Continuation outcomes for intermediate verification supervision. | Downstream outcome supervision itself is established. |
| [ZeroFEC](https://aclanthology.org/2023.acl-long.311/) and [Reference Matters](https://aclanthology.org/2023.acl-long.779/) | Evidence-guided repair and preservation/overcorrection concerns. | The added NLI review gate is an engineering extension, without newly reviewed CAHR training labels. |

The defensible contribution to investigate is the matched factual-unit-loss versus final-only intervention-learning experiment, supported by an auditable benchmark construction and explicit uncertainty. The current work does not establish academic priority, a new GNN architecture, state of the art, natural causal faithfulness, or human-verified containment.
