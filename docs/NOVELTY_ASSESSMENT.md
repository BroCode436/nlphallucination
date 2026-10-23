# Novelty assessment

## Existing methods

Hallucination detection, NLI, evidence alignment, calibration, factual repair, process verification, graph message passing, selective verification and outcome-trained verification allocation all have relevant existing work. Examples include MiPS (https://aclanthology.org/2024.findings-emnlp.429/), TRACK (https://aclanthology.org/2026.eacl-long.273/), selective budget-aware verification SEVRA (https://arxiv.org/abs/2606.19808), ZeroFEC (https://aclanthology.org/2023.acl-long.311/) and Reference Matters (https://aclanthology.org/2023.acl-long.779/). The related-work list is focused, not exhaustive novelty clearance.

## Engineering improvements

Structured-output validation, cached local instruction inference, explicit citation alignment, longer aligned detector input, NLI abstention, validation-only calibration, strict split boundaries, resume support, tests and a clearer UI improve rigor and usability. They are not standalone research novelty claims. LM Format Enforcer (https://github.com/noamgat/lm-format-enforcer) supplies standard format constraints; the local adapter is compatibility engineering.

## Proposed methodological contribution

The research hypothesis is that *measuring observable downstream factual damage* supplies a more informative intervention objective than final-answer correctness alone, permitting better verification allocation at matched cost. The candidate technical mechanism is a source-judged downstream factual-loss target joined to pre-action risk/dependency/semantic features, with harmful repair, missing-label handling and actual joint-intervention controls. A carefully reviewed benchmark/protocol separating hypothetical edges, observed intervention effects and correction validity could be a contribution if prior-work comparison and validation establish a meaningful distinction.

## Required supporting experiment

The central evidence is a matched architecture/objective ablation: richer targets versus final-only targets, identical reasoner, evidence, corrections and budgets; comparison with strong risk/outcome-trained controls; question-cluster intervals; zero-predictor performance; negative effects; and useful transfer. A positive final accuracy score after upgrading the reasoner does not establish the mechanism. Human-validated downstream labels and an independent factual evaluator are especially important because the current NLI checker supplies both risk features and proxy outcome judgments.

## Unsupported claims

The name CHC-Propagate does not prove a novel benchmark. Model-declared dependencies are not certified causal edges. Automatic NLI contradictions are not human gold. Controlled working-note errors do not establish natural hallucination prevalence or access to private chain-of-thought. Missing semantic/graph experiments cannot support architectural superiority. Statistical significance, state of the art, cross-model generalization and universal containment are unsupported unless the actual tables and protocol establish them.

The scientifically valuable outcome may be negative: if the backend ignores corrupted notes, or the judge abstains on most downstream claims, the new measurement can show why final-only learning lacked signal. A more complicated system is not inherently a stronger method.

See RELATED_WORK_COMPARISON.md for newly checked AgentProp-Bench, AgentHallu, SRM/FlowScore, CausalGaze and the October 2026 lifecycle survey. These further narrow any priority claim. Faithful published selective-gate and FlowScore comparisons are unexecuted, and independent human factual labels are absent. A useful implementation and a measured proxy effect can support a research investigation while leaving a novel-method superiority claim unsupported.
