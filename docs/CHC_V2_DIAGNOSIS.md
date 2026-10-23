# CHC V2 initial diagnosis

The prior saved real test outcomes were recomputed before changing experimental code: 3,708 held-out seeded episodes give no-action exact-match accuracy 0.4482200647. This is replay of real saved generations, not a new model run. Previous weights and outputs remain intact on branch `codex/chc-research-v2`.

The old benefit labels contain 2,805 zero, 12 positive and no negative training singleton effects. Mean-square error can reward an almost-always-zero predictor. Larger models or class balancing alone cannot repair this measurement problem. V2 must first collect actual observable continuations with explicit factual claims, preserve all eligible neutral outcomes, and compare against zero benefit.

The old detector has substantial false positives on a small natural RAGTruth sample. Full documents are truncated to 128 tokens; alignment and truncation must be measured separately from model changes. V2 will compare identical sampled questions with sentence-aligned evidence, a longer input, and an external NLI model. Validation-only calibration and threshold tuning must be kept apart from test scoring. NLI neutral means insufficient evidence, not factual contradiction.

The old graph encodes uncertain textual relevance over a few sentences. It does not establish causal links, and the GNN underperformed the tabular model. V2 records model-declared dependencies separately from observed paired intervention effects. No architectural novelty is assumed.

FLAN-T5-small's 24-token answers provide little observable intermediate material. V2 retains it for historical comparison and evaluates a locally feasible instruction backend on development questions before adoption. Longer outputs count only when parsed, structurally validated and evidence checked. Malformed or insufficiently supported starting traces are logged, not silently accepted.

The new primary outcome will explicitly count supported, contradicted, insufficient-evidence and unreviewed downstream claims. An automatic NLI judgment is a proxy requiring human validation, not gold factual truth. Unknown denominators remain missing. A composite loss will be compared with final-answer-only training on identical data and model classes. Experimental complexity is gated on informative pilot targets; full-scale training is not justified when traces or labels fail validation.

The available source files, test definitions and artifact metadata have been inventoried. Model binaries and large data tables are inspected through configurations, hashes, schemas and bounded real rows rather than printed as text. Existing dirty Git changes are preserved. The original 497-question test set is already observed; V2 will reserve a new official-validation slice for final evaluation and separate backend development questions from policy training/tuning.
