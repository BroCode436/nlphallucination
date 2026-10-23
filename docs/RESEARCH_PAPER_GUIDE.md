# Research paper guide

Use a hypothesis-driven title such as “Does Observable Factual Propagation Improve Budgeted Verification?” until results justify a stronger claim. Introduce the sparse final-answer supervision problem, compare with selective verification/process supervision literature, and state exactly how controlled external-note interventions differ from hidden model reasoning.

Describe the data construction with source licenses, bounded sampling, eligibility/rejection counts, separate natural detector and controlled challenge distributions, automatic/human status and all seeds. Show the raw-output and citation-aligned artifact yields separately. Include negative and neutral cases, not only favorable screenshots.

Define the four factual statuses and OHPR denominator before presenting results. Explain missing-label handling, observed factual-loss weights, cost penalty and uncertainty. Present no-intervention/risk/random/zero controls before learned models. Make richer-versus-final-only training the central table. Report harm, coverage, actual calls, token estimates and runtime alongside final accuracy.

Never compare V1's FLAN accuracy with V2's Qwen accuracy as proof of the new intervention objective. The model, evidence format and selected cohort change. Use matched within-V2 baselines to isolate the mechanism. Direct V1 transfer is a distribution-shift diagnostic, not an architectural fairness claim.

Report means and paired question-cluster intervals. Repeated conditions, generation seeds and training seeds are dependent within a question. Intervals including zero do not support a reliable improvement claim. If no justified hypothesis test was run, do not insert a p-value. Correct for multiple comparisons if making formal confirmatory claims across many policies.

Limitations should explicitly cover judge circularity, unreviewed atomicity and graph links, narrow automatic error families, prefix-source filtering, exact-match aliases, small eligible cohorts, no natural trace generalization, and compute limits. The release should include protocol, model snapshot IDs, raw outputs, rejected examples, blank review forms, checkpoints, scripts and tests. A human-reviewed benchmark, independent judge and a preregistered larger test are the next steps before submission.
