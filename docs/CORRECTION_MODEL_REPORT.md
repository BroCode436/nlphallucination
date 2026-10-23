# Implementation and measured results

## Delivered behavior

I built, trained, saved, installed, and tested a local correction-validity model. It evaluates **evidence + original claim + proposed correction**, rather than checking only whether the proposal appears in evidence.

The package includes model weights, training data generation, evaluation scripts, a browser application, a JSON HTTP endpoint, a command-line tool, and regression tests. It requires no paid API or downloaded model for its standalone operation. The existing project's detector remains available, including its previously trained DistilBERT checkpoint; I did not retrain DistilBERT in this work.

This is a working, constrained prototype. The evidence does not justify calling it a production-grade general fact checker or a proven research novelty.

## Architecture

1. Supply an original claim and one or more source passages, optionally with source URLs.
2. Supply a proposed correction, or generate candidate repairs by extracting evidence sentences and attempting number-only substitutions.
3. Normalize Unicode text and derive 18 features jointly from the evidence, original, and candidate.
4. Run two separately trained histogram gradient-boosted classifiers: support and preservation.
5. Require both scores to exceed thresholds selected on validation data.
6. Apply conservative vetoes for missing input, unsupported numerical binding, negation changes, suspicious argument reordering, and dropped conditions.
7. Accept the candidate or retain the original with a review status. Return scores, reasons, evidence excerpts, and supplied source metadata.

The application is claim-oriented. It does not yet perform reliable full-answer atomic claim extraction, live web retrieval, translation, or cross-language question generation.

## What the model learns

The model uses scikit-learn `HistGradientBoostingClassifier` with 100 boosting iterations, at most seven leaves per tree, L2 regularization of 3, and seed 20261023. Separate estimators are trained for evidence support and preservation. The saved artifact contains the two fitted estimators, feature names, version, seed, and thresholds.

Features include original-word recall, candidate precision against the original, preservation of original words that occur in evidence, unsupported candidate fraction, overlap with the most relevant evidence sentence, numeric support and local binding, length ratio, token-sequence similarity, unchanged-text indication, negation change, and evidence availability.

Preservation is not the same as copying the original. An unchanged erroneous number is a support-negative example. An unchanged correct claim is positive. Training also includes valid repairs of unsupported entity and location names so that changing an incorrect name is not categorically forbidden.

The classifiers are genuinely trained models, but their representation is lexical and engineered. They do not implement general semantic entailment, entity resolution, arithmetic, or temporal reasoning. The soft scores are not independently calibrated probabilities of factual correctness.

## Data and splitting

The reproducible generator creates 19,200 synthetic examples:

| Split | Rows | Relation templates | Case families |
|---|---:|---:|---:|
| Train | 9,600 | 6 | 960 |
| Validation | 4,800 | 3 | 480 |
| Test | 4,800 | 3 | 480 |

Each family has ten related examples. Those examples are dependent and should not be treated as independent factual observations. Complete relation templates and generated entity/location names are disjoint between splits. Each validation and test split includes English, Hindi, and Hinglish patterns. This is a narrow structural test, not natural multilingual evaluation.

Examples include valid repairs, wrong-entity copies, deletion of correct details, unchanged errors, wrong numerical binding, unsupported added details, correct unchanged claims, legitimate unsupported-entity/location repairs, and original-swap pairs. In original-swap pairs the same evidence and proposal receive a different validity label after the original changes.

Both acceptance thresholds were selected using validation data; both are 0.95. No test labels were used to select these thresholds. Subsequent manually diagnosed failures informed explicit runtime review checks; those diagnostic sets must therefore be described as development data.

## Evaluation results

| Evaluation | Result | Interpretation |
|---|---|---|
| Generated test set, 4,800 examples | Precision/recall/F1 1.00 | Easy synthetic structural task; not general factual accuracy |
| Earlier 1,440-case diagnostic study | All 480 valid proposals accepted; all 960 invalid proposals rejected | Previously examined synthetic set, not a new independent benchmark |
| Initial 20 author-written challenges | 17/20 correct | Exposed role reversal, paraphrase, and legitimate entity-repair weaknesses |
| Refined 20-case development suite | 19/20 correct | Used during development; remaining failure is paraphrase |
| First run on 12 frozen author-written cases | 9/12 correct | Failed a paraphrase, a conditional statement, and numeric argument reversal |
| Same 12 cases after explicit review checks | 11/12 correct | Development regression result, not independent holdout accuracy |
| Automated project/model/integration tests | 53 passed | Functional regression coverage, not factuality accuracy |

The final runtime was checked directly on all 1,440 earlier study cases in `release_validation.json`. It accepted 240 minimal repairs and 240 reordered valid repairs, while rejecting all 240 examples in each invalid class: wrong-entity copy, dropped detail, unchanged error, and wrong-value binding.

The first frozen 12-case run averaged approximately 14.5 ms per assessment on this local machine, excluding startup and model loading. This is a tiny diagnostic sample, not a production latency benchmark.

I also ran a simple feature-rule baseline and trained the same classifiers without original-swap examples. **Both tied the full model at F1 1.00 on the generated test set.** Therefore this experiment does not establish a learning advantage, a benefit from original-swap augmentation, or a research novelty claim. Natural data and harder comparisons are necessary.

The initial and refined results are retained separately. Labels for all manual examples were written by me and were not independently reviewed. No RAGTruth-wide performance or multilingual benchmark score is claimed.

## Changes to the existing pipeline

`src/correction.py` now loads the new saved verifier when its model artifact is installed. For an edited proposal it passes the evidence, original, and proposed claim to the validity model. A rejected validity result forces rollback and is recorded in the returned result under `correction_validity`.

I removed the averaging loophole in post-verification. Previously a primary hallucination score of 0.9 and a lexical missing-content score of 0 could average to 0.45 and pass a 0.5 threshold. Now either verifier can independently reject. Numerical checks remain independent vetoes.

The original correction module is preserved in `integration_backup/correction.py`. `src/correction_integrated.py` is a snapshot of the installed module for review. The standalone package does not need to import that snapshot.

## Code explanation

The central acceptance decision in `src/correction_validity.py` is conceptually:

```python
support = support_model.predict_proba(features)[0, 1]
preservation = preservation_model.predict_proba(features)[0, 1]
accepted = support >= support_threshold and preservation >= preservation_threshold
```

Runtime vetoes can still turn this into a review decision. The system never averages the scores to compensate for a failed check. `final_claim` is the proposed correction only if accepted; otherwise it is the original text.

`numbers()` normalizes Devanagari digits and preserves signs and decimal values. This prevents treating -5 as 5 or confusing 1.5 with 1.6. Month names are mapped to numerical values for simple date comparison. This is not a complete date or quantity parser.

`same_clause_order()` permits certain conjunction reorders while flagging argument reversals such as “Alice gave Bob 10 books” becoming “Bob gave Alice 10 books.” The condition guard flags source restrictions such as “only if” disappearing from the candidate. These are conservative surface checks; they are not formal semantic proofs and may reject valid rewrites.

`propose()` enumerates evidence sentences and number-substituted original claims, assesses them, and selects an accepted proposal. It can leave an already correct original unchanged. If none passes, it returns `needs_review`. It is an extractive proposal generator, not a generative language model.

## Application and provenance

Run `python app.py --port 8765` and open http://127.0.0.1:8765. The UI accepts source text, an optional URL, the original, and an optional candidate. It displays the final claim and detailed decisions. POST `/api/verify` provides the same functionality for other software.

Source URLs are restricted to HTTP(S), displayed as supplied references, and never fetched or authenticated. The app does not pretend to have searched the internet. Evidence excerpts are selected from supplied text. A URL does not establish that the excerpt actually appears at that page. Source retrieval and citation authentication remain separate future components.

The HTTP page and API were exercised against a running server. The API repaired the example date from August 24 to August 23 while retaining the mission. `evaluation/http_smoke.json` stores the response. Windows sandbox networking initially blocked localhost; after permission and server restart the HTTP checks passed. Proxy handling was disabled for the localhost test.

The initial original-suite run also encountered temporary-directory cleanup permission errors. Rerunning with TEMP/TMP inside the writable workspace succeeded. The final combined run passed all 53 tests; the original failed log is preserved.

## Remaining work

The main limitation is semantic generalization. A natural paraphrase such as “The company has 75 employees” versus “The firm employs 75 people” can be rejected despite being correct. More complex scope, entity binding, comparisons, pronouns, evidence conflicts, arithmetic, and multi-sentence reasoning remain unreliable. Passing these checks is not proof of truth.

Hindi/Hinglish demonstrations establish basic Unicode and lexical operation only. Cross-language consistency verification has not been implemented or validated. Automatic evidence retrieval, source freshness checks, reliable atomic extraction, a multilingual semantic encoder, real annotated correction pairs, and independent human evaluation are the next substantial additions.

The strongest research next step is to evaluate correction-validity training on natural source/original/proposal triples and compare against a semantic judge, overlap-aware ranking, and the simple rule baseline. The current synthetic results cannot support a claim of a superior or novel model.
