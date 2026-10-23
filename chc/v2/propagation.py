"""Transparent propagation denominators and signed downstream intervention loss.

Unsupported claims are not false. Automatic judgments are explicitly distinguished
from human labels. Metrics aggregate by trajectory/question, never by pretending
that dependent claims are independent experimental observations.
"""

from __future__ import annotations
from typing import Any
from ..metrics import exact_match, normalize_answer


def factual_summary(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Count statuses and compute contradicted/(supported+contradicted).

    Returns null rate when no downstream claim can be evaluated. Unknown and
    unreviewed claims are counted separately and excluded from the denominator.
    This is an observed automatic-proxy error rate, not human factual accuracy.
    """
    counts = {
        s: sum(j["status"] == s for j in judgments)
        for s in ("supported", "contradicted", "insufficient_evidence", "unreviewed")
    }
    denominator = counts["supported"] + counts["contradicted"]
    return dict(
        counts,
        downstream_claims=len(judgments),
        evaluable_claims=denominator,
        observed_propagation_rate=counts["contradicted"] / denominator if denominator else None,
        label_origin="automatic_nli_proxy; human_review_pending",
    )


def score_continuation(
    generation: dict[str, Any],
    judgments: list[dict[str, Any]],
    reference: str,
    injection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score only after generation; the evaluator alone receives the reference.

    Invalid generations have missing correctness and propagation scores. Error
    repetition is a conservative exact normalized-text check; it can miss a
    paraphrase. Reported propagation concerns newly generated claims, not the
    injected prefix itself. Graph declarations are kept apart from these scores.
    """
    if generation["trace"] is None:
        return {
            "valid": False,
            "correct": None,
            "factual": None,
            "repeated_original_error": None,
            "generation": generation,
            "judgments": [],
        }
    trace = generation["trace"]
    summary = factual_summary(judgments)
    repeated = bool(
        injection
        and any(
            normalize_answer(c["text"]) == normalize_answer(injection["modified"])
            for c in trace["claims"]
        )
    )
    return {
        "valid": True,
        "correct": exact_match(trace["final_answer"], reference),
        "answer": trace["final_answer"],
        "factual": summary,
        "repeated_original_error": repeated,
        "generation": generation,
        "judgments": judgments,
    }


def observed_loss(
    outcome: dict[str, Any], final_only: bool = False, harmful_repair: bool = False
) -> float | None:
    """Calculate L=.5 downstream contradiction rate + .5 final error + .25 harm.

    Weights are a predeclared equal-weight diagnostic choice, not optimized test
    parameters. Final-only ablation uses 1-final EM. If the required downstream
    denominator is missing, richer loss is unknown rather than assuming zero.
    Harm means a repair of a previously supported prefix becomes contradicted;
    ordinary final-answer deterioration is separately measured at evaluation.
    """
    if not outcome["valid"]:
        return None
    final = 1 - float(outcome["correct"])
    if final_only:
        return final
    rate = outcome["factual"]["observed_propagation_rate"]
    return None if rate is None else 0.5 * rate + 0.5 * final + 0.25 * float(harmful_repair)


def utility(
    before: dict[str, Any],
    after: dict[str, Any],
    final_only: bool = False,
    harmful_repair: bool = False,
) -> float | None:
    """Return signed observed loss reduction; unknown labels remain missing.

    Verification cost is subtracted during selection, not inside supervision, so
    changing lambda does not require relabeling benchmark outcomes. Beneficial,
    neutral and harmful labels follow the actual sign, without forced balancing.
    """
    a = observed_loss(before, final_only)
    b = observed_loss(after, final_only, harmful_repair)
    return None if a is None or b is None else a - b
