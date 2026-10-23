"""Independent pre-outcome validity veto for selected factual corruptions.

A source-specific NLI contradiction is insufficient if another available source
supports the modified statement. Entity swaps can remain true. This guard never
reads downstream outcomes and never forces beneficial/harmful label balance.
"""
from __future__ import annotations
from typing import Any
from ..metrics import normalize_answer


def selected_corruption_validity(injection: dict[str,Any], clean_prefix: list[dict[str,Any]],
                                 evidence: dict[str,dict[str,str]], judge: Any) -> dict[str,Any]:
    """Veto mutations that duplicate known support or have cross-source support.

    Args:
        injection: Previously selected candidate, before continuation generation.
        clean_prefix: Initial automatically supported claims, not future outputs.
        evidence: All available verifier passages, including alternative entities.
        judge: Frozen NLI model; a high entailment score conservatively vetoes
            an asserted contradiction. No human-gold validity is claimed.
    Returns:
        Eligibility flag, exact matching prefix IDs, per-source entailment scores
        and a reason. A rejected selection is logged; no favorable alternative
        is substituted after observing an outcome.
    """
    duplicates=[c['id'] for c in clean_prefix if c['id']!=injection['node_id'] and normalize_answer(c['text'])==normalize_answer(injection['modified'])]
    scores=judge.probabilities([(injection['modified'],e['text']) for e in evidence.values()])
    support={key:p['entailment'] for key,p in zip(evidence,scores)}
    entailing=[key for key,value in support.items() if value>=judge.threshold]
    reasons=[]
    if duplicates:reasons.append('Mutation duplicates another supported clean assertion')
    if entailing:reasons.append('An available source supports the mutation; contradiction not established')
    return {'eligible':not reasons,'reasons':reasons,'duplicate_supported_claim_ids':duplicates,
        'cross_source_entailment':support,'entailing_source_ids':entailing,
        'origin':'Pre-outcome automatic validity veto; human review pending'}
