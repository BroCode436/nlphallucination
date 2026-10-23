"""Controlled note perturbations with explicit validity categories.

An injected sentence is a counterfactual error candidate, not automatically an
observed natural hallucination. Unsupported transformations are labeled unknown;
only mechanically defensible contradictions enter the primary experiment.
"""
from __future__ import annotations

import re
from typing import Any
from .schema import DecisionState


def candidates(state: DecisionState) -> list[dict[str, Any]]:
    """Construct perturbations at explicitly identified supporting-sentence nodes.

    Args:
        state: Clean extracted annotation-supported trace, without a gold answer.
    Returns:
        Candidate injection dictionaries with location, original, modified text,
        error type, and validity basis. No unchanged candidates are returned.
    Limitations:
        Numeric/date substitutions and explicit negation are primary controlled
        conditions. Entity/relationship/misattribution candidates need review and
        are retained as proposed auxiliary conditions rather than false gold labels.
    """
    result: list[dict[str, Any]] = []
    for node in state.claims:
        if node.is_final:
            continue
        text = node.text
        # A year bounded as an independent token is changed by a small offset,
        # avoiding identifier fragments and implausible 4000-year corruptions.
        year = re.search(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b', text)
        if year:
            changed = text[:year.start()] + str(int(year.group())+3) + text[year.end():]
            result.append({'node_id': node.id, 'type': 'incorrect_date', 'original': text,
                           'modified': changed, 'validity': 'controlled_value_contradiction'})
        number = re.search(r'(?<![\w.])\d+(?!\w|\.\d)', text)
        if number:
            changed = text[:number.start()] + str(int(number.group())+2) + text[number.end():]
            result.append({'node_id': node.id, 'type': 'incorrect_number', 'original': text,
                           'modified': changed, 'validity': 'controlled_value_contradiction'})
        # Restrict negation to explicit copular assertions; negating an arbitrary
        # token or a quoted phrase is not a reliable contradiction transformation.
        copula = re.search(r'\b(is|was|are|were)\b(?!\s+not\b)', text)
        if copula and not re.search(r'\b(if|not|never)\b', text, re.I):
            changed = text[:copula.end()]+' not'+text[copula.end():]
            result.append({'node_id': node.id, 'type': 'negation_change', 'original': text,
                           'modified': changed, 'validity': 'controlled_explicit_negation'})
        title = state.evidence[node.evidence_ids[0]]['title']
        other_titles = [e['title'] for k, e in state.evidence.items() if k not in node.evidence_ids]
        if title in text and other_titles:
            result.append({'node_id': node.id, 'type': 'entity_substitution', 'original': text,
                           'modified': text.replace(title, other_titles[0], 1), 'validity': 'human_review_required'})
        if re.search(r'\b(born|founded|married|located)\b', text):
            result.append({'node_id': node.id, 'type': 'incorrect_relationship', 'original': text,
                           'modified': re.sub(r'\b(born|founded|married|located)\b', 'dissolved', text, count=1),
                           'validity': 'human_review_required'})
        result.append({'node_id': node.id, 'type': 'unsupported_conclusion', 'original': text,
                       'modified': text+' Therefore it won the international award.', 'validity': 'human_review_required'})
        if other_titles:
            result.append({'node_id': node.id, 'type': 'evidence_misattribution', 'original': text,
                           'modified': f"According to {other_titles[0]}, {text}", 'validity': 'human_review_required'})
    return result


def select_conditions(state: DecisionState, maximum: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose up to three distinct primary error types without inventing eligibility.

    Args:
        state: Clean trace whose supporting sentences are eligible for perturbation.
        maximum: Cap on primary conditions; unavailable types reduce actual count.
    Returns:
        Primary conditions and auxiliary candidates requiring human validation.
    """
    all_candidates = candidates(state)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in all_candidates:
        if item['validity'].startswith('controlled') and item['type'] not in seen and len(selected) < maximum:
            selected.append(item)
            seen.add(item['type'])
    return selected, [c for c in all_candidates if c['validity']=='human_review_required']
