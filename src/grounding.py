"""Explainable lexical and local numerical grounding (heuristics, not entailment).
"""
from typing import Any as _DocAny
import re
from decimal import Decimal

STOPWORDS = set('the and was were that with from this for are has had have its their into not of in on at to a an is by'.split())
NUMBER = re.compile(r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)')


def clean_text(value: _DocAny) -> _DocAny:
    """Clean text.

    Normalizes supplied text and checks local lexical/numerical consistency.

    Learning notes — input/output contract:
        Parameters:
            value: Value supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    if value is None:
        return ''
    text = str(value).strip()
    return '' if text.lower() in {'nan', 'none', '<na>', 'null'} else text


def content_tokens(text: _DocAny) -> _DocAny:
    """Content tokens.

    Normalizes supplied text and checks local lexical/numerical consistency.

    Learning notes — input/output contract:
        Parameters:
            text: Input text to normalize, split, or transform as described above.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    return set(re.findall(r'\b[a-z]{3,}\b', clean_text(text).lower())) - STOPWORDS


def numerical_conflicts(evidence: _DocAny, claim: _DocAny) -> _DocAny:
    """Check each number against the most lexically relevant evidence clause.

        Local clauses avoid validating Alice's score using Bob's score elsewhere.
        Tied clauses with different numbers are ambiguous, rather than contradictions.
        Equivalent comma/decimal formatting is normalized. Units and entailment remain
        outside this heuristic's scope and must be independently assessed.

    Learning notes — input/output contract:
        Parameters:
            evidence: Source passage supplied to the verifier; presence does not establish source truth.
            claim: Statement being checked against evidence.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    evidence, claim = clean_text(evidence), clean_text(claim)
    clauses = [c for c in re.split(r'(?<=[!?;])\s*|(?<!\d)\.(?!\d)\s*|\n+', evidence) if c.strip()]
    normalized = lambda value: str(Decimal(value.replace(',', '')).normalize())
    records = []
    for match in NUMBER.finditer(claim):
        # Context belongs to this assertion, not the entire multi-claim response.
        left = max(claim.rfind(';', 0, match.start()), claim.rfind('\n', 0, match.start())) + 1
        right_candidates = [p for p in (claim.find(';', match.end()), claim.find('\n', match.end())) if p >= 0]
        context = content_tokens(claim[left:min(right_candidates) if right_candidates else len(claim)])
        scored = [(len(context & content_tokens(c)), c) for c in clauses]
        best = max((score for score, _ in scored), default=0)
        relevant = [c for score, c in scored if score == best and score > 0]
        value = normalized(match.group())
        candidates = [set(normalized(m.group()) for m in NUMBER.finditer(c)) for c in relevant]
        supported = bool(candidates) and all(value in values for values in candidates)
        ambiguous = not candidates or any(value in values for values in candidates) != supported
        records.append({'number': match.group(), 'start': match.start(), 'end': match.end(),
                        'supported': supported, 'ambiguous': ambiguous,
                        'conflict': not supported and not ambiguous,
                        'evidence_clauses': relevant})
    return records
