"""Conservative factual-unit extraction and recorded-output sensitivity scoring.

Only explicit sentence boundaries and shared-subject verb conjunctions are split.
Uncertain relative clauses/lists are flagged rather than inventing paraphrases.
These units are correlated pieces of an observable model artifact, not independent
examples or access to private reasoning. Human atomicity review remains necessary.
"""

from __future__ import annotations
import copy, re
from typing import Any
from .propagation import factual_summary, utility
from .validity import selected_corruption_validity


def factual_units(text: str) -> list[dict[str, Any]]:
    """Extract conservative units while preserving original text and edit provenance.

    Args:
        text: One actual generated factual statement, possibly compound.
    Returns:
        Units with text, original text, extraction rule and a residual-compound
        flag. Shared subjects are copied only when an explicit verb follows
        “and”; noun lists, relative clauses and ambiguous conjunctions remain
        unsplit. The parser does not claim complete semantic atomicity.
    """
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])|;\s*", text) if p.strip()]
    output = []
    verbs = r"is|was|are|were|has|had|died|lives|lived|works|worked|founded|married|published"
    for sentence in sentences:
        match = re.match(r"^(.*?)\b(?:" + verbs + r")\b", sentence, re.I)
        parts = re.split(r"\s+and\s+(?=(?:" + verbs + r")\b)", sentence, flags=re.I)
        if len(parts) > 1 and match and match.group(1).strip():
            subject = match.group(1).strip()
            pieces = [parts[0]] + [subject + " " + p for p in parts[1:]]
            rule = "explicit_shared_subject_verb_split"
        else:
            pieces = [sentence]
            rule = "sentence_boundary" if len(sentences) > 1 else "unchanged_short_assertion"
        for piece in pieces:
            output.append(
                {
                    "text": piece,
                    "original_text": text,
                    "extraction_rule": rule,
                    "residual_compound_possible": bool(
                        re.search(r"\b(and|which|who|while)\b", piece, re.I)
                    ),
                    "human_atomicity_review": "pending",
                }
            )
    return output


def score_atomic_outcome(
    outcome: dict[str, Any], evidence: dict[str, dict[str, str]], judge: Any
) -> dict[str, Any]:
    """Judge units from saved real output without generating or inventing outcomes.

    Invalid structured outputs remain invalid. Existing whole-claim judgments
    are preserved, and unit judgments are stored separately. The factual summary
    is recomputed over evaluable units; insufficient units remain unknown.
    """
    result = copy.deepcopy(outcome)
    if not result["valid"]:
        return result
    units = []
    metadata = []
    for c in result["generation"]["trace"]["claims"]:
        for i, unit in enumerate(factual_units(c["text"])):
            units.append(
                {"id": f"{c['id']}_U{i+1}", "text": unit["text"], "evidence_ids": c["evidence_ids"]}
            )
            metadata.append(dict(unit, parent_claim_id=c["id"]))
    judgments = judge.assess(units, evidence)
    for j, m in zip(judgments, metadata):
        j["extraction"] = m
    result["whole_claim_factual"] = result["factual"]
    result["atomic_judgments"] = judgments
    result["factual"] = factual_summary(judgments)
    result["factual_unit_scope"] = (
        "Conservative rule-extracted units; residual compounds flagged; automatic NLI proxy"
    )
    return result


def atomic_records(records: list[dict[str, Any]], judge: Any) -> list[dict[str, Any]]:
    """Compute the predeclared factual-unit objective from actual recorded outputs.

    Raw sentence-level results remain intact in their original JSONL. Each new
    record preserves its sentence benefits and adds factual-unit benefits. Clean
    prefix eligibility is audited separately; this operation never discards
    neutral/negative interventions based on their effects.
    """
    output = []
    for record in records:
        r = copy.deepcopy(record)
        r["sentence_level_benefits"] = copy.deepcopy(r["benefits"])
        r["selected_corruption_validation"] = selected_corruption_validity(
            r["injection"], r["clean_prefix"], r["evidence"], judge
        )
        prefix_units = []
        for c in r["clean_prefix"]:
            prefix_units += [
                {"id": f"{c['id']}_U{i+1}", "text": u["text"], "evidence_ids": c["evidence_ids"]}
                for i, u in enumerate(factual_units(c["text"]))
            ]
        prefix_j = judge.assess(prefix_units, r["evidence"])
        r["atomic_clean_prefix_judgments"] = prefix_j
        r["atomic_prefix_eligible"] = all(j["status"] == "supported" for j in prefix_j)
        for name in (
            "baseline",
            "clean_continuation",
            "restoration_diagnostic",
            "joint_both",
            "joint_gate_off",
        ):
            if name in r:
                r[name] = score_atomic_outcome(r[name], r["evidence"], judge)
        for name in ("singletons", "without_semantic_gate"):
            r[name] = {n: score_atomic_outcome(o, r["evidence"], judge) for n, o in r[name].items()}
        r["benefits"] = {
            n: utility(
                r["baseline"],
                o,
                harmful_repair=r["verification_actions"][n]["harmful_repair_proxy"],
            )
            for n, o in r["singletons"].items()
        }
        r["evaluation_unit"] = (
            "Conservative factual units; shared-subject rules fixed before final test analysis"
        )
        output.append(r)
    return output
