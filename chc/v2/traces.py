"""Validate observable reasoning artifacts before graph building or evaluation.

Only explicit short factual statements are requested. These are public model
outputs, not measurements of private model reasoning. Malformed structures raise
errors and are retained in the rejection log by the collection pipeline.
"""

from __future__ import annotations
import json
import re
from typing import Any
from ..schema import Claim, DecisionState
from ..graph import validate_graph


def parse_trace(
    raw: str, evidence: dict[str, dict[str, str]], question: str, prefix_ids: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Parse strict JSON and validate ordered, source-linked factual claims.

    Args:
        raw: Actual generated text; only a surrounding Markdown fence is removed.
        evidence: Available source IDs. Unknown citations invalidate the trace.
        question: Serving-visible question, attached by the caller, not a label.
        prefix_ids: Earlier visible claim IDs that a continuation may reference.
    Returns:
        Validated JSON object with the question attached and an atomicity warning.
    Raises:
        ValueError: Invalid JSON, extra/missing fields, empty/duplicate claims,
            forward dependencies, missing evidence, or an oversized artifact.
    Algorithm:
        Topological order is enforced as claims are read. This also prevents
        cycles without guessing dependency relationships from final outcomes.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Generated artifact is not valid JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"claims", "final_answer"}:
        raise ValueError("Trace must contain exactly claims and final_answer")
    if not isinstance(obj["final_answer"], str) or not obj["final_answer"].strip():
        raise ValueError("Final answer must be nonempty text")
    if not isinstance(obj["claims"], list) or not 1 <= len(obj["claims"]) <= 4:
        raise ValueError("Trace requires one to four explicit factual claims")
    seen = set(prefix_ids)
    texts = set()
    for c in obj["claims"]:
        if not isinstance(c, dict) or set(c) != {"id", "text", "depends_on", "evidence_ids"}:
            raise ValueError("Claim fields are not the required strict schema")
        if (
            not isinstance(c["id"], str)
            or not re.fullmatch(r"C[1-9]\d*", c["id"])
            or c["id"] in seen
        ):
            raise ValueError("Claim ID must be unique and of the form C1")
        if (
            not isinstance(c["text"], str)
            or not 3 <= len(c["text"].strip()) <= 450
            or c["text"].casefold() in texts
        ):
            raise ValueError("Empty, duplicate, or oversized claim")
        if "?" in c["text"] or "\n" in c["text"].strip():
            raise ValueError("Claim must be a factual assertion, not a question or paragraph")
        if (
            not isinstance(c["depends_on"], list)
            or any(not isinstance(d, str) for d in c["depends_on"])
            or len(set(c["depends_on"])) != len(c["depends_on"])
            or any(d not in seen for d in c["depends_on"])
        ):
            raise ValueError("Dependencies must reference unique earlier claims")
        if (
            not isinstance(c["evidence_ids"], list)
            or not c["evidence_ids"]
            or any(not isinstance(e, str) or e not in evidence for e in c["evidence_ids"])
        ):
            raise ValueError("Every factual claim needs an available evidence ID")
        seen.add(c["id"])
        texts.add(c["text"].casefold())
    obj["question"] = question
    obj["atomicity"] = (
        "Prompt-constrained short factual claims; independent human atomicity review pending"
    )
    return obj


def trace_state(
    question_id: str,
    question: str,
    claims: list[dict[str, Any]],
    evidence: dict[str, dict[str, str]],
) -> DecisionState:
    """Convert validated visible claims into the existing label-free CHC state.

    Inputs contain text, source references and model-declared dependencies only.
    The final placeholder contains the question, never a reference answer. Edges
    carry explicit uncertain provenance; downstream effects are stored elsewhere.
    Returns a validated DecisionState or raises ValueError for inconsistent input.
    """
    nodes = [Claim(c["id"], c["text"], list(c["evidence_ids"]), i) for i, c in enumerate(claims)]
    nodes.append(Claim("final", question, [], len(nodes), is_final=True, status="not_generated"))
    edges = [
        {
            "source": d,
            "target": c["id"],
            "kind": "model_declared_dependency",
            "confidence": 0.5,
            "verified_causal": False,
        }
        for c in claims
        for d in c["depends_on"]
        if d in {x["id"] for x in claims}
    ]
    edges += [
        {
            "source": c["id"],
            "target": "final",
            "kind": "possible_answer_relevance",
            "confidence": 0.3,
            "verified_causal": False,
        }
        for c in claims
    ]
    state = DecisionState(question_id, question, nodes, edges, evidence)
    validate_graph(state)
    return state
