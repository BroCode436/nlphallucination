"""Explicit citation alignment for already generated observable factual claims.

This is a separately audited processing stage, not silent JSON repair. Invalid
JSON and bad dependencies are rejected. Missing source references are supplied
by documented retrieval, which does not declare a statement true. Clean-prefix
factual acceptance is a separate gate. Claim text and
dependency declarations remain byte-for-byte the model's actual content.
"""

from __future__ import annotations
import copy, json, re
from typing import Any
from .traces import parse_trace


def align_uncited_artifact(
    generation: dict[str, Any],
    question: str,
    evidence: dict[str, dict[str, str]],
    judge: Any,
    prefix_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a new validated citation-aligned artifact or preserve the rejection.

    Args:
        generation: Real raw model output with its original validation outcome.
        question/evidence: Serving-visible data only; no answer/reference labels.
        judge: Automatic source alignment model; thresholds fixed before testing.
        prefix_ids: Prior claims allowed in dependencies.
    Returns:
        Original generation with explicit raw-validation and alignment provenance.
        A processed artifact gets a fresh trace with explicit retrieved-source
        provenance; contradiction and insufficient evidence remain possible.
        The raw output is
        retained unchanged. This artifact is not a raw schema-compliant response.
    """
    result = copy.deepcopy(generation)
    result["raw_validation_error"] = generation.get("validation_error")
    result["citation_alignment"] = []
    if generation.get("trace") is not None:
        result["artifact_origin"] = "raw_model_json"
        return result
    try:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", generation["raw"].strip())
        obj = json.loads(raw)
        if (
            not isinstance(obj, dict)
            or set(obj) != {"claims", "final_answer"}
            or not isinstance(obj["claims"], list)
        ):
            return result
        candidate = copy.deepcopy(obj)
        uncited = []
        for c in candidate["claims"]:
            if not isinstance(c, dict) or "evidence_ids" not in c:
                return result
            if c["evidence_ids"] == []:
                uncited.append(c)
                c["evidence_ids"] = [next(iter(evidence))]
        if not uncited:
            return result
        # Structural validation with temporary source placeholders occurs before
        # NLI. Placeholders are never admitted as final references or labels.
        parse_trace(json.dumps(candidate), evidence, question, prefix_ids)
        pairs = [(c["text"], e["text"]) for c in uncited for e in evidence.values()]
        probabilities = judge.probabilities(pairs)
        ids = list(evidence)
        for i, c in enumerate(uncited):
            block = probabilities[i * len(ids) : (i + 1) * len(ids)]
            supported = [k for k in range(len(ids)) if block[k]["entailment"] >= judge.threshold]
            contradicted = [
                k for k in range(len(ids)) if block[k].get("contradiction", 0) >= judge.threshold
            ]
            if supported:
                chosen = max(supported, key=lambda k: block[k]["entailment"])
                relation = "supported_candidate"
            elif contradicted:
                chosen = max(contradicted, key=lambda k: block[k].get("contradiction", 0))
                relation = "contradicted_candidate"
            else:
                terms = set(re.findall(r"\w+", c["text"].casefold()))
                chosen = max(
                    range(len(ids)),
                    key=lambda k: len(
                        terms & set(re.findall(r"\w+", evidence[ids[k]]["text"].casefold()))
                    ),
                )
                relation = "insufficient_evidence"
            # Retrieval must not filter out false continuations. Otherwise the
            # benchmark would censor the very propagation it intends to measure.
            # Clean starting claims are still checked for support by collection.
            c["evidence_ids"] = [ids[chosen]]
            result["citation_alignment"].append(
                {
                    "claim_id": c["id"],
                    "assigned_evidence_id": ids[chosen],
                    "entailment_score": block[chosen]["entailment"],
                    "contradiction_score": block[chosen].get("contradiction"),
                    "retrieved_source_relation": relation,
                    "origin": "automatic_source_alignment; not model citation or human validation",
                }
            )
        result["trace"] = parse_trace(json.dumps(candidate), evidence, question, prefix_ids)
        result["validation_error"] = None
        result["artifact_origin"] = "explicit_citation_aligned_model_claims"
        return result
    except (ValueError, TypeError, KeyError, StopIteration):
        return result


class AlignedTraceBackend:
    """Compose real raw generation with explicit, reference-free source alignment.

    Args:
        backend: Existing TraceBackend instance, configured without grammar.
        judge: Shared automatic NLI model.
    Attributes delegate execution counters/model identity to the actual backend;
    alignment changes metadata only, never model text or measured answer strings.
    """

    def __init__(self, backend: Any, judge: Any) -> None:
        """Retain the real inference objects; no second language model is loaded."""
        self.backend = backend
        self.judge = judge

    def __getattr__(self, name: str) -> Any:
        """Delegate resource metadata and counters to the underlying real model."""
        return getattr(self.backend, name)

    def generate(
        self,
        question: str,
        evidence: dict[str, dict[str, str]],
        prefix: list[dict[str, Any]] | None = None,
        seed: int = 101,
        visible_evidence: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate raw content then validate separately aligned citations.

        The output retains raw_validation_error and citation_alignment, making
        strict raw-JSON reliability distinguishable from processed-artifact yield.
        """
        output = self.backend.generate(
            question, visible_evidence if visible_evidence is not None else evidence, prefix, seed
        )
        result = align_uncited_artifact(
            output, question, evidence, self.judge, tuple(c["id"] for c in (prefix or []))
        )
        result["source_visibility"] = (
            "full" if visible_evidence is None else "explicit_restricted_continuation_context"
        )
        return result
