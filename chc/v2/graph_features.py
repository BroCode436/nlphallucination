"""Pre-action source-bridge hypotheses, separate from measured propagation.

Model-declared dependencies can be empty even on multi-hop questions. A source
document mentioning another cited document's entity supplies an additional
uncertain relevance feature. It does not certify a causal reasoning edge.
"""

from __future__ import annotations
import copy, re
from typing import Any
from ..schema import DecisionState
from ..graph import validate_graph


def enrich_source_bridges(state: DecisionState) -> DecisionState:
    """Add ordered cross-document entity links using serving-visible evidence only.

    Args:
        state: Current claims and retrieved passages; contains no outcome labels.
    Returns:
        A copied validated state. Existing declared edges are retained. An extra
        earlier-to-later edge is added only if a source passage mentions the
        other claim's document entity, with distinct source IDs. Parenthetical
        and comma suffixes in titles are removed for matching. Minimum four
        letters avoids links on short ambiguous titles. Confidence .35 is a
        heuristic indicator, not a calibrated causal probability.
    Limitations:
        Document mentions do not prove claim dependence. Direction follows
        observable order. This fixed recipe must be ablated and human-reviewed.
    """
    result = copy.deepcopy(state)
    nodes = [c for c in result.claims if not c.is_final]
    existing = {(e["source"], e["target"]) for e in result.edges}
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if (source.id, target.id) in existing or set(source.evidence_ids) & set(
                target.evidence_ids
            ):
                continue
            source_text = " ".join(
                result.evidence[e]["text"] for e in source.evidence_ids
            ).casefold()
            target_text = " ".join(
                result.evidence[e]["text"] for e in target.evidence_ids
            ).casefold()
            source_titles = [
                re.split(r"[,(]", result.evidence[e]["title"])[0].strip()
                for e in source.evidence_ids
            ]
            target_titles = [
                re.split(r"[,(]", result.evidence[e]["title"])[0].strip()
                for e in target.evidence_ids
            ]
            matched = [t for t in target_titles if len(t) >= 4 and t.casefold() in source_text]
            matched += [t for t in source_titles if len(t) >= 4 and t.casefold() in target_text]
            if matched:
                result.edges.append(
                    {
                        "source": source.id,
                        "target": target.id,
                        "kind": "source_entity_bridge_hypothesis",
                        "confidence": 0.35,
                        "verified_causal": False,
                        "matched_entities": matched,
                        "direction_basis": "observable claim order",
                    }
                )
                existing.add((source.id, target.id))
    validate_graph(result)
    return result


def prepare_policy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive fixed source-bridge states without changing any collected outcome.

    All action subsets were already generated, so label-free graph enrichment
    can be applied before model fitting without regenerating language-model
    outputs. The graph recipe is identical across train, validation and test.
    This is feature extraction, never post-hoc editing of experimental benefits.
    """
    result = []
    for record in records:
        state = enrich_source_bridges(DecisionState.from_dict(record["pre_state"]))
        result.append(
            dict(
                record,
                pre_state=state.to_dict(),
                declared_only_pre_state=record.get("declared_only_pre_state", record["pre_state"]),
                graph_recipe="source_bridge_v1; possible edges, not causal",
            )
        )
    return result
