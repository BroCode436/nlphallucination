"""NetworkX textual dependencies and strictly pre-intervention features.

Edges describe annotation-informed or explicit-name textual links. They are
uncertain hypotheses, not verified causal relationships or observed effects.
"""
from __future__ import annotations

import re
from typing import Any
import networkx as nx
import numpy as np
from .schema import Claim, DecisionState

FEATURE_NAMES = ['risk', 'uncertainty', 'descendants', 'depth', 'distance_to_final',
                 'evidence_quality', 'verification_tokens', 'claim_length', 'has_number',
                 'has_negation', 'position', 'in_degree', 'out_degree', 'is_final']
GRAPH_FEATURES = [2, 3, 4, 10, 11, 12]


def build_state(question_id: str, question: str, facts: list[dict[str, str]],
                evidence: dict[str, dict[str, str]]) -> DecisionState:
    """Build a small trace graph from annotated sentences and an unfilled answer.

    Args:
        question_id: Original dataset ID used for grouping.
        question: Question text; never the reference answer.
        facts: Extracted sentence records containing text/title/evidence_id.
        evidence: Full relevant documents available to verification.
    Returns:
        State whose final node contains the question, not a future/gold answer.
    Limitations:
        Sentence nodes are not guaranteed atomic. Final edges mean that annotated
        support may be relevant; they do not prove a counterfactual causal effect.
    """
    claims = [Claim(f'c{i}', f['text'], [f['evidence_id']], i) for i, f in enumerate(facts)]
    claims.append(Claim('final', question, [], len(facts), is_final=True, status='not_generated'))
    edges: list[dict[str, Any]] = []
    for i, fact in enumerate(facts):
        edges.append({'source': f'c{i}', 'target': 'final', 'kind': 'annotation_informed_relevance',
                      'confidence': .5, 'verified_causal': False})
        for j in range(i+1, len(facts)):
            # An explicit document-title mention is stronger than similarity but
            # still only a candidate textual dependency. Earlier->later prevents
            # cycles by construction; validation below independently checks this.
            if facts[j]['title'].casefold() in fact['text'].casefold():
                edges.append({'source': f'c{i}', 'target': f'c{j}', 'kind': 'explicit_name_reference',
                              'confidence': .4, 'verified_causal': False})
    state = DecisionState(question_id, question, claims, edges, evidence)
    validate_graph(state)
    return state


def validate_graph(state: DecisionState) -> nx.DiGraph:
    """Validate a DAG and all evidence references before computing features.

    Args:
        state: Pre-action state with claims, directed edges, and evidence.
    Returns:
        Validated directed NetworkX graph keyed by claim identifiers.
    Raises:
        ValueError: On duplicate nodes/text, missing evidence/endpoints, cycles,
            empty claims, or edges that lack documented dependency metadata.
    """
    ids = [c.id for c in state.claims]
    texts = [c.text.strip().casefold() for c in state.claims]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts) or not all(texts):
        raise ValueError('Duplicate or empty claim nodes')
    if sum(c.is_final for c in state.claims) != 1:
        raise ValueError('Exactly one final placeholder is required')
    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    for c in state.claims:
        if any(e not in state.evidence or not state.evidence[e]['text'].strip() for e in c.evidence_ids):
            raise ValueError('Missing evidence reference or blank evidence')
        if not c.is_final and not c.evidence_ids:
            raise ValueError('Intermediate claims require supplied evidence')
    for edge in state.edges:
        if edge.get('source') not in ids or edge.get('target') not in ids or not edge.get('kind'):
            raise ValueError('Missing edge endpoints or dependency provenance')
        if not 0 <= edge.get('confidence', -1) <= 1:
            raise ValueError('Dependency confidence must be between zero and one')
        if edge.get('verified_causal'):
            raise ValueError('This baseline cannot certify causal edges')
        graph.add_edge(edge['source'], edge['target'])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Claim dependencies contain a cycle')
    return graph


def node_features(state: DecisionState) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create graph/node tensors exclusively from serving-visible state.

    Args:
        state: A DecisionState without references, intervention outcomes, or labels.
    Returns:
        Features [nodes,14], integer edge_index [2,edges], and ordered node IDs.
    Complexity:
        Descendant/path queries cost O(V*(V+E)); graphs are capped at a few nodes.
    """
    graph = validate_graph(state)
    order = [c.id for c in state.claims]
    final = next(c.id for c in state.claims if c.is_final)
    depths: dict[str, int] = {}
    for node in nx.topological_sort(graph):
        depths[node] = max((depths[p]+1 for p in graph.predecessors(node)), default=0)
    rows = []
    for c in state.claims:
        distance = nx.shortest_path_length(graph, c.id, final) if nx.has_path(graph, c.id, final) else len(order)
        rows.append([c.risk, 1-abs(c.risk-.5)*2, len(nx.descendants(graph, c.id)), depths[c.id], distance,
                     c.evidence_quality, c.cost_tokens, len(c.text.split()), float(bool(re.search(r'\d', c.text))),
                     float(bool(re.search(r'\b(not|never|no)\b', c.text, flags=re.I))), c.position,
                     graph.in_degree(c.id), graph.out_degree(c.id), float(c.is_final)])
    pairs = [(order.index(a), order.index(b)) for a, b in graph.edges]
    edge_index = np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    return np.asarray(rows, dtype=np.float32), edge_index, order
