"""Budgeted policies with honest separation of predictions and action outcomes.

All deployable policies receive exactly the same pre-action DecisionState. The
analysis-only oracle is implemented in evaluation code and cannot be called here.
"""
from __future__ import annotations

import random
from typing import Any
import joblib
import networkx as nx
import numpy as np
from pathlib import Path
from .graph import node_features, validate_graph
from .schema import Budget, DecisionState


def select_claims(state: DecisionState, scores: dict[str, float], budget: Budget,
                  overlap_discount: bool = False, minimum_net_benefit: float | None = None) -> list[str]:
    """Greedily select affordable nodes while tracking overlapping descendants.

    Args:
        state: Visible claim graph with predicted risk/cost only.
        scores: Predicted benefit or baseline priority per intermediate node.
        budget: Hard limits on calls, estimated tokens, and estimated latency.
        overlap_discount: Heuristic diminishing priority for covered descendants.
        minimum_net_benefit: Optional per-call cost penalty; nodes below it are
            skipped. None enforces a quota when affordable, enabling matched-call
            comparisons. Discounted scores are not observed joint causal effects.
    Returns:
        Ordered node IDs satisfying all declared budgets.
    Complexity:
        O(B*V*(V+E)) for a few-node graph; not an exact knapsack optimizer.
    """
    graph = validate_graph(state)
    selected: list[str] = []
    covered: set[str] = set()
    tokens = 0
    nodes = {c.id: c for c in state.claims if not c.is_final}
    remaining = set(nodes)
    while remaining and len(selected) < budget.calls:
        affordable = [n for n in remaining if (budget.tokens is None or tokens+nodes[n].cost_tokens<=budget.tokens)
                      and (budget.latency_ms is None or (len(selected)+1)*budget.milliseconds_per_call<=budget.latency_ms)]
        if not affordable:
            break
        def priority(node: str) -> tuple[float, int]:
            """Score one affordable node, with stable position-based tie breaking.

            Args:
                node: Identifier of an unselected intermediate claim.
            Returns:
                Discounted priority and inverse position for deterministic ties.
            """
            descendants = nx.descendants(graph, node)
            discount = 1/(1+len(descendants & covered)) if overlap_discount else 1.
            return float(scores.get(node, 0.))*discount, -nodes[node].position
        best = max(affordable, key=priority)
        if minimum_net_benefit is not None and priority(best)[0] <= minimum_net_benefit:
            break
        selected.append(best)
        tokens += nodes[best].cost_tokens
        covered |= nx.descendants(graph, best)
        remaining.remove(best)
    return selected


def baseline_scores(state: DecisionState, method: str, seed: int = 42) -> dict[str, float]:
    """Compute transparent priorities without seeing outcomes or reference labels.

    Args:
        state: Serving-visible state shared by every baseline and learned policy.
        method: 'random', 'uncertainty', 'dependency', or 'risk_dependency'. The
            uncertainty baseline follows the requested highest-error-probability
            definition, not entropy near 0.5.
        seed: Reproducible seed used only by the random baseline.
    Returns:
        Priority for every nonfinal node.
    Raises:
        ValueError: On an unknown policy name.
    """
    graph = validate_graph(state)
    # Seed per question, so the random policy does not reset to the same node
    # position for every question. Conditions of one question retain matched
    # random priorities, and different training/evaluation seeds give replicates.
    rng = random.Random(f'{seed}:{state.question_id}')
    scores = {}
    for node in state.claims:
        if node.is_final:
            continue
        reach = len(nx.descendants(graph, node.id))
        if method == 'random':
            value = rng.random()
        elif method == 'uncertainty':
            value = node.risk
        elif method == 'dependency':
            value = float(reach)
        elif method == 'risk_dependency':
            value = node.risk*(1+reach)
        else:
            raise ValueError(f'Unknown baseline: {method}')
        scores[node.id] = value
    return scores


class TabularPolicy:
    """Inference adapter for a fitted intervention-benefit regression checkpoint.

    Args:
        path: Trusted locally produced joblib artifact.
    Attributes:
        artifact: Regressor, training-only feature mask, and checkpoint metadata.
    """
    def __init__(self, path: Path) -> None:
        """Load the saved estimator and feature mask.

        Args:
            path: Existing checkpoint trained on paired intervention outcomes.
        Returns:
            None; missing/incompatible artifacts raise normal loading errors.
        """
        self.artifact = joblib.load(path)

    def scores(self, state: DecisionState) -> dict[str, float]:
        """Predict single-action benefit from pre-action features only.

        Args:
            state: Validated serving-visible graph.
        Returns:
            Predicted reductions in final-node error, keyed by candidate ID.
        Limitations:
            Values can be negative and are not calibrated causal probabilities.
        """
        x, _, order = node_features(state)
        prediction = self.artifact['model'].predict(x[:, self.artifact['columns']])
        return {node.id: float(prediction[i]) for i, node in enumerate(state.claims) if not node.is_final}
