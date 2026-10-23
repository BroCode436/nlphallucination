"""Typed data contracts shared by loaders, policies, and experiments.

Only DecisionState is passed to policy feature extraction. Reference answers,
injection identities, and post-intervention labels belong to separate records.
This boundary reduces accidental label leakage; it is not a security sandbox.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass
class Claim:
    """One visible reasoning node; its risk is a prediction, not a gold label.

    Attributes:
        id: Unique node identifier within the question.
        text: Current statement (or the question for the final placeholder).
        evidence_ids: Passage identifiers accessible before selection.
        position: Order in the extracted trace, not a causal timestamp.
        is_final: Whether this is the ungenerated final-answer placeholder.
        risk: Detector score, not necessarily calibrated.
        evidence_quality: Lexical relevance score, not source trustworthiness.
        cost_tokens: Estimated evidence-plus-claim tokens for one verification.
        status: Observable verification state; never the injection label.
    """
    id: str
    text: str
    evidence_ids: list[str]
    position: int
    is_final: bool = False
    risk: float = 0.5
    evidence_quality: float = 0.0
    cost_tokens: int = 1
    status: str = 'unverified'


@dataclass
class DecisionState:
    """Serving-visible inputs, deliberately excluding reference outcomes.

    Attributes:
        question_id: Dataset identifier for grouping; not used as a feature.
        question: User's question, available at decision time.
        claims: Visible nodes including a final placeholder without gold text.
        edges: Directed textual links with kind/confidence provenance.
        evidence: Source passages keyed by stable identifiers.
    """
    question_id: str
    question: str
    claims: list[Claim]
    edges: list[dict[str, Any]]
    evidence: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state without attaching evaluation labels.

        Returns:
            A new nested dictionary; changes to it do not mutate this object.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionState:
        """Parse a saved pre-action state, rejecting unexpected top-level keys.

        Args:
            data: Serialized DecisionState, not a complete labeled episode.
        Returns:
            A typed state suitable for graph validation and policy scoring.
        Raises:
            ValueError: If forbidden label/outcome fields were included.
            TypeError: If required fields or claim structures are malformed.
        """
        allowed = {'question_id', 'question', 'claims', 'edges', 'evidence'}
        if set(data) != allowed:
            raise ValueError(f'State fields must be exactly {sorted(allowed)}')
        return cls(data['question_id'], data['question'],
                   [Claim(**c) for c in data['claims']], data['edges'], data['evidence'])


@dataclass
class Budget:
    """Hard selection limits, independent of observed future outcomes.

    Args:
        calls: Maximum selected claims; zero is a valid no-check budget.
        tokens: Optional sum of estimated verification-token costs.
        latency_ms: Optional predicted verification latency, not a runtime promise.
        milliseconds_per_call: Predeclared latency estimate used for selection.
    """
    calls: int = 1
    tokens: int | None = None
    latency_ms: float | None = None
    milliseconds_per_call: float = 20.0

    def __post_init__(self) -> None:
        """Reject negative limits immediately instead of violating budgets later.

        Raises:
            ValueError: When a cost estimate or budget is negative.
        """
        if not isinstance(self.calls,int) or isinstance(self.calls,bool):
            raise ValueError('Verification-call budget must be an integer')
        if self.tokens is not None and (not isinstance(self.tokens,int) or isinstance(self.tokens,bool)):
            raise ValueError('Estimated-token budget must be an integer')
        values = [self.calls, self.milliseconds_per_call]
        values += [v for v in (self.tokens, self.latency_ms) if v is not None]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Budgets and latency estimates must be finite and nonnegative')
