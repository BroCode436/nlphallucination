"""Counterfactual-aware repair checks layered over the retained lexical verifier.

Source entailment alone cannot establish a valid repair: a different supported
sentence may describe the wrong entity or discard valid information. This module
adds conservative anchor/preservation diagnostics and semantic evidence checks.
It is an untrained safety extension, not a claimed newly trained CAHR model.
"""

from __future__ import annotations
import copy, re, sys, time
from pathlib import Path
from typing import Any


class RepairVerifier:
    """Propose existing lexical repairs and apply an additional semantic gate.

    Args:
        project: Existing root containing correction_validity.joblib.
        judge: Shared evidence-NLI checker; its judgments remain automatic proxies.
    Attributes:
        lexical: Preserved trained support/preservation model.
    """

    def __init__(self, project: Path, judge: Any) -> None:
        """Reuse saved correction weights without synthetic relabeling or refitting."""
        if str(project / "src") not in sys.path:
            sys.path.insert(0, str(project / "src"))
        from correction_validity import CorrectionValidityModel

        self.lexical = CorrectionValidityModel(project / "models/correction_validity.joblib")
        self.judge = judge

    def verify(
        self,
        question: str,
        claim: dict[str, Any],
        evidence: dict[str, dict[str, str]],
        semantic_gate: bool = True,
    ) -> dict[str, Any]:
        """Return a sourced action, with explicit reasons for acceptance/abstention.

        Question and original claim are supplied alongside retrieved evidence and
        the actual proposed correction. Capitalized anchors in the original that
        appear in source evidence must remain in the proposal. This conservative
        heuristic may reject valid paraphrases. The lexical proposal mechanism is
        shared across policies; an ablation can remove only the extra gate.
        """
        start = time.perf_counter()
        ev = " ".join(evidence[e]["text"] for e in claim["evidence_ids"])
        proposal = self.lexical.propose(ev, claim["text"])
        replacement = proposal["final_claim"]
        resolved_semantic_deferral = False
        candidates = proposal.get("candidates", [])
        if candidates:
            # The original verifier explicitly defers negation/numeric semantics
            # even when learned support and preservation exceed their thresholds.
            # The added NLI gate can investigate that deferral. It cannot bypass
            # failed learned score thresholds or silently choose a new candidate.
            ranked = sorted(
                candidates,
                key=lambda x: min(x.get("support_score", 0), x.get("preservation_score", 0)),
                reverse=True,
            )
            best = ranked[0]
            thresholds = best.get("thresholds", {"support": 0.95, "preservation": 0.95})
            score_ok = (
                best.get("support_score", 0) >= thresholds["support"]
                and best.get("preservation_score", 0) >= thresholds["preservation"]
            )
            semantic_only = all(
                "semantic review" in reason.lower() for reason in best.get("reasons", [])
            )
            if score_ok and semantic_only:
                replacement = best.get("proposal", replacement)
                resolved_semantic_deferral = (
                    not best.get("accepted", False) and replacement != claim["text"]
                )
        original_j = self.judge.assess([claim], evidence)[0]
        candidate = dict(claim, text=replacement)
        new_j = self.judge.assess([candidate], evidence)[0]
        anchors = {
            a
            for a in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", claim["text"])
            if a.casefold() in ev.casefold() and a not in ("The", "It", "He", "She", "This")
        }
        missing = sorted(a for a in anchors if a.casefold() not in replacement.casefold())
        reasons = []
        if new_j["status"] != "supported":
            reasons.append("replacement_not_semantically_supported")
        if missing:
            reasons.append("possible_wrong_entity_or_valid_information_deletion")
        if original_j["status"] == "supported" and replacement != claim["text"]:
            reasons.append("original_already_supported")
        accepted = replacement != claim["text"] and (not semantic_gate or not reasons)
        output = replacement if accepted else claim["text"]
        return {
            "claim_id": claim["id"],
            "question": question,
            "original": claim["text"],
            "proposal": replacement,
            "replacement": output,
            "accepted": accepted,
            "status": "corrected" if accepted else "unchanged_or_rejected",
            "rejection_reasons": reasons,
            "missing_anchors": missing,
            "lexical_result": proposal,
            "resolved_lexical_semantic_deferral": resolved_semantic_deferral and accepted,
            "original_judgment": original_j,
            "proposed_judgment": new_j,
            "evidence_ids": claim["evidence_ids"],
            "evidence": [{**evidence[e], "id": e} for e in claim["evidence_ids"]],
            "calls": 1,
            "estimated_tokens": len(ev.split())
            + len(claim["text"].split())
            + len(replacement.split()),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "human_review": "pending",
            "harmful_repair_proxy": bool(
                accepted
                and original_j["status"] == "supported"
                and new_j["status"] == "contradicted"
            ),
        }
