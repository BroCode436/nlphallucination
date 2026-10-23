"""Adapter for the existing real DistilBERT weights and safe correction model.

Prediction scores are explicitly uncalibrated unless a separate calibration
procedure is fitted. The adapter never reads reference answers or injection IDs.
"""
from __future__ import annotations

import sys
import copy
import hashlib
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .resources import choose_device
from .schema import DecisionState


class ExistingDetector:
    """Load the user's saved cross-encoder once and batch actual inference.

    Args:
        project: Existing project root holding results/checkpoints.
        device: 'auto', 'cpu', or one CUDA device.
        batch_size: Maximum inference batch, automatically reduced on OOM.
        max_length: Pair truncation budget, recorded as a limitation.
    Attributes:
        model/tokenizer: Actual locally loaded pretrained/fine-tuned objects.
        fallback_events: OOM events retained for experiment reporting.
    """
    def __init__(self, project: Path, device: str = 'auto', batch_size: int = 16, max_length: int = 128) -> None:
        """Initialize local-only weights; missing artifacts raise an explicit error.

        Args:
            project: Project root with the existing checkpoint.
            device: Requested execution device, with CPU fallback if unavailable.
            batch_size: Positive number of pairs per inference batch.
            max_length: Token limit of the saved detector workflow.
        Returns:
            None; the instance owns loaded model weights and tokenizer.
        """
        self.device = choose_device(device)
        self.path = project/'results'/'checkpoints'/'best_transformer_detector'
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path, local_files_only=True).to(self.device).eval()
        self.batch_size, self.max_length = max(1, batch_size), max_length
        self.fallback_events: list[str] = []
        self.verification_executions = 0
        self.verification_cache_hits = 0
        self._verification_cache: dict[str, dict[str, Any]] = {}
        self.identifier = 'existing-distilbert-local'
        if str(project/'src') not in sys.path:
            sys.path.insert(0, str(project/'src'))
        from correction_validity import CorrectionValidityModel
        self.corrector = CorrectionValidityModel(project/'models'/'correction_validity.joblib')

    def predict_pairs(self, claims: list[str], evidence: list[str]) -> list[float]:
        """Return actual hallucination scores for equally sized text lists.

        Args:
            claims: Candidate claims, not gold correctness labels.
            evidence: Matching source passages in the same order.
        Returns:
            Positive-class softmax scores in input order.
        Raises:
            ValueError: If input lengths disagree.
        OOM handling:
            Halve inference batches, then retry on CPU if a single pair fails.
        """
        if len(claims) != len(evidence):
            raise ValueError('Evidence/claim lengths differ')
        scores: list[float] = []
        offset = 0
        while offset < len(claims):
            size = min(self.batch_size, len(claims)-offset)
            try:
                encoded = self.tokenizer(evidence[offset:offset+size], claims[offset:offset+size],
                                         return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type=='cuda'):
                    logits = self.model(**encoded).logits
                scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
                offset += size
            except torch.cuda.OutOfMemoryError:
                self.fallback_events.append(f'detector_oom_batch_{size}')
                torch.cuda.empty_cache()
                if size > 1:
                    self.batch_size = max(1, size//2)
                else:
                    self.device = torch.device('cpu')
                    self.model.to(self.device)
        return scores

    def predict_hallucination(self, claim: str, evidence: str, claim_id: str = 'claim',
                              evidence_ids: list[str] | None = None) -> dict[str, Any]:
        """Expose the common detector interface for one claim.

        Args:
            claim: Statement to compare with the supplied evidence.
            evidence: Reference passage available before intervention.
            claim_id: Identifier echoed in the result.
            evidence_ids: Optional source identifiers for traceability.
        Returns:
            Score, thresholded label, model ID, and explicit calibration status.
        """
        score = self.predict_pairs([claim], [evidence])[0]
        return {'claim_id': claim_id, 'probability': score, 'label': int(score >= .5),
                'evidence_ids': evidence_ids or [], 'model_id': self.identifier,
                'calibration': 'raw_softmax; no calibration fitted for HotpotQA'}

    def annotate(self, state: DecisionState) -> DecisionState:
        """Attach predicted risk and lexical evidence quality to visible nodes.

        Args:
            state: Mutable pre-intervention state; final placeholder is not scored.
        Returns:
            The same state with detector scores and estimated token costs.
        Limitations:
            Full-document verification can be truncated at max_length.
        """
        nodes = [c for c in state.claims if not c.is_final]
        passages = [' '.join(state.evidence[e]['text'] for e in c.evidence_ids) for c in nodes]
        scores = self.predict_pairs([c.text for c in nodes], passages)
        for c, ev, score in zip(nodes, passages, scores):
            c.risk = score
            cw = set(c.text.casefold().split())
            c.evidence_quality = len(cw & set(ev.casefold().split()))/max(len(cw), 1)
            c.cost_tokens = len(self.tokenizer.encode(ev, c.text, truncation=False))
        return state

    def verify(self, state: DecisionState, claim_id: str, safe: bool = True) -> dict[str, Any]:
        """Retrieve supplied sources and accept only justified candidate repairs.

        Args:
            state: Visible state containing current claims and passage references.
            claim_id: Intermediate node to verify; final placeholders are excluded.
            safe: If false, use lexical-best source substitution for an ablation.
        Returns:
            Action status, original/replacement text, actual elapsed time, estimated
            verification tokens, and full correction evidence/decision details.
        Limitations:
            Retrieval uses supplied annotation-linked documents, not web search.
        """
        start = time.perf_counter()
        node = next(c for c in state.claims if c.id == claim_id and not c.is_final)
        evidence = ' '.join(state.evidence[e]['text'] for e in node.evidence_ids)
        # The verifier is deterministic. Reusing the exact same claim/evidence
        # decision across generation seeds changes no intervention outcome and
        # avoids repeatedly evaluating dozens of sentence candidates on CPU.
        key = hashlib.sha256(repr((evidence, node.text, safe)).encode()).hexdigest()
        if key in self._verification_cache:
            self.verification_cache_hits += 1
            cached = copy.deepcopy(self._verification_cache[key])
            cached.update(claim_id=claim_id, evidence_ids=node.evidence_ids, cache_hit=True,
                          execution_latency_ms=(time.perf_counter()-start)*1000)
            return cached
        self.verification_executions += 1
        if safe:
            result = self.corrector.propose(evidence, node.text)
            replacement = result['final_claim']
        else:
            from correction_validity import clauses, words, recall
            replacement = max(clauses(evidence), key=lambda s: recall(words(node.text), words(s)), default=node.text)
            result = {'status': 'unsafe_lexical_substitution', 'final_claim': replacement}
        status = 'claim_corrected' if replacement != node.text else 'claim_unchanged'
        if not evidence.strip():
            status = 'evidence_insufficient'
        elif safe and result['status'] == 'needs_review':
            status = 'correction_rejected'
        action = {'claim_id': claim_id, 'status': status, 'verification_succeeded': bool(evidence.strip()),
                'original': node.text, 'replacement': replacement, 'evidence_ids': node.evidence_ids,
                'evidence': evidence, 'details': result, 'calls': 1, 'estimated_tokens': node.cost_tokens,
                'latency_ms': (time.perf_counter()-start)*1000, 'cache_hit': False,
                'execution_latency_ms': (time.perf_counter()-start)*1000}
        self._verification_cache[key] = copy.deepcopy(action)
        return action
