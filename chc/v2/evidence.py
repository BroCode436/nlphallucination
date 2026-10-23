"""Aligned NLI evidence checks with explicit abstention and judgment provenance.

Sentence selection reduces the original detector's truncation problem. Three-way
NLI is a fallible automatic measurement, not an oracle. Neutral evidence remains
insufficient rather than being converted into a contradiction label.
"""

from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..resources import choose_device


def align_evidence(claim: str, evidence: str, sentences: int = 3) -> str:
    """Select relevant whole sentences using label-free lexical overlap.

    Args:
        claim: Visible statement; reference answers and labels are not used.
        evidence: Full supplied passage, never evidence retrieved from the future.
        sentences: Maximum number of top ranked sentences, restored to source order.
    Returns:
        A compact passage. Empty evidence stays empty, causing later abstention.
    Assumptions:
        English punctuation is a heuristic sentence boundary. This selector can
        miss paraphrases; aligned length and truncation are recorded separately.
    """
    chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"\d])', evidence) if s.strip()]
    terms = set(re.findall(r"\w+", claim.casefold())) - set("a an the is was of in to and".split())
    scores = [
        len(terms & set(re.findall(r"\w+", s.casefold()))) / max(1, len(terms)) for s in chunks
    ]
    indices = sorted(sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:sentences])
    return " ".join(chunks[i] for i in indices)


class EvidenceJudge:
    """Batched three-way NLI with cache and a stable four-state claim interface.

    Args:
        cache: Shared model cache and independent evidence-judgment JSON cache.
        threshold: Conservative automatic status threshold, fixed before test.
        device: Requested single execution device; CPU is supported.
    Attributes:
        labels: Model-card order is checked against the downloaded configuration.
        calibration: Optional validation-fitted binary contradiction calibration;
            it does not calibrate three-way truth or turn neutral into false.
    """

    model_id = "cross-encoder/nli-deberta-v3-xsmall"

    def __init__(self, cache: Path, threshold: float = 0.85, device: str = "auto") -> None:
        """Load the small NLI checkpoint and verify label semantics.

        Missing weights trigger ordinary Hugging Face loading, never fake scores.
        A wrong label mapping raises an error rather than reversing factual labels.
        """
        self.cache = cache
        self.threshold = threshold
        self.device = choose_device(device)
        self.calibration = None
        self.fallbacks = []
        (cache / "judgments").mkdir(parents=True, exist_ok=True)
        local = any(
            (cache / "weights/models--cross-encoder--nli-deberta-v3-xsmall/snapshots").glob(
                "*/config.json"
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache / "weights", local_files_only=local
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=cache / "weights", local_files_only=local
            )
            .to(self.device)
            .eval()
        )
        mapping = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if mapping != {0: "contradiction", 1: "entailment", 2: "neutral"}:
            raise ValueError(f"Unexpected NLI label mapping: {mapping}")
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def probabilities(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Infer [contradiction, entailment, neutral] for aligned evidence/claim pairs.

        Inputs are (claim, full evidence) tuples. Outputs retain selected evidence,
        actual token lengths and cache identity. Token budget is 384 with
        truncation on the evidence side only so the claim is preserved. On CUDA
        OOM the complete small batch is retried on CPU and the event is recorded.
        """
        result = [None] * len(pairs)
        missing = []
        for i, (claim, ev) in enumerate(pairs):
            aligned = align_evidence(claim, ev)
            key = hashlib.sha256(
                json.dumps([self.model_id, self.revision, claim, aligned, 384]).encode()
            ).hexdigest()
            path = self.cache / "judgments" / f"{key}.json"
            if path.exists():
                result[i] = json.loads(path.read_text(encoding="utf-8"))
            else:
                missing.append((i, claim, aligned, key, path))
        for offset in range(0, len(missing), 16):
            batch = missing[offset : offset + 16]
            evs = [b[2] for b in batch]
            claims = [b[1] for b in batch]
            encoded = self.tokenizer(
                evs,
                claims,
                padding=True,
                truncation="only_first",
                max_length=384,
                return_tensors="pt",
            ).to(self.device)
            try:
                with torch.inference_mode():
                    logits = self.model(**encoded).logits
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.fallbacks.append("nli_oom_cpu")
                with torch.inference_mode():
                    logits = self.model(**encoded.to(self.device)).logits
            for b, p in zip(batch, torch.softmax(logits.float(), dim=-1).cpu().tolist()):
                length = len(self.tokenizer.encode(b[2], b[1], truncation=False))
                item = {
                    "contradiction": p[0],
                    "entailment": p[1],
                    "neutral": p[2],
                    "aligned_evidence": b[2],
                    "untruncated_tokens": length,
                    "truncated": length > 384,
                    "cache_key": b[3],
                }
                b[4].write_text(json.dumps(item), encoding="utf-8")
                result[b[0]] = item
        return result

    def assess(
        self, claims: list[dict[str, Any]], evidence: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Attach fallible supported/contradicted/insufficient evidence judgments.

        Missing citations return unreviewed. High-confidence entailment or
        contradiction returns the corresponding automatic status; all other
        cases abstain. Every result contains exact cited source material and
        human_review='pending'. No benchmark reference answer is consulted.
        """
        pairs = [
            (c["text"], " ".join(evidence[e]["text"] for e in c["evidence_ids"] if e in evidence))
            for c in claims
        ]
        probabilities = self.probabilities(pairs)
        output = []
        for c, (_, ev), p in zip(claims, pairs, probabilities):
            status = "insufficient_evidence"
            if not ev.strip():
                status = "unreviewed"
            elif p["entailment"] >= self.threshold:
                status = "supported"
            elif p["contradiction"] >= self.threshold:
                status = "contradicted"
            output.append(
                dict(
                    p,
                    claim_id=c["id"],
                    text=c["text"],
                    status=status,
                    evidence_ids=c["evidence_ids"],
                    evidence=[dict(evidence[e], id=e) for e in c["evidence_ids"] if e in evidence],
                    hallucination_score=p["contradiction"],
                    predicted_classification=status,
                    uncertainty=1 - max(p["contradiction"], p["entailment"], p["neutral"]),
                    calibrated_probability=None,
                    calibration_status="Not calibrated for HotpotQA three-way factual status",
                    model_id=self.model_id,
                    judgment_origin="automatic_nli_proxy",
                    human_review="pending",
                )
            )
        return output

    def embeddings(self, texts: list[str]) -> np.ndarray:
        """Return frozen mean-pooled contextual embeddings for semantic regression.

        Input strings combine claim/question/evidence selected before actions.
        Padding is masked; [B,T,H] last hidden states become [B,H] vectors. These
        are generic NLI-backbone representations, not tuned sentence embeddings.
        No outcomes or reference answers are embedded. CPU numpy output supports
        small tabular regressors without keeping an autograd graph in GPU memory.
        """
        arrays = []
        for start in range(0, len(texts), 16):
            batch = self.tokenizer(
                texts[start : start + 16],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                states = self.model.base_model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            arrays.append(pooled.cpu().numpy())
        return (
            np.concatenate(arrays)
            if arrays
            else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        )
