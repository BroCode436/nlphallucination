"""
Evidence-guided correction with proposal rollback and post-verification.

Automatic correction success is a verifier-acceptance proxy, not independent
factual truth. The legacy CSR/RHR keys remain for compatibility. Independent
CSR and CIHR require human labels via review_corrections.py; CIHR is null here.
CCPR/UER measure actual final-output edits. VAR counts accepted proposals over
all proposals, including rejected proposals that were rolled back.
"""

from typing import Any as _DocAny

import os
import re
import difflib
import numpy as np
import pandas as pd
from grounding import clean_text, numerical_conflicts


class EvidenceGuidedCorrector:
    """Coordinate evidence-based proposals, verification and rollback.

    Constructor dependencies are an optional verification engine and optional
    trained validity model. Attributes engine and validity_model keep those objects.
    Methods retrieve a relevant source clause, align numbers, propose a minimal or
    source-based edit, then reverify before returning the final text. A failed check
    retains the original claim and records the rejected proposal. This retained
    component supports the original correction application; V2 wraps the saved
    validity model separately for controlled propagation experiments.
    """

    def __init__(self, verification_engine: _DocAny = None, validity_model: _DocAny = None) -> None:
        """Initialize correction with an optional detector and saved validity verifier.

        verification_engine supplies adaptive detector checks when available.
        validity_model may be injected for testing; otherwise a local trained verifier
        is loaded only if its artifact exists. Attributes engine and validity_model
        control later proposal acceptance. Missing optional weights do not create fake
        confidence scores; the fallback path remains explicitly heuristic.
        """
        self.engine = verification_engine
        self.validity_model = validity_model
        if self.validity_model is None:
            from pathlib import Path

            model_path = (
                Path(__file__).resolve().parents[1] / "models" / "correction_validity.joblib"
            )
            if model_path.exists():
                from correction_validity import CorrectionValidityModel

                self.validity_model = CorrectionValidityModel(model_path)

    def extract_salient_evidence_sentence(self, evidence: _DocAny, claim: _DocAny) -> _DocAny:
        """Choose the source sentence sharing the most content words with a claim.

        Inputs are evidence and claim strings. English content words of at least three
        letters form sets; sentence overlap determines the selected original sentence.
        The first source sentence breaks ties. Empty evidence returns an empty string.
        This retrieval heuristic does not establish semantic support and may miss a
        paraphrase; post-verification must assess the resulting proposal.
        """
        ev_str = str(evidence).strip()
        if not ev_str:
            return ""

        ev_sentences = re.split(r"(?<=[.!?])\s+", ev_str)
        if not ev_sentences:
            return ev_str[:150]

        cl_words = set(re.findall(r"\b[a-z]{3,}\b", str(claim).lower()))
        best_sent = ev_sentences[0]
        max_overlap = -1

        for s in ev_sentences:
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s.lower()))
            overlap = len(cl_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sent = s

        return best_sent.strip()

    def align_numerical_entities(self, claim: _DocAny, evidence: _DocAny) -> _DocAny:
        """Propose number replacements using nearby lexical context.

        Inputs are claim and source text. For each claim number missing from evidence,
        the function compares up to four surrounding words against evidence numbers.
        A unique positive-overlap candidate is required. Returns (replacement_map,
        ambiguous_flag); ties or missing contextual support abstain. This reduces
        cross-entity number swaps but cannot certify semantic binding.
        """
        cl_nums = re.findall(r"\b\d+(?:\.\d+)?\b", str(claim))
        ev_nums = re.findall(r"\b\d+(?:\.\d+)?\b", str(evidence))

        if not cl_nums or not ev_nums:
            return {}, True

        # Find numbers in claim that are not supported in evidence
        mismatched_cl_nums = [n for n in cl_nums if n not in ev_nums]
        if not mismatched_cl_nums:
            return {}, False  # All numbers are already supported

        replacements = {}
        stopwords = {
            "the",
            "and",
            "was",
            "were",
            "that",
            "with",
            "from",
            "this",
            "for",
            "are",
            "has",
            "had",
            "have",
            "its",
            "their",
            "not",
            "into",
        }
        for old_n in mismatched_cl_nums:
            # Extract local window around old_n in claim (up to 4 tokens on each side)
            pattern = rf"((?:\S+\s+){{0,4}}){re.escape(old_n)}((?:\s+\S+){{0,4}})"
            match = re.search(pattern, claim)
            if not match:
                continue
            cl_ctx_words = set(
                w
                for w in re.findall(
                    r"\b[a-z]{3,}\b", (match.group(1) + " " + match.group(2)).lower()
                )
                if w not in stopwords
            )

            candidate_scores = []
            for cand_n in set(ev_nums):
                # Search occurrences of cand_n in evidence
                ev_matches = re.finditer(
                    rf"((?:\S+\s+){{0,4}}){re.escape(cand_n)}((?:\s+\S+){{0,4}})", evidence
                )
                best_cand_overlap = -1
                for ev_m in ev_matches:
                    ev_ctx_words = set(
                        w
                        for w in re.findall(
                            r"\b[a-z]{3,}\b", (ev_m.group(1) + " " + ev_m.group(2)).lower()
                        )
                        if w not in stopwords
                    )
                    overlap = len(cl_ctx_words.intersection(ev_ctx_words))
                    if overlap > best_cand_overlap:
                        best_cand_overlap = overlap
                candidate_scores.append((cand_n, best_cand_overlap))

            # Filter candidates with positive context overlap
            candidate_scores = [c for c in candidate_scores if c[1] > 0]
            if not candidate_scores:
                # No contextual grounding for this number -> Ambiguous
                return {}, True

            # Sort by score descending
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_cand, best_score = candidate_scores[0]

            # Check for ambiguity (tie with second best candidate)
            if len(candidate_scores) > 1 and candidate_scores[1][1] == best_score:
                return {}, True  # Ambiguous tie

            replacements[old_n] = best_cand

        return replacements, False

    def correct_claim(self, claim: _DocAny, evidence: _DocAny) -> _DocAny:
        """Generate a repair candidate or an explicit abstention.

        Inputs are claim/evidence text. Missing or very short evidence abstains.
        Unsupported numeric values use context-aligned replacements; ambiguity retains
        the original. Otherwise the most overlapping evidence clause is considered
        only when it has adequate overlap and length. Returns (candidate, abstained,
        reason). This is proposal generation, not final correction approval.
        """
        cl = clean_text(claim)
        ev = clean_text(evidence)

        # 1. Abstention check: If evidence is missing or trivial, DO NOT invent content
        if not ev or len(ev.split()) < 3:
            return cl, True, "Abstained: Insufficient evidence to formulate grounded correction."

        # 2. Context-aligned numerical / temporal entity contradiction repair
        cl_nums = re.findall(r"\b\d+(?:\.\d+)?\b", cl)
        ev_nums = re.findall(r"\b\d+(?:\.\d+)?\b", ev)
        mismatched_cl = [n for n in cl_nums if n not in ev_nums]

        if mismatched_cl:
            replacements, is_ambiguous = self.align_numerical_entities(cl, ev)
            if is_ambiguous or not replacements:
                return cl, True, "Abstained: Ambiguous numerical entity context alignment."

            pattern = (
                r"(?<![\w.])(?:"
                + "|".join(re.escape(n) for n in sorted(replacements, key=len, reverse=True))
                + r")(?!\w|\.\d)"
            )
            corrected = re.sub(pattern, lambda match: replacements[match.group()], cl)
            return corrected, False, "Context-aligned numerical entity correction from evidence."

        # 3. Grounded clause substitution
        best_ev_clause = self.extract_salient_evidence_sentence(ev, cl)
        cl_words = set(re.findall(r"\b[a-z]{3,}\b", cl.lower()))
        ev_words = set(re.findall(r"\b[a-z]{3,}\b", best_ev_clause.lower()))
        overlap_ratio = len(cl_words.intersection(ev_words)) / max(len(cl_words), 1)

        # If evidence clause has negligible semantic relevance to the claim, abstain
        if overlap_ratio < 0.10:
            return cl, True, "Abstained: Evidence lacks semantic alignment with claim."

        if len(best_ev_clause.split()) >= 4:
            corrected = best_ev_clause
            if not corrected.endswith("."):
                corrected += "."
            return corrected, False, "Salient evidence clause substitution."

        return cl, True, "Abstained: Unable to isolate reliable factual assertion."

    def correct_and_reverify(
        self, claim: _DocAny, evidence: _DocAny, is_detected_hallucination: bool = True
    ) -> _DocAny:
        """Apply detection-gated correction with proposal rollback on failed checks.

        Inputs include original claim, evidence and a detector decision supplied by
        inference, never a gold label. Factual predictions preserve the claim. A repair
        proposal is rechecked through the optional three detector checks and numeric
        consistency, or an explicit lexical fallback; the validity verifier can veto it.
        Output distinguishes proposed text from final corrected text and counts actual
        post-verification calls. Rejection returns the original, preserving reviewable
        reasons. Verifier acceptance is not independent factual correctness.
        """
        if not is_detected_hallucination:
            # Preserved intact as factual content
            return {
                "original_claim": claim,
                "corrected_claim": claim,
                "was_edited": False,
                "abstained": False,
                "post_verification_status": "Accepted-Unmodified",
                "is_hallucinated_post": None,
                "edit_reason": "Unchanged (Detected Factual)",
            }

        # Step 1: Perform targeted evidence-guided correction
        corrected_claim, abstained, reason = self.correct_claim(claim, evidence)

        if abstained or corrected_claim.strip() == claim.strip():
            return {
                "original_claim": claim,
                "corrected_claim": claim,
                "was_edited": False,
                "abstained": abstained,
                "post_verification_status": "Abstain-NoReplacement" if abstained else "Unmodified",
                "is_hallucinated_post": 1,
                "edit_reason": reason,
            }

        # Step 2: Post-Correction Re-verification Stage
        if self.engine is not None:
            prob_hal, _ = self.engine.verify_check_1(evidence, corrected_claim)
            prob_cov, _ = self.engine.verify_check_2(evidence, corrected_claim)
            prob_numeric, _ = self.engine.verify_check_3(evidence, corrected_claim)
            numeric_uncertain = any(
                r["ambiguous"] for r in numerical_conflicts(evidence, corrected_claim)
            )
            is_still_hal = (
                1
                if prob_hal >= 0.50 or prob_cov >= 0.50 or prob_numeric >= 0.80 or numeric_uncertain
                else 0
            )
        else:
            ev_words = set(re.findall(r"\b[a-z]{3,}\b", evidence.lower()))
            cor_words = set(re.findall(r"\b[a-z]{3,}\b", corrected_claim.lower()))
            overlap = len(cor_words.intersection(ev_words)) / max(len(cor_words), 1)
            conflict = any(
                item["conflict"] for item in numerical_conflicts(evidence, corrected_claim)
            )
            is_still_hal = 0 if overlap >= 0.40 and not conflict else 1

        validity = None
        if self.validity_model is not None:
            validity = self.validity_model.assess(evidence, claim, corrected_claim)
            if not validity["accepted"]:
                is_still_hal = 1

        status = "Accepted-Verified" if is_still_hal == 0 else "Rejected-PostVerificationFailed"

        return {
            "original_claim": claim,
            "corrected_claim": corrected_claim if is_still_hal == 0 else claim,
            "proposed_claim": corrected_claim,
            "correction_validity": validity,
            "post_verification_calls": 3 if self.engine is not None else 0,
            "proposal_rejected": bool(is_still_hal),
            "was_edited": is_still_hal == 0,
            "abstained": False,
            "post_verification_status": status,
            "is_hallucinated_post": is_still_hal,
            "edit_reason": reason,
        }


# ------------------------------------------------------------------------------
# 5. FORMAL CORRECTION EVALUATION PROTOCOL
# ------------------------------------------------------------------------------
def evaluate_correction_pipeline(
    corrector: _DocAny, test_records: _DocAny, max_eval: int = 400
) -> _DocAny:
    """Evaluate actual detector-driven correction decisions with explicit proxy limits.

    Inputs are a corrector, labeled test records and a bounded evaluation count.
    Each record must provide an inference prediction or an available detector; gold
    labels cannot substitute for missing predictions. The corrector runs before the
    label is used for aggregation. Outputs are a metric dictionary and detailed
    records with proposals, rollback, calls and labels. Acceptance-based correction
    rates are marked proxies. Independent factual correction success and induced
    hallucination rates remain null without reviewed corrected-claim labels.
    """
    eval_subset = test_records[:max_eval]

    n_hal = 0
    n_hal_success = 0
    n_hal_residual = 0

    n_fac = 0
    n_fac_preserved = 0
    n_fac_corrupted = 0
    n_fac_unnecessary_edits = 0

    total_abstained = 0
    total_edited = 0
    edits_accepted = 0

    detailed_results = []

    for item in eval_subset:
        cl = item["claim"]
        ev = item["evidence"]
        lbl = item["label"]  # 1 = Hallucinated, 0 = Factual (strictly for ground truth evaluation)

        # STRICT ZERO-LEAKAGE ASSERTION:
        # Decisions to correct MUST be governed solely by genuine detector predictions.
        # The gold label 'lbl' is strictly sequestered for post-hoc evaluation and NEVER controls edits.
        detection_calls = 0
        if "prediction" in item and item["prediction"] is not None:
            pred = item["prediction"]
        elif corrector.engine is not None:
            # Genuine out-of-sample prediction via verification engine without access to gold label
            ver_res = corrector.engine.adaptive_verify(ev, cl)
            pred = ver_res.get("prediction")
            detection_calls = ver_res.get("verification_calls", 0)
        else:
            raise KeyError(
                "Record is missing 'prediction' and corrector has no verification engine. "
                "Gold labels cannot be used for inference or correction decisions."
            )

        if pred is None:
            # Explicit abstention: detector produced no decisive prediction (abstained)
            rec = {
                "original_claim": cl,
                "corrected_claim": cl,
                "was_edited": False,
                "abstained": True,
                "post_verification_status": "Abstain-NoDetectorPrediction",
                "is_hallucinated_post": 1 if lbl == 1 else 0,
                "edit_reason": "Abstained: Detector produced no decisive prediction",
            }
        else:
            rec = corrector.correct_and_reverify(cl, ev, is_detected_hallucination=(pred == 1))

        rec["detector_verification_calls"] = detection_calls
        rec["evidence"] = ev
        rec["ground_truth"] = lbl
        rec["detector_prediction"] = pred
        detailed_results.append(rec)

        if rec["abstained"]:
            total_abstained += 1

        if rec["was_edited"] or rec.get("proposal_rejected", False):
            total_edited += 1
            if rec["post_verification_status"] == "Accepted-Verified":
                edits_accepted += 1

        if lbl == 1:
            n_hal += 1
            if rec["was_edited"] and rec["post_verification_status"] == "Accepted-Verified":
                n_hal_success += 1
            else:
                n_hal_residual += 1
        else:
            n_fac += 1
            if not rec["was_edited"]:
                n_fac_preserved += 1
            else:
                n_fac_unnecessary_edits += 1
                if rec["is_hallucinated_post"] == 1:
                    n_fac_corrupted += 1

    csr = (n_hal_success / max(n_hal, 1)) * 100.0
    rhr = (n_hal_residual / max(n_hal, 1)) * 100.0
    cihr = (n_fac_corrupted / max(n_fac, 1)) * 100.0
    ccpr = (n_fac_preserved / max(n_fac, 1)) * 100.0
    uer = (n_fac_unnecessary_edits / max(n_fac, 1)) * 100.0
    ar = (total_abstained / max(len(eval_subset), 1)) * 100.0
    var = (edits_accepted / max(total_edited, 1)) * 100.0

    metrics = {
        "Evaluation Basis": "Verifier acceptance proxy; corrected claims lack independent gold labels",
        "Independent Factual Safety Evaluated": False,
        "Mean Total Verification Calls": round(
            sum(
                r.get("detector_verification_calls", 0) + r.get("post_verification_calls", 0)
                for r in detailed_results
            )
            / max(len(detailed_results), 1),
            3,
        ),
        "Rejected Proposals": sum(r.get("proposal_rejected", False) for r in detailed_results),
        "Total Evaluated Claims": len(eval_subset),
        "Targeted Hallucinations (N_hal)": n_hal,
        "Targeted Factual Claims (N_fac)": n_fac,
        "Verifier-Accepted Correction Rate (Proxy %)": round(csr, 2),
        "Correction Success Rate (CSR %)": round(csr, 2),  # Legacy alias: proxy only
        "Independent Correction Success Rate (%)": None,
        "Residual Hallucination Rate (RHR %)": round(rhr, 2),
        "Correction-Induced Hallucination Rate (CIHR %)": None,
        "Rejected Proposal Rate (%)": round(
            sum(r.get("proposal_rejected", False) for r in detailed_results)
            / max(total_edited, 1)
            * 100,
            2,
        ),
        "Correct Content Preservation Rate (CCPR %)": round(ccpr, 2),
        "Unnecessary Edit Rate (UER %)": round(uer, 2),
        "Abstention Rate (AR %)": round(ar, 2),
        "Verification Acceptance Rate (VAR %)": round(var, 2),
    }

    return metrics, detailed_results
