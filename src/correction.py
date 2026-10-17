"""
correction.py
Evidence-Guided Claim Correction and Post-Correction Re-Verification.

Rigorous Operational Metric Definitions:
1. Correction Success Rate (CSR):
   Numerator: Number of ground-truth hallucinated claims (y=1) targeted and corrected into verifiable factual claims (post_status='Accepted-Verified').
   Denominator: Total ground-truth hallucinated claims targeted (N_hal).
   CSR = (N_hal_corrected_and_verified) / N_hal

2. Residual Hallucination Rate (RHR):
   Numerator: Number of ground-truth hallucinated claims that remain contradicted, unverified, or failed post-correction.
   Denominator: Total ground-truth hallucinated claims targeted (N_hal).
   RHR = 1.0 - CSR

3. Correction-Induced Hallucination Rate (CIHR):
   Numerator: Number of originally factual claims (y=0) that were mistakenly edited and became hallucinated post-edit.
   Denominator: Total originally factual claims (N_fac).
   CIHR = (N_fac_corrupted) / N_fac

4. Correct-Content Preservation Rate (CCPR):
   Numerator: Number of factual claims (y=0) preserved intact without modification.
   Denominator: Total originally factual claims (N_fac).
   CCPR = (N_fac_unmodified) / N_fac

5. Unnecessary Edit Rate (UER):
   Numerator: Number of factual claims (y=0) erroneously modified by the corrector.
   Denominator: Total originally factual claims (N_fac).
   UER = (N_fac_modified) / N_fac = 1.0 - CCPR

6. Abstention Rate (AR):
   Numerator: Claims where the system abstained from generating an ungrounded edit due to absent or inadequate evidence.
   Denominator: Total evaluated claims (N_total).
   AR = (N_abstained) / N_total

7. Verification Acceptance Rate (VAR):
   Numerator: Number of proposed corrections that passed post-correction verification.
   Denominator: Total number of proposed corrections edited.
   VAR = (N_edits_accepted) / N_edits
"""

import os
import re
import difflib
import numpy as np
import pandas as pd


class EvidenceGuidedCorrector:
    """
    Evidence-Guided Targeted Claim Corrector with Post-Correction Verification.
    Preserves factual claims intact, repairs factual errors using grounded evidence,
    and abstains from hallucinating replacement content when evidence is insufficient.
    """
    def __init__(self, verification_engine=None):
        self.engine = verification_engine

    def extract_salient_evidence_sentence(self, evidence, claim):
        """Find the sentence in evidence with highest lexical relevance to the claim."""
        ev_str = str(evidence).strip()
        if not ev_str:
            return ""

        ev_sentences = re.split(r'(?<=[.!?])\s+', ev_str)
        if not ev_sentences:
            return ev_str[:150]

        cl_words = set(re.findall(r'\b[a-z]{3,}\b', str(claim).lower()))
        best_sent = ev_sentences[0]
        max_overlap = -1

        for s in ev_sentences:
            s_words = set(re.findall(r'\b[a-z]{3,}\b', s.lower()))
            overlap = len(cl_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sent = s

        return best_sent.strip()

    def align_numerical_entities(self, claim, evidence):
        """
        Aligns numbers in claim with numbers in evidence based on local context windows.
        Returns:
            aligned_replacements: dict of {old_number: new_number} if unambiguous
            is_ambiguous: bool
        """
        cl_nums = re.findall(r'\b\d+(?:\.\d+)?\b', str(claim))
        ev_nums = re.findall(r'\b\d+(?:\.\d+)?\b', str(evidence))

        if not cl_nums or not ev_nums:
            return {}, True

        # Find numbers in claim that are not supported in evidence
        mismatched_cl_nums = [n for n in cl_nums if n not in ev_nums]
        if not mismatched_cl_nums:
            return {}, False  # All numbers are already supported

        replacements = {}
        stopwords = {'the', 'and', 'was', 'were', 'that', 'with', 'from', 'this', 'for', 'are', 'has', 'had', 'have', 'its', 'their', 'not', 'into'}
        for old_n in mismatched_cl_nums:
            # Extract local window around old_n in claim (up to 4 tokens on each side)
            pattern = rf'((?:\S+\s+){{0,4}}){re.escape(old_n)}((?:\s+\S+){{0,4}})'
            match = re.search(pattern, claim)
            if not match:
                continue
            cl_ctx_words = set(w for w in re.findall(r'\b[a-z]{3,}\b', (match.group(1) + " " + match.group(2)).lower()) if w not in stopwords)

            candidate_scores = []
            for cand_n in set(ev_nums):
                # Search occurrences of cand_n in evidence
                ev_matches = re.finditer(rf'((?:\S+\s+){{0,4}}){re.escape(cand_n)}((?:\s+\S+){{0,4}})', evidence)
                best_cand_overlap = -1
                for ev_m in ev_matches:
                    ev_ctx_words = set(w for w in re.findall(r'\b[a-z]{3,}\b', (ev_m.group(1) + " " + ev_m.group(2)).lower()) if w not in stopwords)
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

    def correct_claim(self, claim, evidence):
        """
        Generates an evidence-grounded revision for a targeted claim.
        Returns:
            corrected_claim: str
            abstained: bool
            reason: str
        """
        cl = str(claim).strip()
        ev = str(evidence).strip()

        # 1. Abstention check: If evidence is missing or trivial, DO NOT invent content
        if not ev or len(ev.split()) < 3:
            return cl, True, "Abstained: Insufficient evidence to formulate grounded correction."

        # 2. Context-aligned numerical / temporal entity contradiction repair
        cl_nums = re.findall(r'\b\d+(?:\.\d+)?\b', cl)
        ev_nums = re.findall(r'\b\d+(?:\.\d+)?\b', ev)
        mismatched_cl = [n for n in cl_nums if n not in ev_nums]

        if mismatched_cl:
            replacements, is_ambiguous = self.align_numerical_entities(cl, ev)
            if is_ambiguous or not replacements:
                return cl, True, "Abstained: Ambiguous numerical entity context alignment."

            corrected = cl
            for old_n, new_n in replacements.items():
                corrected = re.sub(rf'\b{re.escape(old_n)}\b', new_n, corrected, count=1)
            return corrected, False, "Context-aligned numerical entity correction from evidence."

        # 3. Grounded clause substitution
        best_ev_clause = self.extract_salient_evidence_sentence(ev, cl)
        cl_words = set(re.findall(r'\b[a-z]{3,}\b', cl.lower()))
        ev_words = set(re.findall(r'\b[a-z]{3,}\b', best_ev_clause.lower()))
        overlap_ratio = len(cl_words.intersection(ev_words)) / max(len(cl_words), 1)

        # If evidence clause has negligible semantic relevance to the claim, abstain
        if overlap_ratio < 0.10:
            return cl, True, "Abstained: Evidence lacks semantic alignment with claim."

        if len(best_ev_clause.split()) >= 4:
            corrected = best_ev_clause
            if not corrected.endswith('.'):
                corrected += '.'
            return corrected, False, "Salient evidence clause substitution."

        return cl, True, "Abstained: Unable to isolate reliable factual assertion."

    def correct_and_reverify(self, claim, evidence, is_detected_hallucination=True):
        """
        Executes full cycle:
        1. Targeted claim revision (if flagged).
        2. Post-Correction Re-verification against evidence.
        """
        if not is_detected_hallucination:
            # Preserved intact as factual content
            return {
                'original_claim': claim,
                'corrected_claim': claim,
                'was_edited': False,
                'abstained': False,
                'post_verification_status': 'Accepted-Unmodified',
                'is_hallucinated_post': 0,
                'edit_reason': 'Unchanged (Detected Factual)'
            }

        # Step 1: Perform targeted evidence-guided correction
        corrected_claim, abstained, reason = self.correct_claim(claim, evidence)

        if abstained or corrected_claim.strip() == claim.strip():
            return {
                'original_claim': claim,
                'corrected_claim': claim,
                'was_edited': False,
                'abstained': abstained,
                'post_verification_status': 'Abstain-NoReplacement' if abstained else 'Unmodified',
                'is_hallucinated_post': 1,
                'edit_reason': reason
            }

        # Step 2: Post-Correction Re-verification Stage
        if self.engine is not None:
            prob_hal, _ = self.engine.verify_check_1(evidence, corrected_claim)
            prob_cov, _ = self.engine.verify_check_2(evidence, corrected_claim)
            final_prob = (prob_hal + prob_cov) / 2.0
            is_still_hal = 1 if final_prob >= 0.50 else 0
        else:
            ev_words = set(re.findall(r'\b[a-z]{3,}\b', evidence.lower()))
            cor_words = set(re.findall(r'\b[a-z]{3,}\b', corrected_claim.lower()))
            overlap = len(cor_words.intersection(ev_words)) / max(len(cor_words), 1)
            is_still_hal = 0 if overlap >= 0.40 else 1

        status = 'Accepted-Verified' if is_still_hal == 0 else 'Rejected-PostVerificationFailed'

        return {
            'original_claim': claim,
            'corrected_claim': corrected_claim,
            'was_edited': True,
            'abstained': False,
            'post_verification_status': status,
            'is_hallucinated_post': is_still_hal,
            'edit_reason': reason
        }


# ------------------------------------------------------------------------------
# 5. FORMAL CORRECTION EVALUATION PROTOCOL
# ------------------------------------------------------------------------------
def evaluate_correction_pipeline(corrector, test_records, max_eval=400):
    """
    Evaluates correction and post-verification under exact mathematical definitions:
    - CSR: Correction Success Rate
    - RHR: Residual Hallucination Rate
    - CIHR: Correction-Induced Hallucination Rate
    - CCPR: Correct Content Preservation Rate
    - UER: Unnecessary Edit Rate
    - AR: Abstention Rate
    - VAR: Verification Acceptance Rate
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
        cl = item['claim']
        ev = item['evidence']
        lbl = item['label']  # 1 = Hallucinated, 0 = Factual (strictly for ground truth evaluation)

        # STRICT ZERO-LEAKAGE ASSERTION:
        # Decisions to correct MUST be governed solely by genuine detector predictions.
        # The gold label 'lbl' is strictly sequestered for post-hoc evaluation and NEVER controls edits.
        if 'prediction' in item and item['prediction'] is not None:
            pred = item['prediction']
        elif corrector.engine is not None:
            # Genuine out-of-sample prediction via verification engine without access to gold label
            ver_res = corrector.engine.adaptive_verify(ev, cl)
            pred = ver_res.get('prediction')
        else:
            raise KeyError(
                "Record is missing 'prediction' and corrector has no verification engine. "
                "Gold labels cannot be used for inference or correction decisions."
            )

        if pred is None:
            # Explicit abstention: detector produced no decisive prediction (abstained)
            rec = {
                'original_claim': cl,
                'corrected_claim': cl,
                'was_edited': False,
                'abstained': True,
                'post_verification_status': 'Abstain-NoDetectorPrediction',
                'is_hallucinated_post': 1 if lbl == 1 else 0,
                'edit_reason': 'Abstained: Detector produced no decisive prediction'
            }
        else:
            rec = corrector.correct_and_reverify(cl, ev, is_detected_hallucination=(pred == 1))

        rec['ground_truth'] = lbl
        rec['detector_prediction'] = pred
        detailed_results.append(rec)

        if rec['abstained']:
            total_abstained += 1

        if rec['was_edited']:
            total_edited += 1
            if rec['post_verification_status'] == 'Accepted-Verified':
                edits_accepted += 1

        if lbl == 1:
            n_hal += 1
            if rec['was_edited'] and rec['post_verification_status'] == 'Accepted-Verified':
                n_hal_success += 1
            else:
                n_hal_residual += 1
        else:
            n_fac += 1
            if not rec['was_edited']:
                n_fac_preserved += 1
            else:
                n_fac_unnecessary_edits += 1
                if rec['is_hallucinated_post'] == 1:
                    n_fac_corrupted += 1

    csr = (n_hal_success / max(n_hal, 1)) * 100.0
    rhr = (n_hal_residual / max(n_hal, 1)) * 100.0
    cihr = (n_fac_corrupted / max(n_fac, 1)) * 100.0
    ccpr = (n_fac_preserved / max(n_fac, 1)) * 100.0
    uer = (n_fac_unnecessary_edits / max(n_fac, 1)) * 100.0
    ar = (total_abstained / max(len(eval_subset), 1)) * 100.0
    var = (edits_accepted / max(total_edited, 1)) * 100.0

    metrics = {
        'Total Evaluated Claims': len(eval_subset),
        'Targeted Hallucinations (N_hal)': n_hal,
        'Targeted Factual Claims (N_fac)': n_fac,
        'Correction Success Rate (CSR %)': round(csr, 2),
        'Residual Hallucination Rate (RHR %)': round(rhr, 2),
        'Correction-Induced Hallucination Rate (CIHR %)': round(cihr, 2),
        'Correct Content Preservation Rate (CCPR %)': round(ccpr, 2),
        'Unnecessary Edit Rate (UER %)': round(uer, 2),
        'Abstention Rate (AR %)': round(ar, 2),
        'Verification Acceptance Rate (VAR %)': round(var, 2)
    }

    return metrics, detailed_results
