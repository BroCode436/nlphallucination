"""Inspect a claim locally and emit an auditable JSON explanation.

python analyze_claim.py --claim "Alice scored 20 points." --evidence "Alice scored 10 points; Bob scored 20 points."
Use --model baseline or transformer for saved, local-only checkpoints.
"""
from typing import Any as _DocAny
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from grounding import clean_text, content_tokens, numerical_conflicts
from verification import VerificationEngine, compute_evidence_quality
from correction import EvidenceGuidedCorrector


def inspect_claim(engine: _DocAny, evidence: _DocAny, claim: _DocAny, model_name: str='heuristic') -> _DocAny:
    """Inspect claim.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            engine: Verification engine containing the selected detector and heuristic checks.
            evidence: Source passage supplied to the verifier; presence does not establish source truth.
            claim: Statement being checked against evidence.
            model_name: Model name supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    evidence, claim = clean_text(evidence), clean_text(claim)
    if not claim:
        raise ValueError('Claim must not be empty')
    category, score = compute_evidence_quality(evidence, claim)
    result = engine.adaptive_verify(evidence, claim)
    correction = None
    if result['prediction'] == 1:
        correction = EvidenceGuidedCorrector(engine).correct_and_reverify(claim, evidence)
    return {'claim': claim, 'model': model_name,
            'assessment_basis': 'Detector and lexical/numerical heuristics; not independently established factual truth',
            'evidence_quality': {'category': category, 'score': score},
            'missing_content_tokens': sorted(content_tokens(claim) - content_tokens(evidence)),
            'numerical_checks': numerical_conflicts(evidence, claim),
            'verification': result, 'correction': correction,
            'needs_review': result['prediction'] != 0 or any(r['conflict'] or r['ambiguous'] for r in numerical_conflicts(evidence, claim)) or category in ('Weak', 'Irrelevant', 'No Evidence')}


def main() -> None:
    """Main.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            None; effects or assertions are performed directly.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim', required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--evidence')
    evidence.add_argument('--evidence-file', type=Path)
    parser.add_argument('--model', choices=('heuristic', 'baseline', 'transformer'), default='heuristic')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    options = {}
    checkpoints = ROOT / 'results' / 'checkpoints'
    if args.model == 'baseline':
        import joblib
        options = {'primary_model': joblib.load(checkpoints / 'best_baseline_clf.joblib'),
                   'vectorizer': joblib.load(checkpoints / 'best_baseline_vec.joblib')}
    elif args.model == 'transformer':
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = checkpoints / 'best_transformer_detector'
        options = {'transformer_model': AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True),
                   'transformer_tokenizer': AutoTokenizer.from_pretrained(path, local_files_only=True)}
    text = args.evidence_file.read_text(encoding='utf-8') if args.evidence_file else args.evidence
    report = inspect_claim(VerificationEngine(**options), text, args.claim, args.model)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
