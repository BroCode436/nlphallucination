"""Audit saved claim splits, including repeated evidence under different IDs.

python audit_dataset.py --output results/tables/dataset_audit.json
Exit 1 signals source overlap or invalid labels. Repeated evidence is a warning
requiring inspection, since generic/short passages can repeat legitimately.
"""
from typing import Any as _DocAny
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def audit_splits(frames: _DocAny) -> _DocAny:
    """Audit splits.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            frames: Frames supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    report = {'splits': {}, 'source_overlaps': {}, 'evidence_fingerprint_overlaps': {}}
    sources, fingerprints = {}, {}
    for name, frame in frames.items():
        required = {'source_id', 'claim', 'evidence', 'label'}
        if not required.issubset(frame.columns):
            raise ValueError(f'{name}: missing columns {sorted(required - set(frame.columns))}')
        sources[name] = set(frame['source_id'])
        texts = frame['evidence'].fillna('').astype(str).str.split().str.join(' ').str.lower()
        fingerprints[name] = {hashlib.sha256(text.encode()).hexdigest() for text in texts if text}
        report['splits'][name] = {'claims': len(frame), 'sources': len(sources[name]),
                                 'hallucinations': int(frame['label'].eq(1).sum()),
                                 'invalid_labels': int((~frame['label'].isin([0, 1])).sum()),
                                 'blank_evidence': int(texts.eq('').sum()),
                                 'blank_claims': int(frame['claim'].fillna('').astype(str).str.strip().eq('').sum())}
    names = list(frames)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            key = f'{first}/{second}'
            report['source_overlaps'][key] = len(sources[first] & sources[second])
            report['evidence_fingerprint_overlaps'][key] = len(fingerprints[first] & fingerprints[second])
    report['passed'] = not any(report['source_overlaps'].values()) and not any(r['invalid_labels'] for r in report['splits'].values())
    report['evidence_overlap_requires_review'] = any(report['evidence_fingerprint_overlaps'].values())
    return report


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
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frames = {name: pd.read_csv(args.data_dir / f'ragtruth_claims_{name}.csv', keep_default_na=False)
              for name in ('train', 'val', 'test')}
    report = audit_splits(frames)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
