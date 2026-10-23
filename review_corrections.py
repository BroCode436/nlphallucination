"""Export a human-review worksheet or score a completed worksheet.

python review_corrections.py prepare --input results/predictions/correction_predictions.csv --output review.csv
python review_corrections.py score --input review.csv --output independent_metrics.json
Reviewers assign independent_post_label: 0 supported, 1 hallucinated, 2 uncertain.
"""
from typing import Any as _DocAny
import argparse
import csv
import json
from pathlib import Path


def score_reviews(rows: _DocAny) -> _DocAny:
    """Score reviews.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            rows: Rows supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    if not rows:
        raise ValueError('Review worksheet is empty')
    for row in rows:
        if row.get('independent_post_label', '') not in ('0', '1', '2') or not row.get('reviewer', '').strip():
            raise ValueError('Every row needs a reviewer and independent_post_label (0, 1, or 2)')
        if str(row.get('ground_truth')) not in ('0', '1'):
            raise ValueError('Every row needs binary original ground_truth')
    hal = [r for r in rows if str(r['ground_truth']) == '1']
    fac = [r for r in rows if str(r['ground_truth']) == '0']
    edited = lambda r: r['original_claim'].strip() != r['corrected_claim'].strip()
    percent = lambda count, total: round(100 * count / total, 2) if total else None
    return {'Evaluation Basis': 'Independent human labels of final output; uncertain outcomes are not successes',
            'Reviewed Claims': len(rows),
            'Independent CSR (%)': percent(sum(edited(r) and r['independent_post_label'] == '0' for r in hal), len(hal)),
            'Independent CIHR (%)': percent(sum(edited(r) and r['independent_post_label'] == '1' for r in fac), len(fac)),
            'Uncertain Outcomes (%)': percent(sum(r['independent_post_label'] == '2' for r in rows), len(rows)),
            'Original Hallucinated Claims': len(hal), 'Original Factual Claims': len(fac)}


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
    parser.add_argument('action', choices=('prepare', 'score'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Use a separate output file to preserve the input')
    with args.input.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('Input contains no rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'score':
        result = score_reviews(rows)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    # Keep gold labels only in the scoring key; reviewers should hide that column.
    fields = ['review_id', 'evidence', 'original_claim', 'corrected_claim', 'independent_post_label', 'reviewer', 'notes', 'ground_truth']
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({'review_id': index, 'evidence': row.get('evidence', ''),
                             'original_claim': row['original_claim'], 'corrected_claim': row['corrected_claim'],
                             'ground_truth': row['ground_truth']})
    print(f'Prepared {len(rows)} rows. Judge final claims against evidence; hide ground_truth while reviewing.')


if __name__ == '__main__':
    main()
