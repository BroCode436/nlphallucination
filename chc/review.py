"""Prepare a human-review sample without filling human judgment fields.

Sampling uses observed outcome strata for coverage, but the worksheet hides
machine correctness/gain labels. A separate key preserves reproducible matching.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
import pandas as pd


def prepare_review(records: list[dict[str, Any]], output: Path, maximum: int = 100, seed: int = 42) -> dict[str, Any]:
    """Sample cases across error/action/gain strata and leave reviewer labels blank.

    Args:
        records: Actually collected episodes with verified singleton outcomes.
        output: Folder for blinded worksheet and private machine-label key.
        maximum: Upper bound on reviewed singleton examples to prepare.
        seed: Deterministic sample order within strata.
    Returns:
        Counts and an explicit pending-review status. This function cannot perform
        or claim independent human annotation.
    """
    rng=random.Random(seed)
    groups: dict[tuple[str,str,int],list[tuple[dict[str,Any],str]]] = {}
    for record in records:
        for node,result in record['singletons'].items():
            action=record['verification_actions'][node]
            gain=int(result['correct'])-int(record['baseline']['correct'])
            groups.setdefault((record['injection']['type'],action['status'],gain),[]).append((record,node))
    for values in groups.values():rng.shuffle(values)
    selected=[]
    # Round-robin coverage avoids allowing the dominant zero-effect stratum to
    # crowd out rare harmful or helpful interventions from the review worksheet.
    while len(selected)<maximum:
        progress=False
        for key in sorted(groups):
            if groups[key] and len(selected)<maximum:
                selected.append(groups[key].pop());progress=True
        if not progress:break
    worksheet,key_rows=[],[]
    for index,(record,node) in enumerate(selected):
        review_id=f'review-{index:03d}'
        action=record['verification_actions'][node]
        worksheet.append({'review_id':review_id,'question':record['question'],'reference_answer':record['reference_answer'],
                          'original_note':action['original'],'proposed_or_retained_note':action['replacement'],
                          'evidence':action['evidence'],'before_answer':record['baseline']['answer'],
                          'after_answer':record['singletons'][node]['answer'],
                          'clean_claim_supported':'','injected_error_valid':'','repair_valid':'',
                          'before_answer_correct':'','after_answer_correct':'','dependency_notes':'','reviewer':'','comments':''})
        key_rows.append({'review_id':review_id,'episode_id':record['episode_id'],'claim_id':node,
                         'machine_before_correct':record['baseline']['correct'],
                         'machine_after_correct':record['singletons'][node]['correct'],'split':record['split']})
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(worksheet).to_csv(output/'human_review_blinded.csv',index=False)
    pd.DataFrame(key_rows).to_csv(output/'human_review_key.csv',index=False)
    report={'prepared_examples':len(worksheet),'review_completed':False,'seed':seed,
            'sampling':'Round-robin error type/action status/observed gain strata; not prevalence-weighted'}
    (output/'review_status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
