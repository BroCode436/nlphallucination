"""Derive readable reports and injection diagnostics from completed real runs.

This module does not train policies or select hyperparameters. It preserves
negative results and distinguishes logical policy replay from new model calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .data import read_jsonl
from .metrics import summarize_policies
from .review import prepare_review


def paired_comparison(frame: pd.DataFrame, method: str, baseline: str, seed: int = 42) -> dict[str, Any]:
    """Compare matched one-call policies with question-cluster bootstrap intervals.

    Args:
        frame: Actual held-out policy replay rows, including training/generation seeds.
        method: Learned policy name to compare.
        baseline: Comparator on identical question/condition/seed episodes.
        seed: Bootstrap resampling seed, not a policy-tuning parameter.
    Returns:
        Both condition-weighted and question-balanced differences, with a 95%
        percentile interval for the latter. No p-value or significance claim.
    """
    subset=frame[(frame.budget==1)&frame.method.isin([method,baseline])]
    paired=subset.pivot(index=['episode_id','question_id','training_seed'],columns='method',values='correct').dropna()
    difference=paired[method]-paired[baseline]
    by_question=difference.groupby(level='question_id').mean().to_numpy()
    rng=np.random.default_rng(seed)
    replicates=[rng.choice(by_question,len(by_question),replace=True).mean() for _ in range(2000)]
    return {'method':method,'baseline':baseline,'questions':len(by_question),'paired_rows':len(paired),
            'condition_weighted_difference':float(difference.mean()),'question_balanced_difference':float(by_question.mean()),
            'question_balanced_ci_low':float(np.quantile(replicates,.025)),
            'question_balanced_ci_high':float(np.quantile(replicates,.975)),
            'interpretation':'Descriptive paired question bootstrap; no multiplicity-adjusted significance test'}


def stage_report(stage: Path) -> dict[str, Any]:
    """Summarize a completed stage without assuming any expected row counts.

    Args:
        stage: Run directory containing episodes, predictions, and real checkpoints.
    Returns:
        Actual counts and mean one-check results, also saved as JSON/Markdown.
    Limitations:
        The loss observes the final answer only, and manual review remains pending.
    """
    episode_path=stage/'episodes_with_gate_ablation.jsonl'
    records=read_jsonl(episode_path if episode_path.exists() else stage/'episodes.jsonl')
    frame=pd.read_csv(stage/'all_policy_predictions.csv')
    config=json.loads((stage/'config.json').read_text(encoding='utf-8'))
    pooled=summarize_policies(frame,cost_lambda=config.get('cost_lambda',.01))
    pooled.to_csv(stage/'pooled_policy_summary.csv',index=False)
    comparisons=[paired_comparison(frame,'chc_tabular',baseline) for baseline in ('random','uncertainty','dependency','risk_dependency')]
    comparisons.append(paired_comparison(frame,'chc_graph','uncertainty'))
    pd.DataFrame(comparisons).to_csv(stage/'paired_comparisons.csv',index=False)
    injections=[]
    gains=[]
    keys:set[str]=set()
    truncated:set[str]=set()
    statuses:dict[str,int]={}
    for record in records:
        injections.append({'split':record['split'],'type':record['injection']['type'],
                           'question_id':record['question_id'],'clean_correct':int(record['clean_continuation']['correct']),
                           'corrupted_correct':int(record['baseline']['correct']),
                           'observed_harm':int(record['clean_continuation']['correct'] and not record['baseline']['correct']),
                           'observed_help':int(not record['clean_continuation']['correct'] and record['baseline']['correct']),
                           'identical_output':int(record['injection_ignored_output'])})
        generated=[record['clean_continuation'],record['baseline'],*record['joint_outcomes'].values(),
                   *record['unsafe_singletons'].values(),*record.get('gate_off_singletons',{}).values()]
        for item in generated:
            keys.add(item['cache_key'])
            if item['truncated']:truncated.add(item['cache_key'])
        for node,result in record['singletons'].items():
            status=record['verification_actions'][node]['status']
            statuses[status]=statuses.get(status,0)+1
            gains.append({'split':record['split'],'gain':int(result['correct'])-int(record['baseline']['correct'])})
    injection_frame=pd.DataFrame(injections)
    injection_frame.groupby(['split','type']).agg(episodes=('question_id','size'),questions=('question_id','nunique'),
        clean_accuracy=('clean_correct','mean'),corrupted_accuracy=('corrupted_correct','mean'),
        observed_harm_fraction=('observed_harm','mean'),observed_help_fraction=('observed_help','mean'),
        identical_output_fraction=('identical_output','mean')).to_csv(stage/'injection_effects.csv')
    gain_frame=pd.DataFrame(gains)
    target_counts=gain_frame.groupby(['split','gain']).size().rename('candidate_actions').reset_index()
    target_counts.to_csv(stage/'observed_benefit_distribution.csv',index=False)
    cases=[]
    for desired in (1,-1,0):
        found=False
        for record in records:
            if record['split']!='test':continue
            for node,result in record['singletons'].items():
                gain=int(result['correct'])-int(record['baseline']['correct'])
                if gain==desired:
                    cases.append({'selection_basis':'Post-hoc illustration by observed gain; not a deployable selector',
                                  'episode_id':record['episode_id'],'question':record['question'],
                                  'injected_note':record['injection']['modified'],'action':record['verification_actions'][node],
                                  'before':record['baseline']['answer'],'after':result['answer'],
                                  'reference':record['reference_answer'],'observed_final_gain':gain})
                    found=True;break
            if found:break
    from .data import write_jsonl
    write_jsonl(stage/'case_studies.jsonl',cases)
    review=prepare_review(records,stage/'review')
    one=pooled[pooled.budget==1]
    report={'stage':stage.name,'questions':len({r['question_id'] for r in records}),
            'questions_by_split':{s:len({r['question_id'] for r in records if r['split']==s}) for s in sorted({r['split'] for r in records})},
            'episodes':len(records),'primary_conditions':len({(r['question_id'],r['injection']['type']) for r in records}),
            'distinct_real_continuations_used':len(keys),'distinct_truncated_continuations':len(truncated),
            'policy_replay_rows':len(frame),'verification_status_counts':statuses,
            'review':review,'one_call_results':one.where(pd.notna(one),None).to_dict('records')}
    # pandas numeric columns retain NaN even after replacing with None; JSON
    # round-tripping through pandas produces standards-compliant null values.
    report['one_call_results']=json.loads(one.to_json(orient='records'))
    (stage/'measured_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=[f'# Measured {stage.name} results','',f"Actual questions: {report['questions']}; seeded episodes: {len(records)}; distinct generated continuations used: {len(keys)}.",'',
           'These are final-answer exact-match outcomes. They are not complete downstream intermediate factuality measurements. Human review is pending.','',
           '| Method | Accuracy | Gain | Mean calls | Harmful selected episodes |','|---|---:|---:|---:|---:|']
    for _,row in one.iterrows():
        lines.append(f"| {row['method']} | {row['final_accuracy']:.4f} | {row['mean_gain']:+.4f} | {row['mean_calls']:.2f} | {row['harmful_rate_per_selected_episode']:.4f} |")
    lines+=['','Means pool recorded training and generation seeds; question-cluster intervals are in pooled_policy_summary.csv. The oracle is analysis-only.','',
            'Inspect injection_effects.csv to see whether injected notes actually changed final correctness. Zero effects are retained. Observed_benefit_distribution.csv shows whether the training signal is sparse.','',
            'The smoke and pilot are different stages, with extraction/eligibility refinements before pilot collection. Do not merge them into a single accuracy estimate.']
    (stage/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report
