"""All-input answer scoring and denominator audits for recorded policy outcomes.

Primary propagation analyses require evaluable structured claims. This separate
analysis also scores answers from every eligible input, exposing format failures
and complete-case selection. It never marks an invalid trace valid or an unknown
claim false. No model is retrained or chosen using these held-out results.
"""
from __future__ import annotations
import json,re
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from ..schema import DecisionState,Budget
from ..policy import TabularPolicy,baseline_scores,select_claims
from ..models import GraphPolicy
from ..metrics import exact_match
from .graph_features import prepare_policy_records
from .learning import visible_features,question_interval


def recorded_answer(outcome: dict[str,Any]) -> str | None:
    """Extract an actual final-answer field even if a trace failed validation.

    Only a nonempty string from an existing answer or parseable raw JSON object
    is accepted. No regex guess, reference-based repair or invented answer is
    allowed. A rejected trace remains rejected regardless of answer availability.
    """
    if isinstance(outcome.get('answer'),str) and outcome['answer'].strip():return outcome['answer']
    raw=outcome['generation']['raw'].strip()
    if raw.startswith('```'):raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw).strip()
    try:
        obj=json.loads(raw);answer=obj.get('final_answer') if isinstance(obj,dict) else None
        return answer if isinstance(answer,str) and answer.strip() else None
    except (ValueError,TypeError):return None


def selected_outcome(record: dict[str,Any], selected: list[str], method: str) -> dict[str,Any]:
    """Retrieve the actual measured subset outcome without adding singleton gains."""
    if not selected:return record['baseline']
    if len(selected)==2:return record['joint_gate_off'] if method.endswith('_without_semantic_gate') else record['joint_both']
    return record['without_semantic_gate'][selected[0]] if method.endswith('_without_semantic_gate') else record['singletons'][selected[0]]


def operational_evaluation(records: list[dict[str,Any]], judge: Any, models: Path,
                           output: Path, historical: Path | None=None) -> dict[str,Any]:
    """Score every primary test input, without post-output complete-case filtering.

    Generation/claim validity and final-answer availability are separate fields.
    Missing answers count as operational answer failures, not factual falsehoods.
    Malformed traces with parseable answer fields can still have measured answer
    EM, but have no claimed downstream factual label. All selectors see only
    serving-visible states; reference strings enter scoring after selection.
    """
    records=prepare_policy_records([r for r in records if r['split']=='test'])
    if not records:return {'status':'No primary test inputs'}
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((models/'training_summary.json').read_text());seeds=summary.get('seeds',[42,43,44])
    semantic={p.stem:joblib.load(p) for p in models.glob('semantic_*.joblib')}
    semantic_lookup={}
    if semantic:
        x,embedding,keys=visible_features(records,judge);values=np.concatenate([x,embedding],axis=1)
        semantic_lookup={key:values[i] for i,key in enumerate(keys)}
    rows=[]
    for seed in seeds:
        policies={}
        for path in models.glob(f'tabular_*_{seed}.joblib'):policies[path.stem.rsplit('_',1)[0]]=TabularPolicy(path)
        path=models/f'original_tabular_retrained_{seed}.joblib'
        if path.exists():policies['original_tabular_retrained']=TabularPolicy(path)
        for path in models.glob(f'graph_*_{seed}.pt'):policies[path.stem.rsplit('_',1)[0]]=GraphPolicy(path)
        if historical:
            for name,filename,cls in [('original_tabular_transfer',f'tabular_{seed}.joblib',TabularPolicy),('original_graph_transfer',f'graph_{seed}.pt',GraphPolicy)]:
                path=historical/filename
                if path.exists():policies[name]=cls(path)
        for r in records:
            state=DecisionState.from_dict(r['pre_state']);ids=[c.id for c in state.claims if not c.is_final]
            scores={m:baseline_scores(state,m,seed) for m in ('random','uncertainty','dependency','risk_dependency')}
            for name,policy in policies.items():
                current=DecisionState.from_dict(r['declared_only_pre_state']) if name.endswith('without_source_bridges') else state
                scores[name]=policy.scores(current)
            for name,artifact in semantic.items():
                scores[name]={n:float(artifact['model'].predict(semantic_lookup[(r['episode_id'],n)][artifact['columns']].reshape(1,-1))[0]) for n in ids}
            scores.update(zero_benefit={n:0. for n in ids},training_mean={n:summary.get('training_mean_control',0.) for n in ids},no_intervention={})
            baseline_answer=recorded_answer(r['baseline']);baseline_correct=exact_match(baseline_answer,r['reference_answer']) if baseline_answer else False
            for method,priority in scores.items():
                variants=[(method,method in ('zero_benefit','training_mean'))]
                if method in ('tabular_richer_all','graph_richer','semantic_richer'):variants.append((method+'_cost_aware',True))
                if method=='tabular_richer_all':variants.append((method+'_without_semantic_gate',False))
                for name,stop in variants:
                    for budget in (0,1,2):
                        selected=[] if method=='no_intervention' else select_claims(state,priority,Budget(calls=budget),minimum_net_benefit=.01 if stop else None)
                        o=selected_outcome(r,selected,name);answer=recorded_answer(o)
                        correct=exact_match(answer,r['reference_answer']) if answer else False
                        rows.append({'method':name,'budget':budget,'seed':seed,'question_id':r['question_id'],'episode_id':r['episode_id'],
                            'answer_available':answer is not None,'trace_valid':o['valid'],'answer_em':int(correct),'baseline_em':int(baseline_correct),
                            'gain':int(correct)-int(baseline_correct),'calls':len(selected),'selected':'|'.join(selected),
                            'factual_status':'Automatically scored structured units' if o['valid'] else 'Unavailable: invalid trace; not labeled false'})
    frame=pd.DataFrame(rows);frame.to_csv(output/'all_input_answer_predictions.csv',index=False);reports=[]
    for (method,budget),g in frame.groupby(['method','budget']):
        q=g.groupby('question_id')[['answer_em','gain','trace_valid','answer_available','calls']].mean();lo,hi=question_interval(q.gain)
        reports.append({'method':method,'budget':int(budget),'questions':len(q),'answer_exact_match_all_inputs':float(q.answer_em.mean()),
            'gain_over_no_action':float(q.gain.mean()),'gain_ci_low':lo,'gain_ci_high':hi,'valid_trace_fraction':float(q.trace_valid.mean()),
            'answer_available_fraction':float(q.answer_available.mean()),'mean_calls':float(q.calls.mean())})
    pd.DataFrame(reports).to_csv(output/'all_input_answer_summary.csv',index=False)
    report={'scope':'All eligible primary test inputs. Parseable answer fields are scored independently of trace validity. Missing answers are operational failures, not labeled factual contradictions.',
        'primary_test_questions':len({r['question_id'] for r in records}),'summaries':reports}
    (output/'all_input_answer_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def augment_denominators(records: list[dict[str,Any]], evaluation: Path) -> None:
    """Add per-unit coverage and logical NLI token accounting to policy CSVs.

    Unknown units remain separate from contradicted units. Logical NLI input
    tokens reflect bounded original/proposal checks, not all upfront detector or
    embedding computation and not uncached physical executions or API billing.
    """
    if not (evaluation/'policy_predictions.csv').exists():return
    frame=pd.read_csv(evaluation/'policy_predictions.csv',keep_default_na=False);lookup={r['episode_id']:r for r in records}
    extras=[]
    for row in frame.to_dict(orient='records'):
        r=lookup[row['episode_id']];selected=[n for n in str(row['selected']).split('|') if n]
        o=selected_outcome(r,selected,row['method']);f=o.get('factual')
        total=f['downstream_claims'] if f else 0;evaluable=f['evaluable_claims'] if f else 0
        tokens=0
        for n in selected:
            action=r['verification_actions'][n]
            tokens+=sum(min(384,action[k].get('untruncated_tokens',0)) for k in ('original_judgment','proposed_judgment'))
        extras.append({'evaluable_unit_fraction':evaluable/total if total else None,'downstream_units':total,
            'evaluable_units':evaluable,'insufficient_or_unreviewed_units':total-evaluable,'logical_nli_input_tokens':tokens})
    for key in extras[0] if extras else []:frame[key]=[e[key] for e in extras]
    frame.to_csv(evaluation/'policy_predictions.csv',index=False)
    summary=pd.read_csv(evaluation/'policy_summary.csv')
    additions=frame.groupby(['method','budget','question_id'])[['evaluable_unit_fraction','logical_nli_input_tokens']].mean().groupby(['method','budget']).mean().reset_index()
    summary=summary.drop(columns=[c for c in additions.columns if c not in ('method','budget') and c in summary.columns])
    summary=summary.merge(additions,on=['method','budget'],how='left');summary.to_csv(evaluation/'policy_summary.csv',index=False)
