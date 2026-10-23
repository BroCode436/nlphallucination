"""Reproducible training with disjoint relation templates and entity names.
"""
from typing import Any as _DocAny
import csv
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import json
import random
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from correction_validity import features, FEATURES, CorrectionValidityModel

# Splits hold out complete relation templates, not rows of the same case.
PATTERNS={
 'train':[
  ('{e} opened in {n} and is located in {l}.','{e} opened in {n}.'),
  ('{e} employs {n} people and operates in {l}.','{e} employs {n} people.'),
  ('{e} contains {n} books and serves {l}.','{e} contains {n} books.'),
  ('{e} ने {n} बसें चलाईं और {l} में सेवा दी।','{e} ने {n} बसें चलाईं।'),
  ('{e} mein {n} rooms hain aur location {l} hai.','{e} mein {n} rooms hain.'),
  ('{e} produced {n} units and exports to {l}.','{e} produced {n} units.')],
 'validation':[
  ('{e} sold {n} tickets and is based in {l}.','{e} sold {n} tickets.'),
  ('{e} ने {n} पदक जीते और {l} का प्रतिनिधित्व किया।','{e} ने {n} पदक जीते।'),
  ('{e} ne {n} matches jeete aur team {l} hai.','{e} ne {n} matches jeete.')],
 'test':[
  ('{e} enrolled {n} students and has a campus in {l}.','{e} enrolled {n} students.'),
  ('{e} में {n} कर्मचारी हैं और कार्यालय {l} में है।','{e} में {n} कर्मचारी हैं।'),
  ('{e} ne {n} awards jeete aur office {l} mein hai.','{e} ne {n} awards jeete.')]
}

def generate(split: _DocAny,seed: _DocAny) -> _DocAny:
    """Generate.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            split: Split supplied by the caller; see the operation and call sites for its concrete format.
            seed: Pseudorandom seed controlling reproducible sampling or model initialization.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    rng=random.Random(seed)
    rows=[]
    for ti,(full,short) in enumerate(PATTERNS[split]):
        for i in range(160):
            entity=f'{split}entity{ti}x{i}'
            other=f'{split}other{ti}x{i}'
            loc=f'{split}place{ti}x{i}'
            otherloc=f'{split}town{ti}x{i}'
            n=rng.randint(10,1900); d=n+rng.randint(2,40); wrong=d+rng.randint(2,40)
            def render(e: _DocAny=entity,v: _DocAny=n,l: _DocAny=loc,brief: bool=False) -> _DocAny:
                """Render.

                This operation participates in the saved local NLP experiment workflow.

                Learning notes — input/output contract:
                    Parameters:
                        e: E supplied by the caller; see the operation and call sites for its concrete format.
                        v: V supplied by the caller; see the operation and call sites for its concrete format.
                        l: L supplied by the caller; see the operation and call sites for its concrete format.
                        brief: Brief supplied by the caller; see the operation and call sites for its concrete format.
                    Returns:
                        The value assembled by this function (score, table, records, or model as
                        described above). The Any type preserves the legacy dynamic interface;
                        inspect the documented call site for concrete tensor/table shapes.
                    Assumptions and failures:
                        Inputs must match the saved dataset/model schema. Missing files, invalid
                        fields, or incompatible checkpoints can raise errors; tests cover normal
                        and selected failure paths. Scores are not independent factual proof.

                """
                return (short if brief else full).format(e=e,n=v,l=l)
            target=render(); distractor=render(other,d,otherloc)
            evidence=' '.join(rng.sample([target,distractor],2))
            original=render(v=wrong)
            group=f'{split}-{ti}-{i}'
            candidates=[('valid',target,1,1),('wrong_entity',distractor,1,0),
                        ('dropped_detail',render(brief=True),1,0),
                        ('unchanged_error',original,0,1),('wrong_binding',render(v=d),0,1),
                        ('unsupported_detail',render(l='Unknownplace'),0,0)]
            for kind,candidate,support,preservation in candidates:
                rows.append(dict(group=group,evidence=evidence,original=original,candidate=candidate,
                                 kind=kind,support=support,preservation=preservation,valid=support*preservation))
            # Original-swap counterpart: same evidence and proposal; changes validity.
            rows.append(dict(group=group,evidence=evidence,original=render(other,wrong,otherloc),
                             candidate=target,kind='original_swap',support=1,preservation=0,valid=0))
            # Do not teach the shortcut that an unchanged answer is always invalid.
            rows.append(dict(group=group,evidence=evidence,original=target,candidate=target,
                             kind='correct_unchanged',support=1,preservation=1,valid=1))
            rows.append(dict(group=group,evidence=evidence,original=render(e='Unlistedentity'),candidate=target,
                             kind='unsupported_entity_repair',support=1,preservation=1,valid=1))
            rows.append(dict(group=group,evidence=evidence,original=render(l='Unlistedplace'),candidate=target,
                             kind='unsupported_location_repair',support=1,preservation=1,valid=1))
    return rows

def metrics(y: _DocAny,pred: _DocAny) -> _DocAny:
    """Metrics.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters:
            y: Y supplied by the caller; see the operation and call sites for its concrete format.
            pred: Pred supplied by the caller; see the operation and call sites for its concrete format.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return dict(precision=float(p),recall=float(r),f1=float(f),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),n=len(y))

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
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'evaluation').mkdir(exist_ok=True)
    datasets={s:generate(s,20261023+i) for i,s in enumerate(PATTERNS)}
    artifact={'version':'1.0.0','features':FEATURES,'seed':20261023,'training_scope':'Synthetic lexical correction validity; not general semantic entailment'}
    train=datasets['train']; val=datasets['validation']
    x=np.asarray([features(r['evidence'],r['original'],r['candidate']) for r in train])
    vx=np.asarray([features(r['evidence'],r['original'],r['candidate']) for r in val])
    for head in ('support','preservation'):
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=3,random_state=20261023,early_stopping=False)
        clf.fit(x,[r[head] for r in train]); artifact[head]=clf
    # Select both thresholds on validation only, then freeze before test scoring.
    sp=artifact['support'].predict_proba(vx)[:,1]; pp=artifact['preservation'].predict_proba(vx)[:,1]
    y=np.array([r['valid'] for r in val]); best=None
    for st in (.5,.7,.8,.9,.95):
        for pt in (.5,.7,.8,.9,.95):
            pred=(sp>=st)&(pp>=pt)&(vx[:,7]>=1)&(vx[:,12]==0)
            m=metrics(y,pred)
            # Prefer fewer false acceptances, then higher recall, then stricter threshold.
            rank=(-m['confusion_matrix'][0][1],m['recall'],st+pt)
            if best is None or rank>best[0]: best=(rank,st,pt)
    artifact['thresholds']={'support':best[1],'preservation':best[2]}
    joblib.dump(artifact,ROOT/'models'/'correction_validity.joblib')
    model=CorrectionValidityModel()
    def batch_assess(rows: _DocAny) -> _DocAny:
        """Batch assess.

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
        bx=np.asarray([features(r['evidence'],r['original'],r['candidate']) for r in rows])
        bs=artifact['support'].predict_proba(bx)[:,1]
        bp=artifact['preservation'].predict_proba(bx)[:,1]
        accepted=(bs>=best[1])&(bp>=best[2])&(bx[:,7]>=1)&(bx[:,12]==0)&(bx[:,13]>0)&(bx[:,14]>0)&(bx[:,15]>0)
        # These generated/study cases all contain numbers, so the non-numeric
        # word-order review constraint in assess() cannot change their decisions.
        assert all(__import__('correction_validity').numbers(r['candidate']) for r in rows)
        return [dict(accepted=bool(a),support_score=float(s),preservation_score=float(p)) for a,s,p in zip(accepted,bs,bp)]
    report={'scope':'Synthetic tests; no independent human annotation. General factual accuracy is not established.',
            'split_policy':'Distinct relation templates and entity/location names across train, validation and test',
            'thresholds':artifact['thresholds'],'splits':{},'external_study':{}}
    for split,rows in datasets.items():
        path=ROOT/'evaluation'/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        results=batch_assess(rows)
        report['splits'][split]=metrics([r['valid'] for r in rows],[r['accepted'] for r in results])
        report['splits'][split]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['splits'][split]['by_kind']={k:metrics([r['valid'] for r in rows if r['kind']==k],
            [a['accepted'] for r,a in zip(rows,results) if r['kind']==k]) for k in sorted({r['kind'] for r in rows})}
        if split=='test':
            with (ROOT/'evaluation'/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['group','kind','gold','accepted','support_score','preservation_score'])
                writer.writeheader()
                for r,a in zip(rows,results):writer.writerow(dict(group=r['group'],kind=r['kind'],gold=r['valid'],**{k:a[k] for k in ('accepted','support_score','preservation_score')}))
    study=ROOT.parent/'correction_validity_study'/'cases.jsonl'
    if study.exists():
        rows=[json.loads(s) for s in study.read_text(encoding='utf-8').splitlines()]
        results=batch_assess(rows)
        report['external_study']=metrics([r['construction_valid_complete_repair'] for r in rows],[r['accepted'] for r in results])
        report['external_study']['by_kind']={k:{'n':sum(r['candidate_kind']==k for r in rows),'accepted':sum(a['accepted'] for r,a in zip(rows,results) if r['candidate_kind']==k)} for k in sorted({r['candidate_kind'] for r in rows})}
        report['external_study']['limitation']='Previously examined synthetic diagnostic set; not an untouched final benchmark.'
    (ROOT/'evaluation'/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
