"""Controlled real-model experiments and matched policy replay.

Every evaluated joint intervention uses an actually generated corresponding
continuation (or an exact prompt/settings cache hit). Singleton gains are never
added to fabricate multi-intervention outcomes. Gold answers stay in evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss, confusion_matrix
from .data import supporting_sentences, write_jsonl
from .detector import ExistingDetector
from .generation import SmallReasoner
from .graph import build_state, node_features
from .injection import select_conditions
from .metrics import exact_match, prediction_metrics, summarize_policies
from .models import GraphPolicy, fit_gnn, fit_tabular
from .policy import TabularPolicy, baseline_scores, select_claims
from .schema import Budget, DecisionState


def apply_actions(state: DecisionState, actions: list[dict[str, Any]]) -> DecisionState:
    """Apply only recorded accepted/unchanged replacements to a copied trace.

    Args:
        state: Pre-intervention state; never mutated by this function.
        actions: Actual verifier decisions with claim IDs and replacement strings.
    Returns:
        Independent state with changed notes and observable action statuses.
    """
    result = copy.deepcopy(state)
    updates = {a['claim_id']: a for a in actions}
    for claim in result.claims:
        if claim.id in updates:
            claim.text = updates[claim.id]['replacement']
            claim.status = updates[claim.id]['status']
    return result


def outcome(reasoner: SmallReasoner, state: DecisionState, seed: int, reference: str) -> dict[str, Any]:
    """Generate and externally score one continuation; never expose reference to prompt.

    Args:
        reasoner: Frozen real model backend.
        state: Current notes and fixed question/evidence.
        seed: Matched generation seed.
        reference: Evaluation-only HotpotQA answer.
    Returns:
        Actual generation metadata plus exact match and final-node loss.
    """
    generated = reasoner.generate(state, seed)
    correct = exact_match(generated['answer'], reference)
    return dict(generated, correct=correct, observed_final_node_error=int(not correct),
                broader_downstream_error_count=None)


def collect_episodes(splits: dict[str, list[dict[str, Any]]], detector: ExistingDetector,
                     reasoner: SmallReasoner, output: Path, seeds: list[int], conditions: int = 3,
                     safe_ablation: bool = True) -> list[dict[str, Any]]:
    """Record controlled paired outcomes on real questions, with crash-safe resume.

    Args:
        splits: Official-train-derived fit/tuning questions and held-out validation.
        detector: Common real detector and correction verifier for all policies.
        reasoner: Same frozen generation model/settings for every intervention.
        output: JSONL episode destination; existing completed IDs are reused.
        seeds: Matched generation seeds; stochastic only if backend sampling is on.
        conditions: Maximum distinct eligible primary error conditions per question.
        safe_ablation: Record unsafe singleton substitution outcomes for comparison.
    Returns:
        Complete episodes, including resumed records, with evaluation labels stored
        separately from pre_state. Auxiliary unvalidated injections go to review.
    Safeguards:
        Unsupported injection types are not silently promoted to gold errors.
        Question evidence and reference are fixed across paired actions. Clean
        output, ignored injections, failures, and cache hits remain auditable.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        from .data import read_jsonl
        existing = read_jsonl(output)
    done = {r['episode_id'] for r in existing}
    rows = list(existing)
    review: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for split, questions in splits.items():
        for qi, question in enumerate(questions):
            facts, evidence = supporting_sentences(question)
            if len(facts)<2 or len({f['evidence_id'] for f in facts})<2:
                skipped.append({'question_id': question['id'], 'reason': 'fewer_than_two_distinct_support_documents'})
                continue
            clean = build_state(question['id'], question['question'], facts, evidence)
            injections, auxiliary = select_conditions(clean, conditions)
            review.extend(dict(question_id=question['id'], question=question['question'], **item) for item in auxiliary)
            if not injections:
                skipped.append({'question_id': question['id'], 'reason': 'no_mechanically_eligible_primary_error'})
            for injection in injections:
                for seed in seeds:
                    episode_id = f"{question['id']}:{injection['type']}:{seed}"
                    if episode_id in done:
                        continue
                    corrupted = copy.deepcopy(clean)
                    next(c for c in corrupted.claims if c.id==injection['node_id']).text = injection['modified']
                    detector.annotate(corrupted)
                    clean_result = outcome(reasoner, clean, seed, question['answer'])
                    baseline = outcome(reasoner, corrupted, seed, question['answer'])
                    actions = {c.id: detector.verify(corrupted, c.id) for c in corrupted.claims if not c.is_final}
                    nodes = list(actions)
                    singletons: dict[str, Any] = {}
                    joint: dict[str, Any] = {'': baseline}
                    # Enumerating all subsets is affordable because the baseline
                    # has at most three intermediate nodes. A cached continuation
                    # is still the exact outcome of that prompt, not a sum of gains.
                    for size in range(1, len(nodes)+1):
                        for chosen in itertools.combinations(nodes, size):
                            changed = apply_actions(corrupted, [actions[n] for n in chosen])
                            result = outcome(reasoner, changed, seed, question['answer'])
                            joint['|'.join(sorted(chosen))] = result
                            if size==1:
                                singletons[chosen[0]] = result
                    unsafe: dict[str, Any] = {}
                    if safe_ablation:
                        for node in nodes:
                            action = detector.verify(corrupted, node, safe=False)
                            unsafe[node] = dict(outcome(reasoner, apply_actions(corrupted, [action]), seed, question['answer']), action=action)
                    record = {'episode_id': episode_id, 'question_id': question['id'], 'split': split,
                              'question': question['question'], 'reference_answer': question['answer'],
                              'provenance': question['_provenance'], 'clean_trace': clean.to_dict(),
                              'clean_continuation': clean_result, 'injection': injection,
                              'pre_state': corrupted.to_dict(), 'baseline': baseline, 'singletons': singletons,
                              'joint_outcomes': joint, 'verification_actions': actions, 'unsafe_singletons': unsafe,
                              'generation_seed': seed, 'injection_ignored_output': clean_result['answer']==baseline['answer'],
                              'human_review': 'pending', 'observed_loss_scope': 'final_answer_only'}
                    with output.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    rows.append(record)
                    done.add(episode_id)
            if (qi+1)%5==0:
                print(f'{split}: {qi+1}/{len(questions)} questions; episodes={len(rows)}; actual_generations={reasoner.actual_generations}; cache_hits={reasoner.cache_hits}', flush=True)
    write_jsonl(output.with_name('injection_review_candidates.jsonl'), review)
    write_jsonl(output.with_name('skipped_questions.jsonl'), skipped)
    return rows


def validate_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check schema, graph integrity, paired settings, and train/test separation.

    Args:
        records: Actual collected episode dictionaries.
    Returns:
        Audit counts and limitations; does not certify semantic injection validity.
    Raises:
        ValueError: On duplicate IDs, malformed graph, split leakage, missing
            outcome actions, modified question/evidence, or seed mismatch.
    """
    from .graph import validate_graph
    ids: set[str] = set()
    split_ids: dict[str, set[str]] = {}
    for record in records:
        if record['episode_id'] in ids:
            raise ValueError('Duplicate episode ID')
        ids.add(record['episode_id'])
        state = DecisionState.from_dict(record['pre_state'])
        validate_graph(state)
        split_ids.setdefault(record['split'], set()).add(record['question_id'])
        if state.question!=record['clean_trace']['question'] or state.evidence!=record['clean_trace']['evidence']:
            raise ValueError('Question or evidence changed during injection')
        if record['injection']['original']==record['injection']['modified']:
            raise ValueError('Injection did not change text')
        for value in record['joint_outcomes'].values():
            if value['seed']!=record['generation_seed'] or value['model_id']!=record['baseline']['model_id']:
                raise ValueError('Paired generation settings differ')
        node_features(state)  # Strict typed state contains no outcome feature slots.
    for a in split_ids:
        for b in split_ids:
            if a<b and split_ids[a]&split_ids[b]:
                raise ValueError('Question leakage between intervention partitions')
    return {'episodes': len(records), 'questions_by_split': {k: len(v) for k,v in split_ids.items()},
            'human_review_completed': False, 'full_downstream_factual_labels_available': False,
            'passed_structural_checks': True}


def train_policies(records: list[dict[str, Any]], directory: Path, seeds: list[int], epochs: int = 25,
                   batch_size: int = 16, accumulation: int = 2, device: str = 'auto', resume: bool = False) -> dict[str, Any]:
    """Train both intervention approaches and retrained feature ablations.

    Args:
        records: Complete episodes; only train/validation partitions are fitted.
        directory: Separate output checkpoint folder for this experiment.
        seeds: Initialization seeds for model variability assessment.
        epochs: Graph training epochs, controlled independently of reasoner seeds.
        batch_size: Graphs per minibatch.
        accumulation: Graph minibatches per optimizer update.
        device: CUDA/CPU training target.
        resume: Restore graph optimizer/RNG state when matching checkpoints exist.
    Returns:
        Per-model training summaries; no test outcomes affect model selection.
    """
    from .graph import GRAPH_FEATURES, FEATURE_NAMES
    train = [r for r in records if r['split']=='train']
    val = [r for r in records if r['split']=='validation']
    result: dict[str, Any] = {}
    for seed in seeds:
        for name, cols in [('tabular', None), ('without_graph', [i for i in range(len(FEATURE_NAMES)) if i not in GRAPH_FEATURES]),
                           ('without_risk', [i for i in range(len(FEATURE_NAMES)) if i not in (0,1)])]:
            result[f'{name}_{seed}'] = fit_tabular(train, val, directory/f'{name}_{seed}.joblib', cols, seed)
        result[f'graph_{seed}'] = fit_gnn(train, val, directory/f'graph_{seed}.pt', seed, epochs, batch_size, accumulation, device, resume)
    (directory/'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def evaluate_policies(records: list[dict[str, Any]], models: Path, output: Path, training_seed: int,
                      budgets: list[int] | None = None, cost_lambda: float = .01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay matched held-out outcomes under each policy and actual budget.

    Args:
        records: Full episodes; only official-validation-derived test is evaluated.
        models: Checkpoints fitted on training and selected with validation only.
        output: Tables/predictions directory for this policy training seed.
        training_seed: Determines which learned checkpoints and random ranking load.
        budgets: Call limits; default 0,1,2,3 covers full verification with <=3 nodes.
        cost_lambda: Penalty per logical call in the reported loss-plus-cost objective.
    Returns:
        Per-episode actual-outcome table and question-cluster summary.
    Oracle:
        Uses observed joint test outcomes only as an explicitly labeled analysis
        bound. It is never deployed or fed into the learned feature pipeline.
    """
    budgets = [0,1,2,3] if budgets is None else budgets
    output.mkdir(parents=True, exist_ok=True)
    learned: dict[str, Any] = {'chc_tabular': TabularPolicy(models/f'tabular_{training_seed}.joblib'),
                               'chc_graph': GraphPolicy(models/f'graph_{training_seed}.pt'),
                               'without_graph': TabularPolicy(models/f'without_graph_{training_seed}.joblib'),
                               'without_risk': TabularPolicy(models/f'without_risk_{training_seed}.joblib')}
    rows, benefits = [], []
    for r in records:
        if r['split']!='test':
            continue
        state = DecisionState.from_dict(r['pre_state'])
        score_sets = {m: baseline_scores(state, m, training_seed) for m in ('random','uncertainty','dependency','risk_dependency')}
        score_sets.update({m: p.scores(state) for m,p in learned.items()})
        for method, scores in score_sets.items():
            if method.startswith('chc'):
                for node, value in scores.items():
                    benefits.append({'method': method, 'question_id': r['question_id'], 'prediction': value,
                                     'observed': float(r['singletons'][node]['correct'])-float(r['baseline']['correct'])})
        for budget in budgets:
            cost_sensitive={'chc_tabular_cost_sensitive':'chc_tabular','chc_graph_cost_sensitive':'chc_graph'}
            for method in ['no_intervention', *score_sets, *cost_sensitive, 'oracle_analysis_only']:
                if method=='no_intervention':
                    chosen: list[str] = []
                elif method=='oracle_analysis_only':
                    keys = [key for key in r['joint_outcomes'] if len(key.split('|'))<=budget or key=='']
                    best = max(keys, key=lambda k: (int(r['joint_outcomes'][k]['correct']), -len(k.split('|')) if k else 0))
                    chosen = best.split('|') if best else []
                elif method in cost_sensitive:
                    # This variant may spend less than its maximum budget when
                    # predicted singleton improvement does not justify call cost.
                    # Quota variants remain separate for matched-call comparisons.
                    chosen=select_claims(state,score_sets[cost_sensitive[method]],Budget(budget),
                                         overlap_discount=True,minimum_net_benefit=cost_lambda)
                else:
                    chosen = select_claims(state, score_sets[method], Budget(budget), overlap_discount=method.startswith('chc'))
                key = '|'.join(sorted(chosen))
                generated = r['joint_outcomes'][key]
                action_cost = sum(r['verification_actions'][n]['estimated_tokens'] for n in chosen)
                rows.append({'episode_id': r['episode_id'], 'question_id': r['question_id'],
                             'generation_seed': r['generation_seed'], 'training_seed': training_seed,
                             'condition': r['injection']['type'], 'method': method, 'budget': budget,
                             'selected': key, 'calls': len(chosen), 'estimated_tokens': action_cost,
                             'correct': int(generated['correct']), 'baseline_correct': int(r['baseline']['correct']),
                             'generated_answer': generated['answer'], 'reference_answer': r['reference_answer'],
                             'safe_correction': True, 'observed_loss_scope': 'final_answer_only'})
            # Safe-correction ablation uses the same tabular ranking, budget one,
            # and actual unsafe singleton generation, not assumed gains.
            if budget==1 and r.get('unsafe_singletons'):
                chosen = select_claims(state, score_sets['chc_tabular'], Budget(1))
                node = chosen[0]
                generated = r['unsafe_singletons'][node]
                rows.append({'episode_id': r['episode_id'], 'question_id': r['question_id'], 'generation_seed': r['generation_seed'],
                             'training_seed': training_seed, 'condition': r['injection']['type'], 'method': 'lexical_correction_variant',
                             'budget': 1, 'selected': node, 'calls': 1, 'estimated_tokens': r['verification_actions'][node]['estimated_tokens'],
                             'correct': int(generated['correct']), 'baseline_correct': int(r['baseline']['correct']),
                             'generated_answer': generated['answer'], 'reference_answer': r['reference_answer'],
                             'safe_correction': False, 'observed_loss_scope': 'final_answer_only'})
            if budget==1 and r.get('gate_off_singletons'):
                chosen = select_claims(state, score_sets['chc_tabular'], Budget(1))
                node = chosen[0]
                generated = r['gate_off_singletons'][node]
                rows.append({'episode_id':r['episode_id'],'question_id':r['question_id'],'generation_seed':r['generation_seed'],
                             'training_seed':training_seed,'condition':r['injection']['type'],'method':'without_validity_gate',
                             'budget':1,'selected':node,'calls':1,'estimated_tokens':r['verification_actions'][node]['estimated_tokens'],
                             'correct':int(generated['correct']),'baseline_correct':int(r['baseline']['correct']),
                             'generated_answer':generated['answer'],'reference_answer':r['reference_answer'],
                             'safe_correction':False,'observed_loss_scope':'final_answer_only'})
    frame = pd.DataFrame(rows)
    summary = summarize_policies(frame,cost_lambda=cost_lambda)
    frame.to_csv(output/'policy_predictions.csv', index=False)
    summary.to_csv(output/'policy_summary.csv', index=False)
    benefit_frame = pd.DataFrame(benefits)
    benefit_frame.to_csv(output/'benefit_predictions.csv', index=False)
    benefit_metrics = {m: prediction_metrics(g['observed'].tolist(),g['prediction'].tolist()) for m,g in benefit_frame.groupby('method')}
    (output/'benefit_metrics.json').write_text(json.dumps(benefit_metrics, indent=2), encoding='utf-8')
    return frame, summary


def add_gate_ablation(records: list[dict[str, Any]], reasoner: SmallReasoner, output: Path) -> list[dict[str, Any]]:
    """Generate a stricter gate-only ablation with unchanged proposal pool/ranking.

    Args:
        records: Collected real episodes with each verifier's assessed candidates.
        reasoner: Same frozen backend and sampling settings as collection.
        output: Separate JSONL destination; original collection is preserved.
    Returns:
        Episodes augmented with actual unfiltered-proposal singleton outcomes.
    Method:
        Safe correction ranks accepted proposals by min(support,preservation).
        This ablation ranks the same full pool by that same score but removes
        acceptance filtering. The older lexical variant changes proposal ranking
        too, so it must not be interpreted as an isolated gate contribution.
    """
    augmented=[]
    for index,record in enumerate(records):
        # This is a held-out evaluation ablation, not an alternative target used
        # to retrain the policy. Do not spend extra generation on fitting rows.
        if record['split']!='test':
            augmented.append(record)
            continue
        state=DecisionState.from_dict(record['pre_state'])
        gate_off={}
        for node,action in record['verification_actions'].items():
            pool=action['details'].get('candidates',[])
            if not pool:
                raise ValueError('Missing assessed candidate pool for gate-only ablation')
            best=max(pool,key=lambda p:min(p['support_score'],p['preservation_score']))
            new_action=dict(action,replacement=best['proposal'],status='unfiltered_candidate_ablation')
            generated=outcome(reasoner,apply_actions(state,[new_action]),record['generation_seed'],record['reference_answer'])
            gate_off[node]=dict(generated,action=new_action)
        augmented.append(dict(record,gate_off_singletons=gate_off))
        if (index+1)%250==0:
            print(f'Gate-only ablation: {index+1}/{len(records)} episodes; new generations={reasoner.actual_generations}',flush=True)
    write_jsonl(output,augmented)
    return augmented


def evaluate_detector(detector: ExistingDetector, frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    """Evaluate the actual local detector on a held-out natural RAGTruth sample.

    Args:
        detector: Existing saved DistilBERT, not a mocked scorer.
        frame: Held-out rows with actual binary labels and source IDs.
        output: JSON/CSV destination directory.
    Returns:
        Precision, recall, F1, Brier, binned calibration error, and confusion matrix.
    Limitations:
        Small natural samples can have few positives. No tuning/calibration is
        fitted on this test sample and uncertainty must not be ignored.
    """
    output.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(detector.predict_pairs(frame['claim'].tolist(),frame['evidence'].tolist()))
    y = frame['label'].to_numpy()
    pred = probabilities>=.5
    precision, recall, f1, _ = precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    calibration = []
    ece = 0.
    for low in np.arange(0,1,.1):
        mask = (probabilities>=low)&(probabilities<low+.1 if low<.9 else probabilities<=1)
        if mask.any():
            confidence, fraction = float(probabilities[mask].mean()), float(y[mask].mean())
            ece += mask.mean()*abs(confidence-fraction)
            calibration.append({'mean_score':confidence,'positive_fraction':fraction,'n':int(mask.sum())})
    result = {'n':len(y),'positive_labels':int(y.sum()),'precision':float(precision),'recall':float(recall),'f1':float(f1),
              'brier':float(brier_score_loss(y,probabilities)),'ece_10_bins':float(ece),
              'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'calibration_bins':calibration,
              'calibration_fitted':False,'scope':'Natural sampled held-out RAGTruth; not causal propagation data',
              'max_pair_tokens':detector.max_length}
    (output/'detector_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    frame=frame.copy();frame['predicted_probability']=probabilities;frame['predicted_label']=pred.astype(int)
    frame.to_csv(output/'detector_predictions.csv',index=False)
    return result
