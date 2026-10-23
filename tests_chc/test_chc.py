"""Offline deterministic unit fixtures; none of their scores claim model quality.

Real benchmark and GPU tests are executed by the separate experiment runner.
These tests exercise schemas, graph direction, budgets, metrics, and checkpoints.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
import torch
from chc.data import supporting_sentences, split_questions, audit_splits, read_jsonl, write_jsonl
from chc.graph import build_state, validate_graph, node_features
from chc.injection import candidates, select_conditions
from chc.schema import Budget, DecisionState
from chc.policy import baseline_scores, select_claims, TabularPolicy
from chc.models import BenefitGNN, fit_gnn, fit_tabular, GraphPolicy, training_graph
from chc.metrics import exact_match, normalize_answer, prediction_metrics, summarize_policies
from chc.experiment import apply_actions, validate_episodes
from chc.resources import choose_device, seed_everything, hardware_report
import pandas as pd


@pytest.fixture
def question() -> dict:
    """Return an explicitly synthetic two-hop loader fixture, not research data.

    Returns:
        HotpotQA-shaped dictionary with invented entities and known sentences.
    """
    return {'id':'fixture','question':'Where is the school attended by Ada?', 'answer':'Lima',
            'context':{'title':['Ada','Orion School'],'sentences':[['Ada attended Orion School in 1999.'],['Orion School is located in Lima.']]},
            'supporting_facts':{'title':['Ada','Orion School'],'sent_id':[0,0]},
            '_provenance':{'dataset':'synthetic_unit_fixture'}}


@pytest.fixture
def state(question: dict) -> DecisionState:
    """Build a typed graph fixture for graph, budget, and target-boundary tests.

    Args:
        question: Synthetic HotpotQA-shaped fixture.
    Returns:
        Two intermediate nodes and a label-free final placeholder.
    """
    facts,evidence=supporting_sentences(question)
    return build_state(question['id'],question['question'],facts,evidence)


def test_supporting_loader(question: dict) -> None:
    """Verify supporting references map to actual context sentences.

    Args:
        question: Synthetic loader fixture.
    Returns:
        None; assertions raise if loader offsets or source links are incorrect.
    """
    facts,evidence=supporting_sentences(question)
    assert len(facts)==2 and evidence['doc0']['title']=='Ada'
    question['supporting_facts']['sent_id'][0]=99
    with pytest.raises(ValueError):supporting_sentences(question)


def test_graph_direction_and_features(state: DecisionState) -> None:
    """Verify explicit-name links point forward and descendants are unique.

    Args:
        state: Two-hop fixture graph.
    Returns:
        None; validates shape, direction, and pre-action feature values.
    """
    graph=validate_graph(state)
    assert graph.has_edge('c0','c1') and not graph.has_edge('c1','c0')
    x,edges,ids=node_features(state)
    assert x.shape==(3,14) and edges.shape[0]==2
    assert x[0,2]==2 and ids[-1]=='final'


@pytest.mark.parametrize('problem',['cycle','missing_node','duplicate','missing_evidence','unproven_causal'])
def test_graph_rejects_bad_structure(state: DecisionState, problem: str) -> None:
    """Reject malformed or falsely certified dependency graphs.

    Args:
        state: Fresh valid graph fixture.
        problem: Parametrized structural error deliberately introduced by the test.
    Returns:
        None; each corruption must cause ValueError.
    """
    if problem=='cycle':state.edges.append({'source':'c1','target':'c0','kind':'test','confidence':.5})
    elif problem=='missing_node':state.edges[0]['target']='absent'
    elif problem=='duplicate':state.claims[1].id='c0'
    elif problem=='missing_evidence':state.claims[0].evidence_ids=['absent']
    else:state.edges[0]['verified_causal']=True
    with pytest.raises(ValueError):validate_graph(state)


def test_policy_state_rejects_labels(state: DecisionState) -> None:
    """Ensure outcome/reference keys cannot accidentally enter the state schema.

    Args:
        state: Valid fixture pre-state.
    Returns:
        None; forbidden reference fields must fail parsing.
    """
    data=state.to_dict();data['reference_answer']='Lima'
    with pytest.raises(ValueError):DecisionState.from_dict(data)


def test_injections_are_changed_and_reviewed(state: DecisionState) -> None:
    """Check that primary and unreviewed error candidates remain distinct.

    Args:
        state: Annotation-shaped fixture with a year and copular assertion.
    Returns:
        None; tests transformation mechanics without asserting real-world validity.
    """
    primary,review=select_conditions(state)
    assert {'incorrect_date','incorrect_number','negation_change'}=={c['type'] for c in primary}
    assert all(c['modified']!=c['original'] for c in primary)
    assert all(c['validity']=='human_review_required' for c in review)


@pytest.mark.parametrize('calls,tokens,latency,expected',[(0,None,None,0),(1,None,None,1),(3,0,None,0),(3,None,0,0),(3,None,20,1)])
def test_budget_constraints(state: DecisionState, calls: int, tokens: int|None, latency: float|None, expected: int) -> None:
    """Verify hard call/token/estimated-latency caps in the selection algorithm.

    Args:
        state: Candidate fixture.
        calls: Allowed checks.
        tokens: Optional token cap.
        latency: Optional predicted latency cap in milliseconds.
        expected: Expected number of selected candidates.
    Returns:
        None; no future outcomes are used in these selection assertions.
    """
    assert len(select_claims(state,{'c0':.8,'c1':.6},Budget(calls,tokens,latency)))==expected


def test_invalid_budget() -> None:
    """Reject negative budgets before any action or model call.

    Returns:
        None; constructor must raise ValueError.
    """
    with pytest.raises(ValueError):Budget(-1)


def test_cost_sensitive_stopping(state: DecisionState) -> None:
    """Prefer no action when predicted improvement is below its declared cost.

    Args:
        state: Synthetic pre-action graph without outcome labels.
    Returns:
        None; the optional objective-aware variant can spend less than its cap.
    """
    assert select_claims(state,{'c0':.005,'c1':-.1},Budget(2),minimum_net_benefit=.01)==[]
    assert select_claims(state,{'c0':.5,'c1':-.1},Budget(2),minimum_net_benefit=.01)==['c0']


@pytest.mark.parametrize('calls',[1.5,True,'1'])
def test_fractional_call_budget_rejected(calls: object) -> None:
    """Prevent noninteger budgets from rounding upward inside a greedy loop.

    Args:
        calls: Deliberately invalid runtime input despite the typed interface.
    Returns:
        None; call limits must reject ambiguous or fractional values.
    """
    with pytest.raises(ValueError):Budget(calls)


def test_nonfinite_budget_rejected() -> None:
    """Reject NaN latency and fractional estimated-token budgets.

    Returns:
        None; missing finite cost constraints cannot silently enter selection.
    """
    with pytest.raises(ValueError):Budget(1,latency_ms=float('nan'))
    with pytest.raises(ValueError):Budget(1,tokens=2.5)


def test_support_cap_covers_second_document(question: dict) -> None:
    """Keep the second annotated hop when the first document has many facts.

    Args:
        question: Synthetic two-document fixture with extra first-document facts.
    Returns:
        None; the three-node cap must still represent both evidence sources.
    """
    question['context']['sentences'][0]=['Ada attended Orion School.','Ada was born in 1999.','Ada won 2 medals.']
    question['supporting_facts']={'title':['Ada','Ada','Ada','Orion School'],'sent_id':[0,1,2,0]}
    facts,evidence=supporting_sentences(question,limit=3)
    assert len(facts)==3 and len({f['evidence_id'] for f in facts})==2


def test_baselines_reproducible(state: DecisionState) -> None:
    """Compare random/risk/dependency rankings under deterministic visible inputs.

    Args:
        state: Visible fixture, not an episode containing labels.
    Returns:
        None; priorities and node coverage are checked independently of outcomes.
    """
    state.claims[0].risk=.9;state.claims[1].risk=.2
    assert baseline_scores(state,'random',42)==baseline_scores(state,'random',42)
    assert baseline_scores(state,'uncertainty')['c0']==.9
    assert baseline_scores(state,'dependency')['c0']>baseline_scores(state,'dependency')['c1']
    with pytest.raises(ValueError):baseline_scores(state,'oracle')


def test_actions_do_not_mutate_source(state: DecisionState) -> None:
    """Ensure intervention application preserves the pre-action comparison state.

    Args:
        state: Original fixture.
    Returns:
        None; corrected copy must differ while original remains unchanged.
    """
    changed=apply_actions(state,[{'claim_id':'c0','replacement':'New claim.','status':'claim_corrected'}])
    assert changed.claims[0].text=='New claim.' and state.claims[0].text!='New claim.'


def test_metrics_and_missing_scope() -> None:
    """Check exact match, regression error, and paired improvement mathematics.

    Returns:
        None; broader downstream errors remain null rather than fabricated zero.
    """
    assert exact_match('The Lima.','Lima') and not exact_match('','Lima')
    result=prediction_metrics([1.,0.],[0.,0.]);assert result['mae']==.5 and result['spearman'] is None
    frame=pd.DataFrame([{'method':'test','budget':1,'question_id':'q','correct':1,'baseline_correct':0,'calls':1,'estimated_tokens':20,'generation_seed':1}])
    summary=summarize_policies(frame,10)
    assert summary.iloc[0]['mean_gain']==1 and summary.iloc[0]['objective_loss_plus_cost']==.01
    assert summary.iloc[0]['broader_downstream_error_count'] is None


def test_question_splits_and_json(question: dict, tmp_path: Path) -> None:
    """Check disjoint question splits and Unicode JSONL round trips.

    Args:
        question: Synthetic row to clone with independent IDs.
        tmp_path: Pytest-managed writable directory.
    Returns:
        None; question leakage must be rejected.
    """
    train=[dict(question,id=f'train{i}') for i in range(5)]
    splits=split_questions(train,[dict(question,id='test')]);assert audit_splits(splits)['counts']['test']==1
    with pytest.raises(ValueError):split_questions(train,[dict(question,id='train0')])
    path=tmp_path/'records.jsonl';write_jsonl(path,[{'text':'हिन्दी'}]);assert read_jsonl(path)[0]['text']=='हिन्दी'


def episode_fixture(state: DecisionState, index: int = 0) -> dict:
    """Create labeled training mechanics fixtures, explicitly not measured results.

    Args:
        state: Synthetic visible graph.
        index: Distinct ID and alternating artificial training targets.
    Returns:
        Minimal episode dictionary accepted by the training-data adapter.
    """
    return {'pre_state':state.to_dict(),'baseline':{'correct':False},
            'singletons':{'c0':{'correct':bool(index%2)},'c1':{'correct':False}}}


def test_graph_tensor_and_cpu_checkpoint(state: DecisionState, tmp_path: Path) -> None:
    """Actually train a tiny CPU fixture model and verify checkpoint/resume paths.

    Args:
        state: Synthetic graph fixture.
        tmp_path: Writable test directory.
    Returns:
        None; this demonstrates training mechanics, not benchmark accuracy.
    """
    rows=[episode_fixture(state,i) for i in range(8)]
    x,e,y,mask=training_graph(rows[0]);assert mask.tolist()==[True,True,False]
    seed_everything(42);network=BenefitGNN();output=network(torch.tensor(x),torch.tensor(e));assert output.shape==(3,)
    path=tmp_path/'graph.pt'
    fit_gnn(rows,rows[:2],path,epochs=1,batch_size=3,device='cpu')
    scores=GraphPolicy(path).scores(state);assert set(scores)=={'c0','c1'}
    fit_gnn(rows,rows[:2],path,epochs=2,batch_size=3,device='cpu',resume=True)
    assert path.with_suffix('.last.pt').exists()
    tab=tmp_path/'tab.joblib';fit_tabular(rows,rows[:2],tab)
    assert set(TabularPolicy(tab).scores(state))=={'c0','c1'}


def test_resources_cpu() -> None:
    """Verify explicit CPU execution and machine-report schema without downloads.

    Returns:
        None; hardware may legitimately have no CUDA device.
    """
    assert str(choose_device('cpu'))=='cpu'
    assert 'cuda_available' in hardware_report()


def test_oom_retry_mechanics(state: DecisionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate one OOM to test retry mechanics; this is not a GPU performance test.

    Args:
        state: Synthetic graph fixture for actual tiny CPU training after retry.
        tmp_path: Writable checkpoint folder.
        monkeypatch: Pytest fixture temporarily injecting one explicit OOM failure.
    Returns:
        None; fallback must produce a real checkpoint rather than a fake score.
    """
    original=BenefitGNN.forward
    attempts={'n':0}
    def fail_once(self: BenefitGNN, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raise one controlled error, then delegate to the actual network.

        Args:
            self: Tiny network currently being trained.
            x: Actual fixture node features.
            edge_index: Actual directed fixture edges.
        Returns:
            Real network predictions after the one deliberate failure.
        """
        attempts['n']+=1
        if attempts['n']==1:raise torch.cuda.OutOfMemoryError('unit-test injected OOM')
        return original(self,x,edge_index)
    monkeypatch.setattr(BenefitGNN,'forward',fail_once)
    rows=[episode_fixture(state,i) for i in range(4)]
    path=tmp_path/'retry.pt'
    result=fit_gnn(rows,rows[:2],path,epochs=1,batch_size=2,device='cpu')
    assert result['oom_fallback'] is True and path.exists()
