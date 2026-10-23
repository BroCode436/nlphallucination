"""Generate PNG/PDF research figures exclusively from measured experiment files.

These plots show final-node outcomes, not a fully annotated multi-step cascade.
Every figure caption states that limitation where it matters.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from .data import read_jsonl
from .graph import validate_graph
from .schema import DecisionState


def save_figure(figure: plt.Figure, directory: Path, name: str, caption: str) -> None:
    """Save one actual-results figure in raster/vector formats with a caption file.

    Args:
        figure: Fully populated Matplotlib figure.
        directory: Output directory, created if missing.
        name: Basename without file extension.
        caption: Scientific interpretation and scope, saved alongside the image.
    Returns:
        None; closes the figure to release plotting memory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory/f'{name}.png',dpi=200,bbox_inches='tight')
    figure.savefig(directory/f'{name}.pdf',bbox_inches='tight')
    (directory/f'{name}.caption.txt').write_text(caption,encoding='utf-8')
    plt.close(figure)


def make_plots(stage: Path) -> None:
    """Render ten families of plots without hardcoded performance values.

    Args:
        stage: Completed experiment folder containing tables, episodes, and detector results.
    Returns:
        None; saves plots and human-readable captions in stage/figures.
    Limitations:
        Training-seed means are descriptive; confidence intervals come from the
        evaluation CSV's question-cluster bootstrap, not from bar aesthetics.
    """
    figures = stage/'figures'
    summary = pd.read_csv(stage/'all_policy_summaries.csv')
    one = summary[summary.budget==1].groupby('method').mean(numeric_only=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for column,name,label in [('final_accuracy','final_answer_accuracy','Final-answer exact match'),
                              ('mean_final_node_error','downstream_final_node_error','Mean final-node factual loss'),
                              ('final_node_cascade_reduction','cascade_reduction','Reduction in final-node errors'),
                              ('mean_gain','ablation_comparison','Accuracy gain versus no intervention')]:
        fig,ax=plt.subplots(figsize=(10,5))
        values=one[column].fillna(0)
        ax.barh(values.index,values.values,color=['#225e9e' if 'chc' in n else '#849bb3' for n in values.index])
        ax.set_xlabel(label);ax.set_title('One-call budget; measured held-out outcomes')
        save_figure(fig,figures,name,'Final-answer-only loss; broader downstream claims are not annotated. Means across training seeds; oracle is analysis-only.')
    fig,ax=plt.subplots(figsize=(9,5))
    for method,group in summary.groupby('method'):
        if method.startswith('without') or method=='oracle_analysis_only':continue
        curve=group.groupby('budget')['final_accuracy'].mean()
        ax.plot(curve.index,curve.values,marker='o',label=method)
    ax.set_xlabel('Maximum verification calls');ax.set_ylabel('Final-answer exact match');ax.legend(fontsize=8)
    save_figure(fig,figures,'budget_accuracy','Hard call budgets; actual calls and estimated token costs are in the tables. Joint effects use generated outcomes, not added singleton gains.')
    benefits=pd.concat([pd.read_csv(p) for p in stage.glob('eval_seed_*/benefit_predictions.csv')],ignore_index=True)
    benefits['absolute_error']=(benefits.prediction-benefits.observed).abs()
    fig,ax=plt.subplots(figsize=(7,4))
    errors=benefits.groupby('method')['absolute_error'].mean()
    ax.bar(errors.index,errors.values,color='#225e9e');ax.set_ylabel('MAE of predicted final-error reduction')
    save_figure(fig,figures,'benefit_prediction_error','Held-out measured singleton benefit targets; negative targets represent harmful interventions.')
    detector=json.loads((stage/'detector'/'detector_metrics.json').read_text())
    fig,ax=plt.subplots(figsize=(5,4))
    matrix=np.asarray(detector['confusion_matrix'])
    im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax)
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center')
    ax.set_xticks([0,1],['supported','hallucinated']);ax.set_yticks([0,1],['supported','hallucinated'])
    ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
    save_figure(fig,figures,'detector_confusion','Actual saved DistilBERT predictions on the sampled RAGTruth test claims; not a causal benchmark.')
    fig,ax=plt.subplots(figsize=(5,4));bins=detector['calibration_bins']
    ax.plot([0,1],[0,1],linestyle='--',color='gray')
    ax.plot([b['mean_score'] for b in bins],[b['positive_fraction'] for b in bins],marker='o')
    ax.set_xlabel('Mean raw hallucination score');ax.set_ylabel('Observed hallucination fraction')
    save_figure(fig,figures,'detector_calibration','Ten score bins; small samples and rare positives make bin fractions noisy. No calibration fitted on test.')
    episodes=read_jsonl(stage/'episodes.jsonl')
    example=next(r for r in episodes if r['split']=='test')
    state=DecisionState.from_dict(example['pre_state']);graph=validate_graph(state)
    fig,ax=plt.subplots(figsize=(9,5));positions=nx.spring_layout(graph,seed=42)
    labels={c.id:c.id+'\n'+textwrap.shorten(c.text,width=45) for c in state.claims}
    nx.draw_networkx(graph,pos=positions,labels=labels,ax=ax,node_color='#c9e0f5',node_size=2200,font_size=7,arrowsize=20)
    ax.set_axis_off();ax.set_title('Textual dependency hypotheses — not verified causal edges')
    save_figure(fig,figures,'example_dependency_graph','Edges encode annotated relevance or explicit-name hypotheses, never observed causal truth.')
    fig,ax=plt.subplots(figsize=(11,6));ax.set_axis_off()
    best=max(example['joint_outcomes'].items(),key=lambda kv:int(kv[1]['correct']))
    lines=[f"Question: {example['question']}",f"Injected note: {example['injection']['modified']}",
           f"Before: {example['baseline']['answer']} (EM={int(example['baseline']['correct'])})",
           f"Example action subset: {best[0] or 'none'}",f"After: {best[1]['answer']} (EM={int(best[1]['correct'])})",
           f"Reference (evaluation only): {example['reference_answer']}",
           'The displayed subset is selected using observed outcomes for explanation, not a deployable selection policy.']
    ax.text(.02,.98,'\n\n'.join(textwrap.fill(s,105) for s in lines),va='top',fontsize=11)
    save_figure(fig,figures,'intervention_case_study','Actual before/after outputs. An unchanged or unsuccessful case is retained rather than replaced with a fabricated success.')
