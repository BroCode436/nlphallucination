"""Command-line entry point for real staged CHC experiments.

Example: python -m chc.cli run --stage smoke --project /path/to/existing/project
Outputs never overwrite the original detector's results or pretrained weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import pandas as pd
from .data import audit_splits, fetch_hotpot, load_ragtruth, read_jsonl, split_questions
from .resources import hardware_report, seed_everything


def main() -> None:
    """Parse reproducible run settings and execute the selected experiment stage.

    Returns:
        None; saves all data/checkpoints/results and prints progress.
    Raises:
        Explicit dependency, download, validation, or model errors instead of
        silently falling back to synthetic experimental performance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run','validate','hardware','serve'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--stage', choices=['smoke','pilot'], default='smoke')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume-training', action='store_true')
    parser.add_argument('--port',type=int,default=8766)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    config = json.loads((args.config or package/'configs'/'chc.json').read_text(encoding='utf-8'))
    work = args.work or args.project/'outputs'/'chc'
    work.mkdir(parents=True, exist_ok=True)
    if args.command=='hardware':
        print(json.dumps(hardware_report(work/'hardware_report.json'),indent=2))
        return
    if args.command=='serve':
        from .web import serve
        serve(args.project,work,args.port)
        return
    stage = work/args.stage
    stage.mkdir(exist_ok=True)
    from .experiment import collect_episodes, evaluate_detector, evaluate_policies, train_policies, validate_episodes
    if args.command=='validate':
        print(json.dumps(validate_episodes(read_jsonl(stage/'episodes.jsonl')),indent=2))
        return
    os.environ.setdefault('HF_HOME', str(work/'model_cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    seed_everything(42)
    before = hardware_report(work/'hardware_report.json')
    config['pre_download_memory_estimate']={'reasoner_fp32_weights_mb':320,'reasoner_inference_budget_mb':1800,
                                           'detector_weights_mb':268,'available_vram_bytes':before['gpus'][0]['free_bytes'] if before['gpus'] else None,
                                           'large_model_download':False}
    (stage/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    limits = config[args.stage]
    # Cache at least 500 actual rows from each official split to support bounded
    # later runs. The stage below still uses only its explicitly declared subset.
    # Pilot questions exclude the earlier smoke questions. The 500-question pilot
    # is the held-out evaluation set, with separate training/tuning questions.
    train_offset = config['smoke']['official_train_questions'] if args.stage=='pilot' else 0
    test_offset = config['smoke']['official_validation_questions'] if args.stage=='pilot' else 0
    train = fetch_hotpot(work/'hotpot','train',max(500,limits['official_train_questions']+train_offset))[train_offset:train_offset+limits['official_train_questions']]
    test = fetch_hotpot(work/'hotpot','validation',max(500,limits['official_validation_questions']+test_offset))[test_offset:test_offset+limits['official_validation_questions']]
    splits = split_questions(train,test)
    (stage/'split_audit.json').write_text(json.dumps(audit_splits(splits),indent=2),encoding='utf-8')
    from .detector import ExistingDetector
    from .generation import SmallReasoner
    detector = ExistingDetector(args.project,config['device'])
    reasoner = SmallReasoner(work/'reasoner',config['reasoner_model'],config['device'],
                             config['max_input_tokens'],config['max_new_tokens'],config['sample_generation'])
    records = collect_episodes(splits,detector,reasoner,stage/'episodes.jsonl',config['generation_seeds'],config['max_conditions'])
    audit = validate_episodes(records)
    (stage/'episode_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    detector_report = evaluate_detector(detector,load_ragtruth(args.project),stage/'detector')
    training = train_policies(records,stage/'models',config['training_seeds'],config['graph_epochs'],
                              config['graph_batch_size'],config['gradient_accumulation'],config['device'],args.resume_training)
    all_frames, summaries = [], []
    for seed in config['training_seeds']:
        frame, summary = evaluate_policies(records,stage/'models',stage/f'eval_seed_{seed}',seed,config['budgets'],config['cost_lambda'])
        all_frames.append(frame)
        summary['training_seed']=seed
        summaries.append(summary)
    pd.concat(all_frames).to_csv(stage/'all_policy_predictions.csv',index=False)
    pd.concat(summaries).to_csv(stage/'all_policy_summaries.csv',index=False)
    from .plots import make_plots
    make_plots(stage)
    manifest = {'stage':args.stage,'requested_questions':sum(len(v) for v in splits.values()),
                'questions_with_episodes':len({r['question_id'] for r in records}),'episodes':len(records),
                'primary_error_conditions':len({(r['question_id'],r['injection']['type']) for r in records}),
                'generation_seeds':config['generation_seeds'],'training_seeds':config['training_seeds'],
                'new_real_generations_this_invocation':reasoner.actual_generations,'cache_hits_this_invocation':reasoner.cache_hits,
                'test_policy_evaluations':sum(len(f) for f in all_frames),
                'cache_reuse_is_not_an_independent_sample':True,'broader_downstream_labels_available':False,
                'verification_executions_this_invocation':detector.verification_executions,
                'verification_cache_hits_this_invocation':detector.verification_cache_hits,
                'policy_calls_are_logical_replay_costs':True,
                'human_review_completed':False,'detector_fallbacks':detector.fallback_events,'reasoner_fallbacks':reasoner.fallback_events,
                'model_identifier':reasoner.model_id,'episode_sha256':hashlib.sha256((stage/'episodes.jsonl').read_bytes()).hexdigest(),
                'after_hardware':hardware_report()}
    (stage/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from .reporting import stage_report
    stage_report(stage)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
