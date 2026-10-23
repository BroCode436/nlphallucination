"""Summarize a completed real collection, then train/evaluate gated policy models."""

from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chc.data import read_jsonl, write_jsonl
from chc.v2.atomic import atomic_records
from chc.v2.analysis import summarize_collection, plot_collection, plot_policies
from chc.v2.evidence import EvidenceJudge
from chc.v2.learning import train_models, evaluate_models
from chc.v2.robustness import operational_evaluation,augment_denominators


def main() -> None:
    """Produce counts before model complexity decisions and preserve missing data.

    Use --collect-only for debugging. Full mode trains only on designated train
    rows, tunes on validation, and evaluates the untouched final test slice once.
    The graph/semantic gate is based on signal counts, never final test accuracy.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--historical", type=Path)
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--atomic-only", action="store_true")
    p.add_argument("--diagnostic-complex", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    summary = summarize_collection(args.stage)
    (args.stage / "collection_summary_sentence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.collect_only:
        plot_collection(args.stage)
        print(json.dumps(summary), flush=True)
        return
    records = (
        read_jsonl(args.stage / "episodes.jsonl")
        if (args.stage / "episodes.jsonl").exists()
        else []
    )
    judge = EvidenceJudge(args.cache)
    records = atomic_records(records, judge)
    write_jsonl(args.stage / "episodes_atomic_scored.jsonl", records)
    summary = summarize_collection(args.stage, records)
    (args.stage / "collection_summary_all_atomic.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.atomic_only:
        plot_collection(args.stage, records)
        print(json.dumps({"factual_unit_summary": summary}), flush=True)
        return
    # Eligibility depends only on the original clean prefix's evidence support,
    # never whether a later intervention helped or harmed. All raw records stay.
    records = [
        r
        for r in records
        if r["atomic_prefix_eligible"] and r["selected_corruption_validation"]["eligible"]
    ]
    write_jsonl(args.stage / "episodes_primary.jsonl", records)
    summary = summarize_collection(args.stage, records)
    (args.stage / "cohort_audit.json").write_text(
        json.dumps(
            {
                "raw_recorded_episodes": json.loads(
                    (args.stage / "collection_summary_all_atomic.json").read_text()
                )["episodes"],
                "primary_episodes": len(records),
                "eligibility_rule": "Clean atomic-prefix support and pre-outcome selected-corruption validity; no filtering on gains",
                "invalid_or_uncertain_records_retained_in": "episodes_atomic_scored.jsonl",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    training = train_models(
        records, judge, args.stage / "models", diagnostic_complex=args.diagnostic_complex
    )
    print(
        json.dumps(
            {"training_status": training.get("status"), "counts": training.get("target_counts")}
        ),
        flush=True,
    )
    evaluation = evaluate_models(
        records, judge, args.stage / "models", args.stage / "evaluation", args.historical
    )
    augment_denominators(records,args.stage/'evaluation')
    operational_evaluation(records,judge,args.stage/'models',args.stage/'evaluation',args.historical)
    plot_collection(args.stage, records)
    plot_policies(args.stage / "evaluation")
    print(
        json.dumps(
            {
                "test_questions": evaluation.get("complete_test_questions"),
                "coverage": evaluation.get("coverage"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
