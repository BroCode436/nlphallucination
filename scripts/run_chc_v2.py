"""Run staged CHC V2 collection; real datasets and models are required."""

from pathlib import Path
import argparse, json, sys, time, hashlib, os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chc.data import read_jsonl, fetch_hotpot, audit_splits
from chc.v2.backend import TraceBackend
from chc.v2.alignment import AlignedTraceBackend
from chc.v2.evidence import EvidenceJudge
from chc.v2.repair import RepairVerifier
from chc.v2.collection import collect
import torch


def main() -> None:
    """Select disjoint source rows, record protocol, then collect resumable outputs.

    Debug uses 20 official train rows 460:480. Pilot uses train 200:400,
    validation 400:440, and new official-validation 510:570. These are bounded
    prefixes/slices, not representative natural hallucination distributions.
    Backend selection uses separate rows 440:460 and never final test outcomes.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--hotpot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["debug", "pilot"], default="debug")
    p.add_argument("--seeds", nargs="+", type=int, default=[101])
    p.add_argument("--limit", type=int)
    p.add_argument("--detector-report", type=Path)
    p.add_argument('--reasoner',choices=['Qwen/Qwen2.5-1.5B-Instruct','google/flan-t5-small'],default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--reasoner-cache',type=Path)
    p.add_argument('--max-new-tokens',type=int,default=280)
    p.add_argument(
        "--visibility",
        choices=["full", "masked_first_source"],
        default=os.environ.get("CHC_V2_VISIBILITY", "full"),
    )
    args = p.parse_args()
    if not 16<=args.max_new_tokens<=512:p.error('Use an output cap between 16 and 512 tokens')
    train = read_jsonl(args.hotpot / "hotpot_train_500.jsonl")
    if args.stage == "debug":
        splits = {"development": train[460:480]}
    else:
        test = fetch_hotpot(args.hotpot, "validation", 570)
        splits = {"train": train[200:400], "validation": train[400:440], "test": test[510:570]}
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    # Visibility is an explicit experimental factor, never selected from observed
    # gains. Full-evidence and compressed-context runs use separate directories.
    visibility = args.visibility
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "stage": args.stage,
        "seeds": args.seeds,
        "questions": {k: [r["id"] for r in v] for k, v in splits.items()},
        "loss_weights": {"downstream": 0.5, "final": 0.5, "harmful_repair": 0.25},
        "cost_lambda": 0.01,
        "conditions_per_question": 1,
        "max_new_tokens": args.max_new_tokens,
        "judge_threshold": 0.85,
        "artifact_pipeline": "Raw model JSON plus explicit citation alignment; malformed and unsupported artifacts rejected",
        "source_sampling": "Bounded slices; controlled challenge-set error distribution",
        "split_audit": audit_splits(splits),
    }
    protocol["risk_detector"] = (
        "NLI contradiction proxy"
        if args.detector_report is None
        else {
            "selection": "Validation-selected calibrated DistilBERT; RAGTruth-to-HotpotQA domain shift",
            "report_sha256": hashlib.sha256(args.detector_report.read_bytes()).hexdigest(),
        }
    )
    if args.reasoner!='Qwen/Qwen2.5-1.5B-Instruct':protocol['reasoning_backend']=args.reasoner
    if args.reasoner_cache:protocol['reasoner_cache_override']=str(args.reasoner_cache.resolve())
    protocol["continuation_visibility"] = visibility
    protocol["source_sha256"] = {
        name: hashlib.sha256((ROOT / "chc/v2" / name).read_bytes()).hexdigest()
        for name in (
            "backend.py",
            "alignment.py",
            "collection.py",
            "repair.py",
            "propagation.py",
            "corruptions.py",
            "validity.py",
        )
    }
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing run protocol differs; use a new output directory")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    judge = EvidenceJudge(args.cache)
    backend = AlignedTraceBackend(TraceBackend(args.reasoner_cache or args.cache,model_id=args.reasoner,max_new_tokens=args.max_new_tokens), judge)
    repair = RepairVerifier(ROOT, judge)
    risk = None
    if args.detector_report:
        from chc.v2.detection import CalibratedDetector

        risk = CalibratedDetector(ROOT, args.detector_report)
    start = time.perf_counter()
    records = collect(
        splits,
        backend,
        judge,
        repair,
        args.output,
        args.seeds,
        risk_detector=risk,
        continuation_visibility=visibility,
    )
    manifest = {
        "episodes": len(records),
        "process_elapsed_seconds": time.perf_counter() - start,
        "new_generations_this_invocation": backend.actual_generations,
        "cache_hits_this_invocation": backend.cache_hits,
        "backend": backend.model_id,
        "snapshot_ids": backend.snapshot_ids,
        "fallbacks": backend.fallbacks + judge.fallbacks,
        "peak_allocated_gpu_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }
    (args.output / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
