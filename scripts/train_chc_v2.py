"""Train fixed V2 intervention models after train/validation collection completes.

Final-test records are explicitly excluded before any factual-unit processing.
This supports freezing model checkpoints while final-test generation continues.
"""

from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chc.data import read_jsonl, write_jsonl
from chc.v2.atomic import atomic_records
from chc.v2.evidence import EvidenceJudge
from chc.v2.learning import train_models


def main() -> None:
    """Require complete tuning attempts; fit using train/validation data only.

    --judge-device cpu avoids competing with an active GPU generation run.
    Graph networks retain their existing small CUDA-aware trainer. The explicit
    diagnostic flag permits fixed architectures as exploratory fits when the
    conservative count gate fails; this is recorded, never hidden as full power.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--judge-device", default="cpu")
    p.add_argument("--diagnostic-complex", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((args.stage / "protocol.json").read_text())
    attempts = read_jsonl(args.stage / "attempts.jsonl")
    for split in ("train", "validation"):
        done = {
            a["question_id"]
            for a in attempts
            if a["split"] == split and a.get("collection_complete")
        }
        if not set(protocol["questions"][split]) <= done:
            raise ValueError("Complete " + split + " collection before model fitting")
    # Ignore an in-progress final JSONL line when generation appends concurrently.
    lines = (args.stage / "episodes.jsonl").read_bytes().split(b"\n")[:-1]
    raw = [
        r
        for r in (json.loads(line) for line in lines if line)
        if r["split"] in ("train", "validation")
    ]
    judge = EvidenceJudge(args.cache, device=args.judge_device)
    scored = atomic_records(raw, judge)
    write_jsonl(args.stage / "training_records_atomic_audit.jsonl", scored)
    primary = [
        r
        for r in scored
        if r["atomic_prefix_eligible"] and r["selected_corruption_validation"]["eligible"]
    ]
    write_jsonl(args.stage / "training_records_primary.jsonl", primary)
    report = train_models(
        primary, judge, args.stage / "models", diagnostic_complex=args.diagnostic_complex
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "training_questions": len(
                    {r["question_id"] for r in primary if r["split"] == "train"}
                ),
                "test_records_used_for_fitting": 0,
                "complex_mode": report.get("complex_execution_mode"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
