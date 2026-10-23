"""Evaluate explicit source-alignment yield using saved real development outputs."""

from pathlib import Path
import sys, argparse, json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chc.data import read_jsonl, write_jsonl
from chc.v2.evidence import EvidenceJudge
from chc.v2.alignment import align_uncited_artifact
from chc.v2.collection import source_evidence
from chc.metrics import exact_match
from chc.v2.traces import parse_trace


def main() -> None:
    """Reuse raw real outputs, record citation provenance, and check unchanged text."""
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--hotpot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    judge = EvidenceJudge(args.cache, device="cpu")
    questions = {r["id"]: r for r in read_jsonl(args.hotpot)}
    rows = []
    for r in read_jsonl(args.records):
        if not r["model"].startswith("Qwen/"):
            continue
        row = questions[r["question_id"]]
        ev = source_evidence(row, limit=4)
        raw=dict(r['generation'])
        try:raw['trace']=parse_trace(raw['raw'],ev,row['question']);raw['validation_error']=None
        except ValueError as exc:raw['trace']=None;raw['validation_error']=str(exc)
        for field in ('citation_alignment','artifact_origin','raw_validation_error'):raw.pop(field,None)
        g = align_uncited_artifact(raw, row["question"], ev, judge)
        js = judge.assess(g["trace"]["claims"], ev) if g["trace"] else []
        rows.append(
            {
                "question_id": row["id"],
                "generation": g,
                "judgments": js,
                "correct": (
                    exact_match(g["trace"]["final_answer"], row["answer"]) if g["trace"] else None
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "aligned_records.jsonl", rows)
    valid = [r for r in rows if r["generation"]["trace"]]
    js = [j for r in valid for j in r["judgments"]]
    report = {
        "questions": len(rows),
        "raw_valid": sum(r["generation"]["raw_validation_error"] is None for r in rows),
        "aligned_valid": len(valid),
        "aligned_yield": len(valid) / len(rows),
        "supported_two_claim_prefixes":sum(len(r['judgments'])>=2 and all(j['status']=='supported' for j in r['judgments'][:2]) for r in rows),
        "accuracy_all_attempts": sum(bool(r["correct"]) for r in rows) / len(rows),
        "supported_claim_fraction": (
            sum(j["status"] == "supported" for j in js) / len(js) if js else None
        ),
        "decision_scope": "Development only; source-aligned artifacts are distinct from raw model JSON; no claim text edited",
    }
    (args.output / "alignment_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
