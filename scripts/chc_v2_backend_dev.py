"""Benchmark real structured-output backends on isolated development questions."""

from pathlib import Path
import argparse, json, sys, gc, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chc.data import read_jsonl, supporting_sentences, write_jsonl
from chc.v2.backend import TraceBackend
from chc.v2.evidence import EvidenceJudge
from chc.v2.alignment import AlignedTraceBackend
from chc.metrics import exact_match
import torch


def main() -> None:
    """Compare models before policy training; save every failed output and timing.

    Development uses official training rows 440:460, isolated from the planned
    V2 policy splits. Existing FLAN weights are reused from the historical cache.
    Reference answers enter only post-generation scoring.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--hotpot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flan-cache", type=Path)
    p.add_argument("--aligned", action="store_true")
    p.add_argument("--constrained", action="store_true")
    args = p.parse_args()
    rows = read_jsonl(args.hotpot)[440:460]
    judge = EvidenceJudge(args.cache)
    report = []
    peaks = {}
    for model in ["Qwen/Qwen2.5-1.5B-Instruct", "google/flan-t5-small"]:
        cache = args.flan_cache if model.startswith("google") and args.flan_cache else args.cache
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        backend = TraceBackend(cache, model, max_new_tokens=280, constrained=args.constrained)
        if args.aligned:
            backend = AlignedTraceBackend(backend, judge)
        for i, row in enumerate(rows):
            facts, ev = supporting_sentences(row, 4)
            evidence = {f"E{k+1}": dict(v) for k, v in enumerate(ev.values())}
            # Use annotation-linked sentences as source passages to control input
            # length, while retaining original document titles and source IDs.
            for e in evidence.values():
                e["text"] = " ".join(f["text"] for f in facts if f["title"] == e["title"])
            result = backend.generate(row["question"], evidence, seed=101)
            judgments = judge.assess(result["trace"]["claims"], evidence) if result["trace"] else []
            report.append(
                {
                    "question_id": row["id"],
                    "model": model,
                    "generation": result,
                    "judgments": judgments,
                    "correct": (
                        exact_match(result["trace"]["final_answer"], row["answer"])
                        if result["trace"]
                        else None
                    ),
                }
            )
            print(model, i + 1, "valid", bool(result["trace"]), flush=True)
            write_jsonl(args.output / "backend_records.jsonl", report)
        peaks[model] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    summaries = []
    for model in sorted({r["model"] for r in report}):
        group = [r for r in report if r["model"] == model]
        valid = [r for r in group if r["generation"]["trace"]]
        js = [j for r in valid for j in r["judgments"]]
        summaries.append(
            {
                "model": model,
                "questions": len(group),
                "valid_artifact_rate": len(valid) / len(group),
                "raw_valid_json_rate": sum(
                    r["generation"].get(
                        "raw_validation_error", r["generation"].get("validation_error")
                    )
                    is None
                    for r in group
                )
                / len(group),
                "explicit_citation_alignment_enabled": args.aligned,
                "grammar_enabled": args.constrained,
                "accuracy_all_attempts": sum(bool(r["correct"]) for r in group) / len(group),
                "automatic_supported_fraction": (
                    sum(j["status"] == "supported" for j in js) / len(js) if js else None
                ),
                "mean_latency_ms": sum(r["generation"]["latency_ms"] for r in group) / len(group),
                "peak_allocated_gpu_bytes": peaks[model],
                "memory_scope": "Per-backend process peak including shared NLI judge",
            }
        )
    (args.output / "backend_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
