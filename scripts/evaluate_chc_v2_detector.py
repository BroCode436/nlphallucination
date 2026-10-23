"""Execute matched natural RAGTruth detector tests and validation-only calibration."""

from pathlib import Path
import argparse, sys, json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chc.v2.evidence import EvidenceJudge
from chc.v2.detection import evaluate_detection
from chc.v2.analysis import plot_detection


def main() -> None:
    """Use cached NLI weights and real processed RAGTruth rows; persist all scores."""
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    judge = EvidenceJudge(args.cache, device=args.device)
    report = evaluate_detection(ROOT, judge, args.output)
    plot_detection(args.output)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "method",
                        "evaluation",
                        "f1",
                        "pr_auc_average_precision",
                        "ece_10_bins",
                    )
                }
                for r in report["results"]
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
