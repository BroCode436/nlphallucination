"""Validate a supplied filled review sheet; keep automatic and reviewed labels apart."""

from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chc.v2.review import import_review


def main() -> None:
    """Require a reviewer ID and preserve existing annotation files on conflict."""
    p = argparse.ArgumentParser()
    p.add_argument("--worksheet", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reviewer", required=True)
    args = p.parse_args()
    print(json.dumps(import_review(args.worksheet, args.key, args.output, args.reviewer)))


if __name__ == "__main__":
    main()
