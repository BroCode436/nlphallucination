"""CLI: python verify_correction.py --original ... --evidence ... [--candidate ...]
"""
from typing import Any as _DocAny
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from correction_validity import CorrectionValidityModel
def main() -> None:
    """Main.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            None; effects or assertions are performed directly.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',required=True);parser.add_argument('--evidence',required=True)
    parser.add_argument('--candidate');args=parser.parse_args();model=CorrectionValidityModel()
    result=model.assess(args.evidence,args.original,args.candidate) if args.candidate else model.propose(args.evidence,args.original)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
