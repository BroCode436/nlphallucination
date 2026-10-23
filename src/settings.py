"""Shared configuration and reproducibility helpers; no model downloads on import.
"""
from typing import Any as _DocAny
import json
import random
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def load_config(path: _DocAny=None) -> _DocAny:
    """Load config.

    Loads configuration and initializes reproducible random generators.

    Learning notes — input/output contract:
        Parameters:
            path: Local input/output path selected by the caller.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    with open(path or ROOT / 'configs' / 'config.json', encoding='utf-8') as stream:
        return json.load(stream)


def seed_everything(seed: int=42) -> None:
    """Seed everything.

    Loads configuration and initializes reproducible random generators.

    Learning notes — input/output contract:
        Parameters:
            seed: Pseudorandom seed controlling reproducible sampling or model initialization.
        Returns:
            None; effects or assertions are performed directly.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
