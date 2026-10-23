"""Import explicitly supplied review labels without overwriting automatic results.

Blank worksheets are not completed reviews. A reviewer identifier and valid label
are required. This tool records supplied annotations; it cannot certify that a
human performed them. Review-sample metrics are not population prevalence.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import pandas as pd
from ..data import write_jsonl


def import_review(
    worksheet: Path, machine_key: Path, output: Path, reviewer: str
) -> dict[str, Any]:
    """Validate supplied annotations and save them separately with provenance.

    Args:
        worksheet: Filled CSV derived from human_review_blank.csv; empty labels
            remain unreviewed and are not converted to factual judgments.
        machine_key: Matching IDs/episode/variant mapping from the same collection.
        output: New annotation JSONL; never replaces raw machine judgments.
        reviewer: Explicit identifier for annotation accountability.
    Returns:
        Imported and pending counts with automatic-versus-reviewed agreement.
        Agreement describes the reviewed workload only, not benchmark accuracy.
    Raises:
        ValueError: Missing reviewer, duplicate/unknown IDs, invalid labels or
            required columns. Existing conflicting annotation files are preserved.
    """
    if not reviewer.strip():
        raise ValueError("Supply an explicit reviewer identifier")
    frame = pd.read_csv(worksheet, keep_default_na=False)
    key = pd.read_csv(machine_key, keep_default_na=False)
    if not {"review_id", "human_status", "notes"} <= set(frame.columns):
        raise ValueError("Review columns are missing")
    if frame.review_id.duplicated().any() or key.review_id.duplicated().any():
        raise ValueError("Duplicate review IDs")
    if not set(frame.review_id) <= set(key.review_id):
        raise ValueError("Review IDs do not match the collection key")
    allowed = {"supported", "contradicted", "insufficient_evidence", "unreviewed"}
    records = []
    lookup = key.set_index("review_id").to_dict(orient="index")
    for row in frame.to_dict(orient="records"):
        label = str(row["human_status"]).strip().lower()
        if not label or label == "unreviewed":
            continue
        if label not in allowed:
            raise ValueError("Unknown factual status " + label)
        records.append(
            {
                "review_id": row["review_id"],
                **lookup[row["review_id"]],
                "reviewed_status": label,
                "reviewer": reviewer,
                "notes": row["notes"],
                "atomicity": row.get("human_atomicity", ""),
                "provenance": "Explicitly supplied review CSV; importer does not certify human origin",
            }
        )
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        proposed = "".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records
        )
        if existing != proposed:
            raise ValueError("Use a new output file to preserve prior reviewer annotations")
    write_jsonl(output, records)
    report = {
        "imported": len(records),
        "pending_or_blank": len(frame) - len(records),
        "reviewer": reviewer,
        "agreement_on_reviewed_workload": (
            sum(r["reviewed_status"] == r["machine_status"] for r in records) / len(records)
            if records
            else None
        ),
        "scope": "Supplied review workload only; no population performance claim",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
