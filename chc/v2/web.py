"""Expose recorded propagation experiments and optional real local live inference.

Recorded examples require no model loading. Live inference uses supplied or
locally retrieved evidence and never invents a reference answer for evaluation.
The old application and routes remain available.
"""

from __future__ import annotations
import json, os, threading
from pathlib import Path
from typing import Any


class PropagationService:
    """Serve measured examples and label-free recommendations through one API.

    Args:
        project: Existing project containing source, checkpoints and V2 outputs.
    Attributes:
        lock: Serializes RNG/model use across live requests.
        models: Lazy inference resources; recorded requests do not allocate GPU.
    """

    def __init__(self, project: Path) -> None:
        """Locate configured results and defer heavyweight imports until needed."""
        self.project = project
        self.lock = threading.Lock()
        self.models = None
        self.results = Path(os.environ.get("CHC_V2_RESULTS", str(project / "outputs/chc_v2/pilot")))

    def examples(self) -> dict[str, Any]:
        """List actual stored benchmark questions without exposing gold to a policy."""
        path = self.results / "episodes.jsonl"
        records = (
            [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            if path.exists()
            else []
        )
        return {
            "examples": [
                {
                    "episode_id": r["episode_id"],
                    "question": r["question"],
                    "error_type": r["injection"]["type"],
                }
                for r in records[:100]
            ],
            "scope": "Recorded controlled challenge examples; automatic NLI labels; human review pending",
        }

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a recorded example or generate actual live factual continuations.

        Recorded mode exposes reference scoring after returning the already
        collected output. Live mode receives no reference and leaves final-answer
        accuracy unknown. Submitted text and evidence have explicit size limits.
        """
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        if data.get("mode") == "examples":
            return self.examples()
        episode_id = data.get("episode_id")
        if episode_id:
            path = (
                self.results / "episodes_atomic_scored.jsonl"
                if (self.results / "episodes_atomic_scored.jsonl").exists()
                else self.results / "episodes.jsonl"
            )
            if not path.exists():
                raise ValueError("No completed V2 episodes are installed")
            records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            record = next((r for r in records if r["episode_id"] == episode_id), None)
            if record is None:
                raise ValueError("Unknown recorded episode ID")
            from ..schema import DecisionState
            from ..policy import baseline_scores

            from .graph_features import enrich_source_bridges

            state = enrich_source_bridges(DecisionState.from_dict(record["pre_state"]))
            scores = baseline_scores(state, "uncertainty")
            policy = "highest_risk"
            path = self.results / "models/tabular_richer_all_42.joblib"
            if path.exists():
                from ..policy import TabularPolicy

                scores = TabularPolicy(path).scores(state)
                policy = "tabular_richer_all"
            selected = max(scores, key=scores.get)
            # Link actual fetched source material without changing experimental
            # prompts or stored evidence. The provenance URL is the real dataset
            # API page used to retrieve this question; row index identifies it.
            source_url = record.get("provenance", {}).get("url", "")
            for outcome in (record["baseline"], record["singletons"][selected]):
                for judgment in outcome.get("judgments", []) + outcome.get("atomic_judgments", []):
                    for passage in judgment.get("evidence", []):
                        if source_url:
                            passage["url"] = source_url
                        passage["dataset_row_index"] = record.get("provenance", {}).get("row_index")
            for passage in record["verification_actions"][selected].get("evidence", []):
                if source_url:
                    passage["url"] = source_url
            return {
                "mode": "recorded",
                "question": record["question"],
                "prefix": record["corrupted_prefix"],
                "pre_state": state.to_dict(),
                "before": record["baseline"],
                "after": record["singletons"][selected],
                "selected": selected,
                "policy": policy,
                "action": record["verification_actions"][selected],
                "injection": record["injection"],
                "evaluation_available": True,
                "warning": "Controlled injection; automatic factual proxy labels; uncertain dependency edges; human review pending",
            }
        question = data.get("question", "")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Enter a question up to 2,000 characters")
        supplied = data.get("evidence", [])
        if not isinstance(supplied, list) or not 1 <= len(supplied) <= 5:
            raise ValueError("Supply one to five source passages for live inference")
        evidence = {}
        for i, e in enumerate(supplied):
            if (
                not isinstance(e, dict)
                or not isinstance(e.get("text"), str)
                or not 1 <= len(e["text"]) <= 5000
            ):
                raise ValueError("Each source needs 1–5,000 characters of text")
            evidence[f"E{i+1}"] = {
                "text": e["text"],
                "title": str(e.get("title", f"Source {i+1}"))[:200],
                "source": "User supplied; not authenticated",
                "url": str(e.get("url", ""))[:2000],
            }
        with self.lock:
            if self.models is None:
                from .backend import TraceBackend
                from .alignment import AlignedTraceBackend
                from .evidence import EvidenceJudge
                from .repair import RepairVerifier

                cache = Path(
                    os.environ.get("CHC_V2_CACHE", str(self.project / "outputs/chc_v2/cache"))
                )
                judge = EvidenceJudge(cache)
                backend = AlignedTraceBackend(TraceBackend(cache), judge)
                repair = RepairVerifier(self.project, judge)
                risk = None
                report = self.results.parent / "detector/detector_report.json"
                if report.exists():
                    from .detection import CalibratedDetector

                    risk = CalibratedDetector(self.project, report)
                self.models = (backend, judge, repair, risk)
            backend, judge, repair, risk = self.models
            from .collection import annotate_state
            from .traces import trace_state
            from ..policy import baseline_scores

            initial = backend.generate(question, evidence, seed=101)
            if initial["trace"] is None:
                raise ValueError(
                    "Model output failed validation: " + str(initial["validation_error"])
                )
            prefix = initial["trace"]["claims"][:2]
            judgments = judge.assess(prefix, evidence)
            state = annotate_state(
                trace_state("live", question, prefix, evidence), judgments, judge, risk
            )
            from .graph_features import enrich_source_bridges

            state = enrich_source_bridges(state)
            scores = baseline_scores(state, "uncertainty")
            policy = "highest_risk"
            checkpoint = self.results / "models/tabular_richer_all_42.joblib"
            if checkpoint.exists():
                from ..policy import TabularPolicy

                scores = TabularPolicy(checkpoint).scores(state)
                policy = "tabular_richer_all"
            selected = max(scores, key=scores.get)
            claim = next(c for c in prefix if c["id"] == selected)
            action = repair.verify(question, claim, evidence)
            before = backend.generate(question, evidence, prefix, 101)
            changed = [
                dict(c, text=action["replacement"]) if c["id"] == selected else dict(c)
                for c in prefix
            ]
            after = backend.generate(question, evidence, changed, 101)

            def display(generation: dict[str, Any]) -> dict[str, Any]:
                """Return factual proxy judgments while leaving unavailable EM null."""
                js = (
                    judge.assess(generation["trace"]["claims"], evidence)
                    if generation["trace"]
                    else []
                )
                from .propagation import factual_summary

                return {
                    "generation": generation,
                    "judgments": js,
                    "correct": None,
                    "answer": generation["trace"]["final_answer"] if generation["trace"] else None,
                    "factual": factual_summary(js),
                }

            return {
                "mode": "live",
                "question": question,
                "prefix": prefix,
                "pre_state": state.to_dict(),
                "before": display(before),
                "after": display(after),
                "selected": selected,
                "policy": policy,
                "action": action,
                "evaluation_available": False,
                "warning": "No reference answer: final factual accuracy is unmeasured. NLI judgments are fallible; edges are possible dependencies.",
            }
