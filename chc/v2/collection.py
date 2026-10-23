"""Resumable CHC-Propagate collection using actual frozen-model continuations.

The collector retains every attempted question and candidate mutation. Selection
of error families occurs before observing continuation outcomes. Clean, corrupt,
restored and verified continuations share question, evidence and sampling seed.
"""

from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
from typing import Any
from ..data import supporting_sentences, read_jsonl, write_jsonl
from .traces import trace_state
from .corruptions import corruption_candidates, validate_candidates
from .propagation import score_continuation, utility
from .validity import selected_corruption_validity


def source_evidence(row: dict[str, Any], limit: int = 6) -> dict[str, dict[str, str]]:
    """Build bounded real source passages from annotated supporting sentences.

    Returns passages with document titles, original sentence indices, dataset
    question ID and official dataset link. This controlled evidence condition is
    easier than retrieval over full distractor documents and is reported as such.
    Nothing is synthesized from the reference answer.
    """
    facts, documents = supporting_sentences(row, limit)
    result = {}
    for i, doc in enumerate(documents.values()):
        selected = [f for f in facts if f["title"] == doc["title"]]
        result[f"E{i+1}"] = {
            "title": doc["title"],
            "text": " ".join(f["text"] for f in selected),
            "source": "HotpotQA annotated supporting context",
            "url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "question_id": row["id"],
            "sentence_indices": ",".join(f["sentence_index"] for f in selected),
        }
    return result


def annotate_state(
    state: Any, judgments: list[dict[str, Any]], judge: Any, risk_detector: Any = None
) -> Any:
    """Attach pre-action predicted risk, source relevance and tokenizer costs.

    Only visible claims and their evidence judgments are consumed. Reference
    answers, injected locations and observed action benefits never enter features.
    Uses the same source-aligned judge as the other V2 verification policies.
    """
    byid = {j["claim_id"]: j for j in judgments}
    predictions = {}
    if risk_detector is not None:
        candidates = [c for c in state.claims if not c.is_final]
        passages = [" ".join(state.evidence[e]["text"] for e in c.evidence_ids) for c in candidates]
        predictions = {
            c.id: p
            for c, p in zip(
                candidates,
                risk_detector.predict(
                    [c.text for c in candidates], passages, [c.evidence_ids for c in candidates]
                ),
            )
        }
    for c in state.claims:
        if c.is_final:
            continue
        j = byid[c.id]
        c.risk = predictions[c.id]["hallucination_score"] if predictions else j["contradiction"]
        c.evidence_quality = j["entailment"]
        c.status = j["status"]
        ev = " ".join(state.evidence[e]["text"] for e in c.evidence_ids)
        c.cost_tokens = len(judge.tokenizer.encode(ev, c.text, truncation=False))
    return state


def collect(
    splits: dict[str, list[dict[str, Any]]],
    backend: Any,
    judge: Any,
    repair: Any,
    directory: Path,
    seeds: list[int],
    conditions: int = 1,
    risk_detector: Any = None,
    continuation_visibility: str = "full",
) -> list[dict[str, Any]]:
    """Collect actual paired outcomes without filtering on benefit or correctness.

    Args:
        splits: Predeclared disjoint development/train/validation/test rows.
        backend/judge/repair: Shared real inference objects with persistent caches.
        directory: New experiment directory; prior V1 outputs are never overwritten.
        seeds: Paired model-generation seeds, not additional independent questions.
        conditions: Maximum eligible distinct error families per question.
    Returns:
        Recorded complete episodes; attempted and rejected records are separate.
    Algorithm:
        Validate a generated trace, keep its first two supported factual claims as
        a prefix, select a valid mutation without observing effects, and continue
        after each real verifier action. An exact original restoration is an
        analysis-only sensitivity diagnostic, never an available serving action.
    Missing data:
        Malformed continuations remain missing and are not reinterpreted as
        correct/neutral outcomes. Complete-case policy evaluation reports coverage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    episode_path = directory / "episodes.jsonl"
    attempt_path = directory / "attempts.jsonl"
    records = read_jsonl(episode_path) if episode_path.exists() else []
    attempts = read_jsonl(attempt_path) if attempt_path.exists() else []
    done = {r["episode_id"] for r in records}
    attempted = {(r["question_id"], r["seed"]) for r in attempts if r.get("collection_complete")}
    candidate_rows = (
        read_jsonl(directory / "corruption_candidates.jsonl")
        if (directory / "corruption_candidates.jsonl").exists()
        else []
    )

    def candidate_key(candidate: dict[str, Any]) -> str:
        """Identify a proposal across resumes without using its eventual benefit.

        A crash after an episode append but before attempt completion can revisit
        candidate construction. Deduplication prevents inflated family counts.
        """
        return json.dumps(
            [
                candidate["question_id"],
                candidate["seed"],
                candidate["node_id"],
                candidate["type"],
                candidate["modified"],
                candidate["evidence_ids"],
            ],
            sort_keys=True,
        )

    candidate_keys = {candidate_key(c) for c in candidate_rows}
    for split, questions in splits.items():
        for qi, row in enumerate(questions):
            evidence = source_evidence(row)
            for seed in seeds:
                if (row["id"], seed) in attempted:
                    continue
                start = backend.generate(row["question"], evidence, seed=seed)
                attempt = {
                    "question_id": row["id"],
                    "split": split,
                    "seed": seed,
                    "initial_generation": start,
                    "collection_complete": True,
                }
                if not start["trace"] or len(start["trace"]["claims"]) < 2:
                    attempt["rejection"] = "invalid_or_short_initial_trace"
                    attempts.append(attempt)
                    write_jsonl(attempt_path, attempts)
                    continue
                prefix = copy.deepcopy(start["trace"]["claims"][:2])
                js = judge.assess(prefix, evidence)
                attempt["prefix_judgments"] = js
                if any(j["status"] != "supported" for j in js):
                    attempt["rejection"] = "initial_prefix_not_automatically_supported"
                    attempts.append(attempt)
                    write_jsonl(attempt_path, attempts)
                    continue
                candidates = validate_candidates(
                    corruption_candidates(prefix, evidence), prefix, evidence, judge
                )
                for c in candidates:
                    item = dict(c, question_id=row["id"], split=split, seed=seed)
                    key = candidate_key(item)
                    if key not in candidate_keys:
                        candidate_rows.append(item)
                        candidate_keys.add(key)
                eligible = [c for c in candidates if c["primary_eligible"]]
                # Rotate family order using question hash, before any outcome is
                # generated. Neutral/harmful effects are never discarded later.
                types = sorted({c["type"] for c in eligible})
                rotation = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % max(
                    1, len(types)
                )
                types = types[rotation:] + types[:rotation]
                chosen = [next(c for c in eligible if c["type"] == t) for t in types[:conditions]]
                attempt["eligible_families"] = types
                if not chosen:
                    attempt["rejection"] = "no_automatically_contradicted_mutation"
                    attempts.append(attempt)
                    write_jsonl(attempt_path, attempts)
                    write_jsonl(directory / "corruption_candidates.jsonl", candidate_rows)
                    continue
                validated_chosen=[]
                for injection in chosen:
                    validation=selected_corruption_validity(injection,prefix,evidence,judge)
                    changed=copy.deepcopy(prefix)
                    for c in changed:
                        if c['id']==injection['node_id']:c.update(text=injection['modified'],evidence_ids=injection['evidence_ids'])
                    try:trace_state(row['id'],row['question'],changed,evidence)
                    except ValueError as exc:
                        validation['eligible']=False;validation['reasons'].append('Invalid mutated graph: '+str(exc))
                    injection['selected_validity']=validation
                    if validation['eligible']:validated_chosen.append(injection)
                    else:attempt.setdefault('selected_candidate_rejections',[]).append(injection)
                chosen=validated_chosen
                if not chosen:
                    attempt['rejection']='selected_corruption_failed_pre_outcome_validity'
                    attempts.append(attempt);write_jsonl(attempt_path,attempts)
                    write_jsonl(directory/'corruption_candidates.jsonl',candidate_rows)
                    continue
                visible = None
                if continuation_visibility == "masked_first_source":
                    # Separate, explicitly labeled context-compression challenge:
                    # the first source text is outside the continuation window,
                    # but remains available to the verifier/evaluator. Which
                    # source is masked is fixed before choosing an error/outcome.
                    visible = copy.deepcopy(evidence)
                    visible[next(iter(visible))][
                        "text"
                    ] = "[This source text is outside the continuation context. Earlier working notes describe this hop.]"
                elif continuation_visibility != "full":
                    raise ValueError("Unknown continuation visibility")
                clean_gen = backend.generate(
                    row["question"], evidence, prefix, seed, visible_evidence=visible
                )
                clean_js = (
                    judge.assess(clean_gen["trace"]["claims"], evidence)
                    if clean_gen["trace"]
                    else []
                )
                clean = score_continuation(clean_gen, clean_js, row["answer"])
                for injection in chosen:
                    episode_id = f"{row['id']}:{injection['type']}:{seed}"
                    if episode_id in done:
                        continue
                    corrupted = copy.deepcopy(prefix)
                    for c in corrupted:
                        if c["id"] == injection["node_id"]:
                            c.update(
                                text=injection["modified"], evidence_ids=injection["evidence_ids"]
                            )

                    def evaluate(current: list[dict[str, Any]]) -> dict[str, Any]:
                        """Generate first, then score with held-out reference and NLI.

                        The generation API receives no reference. This ordering
                        is enforced at the call boundary rather than by deleting
                        fields from a dictionary after accidental exposure.
                        """
                        g = backend.generate(
                            row["question"], evidence, current, seed, visible_evidence=visible
                        )
                        j = judge.assess(g["trace"]["claims"], evidence) if g["trace"] else []
                        return score_continuation(g, j, row["answer"], injection)

                    baseline = evaluate(corrupted)
                    current_js = judge.assess(corrupted, evidence)
                    state = annotate_state(
                        trace_state(row["id"], row["question"], corrupted, evidence),
                        current_js,
                        judge,
                        risk_detector,
                    )
                    outcomes = {}
                    actions = {}
                    gate_off = {}
                    for c in corrupted:
                        action = repair.verify(row["question"], c, evidence)
                        actions[c["id"]] = action
                        changed = copy.deepcopy(corrupted)
                        next(x for x in changed if x["id"] == c["id"])["text"] = action[
                            "replacement"
                        ]
                        outcomes[c["id"]] = evaluate(changed)
                        # Gate-only ablation shares the same actual lexical
                        # proposal; it does not choose a different candidate pool.
                        unchecked = copy.deepcopy(corrupted)
                        next(x for x in unchecked if x["id"] == c["id"])["text"] = action[
                            "proposal"
                        ]
                        gate_off[c["id"]] = evaluate(unchecked)
                    both = copy.deepcopy(corrupted)
                    for c in both:
                        c["text"] = actions[c["id"]]["replacement"]
                    joint = evaluate(both)
                    unchecked_both = copy.deepcopy(corrupted)
                    for c in unchecked_both:
                        c["text"] = actions[c["id"]]["proposal"]
                    joint_gate_off = evaluate(unchecked_both)
                    record = {
                        "episode_id": episode_id,
                        "question_id": row["id"],
                        "split": split,
                        "question": row["question"],
                        "reference_answer": row["answer"],
                        "provenance": row.get("_provenance", {}),
                        "generation_seed": seed,
                        "evidence": evidence,
                        "initial_generation": start,
                        "clean_prefix": prefix,
                        "corrupted_prefix": corrupted,
                        "pre_state": state.to_dict(),
                        "injection": injection,
                        "selected_corruption_validation":injection['selected_validity'],
                        "clean_continuation": clean,
                        "baseline": baseline,
                        "singletons": outcomes,
                        "verification_actions": actions,
                        "without_semantic_gate": gate_off,
                        "joint_both": joint,
                        "joint_gate_off": joint_gate_off,
                        "restoration_diagnostic": clean,
                        "restoration_scope": "Exact clean prefix restoration; analysis only, not a deployable action",
                        "benefits": {
                            n: utility(
                                baseline, o, harmful_repair=actions[n]["harmful_repair_proxy"]
                            )
                            for n, o in outcomes.items()
                        },
                        "final_only_benefits": {
                            n: utility(baseline, o, final_only=True) for n, o in outcomes.items()
                        },
                        "human_review": "pending",
                        "label_scope": "automatic_nli_proxy; controlled_challenge_set",
                    }
                    record["continuation_visibility"] = continuation_visibility
                    record["continuation_evidence"] = visible if visible is not None else evidence
                    with episode_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    records.append(record)
                    done.add(episode_id)
                attempts.append(attempt)
                write_jsonl(attempt_path, attempts)
                write_jsonl(directory / "corruption_candidates.jsonl", candidate_rows)
            print(
                f"{split} {qi+1}/{len(questions)}; episodes={len(records)} new_generations={backend.actual_generations} cache_hits={backend.cache_hits}",
                flush=True,
            )
    return records
