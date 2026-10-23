"""Matched utility learning, zero controls and paired question-level evaluation.

Reuses the original CHC tabular/GNN implementations. The central comparison is
identical architecture/features trained on final-answer-only versus downstream
factual loss targets. Semantic regression is a simple frozen-embedding control,
not an automatic justification for a hybrid network.
"""

from __future__ import annotations
import copy, json, hashlib
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from ..graph import node_features, GRAPH_FEATURES
from ..schema import DecisionState, Budget
from ..models import fit_tabular, fit_gnn, GraphPolicy
from ..policy import TabularPolicy, baseline_scores, select_claims
from ..metrics import prediction_metrics
from .propagation import observed_loss, utility
from .evidence import align_evidence
from .graph_features import prepare_policy_records
from .boosting import fit_boosting


def supervised(records: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    """Join chosen targets without mutating raw collected episodes.

    A complete pre-action baseline is required; candidate masks exclude missing
    utility labels. The original generator outputs and reference scores remain
    intact. The label scope is explicit rather than overloading correctness.
    """
    result = []
    for r in records:
        if not r["baseline"]["valid"] or not any(v is not None for v in r[target].values()):
            continue
        row = dict(r, supervision_scope="v2_explicit_utility", training_benefits=r[target])
        result.append(row)
    return result


def visible_features(
    records: list[dict[str, Any]], judge: Any
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Extract structured and frozen semantic features from visible state only.

    Returns tabular [A,14], semantic [A,H], and (episode,node) indexing. Embeddings
    concatenate the question, current claim and aligned source passages. They do
    not contain injection identity, gold answer, corrected text, or future claims.
    """
    xs = []
    texts = []
    keys = []
    for r in records:
        state = DecisionState.from_dict(r["pre_state"])
        x, _, _ = node_features(state)
        for i, c in enumerate(state.claims):
            if c.is_final:
                continue
            ev = " ".join(state.evidence[e]["text"] for e in c.evidence_ids)
            xs.append(x[i])
            texts.append(
                state.question + " [CLAIM] " + c.text + " [EVIDENCE] " + align_evidence(c.text, ev)
            )
            keys.append((r["episode_id"], c.id))
    return np.asarray(xs), judge.embeddings(texts), keys


def train_models(
    records: list[dict[str, Any]],
    judge: Any,
    output: Path,
    seeds: list[int] = (42, 43, 44),
    diagnostic_complex: bool = False,
) -> dict[str, Any]:
    """Fit tabular, directed graph and semantic models using validation only.

    Retains all neutral outcomes and trains the same model classes for both
    objectives. The zero predictor and training-frequency mean are saved controls.
    Small pilot models are exploratory; no final test performance selects them.
    Hybrid semantic-graph training is intentionally gated on simpler-model results.
    """
    records = prepare_policy_records(records)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for r in records:
        if r["split"] in ("train", "validation"):
            digest.update(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "episode_id",
                            "split",
                            "pre_state",
                            "benefits",
                            "final_only_benefits",
                        )
                    },
                    sort_keys=True,
                ).encode()
            )
    fingerprint = output / "training_input_digest.json"
    identity = {
        "sha256": digest.hexdigest(),
        "graph_recipe": "source_bridge_v1",
        "scope": "Train/validation features and targets only; final test excluded",
        "training_recipe_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("learning.py", "boosting.py", "graph_features.py")
        },
        "seeds": list(seeds),
        "diagnostic_complex": diagnostic_complex,
    }
    if fingerprint.exists() and json.loads(fingerprint.read_text()) != identity:
        raise ValueError(
            "Training inputs changed; use a new model output directory to preserve checkpoints"
        )
    fingerprint.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    completed = output / "training_summary.json"
    if completed.exists():
        saved = json.loads(completed.read_text(encoding="utf-8"))
        if any(not (output / name).exists() for name in saved.get("checkpoint_files", [])):
            raise FileNotFoundError("A completed training manifest references missing checkpoints")
        return dict(saved, reused_existing_checkpoints=True)
    split = {s: [r for r in records if r["split"] == s] for s in ("train", "validation")}
    counts = {
        s: {
            k: sum(
                1
                for r in rs
                for v in r["benefits"].values()
                if v is not None
                and (v > 0 if k == "positive" else v < 0 if k == "negative" else v == 0)
            )
            for k in ("positive", "zero", "negative")
        }
        for s, rs in split.items()
    }
    if not split["train"] or not split["validation"]:
        return {"status": "not_trained_missing_valid_splits", "target_counts": counts}
    complex_gate = (
        counts["train"]["positive"] + counts["train"]["negative"] >= 20
        and len({r["question_id"] for r in split["train"]}) >= 50
    )
    diagnostic_gate = (
        diagnostic_complex
        and len({r["question_id"] for r in split["train"]}) >= 20
        and counts["train"]["positive"] + counts["train"]["negative"] >= 5
    )
    run_complex = complex_gate or diagnostic_gate
    signal = {
        "counts": counts,
        "primary_count_gate_passed": complex_gate,
        "complex_models_executed": run_complex,
        "diagnostic_requested_before_test_evaluation": diagnostic_complex,
        "execution_scope": "Exploratory pilot; count gate is not a statistical power guarantee",
    }
    (output / "training_signal_before_fit.json").write_text(
        json.dumps(signal, indent=2), encoding="utf-8"
    )
    print(json.dumps({"train_validation_signal_before_fitting": signal}), flush=True)
    if run_complex:
        x, emb, keys = visible_features(split["train"] + split["validation"], judge)
    else:
        x = emb = np.zeros((0, 0))
        keys = []
    byid = {r["episode_id"]: r for r in records}
    logs = []
    for scope, target in [
        ("richer", "benefits"),
        ("final_only", "final_only_benefits"),
        ("final_only_full", "final_only_benefits"),
    ]:
        scoped = split
        if scope == "final_only":
            # The central ablation changes the target, not which action labels
            # are available. Keep an additional full-coverage final-only baseline
            # so the comparison does not deliberately weaken ordinary supervision.
            scoped = {
                s: [
                    dict(
                        r,
                        final_only_benefits={
                            n: v if r["benefits"].get(n) is not None else None
                            for n, v in r["final_only_benefits"].items()
                        },
                    )
                    for r in rows
                ]
                for s, rows in split.items()
            }
        train = supervised(scoped["train"], target)
        validation = supervised(scoped["validation"], target)
        if not train or not validation:
            continue
        for seed in seeds:
            if scope == "final_only_full":
                legacy = fit_tabular(
                    train,
                    validation,
                    output / f"original_tabular_retrained_{seed}.joblib",
                    seed=seed,
                )
                logs.append(
                    dict(kind="original_tabular_retrained", scope=scope, seed=seed, **legacy)
                )
            for ablation, columns in [
                ("all", None),
                ("without_risk", [i for i in range(14) if i not in (0, 1)]),
                ("without_graph", [i for i in range(14) if i not in GRAPH_FEATURES]),
                ("without_source_bridges", None),
            ]:
                # Feature ablations are retrained, not merely zeroed at test time.
                if scope != "richer" and ablation != "all":
                    continue
                path = output / f"tabular_{scope}_{ablation}_{seed}.joblib"
                selected_train = train
                selected_validation = validation
                if ablation == "without_source_bridges":
                    selected_train = [
                        dict(r, pre_state=r["declared_only_pre_state"]) for r in train
                    ]
                    selected_validation = [
                        dict(r, pre_state=r["declared_only_pre_state"]) for r in validation
                    ]
                result = fit_boosting(selected_train, selected_validation, path, columns, seed)
                artifact = joblib.load(path)
                artifact["target"] = target
                joblib.dump(artifact, path)
                logs.append(
                    dict(kind="tabular", scope=scope, ablation=ablation, seed=seed, **result)
                )
            if run_complex and scope != "final_only_full":
                result = fit_gnn(
                    train,
                    validation,
                    output / f"graph_{scope}_{seed}.pt",
                    seed=seed,
                    epochs=25,
                    batch_size=16,
                    accumulation=2,
                    resume=True,
                )
                logs.append(dict(kind="graph", scope=scope, seed=seed, **result))
        # Frozen embeddings plus visible numerical features form a lightweight
        # semantic regressor. Standardization is fit on training rows only.
        y = np.array(
            [
                (
                    byid[e][target][n]
                    if byid[e][target][n] is not None
                    and (scope != "final_only" or byid[e]["benefits"][n] is not None)
                    else np.nan
                )
                for e, n in keys
            ]
        )
        trainmask = np.array(
            [byid[e]["split"] == "train" for e, n in keys], dtype=bool
        ) & np.isfinite(y)
        valmask = np.array(
            [byid[e]["split"] == "validation" for e, n in keys], dtype=bool
        ) & np.isfinite(y)
        combined = np.concatenate([x, emb], axis=1)
        if trainmask.any() and valmask.any() and scope != "final_only_full":
            variants = {
                "all": list(range(combined.shape[1])),
                "without_embeddings": list(range(14)),
                "without_graph": [i for i in range(combined.shape[1]) if i not in GRAPH_FEATURES],
            }
            for variant, columns in variants.items():
                if scope == "final_only" and variant != "all":
                    continue
                best = None
                selected_x = combined[:, columns]
                for alpha in (0.1, 1.0, 10.0, 100.0):
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                        selected_x[trainmask], y[trainmask]
                    )
                    mse = float(np.mean((model.predict(selected_x[valmask]) - y[valmask]) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, model, alpha)
                suffix = "" if variant == "all" else "_" + variant
                joblib.dump(
                    {
                        "model": best[1],
                        "columns": columns,
                        "embedding_model": judge.model_id,
                        "target": target,
                        "alpha": best[2],
                    },
                    output / f"semantic_{scope}{suffix}.joblib",
                )
                logs.append(
                    {
                        "kind": "semantic_ridge",
                        "scope": scope,
                        "variant": variant,
                        "validation_mse": best[0],
                        "alpha": best[2],
                        "semantic_dimensions": emb.shape[1],
                    }
                )
    values = [v for r in split["train"] for v in r["benefits"].values() if v is not None]
    report = {
        "status": "trained_exploratory" if logs else "not_trained_no_measurable_targets",
        "graph_recipe": "source_bridge_v1; fixed on development before final-test evaluation",
        "target_counts": counts,
        "complex_model_gate_passed": complex_gate,
        "complex_models_executed": run_complex,
        "diagnostic_complex_requested": diagnostic_complex,
        "complex_execution_mode": (
            "Primary count gate"
            if complex_gate
            else (
                "Exploratory low-signal diagnostic"
                if run_complex
                else "Skipped for insufficient signal"
            )
        ),
        "complex_gate_rule": "At least 50 eligible train questions and 20 nonzero action targets; predeclared resource/measurement gate, not a performance filter",
        "logs": logs,
        "zero_control": 0.0,
        "training_mean_control": float(np.mean(values)) if values else 0.0,
        "seeds": list(seeds),
        "hybrid_status": "Not implemented: first evaluate simpler models; no demonstrated need",
        "checkpoint_files": sorted(
            p.name
            for p in output.iterdir()
            if p.suffix in (".joblib", ".pt") and not p.name.endswith(".last.pt")
        ),
    }
    (output / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def question_interval(values: pd.Series, seed: int = 42) -> tuple[float, float]:
    """Bootstrap question means, preserving dependence of conditions and seeds.

    The caller groups all dependent rows within each question first. Returns a
    descriptive percentile interval from 2,000 resamples. One question cannot
    support a useful population uncertainty estimate and returns a point interval.
    """
    a = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if not len(a):
        return (float("nan"), float("nan"))
    means = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(2000)])
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def evaluate_models(
    records: list[dict[str, Any]],
    judge: Any,
    models: Path,
    output: Path,
    historical: Path | None = None,
) -> dict[str, Any]:
    """Evaluate identical complete test episodes under budgets 0/1/2.

    Every policy uses the same backend, evidence, corrections and outcomes. Budget
    two uses a measured joint continuation, never an additive singleton fiction.
    Direct V1 transfer is labeled separately from V2 retraining. Richer/final-only
    objective comparisons are predeclared, with matched question intervals.
    """
    output.mkdir(parents=True, exist_ok=True)
    records = prepare_policy_records(records)
    test = [r for r in records if r["split"] == "test"]
    complete = [
        r
        for r in test
        if r["baseline"]["valid"]
        and r["joint_both"]["valid"]
        and all(o["valid"] for o in r["singletons"].values())
    ]
    if not complete:
        report = {"status": "no_complete_test_episodes", "attempted_episodes": len(test)}
        (output / "evaluation_summary.json").write_text(json.dumps(report, indent=2))
        return report
    x, emb, keys = visible_features(complete, judge)
    semantic_x = np.concatenate([x, emb], axis=1)
    semantic_lookup = {key: semantic_x[i] for i, key in enumerate(keys)}
    training = (
        json.loads((models / "training_summary.json").read_text())
        if (models / "training_summary.json").exists()
        else {}
    )
    seeds = training.get("seeds", [42, 43, 44])
    rows = []
    prediction_rows = []
    for seed in seeds:
        policies = {}
        semantic = {}
        for scope in ("richer", "final_only", "final_only_full"):
            for ablation in ("all", "without_risk", "without_graph", "without_source_bridges"):
                path = models / f"tabular_{scope}_{ablation}_{seed}.joblib"
                if path.exists():
                    policies[f"tabular_{scope}_{ablation}"] = TabularPolicy(path)
            path = models / f"graph_{scope}_{seed}.pt"
            if path.exists():
                policies[f"graph_{scope}"] = GraphPolicy(path)
            for variant in ("", "_without_embeddings", "_without_graph"):
                path = models / f"semantic_{scope}{variant}.joblib"
                if path.exists():
                    semantic[f"semantic_{scope}{variant}"] = joblib.load(path)
        if historical:
            for name, filename, cls in [
                ("original_tabular_transfer", f"tabular_{seed}.joblib", TabularPolicy),
                ("original_graph_transfer", f"graph_{seed}.pt", GraphPolicy),
            ]:
                path = historical / filename
                if path.exists():
                    policies[name] = cls(path)
        legacy_retrained = models / f"original_tabular_retrained_{seed}.joblib"
        if legacy_retrained.exists():
            policies["original_tabular_retrained"] = TabularPolicy(legacy_retrained)
        for r in complete:
            state = DecisionState.from_dict(r["pre_state"])
            ids = [c.id for c in state.claims if not c.is_final]
            scores = {
                m: baseline_scores(state, m, seed)
                for m in ("random", "uncertainty", "dependency", "risk_dependency")
            }
            scores.update(
                {
                    m: p.scores(
                        DecisionState.from_dict(r["declared_only_pre_state"])
                        if m.endswith("without_source_bridges")
                        else state
                    )
                    for m, p in policies.items()
                }
            )
            scores.update(
                {
                    m: {
                        n: float(
                            p["model"].predict(
                                semantic_lookup[(r["episode_id"], n)][p["columns"]].reshape(1, -1)
                            )[0]
                        )
                        for n in ids
                    }
                    for m, p in semantic.items()
                }
            )
            scores["zero_benefit"] = {n: 0.0 for n in ids}
            scores["training_mean"] = {n: training.get("training_mean_control", 0.0) for n in ids}
            for m, score in scores.items():
                for n, p in score.items():
                    objective = (
                        "final_only"
                        if "final_only" in m or m.startswith("original_")
                        else (
                            "richer"
                            if m.startswith(("tabular_", "graph_", "semantic_"))
                            or m in ("zero_benefit", "training_mean")
                            else "priority"
                        )
                    )
                    y = (
                        r["final_only_benefits"][n]
                        if objective == "final_only"
                        else r["benefits"][n]
                    )
                    if y is not None:
                        prediction_rows.append(
                            {
                                "method": m,
                                "question_id": r["question_id"],
                                "episode_id": r["episode_id"],
                                "node_id": n,
                                "seed": seed,
                                "observed_benefit": y,
                                "scoring_objective": objective,
                                "prediction": p,
                            }
                        )
                variants = [(m, m in ("zero_benefit", "training_mean"))]
                if m in ("tabular_richer_all", "graph_richer", "semantic_richer"):
                    variants += [(m + "_cost_aware", True)]
                if m == "tabular_richer_all":
                    variants += [(m + "_without_semantic_gate", False)]
                for name, stop in variants:
                    for budget in (0, 1, 2):
                        selected = select_claims(
                            state,
                            score,
                            Budget(calls=budget),
                            minimum_net_benefit=0.01 if stop else None,
                        )
                        if not selected:
                            o = r["baseline"]
                        elif len(selected) == 2:
                            o = (
                                r["joint_gate_off"]
                                if name.endswith("_without_semantic_gate")
                                else r["joint_both"]
                            )
                        elif name.endswith("_without_semantic_gate"):
                            o = r["without_semantic_gate"][selected[0]]
                        else:
                            o = r["singletons"][selected[0]]
                        if not o["valid"]:
                            continue
                        before_loss = observed_loss(r["baseline"])
                        after_loss = observed_loss(o)
                        rows.append(
                            {
                                "method": name,
                                "budget": budget,
                                "seed": seed,
                                "question_id": r["question_id"],
                                "episode_id": r["episode_id"],
                                "error_type": r["injection"]["type"],
                                "correct": int(o["correct"]),
                                "baseline_correct": int(r["baseline"]["correct"]),
                                "gain": int(o["correct"]) - int(r["baseline"]["correct"]),
                                "factual_rate": o["factual"]["observed_propagation_rate"],
                                "baseline_factual_rate": r["baseline"]["factual"][
                                    "observed_propagation_rate"
                                ],
                                "downstream_evaluable_claims": o["factual"]["evaluable_claims"],
                                "downstream_total_claims": o["factual"]["downstream_claims"],
                                "utility": (
                                    before_loss - after_loss
                                    if before_loss is not None and after_loss is not None
                                    else None
                                ),
                                "harmful": int(bool(r["baseline"]["correct"]) and not o["correct"]),
                                "calls": len(selected),
                                "estimated_tokens": sum(
                                    r["verification_actions"][n]["estimated_tokens"]
                                    for n in selected
                                ),
                                "verification_runtime_ms": sum(
                                    r["verification_actions"][n]["latency_ms"] for n in selected
                                ),
                                "continuation_runtime_ms": o["generation"]["latency_ms"],
                                "selected": "|".join(selected),
                            }
                        )
            for budget in (0, 1, 2):
                o = r["baseline"]
                rows.append(
                    {
                        "method": "no_intervention",
                        "budget": budget,
                        "seed": seed,
                        "question_id": r["question_id"],
                        "episode_id": r["episode_id"],
                        "error_type": r["injection"]["type"],
                        "correct": int(o["correct"]),
                        "baseline_correct": int(o["correct"]),
                        "gain": 0,
                        "factual_rate": o["factual"]["observed_propagation_rate"],
                        "baseline_factual_rate": o["factual"]["observed_propagation_rate"],
                        "utility": 0,
                        "harmful": 0,
                        "calls": 0,
                        "estimated_tokens": 0,
                        "verification_runtime_ms": 0,
                        "continuation_runtime_ms": o["generation"]["latency_ms"],
                        "selected": "",
                    }
                )
    frame = pd.DataFrame(rows)
    frame["factual_reduction"] = frame.baseline_factual_rate - frame.factual_rate
    frame.to_csv(output / "policy_predictions.csv", index=False)
    summaries = []
    for (method, budget), g in frame.groupby(["method", "budget"]):
        question = g.groupby("question_id")[
            [
                "correct",
                "gain",
                "utility",
                "factual_rate",
                "factual_reduction",
                "harmful",
                "calls",
                "estimated_tokens",
                "verification_runtime_ms",
            ]
        ].mean()
        lo, hi = question_interval(question.gain)
        accuracy_lo, accuracy_hi = question_interval(question.correct)
        utility_lo, utility_hi = question_interval(question.utility.dropna())
        reduction_lo, reduction_hi = question_interval(question.factual_reduction.dropna())
        summaries.append(
            {
                "method": method,
                "budget": int(budget),
                "questions": len(question),
                "episodes": int(g.episode_id.nunique()),
                "accuracy": float(question.correct.mean()),
                "accuracy_ci_low": accuracy_lo,
                "accuracy_ci_high": accuracy_hi,
                "accuracy_gain": float(question.gain.mean()),
                "gain_ci_low": lo,
                "gain_ci_high": hi,
                "factual_rate_defined_fraction": float(g.factual_rate.notna().mean()),
                "observed_downstream_error_rate": (
                    float(question.factual_rate.mean())
                    if question.factual_rate.notna().any()
                    else None
                ),
                "mean_utility": float(question.utility.mean()),
                "utility_ci_low": utility_lo,
                "utility_ci_high": utility_hi,
                "downstream_reduction": (
                    float(question.factual_reduction.mean())
                    if question.factual_reduction.notna().any()
                    else None
                ),
                "reduction_ci_low": reduction_lo,
                "reduction_ci_high": reduction_hi,
                "factual_comparison_questions": int(question.factual_reduction.notna().sum()),
                "harmful_episode_rate": float(question.harmful.mean()),
                "harmful_rate_per_selected_episode": (
                    float(g[g.calls > 0].harmful.mean()) if (g.calls > 0).any() else None
                ),
                "mean_calls": float(question.calls.mean()),
                "accuracy_gain_per_call": (
                    float(question.gain.mean() / question.calls.mean())
                    if question.calls.mean() > 0
                    else None
                ),
                "mean_estimated_tokens": float(question.estimated_tokens.mean()),
                "mean_verification_runtime_ms": float(question.verification_runtime_ms.mean()),
            }
        )
    pd.DataFrame(summaries).to_csv(output / "policy_summary.csv", index=False)
    pf = pd.DataFrame(prediction_rows)
    pf.to_csv(output / "benefit_predictions.csv", index=False)
    quality = []
    for m, g in pf.groupby("method"):
        metric_values = prediction_metrics(g.observed_benefit.tolist(), g.prediction.tolist())
        if g.scoring_objective.iloc[0] == "priority":
            # Risk/reach are ranking priorities, not predictions in utility units.
            metric_values["mae"] = None
            metric_values["rmse"] = None
        quality.append(
            dict(
                method=m,
                scoring_objective=g.scoring_objective.iloc[0],
                **metric_values,
                positive_average_precision=(
                    float(average_precision_score(g.observed_benefit > 0, g.prediction))
                    if (g.observed_benefit > 0).any()
                    else None
                ),
                negative_average_precision=(
                    float(average_precision_score(g.observed_benefit < 0, -g.prediction))
                    if (g.observed_benefit < 0).any()
                    else None
                ),
            )
        )
    pd.DataFrame(quality).to_csv(output / "benefit_quality.csv", index=False)
    paired = []
    one = frame[frame.budget == 1]
    for a, b in [
        ("tabular_richer_all", "tabular_final_only_all"),
        ("tabular_richer_all", "tabular_final_only_full_all"),
        ("graph_richer", "graph_final_only"),
        ("semantic_richer", "semantic_final_only"),
        ("tabular_richer_all", "uncertainty"),
        ("tabular_richer_all", "risk_dependency"),
    ]:
        ga = one[one.method == a]
        gb = one[one.method == b]
        match = ga.merge(gb, on=["question_id", "episode_id", "seed"], suffixes=("_a", "_b"))
        if match.empty:
            continue
        for metric in ("correct", "utility"):
            values = (
                (match[f"{metric}_a"] - match[f"{metric}_b"])
                .groupby(match.question_id)
                .mean()
                .dropna()
            )
            if values.empty:
                continue
            lo, hi = question_interval(values)
            paired.append(
                {
                    "method": a,
                    "baseline": b,
                    "metric": metric,
                    "questions": len(values),
                    "difference": float(values.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "inference": "Descriptive question bootstrap; no multiplicity-adjusted significance claim",
                }
            )
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    report = {
        "attempted_test_episodes": len(test),
        "complete_test_episodes": len(complete),
        "complete_test_questions": len({r["question_id"] for r in complete}),
        "coverage": len(complete) / len(test),
        "summaries": summaries,
        "paired": paired,
        "benefit_quality": quality,
        "evaluation_scope": "Complete-case automatic-NLI-labeled controlled challenge set; not natural hallucination prevalence",
    }
    # DataFrame serialization converts undefined numeric metrics to JSON null.
    report = json.loads(pd.Series([report]).to_json(orient="values"))[0]
    (output / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
