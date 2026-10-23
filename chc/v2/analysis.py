"""Derive propagation diagnostics, review sheets and figures from actual episodes.

No favorable values are hardcoded. Missing/unknown labels remain visible and
figures distinguish automatic evidence judgments from human-reviewed truth.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..data import read_jsonl
from .propagation import utility


def summarize_collection(
    directory: Path, records: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Summarize all attempts, benefit signs, labels, effects and review status.

    Downstream contradiction rates exclude unsupported/unreviewed claims. Error
    amplification is a descriptive count by model-declared graph depth, not a
    causal edge effect. A changed final answer is not necessarily a worse answer.
    Human review files leave decisions blank and keep a separate machine key.
    """
    episodes = (
        read_jsonl(directory / "episodes.jsonl") if (directory / "episodes.jsonl").exists() else []
    )
    if records is not None:
        episodes = records
    attempts = (
        read_jsonl(directory / "attempts.jsonl") if (directory / "attempts.jsonl").exists() else []
    )
    candidates = (
        read_jsonl(directory / "corruption_candidates.jsonl")
        if (directory / "corruption_candidates.jsonl").exists()
        else []
    )
    benefits = [v for r in episodes for v in r["benefits"].values() if v is not None]
    final = [v for r in episodes for v in r["final_only_benefits"].values() if v is not None]
    actions = [a for r in episodes for a in r["verification_actions"].values()]
    effects = []
    claimrows = []
    review = []
    key = []
    for r in episodes:
        before = r["baseline"]
        clean = r["clean_continuation"]
        effects.append(
            {
                "question_id": r["question_id"],
                "episode_id": r["episode_id"],
                "split": r["split"],
                "error_type": r["injection"]["type"],
                "clean_valid": clean["valid"],
                "corrupt_valid": before["valid"],
                "clean_correct": clean.get("correct"),
                "corrupt_correct": before.get("correct"),
                "final_answer_changed": (
                    clean.get("answer") != before.get("answer")
                    if clean["valid"] and before["valid"]
                    else None
                ),
                "no_observable_output_change": clean["generation"]["raw"]
                == before["generation"]["raw"],
                "repeated_error_exact": before.get("repeated_original_error"),
                "caused_additional_contradicted_claim_proxy": (
                    before["factual"]["contradicted"] > clean["factual"]["contradicted"]
                    if before["valid"] and clean["valid"]
                    else None
                ),
                "clean_contradicted": clean["factual"]["contradicted"] if clean["valid"] else None,
                "corrupt_contradicted": (
                    before["factual"]["contradicted"] if before["valid"] else None
                ),
                "clean_rate": (
                    clean["factual"]["observed_propagation_rate"] if clean["valid"] else None
                ),
                "corrupt_rate": (
                    before["factual"]["observed_propagation_rate"] if before["valid"] else None
                ),
                "restoration_utility": utility(before, clean),
                "spontaneously_detected_error": "not independently labeled; no inference from correctness alone",
            }
        )
        for variant, o in [("clean", clean), ("corrupt", before)] + [
            (f"verify_{n}", v) for n, v in r["singletons"].items()
        ]:
            if not o["valid"]:
                continue
            claims = o["generation"]["trace"]["claims"]
            prefix = r["clean_prefix"] if variant == "clean" else r["corrupted_prefix"]
            graph = nx.DiGraph()
            graph.add_nodes_from(c["id"] for c in prefix + claims)
            graph.add_edges_from((d, c["id"]) for c in prefix + claims for d in c["depends_on"])
            for j in o.get("atomic_judgments", o["judgments"]):
                node = j.get("extraction", {}).get("parent_claim_id", j["claim_id"])
                source = r["injection"]["node_id"]
                dependent = source in graph and nx.has_path(graph, source, node)
                depth = nx.shortest_path_length(graph, source, node) if dependent else None
                claimrows.append(
                    {
                        "question_id": r["question_id"],
                        "episode_id": r["episode_id"],
                        "variant": variant,
                        "claim_id": j["claim_id"],
                        "parent_claim_id": node,
                        "status": j["status"],
                        "declared_dependent_on_injection": dependent,
                        "declared_dependency_depth": depth,
                        "error_type": r["injection"]["type"],
                    }
                )
                reviewid = f"R{len(review)+1:05d}"
                review.append(
                    {
                        "review_id": reviewid,
                        "question": r["question"],
                        "claim": j["text"],
                        "original_generated_claim": j.get("extraction", {}).get(
                            "original_text", j["text"]
                        ),
                        "sources": json.dumps(j["evidence"], ensure_ascii=False),
                        "human_status": "",
                        "human_atomicity": "",
                        "notes": "",
                    }
                )
                key.append(
                    {
                        "review_id": reviewid,
                        "episode_id": r["episode_id"],
                        "variant": variant,
                        "machine_status": j["status"],
                        "label_origin": j["judgment_origin"],
                    }
                )

    def distribution(values: list[float]) -> dict[str, int]:
        """Count measured signs without resampling or dropping neutral outcomes."""
        return {
            "positive": sum(v > 1e-9 for v in values),
            "zero": sum(abs(v) <= 1e-9 for v in values),
            "negative": sum(v < -1e-9 for v in values),
        }

    report = {
        "attempted_question_seed_pairs": len(attempts),
        "attempted_unique_questions": len({a["question_id"] for a in attempts}),
        "episodes": len(episodes),
        "eligible_questions": len({r["question_id"] for r in episodes}),
        "questions_by_split": {
            s: len({r["question_id"] for r in episodes if r["split"] == s})
            for s in sorted({r["split"] for r in episodes})
        },
        "rejection_counts": dict(
            pd.Series([a.get("rejection", "eligible") for a in attempts], dtype=str).value_counts()
        ),
        "richer_benefit_counts": distribution(benefits),
        "atomic_prefix_eligible_episodes": sum(
            r.get("atomic_prefix_eligible", True) for r in episodes
        ),
        "selected_corruption_valid_episodes": sum(
            r.get("selected_corruption_validation", {}).get("eligible", True) for r in episodes
        ),
        "harmful_repair_proxy_actions": sum(a["harmful_repair_proxy"] for a in actions),
        "richer_missing_targets": sum(v is None for r in episodes for v in r["benefits"].values()),
        "final_only_benefit_counts": distribution(final),
        "accepted_corrections": sum(a["accepted"] for a in actions),
        "rejected_or_unchanged": sum(not a["accepted"] for a in actions),
        "unchanged_final_answer_actions": sum(
            r["baseline"].get("answer") == o.get("answer")
            for r in episodes
            for o in r["singletons"].values()
            if r["baseline"]["valid"] and o["valid"]
        ),
        "candidate_counts_by_family": dict(
            pd.Series([c["type"] for c in candidates], dtype=str).value_counts()
        ),
        "eligible_candidates_by_family": dict(
            pd.Series(
                [c["type"] for c in candidates if c["primary_eligible"]], dtype=str
            ).value_counts()
        ),
        "human_review_completed": False,
        "scope": "Automatic NLI proxy labels on a filtered controlled challenge set; not human-validated or natural error prevalence",
    }
    # Convert numpy counters for strict standard JSON.
    report = json.loads(json.dumps(report, default=lambda x: int(x)))
    (directory / "collection_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(effects).to_csv(directory / "propagation_effects.csv", index=False)
    pd.DataFrame(claimrows).to_csv(directory / "downstream_claims.csv", index=False)
    if review:
        # Systematic bounded sample covers the full stored ordering. It is a
        # review workload, not a prevalence-estimation or validated label set.
        indices = np.linspace(0, len(review) - 1, min(100, len(review)), dtype=int)
        pd.DataFrame([review[i] for i in indices]).to_csv(
            directory / "human_review_blank.csv", index=False
        )
        pd.DataFrame([key[i] for i in indices]).to_csv(
            directory / "machine_review_key.csv", index=False
        )
    return report


def save_figure(fig: Any, output: Path, stem: str, caption: str) -> None:
    """Save a measured figure in PNG/PDF with a separate interpretation caption."""
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    (output / f"{stem}.caption.txt").write_text(caption, encoding="utf-8")


def plot_collection(directory: Path, records: list[dict[str, Any]] | None = None) -> None:
    """Plot benefits, labeled error rates and declared-depth diagnostics.

    Missing plots are skipped when no measured rows exist. No empty graph is
    filled with invented data. The before/after graph is a labeled real example,
    with status colors from the automatic source judgment only.
    """
    episodes = (
        read_jsonl(directory / "episodes.jsonl") if (directory / "episodes.jsonl").exists() else []
    )
    if records is not None:
        episodes = records
    if not episodes:
        return
    figures = directory / "figures"
    benefits = [v for r in episodes for v in r["benefits"].values() if v is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(benefits, bins=np.linspace(-1.25, 1.25, 21))
    ax.set(
        xlabel="Observed richer-loss reduction",
        ylabel="Candidate actions",
        title="Measured intervention benefits (automatic proxy)",
    )
    save_figure(
        fig,
        figures,
        "benefit_distribution",
        "Includes all measurable eligible action targets; zero outcomes retained; unknown targets excluded and counted separately.",
    )
    claims = pd.read_csv(directory / "downstream_claims.csv")
    group = claims[claims.status.isin(["supported", "contradicted"])].copy()
    group["contradicted"] = (group.status == "contradicted").astype(int)
    if len(group):
        values = group.groupby("variant").contradicted.mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        values.plot.bar(ax=ax)
        ax.set(
            ylabel="Contradicted / evaluable claims",
            title="Observed downstream factual error (NLI proxy)",
        )
        ax.tick_params(axis="x", rotation=25)
        save_figure(
            fig,
            figures,
            "downstream_factual_error",
            "Claim-weighted descriptive rate; unsupported claims excluded, not counted false. Policy confidence intervals use question aggregation separately.",
        )
        dependent = group[group.declared_dependency_depth.notna()]
        if len(dependent):
            table = (
                dependent.groupby(["declared_dependency_depth", "variant"])
                .contradicted.mean()
                .unstack()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            table.plot(marker="o", ax=ax)
            ax.set(
                xlabel="Model-declared shortest dependency depth",
                ylabel="Contradicted / evaluable claims",
                title="Error by uncertain dependency depth",
            )
            save_figure(
                fig,
                figures,
                "propagation_by_depth",
                "Dependency declarations are unverified; this plot does not establish causal edges or independent samples.",
            )
    effects = pd.read_csv(directory / "propagation_effects.csv")
    if len(effects):
        effects["reduction"] = effects.corrupt_rate - effects.clean_rate
        values = effects.groupby("error_type").reduction.mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        values.plot.bar(ax=ax)
        ax.set(
            ylabel="Corrupt minus restored error rate",
            title="Restoration sensitivity by error type",
        )
        ax.tick_params(axis="x", rotation=25)
        save_figure(
            fig,
            figures,
            "restoration_by_error_type",
            "Exact original-prefix restoration is an analysis-only sensitivity check, not a deployable learned correction.",
        )
    r = episodes[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "supported": "#3b9a6d",
        "contradicted": "#d95d5d",
        "insufficient_evidence": "#deb454",
        "unreviewed": "#a4aab3",
    }
    action = next(iter(r["singletons"]))
    for ax, label, o in [
        (axes[0], "Corrupted continuation", r["baseline"]),
        (axes[1], f"After verifying {action}", r["singletons"][action]),
    ]:
        if not o["valid"]:
            ax.text(0.1, 0.5, "Invalid continuation")
            ax.axis("off")
            continue
        cs = o["generation"]["trace"]["claims"]
        g = nx.DiGraph()
        g.add_nodes_from(c["id"] for c in r["corrupted_prefix"] + cs)
        g.add_edges_from((d, c["id"]) for c in cs for d in c["depends_on"])
        status = {j["claim_id"]: j["status"] for j in o["judgments"]}
        nx.draw_networkx(
            g,
            pos=nx.spring_layout(g, seed=42),
            ax=ax,
            node_color=[colors.get(status.get(n), "#a4aab3") for n in g],
            node_size=950,
            font_size=9,
            arrows=True,
        )
        ax.set_title(label)
        ax.axis("off")
    save_figure(
        fig,
        figures,
        "before_after_graph",
        "Real first stored example; prefix statuses are unreviewed here. Edges are model-declared possibilities, not established causal mechanisms.",
    )


def plot_detection(directory: Path) -> None:
    """Plot every detector variant's actual PR and calibration curves.

    The selected confusion matrix is chosen by validation F1, not test scores.
    Test curves are descriptive and retain the natural sampled positive rate.
    """
    from sklearn.metrics import precision_recall_curve

    report = json.loads((directory / "detector_report.json").read_text())
    frame = pd.read_csv(directory / "predictions.csv")
    figures = directory / "figures"
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, g in frame.groupby("method"):
        p, r, _ = precision_recall_curve(g.label, g.score)
        ax.plot(r, p, label=method, linewidth=1)
    ax.set(xlabel="Recall", ylabel="Precision", title="Natural RAGTruth sample: precision–recall")
    ax.legend(fontsize=7)
    save_figure(
        fig,
        figures,
        "precision_recall",
        "Average precision values and validation-selected thresholds appear in metrics.csv. Curves are not positive-enriched.",
    )
    results = [r for r in report["results"] if r["evaluation"] == "test_1024"]
    best = max([r for r in results if "validation_f1" in r], key=lambda r: r["validation_f1"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for r in results:
        bins = r["reliability_bins"]
        ax.plot(
            [b["mean_probability"] for b in bins],
            [b["positive_fraction"] for b in bins],
            marker=".",
            label=r["method"],
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Detector reliability",
    )
    ax.legend(fontsize=6)
    save_figure(
        fig,
        figures,
        "calibration",
        "Platt parameters are fitted only on validation; calibration can still generalize poorly.",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array(best["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted label",
        ylabel="True label",
        title=best["method"],
    )
    save_figure(
        fig,
        figures,
        "confusion_matrix",
        "Method chosen by validation F1; values are held-out natural-distribution test predictions.",
    )


def plot_policies(directory: Path) -> None:
    """Visualize completed policy tables, ablations, budgets, harm and prediction error."""
    if not (directory / "policy_summary.csv").exists():
        return
    frame = pd.read_csv(directory / "policy_summary.csv")
    figures = directory / "figures"
    one = frame[frame.budget == 1]
    for column, stem, title in [
        ("accuracy", "final_accuracy", "Final-answer accuracy"),
        (
            "observed_downstream_error_rate",
            "policy_downstream_error",
            "Downstream contradiction rate",
        ),
        ("harmful_episode_rate", "harmful_interventions", "Harmful final-answer interventions"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(one.method, one[column])
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.set(ylabel=column, title=title)
        save_figure(
            fig,
            figures,
            stem,
            "Matched complete test episodes; question-balanced aggregation. Automatic downstream labels require human review.",
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, g in frame.groupby("method"):
        if m in (
            "no_intervention",
            "uncertainty",
            "risk_dependency",
            "tabular_richer_all",
            "tabular_final_only_all",
            "graph_richer",
            "semantic_richer",
        ):
            ax.plot(g.budget, g.accuracy, marker="o", label=m)
    ax.set(
        xlabel="Maximum verification calls",
        ylabel="Final accuracy",
        title="Measured budget trade-off",
    )
    ax.legend(fontsize=8)
    save_figure(
        fig,
        figures,
        "budget_accuracy",
        "Budget-two outcomes are actual joint generations. Cost-aware methods can use fewer calls than their cap.",
    )
    ab = one[one.method.str.contains("tabular_|graph_|semantic_")]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ab.method, ab.mean_utility)
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set(
        ylabel="Observed richer-loss reduction", title="Matched architecture/objective ablations"
    )
    save_figure(
        fig,
        figures,
        "ablations",
        "Central comparison: final-only versus richer-target training with identical architectures and visible features.",
    )
    quality = pd.read_csv(directory / "benefit_quality.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(quality.method, quality.mae)
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set(ylabel="MAE", title="Benefit prediction, including zero control")
    save_figure(
        fig,
        figures,
        "benefit_prediction_quality",
        "All measured targets, including neutral outcomes; sparse targets can make zero prediction competitive.",
    )
