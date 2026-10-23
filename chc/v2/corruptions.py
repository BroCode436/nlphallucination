"""Auditable factual perturbation candidates with conservative validity gates.

Ten error families are implemented where their prerequisites exist. Generated
mutations are candidates, not automatically true error labels. Primary inclusion
requires a supported original and a contradicted mutation from the automatic NLI
proxy. All provenance/disambiguation/composition cases still need human review.
"""

from __future__ import annotations
import copy, re
from typing import Any


def corruption_candidates(
    claims: list[dict[str, Any]], evidence: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Generate eligible mutations without references or intervention outcomes.

    Args:
        claims: Validated clean prefix; only explicit textual anchors are changed.
        evidence: Supplies alternative real entity names and citation IDs.
    Returns:
        Candidates containing original/modified text, source claim ID, error
        family, exact transformation and its proposed justification. Unsupported
        assertions are not assumed false. Missing prerequisites yield no example
        for that family instead of inventing a label or an entity relationship.
    """
    output = []
    for c in claims:
        text = c["text"]

        def add(kind: str, modified: str, basis: str, changed_ids: list[str] | None = None) -> None:
            """Append a changed candidate with a machine-readable review obligation.

            The nested helper centralizes provenance, but deliberately grants no
            factual-validity status. Later evidence checks decide eligibility.
            """
            if modified != text or changed_ids:
                output.append(
                    {
                        "node_id": c["id"],
                        "type": kind,
                        "original": text,
                        "modified": modified,
                        "evidence_ids": changed_ids or c["evidence_ids"],
                        "justification": basis,
                        "validity": "candidate_unreviewed",
                        "human_review": "pending",
                    }
                )

        years = list(re.finditer(r"\b(?:18|19|20)\d{2}\b", text))
        if years:
            m = years[0]
            add(
                "incorrect_date",
                text[: m.start()] + str(int(m.group()) + 3) + text[m.end() :],
                f"Changes explicit year {m.group()} to {int(m.group())+3}; compare same relation against cited evidence",
            )
        number = re.search(r"(?<![\w.])\d+(?!\w|\.\d)", text)
        if number:
            m = number
            add(
                "incorrect_number",
                text[: m.start()] + str(int(m.group()) + 2) + text[m.end() :],
                f"Changes explicit numeric value {m.group()} by +2; dates also retained as overlapping family candidates",
            )
        copula = re.search(r"\b(is|was|are|were|has|had)\b(?!\s+not\b)", text)
        if copula and not re.search(r"\b(not|never|if|might)\b", text, re.I):
            add(
                "negation_error",
                text[: copula.end()] + " not" + text[copula.end() :],
                "Negates an explicit affirmative assertion; requires contradiction evidence",
            )
        titles = [e["title"] for e in evidence.values()]
        for title in titles:
            if title.casefold() in text.casefold():
                other = next((t for t in titles if t != title), None)
                if other:
                    modified = re.sub(re.escape(title), lambda _: other, text, count=1, flags=re.I)
                    add(
                        "entity_substitution",
                        modified,
                        f"Replaces subject/entity {title} with distinct supplied entity {other}; truth must be checked",
                    )
                    add(
                        "entity_disambiguation",
                        modified,
                        f"Attributes the same fact to different entity {other}; ambiguous identity requires independent review",
                    )
                break
        relation = re.fullmatch(
            r"(.+?) (is|was) (?:the )?(parent|child|predecessor|successor|north|south|east|west) (?:of )?(.+)",
            text,
            re.I,
        )
        inverse = {
            "parent": "child",
            "child": "parent",
            "predecessor": "successor",
            "successor": "predecessor",
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        }
        if relation:
            a, verb, r, b = relation.groups()
            add(
                "relationship_reversal",
                f"{a} {verb} {inverse[r.lower()]} of {b}",
                f"Reverses asymmetric relation {r}; verify source and direction",
            )
        alternate = [e for e in evidence if e not in c["evidence_ids"]]
        if alternate:
            add(
                "evidence_misattribution",
                text,
                "Changes citation to a different document; a provenance error is not automatically factual falsity",
                [alternate[0]],
            )
        add(
            "unsupported_conclusion",
            text + " It received an international award.",
            "Appends a claim absent from the known support; insufficient evidence is not a contradiction",
        )
        arithmetic = re.search(r"(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(\d+)", text)
        if arithmetic:
            a, op, b, answer = arithmetic.groups()
            actual = {"+": int(a) + int(b), "-": int(a) - int(b), "*": int(a) * int(b)}[op]
            if int(answer) == actual:
                add(
                    "intermediate_calculation",
                    text[: arithmetic.start(4)] + str(actual + 1) + text[arithmetic.end(4) :],
                    f"Exact integer arithmetic: {a}{op}{b}={actual}, not {actual+1}",
                )
        if c["depends_on"]:
            dep = next((d for d in claims if d["id"] == c["depends_on"][0]), None)
            if dep and dep["text"] != text:
                add(
                    "multi_hop_composition",
                    dep["text"],
                    "Replaces a dependent conclusion with an earlier premise; not necessarily false, needs review",
                )
    return output


def validate_candidates(
    candidates: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence: dict[str, dict[str, str]],
    judge: Any,
) -> list[dict[str, Any]]:
    """Check original/mutated statements and retain uncertainty explicitly.

    Every candidate survives in the audit. Only a high-confidence automatically
    supported original and contradicted mutation can enter the primary automatic
    challenge set. This does not imply human validation or natural prevalence.
    """
    byid = {c["id"]: c for c in claims}
    original = {j["claim_id"]: j for j in judge.assess(claims, evidence)}
    mutations = [
        dict(byid[c["node_id"]], text=c["modified"], evidence_ids=c["evidence_ids"])
        for c in candidates
    ]
    assessed = judge.assess(mutations, evidence)
    for c, j in zip(candidates, assessed):
        c["original_judgment"] = original[c["node_id"]]
        c["mutation_judgment"] = j
        factual_mutation = c["type"] not in (
            "evidence_misattribution",
            "unsupported_conclusion",
            "multi_hop_composition",
            "entity_disambiguation",
        )
        # Wrong citations and an irrelevant-but-true premise are not evidence of
        # false factual content. Even an overconfident NLI result cannot promote
        # these transformations into the primary contradiction benchmark.
        c["primary_eligible"] = (
            factual_mutation
            and original[c["node_id"]]["status"] == "supported"
            and j["status"] == "contradicted"
        )
        c["factual_mutation_gate"] = factual_mutation
        c["validity"] = (
            "automatic_contradiction_candidate"
            if c["primary_eligible"]
            else "requires_human_review"
        )
    return candidates
