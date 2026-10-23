"""Deterministic V2 software fixtures; these are not research measurements.

Tests enforce abstention, schema validation, leakage boundaries, denominator
rules and signed labels before expensive real-model collection is trusted.
"""

import copy, json
import numpy as np
import pytest
from chc.v2.traces import parse_trace, trace_state
from chc.v2.evidence import align_evidence
from chc.v2.propagation import factual_summary, observed_loss, utility
from chc.v2.corruptions import corruption_candidates, validate_candidates
from chc.v2.learning import supervised, question_interval
from chc.models import training_graph
from chc.v2.detection import binary_metrics, reliability


@pytest.fixture
def source():
    """Return a tiny explicit evidence fixture for invariant testing only."""
    return {
        "E1": {"title": "Ada", "text": "Ada was born in 1980. Ada is a scientist."},
        "E2": {"title": "Ben", "text": "Ben is an artist."},
    }


@pytest.fixture
def trace():
    """Return a hand-authored fixture clearly excluded from benchmark collection."""
    return {
        "claims": [
            {"id": "C1", "text": "Ada was born in 1980.", "depends_on": [], "evidence_ids": ["E1"]},
            {
                "id": "C2",
                "text": "Ada is a scientist.",
                "depends_on": ["C1"],
                "evidence_ids": ["E1"],
            },
        ],
        "final_answer": "Ada",
    }


def test_valid_trace_and_graph(trace, source):
    """Explicit prior-to-later dependencies survive conversion to existing CHC."""
    parsed = parse_trace(json.dumps(trace), source, "Who?")
    state = trace_state("q", "Who?", parsed["claims"], source)
    assert state.edges[0]["source"] == "C1" and state.edges[0]["target"] == "C2"
    assert not any(e["verified_causal"] for e in state.edges)
    assert state.claims[-1].text == "Who?"


@pytest.mark.parametrize(
    "change", ["citation", "forward", "duplicate", "extra", "blank", "notjson"]
)
def test_strict_rejection(trace, source, change):
    """Malformed or uncited output cannot silently enter a valid benchmark."""
    if change == "citation":
        trace["claims"][0]["evidence_ids"] = []
    if change == "forward":
        trace["claims"][0]["depends_on"] = ["C2"]
    if change == "duplicate":
        trace["claims"][1]["id"] = "C1"
    if change == "extra":
        trace["reference_answer"] = "leak"
    if change == "blank":
        trace["final_answer"] = ""
    with pytest.raises(ValueError):
        parse_trace("nonsense" if change == "notjson" else json.dumps(trace), source, "Who?")


def test_continuation_prefix_ids(source):
    """A continuation may depend on a visible prefix, without repeating its IDs."""
    value = {
        "claims": [
            {
                "id": "C3",
                "text": "Ada is a scientist.",
                "depends_on": ["C1"],
                "evidence_ids": ["E1"],
            }
        ],
        "final_answer": "Ada",
    }
    assert parse_trace(json.dumps(value), source, "Who?", ("C1", "C2"))["claims"][0][
        "depends_on"
    ] == ["C1"]


def test_unknown_not_false():
    """Only supported and contradicted claims enter the factual-rate denominator."""
    summary = factual_summary(
        [
            {"status": s}
            for s in ["supported", "contradicted", "insufficient_evidence", "unreviewed"]
        ]
    )
    assert summary["observed_propagation_rate"] == 0.5 and summary["evaluable_claims"] == 2
    assert (
        factual_summary([{"status": "insufficient_evidence"}])["observed_propagation_rate"] is None
    )


def test_signed_richer_utility():
    """A harmful downstream intervention stays negative even with unchanged EM."""
    good = {"valid": True, "correct": True, "factual": {"observed_propagation_rate": 0.0}}
    bad = {"valid": True, "correct": True, "factual": {"observed_propagation_rate": 1.0}}
    assert utility(good, bad) == -0.5 and utility(good, bad, final_only=True) == 0
    assert utility(bad, good) == 0.5
    assert observed_loss({"valid": False}) is None
    unknown = dict(good, factual={"observed_propagation_rate": None})
    assert utility(good, unknown) is None


def test_alignment_uses_later_relevant_sentence():
    """The selector can recover a relevant sentence beyond an irrelevant prefix."""
    ev = "Noise is irrelevant. " * 200 + "Ada was born in 1980."
    assert "1980" in align_evidence("When was Ada born?", ev, 1)


def test_mutations_are_not_gold(trace, source):
    """Changing dates or entity names does not automatically establish falsity."""
    candidates = corruption_candidates(trace["claims"], source)
    assert any(c["type"] == "incorrect_date" for c in candidates)
    assert all(c["validity"] == "candidate_unreviewed" for c in candidates)
    assert all(c["human_review"] == "pending" for c in candidates)


def test_candidate_neutral_rejected(trace, source):
    """Even a plausible mutation is excluded when the checker cannot contradict it."""

    class AbstainingJudge:
        """Fixture checker deliberately returns unknown labels, not predictions."""

        def assess(self, claims, evidence):
            """Return input-indexed unknown labels for an eligibility invariant."""
            return [{"claim_id": c["id"], "status": "insufficient_evidence"} for c in claims]

    cs = validate_candidates(
        corruption_candidates(trace["claims"], source), trace["claims"], source, AbstainingJudge()
    )
    assert not any(c["primary_eligible"] for c in cs)


def test_v2_targets_join_after_features(trace, source):
    """Changing future utility labels changes targets but never the input matrix."""
    state = trace_state("q", "Who?", trace["claims"], source)
    r = {
        "pre_state": state.to_dict(),
        "baseline": {"correct": False},
        "singletons": {"C1": {"correct": True}, "C2": {"correct": False}},
        "supervision_scope": "v2_explicit_utility",
        "training_benefits": {"C1": -0.5, "C2": None},
    }
    x, e, y, mask = training_graph(r)
    assert y[0] == -0.5 and not mask[1]
    changed = copy.deepcopy(r)
    changed["training_benefits"]["C1"] = 0.5
    assert np.array_equal(x, training_graph(changed)[0])


def test_calibration_bins_and_metrics():
    """Boundary probabilities count once; PR area uses natural binary labels."""
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    ece, bins = reliability(y, p)
    assert ece == 0 and sum(b["n"] for b in bins) == 2
    assert binary_metrics(y, p, 0.5)["pr_auc_average_precision"] == 1


def test_question_bootstrap():
    """A constant question-level effect has a matching deterministic interval."""
    import pandas as pd

    assert question_interval(pd.Series([0.25, 0.25, 0.25])) == (0.25, 0.25)


def test_alignment_preserves_model_claim_text(trace, source):
    """Explicit source assignment cannot rewrite model facts or hide raw failure."""
    from chc.v2.alignment import align_uncited_artifact

    trace["claims"][0]["evidence_ids"] = []

    class SupportingJudge:
        """Fixture entailment scores for testing citation provenance only."""

        threshold = 0.85

        def probabilities(self, pairs):
            """Return predetermined support for the Ada evidence fixture."""
            return [{"entailment": 0.95 if "1980" in e else 0.1} for c, e in pairs]

    raw = json.dumps(trace)
    g = {"raw": raw, "trace": None, "validation_error": "Missing citation"}
    out = align_uncited_artifact(g, "Who?", source, SupportingJudge())
    assert out["trace"]["claims"][0]["text"] == trace["claims"][0]["text"]
    assert out["raw"] == raw and out["raw_validation_error"] == "Missing citation"
    assert out["trace"]["claims"][0]["evidence_ids"] == ["E1"]
    assert out["artifact_origin"] == "explicit_citation_aligned_model_claims"
    assert g["trace"] is None


def test_alignment_cannot_repair_bad_dependency(trace, source):
    """Source alignment must not silently invent or reverse dependency links."""
    from chc.v2.alignment import align_uncited_artifact

    trace["claims"][0].update(evidence_ids=[], depends_on=["C2"])

    class NeverCalledJudge:
        """An invariant guard: malformed structures never reach an evidence model."""

        def probabilities(self, pairs):
            """Fail if invalid dependencies bypass structural validation."""
            raise AssertionError("Must validate structure first")

    g = {"raw": json.dumps(trace), "trace": None, "validation_error": "Forward dependency"}
    assert align_uncited_artifact(g, "Who?", source, NeverCalledJudge())["trace"] is None


def test_api_requires_evidence_for_live(tmp_path):
    """Bad live requests are rejected before downloading/loading any model."""
    from chc.v2.web import PropagationService

    service = PropagationService(tmp_path)
    assert service.examples()["examples"] == []
    with pytest.raises(ValueError):
        service.analyze({"question": "Who?"})
    with pytest.raises(ValueError):
        service.analyze({"episode_id": "missing"})


def test_semantic_review_resolves_only_score_passing_deferral():
    """The semantic layer can resolve a negation deferral, not failed lexical scores."""
    from chc.v2.repair import RepairVerifier

    class LexicalFixture:
        """Supply a high-scoring but explicitly deferred real-code-shaped candidate."""

        support = 0.99

        def propose(self, ev, original):
            """Return a deterministic software fixture, never a research outcome."""
            return {
                "final_claim": original,
                "candidates": [
                    {
                        "proposal": "Ada is a scientist.",
                        "support_score": self.support,
                        "preservation_score": 0.99,
                        "thresholds": {"support": 0.95, "preservation": 0.95},
                        "reasons": ["Negation changed; requires semantic review"],
                        "accepted": False,
                    }
                ],
            }

    class SemanticFixture:
        """Make negation contradiction explicit for the repair-interface test."""

        def assess(self, claims, evidence):
            """Return fixed supported/contradicted labels for test statements."""
            return [
                {"status": "contradicted" if "not" in c["text"] else "supported"} for c in claims
            ]

    verifier = RepairVerifier.__new__(RepairVerifier)
    verifier.lexical = LexicalFixture()
    verifier.judge = SemanticFixture()
    claim = {"id": "C1", "text": "Ada is not a scientist.", "evidence_ids": ["E1"]}
    ev = {"E1": {"title": "Ada", "text": "Ada is a scientist."}}
    result = verifier.verify("Who is a scientist?", claim, ev)
    assert result["accepted"] and result["resolved_lexical_semantic_deferral"]
    verifier.lexical.support = 0.1
    assert not verifier.verify("Who is a scientist?", claim, ev)["accepted"]


def test_citation_retrieval_does_not_censor_false_claims(trace, source):
    """A false uncited continuation remains evaluable, not rejected as invalid."""
    from chc.v2.alignment import align_uncited_artifact

    trace["claims"][0].update(text="Ada was born in 1983.", evidence_ids=[])

    class ContradictionJudge:
        """Fixture distinguishes source relevance from factual support."""

        threshold = 0.85

        def probabilities(self, pairs):
            """Return contradiction for the relevant source, unknown otherwise."""
            return [
                {"entailment": 0.01, "contradiction": 0.97 if "1980" in e else 0.01}
                for c, e in pairs
            ]

    g = {"raw": json.dumps(trace), "trace": None, "validation_error": "Missing citation"}
    result = align_uncited_artifact(g, "When?", source, ContradictionJudge())
    assert result["trace"]["claims"][0]["text"] == "Ada was born in 1983."
    assert result["citation_alignment"][0]["retrieved_source_relation"] == "contradicted_candidate"


def test_provenance_mutation_is_not_factual_falsity(trace, source):
    """An overconfident NLI result cannot label a wrong citation a false fact."""

    class WrongCitationJudge:
        """Return source-dependent fixture labels to exercise the family gate."""

        def assess(self, claims, evidence):
            """Use cited ID for this fixture, not actual model inference."""
            return [
                {
                    "claim_id": c["id"],
                    "status": "supported" if c["evidence_ids"] == ["E1"] else "contradicted",
                }
                for c in claims
            ]

    candidates = [
        c
        for c in corruption_candidates(trace["claims"], source)
        if c["type"] == "evidence_misattribution"
    ]
    assert candidates
    assert not any(
        c["primary_eligible"]
        for c in validate_candidates(candidates, trace["claims"], source, WrongCitationJudge())
    )


def test_source_bridge_uses_only_available_passages():
    """A source mention adds an explicitly uncertain edge without future outputs."""
    from chc.v2.graph_features import enrich_source_bridges

    claims = [
        {"id": "C1", "text": "Ada is an author.", "depends_on": [], "evidence_ids": ["E1"]},
        {"id": "C2", "text": "Ben is an artist.", "depends_on": [], "evidence_ids": ["E2"]},
    ]
    evidence = {
        "E1": {"title": "Ada Author", "text": "Ada Author works with Ben Artist."},
        "E2": {"title": "Ben Artist", "text": "Ben Artist lives in Lima."},
    }
    original = trace_state("q", "Who?", claims, evidence)
    state = enrich_source_bridges(original)
    extra = [e for e in state.edges if e["kind"] == "source_entity_bridge_hypothesis"]
    assert len(extra) == 1 and extra[0]["source"] == "C1" and extra[0]["target"] == "C2"
    assert not extra[0]["verified_causal"] and len(original.edges) == 2


def test_atomic_units_copy_only_explicit_shared_subject():
    """Split an explicit coordinated predicate; do not invent a subject/entity."""
    from chc.v2.atomic import factual_units

    units = factual_units("Ada was born in 1980 and died in 2020.")
    assert [u["text"] for u in units] == ["Ada was born in 1980", "Ada died in 2020."]
    assert all(u["original_text"] == "Ada was born in 1980 and died in 2020." for u in units)
    listed = factual_units("The district includes Putnam and Westchester counties.")
    assert len(listed) == 1 and listed[0]["residual_compound_possible"]


def test_blank_review_does_not_create_human_labels(tmp_path):
    """A prepared worksheet must never masquerade as a completed review."""
    import pandas as pd
    from chc.v2.review import import_review

    sheet = tmp_path / "review.csv"
    key = tmp_path / "key.csv"
    output = tmp_path / "labels.jsonl"
    pd.DataFrame([{"review_id": "R1", "human_status": "", "notes": ""}]).to_csv(sheet, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "episode_id": "q",
                "variant": "corrupt",
                "machine_status": "supported",
            }
        ]
    ).to_csv(key, index=False)
    result = import_review(sheet, key, output, "reviewer_fixture")
    assert result["imported"] == 0 and result["pending_or_blank"] == 1
    assert output.read_text() == ""
    with pytest.raises(ValueError):
        import_review(sheet, key, output, "")


def test_sparse_training_gate_keeps_working_tabular_path(trace, source, tmp_path):
    """A failed complexity gate still trains the small control without embeddings."""
    from chc.v2.learning import train_models

    state = trace_state("q", "Who?", trace["claims"], source)
    common = {
        "pre_state": state.to_dict(),
        "baseline": {"valid": True, "correct": False},
        "singletons": {"C1": {"correct": False}, "C2": {"correct": False}},
        "benefits": {"C1": 0.0, "C2": None},
        "final_only_benefits": {"C1": 0.0, "C2": 0.0},
    }
    records = [
        dict(common, episode_id="t", question_id="t", split="train"),
        dict(common, episode_id="v", question_id="v", split="validation"),
    ]

    class NoEmbeddingJudge:
        """The sparse branch must not call an expensive semantic encoder."""

        model_id = "fixture_only"

        def embeddings(self, texts):
            """Fail if the model-signal gate is bypassed."""
            raise AssertionError("Sparse data should not run embedding training")

    result = train_models(records, NoEmbeddingJudge(), tmp_path, seeds=[42])
    assert not result["complex_model_gate_passed"]
    assert (tmp_path / "tabular_richer_all_42.joblib").exists()
    assert not (tmp_path / "graph_richer_42.pt").exists()


def test_selected_entity_swap_can_remain_true():
    """A real observed failure pattern is vetoed before any continuation is seen."""
    from chc.v2.validity import selected_corruption_validity

    prefix = [{"id": "C1", "text": "Ada is a writer."}, {"id": "C2", "text": "Ben is a writer."}]
    candidate = {"node_id": "C1", "modified": "Ben is a writer."}
    sources = {"E1": {"text": "Ada is a writer."}, "E2": {"text": "Ben is a writer."}}

    class SourceSupport:
        """Fixture support checker for a duplicated, still-true entity substitution."""

        threshold = 0.85

        def probabilities(self, pairs):
            """Only the matching subject's source entails the substituted claim."""
            return [{"entailment": 0.99 if c == e else 0.01} for c, e in pairs]

    result = selected_corruption_validity(candidate, prefix, sources, SourceSupport())
    assert not result["eligible"] and result["duplicate_supported_claim_ids"] == ["C2"]
    assert result["entailing_source_ids"] == ["E2"]


def test_reasoner_oom_retries_cpu_with_recorded_status(tmp_path):
    """A simulated CUDA OOM must recover explicitly; it is not a model benchmark."""
    import torch
    from chc.v2.backend import TraceBackend

    class Encoding(dict):
        """Keep fixture tensors on CPU while exercising the recovery control flow."""

        def to(self, device):
            """Avoid allocating CUDA for a deterministic software-only fixture."""
            return self

    class Tokenizer:
        """Minimal tokenizer interface for the backend error-path contract."""

        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            """Return a stable fixture prompt with no hidden labels."""
            return "fixture prompt"

        def __call__(self, *args, **kwargs):
            """Produce bounded token/mask tensors compatible with generation code."""
            return Encoding(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, *args, **kwargs):
            """Return schema-valid fixture content, never a reported real outcome."""
            return json.dumps(
                {
                    "claims": [
                        {
                            "id": "C1",
                            "text": "Ada is a writer.",
                            "depends_on": [],
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "final_answer": "Ada",
                }
            )

    class Model:
        """Raise once, then return CPU tokens after the explicit fallback."""

        calls = 0

        def generate(self, **kwargs):
            """Simulate only the exception mechanism, not language-model behavior."""
            self.calls += 1
            if self.calls == 1:
                raise torch.cuda.OutOfMemoryError("Fixture OOM")
            return torch.tensor([[1, 2, 3, 4]])

        def float(self):
            """Preserve the fixture while matching the real fallback interface."""
            return self

        def to(self, device):
            """Record-free fixture device conversion; result is already CPU."""
            return self

    b = TraceBackend.__new__(TraceBackend)
    b.cache = tmp_path
    (tmp_path / "responses").mkdir()
    b.model_id = "fixture"
    b.revision = "fixture"
    b.snapshot_ids = ["fixture"]
    b.max_new_tokens = 20
    b.constrained = False
    b.causal = True
    b.tokenizer = Tokenizer()
    b.model = Model()
    b.device = torch.device("cuda")
    b.fallbacks = []
    b.actual_generations = 0
    b.cache_hits = 0
    result = b.generate("Who?", {"E1": {"text": "Ada is a writer.", "title": "Ada"}}, seed=1)
    assert result["device"] == "cpu" and b.fallbacks == ["generation_oom_cpu"]
    assert b.model.calls == 2 and result["trace"] is not None
