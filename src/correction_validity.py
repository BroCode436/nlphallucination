"""Local, trained correction verifier. Lexical features are not semantic proof."""

from typing import Any as _DocAny
import re
from functools import lru_cache
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import joblib
from decimal import Decimal

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "जनवरी": "01",
    "फरवरी": "02",
    "मार्च": "03",
    "अप्रैल": "04",
    "मई": "05",
    "जून": "06",
    "जुलाई": "07",
    "अगस्त": "08",
    "सितंबर": "09",
    "अक्टूबर": "10",
    "नवंबर": "11",
    "दिसंबर": "12",
}
STOP = set(
    "the a an is was were in on at and it its by of to has had है था में को और ने यह".split()
)


def tokens(text: _DocAny) -> _DocAny:
    """Return a fresh list of normalized multilingual content tokens.

    Input text is converted to a string and passed through a bounded immutable
    cache. Unicode normalization preserves Hindi combining marks; month names are
    mapped to numbers and common stopwords are removed. The fresh list prevents
    callers from mutating cached state. This lexical representation cannot identify
    truth or semantic entailment by itself.
    """
    return list(_cached_tokens(str(text)))


@lru_cache(maxsize=8192)
def _cached_tokens(text: str) -> tuple[str, ...]:
    """Cache immutable Unicode tokens for repeated claim/evidence comparisons.

    Args:
        text: Exact input string; changing text produces a different cache entry.
    Returns:
        Immutable token tuple. The public wrapper returns a fresh list to preserve
        its previous interface and avoid shared mutable cache state.
    Limitations:
        Cache size is bounded; this changes runtime cost, not model predictions.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    # Preserve combining marks in Hindi instead of breaking words with ASCII regexes.
    text = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in text)
    return tuple(MONTHS.get(t, t) for t in text.split() if t not in STOP)


def words(text: _DocAny) -> _DocAny:
    """Return the unique non-integer content tokens used in lexical matching.

    The input is any string-like value accepted by tokens(). Output is a set, so
    word repetition and argument order are lost. Decimal strings and ordinary words
    are treated according to the shared tokenizer. Numeric consistency is checked
    separately; a high word overlap is not proof of a valid factual correction.
    """
    return set(t for t in tokens(text) if not t.isdigit())


def numbers(text: _DocAny) -> _DocAny:
    """Return a fresh set of canonical numeric/date components in text.

    The bounded helper normalizes Unicode digits and Decimal values, including
    signed/grouped decimals. Month names contribute month numbers. The public fresh
    set preserves caller isolation from the cached frozenset. It extracts values,
    not their semantic entity binding; later features compare relevant sentences.
    """
    return set(_cached_numbers(str(text)))


@lru_cache(maxsize=8192)
def _cached_numbers(text: str) -> frozenset[str]:
    """Reuse exact signed/decimal normalization without sharing a mutable set.

    Args:
        text: Exact evidence or claim string.
    Returns:
        Immutable normalized numeric values. Public numbers() returns a fresh set.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in normalized)
    values = {
        format(Decimal(m.group().replace(",", "")).normalize(), "f")
        for m in re.finditer(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)", normalized)
    }
    values.update(str(int(MONTHS[t])) for t in normalized.split() if t in MONTHS)
    return frozenset(values)


def clauses(text: _DocAny) -> _DocAny:
    """Split source text into nonempty sentence/clause candidate strings.

    English sentence punctuation, Hindi danda, semicolons and newlines delimit
    candidates. The original wording is preserved after surrounding whitespace is
    stripped. Abbreviations and complex punctuation can be split incorrectly; this
    is a lightweight proposal generator rather than a trained atomic-claim parser.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+|[;\n]+", text) if s.strip()]


def recall(a: _DocAny, b: _DocAny) -> _DocAny:
    """Measure set coverage as |a intersect b| / max(|a|, 1).

    Inputs must be sets of tokens or normalized numbers. Output is a float in
    [0,1]. An empty first set yields zero, avoiding division by zero. Callers that
    want vacuous numeric coverage explicitly handle the no-number case themselves.
    This asymmetric coverage differs from precision and ignores order.
    """
    return len(a & b) / max(len(a), 1)


def same_clause_order(a: _DocAny, b: _DocAny) -> _DocAny:
    """Check whether conjunction-separated arguments match after a shared prefix.

    Inputs are original/candidate strings. The function removes their common token
    prefix, splits conjunctions before stopword removal, and compares sorted clause
    token lists. This detects some order/role changes while tolerating reordered
    independent clauses. It is a heuristic guard, not a semantic relation parser.
    """
    at, bt = tokens(a), tokens(b)
    prefix = 0
    while prefix < min(len(at), len(bt)) and at[prefix] == bt[prefix]:
        prefix += 1
    # Split before stopword removal so conjunctions remain available.
    split = lambda s: [tokens(c) for c in re.split(r"\b(?:and|aur)\b|और", s, flags=re.I)]
    ac, bc = split(a), split(b)
    if ac:
        ac[0] = ac[0][prefix:]
    if bc:
        bc[0] = bc[0][prefix:]
    return sorted(ac) == sorted(bc)


FEATURES = [
    "original_word_recall",
    "candidate_word_precision",
    "supported_original_recall",
    "unsupported_candidate_fraction",
    "best_sentence_precision",
    "best_sentence_recall",
    "number_support",
    "number_binding",
    "original_number_overlap",
    "length_ratio",
    "sequence_similarity",
    "identical",
    "negation_changed",
    "evidence_present",
    "candidate_present",
    "original_present",
    "original_support",
    "candidate_support_delta",
]


def features(evidence: _DocAny, original: _DocAny, candidate: _DocAny) -> _DocAny:
    """Build the 18-value lexical support/preservation feature vector.

    Inputs are evidence, original claim and proposed replacement strings. Output
    order exactly matches FEATURES and the saved estimators. Features cover token
    recall/precision, unsupported content, best-sentence overlap, numeric support
    and binding, edit length, sequence similarity, identity, negation and presence.
    Relevant-sentence numeric binding avoids accepting a number merely because it
    appears elsewhere in a document. Empty inputs produce bounded features, while
    assess() records explicit missing-input rejection reasons. These features are
    learned proxies and can miss paraphrase, entailment or entity-role errors.
    """
    ew, ow, cw = words(evidence), words(original), words(candidate)
    cn, on = numbers(candidate), numbers(original)
    sents = clauses(evidence)
    best = max(sents, key=lambda s: (recall(cw, words(s)), recall(words(s), cw)), default="")
    # Numbers must be supported by the same relevant sentence, not just anywhere.
    ns = recall(cn, numbers(evidence)) if cn else 1.0
    nb = recall(cn, numbers(best)) if cn else 1.0
    cp = recall(cw, ew)
    op = recall(ow, ew)
    neg = {"not", "never", "no", "नहीं", "nahi", "nahin"}
    return [
        recall(ow, cw),
        recall(cw, ow),
        recall(ow & ew, cw),
        1 - cp,
        recall(cw, words(best)),
        recall(words(best), cw),
        ns,
        nb,
        recall(on, cn) if on else 1.0,
        len(tokens(candidate)) / max(len(tokens(original)), 1),
        SequenceMatcher(None, tokens(original), tokens(candidate)).ratio(),
        float(tokens(original) == tokens(candidate)),
        float(bool(ow & neg) != bool(cw & neg)),
        float(bool(ew)),
        float(bool(cw)),
        float(bool(ow)),
        op,
        cp - op,
    ]


class CorrectionValidityModel:
    """Assess factual repair support and original-information preservation.

    Initialize with a trusted saved model path, or use the project's existing
    default. Attributes path/artifact expose fitted support and preservation
    estimators, decision thresholds and artifact version. assess() evaluates one
    proposal; propose() constructs source-based alternatives and rolls back failed
    edits. The class implements the retained lexical safety baseline. It explicitly
    defers some semantic cases to review; V2's separate RepairVerifier investigates
    those cases without retraining or changing these existing weights.
    """

    def __init__(self, model_path: _DocAny = None) -> None:
        """Load the trusted local correction-validity joblib artifact.

        model_path is optional; by default the project models/correction_validity.joblib
        is used. Attributes path and artifact retain its provenance, fitted support and
        preservation classifiers, thresholds and version. Missing/incompatible files
        raise normal loading errors. Only trusted local artifacts should be loaded.
        """
        self.path = (
            Path(model_path)
            if model_path
            else Path(__file__).resolve().parents[1] / "models" / "correction_validity.joblib"
        )
        self.artifact = joblib.load(self.path)

    def assess(self, evidence: _DocAny, original: _DocAny, candidate: _DocAny) -> _DocAny:
        """Assess one proposal without losing the original claim on rejection.

        Inputs are supplied evidence, original text and candidate text. The two saved
        classifiers receive a [1,18] feature array and return support/preservation
        scores. Learned threshold failures plus explicit number, negation, argument
        order and qualifier checks populate rejection reasons. Output includes scores,
        thresholds, cited excerpt and final_claim: candidate if accepted, otherwise the
        original. Semantic-review flags are deliberate abstentions; V2 can investigate
        them without changing this retained lexical verifier. Acceptance is a proxy,
        not independent proof of truth or preservation.
        """
        x = np.asarray([features(evidence, original, candidate)])
        support = float(self.artifact["support"].predict_proba(x)[0, 1])
        preservation = float(self.artifact["preservation"].predict_proba(x)[0, 1])
        thresholds = self.artifact["thresholds"]
        reasons = []
        if not all(str(t).strip() for t in (evidence, original, candidate)):
            reasons.append("Missing evidence, original, or proposal")
        if support < thresholds["support"]:
            reasons.append("Proposal lacks sufficient learned evidence support")
        if preservation < thresholds["preservation"]:
            reasons.append("Proposal may change the subject or remove supported details")
        # Hard constraints cannot be outweighed by another score.
        if x[0, 7] < 1:
            reasons.append(
                "A proposed number/date is absent from the most relevant evidence sentence"
            )
        if x[0, 12]:
            reasons.append("Negation changed; requires semantic review")
        best_sentence = max(
            clauses(evidence), key=lambda s: recall(words(candidate), words(s)), default=""
        )
        negations = {"not", "never", "no", "नहीं", "nahi", "nahin"}
        if bool(words(candidate) & negations) != bool(words(best_sentence) & negations):
            reasons.append("Proposal and evidence differ in negation; requires semantic review")
        if (
            words(candidate) == words(best_sentence)
            and numbers(candidate) == numbers(best_sentence)
            and not same_clause_order(candidate, best_sentence)
        ):
            reasons.append("Relation argument order changed; requires semantic review")
        qualifiers = {
            "if",
            "unless",
            "only",
            "except",
            "until",
            "यदि",
            "अगर",
            "केवल",
            "agar",
            "sirf",
        }
        if (words(best_sentence) & qualifiers) - (words(candidate) & qualifiers):
            reasons.append("Evidence contains a condition or restriction absent from the proposal")
        accepted = not reasons
        cited = max(clauses(evidence), key=lambda s: recall(words(candidate), words(s)), default="")
        return {
            "accepted": accepted,
            "support_score": support,
            "preservation_score": preservation,
            "thresholds": thresholds,
            "reasons": reasons,
            "status": "accepted" if accepted else "needs_review",
            "final_claim": candidate if accepted else original,
            "evidence_excerpt": cited,
            "model_version": self.artifact["version"],
            "assessment_basis": "Trained lexical verifier; not independent factual proof",
        }

    def propose(self, evidence: _DocAny, original: _DocAny) -> _DocAny:
        """Generate and rank source-grounded lexical repair proposals.

        Inputs are evidence and the original claim. Candidates include the original,
        source clauses and numeric replacements when number counts match. Each unique
        candidate is assessed by the unchanged verifier. Among accepted candidates the
        maximum of the minimum support/preservation score is selected. The returned
        dictionary preserves every rejected proposal and returns the original when
        none passes. V2 explicitly inspects semantic-review deferrals rather than
        mistaking this unchanged fallback for the only available proposal.
        """
        proposals = [original]
        for sentence in clauses(evidence):
            proposals.append(sentence)
            pattern = r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)"
            old_nums = re.findall(pattern, original)
            new_nums = re.findall(pattern, sentence)
            if len(old_nums) == len(new_nums) and old_nums:
                it = iter(new_nums)
                proposals.append(re.sub(pattern, lambda m: next(it), original))
        assessed = [
            dict(self.assess(evidence, original, p), proposal=p) for p in dict.fromkeys(proposals)
        ]
        valid = [r for r in assessed if r["accepted"]]
        best = (
            max(valid, key=lambda r: min(r["support_score"], r["preservation_score"]))
            if valid
            else None
        )
        return {
            "original": original,
            "final_claim": best["proposal"] if best else original,
            "status": (
                ("unchanged" if best["proposal"] == original else "repaired")
                if best
                else "needs_review"
            ),
            "selected": best,
            "candidates": assessed,
        }
