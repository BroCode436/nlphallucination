"""Real HotpotQA/RAGTruth loading and explicit dataset integrity checks.

HotpotQA rows are downloaded from Hugging Face's official dataset viewer API in
bounded pages. No generated fixture is substituted when the network fails.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty UTF-8 JSONL records with ordinary parse errors preserved.

    Args:
        path: Existing JSONL file, one JSON object per line.
    Returns:
        Parsed dictionaries in file order.
    """
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save reproducible Unicode JSONL without ASCII-escaping Hindi characters.

    Args:
        path: Output file; parent directories are created.
        rows: Serializable record dictionaries.
    Returns:
        None; replaces only the explicitly named output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def fetch_hotpot(cache: Path, split: str, count: int) -> list[dict[str, Any]]:
    """Fetch a bounded, reproducibly selected prefix of a real official split.

    Args:
        cache: Local directory holding original downloaded row dictionaries.
        split: Official 'train' or 'validation'; never hidden test answers.
        count: Maximum rows to retrieve in pages of at most 100.
    Returns:
        Real dataset rows, each with download provenance attached.
    Raises:
        ValueError: For invalid count/split or an unexpectedly short response.
        RuntimeError: After three unsuccessful HTTP attempts for a page.
    Limitations:
        A prefix is not a random representative sample of the full benchmark.
    """
    if split not in ('train', 'validation') or count <= 0:
        raise ValueError('Use train/validation and a positive row count')
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f'hotpot_{split}_{count}.jsonl'
    if dest.exists():
        return read_jsonl(dest)
    rows: list[dict[str, Any]] = []
    for offset in range(0, count, 100):
        params = urllib.parse.urlencode({'dataset': 'hotpotqa/hotpot_qa', 'config': 'distractor',
                                         'split': split, 'offset': offset, 'length': min(100, count-offset)})
        url = 'https://datasets-server.huggingface.co/rows?' + params
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'CHC-research/0.1'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    page = json.load(response)
                break
            except Exception as error:
                if attempt == 2:
                    raise RuntimeError(f'HotpotQA page download failed: {url}') from error
                time.sleep(1 + attempt)
        for entry in page['rows']:
            row = entry['row']
            row['_provenance'] = {'dataset': 'hotpotqa/hotpot_qa', 'config': 'distractor',
                                  'official_split': split, 'row_index': entry['row_idx'], 'url': url,
                                  'license': 'CC-BY-SA-4.0; retain attribution'}
            rows.append(row)
    if len(rows) != count:
        raise ValueError(f'Expected {count} rows, received {len(rows)}')
    write_jsonl(dest, rows)
    return rows


def supporting_sentences(row: dict[str, Any], limit: int = 3) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Extract explicitly annotated supporting sentences, not invented reasoning.

    Args:
        row: One official HotpotQA row containing context and supporting_facts.
        limit: Maximum intermediate nodes, kept small for laptop experiments.
    Returns:
        Node source records and a passage dictionary containing full relevant docs.
    Raises:
        ValueError: When annotations point outside the provided context.
    Limitations:
        A supporting sentence can contain several propositions. This baseline is
        sentence extraction, not a validated semantic atomic-claim parser.
    """
    if limit < 1:
        raise ValueError('Supporting sentence limit must be positive')
    docs = dict(zip(row['context']['title'], row['context']['sentences']))
    facts: list[dict[str, str]] = []
    evidence: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for title, index in zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id']):
        if title not in docs or not 0 <= index < len(docs[title]):
            raise ValueError('Supporting annotation points outside context')
        text = docs[title][index].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        passage_id = f'doc{list(docs).index(title)}'
        evidence[passage_id] = {'title': title, 'text': ' '.join(docs[title]), 'source': 'HotpotQA context'}
        facts.append({'text': text, 'evidence_id': passage_id, 'title': title,
                      'sentence_index': str(index), 'validity_basis': 'official_support_annotation; human_review_pending'})
    # Cover distinct annotated documents before taking additional sentences from
    # the same document. Otherwise a three-node cap can accidentally remove the
    # second hop when the first document has three supporting annotations.
    first_per_document: list[int] = []
    represented: set[str] = set()
    for index, fact in enumerate(facts):
        if fact['evidence_id'] not in represented:
            first_per_document.append(index)
            represented.add(fact['evidence_id'])
    chosen = first_per_document[:limit]
    for index in range(len(facts)):
        if index not in chosen and len(chosen)<limit:
            chosen.append(index)
    return [facts[i] for i in sorted(chosen)], evidence


def split_questions(train: list[dict[str, Any]], test: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Partition official train into fit/tuning and reserve official validation.

    Args:
        train: Official training rows only.
        test: Official validation rows held out for final analysis.
        seed: Seed for question-level shuffle of the training rows.
    Returns:
        Dictionary with 'train', 'validation', and 'test' question lists.
    Raises:
        ValueError: If question IDs overlap or training has fewer than five rows.
    """
    if len(train) < 5 or {r['id'] for r in train} & {r['id'] for r in test}:
        raise ValueError('Need disjoint questions and at least five training rows')
    shuffled = list(train)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(.8 * len(shuffled)))
    return {'train': shuffled[:cut], 'validation': shuffled[cut:], 'test': list(test)}


def audit_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Audit question/evidence overlap and report shared entities as limitations.

    Args:
        splits: Question-level partitions, prior to error-condition expansion.
    Returns:
        Counts, pairwise overlap counts, and a nonrandom-prefix sampling warning.
    Raises:
        ValueError: If question IDs overlap between any split pair.
    """
    report: dict[str, Any] = {'counts': {k: len(v) for k, v in splits.items()}, 'overlaps': {},
                              'sampling': 'Bounded official prefixes; not population-random',
                              'template_overlap': 'Natural questions; no independently annotated template IDs available'}
    for a, ar in splits.items():
        for b, br in splits.items():
            if a >= b:
                continue
            ids = len({r['id'] for r in ar} & {r['id'] for r in br})
            entities = len({t for r in ar for t in r['context']['title']} & {t for r in br for t in r['context']['title']})
            report['overlaps'][f'{a}/{b}'] = {'question_ids': ids, 'document_titles': entities}
            if ids:
                raise ValueError('Question leakage detected')
    return report


def load_ragtruth(project: Path, count: int = 256, seed: int = 42) -> pd.DataFrame:
    """Load an actual held-out detector sample without using it for CHC training.

    Args:
        project: Existing project root containing processed RAGTruth CSVs.
        count: Maximum uniformly sampled test claims.
        seed: Sampling seed recorded in the detector evaluation report.
    Returns:
        Natural-distribution test rows with evidence, claim, label, and source_id.
    Raises:
        FileNotFoundError: If the real processed dataset is unavailable.
    """
    frame = pd.read_csv(project/'data'/'processed'/'ragtruth_claims_test.csv')
    return frame.sample(min(count, len(frame)), random_state=seed).reset_index(drop=True)
