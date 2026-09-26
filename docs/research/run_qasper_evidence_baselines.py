"""Offline, given-document evidence-selection diagnostic on a frozen dev pool.

BM25/source-order never receive reference answers. Gold is used only by scoring
and an explicitly non-deployable oracle over subsets of reference-matched units.
Budgets count the actual rendered pack with the pinned local BGE tokenizer;
these are encoder-token diagnostics, not a generator or full RAG evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import tarfile
import time
import unicodedata

from transformers import AutoTokenizer

from qasper_metrics import evidence_metrics, references_from_annotations
from qasper_alignment_v2 import native_text_blocks


@dataclass(frozen=True)
class Unit:
    unit_id: str
    order: int
    kind: str
    start: int
    end: int
    text: str
    native_text: str


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def lexical_tokens(text):
    return re.findall(r"\w+", text.casefold())


def native_pointer(paper, pointer, source_id):
    prefix = f"/{source_id}/"
    if not pointer.startswith(prefix):
        raise ValueError("native locator refers to a different paper")
    parts = pointer[len(prefix):].split("/")
    # The whitelist cannot read qas, answers, worker fields or other gold.
    if parts in (["title"], ["abstract"]):
        value = paper[parts[0]]
    elif len(parts) in (3, 4) and parts[0] == "full_text" and parts[2] in {"section_name", "paragraphs"}:
        section = paper["full_text"][int(parts[1])]
        if parts[2] == "section_name" and len(parts) == 3:
            value = section["section_name"]
        elif parts[2] == "paragraphs" and len(parts) == 4:
            value = section["paragraphs"][int(parts[3])]
        else:
            raise ValueError("unsupported source locator")
    else:
        raise ValueError("source locator is outside the model-visible whitelist")
    if not isinstance(value, str):
        raise ValueError("source unit is not text")
    return value


def build_units(document, paper):
    units = []
    blocks = {}
    for block in document["blocks"]:
        blocks.setdefault((block["source_locator"]["json_pointer"], block["kind"]), []).append(block)
    for kind, native_locator, canonical_locator, native in native_text_blocks(paper, document["source_id"]):
        if not native.strip():
            continue
        native = native_pointer(paper, native_locator, document["source_id"])
        matches = blocks.get((canonical_locator, "heading" if kind == "title" else kind), [])
        if len(matches) != 1:
            raise ValueError("native unit must map to exactly one canonical block")
        block = matches[0]
        start, end = block["char_span"]
        if not 0 <= start <= end <= len(document["canonical_text"]):
            raise ValueError("invalid canonical span")
        text = document["canonical_text"][start:end]
        if normalized(text) != normalized(native):
            raise ValueError("native and canonical unit do not match")
        if text.strip():
            units.append(Unit(block["block_id"], len(units), kind, start, end, text, native))
    if not units:
        raise ValueError("empty candidate document")
    return units


def bm25_ranking(question, units, k1=1.5, b=.75):
    terms = [Counter(lexical_tokens(unit.text)) for unit in units]
    lengths = [sum(counter.values()) for counter in terms]
    average = sum(lengths) / max(len(lengths), 1)
    df = Counter(term for counter in terms for term in counter)
    query = set(lexical_tokens(question))
    scores = []
    for index, counter in enumerate(terms):
        value = 0.
        for term in query:
            freq = counter[term]
            if freq:
                idf = math.log(1 + (len(units) - df[term] + .5) / (df[term] + .5))
                value += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * lengths[index] / max(average, 1)))
        scores.append(value)
    return sorted(range(len(units)), key=lambda index: (-scores[index], units[index].order))


def render_pack(units, selected):
    return "\n\n".join(f"[{units[index].unit_id}]\n{units[index].text}"
                       for index in sorted(selected, key=lambda index: units[index].order))


class PackCounter:
    def __init__(self, tokenizer, units, deadline=math.inf):
        self.tokenizer, self.units, self.cache = tokenizer, units, {(): 0}
        self.deadline = deadline

    def __call__(self, selected):
        check_time(self.deadline)
        key = tuple(sorted(selected))
        if key not in self.cache:
            self.cache[key] = len(self.tokenizer.encode(render_pack(self.units, key), add_special_tokens=True, truncation=False))
        return self.cache[key]


def pack_ranked(units, ranking, budget, count, max_units=None):
    chosen, seen_text = [], set()
    for index in ranking:
        # Exact native duplicates are a single evidence item; retain first ranked location.
        if units[index].native_text in seen_text:
            continue
        if count([*chosen, index]) <= budget:
            chosen.append(index)
            seen_text.add(units[index].native_text)
            if max_units is not None and len(chosen) >= max_units:
                break
    return sorted(chosen)


def check_time(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("bounded development diagnostic time budget reached")


def oracle_candidates(units, references, count, max_reference_units=16, deadline=math.inf):
    by_text = {}
    for index, unit in enumerate(units):
        # Candidate alternatives with identical native text are equivalent for the
        # official metric. This oracle uses the first source location, explicitly.
        by_text.setdefault(unit.native_text, index)
    selections = {()}
    for reference in references:
        available = sorted({by_text[text] for text in reference if text in by_text})
        if len(available) > max_reference_units:
            raise ValueError("oracle exhaustive-subset cap exceeded; no approximate fallback")
        for size in range(1, len(available) + 1):
            check_time(deadline)
            selections.update(itertools.combinations(available, size))
    return [(selected, count(selected)) for selected in sorted(selections)]


def choose_oracle(options, budget, units, annotations, deadline=math.inf):
    best, best_key = (), None
    for selected, tokens in options:
        check_time(deadline)
        if tokens > budget:
            continue
        metric = evidence_metrics([units[index].native_text for index in selected], annotations)
        key = (metric["evidence_f1"], metric["evidence_recall"], -tokens)
        if best_key is None or key > best_key:
            best, best_key = selected, key
    return list(best)


def score_selection(units, chosen, annotations, count, budget):
    tokens = count(chosen)
    if tokens > budget:
        raise ValueError("actual rendered evidence exceeds token budget")
    predicted = [units[index].native_text for index in chosen]
    metrics = evidence_metrics(predicted, annotations)
    text_metrics = evidence_metrics(predicted, annotations, text_evidence_only=True)
    return {"official_evidence_f1": metrics["evidence_f1"],
            "reference_evidence_recall": metrics["evidence_recall"],
            "official_text_only_evidence_f1": text_metrics["evidence_f1"],
            "actual_evidence_tokens": tokens, "selected_units": len(chosen),
            "selected_ids": [units[index].unit_id for index in chosen],
            "pack_sha256": hashlib.sha256(render_pack(units, chosen).encode()).hexdigest()}


def aggregate(records):
    groups = {}
    for row in records:
        groups.setdefault((row["method"], row["budget"]), []).append(row)
    summaries = []
    for (method, budget), values in sorted(groups.items()):
        by_doc = {}
        for row in values:
            by_doc.setdefault(row["doc_id"], []).append(row)
        item = {"method": method, "budget": budget, "questions": len(values), "documents": len(by_doc)}
        for field in ("official_evidence_f1", "reference_evidence_recall", "official_text_only_evidence_f1", "actual_evidence_tokens"):
            item[field + "_question_macro"] = sum(row[field] for row in values) / len(values)
            item[field + "_document_macro"] = sum(sum(row[field] for row in rows) / len(rows) for rows in by_doc.values()) / len(by_doc)
        summaries.append(item)
    return summaries


def load_frozen_pool(pool, sidecar):
    pool, sidecar = Path(pool), Path(sidecar)
    manifest = json.loads((pool / "pool_manifest.json").read_text(encoding="utf-8"))
    candidates_path = pool / "candidates.jsonl"
    if digest(candidates_path) != manifest["candidate_manifest_sha256"]:
        raise ValueError("candidate manifest hash mismatch")
    audit_path = sidecar.parent / "alignment_audit_v2.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(sidecar) != audit["sidecar_sha256"]:
        raise ValueError("v2 sidecar hash mismatch")
    if audit["input_sha256"].get(str(candidates_path.resolve())) != digest(candidates_path):
        raise ValueError("v2 sidecar references a different frozen pool")
    candidates = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    if not 1 <= len(candidates) <= 32 or any(row["official_split"] != "validation" for row in candidates):
        raise ValueError("requires a frozen non-test validation pool of at most 32 documents")
    by_doc = {row["doc_id"]: row for row in candidates}
    if len(by_doc) != len(candidates):
        raise ValueError("duplicate candidate document ID")
    qa_rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    if len(qa_rows) > 200 or any(row["doc_id"] not in by_doc or row["official_split"] != "validation" for row in qa_rows):
        raise ValueError("sidecar falls outside diagnostic scope")
    if len({row["question_id"] for row in qa_rows}) != len(qa_rows):
        raise ValueError("duplicate question ID")
    if {row["doc_id"] for row in qa_rows} != set(by_doc):
        raise ValueError("sidecar is missing a frozen document")
    for row in qa_rows:
        candidate = by_doc[row["doc_id"]]
        if row["source_id"] != candidate["source_id"] or row["family_id"] != candidate["family_id"] or row["canonical_body_sha256"] != candidate["normalized_body_sha256"]:
            raise ValueError("sidecar/candidate lineage mismatch")
    return manifest, candidates, qa_rows


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pool = Path(args.pool)
    manifest, candidates, qa_rows = load_frozen_pool(pool, args.sidecar)
    wanted = {row["doc_id"] for row in candidates}
    shard = Path(next(path for path in manifest["input_sha256"] if "documents-" in path))
    alignment = json.loads((pool / "native_qa_alignment.json").read_text(encoding="utf-8"))
    archive = Path(next(path for path in alignment["input_sha256"] if path.endswith("qasper-train-dev-v0.3.tgz")))
    for path, expected in ((shard, manifest["input_sha256"][str(shard)]), (archive, alignment["input_sha256"][str(archive)])):
        if digest(path) != expected:
            raise ValueError("source hash differs from frozen lineage")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("qasper-dev-v0.3.json")
        if not member.isfile() or member.size > 32 * 1024 * 1024:
            raise ValueError("unexpected native validation member")
        with tar.extractfile(member) as stream:
            raw = json.load(stream)
    documents = {}
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["doc_id"] in wanted:
                if row["original_split"] != "validation":
                    raise ValueError("canonical split mismatch")
                # Explicitly exclude qas before candidate construction.
                paper = {key: raw[row["source_id"]][key] for key in ("title", "abstract", "full_text")}
                documents[row["doc_id"]] = build_units(row, paper)
    del raw
    if set(documents) != wanted:
        raise ValueError("missing canonical candidates")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_paths = [Path(args.tokenizer) / name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "sentencepiece.bpe.model") if (Path(args.tokenizer) / name).exists()]
    input_paths = [pool / "candidates.jsonl", pool / "pool_manifest.json", Path(args.sidecar), Path(args.sidecar).parent / "alignment_audit_v2.json", shard, archive, *tokenizer_paths]
    input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
    deadline = started + args.max_seconds
    counters = {doc_id: PackCounter(tokenizer, units, deadline) for doc_id, units in documents.items()}
    records, coverage = [], []
    for qa in qa_rows:
        if time.monotonic() > deadline:
            raise TimeoutError("bounded development diagnostic time budget reached")
        units = documents[qa["doc_id"]]
        counter = counters[qa["doc_id"]]
        annotations = qa["answer_annotations"]
        references = [reference["evidence"] for reference in references_from_annotations(annotations)]
        rankings = {"bm25": bm25_ranking(qa["question"], units), "source_order": list(range(len(units)))}
        options = oracle_candidates(units, references, counter, deadline=deadline)
        native_strings = {unit.native_text for unit in units}
        coverage.append({"doc_id": qa["doc_id"], "question_id": qa["question_id"],
            "reference_sizes": [len(ref) for ref in references],
            "reachable_reference_items": [sum(text in native_strings for text in ref) for ref in references],
            "oracle_subsets_enumerated": len(options)})
        for budget in args.budgets:
            selections = {method: pack_ranked(units, ranking, budget, counter) for method, ranking in rankings.items()}
            for topk in (1, 3, 5):
                selections[f"bm25_top{topk}"] = pack_ranked(units, rankings["bm25"], budget, counter, max_units=topk)
            selections["empty"] = []
            selections["gold_subset_oracle"] = choose_oracle(options, budget, units, annotations, deadline)
            for method, selected in selections.items():
                result = score_selection(units, selected, annotations, counter, budget)
                records.append({"doc_id": qa["doc_id"], "question_id": qa["question_id"],
                                "method": method, "budget": budget, **result})
    if any(digest(Path(path)) != expected for path, expected in input_hashes.items()):
        raise ValueError("an input changed during the diagnostic")
    report = {"status": "completed", "scope": "given-document evidence-selection development diagnostic",
        "independent_evaluation": False, "answer_generation_performed": False,
        "test_payload_read": False, "api_calls": 0, "budgets": args.budgets,
        "token_budget_definition": "BGE tokenizer, complete rendered evidence pack including unit IDs and special tokens; query/instructions excluded; empty pack zero",
        "tokenizer": str(Path(args.tokenizer).resolve()), "tokenizer_sha256": digest(Path(args.tokenizer) / "tokenizer.json"),
        "candidate_policy": "All nonempty native title/section heading/abstract/paragraph units verified against canonical spans; synthetic Abstract heading excluded; full units only; exact duplicate strings selected once",
        "bm25": {"k1": 1.5, "b": .75, "tokenization": "Unicode word runs, casefold", "ties": "source order", "top_k": [1, 3, 5], "top_k_policy": "up to k distinct complete units that fit; overlong units skipped"},
        "oracle_scope": "Exact exhaustive subsets within each reachable gold reference, at most 16 unique units; first source occurrence for duplicate strings. Feasible gold-guided reference, not an unrestricted oracle over arbitrary spans.",
        "question_count": len(qa_rows), "document_count": len(documents),
        "input_sha256": input_hashes, "input_hashes_unchanged": True,
        "source_sha256": {path.name: digest(path) for path in (Path(__file__), Path(__file__).with_name("qasper_metrics.py"), Path(__file__).with_name("qasper_alignment_v2.py"))},
        "all_packs_within_budget": all(row["actual_evidence_tokens"] <= row["budget"] for row in records),
        "elapsed_seconds": time.monotonic() - started, "metrics": aggregate(records),
        "limits": ["No answer quality or JEV architecture claim.", "This exposed development pool is not independent confirmation data.",
                   "BGE evidence budgets cannot be substituted for an unselected generator tokenizer.",
                   "Official string evidence matching and heading annotations do not establish semantic sufficiency."]}
    for name, rows in (("per_question.jsonl", records), ("candidate_coverage.jsonl", coverage)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "question_count", "document_count", "elapsed_seconds", "all_packs_within_budget", "metrics")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--max_seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 900 or not args.budgets or len(args.budgets) > 3 or any(budget < 64 or budget > 4096 for budget in args.budgets):
        raise ValueError("development diagnostic caps exceeded")
    run(args)
