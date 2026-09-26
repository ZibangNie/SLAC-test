"""Frozen CPU BM25 versus cached dense retrieval, with independent replay audit.

Source-only preparation precedes outcomes. No model inference, GPU, provider,
key, new dependency, query filtering, parameter sweep or official test QA.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time
import unicodedata

import numpy as np
import torch

import prepare_qasper_relation_pilot as pilot
import run_qasper_corpus_bridge as bridge
from run_qasper_evidence_baselines import PackCounter, check_time, digest, lexical_tokens, score_selection
from qasper_metrics import evidence_metrics


SCHEMA = "slac-qasper-lexical-baseline-v1"
SCOPES = ("given_document", "corpus_32")
METHODS = ("dense_cached", "bm25")
METRICS = ("source_qualified_evidence_f1", "source_qualified_evidence_recall",
    "candidate_source_qualified_evidence_recall", "official_evidence_f1", "reference_evidence_recall",
    "official_text_only_evidence_f1", "candidate_official_string_evidence_recall", "actual_evidence_tokens",
    "selected_units", "candidate_count", "candidate_document_count", "empty_pack",
    "selected_zero_lexical_score_units", "candidate_zero_lexical_score_units", "seed_zero_lexical_score_units",
    "source_doc_hit_at_1_units", "source_doc_hit_at_5_units", "source_doc_hit_at_8_units")
CONFIG = {"queries": 77, "families": 24, "documents": 32, "native_units": 1850,
    "scopes": list(SCOPES), "methods": list(METHODS), "k1": 1.5, "b": .75,
    "tokenization": "existing lexical_tokens: Unicode re.findall(r'\\w+', text.casefold())",
    "lexical_text": "complete frozen canonical Unit.text; text itself is not modified",
    "query_terms": "sorted unique lexical tokens; each term contributes once",
    "idf": "ln(1+(N-df+0.5)/(df+0.5))",
    "term_weight": "idf * f*(k1+1)/(f+k1*(1-b+b*dl/max(avgdl,1)))",
    "statistics_scope": "one global index of all1850 native units; given_document only filters ranking",
    "stopwords": False, "stemming": False, "query_expansion": False, "anchor_boost": False,
    "query_instruction": None, "positive_score_threshold": None,
    "zero_score_policy": "retain zero-score units; break ties by frozen global source row, including empty/all-OOV queries",
    "seed_units": 8, "candidate_cap": 16,
    "neighbors": "all top8 seeds first, then immediate same-document left/right in seed order until cap16",
    "candidate_ranking": "own method score descending, then frozen global row",
    "packing": "greedy whole units in candidate rank order; max3 and actual rendered BGE tokens<=1024",
    "deduplication": "(doc_id,native_text), first ranked fitting location",
    "renderer": "unchanged [unit_id] header and full canonical text; final global source order",
    "dense_representation": "verified cached CLS/FP32 L2 vectors; CPU FP32 exhaustive inner product",
    "cpu_threads": 1, "max_seconds": 600,
    "comparison": "BM25 minus dense within each scope; retain both scopes separately",
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "bootstrap": "whole-family multinomial resampling with PCG64, shared draws, linear percentile95",
    "weightings": ["question_weighted", "family_balanced"], "tie_tolerance": 1e-12,
    "multiple_comparison_adjustment": "none", "p_values": False, "gold_used_for_specification_or_ranking": False}
SOURCES = [
    {"url": "https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/search/similarities/BM25Similarity.html",
     "role": "Primary implementation documentation for positive IDF, length normalization and defaults; its k1=1.2 is not our inherited1.5."},
    {"url": "https://apache.googlesource.com/lucene-solr/+/7ada4032180b516548fc0263f42da6a7a917f92b/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java",
     "role": "Pinned Apache source for IDF/TF; no claim of Lucene analyzer, quantized lengths or bitwise scores. Constant(k1+1) retained from existing research BM25."}]
LIMITS = [
    "Exposed validation development diagnostic; no independent confirmation, tuning, SOTA or significance claim.",
    "This changes candidate retrieval as well as final ranking; it is not fixed-candidate reranking.",
    "Given-document uses source knowledge; corpus32 is query-only stress testing, not standard full-corpus Qasper evaluation.",
    "Inherited k1=1.5 differs from Lucene default1.2; no parameter was selected by gold or new outcomes.",
    "Unicode word runs retain stopwords and do not stem; scientific punctuation/formulas and inflection may be poorly represented.",
    "Length normalization can favor short headings; zero-score source-order fallback is explicitly retained and diagnosed.",
    "IDF treats native units, not papers, as BM25 documents; repeated headings/paragraphs contribute to global statistics.",
    "Source-qualified packing differs from earlier A text-only packing; dense is recomputed under the same new policy.",
    "Exact source-qualified evidence matching is diagnostic; official-string compatibility metrics are also retained.",
    "BGE evidence tokens exclude generator question/instructions; original headers may repeat across papers, so packs are not a generation interface.",
    "Cached dense vectors are reused; index/scoring/packing costs are not cold-start model-to-answer latency.",
    "No API, model inference, training, relation labels, GPU, answer generation or official test QA is used.",
    "Bootstrap intervals describe one fixed run and do not account for new corpora, parameter choices or independent model reruns."]
PLAN_FILES = ("plan.json", "lexical_index.json", "plan_seal.json")
RUN_FILES = ("per_question.jsonl", "public_aggregate.json", "summary.json")


def environment():
    return {"python": platform.python_version(), "unicode": unicodedata.unidata_version,
        **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "tokenizers")}}


def code_hashes():
    names = (Path(__file__).name, "run_qasper_corpus_bridge.py", "prepare_qasper_extended_development.py",
        "prepare_qasper_relation_pilot.py", "run_qasper_dense_baseline.py", "run_qasper_evidence_baselines.py",
        "qasper_metrics.py", "qasper_alignment_v2.py")
    return {str(Path(__file__).with_name(name).resolve()): digest(Path(__file__).with_name(name)) for name in names}


def load_inputs(prepared, dense, *, deadline=math.inf):
    source = bridge.load_verified_inputs(prepared, dense, deadline=deadline)
    for path, value in code_hashes().items():
        if path in source["input_sha256"] and source["input_sha256"][path] != value:
            raise ValueError("source code binding conflict")
        source["input_sha256"][path] = value
    return source


def identity(source):
    return {"queries": [{name: q[name] for name in ("family_id", "doc_id", "question_id", "query")}
                        for q in source["prepared"]["queries"]],
        "units": [{"doc_id": key[0], "unit_id": key[1], "text_sha256": hashlib.sha256(unit.text.encode()).hexdigest()}
                  for key, unit in zip(source["keys"], source["units"], strict=True)]}


def build_index(texts):
    terms = [dict(sorted(Counter(lexical_tokens(text)).items())) for text in texts]
    if not terms:
        raise ValueError("cannot index an empty corpus")
    lengths = [sum(row.values()) for row in terms]
    frequencies = Counter(term for row in terms for term in row)
    return {"N": len(terms), "term_frequencies": terms, "lengths": lengths,
        "df": dict(sorted(frequencies.items())), "avgdl": sum(lengths) / len(lengths),
        "lexical_tokens": sum(lengths), "vocabulary_size": len(frequencies),
        "zero_lexical_units": sum(value == 0 for value in lengths)}


def bm25_scores(question, index):
    """One global score array. This function has no document filter or gold."""
    query = sorted(set(lexical_tokens(question)))
    scores = [0.] * index["N"]
    k1, b = CONFIG["k1"], CONFIG["b"]
    for term in query:
        df = index["df"].get(term, 0)
        if not df:
            continue
        idf = math.log(1 + (index["N"] - df + .5) / (df + .5))
        for row, counts in enumerate(index["term_frequencies"]):
            freq = counts.get(term, 0)
            if freq:
                scores[row] += idf * freq * (k1 + 1) / (
                    freq + k1 * (1 - b + b * index["lengths"][row] / max(index["avgdl"], 1)))
    return scores


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("lexical preparation already exists")
    torch.set_num_threads(1)
    source = load_inputs(args.prepared, args.dense)
    index = build_index([unit.text for unit in source["units"]])
    pilot.verify_hashes(source["input_sha256"])
    plan = {"schema": SCHEMA, "status": "prepared_before_lexical_scoring", "config": CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
        "prepared": str(Path(args.prepared).resolve()), "dense": str(Path(args.dense).resolve()),
        "input_sha256": source["input_sha256"], "input_binding_sha256": pilot.stable_hash(source["input_sha256"]),
        "identity": identity(source), "index_sha256": pilot.stable_hash(index), "sources": SOURCES, "limits": LIMITS,
        "api_calls": 0, "model_inference_performed": False, "gpu_used": False,
        "source_only_audit": {name: index[name] for name in ("N", "avgdl", "lexical_tokens", "vocabulary_size", "zero_lexical_units")}}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "plan.json", plan)
    pilot.write_json(output / "lexical_index.json", index)
    pilot.write_json(output / "plan_seal.json", {name: digest(output / name) for name in PLAN_FILES[:-1]})
    return {"status": plan["status"], "source_only_audit": plan["source_only_audit"],
        "plan_sha256": digest(output / "plan.json"), "api_calls": 0, "gpu_used": False}


def load_plan(directory):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != set(PLAN_FILES):
        raise ValueError("plan inventory differs")
    buffers = {name: (directory / name).read_bytes() for name in PLAN_FILES}
    hashes = {str(directory / name): hashlib.sha256(value).hexdigest() for name, value in buffers.items()}
    seal, plan, index = (json.loads(buffers[name]) for name in ("plan_seal.json", "plan.json", "lexical_index.json"))
    if seal != {name: hashes[str(directory / name)] for name in PLAN_FILES[:-1]}:
        raise ValueError("plan bytes differ from seal")
    fixed = {"schema": SCHEMA, "status": "prepared_before_lexical_scoring", "config": CONFIG,
        "environment": environment(), "sources": SOURCES, "limits": LIMITS,
        "api_calls": 0, "model_inference_performed": False, "gpu_used": False}
    if set(plan) != set(fixed) | {"created_at_utc", "prepared", "dense", "input_sha256", "input_binding_sha256",
            "identity", "index_sha256", "source_only_audit"} or any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError("fixed lexical plan contract differs")
    if plan["index_sha256"] != pilot.stable_hash(index) or plan["input_binding_sha256"] != pilot.stable_hash(plan["input_sha256"]):
        raise ValueError("lexical index/input binding differs")
    pilot.verify_hashes(plan["input_sha256"])
    return plan, index, hashes


def reloaded(plan, index, *, deadline=math.inf):
    source = load_inputs(plan["prepared"], plan["dense"], deadline=deadline)
    if source["input_sha256"] != plan["input_sha256"] or identity(source) != plan["identity"]:
        raise ValueError("reloaded source identity/binding differs")
    if build_index([unit.text for unit in source["units"]]) != index:
        raise ValueError("global lexical index does not reproduce complete source text")
    audit = {name: index[name] for name in ("N", "avgdl", "lexical_tokens", "vocabulary_size", "zero_lexical_units")}
    if plan["source_only_audit"] != audit:
        raise ValueError("source-only preparation aggregate differs")
    return source


def evaluate_query(q, source, lexical, dense, index, *, deadline=math.inf):
    units, keys = source["units"], source["keys"]
    if len(lexical) != len(units) or len(dense) != len(units):
        raise ValueError("score array does not cover every native unit")
    terms = set(lexical_tokens(q["query"]))
    oov = terms - set(index["df"])
    query_info = {"query_unique_terms": len(terms), "query_oov_unique_terms": len(oov),
        "query_empty_lexical": int(not terms), "query_nonempty_all_oov": int(bool(terms) and len(oov) == len(terms)),
        "query_oov_unique_fraction": len(oov) / len(terms) if terms else 0.}
    counter = PackCounter(source["tokenizer"], units, deadline)
    selections = []
    for scope in SCOPES:
        allowed = source["positions"][q["doc_id"]] if scope == "given_document" else None
        scope_indices = list(range(len(units))) if allowed is None else allowed
        for method, scores in (("dense_cached", dense), ("bm25", lexical)):
            started = time.perf_counter()
            ranking = bridge.rank_scores(scores, allowed)
            seeds, candidates = bridge.expand_corpus_candidates(ranking, keys, source["positions"])
            candidate_set = set(candidates)
            ranked_candidates = [i for i in ranking if i in candidate_set]
            retrieval_seconds = time.perf_counter() - started
            started = time.perf_counter()
            selected = bridge.pack_source_qualified(units, keys, ranked_candidates, 1024, counter, max_units=3)
            selections.append((scope, method, ranking, seeds, ranked_candidates, selected, retrieval_seconds,
                time.perf_counter() - started, sum(lexical[i] > 0 for i in scope_indices)))
    # Gold cannot affect lexical statistics, score computation, retrieval, or packing.
    gold = source["qa_by_key"][(q["doc_id"], q["question_id"])]["answer_annotations"]
    pairs = lambda ids: [(keys[i][0], units[i].native_text) for i in ids]
    records = []
    for scope, method, ranking, seeds, candidates, selected, retrieval_seconds, packing_seconds, positive in selections:
        qualified = bridge.source_qualified_metrics(pairs(selected), q["doc_id"], gold)
        candidate_metrics = bridge.source_qualified_metrics(pairs(candidates), q["doc_id"], gold)
        records.append({**{name: q[name] for name in ("family_id", "doc_id", "question_id")},
            "scope": scope, "method": method, **query_info,
            **score_selection(units, selected, gold, counter, 1024),
            "source_qualified_evidence_f1": qualified["evidence_f1"],
            "source_qualified_evidence_recall": qualified["evidence_recall"],
            "candidate_source_qualified_evidence_recall": candidate_metrics["evidence_recall"],
            "candidate_official_string_evidence_recall": evidence_metrics([units[i].native_text for i in candidates], gold)["evidence_recall"],
            "scope_positive_lexical_units": positive, "scope_no_positive_lexical_score": int(positive == 0),
            "seed_zero_lexical_score_units": sum(lexical[i] == 0 for i in seeds),
            "candidate_zero_lexical_score_units": sum(lexical[i] == 0 for i in candidates),
            "selected_zero_lexical_score_units": sum(lexical[i] == 0 for i in selected),
            **{f"source_doc_hit_at_{k}_units": int(any(keys[i][0] == q["doc_id"] for i in ranking[:k])) for k in (1, 5, 8)},
            "candidate_count": len(candidates), "candidate_document_count": len({keys[i][0] for i in candidates}),
            "empty_pack": int(not selected), "duplicate_rendered_headers_in_pack": len(selected)-len({units[i].unit_id for i in selected}),
            **bridge.duplicate_diagnostics(units, keys, candidates, selected, q["doc_id"]),
            "seed_global_indices": seeds, "candidate_global_indices": candidates, "selected_global_indices": selected,
            "candidate_scores": [dense[i] if method == "dense_cached" else lexical[i] for i in candidates],
            "retrieval_wall_seconds": retrieval_seconds, "packing_wall_seconds": packing_seconds})
    return records


def evaluate_all(source, index, deadline):
    records, times = [], {"dense_score_wall_seconds": 0., "lexical_score_wall_seconds": 0.}
    for q in source["prepared"]["queries"]:
        check_time(deadline)
        key = (q["doc_id"], q["question_id"])
        started = time.perf_counter()
        dense = (source["candidate_vectors"] @ source["query_vectors"][source["query_positions"][key]]).tolist()
        times["dense_score_wall_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        lexical = bm25_scores(q["query"], index)
        times["lexical_score_wall_seconds"] += time.perf_counter() - started
        bridge.validate_within_document_replay({"queries": [q]}, source["documents"], source["units"], source["keys"],
            source["positions"], {key: dense}, source["rankings"], source["tokenizer"], deadline)
        records.extend(evaluate_query(q, source, lexical, dense, index, deadline=deadline))
    return records, times


def summarize(records, queries):
    identities = sorted((q["family_id"], q["doc_id"], q["question_id"]) for q in queries)
    expected = {(scope, method, *key) for scope in SCOPES for method in METHODS for key in identities}
    indexed = {(r["scope"], r["method"], r["family_id"], r["doc_id"], r["question_id"]): r for r in records}
    if not identities or len(set(identities)) != len(identities) or len(records) != len(expected) or set(indexed) != expected:
        raise ValueError("requires complete unique four-arm question denominator")
    families = sorted({key[0] for key in identities})
    groups = [np.array([i for i, key in enumerate(identities) if key[0] == f]) for f in families]
    draws = np.random.Generator(np.random.PCG64(CONFIG["bootstrap_seed"])).multinomial(
        len(groups), np.full(len(groups), 1/len(groups)), size=CONFIG["bootstrap_replicates"])
    sizes = np.array([len(g) for g in groups])
    def estimate(values):
        return {"question_weighted": float(values.mean()), "family_balanced": float(np.mean([values[g].mean() for g in groups]))}
    def comparison(values):
        sums = np.array([values[g].sum() for g in groups])
        return {**estimate(values),
            "question_weighted_percentile95": np.quantile((draws@sums)/(draws@sizes), [.025,.975], method="linear").tolist(),
            "family_balanced_percentile95": np.quantile((draws@(sums/sizes))/len(groups), [.025,.975], method="linear").tolist(),
            "positive_questions": int((values > 1e-12).sum()), "negative_questions": int((values < -1e-12).sum()),
            "tied_questions": int((np.abs(values) <= 1e-12).sum())}
    reports = []
    for scope in SCOPES:
        tables, methods = {}, []
        for method in METHODS:
            rows = [indexed[(scope, method, *key)] for key in identities]
            tables[method] = {field: np.array([r[field] for r in rows], dtype=float) for field in METRICS}
            if any(not np.isfinite(v).all() for v in tables[method].values()) or any(
                    not 0 <= r["actual_evidence_tokens"] <= 1024 or not 0 <= r["selected_units"] <= 3 for r in rows):
                raise ValueError("nonfinite metric or evidence budget violation")
            methods.append({"method": method, "metrics": {field: estimate(v) for field, v in tables[method].items()},
                "timing": {field: {"total": sum(r[field] for r in rows), "median": statistics.median(r[field] for r in rows)}
                    for field in ("retrieval_wall_seconds", "packing_wall_seconds")}})
        lexical_rows = [indexed[(scope, "bm25", *key)] for key in identities]
        reports.append({"scope": scope, "questions": len(identities), "families": len(groups), "methods": methods,
            "comparison": {"plus": "bm25", "minus": "dense_cached",
                "metrics": {field: comparison(tables["bm25"][field]-tables["dense_cached"][field]) for field in METRICS},
                "selected_set_changed_questions": sum(set(indexed[(scope,"bm25",*key)]["selected_global_indices"])
                    != set(indexed[(scope,"dense_cached",*key)]["selected_global_indices"]) for key in identities),
                "sign_interpretation": "positive quality delta is a win; positive tokens/counts mean larger, not better"},
            "lexical_query_diagnostics": {"empty_lexical_queries": sum(r["query_empty_lexical"] for r in lexical_rows),
                "nonempty_all_oov_queries": sum(r["query_nonempty_all_oov"] for r in lexical_rows),
                "queries_without_positive_score_in_scope": sum(r["scope_no_positive_lexical_score"] for r in lexical_rows),
                "mean_unique_query_oov_fraction": statistics.mean(r["query_oov_unique_fraction"] for r in lexical_rows)},
            "rendered_header_collision_occurrences": sum(r["duplicate_rendered_headers_in_pack"] for r in lexical_rows)})
    return {"scopes": reports, "shared_bootstrap_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}


def public_result(plan, source, records, execution):
    return {"schema": SCHEMA, "status": "completed", "config": CONFIG, "environment": environment(),
        "question_count": len(source["prepared"]["queries"]), "family_count": len({q["family_id"] for q in source["prepared"]["queries"]}),
        "corpus_units": len(source["units"]), "corpus_documents": len(source["documents"]),
        "source_only_audit": plan["source_only_audit"], "input_binding_sha256": plan["input_binding_sha256"],
        "all_packs_within_budget": True, "api_calls": 0, "paid_api_cost_usd": "0", "gpu_used": False,
        "model_inference_performed": False, "test_payload_read": False, "independent_confirmation": False,
        "source_hashes_unchanged": True, "execution": execution, "sources": SOURCES, "limits": LIMITS,
        **summarize(records, source["prepared"]["queries"])}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("lexical run output exists")
    started = time.monotonic(); deadline = started + CONFIG["max_seconds"]
    wall_started = time.perf_counter()
    torch.set_num_threads(1)
    plan, index, plan_hashes = load_plan(args.plan)
    source = reloaded(plan, index, deadline=deadline)
    pilot.verify_hashes(plan_hashes)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "started", "plan_sha256": plan_hashes, "input_sha256": plan["input_sha256"]}
    try:
        records, scoring = evaluate_all(source, index, deadline)
        execution = {**scoring, "wall_seconds_before_aggregate": time.perf_counter()-wall_started, "device": "cpu", "threads": 1,
            "timing_scope": "shared global scoring per query; individual scope ranking/expansion/packing; cached dense vectors, no model load"}
        public = public_result(plan, source, records, execution)
        pilot.verify_hashes(plan["input_sha256"]); pilot.verify_hashes(plan_hashes); check_time(deadline)
        with (output / "per_question.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        pilot.write_json(output / "public_aggregate.json", public)
        report.update(status="completed", public=public,
            output_sha256={name: digest(output/name) for name in RUN_FILES[:-1]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, retry_or_fallback=False)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-wall_started
        pilot.write_json(output / "summary.json", report)
    return {"status": "completed", "records": len(records), "gpu_used": False, "api_calls": 0}


def validate_timing(report, saved):
    execution = report["public"]["execution"]
    expected = {"dense_score_wall_seconds", "lexical_score_wall_seconds", "wall_seconds_before_aggregate", "device", "threads", "timing_scope"}
    if (set(execution) != expected or execution["device"] != "cpu" or execution["threads"] != 1
            or execution["timing_scope"] != "shared global scoring per query; individual scope ranking/expansion/packing; cached dense vectors, no model load"):
        raise ValueError("execution metadata differs")
    times = [report["elapsed_seconds"], execution["wall_seconds_before_aggregate"],
             execution["dense_score_wall_seconds"], execution["lexical_score_wall_seconds"]]
    times += [row[field] for row in saved for field in ("retrieval_wall_seconds", "packing_wall_seconds")]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 600 for v in times):
        raise ValueError("execution timing outside frozen bound")
    if not (execution["dense_score_wall_seconds"]+execution["lexical_score_wall_seconds"]
            <= execution["wall_seconds_before_aggregate"] <= report["elapsed_seconds"]):
        raise ValueError("timing components inconsistent")


def audit(args):
    started = time.monotonic(); deadline = started+600
    directory = Path(args.run).resolve()
    if {p.name for p in directory.iterdir()} != set(RUN_FILES):
        raise ValueError("complete run inventory differs")
    buffers = {name: (directory/name).read_bytes() for name in RUN_FILES}
    bindings = {str(directory/name): hashlib.sha256(value).hexdigest() for name,value in buffers.items()}
    plan, index, plan_hashes = load_plan(args.plan)
    source = reloaded(plan, index, deadline=deadline)
    report, public = json.loads(buffers["summary.json"]), json.loads(buffers["public_aggregate.json"])
    if (set(report) != {"schema","status","plan_sha256","input_sha256","public","output_sha256","elapsed_seconds"}
            or report["schema"] != SCHEMA or report["status"] != "completed" or report["plan_sha256"] != plan_hashes
            or report["input_sha256"] != plan["input_sha256"] or report["public"] != public
            or report["output_sha256"] != {name: bindings[str(directory/name)] for name in RUN_FILES[:-1]}):
        raise ValueError("saved summary/metadata/seal differs")
    saved = [json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    validate_timing(report, saved)
    torch.set_num_threads(1)
    records, _ = evaluate_all(source, index, deadline)
    scrub = lambda row: {k:v for k,v in row.items() if k not in ("retrieval_wall_seconds", "packing_wall_seconds")}
    if [scrub(row) for row in saved] != [scrub(row) for row in records]:
        raise ValueError("recomputed complete records/rankings/scores differ")
    if public_result(plan, source, saved, public["execution"]) != public:
        raise ValueError("recomputed complete aggregate/comparisons differ")
    pilot.verify_hashes(plan["input_sha256"]); pilot.verify_hashes(plan_hashes); pilot.verify_hashes(bindings)
    check_time(deadline)
    return {"status":"verified", "records":len(saved), "all_rankings_metrics_comparisons_recomputed":True,
        "all_source_plan_output_hashes_unchanged":True, "gpu_used":False, "model_inference_performed":False, "api_calls":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("prepared", "dense", "output"):
        prepare_parser.add_argument("--"+name, required=True)
    run_parser = sub.add_parser("run")
    for name in ("plan", "output"):
        run_parser.add_argument("--"+name, required=True)
    audit_parser = sub.add_parser("audit")
    for name in ("plan", "run"):
        audit_parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":run,"audit":audit}[args.command](args), indent=2))


if __name__ == "__main__":
    main()
