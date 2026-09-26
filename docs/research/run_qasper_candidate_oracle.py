"""Exact fixed-candidate evidence upper bound; CPU only, never for generation.

Prepare seals sources without computing oracle scores. Run enumerates every
subset of size at most three, including empty, on all six frozen native arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import analyze_qasper_extended_results as statistics
import run_qasper_native_dual_index_v2 as native
import run_qasper_native_owner_order as owner

pilot, bridge = native.pilot, native.bridge
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "slac-qasper-fixed-candidate-oracle-v1"
SCOPES, METHODS = native.SCOPES, native.METHODS
PAIRS = (("leaf_owner", "leaf_direct"), ("dual_owner", "leaf_owner"))
METRICS = ("candidate_source_qualified_evidence_recall", "actual_source_qualified_evidence_f1",
    "actual_source_qualified_evidence_recall", "oracle_source_qualified_evidence_f1",
    "oracle_source_qualified_evidence_recall", "oracle_minus_actual_f1", "actual_evidence_tokens",
    "oracle_evidence_tokens", "oracle_selected_units")
CONFIG = {"questions": 77, "families": 24, "records": 462, "scopes": list(SCOPES), "methods": list(METHODS),
    "pairs_per_scope": [list(p) for p in PAIRS], "primary_metric": "oracle_source_qualified_evidence_f1",
    "candidate_cap": 16, "max_selected_units": 3, "budget_bge_tokens": 1024,
    "enumerated_subsets_including_empty": 311028, "unique_global_index_subsets": 150461,
    "selection": "all subsets; maximize exact Fraction F1, then independent max-reference recall, then minimize actual tokens, then sorted global indices lexicographically",
    "deduplication": "at most one selected unit per exact (document,native_text); alternative locations remain candidates",
    "rendering": "unchanged whole-native render_pack and PackCounter; no truncation or generation",
    "bootstrap_seed": statistics.BOOTSTRAP_SEED, "bootstrap_replicates": statistics.BOOTSTRAP_REPLICATES,
    "bootstrap": "shared whole-family PCG64 multinomial; two-sided linear percentile 95%",
    "multiple_comparison_adjustment": "none", "pytorch_cpu_threads": 1,
    "enumeration_execution": "serial", "numpy_blas_threads": "not independently constrained", "max_seconds": 1200,
    "gold_used_for_oracle_selection": True, "oracle_only": True, "deployable_retrieval_result": False,
    "answer_generation_allowed": False, "posthoc_development": True, "independent_confirmation": False,
    "official_test_used": False}
LIMITS = [
    "Gold-guided fixed-candidate upper bound on the same exposed 77 development questions, not a deployable selector or independent confirmation.",
    "All six arms and all questions remain, including unanswerable, empty-reference and FLOAT evidence annotations.",
    "Any empty reference gives empty prediction F1=recall=1; oracle abstention is not a measured answerability capability.",
    "Candidate recall, actual selection and budgeted oracle are distinct; empty references break a simple candidate-recall upper-bound inequality.",
    "Equal maximum caps do not imply equal actual evidence length; oracle tokens are reported separately.",
    "Cross-corpus query-only scope is a stress diagnostic and is not standard given-document Qasper.",
    "Cross-document header ambiguity remains; no oracle pack may be sent to answer generation.",
    "The old 104-question subset-oracle result is not a matched denominator or comparator here.",
    "No JEV, Boundary Refiner, new embeddings, API calls, training or GPU inference are introduced.",
    "Enumeration is serial and PyTorch uses one CPU thread; NumPy/BLAS thread count is not independently constrained.",
]
RUN_FILES = ("per_question.jsonl", "public_aggregate.json", "summary.json")


def subset_inventory(rows):
    groups, unique = Counter(), set()
    for row in rows:
        candidates = sorted(row["candidate_global_indices"])
        if len(candidates) > 16 or len(set(candidates)) != len(candidates) or any(type(i) is not int or i < 0 for i in candidates):
            raise ValueError("invalid frozen candidate indices")
        for size in range(min(3, len(candidates)) + 1):
            for selected in itertools.combinations(candidates, size):
                groups[(row["scope"], row["method"])] += 1
                unique.add(selected)
    return {"enumerated_subsets_including_empty": sum(groups.values()), "unique_global_index_subsets": len(unique),
        "groups": [{"scope": scope, "method": method, "subsets": groups[scope, method]} for scope in SCOPES for method in METHODS]}


def source_snapshot(native_plan, native_run, native_audit):
    plan, _, plan_hashes = native.load_plan(native_plan)
    buffers, run_hashes = owner.snapshot(native_run, owner.SOURCE_FILES)
    report, public = json.loads(buffers["summary.json"]), json.loads(buffers["public_aggregate.json"])
    if (report["status"] != "completed" or report["schema"] != native.SCHEMA or report["config"] != native.CONFIG
            or report["plan_sha256"] != plan_hashes or report["input_sha256"] != plan["input_sha256"] or report["public"] != public
            or report["output_sha256"] != {name: hashlib.sha256(buffers[name]).hexdigest() for name in owner.SOURCE_FILES if name != "summary.json"}):
        raise ValueError("native source metadata or output seal differs")
    audit_path = Path(native_audit).resolve()
    audit_bytes = audit_path.read_bytes(); audit = json.loads(audit_bytes)
    for key, value in {"status": "verified", "records": 462, "configurations": 6,
            "saved_tensor_hash_index_norm_verified": True, "selection_metrics_and_comparisons_recomputed": True,
            "gpu_used": False, "model_loaded": False, "api_calls": 0}.items():
        if type(audit.get(key)) is not type(value) or audit.get(key) != value:
            raise ValueError("complete native CPU audit receipt required")
    rows = [json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    owner.validate_source_rows(rows, plan)
    inventory = subset_inventory(rows)
    if any(inventory[key] != CONFIG[key] for key in ("enumerated_subsets_including_empty", "unique_global_index_subsets")):
        raise ValueError("fixed complete candidate subset inventory differs")
    files = [Path(__file__).resolve(), ROOT / "tests/research/test_qasper_candidate_oracle.py",
        ROOT / "docs/research/CANDIDATE_ORACLE_PROTOCOL_20260927.md", Path(owner.__file__).resolve(), Path(statistics.__file__).resolve()]
    bindings = native.merge_hashes(plan["input_sha256"], plan_hashes, run_hashes,
        {str(audit_path): hashlib.sha256(audit_bytes).hexdigest()}, {str(p): native.digest(p) for p in files})
    pilot.verify_hashes(bindings)
    return plan, rows, inventory, bindings


def prepare(args):
    output, run_output = Path(args.output).resolve(), Path(args.run_output).resolve()
    source_paths = {name: str(Path(getattr(args, name)).resolve()) for name in ("native_plan", "native_run", "native_audit")}
    if output.exists() or run_output.exists(): raise FileExistsError("oracle plan and fixed run output must be unused")
    if output.is_relative_to(run_output) or run_output.is_relative_to(output) or any(
            new.is_relative_to(Path(old)) or Path(old).is_relative_to(new) for new in (output, run_output) for old in source_paths.values()):
        raise ValueError("oracle outputs overlap source or each other")
    _, _, inventory, bindings = source_snapshot(**source_paths)
    plan = {"schema": SCHEMA, "status": "prepared_not_oracle_scored", "config": CONFIG, "limits": LIMITS,
        **source_paths, "run_output": str(run_output), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": bindings, "input_binding_sha256": pilot.stable_hash(bindings), "subset_inventory": inventory,
        "api_calls": 0, "oracle_scores_computed": False, "gold_read_for_new_oracle": False}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "plan.json", plan)
    pilot.write_json(output / "plan_seal.json", {"plan.json": native.digest(output / "plan.json")})
    return {"status": plan["status"], "plan_sha256": native.digest(output / "plan.json"), "oracle_scores_computed": False}


def load_plan(directory):
    buffers, hashes = owner.snapshot(directory, ("plan.json", "plan_seal.json"))
    if json.loads(buffers["plan_seal.json"]) != {"plan.json": hashlib.sha256(buffers["plan.json"]).hexdigest()}:
        raise ValueError("oracle plan seal differs")
    plan = json.loads(buffers["plan.json"])
    fixed = {"schema": SCHEMA, "status": "prepared_not_oracle_scored", "config": CONFIG, "limits": LIMITS,
        "api_calls": 0, "oracle_scores_computed": False, "gold_read_for_new_oracle": False}
    if (set(plan) != set(fixed) | {"native_plan", "native_run", "native_audit", "run_output", "created_at_utc", "input_sha256", "input_binding_sha256", "subset_inventory"}
            or any(plan.get(k) != v for k,v in fixed.items()) or plan["input_binding_sha256"] != pilot.stable_hash(plan["input_sha256"])):
        raise ValueError("oracle frozen contract differs")
    pilot.verify_hashes(plan["input_sha256"])
    return plan, hashes


def references(annotations, source_doc):
    return [[(source_doc, text) for text in ref["evidence"]] for ref in bridge.references_from_annotations(annotations)]


def fraction_metrics(predicted, refs):
    """Exact official list-length denominators and independent maxima."""
    if not refs: raise ValueError("references cannot be omitted")
    predictions = set(predicted)
    f1s, recalls = [], []
    for ref in refs:
        common = len(predictions.intersection(ref))
        f1s.append(Fraction(1) if not predicted and not ref else Fraction(2*common, len(predicted)+len(ref)))
        recalls.append(Fraction(int(not predicted)) if not ref else Fraction(common, len(ref)))
    return max(f1s), max(recalls)


def legal_identity(selected, units, keys):
    return len({(keys[i][0], units[i].native_text) for i in selected}) == len(selected)


def exact_oracle(candidates, units, keys, refs, count, deadline=math.inf):
    candidates = sorted(candidates)
    if len(candidates) > 16 or len(candidates) != len(set(candidates)) or any(type(i) is not int or not 0 <= i < len(units) for i in candidates):
        raise ValueError("oracle candidates must be distinct valid indices")
    best_key, best = None, None
    stats = {"enumerated_subsets": 0, "duplicate_rejected_subsets": 0, "overbudget_subsets": 0, "feasible_subsets": 0}
    for size in range(min(3, len(candidates)) + 1):
        for selected in itertools.combinations(candidates, size):
            native.check_time(deadline)
            stats["enumerated_subsets"] += 1
            if not legal_identity(selected, units, keys):
                stats["duplicate_rejected_subsets"] += 1; continue
            tokens = count(selected)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("actual whole-pack token count must be a nonnegative integer")
            if tokens > 1024:
                stats["overbudget_subsets"] += 1; continue
            stats["feasible_subsets"] += 1
            f1, recall = fraction_metrics([(keys[i][0], units[i].native_text) for i in selected], refs)
            key = (-f1, -recall, tokens, selected)
            if best_key is None or key < best_key:
                best_key, best = key, (selected, f1, recall, tokens)
    if best is None: raise ValueError("empty subset must always be feasible")
    return best, stats


def verified_source(plan):
    original, rows, inventory, bindings = source_snapshot(plan["native_plan"], plan["native_run"], plan["native_audit"])
    if bindings != plan["input_sha256"] or inventory != plan["subset_inventory"]:
        raise ValueError("oracle source binding/candidate inventory changed")
    proof = native.audit(SimpleNamespace(plan=plan["native_plan"], run=plan["native_run"]))
    if proof["status"] != "verified" or proof["records"] != 462:
        raise ValueError("original native source replay failed")
    source = native.load_inputs(original["prepared"], original["dense"], original["chunks"])
    if source["input_sha256"] != original["input_sha256"]: raise ValueError("reloaded original inputs differ")
    return source, rows


def evaluate(source, actual_rows, deadline=math.inf):
    units, keys = source["units"], source["keys"]
    count = native.PackCounter(source["tokenizer"], units, deadline)
    # Validate all actual witnesses before using gold to choose any oracle pack.
    for row in actual_rows:
        selected = row["selected_global_indices"]
        if (any(type(i) is not int or not 0 <= i < len(units) for i in selected)
                or len(selected) != len(set(selected)) or len(selected) > 3 or not set(selected) <= set(row["candidate_global_indices"])
                or not legal_identity(selected, units, keys) or count(selected) > 1024
                or count(selected) != row["actual_evidence_tokens"]
                or hashlib.sha256(bridge.render_pack(units, selected).encode()).hexdigest() != row["pack_sha256"]):
            raise ValueError("actual selection is not an exact feasible oracle witness")
    records = []
    totals = Counter()
    for row in actual_rows:
        annotations = source["qa_by_key"][(row["doc_id"], row["question_id"])]["answer_annotations"]
        refs = references(annotations, row["doc_id"])
        result, stats = exact_oracle(row["candidate_global_indices"], units, keys, refs, count, deadline)
        selected, f1, recall, tokens = result
        actual = [(keys[i][0], units[i].native_text) for i in row["selected_global_indices"]]
        actual_f1, actual_recall = fraction_metrics(actual, refs)
        checked = bridge.source_qualified_metrics([(keys[i][0], units[i].native_text) for i in selected], row["doc_id"], annotations)
        candidate = bridge.source_qualified_metrics([(keys[i][0], units[i].native_text) for i in row["candidate_global_indices"]], row["doc_id"], annotations)
        if (f1 < actual_f1 or abs(float(actual_f1)-row["source_qualified_evidence_f1"]) > 1e-12
                or abs(float(actual_recall)-row["source_qualified_evidence_recall"]) > 1e-12
                or abs(float(f1)-checked["evidence_f1"]) > 1e-12 or abs(float(recall)-checked["evidence_recall"]) > 1e-12
                or candidate["evidence_recall"] != row["candidate_source_qualified_evidence_recall"]):
            raise ValueError("exact oracle/actual/reference scoring disagrees with frozen metric semantics")
        record = {k: row[k] for k in (*owner.IDENTITY, "scope", "method")}
        record.update(candidate_source_qualified_evidence_recall=candidate["evidence_recall"],
            actual_source_qualified_evidence_f1=float(actual_f1), actual_source_qualified_evidence_recall=float(actual_recall),
            oracle_source_qualified_evidence_f1=float(f1), oracle_source_qualified_evidence_recall=float(recall),
            oracle_minus_actual_f1=float(f1-actual_f1), actual_evidence_tokens=row["actual_evidence_tokens"],
            oracle_evidence_tokens=tokens, oracle_selected_units=len(selected), oracle_empty_pack=int(not selected),
            reference_count=len(refs), empty_reference_count=sum(not ref for ref in refs), any_empty_reference=int(any(not ref for ref in refs)),
            oracle_f1_fraction=[f1.numerator,f1.denominator], oracle_recall_fraction=[recall.numerator,recall.denominator],
            candidate_global_indices=row["candidate_global_indices"], actual_selected_global_indices=row["selected_global_indices"],
            oracle_selected_global_indices=list(selected), oracle_pack_sha256=hashlib.sha256(bridge.render_pack(units,selected).encode()).hexdigest(), **stats)
        records.append(record); totals.update(stats)
    if totals["enumerated_subsets"] != CONFIG["enumerated_subsets_including_empty"]:
        raise ValueError("not all frozen candidate subsets were evaluated")
    if sum(totals[k] for k in ("duplicate_rejected_subsets","overbudget_subsets","feasible_subsets")) != totals["enumerated_subsets"]:
        raise ValueError("subset feasibility partition does not reconcile")
    if len(count.cache) > CONFIG["unique_global_index_subsets"]:
        raise ValueError("token cache exceeded the complete global subset inventory")
    return records, {**dict(totals), "token_cache_entries": len(count.cache)}


def summarize(records, queries):
    questions = statistics.questions_from({"queries":queries})
    expected = {(scope, method, *q) for scope in SCOPES for method in METHODS for q in questions}
    indexed = {owner.row_key(r):r for r in records}
    if len(records) != len(expected) or set(indexed) != expected or len(questions) != 77 or len({q[0] for q in questions}) != 24:
        raise ValueError("oracle aggregation requires all six arms and the fixed complete denominator")
    groups, draws = statistics.family_resamples(questions)
    means, comparisons = [], []
    for scope in SCOPES:
        tables = {}
        for method in METHODS:
            rows = [indexed[(scope,method,*q)] for q in questions]
            table = {k:np.asarray([r[k] for r in rows],dtype=np.float64) for k in METRICS}
            if any(not np.isfinite(v).all() for v in table.values()): raise ValueError("nonfinite oracle aggregate")
            means.append({"scope":scope,"method":method,"questions":77,"families":24,
                "metrics":{k:{"question_weighted":float(v.mean()),"family_balanced":float(np.mean([v[g].mean() for g in groups]))} for k,v in table.items()},
                "oracle_empty_questions":sum(r["oracle_empty_pack"] for r in rows),
                "any_empty_reference_questions":sum(r["any_empty_reference"] for r in rows),
                "actual_witness_dominated_or_tied_questions":sum(r["oracle_minus_actual_f1"] >= 0 for r in rows)})
            tables[method] = table
        metric = CONFIG["primary_metric"]
        for plus,minus in PAIRS:
            comparisons.append({"scope":scope,"plus":plus,"minus":minus,"metric":metric,
                **statistics.clustered_delta(tables[plus][metric]-tables[minus][metric],groups,draws,metric)})
    return {"methods":means,"comparisons":comparisons,"shared_bootstrap_draws_sha256":hashlib.sha256(draws.tobytes()).hexdigest()}


def public_result(plan, rows, source, stats):
    return {"schema":SCHEMA,"status":"completed","config":CONFIG,"limits":LIMITS,
        "question_count":77,"family_count":24,"record_count":len(rows),"input_binding_sha256":plan["input_binding_sha256"],
        "api_calls":0,"gpu_used":False,"model_inference_performed":False,"answer_generation_performed":False,
        "test_payload_read":False,"raw_text_or_question_ids_in_public_output":False,
        "all_actual_witnesses_validated":True,"all_subsets_enumerated":True,"subset_accounting":stats,
        **summarize(rows,source["prepared"]["queries"])}


def run(args):
    started=time.perf_counter(); deadline=time.monotonic()+CONFIG["max_seconds"]
    plan, hashes=load_plan(args.plan); output=Path(plan["run_output"])
    if output.exists(): raise FileExistsError("fixed oracle run exists; no overwrite or renamed rerun")
    output.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(1)
        source, actual=verified_source(plan)
        records,stats=evaluate(source,actual,deadline)
        public=public_result(plan,records,source,stats)
        pilot.verify_hashes(plan["input_sha256"]); pilot.verify_hashes(hashes); native.check_time(deadline)
        with (output/"per_question.jsonl").open("x",encoding="utf-8") as stream:
            for row in records: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        pilot.write_json(output/"public_aggregate.json",public)
        output_hashes={name:native.digest(output/name) for name in RUN_FILES[:-1]}
        native.check_time(deadline)
        pilot.write_json(output/"summary.json",{"schema":SCHEMA,"status":"completed","plan_sha256":hashes,
            "input_sha256":plan["input_sha256"],"public":public,
            "output_sha256":output_hashes,
            "elapsed_seconds":time.perf_counter()-started})
        return {"status":"completed","records":len(records),"oracle_only":True,"api_calls":0}
    except BaseException as exc:
        pilot.write_json(output/"failure.json",{"schema":SCHEMA,"status":"failed","error_class":type(exc).__name__,
            "elapsed_seconds":time.perf_counter()-started,"complete_oracle_available":False,"api_calls":0,"automatic_retries":0})
        raise


def audit(args):
    deadline=time.monotonic()+CONFIG["max_seconds"]
    plan,hashes=load_plan(args.plan)
    if Path(args.run).resolve()!=Path(plan["run_output"]): raise ValueError("oracle audit run differs from fixed output")
    buffers,output_hashes=owner.snapshot(args.run,RUN_FILES)
    report,public=json.loads(buffers["summary.json"]),json.loads(buffers["public_aggregate.json"])
    if (set(report)!={"schema","status","plan_sha256","input_sha256","public","output_sha256","elapsed_seconds"}
            or report["schema"]!=SCHEMA or report["status"]!="completed" or report["plan_sha256"]!=hashes
            or report["input_sha256"]!=plan["input_sha256"] or report["public"]!=public
            or report["output_sha256"]!={name:hashlib.sha256(buffers[name]).hexdigest() for name in RUN_FILES[:-1]}
            or type(report["elapsed_seconds"]) not in (int,float) or not math.isfinite(report["elapsed_seconds"])
            or not 0<=report["elapsed_seconds"]<=CONFIG["max_seconds"]):
        raise ValueError("complete oracle output metadata/seals differ")
    torch.set_num_threads(1)
    source,actual=verified_source(plan)
    rows,stats=evaluate(source,actual,deadline)
    saved=[json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    if saved!=rows or public!=public_result(plan,rows,source,stats): raise ValueError("exact oracle witness/aggregate replay differs")
    pilot.verify_hashes(plan["input_sha256"]);pilot.verify_hashes(hashes);pilot.verify_hashes(output_hashes);native.check_time(deadline)
    return {"status":"verified","records":462,"all_subsets_enumerated":True,"exact_fraction_ties_replayed":True,
        "all_actual_witnesses_validated":True,"all_four_comparisons_recomputed":True,"oracle_only":True,
        "api_calls":0,"gpu_used":False,"model_inference_performed":False,"answer_generation_performed":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="command",required=True)
    freeze=commands.add_parser("prepare")
    for name in ("native-plan","native-run","native-audit","output","run-output"): freeze.add_argument("--"+name,required=True)
    execute=commands.add_parser("run");execute.add_argument("--plan",required=True)
    inspect=commands.add_parser("audit");inspect.add_argument("--plan",required=True);inspect.add_argument("--run",required=True)
    args=parser.parse_args();print(json.dumps({"prepare":prepare,"run":run,"audit":audit}[args.command](args),indent=2))
