"""Describe the completed local-answer run's physical requests, without new calls.

This is request accounting, not an end-to-end latency or cache-speed benchmark.
The already published complete-result receipt must bind the summary and ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("resource observations must be finite nonnegative numbers")
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered) - 1) * p
        lo = math.floor(index)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)
    return {"count": len(values), "sum": math.fsum(values), "mean": statistics.fmean(values),
            "min": ordered[0], "p50": percentile(.5), "p90": percentile(.9),
            "p95": percentile(.95), "p99": percentile(.99), "max": ordered[-1]}


def aggregate(ledger, responses, expected_count, expected_cost):
    rows = ledger["attempts"]
    if (ledger.get("halt_reason") is not None or ledger.get("automatic_retries") != 0
            or len(rows) != expected_count or len(responses) != expected_count
            or len({r["cache_key"] for r in rows}) != expected_count
            or [r["attempt"] for r in rows] != list(range(1, expected_count + 1))):
        raise ValueError("only a complete distinct sequential request ledger is supported")
    totals = Counter()
    costs, elapsed, prompt, completion = [], [], [], []
    finish = Counter()
    for row, response in zip(rows, responses):
        usage = response["usage"]
        if row["status"] != "completed" or row["usage"] != usage:
            raise ValueError("incomplete response or usage mismatch")
        cost = Decimal(str(usage["cost"]))
        if not cost.is_finite() or cost < 0 or cost != Decimal(row["actual_cost_usd"]):
            raise ValueError("invalid or inconsistent known cost")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage[name]) is not int or usage[name] < 0:
                raise ValueError("invalid provider token count")
            totals[name] += usage[name]
        if (usage["prompt_tokens"] != row["input_tokens"]
                or usage["completion_tokens"] != row["output_tokens"]
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("provider token counts do not reconcile")
        for source, key in (("prompt_tokens_details", "cached_tokens"),
                            ("completion_tokens_details", "reasoning_tokens")):
            value = usage.get(source, {}).get(key)
            if type(value) is not int or value < 0:
                raise ValueError("required provider detail unavailable")
            upper = usage["prompt_tokens" if key == "cached_tokens" else "completion_tokens"]
            if value > upper:
                raise ValueError("provider detail exceeds token total")
            totals[key] += value
        if len(response["choices"]) != 1:
            raise ValueError("expected exactly one completed response choice")
        finish[str(response["choices"][0]["finish_reason"])] += 1
        costs.append(cost)
        elapsed.append(row["elapsed_seconds"])
        prompt.append(row["input_tokens"])
        completion.append(row["output_tokens"])
    total_cost = sum(costs, Decimal("0"))
    if total_cost != Decimal(expected_cost) or total_cost != Decimal(ledger["actual_reported_cost_usd"]):
        raise ValueError("known cost differs from audited full result")
    return {"physical_requests": expected_count, "successful_requests": expected_count,
            "unknown_cost_attempts": 0, "automatic_retries": 0,
            "known_provider_cost_usd": str(total_cost), "provider_tokens": dict(totals),
            "request_cycle_seconds": distribution(elapsed),
            "prompt_tokens_per_request": distribution(prompt),
            "completion_tokens_per_request": distribution(completion),
            "reported_cost_usd_per_request": distribution([float(v) for v in costs]),
            "finish_reason_counts": dict(finish)}


def analyze(run, public_results, output):
    run, public_results, output = Path(run).resolve(), Path(public_results).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("resource analysis output already exists")
    public = read(public_results)
    if (public.get("schema") != "slac-qasper-local-answer-publication-v1"
            or public.get("status") != "complete_audited_development_results"
            or public.get("unique_requests") != 367 or public.get("logical_predictions") != 462):
        raise ValueError("published complete six-method result required")
    summary_path = run / "summary.json"
    summary = read(summary_path)
    if digest(summary_path) != public["publication_provenance"]["summary_sha256"]:
        raise ValueError("summary differs from published audited result")
    bindings = {str(summary_path): digest(summary_path), str(public_results): digest(public_results)}
    for relative, expected in summary["output_sha256"].items():
        path = (run / Path(relative.replace("\\", "/"))).resolve()
        if not path.is_relative_to(run) or digest(path) != expected:
            raise ValueError("audited output path or hash differs")
        bindings[str(path)] = expected
    ledger_path = run / "provider_calls/ledger.json"
    if str(ledger_path) not in bindings:
        raise ValueError("ledger is not bound by audited summary")
    ledger = read(ledger_path)
    responses = []
    for index in range(1, 368):
        path = run / f"provider_calls/response_{index:03d}.json"
        if str(path) not in bindings:
            raise ValueError("response is not bound by audited summary")
        responses.append(read(path))
    result = aggregate(ledger, responses, 367, summary["known_generation_cost_usd"])
    result.update(schema="slac-qasper-local-answer-physical-resources-v1",
        source_summary_sha256=bindings[str(summary_path)],
        source_public_results_sha256=bindings[str(public_results)],
        source_ledger_sha256=bindings[str(ledger_path)], analyzer_sha256=digest(__file__),
        api_calls_by_analysis=0, logical_predictions=462, exact_payload_reuses=95,
        latency_scope="Sequential client request cycle after durable reservation, including HTTP, response read, redaction, response serialization and validation; before final ledger save. Excludes retrieval, prompt construction, startup, and final result audit.",
        limitations=["One run, one generator/provider, no concurrency or repeated timing trials.",
                     "Physical requests can serve multiple methods; no method-level latency superiority is estimated.",
                     "Exact payload reuse is an experimental control, not measured deployment cache speedup.",
                     "Provider cached_tokens is provider-reported prompt caching, distinct from local exact-payload reuse.",
                     "The earlier JEV support timeout and its unknown cost remain outside this complete generation-only resource summary.",
                     "No end-to-end system cost, local hardware/download cost or future price claim."])
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("bound inputs changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--public-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.public_results, args.output)
    print(json.dumps({key: result[key] for key in ("physical_requests", "known_provider_cost_usd", "provider_tokens", "request_cycle_seconds")}))
