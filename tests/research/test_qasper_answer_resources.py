import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("answer_resources", Path(__file__).parents[2] / "docs/research/analyze_qasper_answer_resources.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    usage = {"cost": .1, "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
             "prompt_tokens_details": {"cached_tokens": 20},
             "completion_tokens_details": {"reasoning_tokens": 0}}
    row = {"attempt": 1, "cache_key": "a", "status": "completed", "usage": copy.deepcopy(usage),
           "actual_cost_usd": "0.1", "input_tokens": 100, "output_tokens": 10, "elapsed_seconds": 2.5}
    ledger = {"attempts": [row], "halt_reason": None, "automatic_retries": 0, "actual_reported_cost_usd": "0.1"}
    response = {"usage": usage, "choices": [{"finish_reason": "stop"}]}
    return ledger, [response]


def test_physical_cost_and_provider_cache_are_not_reuse_counts():
    ledger, responses = fixture()
    report = mod.aggregate(ledger, responses, 1, "0.1000")
    assert report["physical_requests"] == 1
    assert report["provider_tokens"]["cached_tokens"] == 20
    assert report["known_provider_cost_usd"] == "0.1"


@pytest.mark.parametrize("mutation", [
    lambda l, r: l.update(halt_reason="TimeoutError"),
    lambda l, r: l.update(automatic_retries=1),
    lambda l, r: l["attempts"][0].update(status="in_flight"),
    lambda l, r: l["attempts"][0].update(actual_cost_usd="NaN"),
    lambda l, r: l["attempts"][0].update(input_tokens=99),
    lambda l, r: l["attempts"][0].update(elapsed_seconds=float("inf")),
    lambda l, r: l["attempts"][0].update(elapsed_seconds=True),
    lambda l, r: l.update(actual_reported_cost_usd="0.2"),
    lambda l, r: r[0]["usage"].update(cost=.2),
])
def test_reject_incomplete_or_inconsistent_accounting(mutation):
    ledger, responses = fixture()
    mutation(ledger, responses)
    with pytest.raises(ValueError):
        mod.aggregate(ledger, responses, 1, "0.1")


def test_reject_repeated_payload_and_missing_request():
    ledger, responses = fixture()
    with pytest.raises(ValueError):
        mod.aggregate(ledger, responses, 2, "0.1")
    ledger["attempts"].append({**ledger["attempts"][0], "attempt": 2})
    with pytest.raises(ValueError):
        mod.aggregate(ledger, responses * 2, 2, "0.2")


@pytest.mark.parametrize("field,value", [("total_tokens", 111), ("prompt_tokens", True)])
def test_reject_even_mutually_agreeing_invalid_usage(field, value):
    ledger, responses = fixture()
    responses[0]["usage"][field] = value
    ledger["attempts"][0]["usage"][field] = value
    with pytest.raises(ValueError):
        mod.aggregate(ledger, responses, 1, "0.1")


def test_unknown_provider_cache_is_not_zero():
    ledger, responses = fixture()
    responses[0]["usage"]["prompt_tokens_details"].clear()
    ledger["attempts"][0]["usage"]["prompt_tokens_details"].clear()
    with pytest.raises(ValueError):
        mod.aggregate(ledger, responses, 1, "0.1")


def test_linear_quantiles_and_no_implicit_zero():
    assert mod.distribution([1, 3, 7, 9])["p95"] == pytest.approx(8.7)
    with pytest.raises(ValueError):
        mod.distribution([])
