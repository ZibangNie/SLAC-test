from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import openrouter_decision_client as client


@pytest.fixture(autouse=True)
def restore_profile():
    client.select_general_profile("gpt41mini")
    yield
    client.select_general_profile("gpt41mini")


def tasks():
    return [{"id": "support:one", "item": {"query": "What is stated?", "unit": {"id": "u1", "text": "A fact."}}}]


def test_profile_switch_keeps_jev_payload_exact_and_general_visible_state():
    jev_before = client.canonical_bytes(client.make_payload(tasks(), "support", "jev"))
    general_before = client.make_payload(tasks(), "support", "general")
    client.select_general_profile("qwen36plus")
    after = client.make_payload(tasks(), "support", "general")
    assert client.canonical_bytes(client.make_payload(tasks(), "support", "jev")) == jev_before
    assert after["messages"] == general_before["messages"]
    assert after["response_format"] == general_before["response_format"]
    assert after["model"] == "qwen/qwen3.6-plus"
    assert after["provider"]["only"] == ["alibaba"]
    assert after["provider"]["allow_fallbacks"] is False
    assert after["provider"]["max_price"] == {"prompt": "0.325", "completion": "1.95", "request": "0"}
    assert after["reasoning"] == {"enabled": False}
    assert "openai/gpt-4.1-mini" not in client.RESPONSE_MODELS["general"]


def test_unknown_profile_rejected_without_mutation():
    before = client.canonical_bytes(client.MODELS)
    with pytest.raises(ValueError, match="unknown"):
        client.select_general_profile("untrusted-provider")
    assert client.canonical_bytes(client.MODELS) == before


def test_profile_return_is_not_shared_mutable_state():
    returned = client.select_general_profile("qwen36plus")
    returned["reasoning"]["enabled"] = True
    assert client.MODELS["general"]["reasoning"] == {"enabled": False}


def test_json_object_profile_preserves_questions_and_jev_payload():
    strict = client.make_payload(tasks(), "support", "general")
    jev = client.canonical_bytes(client.make_payload(tasks(), "support", "jev"))
    client.select_general_profile("qwen36plus-json")
    payload = client.make_payload(tasks(), "support", "general")
    assert payload["messages"][1] == strict["messages"][1]
    assert payload["response_format"] == {"type": "json_object"}
    assert client.canonical_bytes(strict["response_format"]["json_schema"]["schema"]).decode() in payload["messages"][0]["content"]
    assert client.payload_task_ids(payload, "general") == ["support:one"]
    assert client.canonical_bytes(client.make_payload(tasks(), "support", "jev")) == jev
    response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"support:one":"yes"}'}}]}
    assert client.parse_labels(response, payload, "general", "support") == {"support:one": "yes"}


@pytest.mark.parametrize("content", ['{}', '{"other":"yes"}', '{"support:one":"dependent"}',
                                    '{"support:one":"yes","support:one":"no"}',
                                    '```json\n{"support:one":"yes"}\n```'])
def test_json_object_profile_rejects_invalid_labels_without_repair(content):
    client.select_general_profile("qwen36plus-json")
    payload = client.make_payload(tasks(), "support", "general")
    response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    with pytest.raises(ValueError):
        client.parse_labels(response, payload, "general", "support")


def test_json_object_profile_rejects_misaligned_state():
    client.select_general_profile("qwen36plus-json")
    payload = client.make_payload(tasks(), "support", "general")
    visible = json.loads(payload["messages"][1]["content"])
    visible["state"]["items"] = {}
    payload["messages"][1]["content"] = json.dumps(visible)
    with pytest.raises(ValueError, match="IDs differ"):
        client.payload_task_ids(payload, "general")
