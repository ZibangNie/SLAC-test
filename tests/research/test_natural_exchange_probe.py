"""Authored sources and fake counter/transport only; no private corpus or keys."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys

import pytest

from docs.research import run_natural_exchange_probe as runner


@pytest.fixture
def sources(monkeypatch):
    def unit(uid, order, text):
        return {"unit_id": uid, "source_order": order, "text": text,
                "native_text_sha256": runner.sha(text.encode()), "retrieval_text_sha256": runner.sha(text.encode())}
    cases = []
    for ordinal in range(1, 7):
        a, r, c = unit("a", 0, "A red switch."), unit("r", 1, "A cell X."), unit("c", 2, "A cell X at 3 volts.")
        cases.append({"ordinal": ordinal, "doc_id": "invented", "question_id": str(ordinal),
                      "query": f"What powers lamp {ordinal}?", "original_pack": [a,r],
                      "proposed_pack": [a,c], "candidate": c, "removed": r})
    packet = runner.canonical({"schema": "slac-natural-exchange-review-inputs-v1", "cases": cases})
    matrix = [["no","no","no"], ["yes","no","no"], ["no","no","no"],
              ["no","yes","no"], ["no","no","no"], ["unknown","no","no"]]
    def review(b=False):
        values = deepcopy(matrix)
        if b:
            values[4][0], values[5][1] = "unknown", "yes"
        return runner.canonical({"packet_sha256": runner.sha(packet), "reviews": [
            {"ordinal": i, "judgments": {d: {"label": label, "reason": "NEVER_SEND_REVIEW"}
             for d,label in zip(runner.DIMENSIONS, row)}} for i,row in enumerate(values,1)]})
    result = {name: b"invented source" for name in runner.SOURCE_NAMES}
    result.update({runner.PACKET: packet, runner.REVIEW_A: review(), runner.REVIEW_B: review(True), runner.SAMPLE_PLAN: b"{}"})
    monkeypatch.setattr(runner, "PINNED_INPUT_SHA256", {p: runner.sha(result[p]) for p in runner.PINNED_INPUT_SHA256})
    monkeypatch.setattr(runner, "_snapshots", lambda: result.copy())
    monkeypatch.setattr(runner, "_load_counter", lambda _: (len, {"version": "TEST_ONLY_FAKE_COUNTER"}))
    return result


def test_fixed_projection_preserves_text_identity_and_separates_renderers(sources):
    plan, wires = runner._assemble(sources)
    assert len(wires) == len({w.cache_key for w in wires}) == 6
    assert sum(len(w.expected_ids) for w in wires) == 18
    assert plan["limits"]["generation_max_tokens"] == 1024
    assert plan["limits"]["judge_max_tokens"] == 2048
    for entry, wire in zip(plan["requests"], wires):
        assert entry["surfaces"]["original"]["render_sha256"] != entry["surfaces"]["legacy_original"]["render_sha256"]
        assert wire.plan.max_units == 3 and wire.plan.max_tokens == 1024 and wire.plan.max_judge_tokens == 2048
        assert wire.payload()["state"]["current_pack"][0]["text"] == "A red switch."
        assert not any(term in wire.payload_bytes for term in (b"question_id",b"ordinal",b"NEVER_SEND",b"reviewer",b"rank"))
    categories = plan["readout_only"]["categories"]
    assert [len(categories[k]) for k in ("agreed_determinate","agreed_unknown","disagreed")] == [15,1,2]
    assert sum(r["A"] == "yes" for r in categories["agreed_determinate"]) == 2


def test_supervision_and_extra_input_metadata_cannot_change_requests(sources):
    packet = json.loads(sources[runner.PACKET])
    before = runner.freeze_requests(packet, len)[1]
    for case in packet["cases"]:
        case["ranking"] = "DO_NOT_SEND"
        case["judgments"] = {"loss": "yes"}
        case["candidate"]["annotation"] = "DO_NOT_SEND"
    after = runner.freeze_requests(packet, len)[1]
    assert [w.payload_bytes for w in before] == [w.payload_bytes for w in after]
    changed = sources.copy()
    b = json.loads(changed[runner.REVIEW_B])
    b["reviews"][0]["judgments"][runner.DIMENSIONS[0]]["label"] = "yes"
    changed[runner.REVIEW_B] = runner.canonical(b)
    assert runner.review_readout(changed) != runner.review_readout(sources)
    assert all(b"DO_NOT_SEND" not in w.payload_bytes for w in after)


@pytest.mark.parametrize("kind", ["order","text","removed","proposed","generation_budget","judge_budget"])
def test_input_drift_and_complete_render_budget_rejected(sources, kind):
    packet = json.loads(sources[runner.PACKET])
    count = len
    if kind == "order": packet["cases"].reverse()
    elif kind == "text": packet["cases"][0]["candidate"]["text"] += "tampered"
    elif kind == "removed": packet["cases"][0]["removed"]["unit_id"] = "absent"
    elif kind == "proposed": packet["cases"][0]["proposed_pack"].reverse()
    elif kind == "generation_budget": count = lambda _: 1025
    else: count = lambda text: 2049 if text.count("[invented/") == 3 else 1
    with pytest.raises(ValueError): runner.freeze_requests(packet, count)


@pytest.mark.parametrize("kind", ["source","private_packet","plan","tokenizer"])
def test_verification_drift_prevents_client_and_claim(tmp_path, monkeypatch, sources, kind):
    directory = tmp_path / "plan"
    runner.prepare_plan(directory)
    path = directory / "plan.json"
    if kind in ("source","private_packet"):
        sources[runner.PACKET if kind == "private_packet" else runner.SOURCE_NAMES[0]] += b" "
    elif kind == "plan":
        value = json.loads(path.read_bytes())
        value["requests"][0]["wire_body_sha256"] = "0"*64
        path.write_bytes(runner.canonical(value)+b"\n")
    else:
        monkeypatch.setattr(runner, "_load_counter", lambda _: (_ for _ in ()).throw(ValueError("tokenizer drift")))
    monkeypatch.setattr(runner.client,"ExchangeProbeClient",lambda *a,**k: pytest.fail("no client after drift"))
    with pytest.raises(ValueError):
        runner.execute_plan(path,tmp_path/"run",key_file="never-read",live=True)
    assert not (directory/"execution_claim.json").exists()
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("fail_at", [None,2])
def test_fake_run_complete_or_partial_is_terminal_for_plan(tmp_path, monkeypatch, sources, fail_at):
    monkeypatch.setattr(socket,"socket",lambda *a,**k: pytest.fail("network forbidden"))
    monkeypatch.setattr(runner.client.base,"read_key",lambda *a,**k: pytest.fail("key forbidden"))
    directory = tmp_path/"plan"
    runner.prepare_plan(directory)
    calls=[]
    def fake(wire, timeout):
        calls.append(wire.cache_key)
        if len(calls)==fail_at: raise TimeoutError("private error")
        return {"model":runner.client.RESPONSE_MODEL,"provider":"typesafe",
                "usage":{"cost":"0.0001","input_tokens":25,"output_tokens":3},
                "answers":{d:{"type":"choice","choice":"unknown"} for d in wire.expected_ids}}
    ledger=runner.execute_plan(directory/"plan.json",tmp_path/"run",transport=fake,key_file="never-read")
    assert len(calls)==(6 if fail_at is None else 2)
    assert ledger["status"]==("completed" if fail_at is None else "halted")
    with pytest.raises(FileExistsError):
        runner.execute_plan(directory/"plan.json",tmp_path/"another-run",transport=fake)
    assert not (tmp_path/"another-run").exists()


def test_bounded_cli_uses_fixed_script_watchdog_and_sanitized_output(tmp_path, monkeypatch, capsys):
    plan=tmp_path/"plan.json"
    plan.write_bytes(b"invented immutable plan")
    monkeypatch.setattr(runner,"_contained",lambda p: Path(p))
    captured={}
    def supervise(command, output, receipt, **kwargs):
        captured.update(command=command,output=output,receipt=receipt,**kwargs)
        return {"status":"worker_exited","returncode":0,"secret":"MUST_NOT_PRINT"}
    monkeypatch.setattr(runner,"supervise",supervise)
    monkeypatch.setattr(sys,"argv",["runner","bounded","--plan",str(plan),"--output-dir",str(tmp_path/"out"),
                                    "--receipt",str(tmp_path/"receipt.json"),"--key-file","never-read"])
    assert runner.main()==0
    assert captured["timeout_seconds"]==180
    assert captured["command"][:3]==[sys.executable,str(Path(runner.__file__).resolve()),"run"]
    assert captured["command"][-2:]==["--proxy","http://127.0.0.1:7897"]
    assert json.loads(capsys.readouterr().out)=={"command":"bounded","status":"worker_exited"}
