"""Offline wire feasibility only: no credential, transport, execution plan or scores."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket


def denied(*args, **kwargs):
    raise RuntimeError("network forbidden during JEV wire inspection")


socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied

import openrouter_decision_client as client


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "artifacts/research-foundation/offline-20261005/candidate-granularity-mechanism-01"
INPUT_SHA = "aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    raw = (PHASE / "selected_inputs.json").read_bytes()
    assert sha(raw) == INPUT_SHA
    data = json.loads(raw)
    assert len(data["cases"]) == 2 and len(data["pairs"]) == 117
    assert client.MODELS["jev"]["id"] == "typesafe/jev-1.13"
    assert client.MODELS["jev"]["prompt_per_million"] == "0.042"
    assert client.MODELS["jev"]["completion_per_million"] == "0"
    rows, seen = [], set()
    for case in sorted(data["cases"], key=lambda case: case["ordinal"]):
        pairs = [pair for pair in data["pairs"]
                 if (pair["doc_id"], pair["question_id"]) == (case["doc_id"], case["question_id"])]
        assert pairs and all(pair["query"] == case["query"] for pair in pairs)
        tasks = [{"id": pair["task_id"], "item": {"query": pair["query"],
                  "unit": {"id": pair["task_id"], "text": pair["passage"]}}} for pair in pairs]
        for task in tasks:
            assert task["id"] not in seen
            seen.add(task["id"])
        for batch_index, batch in enumerate(client.task_batches(tasks, "support")):
            payload = client.make_payload(batch, "support", "jev")
            serialized = client.canonical_bytes(payload)
            reserved, input_allowance, output_allowance = client.reservation(payload, "jev")
            assert len(serialized) <= 24000 and 1 <= len(batch) <= 8
            # Roundtrip verifies that serialized units retain the exact source slices.
            decoded = json.loads(serialized)
            assert decoded["state"]["items"] == {task["id"]: task["item"] for task in batch}
            rows.append({"ordinal": case["ordinal"], "batch_index": batch_index,
                         "decisions": len(batch), "wire_bytes": len(serialized),
                         "payload_sha256": sha(serialized), "historical_reservation_usd": str(reserved),
                         "input_allowance": input_allowance, "output_allowance": output_allowance})
    assert len(seen) == 117 and seen == {pair["task_id"] for pair in data["pairs"]}
    assert sum(row["decisions"] for row in rows) == 117
    result = {
        "schema": "slac-granularity-jev-wire-inspection-v1", "status": "offline_feasibility_only",
        "input_sha256": INPUT_SHA,
        "source_sha256": {path.name: sha(path.read_bytes()) for path in (
            Path(__file__).resolve(), Path(client.__file__).resolve())},
        "model_id_from_historical_client": client.MODELS["jev"]["id"],
        "historical_price_usd_per_million": {"input": "0.042", "output": "0"},
        "price_refreshed_this_inspection": False, "batches": rows,
        "summary": {"questions": 2, "decisions": 117, "requests_if_executed": len(rows),
                    "wire_bytes_total": sum(row["wire_bytes"] for row in rows),
                    "wire_bytes_maximum": max(row["wire_bytes"] for row in rows),
                    "historical_reservation_usd": str(sum(
                        (Decimal(row["historical_reservation_usd"]) for row in rows), Decimal(0)))},
        "api_calls": 0, "credentials_read": False, "references_read": False,
        "tokenizer_calls": 0, "model_calls": 0, "execution_plan_created": False,
        "scores_available": False,
        "limits": ["Historical client price is not a verified current quote or measured expense.",
                   "Reservation is client accounting, not an enforced supplier monetary cap.",
                   "Future JEV scores require a separate selection and execution protocol; not BGE logits."]}
    output = PHASE / "jev-wire-feasibility-01"
    public = ROOT / "docs/research/results/granularity_jev_wire_feasibility_20261005.json"
    assert not output.exists() and not public.exists()
    output.mkdir()
    write_new(output / "report.json", result)
    write_new(public, result)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
