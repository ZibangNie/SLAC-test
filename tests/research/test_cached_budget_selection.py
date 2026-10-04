"""Synthetic-only cached runner checks; no corpus, tokenizer or API access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/research"))
import run_cached_budget_selection as runner


def unit(index):
    return {"unit_id": f"u{index}", "order": index, "kind": "paragraph",
            "start": index, "end": index + 1,
            "text": f"synthetic text {index}", "native_text": f"native {index}"}


def sample_inputs():
    units = [unit(i) for i in range(5)]
    query = {"doc_id": "synthetic-doc", "question_id": "synthetic-question",
             "candidate_ids": ["u4", "u1", "u3"], "ranked_ids": ["u3", "u4", "u1"]}
    support = {(query["doc_id"], query["question_id"], uid): f"task-{uid}"
               for uid in query["candidate_ids"]}
    labels = {"task-u4": "unknown", "task-u1": "no", "task-u3": "yes"}
    scores = {tid: {"yes": .6, "no": .2, "unknown": .2} for tid in labels}
    return query, units, support, labels, scores


def test_candidate_projection_retains_unknown_excludes_no_and_uses_saved_rank():
    query, units, support, labels, scores = sample_inputs()
    candidates, indices = runner.build_candidates(query, units, support, labels, scores)
    assert indices == (3, 4)
    assert [c.priority for c in candidates] == [0, 1]
    assert [c.utility for c in candidates] == [.6, .6]
    assert [c.duplicate_key for c in candidates] == ["native 3", "native 4"]
    # Canonical render text is deliberately different from duplicate identity.
    assert runner.render(units, (4, 3)) == "[u3]\nsynthetic text 3\n\n[u4]\nsynthetic text 4"


@pytest.mark.parametrize("score", [
    {"yes": True, "no": 0, "unknown": 0},
    {"yes": float("nan"), "no": 0, "unknown": 0},
    {"yes": .5, "no": .5},
    {"yes": .98, "no": 0, "unknown": 0},
    {"yes": 1.01, "no": 0, "unknown": 0},
])
def test_rejects_invalid_reported_scores_even_on_excluded_candidate(score):
    args = sample_inputs()
    args[-1]["task-u1"] = score
    with pytest.raises(ValueError, match="cached JEV"):
        runner.build_candidates(*args)


def test_reported_scores_are_preserved_without_normalization():
    args = sample_inputs()
    args[-1]["task-u3"] = {"yes": .7, "no": .2, "unknown": .09}
    candidates, _ = runner.build_candidates(*args)
    assert candidates[0].utility == .7


def test_sample_is_order_independent_and_bounded_per_family():
    queries = [{"family_id": f"family-{f}", "doc_id": f"doc-{f}", "question_id": f"question-{q}"}
               for f in range(12) for q in range(4)]
    selected = runner.sample_identities(queries)
    assert selected == runner.sample_identities(list(reversed(queries)))
    assert len(selected) == 16 and len({q[0] for q in selected}) == 8
    assert all(sum(q[0] == family for q in selected) == 2 for family in {q[0] for q in selected})


def put(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_cost(indices):
    # Full-pack cost is deliberately not the sum of singleton costs.
    return sum((i + 1) * 13 for i in indices) + len(indices) ** 2 + 2 if indices else 0


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(runner, "ARTIFACTS", artifact_root)
    documents = {f"doc-{d}": [unit(i) for i in range(16)] for d in range(24)}
    queries, support, labels, scores, records = [], [], {}, {}, []
    for i in range(77):
        doc = f"doc-{i % 24}"
        identity = {"family_id": f"family-{i % 24}", "doc_id": doc, "question_id": f"question-{i}"}
        ids = [f"u{k}" for k in range(15 if i < 18 else 16)]
        queries.append({**identity, "query": "synthetic query", "candidate_ids": ids,
                        "ranked_ids": list(reversed(ids)), "seed_ids": ids[:8]})
        for uid in ids:
            tid = f"task-{i}-{uid}"
            support.append({"id": tid, "doc_id": doc, "question_id": identity["question_id"], "unit_id": uid})
            k = int(uid[1:])
            labels[tid] = "unknown" if k == 5 else "yes" if k in (3, 9) else "no"
            score = {3: .4, 5: .6, 9: .8}.get(k, .9)
            scores[tid] = {"yes": score, "no": 1 - score, "unknown": 0}
        selection = (3, 5, 9)
        records.append({**identity, "method": "p_yes_only_k3", "selected_ids": [f"u{k}" for k in selection],
                        "pack_sha256": runner.text_hash(runner.render(documents[doc], selection)),
                        "actual_evidence_tokens": packed_cost(selection),
                        # Legacy containers can carry metrics; the runner must not use them.
                        "official_evidence_f1": "UNREAD_SENTINEL", "official_answer_f1": {"ignored": True}})
    inputs = {runner.PREPARED: {"documents": documents, "queries": queries, "support_tasks": support},
              runner.LABELS: {"jev": labels}, runner.SCORES: scores, runner.OLD_RECORDS: records}
    hashes = {name: put(artifact_root / name, value, jsonl=name.endswith("jsonl")) for name, value in inputs.items()}
    tokenizer_dir = tmp_path / "synthetic-tokenizer"
    tok_hash = put(tokenizer_dir / "tokenizer.json", {"synthetic": True})
    contract = {"verified_consumed_sha256": hashes,
                "tokenizer": {"path": str(tokenizer_dir), "verified_file_sha256": {"tokenizer.json": tok_hash}}}
    contract_path = artifact_root / "contract.json"
    put(contract_path, contract)
    encodes = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": True, "truncation": False}
            indices = tuple(int(k) for k in re.findall(r"^\[u(\d+)\]", text, re.M))
            encodes.append(indices)
            return [0] * packed_cost(indices)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path) == tokenizer_dir
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    args = SimpleNamespace(contract=str(contract_path), output=str(artifact_root / "output"), max_seconds=300)
    return SimpleNamespace(args=args, root=artifact_root, contract=contract, inputs=inputs,
                           encodes=encodes, contract_path=contract_path)


def test_complete_synthetic_run_uses_bound_pack_cache_and_no_reference_fields(synthetic_run):
    fixture = synthetic_run
    result = runner.run(fixture.args)
    assert result["status"] == "completed" and result["baseline_reproduced_questions"] == 77
    assert result["api_calls"] == 0 and result["quality_metrics_computed"] is False
    assert result["reference_input"] is None
    # One seeded triple per doc; exactly six nonempty unseen subsets per doc.
    assert result["new_tokenizations"] == len(fixture.encodes) == 24 * 6
    assert (3, 5, 9) not in fixture.encodes
    assert result["methods"]["score_greedy"]["tokens"]["min"] == packed_cost((3, 5, 9))
    assert result["methods"]["score_exact"]["utility_win_tie_loss"] == [0, 77, 0]
    encoded_summary = json.dumps(result)
    assert "UNREAD_SENTINEL" not in encoded_summary and "synthetic query" not in encoded_summary


def test_changed_input_hash_stops_before_tokenizer(synthetic_run):
    fixture = synthetic_run
    path = fixture.root / runner.SCORES
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source binding"):
        runner.run(fixture.args)
    assert fixture.encodes == []
    assert not (Path(fixture.args.output) / "summary.json").exists()


def test_wrong_saved_full_pack_token_count_rejects_baseline(synthetic_run):
    fixture = synthetic_run
    rows = deepcopy(fixture.inputs[runner.OLD_RECORDS])
    for row in rows:
        row["actual_evidence_tokens"] = 2048
    fixture.contract["verified_consumed_sha256"][runner.OLD_RECORDS] = put(
        fixture.root / runner.OLD_RECORDS, rows, jsonl=True)
    put(fixture.contract_path, fixture.contract)
    with pytest.raises(ValueError, match="baseline"):
        runner.run(fixture.args)
    assert not (Path(fixture.args.output) / "summary.json").exists()


def test_artifact_boundary_rejects_traversal(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(runner, "ARTIFACTS", artifact_root)
    with pytest.raises(ValueError, match="artifact directory"):
        runner.under_artifacts(artifact_root / ".." / "outside")
    with pytest.raises(ValueError, match="artifact directory"):
        runner.under_artifacts(artifact_root)


def test_cli_disables_network_before_run_and_redacts_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner", "--contract", "unused", "--output", "unused"])
    # main mutates socket functions; enlist monkeypatch so teardown restores them.
    monkeypatch.setattr(runner.socket, "create_connection", runner.socket.create_connection)
    monkeypatch.setattr(runner.socket.socket, "connect", runner.socket.socket.connect)
    monkeypatch.setattr(runner.socket.socket, "connect_ex", runner.socket.socket.connect_ex)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    def stopped(args):
        assert runner.os.environ["HF_HUB_OFFLINE"] == "1"
        assert runner.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        with pytest.raises(RuntimeError, match="network disabled"):
            runner.socket.create_connection(("synthetic.invalid", 443))
        with pytest.raises(RuntimeError, match="network disabled"):
            runner.socket.socket.connect(None, ("synthetic.invalid", 443))
        raise ValueError("PRIVATE-SYNTHETIC-SENTINEL")

    monkeypatch.setattr(runner, "run", stopped)
    assert runner.main() == 1
    output = capsys.readouterr().out
    assert "PRIVATE-SYNTHETIC-SENTINEL" not in output
    assert json.loads(output) == {"status": "failed", "error_class": "ValueError", "api_calls": 0}
