"""Synthetic-only gates for the four-case evidence evaluator; no real references."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import evaluate_score_transfer_microdiagnostic as m


def annotations(evidence=(), *, unanswerable=False):
    return [{"native_answer": {"unanswerable": unanswerable,
        "extractive_spans": [], "free_form_answer": "synthetic answer",
        "yes_no": None, "evidence": list(evidence)}}]


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    import prepare_qasper_confirmation_candidates as candidates
    from run_qasper_evidence_baselines import Unit, PackCounter, render_pack

    prep = tmp_path / "prep"
    prep.mkdir()
    monkeypatch.setattr(m, "ART", tmp_path)
    monkeypatch.setattr(m, "PREP", prep)
    monkeypatch.setattr(m, "REFERENCE", tmp_path / "never-open-real-reference.jsonl")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    class Tokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and truncation is False
            return [0] * (len(text.split()) + 1)

    model = tmp_path / "fake-tokenizer"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    monkeypatch.setattr(candidates, "DEFAULT_MODEL", model)
    monkeypatch.setattr(candidates, "TOKENIZER_HASHES",
                        {"config.json": hashlib.sha256(b"{}").hexdigest()})
    monkeypatch.setattr(candidates, "tokenizer_for", lambda path: Tokenizer())
    prepared = {"documents": {}, "queries": [], "support_tasks": [], "static_tasks": []}
    baselines, selected, labels, scores, references = [], [], {}, {}, {}
    for ordinal, line in enumerate((0, 4, 10, 17), 1):
        doc, qid = f"synthetic-doc-{ordinal}", f"synthetic-query-{ordinal}"
        identity = {"doc_id": doc, "question_id": qid, "family_id": f"component-{ordinal}"}
        units = [Unit(f"u{i:02d}", i, "paragraph", i * 10, i * 10 + 9,
                      f"display only {ordinal} {i}", f"native {ordinal} {i}") for i in range(16)]
        if ordinal == 3:
            units[2] = Unit("u02", 2, "figure", 20, 29, "display figure", "FLOAT SELECTED synthetic figure")
        prepared["documents"][doc] = [vars(u) for u in units]
        query = identity | {"query": "An invented query?", "source_id": f"source-{ordinal}",
            "original_family_id": f"original-{ordinal}", "candidate_ids": [u.unit_id for u in units],
            "ranked_ids": [u.unit_id for u in units]}
        prepared["queries"].append(query)
        selected.append({"question_zero_based_line": line, "doc_id": doc, "question_id": qid,
            "component_key": identity["family_id"], "source_id": query["source_id"],
            "original_family_id": query["original_family_id"]})
        for i, unit in enumerate(units):
            task_id = f"task-{ordinal}-{i}"
            prepared["support_tasks"].append({"id": task_id, "doc_id": doc,
                "question_id": qid, "unit_id": unit.unit_id})
            labels[task_id] = "yes" if i == 2 and ordinal != 4 else "no"
            scores[task_id] = {"yes": .8, "no": .1, "unknown": .1}
        count = PackCounter(Tokenizer(), units)
        packs = []
        for method, index in (("dense_k3", 0), ("reranker_k3", 1)):
            rendered = render_pack(units, [index])
            packs.append(identity | {"method": method, "budget": 1024,
                "selected_ids": [units[index].unit_id], "actual_evidence_tokens": count([index]),
                "rendered_pack": rendered, "pack_sha256": hashlib.sha256(rendered.encode()).hexdigest()})
        baselines.append({"identity": identity, "dense": packs[0], "bge": packs[1]})
        if ordinal == 1:
            refs = annotations((units[1].native_text, units[2].native_text))
        elif ordinal == 2:
            refs = annotations((units[0].native_text,)) + annotations((units[2].native_text,))
        elif ordinal == 3:
            refs = annotations((units[2].native_text,))
        else:
            refs = annotations(unanswerable=True)
        references[doc, qid] = refs
    identities = {"selected": selected}
    put(prep / "bounded_prepared.json", prepared)
    put(prep / "selected_identities.json", identities)
    monkeypatch.setattr(m, "frozen_inputs", lambda: deepcopy((prepared, baselines, identities)))
    execution = {"complete": True, "labels": labels, "reported_scores": scores, "bindings": {}}
    runner = SimpleNamespace(verify_run=lambda plan: deepcopy(execution))
    monkeypatch.setitem(sys.modules, "run_score_transfer_microdiagnostic", runner)
    return SimpleNamespace(root=tmp_path, output=tmp_path / "evaluation", runner=runner,
        execution=execution, references=references, identities=identities)


@pytest.mark.parametrize("failure", ["verification_error", "incomplete", "missing_judgment"])
def test_failed_support_cannot_access_reference_or_create_freeze(sandbox, monkeypatch, failure):
    if failure == "verification_error":
        def reject(plan):
            raise ValueError("failed execution verification")
        sandbox.runner.verify_run = reject
    elif failure == "incomplete":
        sandbox.execution["complete"] = False
    else:
        sandbox.execution["labels"].pop(next(iter(sandbox.execution["labels"])))
    accesses = []
    monkeypatch.setattr(m, "selected_references", lambda *_: accesses.append("forbidden"))
    with pytest.raises(ValueError):
        m.freeze(sandbox.root / "fake-plan.json", sandbox.output)
    assert accesses == []
    assert not sandbox.output.exists()


def test_seal_precedes_reference_read_and_official_native_metric(sandbox, monkeypatch):
    assert not m.REFERENCE.exists()
    frozen = m.freeze(sandbox.root / "fake-plan.json", sandbox.output)
    assert frozen == {"status": "frozen_before_reference_read", "cases": 4,
                      "packs": 12, "references_read": False}
    accesses = []

    def selected_only_after_seal(identities):
        manifest = m.read(sandbox.output / "frozen_manifest.json")
        assert manifest["pack_sha256"] == m.digest(sandbox.output / "frozen_packs.json")
        assert (sandbox.output / "reference_read_started.json").is_file()
        accesses.append("four synthetic references")
        return deepcopy(sandbox.references)

    monkeypatch.setattr(m, "selected_references", selected_only_after_seal)
    result = m.score(sandbox.output)
    assert accesses == ["four synthetic references"]
    # Independent hand calculation: multi-reference max, figure retained,
    # exact native text rather than display text, and empty/unanswerable=1.
    means = result["aggregate"]
    assert means["dense_k3"]["official_evidence_f1"] == pytest.approx(1 / 4)
    assert means["reranker_k3"]["official_evidence_f1"] == pytest.approx(1 / 6)
    assert means["p_yes_only_k3"]["official_evidence_f1"] == pytest.approx(11 / 12)
    assert result["primary_mean_delta"] == pytest.approx(3 / 4)
    stored = m.read(sandbox.output / "evidence_readout.json")
    assert len(stored["records"]) == 12
    assert not stored["answer_f1_computed"] and not stored["answer_generation"]


def test_changed_sealed_packs_block_reference_access(sandbox, monkeypatch):
    m.freeze(sandbox.root / "fake-plan.json", sandbox.output)
    with (sandbox.output / "frozen_packs.json").open("ab") as stream:
        stream.write(b" ")
    accesses = []
    monkeypatch.setattr(m, "selected_references", lambda *_: accesses.append("forbidden"))
    with pytest.raises(ValueError, match="packs changed"):
        m.score(sandbox.output)
    assert accesses == []
    assert not (sandbox.output / "reference_read_started.json").exists()


def test_changed_metric_source_blocks_reference_access(sandbox, monkeypatch):
    m.freeze(sandbox.root / "fake-plan.json", sandbox.output)
    real_digest = m.digest
    monkeypatch.setattr(m, "digest", lambda path: "0" * 64
                        if Path(path).name == "qasper_metrics.py" else real_digest(path))
    accesses = []
    monkeypatch.setattr(m, "selected_references", lambda *_: accesses.append("forbidden"))
    with pytest.raises(ValueError, match="metric"):
        m.score(sandbox.output)
    assert accesses == []
    assert not (sandbox.output / "reference_read_started.json").exists()


def reference_file(sandbox, monkeypatch, mismatch=None):
    rows = [b"deliberately invalid unselected JSON\n"] * 18
    for identity in sandbox.identities["selected"]:
        row = {k: identity[k] for k in ("doc_id", "question_id", "source_id", "original_family_id")}
        row["family_id"] = identity["component_key"]
        row["answer_annotations"] = sandbox.references[row["doc_id"], row["question_id"]]
        if mismatch and identity["question_zero_based_line"] == 4:
            row[mismatch] = "wrong-identity"
        rows[identity["question_zero_based_line"]] = (json.dumps(row) + "\n").encode()
    raw = b"".join(rows)
    path = sandbox.root / "synthetic-references.jsonl"
    path.write_bytes(raw)
    monkeypatch.setattr(m, "REFERENCE", path)
    monkeypatch.setattr(m, "REFERENCE_SHA", hashlib.sha256(raw).hexdigest())
    return path, raw


def test_reference_loader_decodes_only_the_four_selected_rows(sandbox, monkeypatch):
    reference_file(sandbox, monkeypatch)
    decoded = []
    original = json.loads

    def record_load(raw, *args, **kwargs):
        decoded.append(raw)
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(m.json, "loads", record_load)
    assert m.selected_references(sandbox.identities) == sandbox.references
    assert len(decoded) == 4
    assert all(b"deliberately invalid" not in raw for raw in decoded)


@pytest.mark.parametrize("field", ["doc_id", "question_id", "family_id", "source_id", "original_family_id"])
def test_reference_loader_rejects_identity_mismatch(sandbox, monkeypatch, field):
    reference_file(sandbox, monkeypatch, mismatch=field)
    with pytest.raises(ValueError, match="reference identity"):
        m.selected_references(sandbox.identities)


def test_reference_snapshot_cannot_change_between_hash_and_decode(sandbox, monkeypatch):
    path, original_raw = reference_file(sandbox, monkeypatch)
    original_open = Path.open
    reads = []

    def open_with_race(target, mode="r", *args, **kwargs):
        if target == path and mode == "rb":
            reads.append(True)
            if len(reads) == 2:
                # A second open would see different, still well-formed targets
                # under the same identities after a successful first hash.
                path.write_bytes(original_raw.replace(b"native 1 2", b"tampered target"))
        return original_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_race)
    assert m.selected_references(sandbox.identities) == sandbox.references
