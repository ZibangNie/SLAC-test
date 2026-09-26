"""Synthetic-only wrapper checks; never access the project data or network."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/research/screen_qasper_remaining_validation.py"
spec = importlib.util.spec_from_file_location("remaining_screen", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rows(path, values):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value) + "\n")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "EXPECTED", {"qasper_train": 1, "qasper_validation": 5, "legacy_train": 1, "legacy_dev": 1, "current": 1})
    monkeypatch.setattr(m, "BATCH_SIZE", 2)
    pool_dir, canonical = tmp_path / "pool", tmp_path / "canonical"
    pool_dir.mkdir()
    canonical.mkdir()
    text = " ".join(f"synthetic{i}" for i in range(160))
    body_hash = m.hashlib.sha256(m.old.normalized(text).encode()).hexdigest()
    documents, indexes = [], []
    for i in range(6):
        doc = {"doc_id": f"qasper:2000.0000{i}", "source_id": f"2000.0000{i}", "family_id": f"family{i}",
               "source": "qasper", "original_split": "train" if i == 0 else "validation",
               "canonical_text": text, "blocks": [{"kind": "paragraph", "char_span": [0, len(text)]}]}
        index = {k: doc[k] for k in ("doc_id", "source_id", "family_id", "source", "original_split")}
        index.update(normalized_body_sha256=body_hash, shard="documents-00000.jsonl.gz", row_in_shard=i)
        documents.append(doc)
        indexes.append(index)
    index_path, shard = canonical / "index.jsonl.gz", canonical / "documents-00000.jsonl.gz"
    rows(index_path, indexes)
    rows(shard, documents)
    candidates = pool_dir / "candidates.jsonl"
    rows(candidates, [{**{k: indexes[1][k] for k in ("doc_id", "source_id", "family_id", "normalized_body_sha256")}, "official_split": "validation"}])
    eligibility = pool_dir / "eligibility_audit.json"
    dump(eligibility, {"eligible_ids": [r["doc_id"] for r in indexes[1:]], "excluded": []})
    paths = [index_path, shard]
    for split in ("train", "dev"):
        path = tmp_path / f"refiner_{split}.jsonl"
        rows(path, [{"doc_id": "same-legacy-id", "atoms": [text], "orig_split": "test" if split == "train" else "dev", "meta": {"split": split}}])
        paths.append(path)
    pool = pool_dir / "pool_manifest.json"
    manifest = {"selected_documents": 1, "input_sha256": {str(p): m.old.digest(p) for p in paths},
                "candidate_manifest_sha256": m.old.digest(candidates), "all_input_hashes_unchanged": True,
                "counts": {"qa_field_present": 0, "qa_not_exported_metadata": 6, "canonical_train": 1, "canonical_validation": 5}}
    dump(pool, manifest)
    prior_dir = tmp_path / "prior"
    m.old.screen(pool, prior_dir)
    prior = prior_dir / "overlap_report.json"
    receipt = tmp_path / "metadata_review.json"
    dump(receipt, {"source_sha256": {str(p): m.old.digest(p) for p in (pool, candidates, index_path, eligibility)}})
    return dict(pool=pool, prior=prior, receipt=receipt, output=tmp_path / "new",
                index=index_path, shard=shard, candidates=candidates, eligibility=eligibility,
                indexes=indexes, documents=documents, text=text)


def prepare(f):
    return m.prepare(f["pool"], f["prior"], f["receipt"], f["output"])


def rewrite_metadata_bindings(f):
    pool = m.read_json(f["pool"])
    pool["input_sha256"][str(f["index"])] = m.old.digest(f["index"])
    dump(f["pool"], pool)
    prior = m.read_json(f["prior"])
    prior["input_sha256"][str(f["index"])] = m.old.digest(f["index"])
    prior["input_sha256"][str(f["pool"])] = m.old.digest(f["pool"])
    dump(f["prior"], prior)
    receipt = m.read_json(f["receipt"])
    for p in (f["pool"], f["index"], f["eligibility"]):
        receipt["source_sha256"][str(p)] = m.old.digest(p)
    dump(f["receipt"], receipt)


def test_real_size_batches_fixed_order_complete_no_duplicate():
    old_size = m.BATCH_SIZE
    try:
        m.BATCH_SIZE = 32
        values = [f"doc{i:03d}" for i in reversed(range(249))]
        batches = m.batches_for(values)
        assert list(map(len, batches)) == [32] * 7 + [25]
        assert sum(batches, []) == sorted(values)
        with pytest.raises(ValueError):
            m.batches_for(values + values[:1])
        with pytest.raises(ValueError):
            m.batches_for([])
    finally:
        m.BATCH_SIZE = old_size


def test_prepare_keeps_all_remaining_no_new_evaluation_pool(fixture):
    f = fixture
    result = prepare(f)
    assert result["batch_sizes"] == [2, 2]
    plan, data = m.load_plan(f["output"])
    assert sum(plan["batch_doc_ids"], []) == sorted({r["doc_id"] for r in f["indexes"][2:]})
    assert plan["purpose"] == "screen-only-not-evaluation-pool"
    assert not plan["holdout_selected"] and not plan["cleared_for_evaluation"]
    assert len(data["remaining"]) == 4
    for path in f["output"].rglob("*.json*"):
        assert f["text"] not in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare(f)


def test_full_aggregate_deduplicates_cross_batch_but_keeps_source_rows(fixture):
    f = fixture
    prepare(f)
    public = m.run(f["output"])
    assert public["status"] == "complete_document_screen_with_limits"
    assert public["flagged_pairs_raw"] == 26
    assert public["flagged_pairs_unique"] == 22
    assert public["flagged_remaining_documents"] == 4
    assert public["comparison_scopes"]["remaining_validation"] == {
        "raw_pairs_compared": 10, "unique_pairs_compared": 6, "raw_flagged_pairs": 10, "unique_flagged_pairs": 6}
    assert public["comparison_scopes"]["current_development"]["unique_pairs_compared"] == 4
    assert public["legacy_orig_split_counts_unique_source_rows"]["legacy_train_orig_split_test"] == 1
    assert public["legacy_orig_split_counts_across_batch_scans"]["legacy_train_orig_split_test"] == 2
    assert not public["official_test_payload_read"]
    assert not public["near_duplicate_clearance"]
    assert m.audit(f["output"])["status"] == "verified_complete_document_screen"
    assert "qasper:2000" not in json.dumps(public)
    assert "same-legacy-id" not in json.dumps(public)
    with pytest.raises(FileExistsError):
        m.run(f["output"])


@pytest.mark.parametrize("bad_key", ["qa_field_present", "qa_not_exported_metadata"])
def test_missing_no_qa_proof_rejected_before_payload_hash(fixture, monkeypatch, bad_key):
    f = fixture
    pool = m.read_json(f["pool"])
    pool["counts"].pop(bad_key)
    dump(f["pool"], pool)
    seen = []
    original = m.old.digest
    def spy(p):
        seen.append(Path(p))
        return original(p)
    monkeypatch.setattr(m.old, "digest", spy)
    with pytest.raises(ValueError, match="no-QA"):
        prepare(f)
    assert f["shard"] not in seen


def test_test_index_rejected_before_payload_hash_or_parse(fixture, monkeypatch):
    f = fixture
    f["indexes"][0]["original_split"] = "test"
    rows(f["index"], f["indexes"])
    rewrite_metadata_bindings(f)
    touched = []
    original_digest, original_jsonl = m.old.digest, m.old.jsonl
    def spy_hash(p):
        touched.append(Path(p))
        return original_digest(p)
    def spy_rows(p):
        touched.append(Path(p))
        yield from original_jsonl(p)
    monkeypatch.setattr(m.old, "digest", spy_hash)
    monkeypatch.setattr(m.old, "jsonl", spy_rows)
    with pytest.raises(ValueError, match="unsupported"):
        prepare(f)
    assert f["shard"] not in touched


def test_changed_index_rejected_before_payload_hash(fixture, monkeypatch):
    f = fixture
    with f["index"].open("ab") as handle:
        handle.write(b"change")
    original = m.old.digest
    def spy(p):
        assert Path(p) != f["shard"]
        return original(p)
    monkeypatch.setattr(m.old, "digest", spy)
    with pytest.raises(ValueError, match="index hash"):
        prepare(f)


def test_unbound_named_payload_cannot_enter_screen(fixture):
    f = fixture
    pool = m.read_json(f["pool"])
    pool["input_sha256"][str(f["pool"].parent / "documents-00001.jsonl.gz")] = "0" * 64
    dump(f["pool"], pool)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(f)


def test_unrecognized_qa_or_test_paths_are_never_opened(fixture, monkeypatch):
    f = fixture
    pool = m.read_json(f["pool"])
    forbidden = f["pool"].parent / "official_test_qa.json"
    pool["input_sha256"][str(forbidden)] = "0" * 64
    dump(f["pool"], pool)
    rewrite_metadata_bindings(f)
    original = m.old.digest
    def spy(p):
        assert Path(p) != forbidden
        return original(p)
    monkeypatch.setattr(m.old, "digest", spy)
    prepare(f)
    plan, data = m.load_plan(f["output"])
    assert str(forbidden) not in data["allowed"]
    assert str(forbidden) not in plan["input_sha256"]


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_eligibility_cannot_filter_or_duplicate_remaining(fixture, change):
    f = fixture
    audit = m.read_json(f["eligibility"])
    if change == "missing":
        audit["eligible_ids"].pop()
    else:
        audit["eligible_ids"].append(audit["eligible_ids"][0])
    dump(f["eligibility"], audit)
    rewrite_metadata_bindings(f)
    with pytest.raises(ValueError, match="eligibility"):
        prepare(f)


def test_batch_failure_never_yields_full_result_or_clearance(fixture, monkeypatch):
    f = fixture
    prepare(f)
    original = m.old.screen
    calls = []
    def fake(*args):
        calls.append(args)
        if len(calls) == 2:
            return {"status": "partial_runtime_limit", "all_input_hashes_unchanged": True}
        return original(*args)
    monkeypatch.setattr(m.old, "screen", fake)
    with pytest.raises(ValueError, match="incomplete"):
        m.run(f["output"])
    failure = m.read_json(f["output"] / "failure.json")
    assert failure["completed_batches"] == 1
    assert not failure["cleared_for_evaluation"]
    assert not (f["output"] / "public_aggregate.json").exists()
    with pytest.raises(ValueError, match="failed"):
        m.audit(f["output"])
    with pytest.raises(FileExistsError):
        m.run(f["output"])


def test_plan_batch_identity_change_even_resealed_rejected(fixture):
    f = fixture
    prepare(f)
    plan_file = f["output"] / "plan.json"
    plan = m.read_json(plan_file)
    plan["batch_doc_ids"][0][0] = plan["batch_doc_ids"][1][0]
    dump(plan_file, plan)
    seal_path = f["output"] / "prepare_seal.json"
    seal = m.read_json(seal_path)
    seal["plan.json"] = m.old.digest(plan_file)
    dump(seal_path, seal)
    with pytest.raises(ValueError, match="exactly"):
        m.load_plan(f["output"])


def test_public_tamper_and_reseal_not_self_consistent(fixture):
    f = fixture
    prepare(f)
    m.run(f["output"])
    public_path = f["output"] / "public_aggregate.json"
    public = m.read_json(public_path)
    public["flagged_pairs_unique"] = 0
    dump(public_path, public)
    complete_path = f["output"] / "completion.json"
    complete = m.read_json(complete_path)
    complete["output_sha256"]["public_aggregate.json"] = m.old.digest(public_path)
    dump(complete_path, complete)
    with pytest.raises(ValueError, match="replay"):
        m.audit(f["output"])


def test_incomplete_denominator_rejected(fixture):
    f = fixture
    prepare(f)
    m.run(f["output"])
    path = f["output"] / "batch-01/screen/overlap_report.json"
    report = m.read_json(path)
    report["counts"]["qasper_validation_pairs"] -= 1
    dump(path, report)
    plan, data = m.load_plan(f["output"])
    with pytest.raises(ValueError, match="denominator"):
        m.aggregate(f["output"], plan, data)


def test_duplicate_reverse_pair_metrics_rejected(fixture):
    f = fixture
    prepare(f)
    m.run(f["output"])
    path = f["output"] / "batch-02/screen/flagged_pairs.jsonl"
    pairs = [p for _, p in m.old.jsonl(path)]
    pair = next(p for p in pairs if p["reference_doc_id"] == f["indexes"][2]["doc_id"])
    pair["metrics"] = m.old.overlap(155, 157, 158)
    rows(path, pairs)
    report_path = path.parent / "overlap_report.json"
    report = m.read_json(report_path)
    report["output_sha256"][path.name] = m.old.digest(path)
    dump(report_path, report)
    plan, data = m.load_plan(f["output"])
    with pytest.raises(ValueError, match="reverse"):
        m.aggregate(f["output"], plan, data)


def test_legacy_same_document_id_different_rows_not_collapsed(fixture):
    f = fixture
    known = {r["doc_id"]: r for r in f["indexes"]}
    query = f["indexes"][2]["doc_id"]
    pair = dict(query_doc_id=query, reference_doc_id="same", reference_scope="legacy_train", reference_line=1,
                metrics=m.old.overlap(120, 120, 120), review_reasons=["high_lexical_overlap_review"])
    _, first, _ = m.pair_identity(pair, {query}, set(), known)
    pair["reference_scope"] = "legacy_dev"
    _, second, _ = m.pair_identity(pair, {query}, set(), known)
    assert first != second


def test_missing_reverse_cross_batch_flag_is_not_silently_deduplicated(fixture):
    f = fixture
    prepare(f)
    m.run(f["output"])
    path = f["output"] / "batch-02/screen/flagged_pairs.jsonl"
    pairs = [p for _, p in m.old.jsonl(path)]
    target = next(p for p in pairs if p["reference_doc_id"] == f["indexes"][2]["doc_id"])
    pairs.remove(target)
    rows(path, pairs)
    report_path = path.parent / "overlap_report.json"
    report = m.read_json(report_path)
    report["output_sha256"][path.name] = m.old.digest(path)
    report["flagged_pairs"] -= 1
    for reason in target["review_reasons"]:
        report["flagged_reason_counts"][reason] -= 1
        if report["flagged_reason_counts"][reason] == 0:
            del report["flagged_reason_counts"][reason]
    dump(report_path, report)
    plan, data = m.load_plan(f["output"])
    with pytest.raises(ValueError, match="missing reverse"):
        m.aggregate(f["output"], plan, data)


def test_version_ids_reuse_existing_identity_rule():
    assert m.old.identities({"source_id": "2000.12345v1"})[1] == m.old.identities({"source_id": "2000.12345v9"})[1]
    assert m.old.identities({"source_family": "qasper"})[0] is None
