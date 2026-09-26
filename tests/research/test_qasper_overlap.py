"""Synthetic tests; no project corpus, model, API or credentials are accessed."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/research/screen_qasper_overlap.py"
spec = importlib.util.spec_from_file_location("screen_qasper_overlap", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unicode_normalization_and_punctuation_are_explicit():
    assert module.shingles("CAFÉ, a b c d e") == module.shingles("cafe\u0301 a b c d e")
    assert module.shingles("") == set()
    assert module.shingles("One two") == {"one two"}


def test_exact_overlap_and_distinct_thresholds():
    values = module.overlap(50, 100, 150)
    assert values["jaccard"] == 0.25
    assert values["query_containment"] == 0.5
    assert values["reference_containment"] == pytest.approx(1 / 3)
    assert module.lexical_flags(values) == ["moderate_lexical_overlap_review"]
    assert module.lexical_flags(module.overlap(120, 125, 600)) == ["high_lexical_overlap_review"]
    assert not module.lexical_flags(module.overlap(20, 20, 20))


def test_arxiv_version_identity_does_not_use_text_or_domain():
    family, identifiers = module.identities({"doc_id": "qasper:1909.00694v3", "source_family": "qasper", "canonical_text": "Citation arXiv:1234.56789"})
    assert family is None
    assert identifiers == {"1909.00694"}


def test_whole_body_preserves_non_heading_atoms():
    assert module.legacy_body({"atoms": [{"type": "heading", "text": "title"}, {"text": "one"}, "two"]}) == "one\ntwo"
    with pytest.raises(ValueError):
        module.body({"canonical_text": "word", "blocks": [{"kind": "paragraph", "char_span": [0, 10]}]})


def dump_jsonl(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fixture_pool(tmp_path, *, include_test_index=False):
    pool = tmp_path / "pool"
    pool.mkdir()
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    text = " ".join(f"term{i}" for i in range(180))
    texts = [text, text + " revised document", " ".join(f"unrelated{i}" for i in range(180))]
    rows, indexes, candidates = [], [], []
    for i, value in enumerate(texts):
        doc_id = f"qasper:1909.0069{i}"
        split = "validation" if i in {0, 2} else "train"
        body_hash = hashlib.sha256(module.normalized(value).encode()).hexdigest()
        row = {"source": "qasper", "doc_id": doc_id, "source_id": f"1909.0069{i}", "family_id": f"1909.0069{i}",
               "original_split": split, "canonical_text": value, "blocks": [{"kind": "paragraph", "char_span": [0, len(value)]}]}
        index = {k: row[k] for k in ("source", "doc_id", "source_id", "family_id", "original_split")}
        index.update(shard="documents-00000.jsonl.gz", row_in_shard=i, normalized_body_sha256=body_hash)
        if i == 0:
            candidates.append({"doc_id": doc_id, "family_id": row["family_id"], "source_id": row["source_id"], "official_split": split, "normalized_body_sha256": body_hash})
        indexes.append(index)
        rows.append(row)
    if include_test_index:
        indexes[-1]["original_split"] = "test"
    dump_jsonl(canonical / "index.jsonl.gz", indexes)
    dump_jsonl(canonical / "documents-00000.jsonl.gz", rows)
    dump_jsonl(pool / "candidates.jsonl", candidates)
    legacy = []
    for split in ("train", "dev"):
        path = tmp_path / f"refiner_{split}.jsonl"
        dump_jsonl(path, [{"doc_id": "legacy_" + split, "doc_name": "1909.00690v2" if split == "train" else "other", "orig_split": "test" if split == "train" else "train", "atoms": [text] if split == "train" else [texts[2]], "meta": {"split": split}}])
        legacy.append(path)
    paths = [canonical / "index.jsonl.gz", canonical / "documents-00000.jsonl.gz", *legacy]
    manifest = {"selected_documents": 1, "candidate_manifest_sha256": module.digest(pool / "candidates.jsonl"), "input_sha256": {str(p): module.digest(p) for p in paths}}
    path = pool / "pool_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, texts


def test_full_screen_counts_flags_privacy_and_immutability(tmp_path):
    manifest, texts = fixture_pool(tmp_path)
    before = manifest.read_bytes()
    output = tmp_path / "output"
    result = module.screen(manifest, output)
    assert result["status"] == "complete_lexical_screen_with_limits"
    assert result["counts"]["qasper_train_pairs"] == 1
    assert result["counts"]["qasper_validation_pairs"] == 1  # Self excluded.
    assert result["counts"]["legacy_train_orig_split_test"] == 1
    assert result["flagged_pairs"] == 2
    assert result["flagged_reason_counts"]["arxiv_base_identity_review"] == 1
    assert result["all_input_hashes_unchanged"]
    assert not result["near_duplicate_clearance"]
    assert manifest.read_bytes() == before
    for path in output.iterdir():
        content = path.read_text(encoding="utf-8")
        assert not any(value in content for value in texts)
    with pytest.raises(FileExistsError):
        module.screen(manifest, output)


def test_test_index_rejected_before_canonical_payload(tmp_path, monkeypatch):
    manifest, _ = fixture_pool(tmp_path, include_test_index=True)
    original = module.jsonl
    original_digest = module.digest
    seen = []
    hashed = []
    def spy(path):
        seen.append(Path(path).name)
        yield from original(path)
    monkeypatch.setattr(module, "jsonl", spy)
    def hash_spy(path):
        hashed.append(Path(path).name)
        return original_digest(path)
    monkeypatch.setattr(module, "digest", hash_spy)
    with pytest.raises(ValueError, match="unsupported"):
        module.screen(manifest, tmp_path / "output")
    assert "documents-00000.jsonl.gz" not in seen
    assert "documents-00000.jsonl.gz" not in hashed


def test_candidate_pairs_are_deduplicated_with_exact_intersections(tmp_path):
    manifest_path, texts = fixture_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    candidate_path = manifest_path.parent / "candidates.jsonl"
    candidates = [row for _, row in module.jsonl(candidate_path)]
    candidates.append({"doc_id": "qasper:1909.00692", "source_id": "1909.00692", "family_id": "1909.00692", "official_split": "validation", "normalized_body_sha256": hashlib.sha256(module.normalized(texts[2]).encode()).hexdigest()})
    dump_jsonl(candidate_path, candidates)
    manifest.update(selected_documents=2, candidate_manifest_sha256=module.digest(candidate_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"
    report = module.screen(manifest_path, output)
    assert report["counts"]["qasper_train_pairs"] == 2
    assert report["counts"]["qasper_validation_pairs"] == 1
    pairs = [row for _, row in module.jsonl(output / "flagged_pairs.jsonl")]
    pair = next(p for p in pairs if p["query_doc_id"] == "qasper:1909.00690" and p["reference_scope"] == "qasper_train")
    query, reference = module.shingles(texts[0]), module.shingles(texts[1])
    assert pair["metrics"] == module.overlap(len(query & reference), len(query), len(reference))


def test_frozen_hash_mismatch_refused(tmp_path):
    manifest, _ = fixture_pool(tmp_path)
    with (manifest.parent / "candidates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="candidate file"):
        module.screen(manifest, tmp_path / "output")


def test_runtime_cap_report_is_partial(tmp_path, monkeypatch):
    manifest, _ = fixture_pool(tmp_path)
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, 2.0))
    report = module.screen(manifest, tmp_path / "output", max_seconds=1)
    assert report["status"] == "partial_runtime_limit"
    assert report["completed_scopes"] == []


def test_resource_cap_is_not_silent_truncation(tmp_path, monkeypatch):
    manifest, _ = fixture_pool(tmp_path)
    monkeypatch.setattr(module, "MAX_DOCUMENT_CHARACTERS", 10)
    with pytest.raises(ValueError, match="resource bound"):
        module.screen(manifest, tmp_path / "output")
