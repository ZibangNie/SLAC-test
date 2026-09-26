from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_refiner_labels.py"
spec = importlib.util.spec_from_file_location("repair_refiner_labels", MODULE_PATH)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def example(split):
    return {
        "sample_id": split + "-sample",
        "doc_id": "shared-doc",
        "atoms": [{"text": str(index)} for index in range(18)],
        "b0": [int(g in {12, 13}) for g in range(17)],
        "b_gold": [int(g in {14, 16}) for g in range(17)],
        "labels": {"edit": [{"g": 12, "y": "SHIFT:4"}, {"g": 13, "y": "SHIFT:1"}], "insert": [0] * 17},
        "meta": {"split": split, "K": 6, "noise": {"historical": True}},
        "orig_split": "test",
        "source_family": "synthetic-test-fixture",
    }


def inputs(tmp_path):
    paths = []
    for split in ("train", "dev"):
        path = tmp_path / f"refiner_{split}.jsonl"
        path.write_text(json.dumps(example(split)) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_export_preserves_sources_and_content_with_explicit_lineage(tmp_path):
    train, dev = inputs(tmp_path)
    original = {path: path.read_bytes() for path in (train, dev)}
    destination = tmp_path / "out"
    manifest = repair.run_export(train, dev, destination)
    assert manifest["status"] == "complete"
    assert manifest["test_split_payload_read"] is False
    assert manifest["flags"]["cleared_for_training"] is False
    assert manifest["train_dev_identity_overlap"] == {"doc_id": 1, "sample_id": 0, "exact_atom_texts": 1}
    for split, path in (("train", train), ("dev", dev)):
        assert path.read_bytes() == original[path]
        report = manifest["splits"][split]
        assert report["input_sha256"] == hashlib.sha256(original[path]).hexdigest()
        assert report["original_split_counts"] == {"test": 1}
        assert report["counts"]["original_nonmonotone_rows"] == 1
        assert report["counts"]["validated_monotone_rows"] == 1
        output = destination / report["output_file"]
        assert report["output_sha256"] == repair.sha256_file(output)
        result = json.loads(output.read_text(encoding="utf-8"))
        before = example(split)
        for field in set(before) - {"labels", "meta"}:
            assert result[field] == before[field]
        assert result["meta"]["noise"] == before["meta"]["noise"]
        assert result["meta"]["label_repair"]["flags"]["independent_evaluation"] is False
        assert result["labels"]["edit"] == [{"g": 12, "y": "SHIFT:2"}, {"g": 13, "y": "SHIFT:3"}]


def test_output_directory_must_not_exist(tmp_path):
    train, dev = inputs(tmp_path)
    destination = tmp_path / "out"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        repair.run_export(train, dev, destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(destination.iterdir()) == [sentinel]


def test_rejects_globs_and_test_input_filename(tmp_path):
    with pytest.raises(ValueError, match="glob"):
        repair.resolve_input(tmp_path / "refiner_*.jsonl", "train")
    path = tmp_path / "refiner_train_test.jsonl"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        repair.resolve_input(path, "train")


def test_bad_row_fails_closed_with_failed_manifest(tmp_path):
    train, dev = inputs(tmp_path)
    broken = example("dev")
    broken["b_gold"][0] = 1
    dev.write_text(json.dumps(broken), encoding="utf-8")
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="dev line 1"):
        repair.run_export(train, dev, destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["flags"]["cleared_for_training"] is False


def test_row_shape_and_split_are_checked():
    row = example("train")
    row["meta"]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        repair.repair_row(row, split="train", line_number=1, input_sha256="fixture")
