"""The supplemental metadata gate rejects resealed descriptions, without GPU."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import audit_qasper_reranker_metadata as metadata
from test_qasper_reranker_baseline import saved_run_fixture, reseal_test_run


@pytest.fixture
def complete(tmp_path, monkeypatch):
    runner = metadata.runner
    plan, run = saved_run_fixture(tmp_path, monkeypatch)
    summary = runner.read(run / "summary.json")
    pair_audit = [json.loads(line) for line in (plan / "pair_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    summary.update(copy.deepcopy(metadata.FIXED_FIELDS))
    summary.update(compute={**metadata.expected_padding(pair_audit), "peak_allocated_bytes": 100,
                            "peak_reserved_bytes": 200}, elapsed_seconds=1.5,
        execution={"torch": str(runner.torch.__version__), "transformers": runner.transformers.__version__,
            "cuda_runtime": runner.torch.version.cuda, "device": "cuda:0", "dtype": "float16",
            "gpu_name": "synthetic GPU", "gpu_capability": [12, 0]},
        gpu_admission_samples=[{"name": "synthetic GPU", "free_mib": 5000, "utilization_percent": 0} for _ in range(3)])
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    reseal_test_run(run)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata audit must not load a model or query GPU/processes")
    monkeypatch.setattr(runner.AutoModelForSequenceClassification, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "gpu_sample", forbidden)
    monkeypatch.setattr(runner, "user_game_running", forbidden)
    monkeypatch.setattr(runner, "ensure_gpu_ready", forbidden)
    return SimpleNamespace(plan=plan, run=run, output=tmp_path / "metadata_audit")


def change_summary(args, mutate):
    summary = metadata.runner.read(args.run / "summary.json")
    mutate(summary)
    (args.run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    reseal_test_run(args.run)


def test_complete_saved_scientific_run_has_supplemental_metadata_and_no_gpu_probe(complete):
    result = metadata.audit(complete)
    assert result["status"] == "verified" and result["scientific_audit"]["records"] == 3
    assert result["metadata_audit"]["derived_counts"]["question_count"] == 1
    assert result["gpu_or_process_probe_performed"] is False
    assert result["metadata_audit"]["fifa_veto"]["historical_process_absence_independently_verified"] is False
    assert "doc-secret" not in json.dumps(result)
    assert "question-secret" not in json.dumps(result)
    binding = metadata.runner.read(complete.output / "source_binding.json")["input_sha256"]
    metadata.runner.verify_hashes(binding)


@pytest.mark.parametrize("field", list(metadata.FIXED_FIELDS))
def test_resealed_wrong_fixed_description_is_rejected(complete, field):
    expected = metadata.FIXED_FIELDS[field]
    wrong = not expected if type(expected) is bool else 99 if type(expected) is int else [] if isinstance(expected, list) else "wrong"
    change_summary(complete, lambda row: row.update({field: wrong}))
    # Original scientific audit still passes: this is the known supplemented gap.
    assert metadata.runner.audit_saved_run(complete.plan, complete.run)["status"] == "verified"
    with pytest.raises(ValueError, match="metadata field"):
        metadata.audit(complete)
    assert not complete.output.exists()


@pytest.mark.parametrize("change", ["free", "utilization", "sample_count", "dtype", "package", "negative_peak", "bad_padding", "elapsed_nan", "private_id"])
def test_recorded_resource_and_public_identity_checks(complete, change):
    def mutate(row):
        if change == "free": row["gpu_admission_samples"][0]["free_mib"] = 4095
        elif change == "utilization": row["gpu_admission_samples"][0]["utilization_percent"] = 21
        elif change == "sample_count": row["gpu_admission_samples"].pop()
        elif change == "dtype": row["execution"]["dtype"] = "float32"
        elif change == "package": row["execution"]["torch"] = "other-version"
        elif change == "negative_peak": row["compute"]["peak_allocated_bytes"] = -1
        elif change == "bad_padding": row["compute"]["padded_input_tokens"] += 1
        elif change == "elapsed_nan": row["elapsed_seconds"] = float("nan")
        elif change == "private_id": row["unexpected_private_id"] = "question-secret"
    change_summary(complete, mutate)
    with pytest.raises(ValueError):
        metadata.audit(complete)
    assert not complete.output.exists()


def test_scientific_audit_precedes_metadata_and_cannot_be_bypassed(complete, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("scientific records do not replay")
    monkeypatch.setattr(metadata.runner, "audit_saved_run", refuse)
    with pytest.raises(ValueError, match="scientific records"):
        metadata.audit(complete)
    assert not complete.output.exists()


def test_extra_run_file_refused_without_mutating_sealed_inputs(complete):
    (complete.run / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        metadata.audit(complete)
    assert not complete.output.exists()


@pytest.mark.parametrize("phase", ["scientific_audit", "metadata"])
def test_bound_input_change_during_audit_is_rejected(complete, monkeypatch, phase):
    if phase == "scientific_audit":
        target = complete.run / "summary.json"
        owner, name = metadata.runner, "audit_saved_run"
    else:
        config = metadata.runner.read(complete.plan / "experiment_config.json")
        target = Path(config["prepared_dir"]) / "prepared.json"
        owner, name = metadata, "validate_metadata"
    original = getattr(owner, name)
    def change_after_check(*args, **kwargs):
        result = original(*args, **kwargs)
        target.write_bytes(target.read_bytes() + b" ")
        return result
    monkeypatch.setattr(owner, name, change_after_check)
    with pytest.raises(ValueError, match="hash mismatch"):
        metadata.audit(complete)
    assert not complete.output.exists()


def test_padded_token_totals_include_incomplete_last_microbatch():
    assert metadata.expected_padding([{"pair_tokens": n} for n in [1, 4, 2, 5, 3]]) == {
        "padded_input_tokens": 21, "max_padded_tokens_per_batch": 20, "max_microbatch": 4}
