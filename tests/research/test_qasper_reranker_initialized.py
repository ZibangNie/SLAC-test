"""Single-attempt initialization correction; all CUDA/model calls are mocked."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_reranker_initialized as launch
from test_qasper_reranker_baseline import preparation_fixture


@pytest.fixture
def prepared_launch(tmp_path, monkeypatch):
    runner = launch.runner
    base, _, _ = preparation_fixture(tmp_path, monkeypatch)
    runner.prepare(base)
    failed = tmp_path / "old_failed_run"
    failed.mkdir()
    runner.write(failed / "failure.json", {"status": "failed", "stage": "run",
        "error_class": "RuntimeError", "elapsed_seconds": 13.187, "api_calls": 0,
        "automatic_retries": 0, "cpu_fallback": False, "experimental_scores_available": False})
    stderr = tmp_path / "old_failure.stderr.log"
    stderr.write_text("torch.cuda.reset_peak_memory_stats(0)\nRuntimeError: Invalid device argument\n", encoding="utf-8")
    args = SimpleNamespace(base_plan=base.output, failed_run=failed, failed_stderr=stderr,
        run_output=tmp_path / "new_run", receipt_output=tmp_path / "launch" / "run",
        output=tmp_path / "launch" / "plan")
    monkeypatch.setattr(launch, "now", lambda: datetime(2026, 9, 26, 19, tzinfo=timezone.utc))
    monkeypatch.setattr(runner.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *a, **kw: pytest.fail("no model load in launcher tests"))
    return args


def wire_execution(monkeypatch, *, fail=None):
    events = []
    initialized = False
    def guard():
        events.append("guard")
        if fail == "guard":
            raise RuntimeError("synthetic admission refusal")
        return [{"name": "synthetic GPU", "free_mib": 5000, "utilization_percent": 0}] * 3
    def init():
        nonlocal initialized
        events.append("init")
        if fail == "init":
            raise RuntimeError("synthetic initialization failure")
        initialized = True
    def old_run(args):
        events.append("runner")
        # The original runner remains responsible for this second guard.
        guard()
        assert initialized
        if fail == "runner":
            raise RuntimeError("synthetic original runner failure")
        output = Path(args.output)
        output.mkdir()
        summary = {"status": "completed", "question_count": 1, "pair_count": 3,
                   "record_count": 3, "elapsed_seconds": 0.125}
        launch.runner.write(output / "summary.json", summary)
        launch.runner.write(output / "run_manifest.json", {"status": "completed"})
        for name in launch.runner.OUTPUT_FILES:
            (output / name).write_text("", encoding="utf-8")
        return summary
    monkeypatch.setattr(launch.runner, "ensure_gpu_ready", guard)
    monkeypatch.setattr(launch.runner.torch.cuda, "init", init)
    monkeypatch.setattr(launch.runner.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(launch.runner, "run", old_run)
    return events


def test_prepare_seals_original_failure_and_sources_without_cuda(prepared_launch, monkeypatch):
    monkeypatch.setattr(launch.runner.torch.cuda, "init", lambda: pytest.fail("prepare must not initialize CUDA"))
    monkeypatch.setattr(launch.runner, "ensure_gpu_ready", lambda: pytest.fail("prepare must not probe GPU/processes"))
    before = launch.snapshot(prepared_launch.base_plan, launch.BASE_FILES)[1]
    config = launch.prepare(prepared_launch)
    restored, binding = launch.load_plan(prepared_launch.output)
    assert restored == config and config["cuda_initialization_performed"] is False
    assert config["base_plan_sha256"] == launch.runner.digest(Path(prepared_launch.base_plan) / "experiment_config.json")
    assert config["prior_failure_sha256"] == launch.runner.digest(prepared_launch.failed_run / "failure.json")
    assert {str(path) for path in launch.sources()} <= set(binding)
    assert before == launch.snapshot(prepared_launch.base_plan, launch.BASE_FILES)[1]
    assert not prepared_launch.run_output.exists() and not prepared_launch.receipt_output.exists()


def test_guard_then_init_then_unchanged_runner_and_separate_receipt(prepared_launch, monkeypatch):
    launch.prepare(prepared_launch)
    events = wire_execution(monkeypatch)
    result = launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert events == ["guard", "init", "runner", "guard"]
    assert result["status"] == "completed"
    assert result["cuda_initialized_before"] is False and result["cuda_initialized_after"] is True
    assert result["runner_invocations"] == 1 and result["automatic_retries"] == 0
    assert result["scientific_outputs_independently_audited"] is False
    assert result["initialization_and_launcher_overhead_seconds"] >= 0
    assert result["total_launcher_seconds"] >= result["runner_elapsed_seconds"]
    assert set(path.name for path in prepared_launch.run_output.iterdir()) == {
        *launch.runner.OUTPUT_FILES, "summary.json", "run_manifest.json"}
    assert set(path.name for path in prepared_launch.receipt_output.iterdir()) == {"started.json", "receipt.json", "receipt_manifest.json"}
    with pytest.raises(FileExistsError, match="no retry"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert events == ["guard", "init", "runner", "guard"]


@pytest.mark.parametrize("failure,stage,expected", [
    ("guard", "outer_gpu_admission", ["guard"]),
    ("init", "cuda_initialization", ["guard", "init"]),
    ("runner", "unchanged_runner", ["guard", "init", "runner", "guard"]),
])
def test_failed_attempt_records_stage_and_is_never_retried(prepared_launch, monkeypatch, failure, stage, expected):
    launch.prepare(prepared_launch)
    events = wire_execution(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match="synthetic"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    receipt = launch.runner.read(prepared_launch.receipt_output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["failure_stage"] == stage
    assert receipt["experimental_scores_available"] is False
    assert receipt["automatic_retries"] == 0 and receipt["api_calls"] == 0
    with pytest.raises(FileExistsError):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert events == expected


def test_source_change_is_refused_before_admission_or_receipt(prepared_launch, monkeypatch):
    launch.prepare(prepared_launch)
    events = wire_execution(monkeypatch)
    prepared_launch.failed_stderr.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert events == [] and not prepared_launch.receipt_output.exists()


def test_cutoff_is_checked_before_gpu_work(prepared_launch, monkeypatch):
    launch.prepare(prepared_launch)
    events = wire_execution(monkeypatch)
    monkeypatch.setattr(launch, "now", lambda: datetime(2026, 9, 27, 1, tzinfo=timezone.utc))
    with pytest.raises(TimeoutError, match="cutoff"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert events == []
    assert launch.runner.read(prepared_launch.receipt_output / "receipt.json")["failure_stage"] == "deadline_before_admission"


def test_original_game_veto_prevents_initialization(prepared_launch, monkeypatch):
    launch.prepare(prepared_launch)
    monkeypatch.setattr(launch.runner, "user_game_running", lambda: True)
    monkeypatch.setattr(launch.runner, "gpu_sample", lambda: pytest.fail("FIFA veto must precede GPU sampling"))
    monkeypatch.setattr(launch.runner.torch.cuda, "init", lambda: pytest.fail("FIFA veto must precede initialization"))
    with pytest.raises(RuntimeError, match="FIFA18"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    assert launch.runner.read(prepared_launch.receipt_output / "receipt.json")["cuda_init_invoked"] is False


def test_receipt_cannot_pollute_immutable_plan_or_run(prepared_launch):
    prepared_launch.receipt_output = prepared_launch.run_output / "receipt"
    with pytest.raises(ValueError, match="separate"):
        launch.prepare(prepared_launch)
    assert not prepared_launch.output.exists()


def test_only_diagnosed_initialization_failure_is_eligible(prepared_launch):
    prepared_launch.failed_stderr.write_text("RuntimeError: CUDA out of memory", encoding="utf-8")
    with pytest.raises(ValueError, match="initialization failure"):
        launch.prepare(prepared_launch)


def test_postrun_source_change_invalidates_receipt(prepared_launch, monkeypatch):
    launch.prepare(prepared_launch)
    wire_execution(monkeypatch)
    original = launch.runner.run
    def changed(args):
        result = original(args)
        prepared_launch.failed_stderr.write_text("changed", encoding="utf-8")
        return result
    monkeypatch.setattr(launch.runner, "run", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        launch.run(SimpleNamespace(plan=prepared_launch.output))
    receipt = launch.runner.read(prepared_launch.receipt_output / "receipt.json")
    assert receipt["all_bound_inputs_unchanged"] is False
    assert receipt["experimental_scores_available"] is False
    assert receipt["failure_stage"] == "postrun_source_verification"
