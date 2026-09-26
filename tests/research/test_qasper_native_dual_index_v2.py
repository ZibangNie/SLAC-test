"""Only the verified local WDDM host UI may pass the unchanged idle gate."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_qasper_native_dual_index as v1
import run_qasper_native_dual_index_v2 as v2


def host(pid=19932):
    return {"Id": pid, "ProcessName": "ChatGPT", "Path": v2.VERIFIED_HOST_UI_PATH}


def sample(processes, blocked=False):
    return {"fifa_process_present": False, "compute_process_present": blocked,
        "observed_gpu_processes": processes, "gpu_utilization_percent": 0,
        "memory_used_mib": 313, "memory_total_mib": 8192, "memory_free_mib": 7879}


def test_all_scientific_functions_and_existing_parameters_remain_unchanged():
    functions = lambda module: {node.name: ast.dump(node, include_attributes=False) for node in
        ast.parse(Path(module.__file__).read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)}
    original, revised = functions(v1), functions(v2)
    changed = {name for name in original if original[name] != revised[name]}
    assert changed == {"source_hashes", "gpu_sample", "assert_idle"}
    assert {key: v2.CONFIG[key] for key in v1.CONFIG} == v1.CONFIG
    assert set(v2.CONFIG) - set(v1.CONFIG) == {"gpu_process_admission"}
    assert v2.LIMITS[:len(v1.LIMITS)] == v1.LIMITS
    assert v2.SCHEMA != v1.SCHEMA


@pytest.mark.parametrize("reported", ["ChatGPT.exe", "CHATGPT.EXE", v2.VERIFIED_HOST_UI_PATH.replace("/", "\\")])
def test_only_verified_host_ui_can_pass_with_matching_live_pid_and_path(reported):
    rows, blocked = v2.classify_gpu_processes(f"19932, {reported}", host)
    assert not blocked and len(rows) == 1 and v2.verified_host_ui(rows[0])
    v2.assert_idle(sample(rows, blocked))


@pytest.mark.parametrize("reported", ["[N/A]", "N/A", "unknown", "python.exe", "FIFA18.exe", "C:/other/ChatGPT.exe"])
def test_other_unknown_and_unverified_reported_names_are_blocked_without_lookup(reported):
    rows, blocked = v2.classify_gpu_processes(f"1234, {reported}", lambda _: pytest.fail("unapproved name must not become allowed by lookup"))
    assert blocked
    with pytest.raises(RuntimeError):
        v2.assert_idle(sample(rows, blocked))


@pytest.mark.parametrize("change", [
    {"Id":999}, {"ProcessName":"other"}, {"Path":"C:/other/ChatGPT.exe"}, {"Path":None},
    {"Path":"ChatGPT.exe"}, {"Path":v2.VERIFIED_HOST_UI_PATH.replace("26.917.9434.0","999.0")},
])
def test_live_os_process_identity_must_still_match(change):
    rows, blocked = v2.classify_gpu_processes("19932, ChatGPT.exe", lambda pid: {**host(pid), **change})
    assert blocked and not v2.verified_host_ui(rows[0])


def test_disappeared_or_inaccessible_pid_remains_blocked():
    def missing(pid):
        raise subprocess.CalledProcessError(1, ["Get-Process"])
    rows, blocked = v2.classify_gpu_processes("19932, ChatGPT.exe", missing)
    assert blocked and rows[0]["os_identity"] is None
    rows, blocked = v2.classify_gpu_processes("19932, ChatGPT.exe", lambda _: None)
    assert blocked


@pytest.mark.parametrize("raw", ["N/A, ChatGPT.exe", "0, ChatGPT.exe", "19932", "19932,", "19932,ChatGPT.exe,extra",
    "19932,ChatGPT.exe\n19932,ChatGPT.exe"])
def test_malformed_or_duplicate_gpu_identity_fails_closed(raw):
    with pytest.raises(RuntimeError):
        v2.classify_gpu_processes(raw, host)


def test_allowed_host_does_not_mask_second_compute_process():
    rows, blocked = v2.classify_gpu_processes("19932, ChatGPT.exe\n1234, python.exe", host)
    assert len(rows) == 2 and blocked
    with pytest.raises(RuntimeError):
        v2.assert_idle(sample(rows, blocked))
    with pytest.raises(ValueError, match="classification"):
        v2.assert_idle(sample(rows, False))


@pytest.mark.parametrize("field,value", [("fifa_process_present",True), ("gpu_utilization_percent",11), ("memory_free_mib",4095)])
def test_host_exception_never_relaxes_game_utilization_or_memory_threshold(field,value):
    rows,_ = v2.classify_gpu_processes("19932, ChatGPT.exe", host)
    row=sample(rows);row[field]=value
    with pytest.raises(RuntimeError):
        v2.assert_idle(row)


def test_gpu_sample_queries_pid_and_name_and_reads_current_path_without_terminating(monkeypatch):
    commands=[]
    def fake_run(args,**kwargs):
        commands.append(args)
        if args[0]=="powershell" and "Get-Process |" in args[-1]:
            return SimpleNamespace(stdout="")
        if args[0]=="nvidia-smi" and "--query-gpu=utilization.gpu,memory.used,memory.total" in args:
            return SimpleNamespace(stdout="0, 313, 8192")
        if args[0]=="nvidia-smi":
            assert "--query-compute-apps=pid,process_name" in args
            return SimpleNamespace(stdout="19932, ChatGPT.exe")
        assert "Get-Process -Id 19932 -ErrorAction Stop" in args[-1]
        return SimpleNamespace(stdout=json.dumps(host()))
    monkeypatch.setattr(v2.subprocess,"run",fake_run)
    row=v2.gpu_sample();v2.assert_idle(row)
    assert not row["compute_process_present"] and row["observed_gpu_processes"][0]["pid"]==19932
    assert len(commands)==4
    assert all(not any("Stop-Process" in token or "taskkill" in token for token in command) for command in commands)


def test_empty_compute_list_still_admits_only_with_idle_thresholds():
    rows,blocked=v2.classify_gpu_processes("",lambda _:pytest.fail("no process lookup expected"))
    assert rows==[] and not blocked
    v2.assert_idle(sample(rows,blocked))
