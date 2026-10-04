"""One-shot outer deadline for the separately frozen six-request JEV probe.

The parent never loads a credential or edits the worker's accounting ledger.
Killing the worker cannot cancel a request already accepted by the provider.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "artifacts/research-foundation/offline-20261004"
RUNNER = ROOT / "docs/research/run_conditional_probe.py"
HARD_DEADLINE_SECONDS = 180


def _contained(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == OUTPUT_ROOT.resolve() or not resolved.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("probe output must be a child of its dedicated artifact directory")
    return resolved


def _save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _ledger_observation(run_dir: Path) -> dict:
    try:
        with (run_dir / "ledger.json").open("rb") as stream:
            raw = stream.read(131073)
        if len(raw) > 131072:
            raise ValueError("oversized ledger")
        ledger = json.loads(raw)
        attempts = ledger["attempts"]
        if not isinstance(attempts, list) or any(not isinstance(item, dict) for item in attempts):
            raise ValueError("invalid attempt records")
    except (OSError, ValueError, KeyError, TypeError):
        return {"ledger_observed": False, "attempt_count": None,
                "unresolved_cost_attempts": None}
    return {"ledger_observed": True, "attempt_count": len(attempts),
            "in_flight_attempts": sum(item.get("status") == "in_flight" for item in attempts),
            "unresolved_cost_attempts": sum(item.get("cost_status") != "provider_reported" for item in attempts)}


def supervise(command: list[str], run_dir: Path, receipt: Path, *,
              timeout_seconds: float = HARD_DEADLINE_SECONDS, plan_sha256: str) -> dict:
    """Run one worker once; command injection is impossible with shell=False.

    The CLI supplies the fixed Python runner. A direct command parameter lets
    tests exercise actual process termination with a non-network worker.
    """
    if not 0 < timeout_seconds <= HARD_DEADLINE_SECONDS:
        raise ValueError("invalid process deadline")
    run_dir, receipt = _contained(Path(run_dir)), _contained(Path(receipt))
    if run_dir.exists():
        raise FileExistsError("probe run directory already exists; resumption is forbidden")
    if not receipt.parent.is_dir() or receipt.is_relative_to(run_dir):
        raise ValueError("controller receipt must be outside the new worker directory")
    result = {"schema": "slac-conditional-probe-controller-v1", "status": "starting",
              "started_utc": datetime.now(timezone.utc).isoformat(), "pid": None,
              "hard_deadline_seconds": timeout_seconds, "plan_sha256": plan_sha256,
              "automatic_restarts": 0, "upstream_cancellation_claimed": False}
    with receipt.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(result, sort_keys=True) + "\n")
    start = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result.update(status="running", pid=process.pid)
        _save(receipt, result)
        try:
            code = process.wait(timeout=max(0.001, timeout_seconds - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            result["status"] = "hard_deadline_exceeded"
        else:
            result["status"] = "worker_exited"
        result["returncode"] = code
        result["worker_terminal_verified"] = process.poll() is not None
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        result.update(status="controller_aborted", error_class=type(exc).__name__,
                      worker_terminal_verified=process is None or process.poll() is not None)
        # No arbitrary exception text, command arguments or credentials retained.
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result.update(_ledger_observation(run_dir))
        _save(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = parser.parse_args()
    if args.proxy != "http://127.0.0.1:7897":
        raise ValueError("probe uses the fixed local proxy only")
    plan = _contained(args.plan)
    # Read only the plan, never the credential. Worker performs the complete
    # source/wire verification again before any credential access or model call.
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    command = [sys.executable, str(RUNNER), "run", "--plan", str(plan),
               "--output-dir", str(_contained(args.output_dir)),
               "--key-file", str(args.key_file.resolve()), "--proxy", args.proxy]
    result = supervise(command, args.output_dir, args.receipt, plan_sha256=plan_sha)
    print(json.dumps({key: result.get(key) for key in
                      ("status", "returncode", "worker_terminal_verified", "attempt_count",
                       "unresolved_cost_attempts", "elapsed_seconds")}))
    return 0 if result["status"] == "worker_exited" and result.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
