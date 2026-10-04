"""One new exchange worker under the existing verified process watchdog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research.run_conditional_probe_bounded import _contained, supervise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    plan = _contained(args.plan)
    output = _contained(args.output_dir)
    receipt = _contained(args.receipt)
    # The key's path is passed as an argument, never loaded by this parent.
    # The worker verifies its complete immutable plan before credential access.
    command = [sys.executable, str(ROOT / "docs/research/run_exchange_probe.py"),
               "run", "--plan", str(plan), "--output-dir", str(output),
               "--key-file", str(args.key_file.resolve()), "--proxy", "http://127.0.0.1:7897"]
    result = supervise(command, output, receipt, timeout_seconds=180,
                       plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    print(json.dumps({key: result.get(key) for key in
                      ("status", "returncode", "worker_terminal_verified", "attempt_count",
                       "unresolved_cost_attempts", "elapsed_seconds")}))
    return 0 if result["status"] == "worker_exited" and result.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
