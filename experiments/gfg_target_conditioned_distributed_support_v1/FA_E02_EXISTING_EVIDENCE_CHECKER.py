from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .FA_E02_EXISTING_EVIDENCE_AUDIT import DEFAULT_GFG_ROOT, DEFAULT_OUTPUT, file_sha256, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfg-root", type=Path, default=DEFAULT_GFG_ROOT)
    parser.add_argument("--expected", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="fa_e02_check_") as directory:
        observed_path = Path(directory) / "audit.json"
        run(args.gfg_root, observed_path)
        observed = json.loads(observed_path.read_text(encoding="utf-8"))
        if expected != observed:
            raise RuntimeError("FA_E02_RECOMPUTATION_MISMATCH")
    report = {
        "schema": "fa-e02-existing-evidence-check-v1",
        "status": "PASS",
        "expected_sha256": file_sha256(args.expected),
        "source_run_count": expected["summary"]["source_scope"]["source_run_count"],
        "new_native_executions": expected["summary"]["source_scope"]["new_forward_executions"],
        "checks": [
            "source GFG validation status",
            "database SHA-256",
            "native tensor file and raw-array SHA-256",
            "independent full audit recomputation",
            "byte-equivalent structured result",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
