from __future__ import annotations

import json

from ..materialize import materialize
from ..runner import run_all_scientific


def main() -> int:
    result, auxiliary = run_all_scientific()
    manifest = materialize(result, auxiliary)
    print(
        json.dumps(
            {
                "final_status_before_test_finalization": result[
                    "final_status"
                ],
                "scientific_sha256": result["scientific_sha256"],
                "failed_mandatory_gates": result[
                    "failed_mandatory_gates"
                ],
                "artifact_count": manifest["artifact_count"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    scientific_gates = [
        gate
        for gate in result["failed_mandatory_gates"]
        if not gate.endswith("_tests_passed")
    ]
    return 0 if not scientific_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
