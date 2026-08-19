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
                "final_status": result["final_status"],
                "failure_reason": result["failure_reason"],
                "scientific_sha256": result["scientific_sha256"],
                "unmappable_primitive_relation_count": result[
                    "unmappable_primitive_relation_count"
                ],
                "artifact_count": manifest["artifact_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
