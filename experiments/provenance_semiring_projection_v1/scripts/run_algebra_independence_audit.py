from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.algebra_independence import (
    build_algebra_independence_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Native/Candidate algebra independence")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    experiment_root = Path(__file__).parents[1]
    audit = build_algebra_independence_audit(experiment_root)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "native_candidate_algebra_independence.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if audit["status"] == "NATIVE_CANDIDATE_ALGEBRA_INDEPENDENCE_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
