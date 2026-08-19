from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.lower_strictness import build_unification_artifacts, evaluate_lower_strictness


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Construct strictness witnesses below N[X]")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    lower, joint = evaluate_lower_strictness()
    hierarchy, result, w3c = build_unification_artifacts(args.artifact_root, lower, joint)
    _write(args.artifact_root / "lower_projection_strictness_constructions.json", lower)
    _write(args.artifact_root / "joint_lower_projection_strictness.json", joint)
    _write(args.artifact_root / "two_level_unification_hierarchy_v2.json", hierarchy)
    _write(args.artifact_root / "unification_of_unification_result_v2.json", result)
    _write(args.artifact_root / "two_level_unification_hierarchy.json", {
        "schema_version": "superseded-artifact-pointer-v1",
        "status": hierarchy["status"],
        "superseded_by": "two_level_unification_hierarchy_v2.json",
        "reason": "v2 separates formal algebraic targets from non-semiring task projections",
    })
    _write(args.artifact_root / "unification_of_unification_result.json", {
        "schema_version": "superseded-artifact-pointer-v1",
        "status": result["status"],
        "superseded_by": "unification_of_unification_result_v2.json",
        "reason": "v2 states the formal projection boundary explicitly",
    })
    _write(args.artifact_root / "nx_w3c_relation_scope.json", w3c)
    return 0 if result["status"] == "UNIFICATION_OF_UNIFICATION_FORMAL_BOUNDARY_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
