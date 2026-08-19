from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.database_which_bridge import evaluate_existing_database_which_bridge


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect N[X] Vars to the frozen existing Database which-lineage")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    bridge, hierarchy = evaluate_existing_database_which_bridge(args.repo_root.resolve())
    _write(args.artifact_root / "nx_to_existing_which_lineage.json", bridge)
    _write(args.artifact_root / "database_lineage_hierarchy.json", hierarchy)
    return 0 if bridge["status"] == "THREE_WAY_EXACT_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
