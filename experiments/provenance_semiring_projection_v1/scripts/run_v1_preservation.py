from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.v1_preservation import (
    build_v1_result_preservation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalidate PR #19 P1/P2 and protected-tree conclusions")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_v1_result_preservation(args.repo_root.resolve(), args.artifact_root.resolve())
    path = args.artifact_root / "v1_result_preservation.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if result["status"] == "PR19_V1_RESULTS_PRESERVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

