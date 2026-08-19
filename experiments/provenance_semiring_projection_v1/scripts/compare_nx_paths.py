from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.exact_comparison import (
    compare_nx_corpora,
    compare_nx_corpora_v2,
)


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare canonical Native and Candidate N[X] results without repair")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--comparison-v2", type=Path)
    args = parser.parse_args()
    native = json.loads(args.native.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison, coverage = compare_nx_corpora(native, candidate)
    _write(args.comparison, comparison)
    _write(args.coverage, coverage)
    comparison_v2 = compare_nx_corpora_v2(native, candidate)
    if args.comparison_v2 is not None:
        _write(args.comparison_v2, comparison_v2)
    return 0 if comparison["status"] == "EXACT_SUPPORTED" and comparison_v2["status"] == "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
