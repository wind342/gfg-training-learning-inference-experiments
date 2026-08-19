from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.nx_strictness import evaluate_nx_strictness


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real paired executions proving Gamma-to-N[X] strictness")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    strictness, reverse = evaluate_nx_strictness()
    _write(args.artifact_root / "nx_strictness_counterexamples.json", strictness)
    _write(args.artifact_root / "reverse_reconstruction_impossibility.json", reverse)
    return 0 if strictness["status"] == "STRICTNESS_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
