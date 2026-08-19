from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.negative_controls import run_negative_controls


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute all unique fail-closed mutations exactly once")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    report, classification = run_negative_controls(args.artifact_root)
    _write(args.artifact_root / "negative_controls.json", report)
    _write(args.artifact_root / "negative_control_classification.json", classification)
    return 0 if report["status"] == "ALL_NEGATIVE_CONTROLS_FAILED_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
