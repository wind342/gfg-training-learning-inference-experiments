from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.report_statistics import (
    compute_report_statistics,
    inject_report_statistics,
    verify_report_artifact_consistency,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify artifact-derived report statistics")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path)
    parser.add_argument("--consistency-output", type=Path)
    parser.add_argument("--rendered-report-output", type=Path)
    args = parser.parse_args()
    statistics_output = args.statistics_output or args.artifact_root / "report_statistics.json"
    consistency_output = args.consistency_output or args.artifact_root / "report_artifact_consistency.json"
    rendered_report_output = args.rendered_report_output or args.report
    statistics = compute_report_statistics(args.artifact_root)
    rendered = inject_report_statistics(args.report.read_text(encoding="utf-8"), statistics)
    rendered_report_output.parent.mkdir(parents=True, exist_ok=True)
    rendered_report_output.write_text(rendered, encoding="utf-8", newline="\n")
    _write_json(statistics_output, statistics)
    consistency = verify_report_artifact_consistency(
        args.artifact_root,
        rendered_report_output,
        statistics,
    )
    _write_json(consistency_output, consistency)
    return 0 if consistency["status"] == "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
