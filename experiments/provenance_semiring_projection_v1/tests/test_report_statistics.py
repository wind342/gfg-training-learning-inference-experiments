from __future__ import annotations

import copy
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.report_statistics import (
    compute_report_statistics,
    inject_report_statistics,
    verify_report_artifact_consistency,
)


def _paths() -> tuple[Path, Path]:
    experiment = Path(__file__).resolve().parents[1]
    return experiment / "artifacts", experiment / "EXPERIMENT_REPORT.md"


def test_report_statistics_recompute_from_machine_artifacts() -> None:
    artifact_root, _report = _paths()
    statistics = compute_report_statistics(artifact_root)
    values = statistics["statistics"]
    assert values["case_count"] == 13
    assert values["source_variable_observation_count"] == 135
    assert values["output_count"] == 42
    assert values["polynomial_term_count"] == 197
    assert values["monomial_factor_count"] == 332
    assert values["exponent_observation_count"] == 332


def test_persisted_report_statistics_are_exact() -> None:
    artifact_root, report = _paths()
    persisted = compute_report_statistics(artifact_root)
    consistency = verify_report_artifact_consistency(artifact_root, report, persisted)
    assert consistency["status"] == "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS"


def test_wrong_report_number_fails_closed_without_repair(tmp_path: Path) -> None:
    artifact_root, report = _paths()
    statistics = compute_report_statistics(artifact_root)
    rendered = inject_report_statistics(report.read_text(encoding="utf-8"), statistics)
    wrong = rendered.replace("| Logical outputs | 42 |", "| Logical outputs | 41 |")
    wrong_report = tmp_path / "wrong_report.md"
    wrong_report.write_text(wrong, encoding="utf-8", newline="\n")
    persisted = copy.deepcopy(statistics)
    consistency = verify_report_artifact_consistency(artifact_root, wrong_report, persisted)
    assert consistency["status"] == "BLOCK"
    assert consistency["automatic_repair"] is False
    assert "report_generated_block_exact" in consistency["blocking_reasons"]
