from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_STATS_BEGIN = "<!-- BEGIN MACHINE-GENERATED REPORT STATISTICS -->"
REPORT_STATS_END = "<!-- END MACHINE-GENERATED REPORT STATISTICS -->"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(name: str, path: Path, document: dict[str, Any]) -> str:
    if name != "manifest":
        return _sha(path)
    stable_summary = {
        "schema_version": document["schema_version"],
        "status": document["status"],
        "file_count": document["file_count"],
        "artifact_file_count": document["artifact_file_count"],
        "rehash_mismatch_count": document["rehash_mismatch_count"],
    }
    return hashlib.sha256(
        json.dumps(stable_summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _coverage_count(coverage: dict[str, Any], field: str) -> int:
    row = next(item for item in coverage["required_fields"] if item["field"] == field)
    return int(row["native_observation_count"])


def compute_report_statistics(artifact_root: Path) -> dict[str, Any]:
    paths = {
        "p1": artifact_root / "nx_exact_comparison.json",
        "coverage": artifact_root / "nx_field_coverage.json",
        "native": artifact_root / "native_nx_polynomials.json",
        "p2": artifact_root / "nx_strictness_counterexamples.json",
        "lower": artifact_root / "hierarchical_projection_exact_comparison.json",
        "negative": artifact_root / "negative_controls.json",
        "manifest": artifact_root / "artifact_manifest.json",
    }
    local_tests = artifact_root / "test_results.json"
    hardening_tests = artifact_root / "hardening_runs" / "run_1" / "test_results.json"
    paths["tests"] = (
        local_tests
        if local_tests.is_file()
        else hardening_tests
        if hardening_tests.is_file()
        else artifact_root / "runs" / "run_1" / "test_results.json"
    )
    missing = [path.as_posix() for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing report-statistic artifacts: {missing}")
    documents = {name: _read(path) for name, path in paths.items()}
    coverage = documents["coverage"]
    native = documents["native"]
    unique_source_identities = {
        row["source_identity"]
        for case in native["results"]
        for row in case["source_variables"]
    }
    lower_cases = documents["lower"]["cases"]
    algebraic_ids = set(documents["lower"]["required_algebraic_domains"])
    task_ids = set(documents["lower"]["required_task_projections"])
    statistics = {
        "case_count": int(documents["p1"]["actual_case_count"]),
        "source_variable_observation_count": _coverage_count(coverage, "source_variables[].variable"),
        "unique_source_identity_count": len(unique_source_identities),
        "output_count": _coverage_count(coverage, "outputs[].logical_output_key"),
        "polynomial_term_count": _coverage_count(coverage, "outputs[].polynomial.terms[].coefficient"),
        "coefficient_observation_count": _coverage_count(coverage, "outputs[].polynomial.terms[].coefficient"),
        "monomial_factor_count": _coverage_count(coverage, "outputs[].polynomial.terms[].monomial[].variable"),
        "exponent_observation_count": _coverage_count(coverage, "outputs[].polynomial.terms[].monomial[].exponent"),
        "strictness_pair_count": int(documents["p2"]["actual_pair_count"]),
        "real_execution_count": int(documents["p2"]["real_execution_count"]),
        "lower_domain_comparison_count": len(lower_cases),
        "formal_algebraic_comparison_count": sum(item["domain_id"] in algebraic_ids for item in lower_cases),
        "task_projection_comparison_count": sum(item["domain_id"] in task_ids for item in lower_cases),
        "negative_control_count": int(documents["negative"]["actual_control_count"]),
        "test_count": int(documents["tests"]["passed_count"]),
        "manifest_file_count": int(documents["manifest"]["file_count"]),
        "manifest_artifact_file_count": int(documents["manifest"]["artifact_file_count"]),
    }
    return {
        "schema_version": "report-statistics-v1",
        "status": "REPORT_STATISTICS_DERIVED_FROM_ARTIFACTS",
        "statistics": statistics,
        "sources": {
            name: {
                "path": path.relative_to(artifact_root).as_posix(),
                "sha256": _source_digest(name, path, documents[name]),
            }
            for name, path in paths.items()
        },
        "derivation_rules": {
            "observations": "read native_observation_count from nx_field_coverage required fields",
            "unique_sources": "set cardinality of source_identity across Native corpus source-variable maps",
            "lower_comparisons": "count classified comparison cases and split by algebraic/task domain IDs",
            "tests": "passed_count from the first hardening run when present, otherwise the frozen v1 run",
            "manifest": "file_count fields and a stable summary digest from artifact_manifest.json; per-file hashes remain verified by the manifest gate",
        },
        "manual_statistic_literals_allowed": False,
    }


def render_report_statistics_block(report: dict[str, Any]) -> str:
    stats = report["statistics"]
    rows = [
        ("Frozen P1 cases", "case_count"),
        ("Source-variable observations", "source_variable_observation_count"),
        ("Unique source identities", "unique_source_identity_count"),
        ("Logical outputs", "output_count"),
        ("Polynomial terms", "polynomial_term_count"),
        ("Coefficient observations", "coefficient_observation_count"),
        ("Monomial factors", "monomial_factor_count"),
        ("Exponent observations", "exponent_observation_count"),
        ("Strictness pairs", "strictness_pair_count"),
        ("Real strictness executions", "real_execution_count"),
        ("All classified lower comparisons", "lower_domain_comparison_count"),
        ("Formal algebraic comparisons", "formal_algebraic_comparison_count"),
        ("Task-projection comparisons", "task_projection_comparison_count"),
        ("Negative controls", "negative_control_count"),
        ("Passing bottom-up test areas", "test_count"),
        ("Manifested files", "manifest_file_count"),
        ("Manifested artifact files", "manifest_artifact_file_count"),
    ]
    return "\n".join(
        [
            REPORT_STATS_BEGIN,
            "## Machine-derived experiment statistics",
            "",
            "This block is generated from final machine artifacts; it is not maintained by hand.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *(f"| {label} | {stats[key]} |" for label, key in rows),
            REPORT_STATS_END,
        ]
    )


def inject_report_statistics(report_text: str, statistics: dict[str, Any]) -> str:
    if report_text.count(REPORT_STATS_BEGIN) != 1 or report_text.count(REPORT_STATS_END) != 1:
        raise ValueError("report must contain exactly one machine-statistics marker pair")
    prefix, remainder = report_text.split(REPORT_STATS_BEGIN, 1)
    _old, suffix = remainder.split(REPORT_STATS_END, 1)
    return prefix + render_report_statistics_block(statistics) + suffix


def verify_report_artifact_consistency(
    artifact_root: Path,
    report_path: Path,
    persisted_statistics: dict[str, Any],
) -> dict[str, Any]:
    recomputed = compute_report_statistics(artifact_root)
    report_text = report_path.read_text(encoding="utf-8")
    expected_block = render_report_statistics_block(recomputed)
    actual_block = (
        REPORT_STATS_BEGIN
        + report_text.split(REPORT_STATS_BEGIN, 1)[1].split(REPORT_STATS_END, 1)[0]
        + REPORT_STATS_END
        if REPORT_STATS_BEGIN in report_text and REPORT_STATS_END in report_text
        else ""
    )
    checks = {
        "persisted_statistics_equal_recomputation": persisted_statistics["statistics"] == recomputed["statistics"],
        "persisted_sources_equal_recomputation": persisted_statistics["sources"] == recomputed["sources"],
        "report_generated_block_exact": actual_block == expected_block,
        "stale_report_tuple_absent": "155 source-variable observations, 27 outputs, 592 polynomial terms" not in report_text,
        "all_statistic_source_files_present": all(
            (artifact_root / item["path"]).is_file()
            for item in persisted_statistics["sources"].values()
        ),
    }
    return {
        "schema_version": "report-artifact-consistency-v1",
        "status": "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS" if all(checks.values()) else "BLOCK",
        "gate": "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS",
        "checks": checks,
        "blocking_reasons": [name for name, passed in checks.items() if not passed],
        "automatic_repair": False,
        "report_path": report_path.name,
        "report_sha256": _sha(report_path),
        "statistics_sha256": hashlib.sha256(
            (json.dumps(persisted_statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest(),
    }
