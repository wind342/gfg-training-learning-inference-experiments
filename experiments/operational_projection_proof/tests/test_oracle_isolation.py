from pathlib import Path

from experiments.operational_projection_proof.src.isolation_checks import (
    scan_candidate_isolation,
)


def test_database_candidate_has_no_oracle_or_native_file_access(proof_reports) -> None:
    report = proof_reports["oracle_isolation.json"]
    assert report["status"] == "SUPPORTED"
    assert report["oracle_leakage_count"] == 0
    assert report["native_domain_result_read_count"] == 0
    assert report["oracle_runtime_trap_passed"] is True


def test_import_from_symbol_leak_is_detected(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "from experiments.database_lineage.src import synthetic_oracle\n",
        encoding="utf-8",
    )
    report = scan_candidate_isolation(
        [candidate], forbidden_modules=("synthetic_oracle",)
    )
    assert report["oracle_leakage_count"] == 1
    assert report["status"] == "NOT_SUPPORTED"
