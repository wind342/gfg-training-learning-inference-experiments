from __future__ import annotations

from typing import Any

import pytest

from experiments.five_profile_unified_projection_proof.src.mechanism_entry import CORE_COMMIT, SOURCE_COMMITS
from experiments.five_profile_unified_projection_proof.src.result_validation import MECHANISMS


@pytest.fixture
def result_factory():
    def build(mechanism: str) -> dict[str, Any]:
        return {
            "mechanism": mechanism,
            "profile_name": f"{mechanism}-v1",
            "source_commit": SOURCE_COMMITS[mechanism],
            "core_commit": CORE_COMMIT,
            "run_status": "PASS",
            "p1": {
                "status": "PASS",
                "candidate_record_count": 1,
                "native_record_count": 1,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "field_mismatch_count": 0,
                "multiplicity_mismatch_count": 0,
                "byte_equal": True,
                "query_count": 1,
                "query_mismatch_count": 0,
            },
            "p2": {
                "status": "PASS",
                "witness_count": 1,
                "valid_witness_count": 1,
                "snapshot_distinct": True,
                "target_equal": True,
                "witness_summaries": [{"id": f"{mechanism}-witness", "status": "SUPPORTED"}],
            },
            "external_independence": {"rating": "A", "basis": "test fixture"},
            "ordinary_output_orthogonality": {"checked": True, "status": "PASS"},
            "determinism": {"checked": False, "status": "FAIL", "run_1_hash": None, "run_2_hash": None},
            "artifact_hashes": {"science": "0" * 64},
        }

    return build


@pytest.fixture
def complete_results(result_factory):
    return {mechanism: result_factory(mechanism) for mechanism in MECHANISMS}

