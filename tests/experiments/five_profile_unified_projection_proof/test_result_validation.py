from __future__ import annotations

import pytest

from experiments.five_profile_unified_projection_proof.src.result_validation import (
    MECHANISMS,
    ResultValidationError,
    validate_complete_result_set,
    validate_mechanism_result,
)


def test_complete_five_mechanism_set_passes(complete_results):
    validate_complete_result_set(complete_results)


def test_missing_mechanism_fails_closed(complete_results):
    complete_results.pop(MECHANISMS[-1])
    with pytest.raises(ResultValidationError, match="missing"):
        validate_complete_result_set(complete_results)


@pytest.mark.parametrize("section", ["p1", "p2"])
def test_any_projection_obligation_failure_fails_closed(result_factory, section):
    result = result_factory(MECHANISMS[0])
    result[section]["status"] = "FAIL"
    with pytest.raises(ResultValidationError):
        validate_mechanism_result(result)


@pytest.mark.parametrize("token", ["SKIP", "BLOCKED", "UNAVAILABLE", "NOT_INSTALLED"])
def test_non_execution_status_cannot_be_success(result_factory, token):
    result = result_factory(MECHANISMS[0])
    result["external_independence"]["basis"] = token
    with pytest.raises(ResultValidationError, match="forbidden"):
        validate_mechanism_result(result)


def test_artifact_schema_requires_every_p1_field(result_factory):
    result = result_factory(MECHANISMS[0])
    del result["p1"]["field_mismatch_count"]
    with pytest.raises(ResultValidationError, match="p1 missing"):
        validate_mechanism_result(result)


def test_artifact_hashes_must_be_sha256(result_factory):
    result = result_factory(MECHANISMS[0])
    result["artifact_hashes"] = {"science": "not-a-hash"}
    with pytest.raises(ResultValidationError, match="artifact hashes"):
        validate_mechanism_result(result)

