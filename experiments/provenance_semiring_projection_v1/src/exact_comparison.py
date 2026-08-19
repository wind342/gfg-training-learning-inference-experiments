from __future__ import annotations

from collections import Counter
from typing import Any


SEMANTIC_FIELDS = [
    "workload_id",
    "variant",
    "source_variables[].variable",
    "source_variables[].source_identity",
    "outputs[].logical_output_key",
    "outputs[].values",
    "outputs[].polynomial.schema_version",
    "outputs[].polynomial.terms[].coefficient",
    "outputs[].polynomial.terms[].monomial[].variable",
    "outputs[].polynomial.terms[].monomial[].exponent",
]


def _case_key(item: dict[str, Any]) -> tuple[str, str]:
    return item["workload_id"], item["variant"]


def _semantic_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_id": item["workload_id"],
        "variant": item["variant"],
        "source_variables": item["source_variables"],
        "outputs": item["outputs"],
    }


def compare_nx_corpora(native: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if native.get("schema_version") != "native-nx-corpus-v1":
        raise ValueError("unexpected Native corpus")
    if candidate.get("schema_version") != "core-projected-nx-corpus-v1":
        raise ValueError("unexpected Candidate corpus")
    native_by_case = {_case_key(item): item for item in native["results"]}
    candidate_by_case = {_case_key(item): item for item in candidate["results"]}
    if len(native_by_case) != len(native["results"]):
        raise ValueError("duplicate Native case")
    if len(candidate_by_case) != len(candidate["results"]):
        raise ValueError("duplicate Candidate case")
    all_keys = sorted(set(native_by_case) | set(candidate_by_case))
    cases = []
    mismatch_count = 0
    for key in all_keys:
        native_item = native_by_case.get(key)
        candidate_item = candidate_by_case.get(key)
        missing_side = None
        exact = False
        if native_item is None:
            missing_side = "native"
        elif candidate_item is None:
            missing_side = "candidate"
        else:
            exact = _semantic_result(native_item) == _semantic_result(candidate_item)
        if not exact:
            mismatch_count += 1
        cases.append(
            {
                "workload_id": key[0],
                "variant": key[1],
                "exact": exact,
                "missing_side": missing_side,
                "native_output_count": None if native_item is None else len(native_item["outputs"]),
                "candidate_output_count": None if candidate_item is None else len(candidate_item["outputs"]),
            }
        )
    expected_keys = {(f"W{index}", "default") for index in range(1, 13) if index != 11} | {("W11", "plan_a"), ("W11", "plan_b")}
    expected_case_coverage = set(all_keys) == expected_keys
    status = "EXACT_SUPPORTED" if mismatch_count == 0 and expected_case_coverage else "NOT_ESTABLISHED"
    comparison = {
        "schema_version": "nx-exact-comparison-v1",
        "claim": "Native K-relation N[X] equals Core-only Candidate N[X] exactly",
        "status": status,
        "expected_case_count": 13,
        "actual_case_count": len(all_keys),
        "expected_case_coverage": expected_case_coverage,
        "mismatch_count": mismatch_count,
        "repair_count": 0,
        "cases": cases,
    }
    field_counts = Counter()
    for item in native["results"]:
        field_counts["workload_id"] += 1
        field_counts["variant"] += 1
        for source in item["source_variables"]:
            field_counts["source_variables[].variable"] += "variable" in source
            field_counts["source_variables[].source_identity"] += "source_identity" in source
        for output in item["outputs"]:
            field_counts["outputs[].logical_output_key"] += "logical_output_key" in output
            field_counts["outputs[].values"] += "values" in output
            polynomial = output["polynomial"]
            field_counts["outputs[].polynomial.schema_version"] += "schema_version" in polynomial
            for term in polynomial["terms"]:
                field_counts["outputs[].polynomial.terms[].coefficient"] += "coefficient" in term
                for factor in term["monomial"]:
                    field_counts["outputs[].polynomial.terms[].monomial[].variable"] += "variable" in factor
                    field_counts["outputs[].polynomial.terms[].monomial[].exponent"] += "exponent" in factor
    coverage = {
        "schema_version": "nx-field-coverage-v1",
        "status": status,
        "required_fields": [
            {
                "field": field,
                "native_observation_count": field_counts[field],
                "candidate_compared": True,
                "mismatch_count": mismatch_count,
            }
            for field in SEMANTIC_FIELDS
        ],
        "all_required_fields_observed": all(field_counts[field] > 0 for field in SEMANTIC_FIELDS),
        "uncompared_required_field_count": 0,
        "repair_count": 0,
    }
    if not coverage["all_required_fields_observed"]:
        coverage["status"] = "NOT_ESTABLISHED"
        comparison["status"] = "NOT_ESTABLISHED"
    return comparison, coverage


def _observation_mismatch_count(left: dict[object, object], right: dict[object, object]) -> int:
    return sum(left.get(key) != right.get(key) for key in set(left) | set(right))


def _case_observations(item: dict[str, Any]) -> dict[str, dict[object, object]]:
    variables = {
        row["variable"]: row["source_identity"] for row in item["source_variables"]
    }
    outputs: dict[object, object] = {}
    coefficients: dict[object, object] = {}
    exponents: dict[object, object] = {}
    for output in item["outputs"]:
        output_key = output["logical_output_key"]
        outputs[output_key] = output["values"]
        for term in output["polynomial"]["terms"]:
            monomial_key = tuple(
                (factor["variable"], factor["exponent"])
                for factor in term["monomial"]
            )
            coefficients[(output_key, monomial_key)] = term["coefficient"]
            for factor in term["monomial"]:
                exponents[(output_key, monomial_key, factor["variable"])] = factor["exponent"]
    return {
        "variables": variables,
        "outputs": outputs,
        "coefficients": coefficients,
        "exponents": exponents,
    }


def compare_nx_corpora_v2(native: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare only final canonical documents from the two isolated authorities."""

    native_by_case = {_case_key(item): item for item in native["results"]}
    candidate_by_case = {_case_key(item): item for item in candidate["results"]}
    keys = sorted(set(native_by_case) | set(candidate_by_case))
    totals = {
        "variable_identity_mismatch_count": 0,
        "output_identity_mismatch_count": 0,
        "coefficient_mismatch_count": 0,
        "exponent_mismatch_count": 0,
        "canonical_polynomial_mismatch_count": 0,
    }
    cases = []
    for key in keys:
        left = native_by_case.get(key)
        right = candidate_by_case.get(key)
        if left is None or right is None:
            case_counts = {name: 1 for name in totals}
            exact = False
        else:
            left_obs = _case_observations(left)
            right_obs = _case_observations(right)
            case_counts = {
                "variable_identity_mismatch_count": _observation_mismatch_count(
                    left_obs["variables"], right_obs["variables"]
                ),
                "output_identity_mismatch_count": _observation_mismatch_count(
                    left_obs["outputs"], right_obs["outputs"]
                ),
                "coefficient_mismatch_count": _observation_mismatch_count(
                    left_obs["coefficients"], right_obs["coefficients"]
                ),
                "exponent_mismatch_count": _observation_mismatch_count(
                    left_obs["exponents"], right_obs["exponents"]
                ),
                "canonical_polynomial_mismatch_count": sum(
                    left_output.get("polynomial") != right_output.get("polynomial")
                    for output_key in set(left_obs["outputs"]) | set(right_obs["outputs"])
                    for left_output, right_output in [
                        (
                            next((item for item in left["outputs"] if item["logical_output_key"] == output_key), {}),
                            next((item for item in right["outputs"] if item["logical_output_key"] == output_key), {}),
                        )
                    ]
                ),
            }
            exact = not any(case_counts.values())
        for name, value in case_counts.items():
            totals[name] += value
        cases.append(
            {
                "workload_id": key[0],
                "variant": key[1],
                "exact": exact,
                **case_counts,
            }
        )
    expected = {(f"W{index}", "default") for index in range(1, 13) if index != 11} | {
        ("W11", "plan_a"),
        ("W11", "plan_b"),
    }
    exact = set(keys) == expected and not any(totals.values()) and all(item["exact"] for item in cases)
    return {
        "schema_version": "native-candidate-nx-exact-comparison-v2",
        "status": "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED" if exact else "NOT_ESTABLISHED",
        "claim": "The independent Native polynomial oracle and Core-only Candidate agree on final canonical documents",
        "comparison_authority": "pure structural parsing of isolated final JSON documents",
        "automatic_repair_count": 0,
        "expected_case_count": 13,
        "actual_case_count": len(keys),
        **totals,
        "cases": cases,
    }
