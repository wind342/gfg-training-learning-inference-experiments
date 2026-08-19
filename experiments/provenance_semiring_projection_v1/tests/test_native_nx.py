from __future__ import annotations

from experiments.provenance_semiring_projection_v1.src.native_nx import evaluate_native_nx
from experiments.provenance_semiring_projection_v1.src.workloads import workload_by_id


def _only_polynomial(workload_id: str) -> dict:
    result = evaluate_native_nx(workload_by_id(workload_id))
    outputs = result["outputs"]
    assert isinstance(outputs, list) and len(outputs) == 1
    return outputs[0]["polynomial"]


def test_projection_sums_identity_distinct_alternatives() -> None:
    polynomial = _only_polynomial("W3")
    assert len(polynomial["terms"]) == 2
    assert all(term["coefficient"] == 1 for term in polynomial["terms"])


def test_repeated_alternative_has_explicit_coefficient_two() -> None:
    polynomial = _only_polynomial("W4")
    assert len(polynomial["terms"]) == 1
    term = polynomial["terms"][0]
    assert term["coefficient"] == 2
    assert [factor["exponent"] for factor in term["monomial"]] == [1, 1]


def test_duplicate_values_preserve_two_source_variables() -> None:
    polynomial = _only_polynomial("W5")
    assert len(polynomial["terms"]) == 2
    assert len({factor["variable"] for term in polynomial["terms"] for factor in term["monomial"]}) == 2


def test_self_join_records_exponent_two() -> None:
    polynomial = _only_polynomial("W6")
    term = polynomial["terms"][0]
    assert term["coefficient"] == 1
    assert len(term["monomial"]) == 1
    assert term["monomial"][0]["exponent"] == 2


def test_plan_equivalent_w11_queries_have_equal_native_nx() -> None:
    workload = workload_by_id("W11")
    plan_a = evaluate_native_nx(workload, variant="plan_a")
    plan_b = evaluate_native_nx(workload, variant="plan_b")
    assert plan_a["outputs"] == plan_b["outputs"]


def test_stress_polynomials_have_coefficients_exponents_and_varied_terms() -> None:
    result = evaluate_native_nx(workload_by_id("W12"))
    outputs = result["outputs"]
    assert isinstance(outputs, list) and len(outputs) == 25
    terms = [term for output in outputs for term in output["polynomial"]["terms"]]
    assert len(terms) >= 100
    assert any(term["coefficient"] > 1 for term in terms)
    assert any(any(factor["exponent"] > 1 for factor in term["monomial"]) for term in terms)
