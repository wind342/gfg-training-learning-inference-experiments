from __future__ import annotations

from typing import Any

from .nx_polynomial import NXPolynomial


def _derived_absorb(
    terms: frozenset[frozenset[str]],
) -> frozenset[frozenset[str]]:
    return frozenset(
        term for term in terms if not any(other < term for other in terms)
    )


def _bag(polynomial: NXPolynomial) -> int:
    return sum(coefficient for _, coefficient in polynomial.coefficients)


def _boolean(polynomial: NXPolynomial) -> bool:
    return bool(polynomial.coefficients)


def _flat_source_support(polynomial: NXPolynomial) -> dict[str, list[str]]:
    return {"variables": polynomial.variables()}


def _positive_boolean(polynomial: NXPolynomial) -> dict[str, list[list[str]]]:
    terms = _derived_absorb(frozenset(frozenset(variable for variable, _ in monomial) for monomial, _ in polynomial.coefficients))
    return {"terms": sorted((sorted(term) for term in terms), key=lambda term: (len(term), term))}


def derive_lower_domains_from_nx(native_nx_result: dict[str, Any]) -> dict[str, Any]:
    domain_functions = {
        "bag_naturals": _bag,
        "boolean": _boolean,
        "positive_boolean_lineage": _positive_boolean,
        "flat_source_support_view": _flat_source_support,
    }
    domains = []
    for domain_id, function in domain_functions.items():
        outputs = []
        for output in native_nx_result["outputs"]:
            polynomial = NXPolynomial.from_document(output["polynomial"])
            outputs.append({
                "logical_output_key": output["logical_output_key"],
                "values": output["values"],
                "annotation": function(polynomial),
            })
        domains.append({"domain_id": domain_id, "outputs": outputs})
    return {"workload_id": native_nx_result["workload_id"], "variant": native_nx_result["variant"], "domains": domains}


def compare_lower_hierarchy(direct: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    direct_cases = {(item["workload_id"], item["variant"]): item for item in direct["results"]}
    derived_cases = {(item["workload_id"], item["variant"]): item for item in derived["results"]}
    cases = []
    mismatch_count = 0
    domain_case_counts: dict[str, int] = {}
    for key in sorted(set(direct_cases) | set(derived_cases)):
        left = direct_cases.get(key)
        right = derived_cases.get(key)
        left_domains = {} if left is None else {domain["domain_id"]: domain["outputs"] for domain in left["domains"]}
        right_domains = {} if right is None else {domain["domain_id"]: domain["outputs"] for domain in right["domains"]}
        for domain_id in sorted(set(left_domains) | set(right_domains)):
            exact = left_domains.get(domain_id) == right_domains.get(domain_id)
            mismatch_count += not exact
            domain_case_counts[domain_id] = domain_case_counts.get(domain_id, 0) + 1
            cases.append({"workload_id": key[0], "variant": key[1], "domain_id": domain_id, "exact": exact})
    algebraic_domains = {"bag_naturals", "boolean", "positive_boolean_lineage"}
    task_projections = {"flat_source_support_view"}
    required_domains = algebraic_domains | task_projections
    coverage = set(domain_case_counts) == required_domains and all(
        count == 13 for count in domain_case_counts.values()
    )
    return {
        "schema_version": "hierarchical-projection-exact-comparison-v2",
        "claim": "Direct target execution equals the N[X]-derived result, with algebraic targets and task projections classified separately",
        "status": "FORMAL_PROJECTION_HIERARCHY_EXACT_SUPPORTED" if mismatch_count == 0 and coverage else "NOT_ESTABLISHED",
        "direct_path_computes_nx_first": False,
        "required_algebraic_domains": sorted(algebraic_domains),
        "required_task_projections": sorted(task_projections),
        "domain_classifications": {
            "bag_naturals": "COMMUTATIVE_SEMIRING_TARGET",
            "boolean": "COMMUTATIVE_SEMIRING_TARGET",
            "positive_boolean_lineage": "SEMIRING_QUOTIENT_OR_HOMOMORPHIC_IMAGE",
            "flat_source_support_view": "PARTIAL_NONZERO_SUPPORT_VIEW",
        },
        "domain_case_counts": domain_case_counts,
        "coverage_complete": coverage,
        "mismatch_count": mismatch_count,
        "repair_count": 0,
        "cases": cases,
    }
