from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidate_nx import project_snapshot_to_nx
from .core_capture import core_snapshot_from_events
from .native_lower_k import evaluate_direct_lower_domains
from .native_nx import evaluate_native_nx


def _coefficient_workload() -> dict[str, Any]:
    join_a = {
        "op": "join", "stage": "ls_coefficient_join_a",
        "left": {"op": "base", "relation": "R"}, "right": {"op": "base", "relation": "S"},
        "left_keys": ["key"], "right_keys": ["key"], "right_prefix": "s_",
    }
    join_b = {
        "op": "join", "stage": "ls_coefficient_join_b",
        "left": {"op": "base", "relation": "R"}, "right": {"op": "base", "relation": "S"},
        "left_keys": ["key"], "right_keys": ["key"], "right_prefix": "s_",
    }
    return {
        "id": "LS_COEFFICIENT",
        "relations": {
            "R": [{"source_identity": "LSC:R:r1", "values": {"key": 1, "value": "same"}}],
            "S": [{"source_identity": "LSC:S:s1", "values": {"key": 1, "tag": "t"}}],
        },
        "queries": {
            "single": {"op": "project", "stage": "ls_coefficient_project_single", "input": join_a, "fields": [{"output": "value", "input": "value"}]},
            "double": {
                "op": "union", "stage": "ls_coefficient_union", "inputs": [
                    {"op": "project", "stage": "ls_coefficient_project_a", "input": join_a, "fields": [{"output": "value", "input": "value"}]},
                    {"op": "project", "stage": "ls_coefficient_project_b", "input": join_b, "fields": [{"output": "value", "input": "value"}]},
                ],
            },
        },
        "default_query": "single",
    }


def _balanced_bag_workload() -> dict[str, Any]:
    return {
        "id": "LS_BALANCED_BAG",
        "relations": {
            "A": [{"source_identity": "LSB:A:a1", "values": {"value": "same"}}],
            "B": [{"source_identity": "LSB:B:b1", "values": {"value": "same"}}],
        },
        "queries": {
            "two_sources": {"op": "union", "stage": "ls_bag_union_sources", "inputs": [{"op": "base", "relation": "A"}, {"op": "base", "relation": "B"}]},
            "one_source_twice": {"op": "union", "stage": "ls_bag_union_duplicate", "inputs": [{"op": "base", "relation": "A"}, {"op": "base", "relation": "A"}]},
        },
        "default_query": "two_sources",
    }


def _exponent_workload() -> dict[str, Any]:
    return {
        "id": "LS_EXPONENT",
        "relations": {"R": [{"source_identity": "LSE:R:r1", "values": {"key": 1, "value": "same"}}]},
        "queries": {
            "single_use": {"op": "project", "stage": "ls_exponent_project_single", "input": {"op": "base", "relation": "R"}, "fields": [{"output": "value", "input": "value"}]},
            "self_join": {
                "op": "project", "stage": "ls_exponent_project_self",
                "input": {"op": "join", "stage": "ls_exponent_self_join", "left": {"op": "base", "relation": "R"}, "right": {"op": "base", "relation": "R"}, "left_keys": ["key"], "right_keys": ["key"], "right_prefix": "r2_"},
                "fields": [{"output": "value", "input": "value"}],
            },
        },
        "default_query": "single_use",
    }


PAIR_SPECS = [
    ("lower-coefficient", _coefficient_workload(), "single", "double"),
    ("lower-balanced-bag", _balanced_bag_workload(), "two_sources", "one_source_twice"),
    ("lower-exponent", _exponent_workload(), "single_use", "self_join"),
]


def _execute(workload: dict[str, Any], variant: str) -> dict[str, Any]:
    ordinary, _measurements, snapshot, validation = core_snapshot_from_events(
        workload, variant=variant, run_id="lower-strictness", variant_tag="baseline"
    )
    native = evaluate_native_nx(workload, variant=variant)
    candidate = project_snapshot_to_nx(snapshot, validation)
    direct_lower = evaluate_direct_lower_domains(workload, variant=variant)
    if native["outputs"] != candidate["outputs"]:
        raise ValueError("strictness execution has Native/Candidate N[X] drift")
    return {
        "ordinary_rows": json.loads(ordinary)["rows"],
        "snapshot_id": snapshot.snapshot_id,
        "nx_outputs": native["outputs"],
        "lower_domains": {item["domain_id"]: item["outputs"] for item in direct_lower["domains"]},
    }


def evaluate_lower_strictness() -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = []
    for pair_id, workload, left_variant, right_variant in PAIR_SPECS:
        left = _execute(workload, left_variant)
        right = _execute(workload, right_variant)
        same_domains = sorted(
            domain_id for domain_id in left["lower_domains"]
            if left["lower_domains"][domain_id] == right["lower_domains"][domain_id]
        )
        nx_different = left["nx_outputs"] != right["nx_outputs"]
        gamma_different = left["snapshot_id"] != right["snapshot_id"]
        pairs.append(
            {
                "pair_id": pair_id,
                "workload_id": workload["id"],
                "left_variant": left_variant,
                "right_variant": right_variant,
                "real_execution_count": 2,
                "ordinary_rows_equal": left["ordinary_rows"] == right["ordinary_rows"],
                "native_candidate_exact_both_sides": True,
                "nx_different": nx_different,
                "gamma_snapshot_ids_different": gamma_different,
                "same_lower_domains": same_domains,
                "left_nx": left["nx_outputs"],
                "right_nx": right["nx_outputs"],
                "supported": nx_different and gamma_different,
            }
        )
    requirements = {
        "same_which_vars": any("flat_source_support_view" in pair["same_lower_domains"] and pair["nx_different"] for pair in pairs),
        "same_bag_naturals": any("bag_naturals" in pair["same_lower_domains"] and pair["nx_different"] for pair in pairs),
        "same_boolean": any("boolean" in pair["same_lower_domains"] and pair["nx_different"] for pair in pairs),
        "same_flat_source_support_view": any("flat_source_support_view" in pair["same_lower_domains"] and pair["nx_different"] for pair in pairs),
        "same_positive_boolean_lineage": any("positive_boolean_lineage" in pair["same_lower_domains"] and pair["nx_different"] for pair in pairs),
    }
    required_domains = {"bag_naturals", "boolean", "flat_source_support_view", "positive_boolean_lineage"}
    joint_pairs = [pair for pair in pairs if required_domains <= set(pair["same_lower_domains"]) and pair["nx_different"]]
    supported = all(requirements.values()) and all(pair["supported"] for pair in pairs)
    lower = {
        "schema_version": "lower-projection-strictness-constructions-v1",
        "claim": "Each evaluated lower projection forgets distinctions retained by N[X]",
        "status": "LOWER_PROJECTION_STRICTNESS_SUPPORTED" if supported else "NOT_ESTABLISHED",
        "requirements": requirements,
        "pair_count": len(pairs),
        "real_execution_count": sum(pair["real_execution_count"] for pair in pairs),
        "pairs": pairs,
    }
    joint = {
        "schema_version": "joint-lower-projection-strictness-v1",
        "claim": "The joint tuple of three algebraic targets and the flat-support task projection is non-injective with respect to N[X]",
        "status": "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED" if joint_pairs else "NOT_ESTABLISHED",
        "evaluated_domains": sorted(required_domains),
        "joint_witness_count": len(joint_pairs),
        "joint_witness_pair_ids": [pair["pair_id"] for pair in joint_pairs],
        "honesty_rule": "NOT_ESTABLISHED would be emitted if no real joint witness existed",
    }
    return lower, joint


def build_unification_artifacts(artifact_root: Path, lower: dict[str, Any], joint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def read(name: str) -> dict[str, Any]:
        return json.loads((artifact_root / name).read_text(encoding="utf-8"))

    gates = {
        "p1_exact_nx": read("nx_exact_comparison.json")["status"] == "EXACT_SUPPORTED",
        "independent_native_candidate_exact": read("native_candidate_nx_exact_comparison_v2.json")["status"] == "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED",
        "algebra_independence": read("native_candidate_algebra_independence.json")["status"] == "NATIVE_CANDIDATE_ALGEBRA_INDEPENDENCE_SUPPORTED",
        "p2_gamma_strictness": read("nx_strictness_counterexamples.json")["status"] == "STRICTNESS_SUPPORTED",
        "p3_classified_projection_hierarchy": read("hierarchical_projection_exact_comparison.json")["status"] == "FORMAL_PROJECTION_HIERARCHY_EXACT_SUPPORTED",
        "three_formal_algebraic_targets": read("formal_target_semantics_audit.json")["formal_algebraic_target_count"] >= 3,
        "flat_support_classified_as_task_projection": read("flat_support_view_formal_classification.json")["classification"] == "PARTIAL_NONZERO_SUPPORT_VIEW",
        "flat_support_exact": read("flat_support_view_exact_comparison.json")["status"] == "FLAT_SOURCE_SUPPORT_VIEW_EXACT_PROJECTION_SUPPORTED",
        "existing_database_which": read("nx_to_existing_which_lineage.json")["status"] == "THREE_WAY_EXACT_SUPPORTED",
        "lower_strictness": lower["status"] == "LOWER_PROJECTION_STRICTNESS_SUPPORTED",
        "joint_lower_strictness": joint["status"] == "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED",
    }
    hierarchy = {
        "schema_version": "two-level-unification-hierarchy-v2",
        "status": "TWO_LEVEL_FORMAL_HIERARCHY_SUPPORTED" if all(gates.values()) else "NOT_ESTABLISHED",
        "levels": [
            {
                "level": 0,
                "domain": "complete Core v3 generation facts Gamma_G(omega)",
                "outgoing_arrow": "exact strict projection",
            },
            {
                "level": 1,
                "domain": "canonical provenance polynomial N[X]",
                "incoming_arrow": "exact strict projection from Gamma_G(omega)",
            },
            {
                "level": "2A",
                "classification": "formal algebraic targets",
                "incoming_arrow": "exact semiring homomorphism or quotient projection from N[X]",
                "domains": ["bag N", "Boolean B", "PosBool(X)"],
            },
            {
                "level": "2B",
                "classification": "non-semiring task projections",
                "incoming_arrow": "exact task projection from N[X]",
                "domains": ["flat source-support view", "Vars(N[X])", "existing Database which-lineage"],
            },
        ],
        "forbidden_conflation": "Level 2A and Level 2B are not jointly described as semiring homomorphic images",
        "not_evaluated": ["Which(X)", "Trio(X)", "witness-basis Why provenance"],
        "gates": gates,
    }
    result = {
        "schema_version": "unification-of-unification-result-v2",
        "status": "UNIFICATION_OF_UNIFICATION_FORMAL_BOUNDARY_SUPPORTED" if all(gates.values()) else "NOT_ESTABLISHED",
        "statement": "Complete generation facts exactly and strictly project to N[X]; N[X] then has exact algebraic projections to three formally verified targets and separate exact task projections to flat support and existing which-lineage.",
        "established_arrows": [
            "Gamma_G(omega) -- exact strict projection --> N[X]",
            "N[X] -- exact algebraic projection --> bag N",
            "N[X] -- exact algebraic projection --> Boolean B",
            "N[X] -- exact algebraic projection --> PosBool(X)",
            "N[X] -- exact task projection --> flat source-support view",
            "N[X] -- exact task projection --> Vars(N[X]) --> existing Database which-lineage",
        ],
        "boundary": "No task projection is counted toward the formal semiring-target threshold.",
        "gates": gates,
        "failed_gates": sorted(name for name, value in gates.items() if not value),
    }
    w3c = {
        "schema_version": "nx-w3c-relation-scope-v1",
        "evaluation_status": "NOT_EVALUATED",
        "status": "NOT_EVALUATED",
        "reason": "This experiment evaluates semiring provenance under finite positive relational algebra; it does not evaluate or claim equivalence with W3C PROV-DM generation qualified-relation semantics.",
        "claim_escalation_permitted": False,
    }
    return hierarchy, result, w3c
