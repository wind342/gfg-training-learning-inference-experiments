from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from experiments.operational_projection_proof.scripts.run_all import (
    PROFILE_ROOT as EXISTING_PROFILE_ROOT,
    _business_projection,
    _many_to_many_projection,
)
from experiments.operational_projection_proof.src.database_reference import (
    business_oracle_result,
    many_to_many_oracle_result,
)
from experiments.operational_projection_proof.src.projection_profile import load_profile
from experiments.operational_projection_proof.src.projection_result import combine_results

from .nx_polynomial import NXPolynomial
from .structural import variable_for_source


FROZEN_DATABASE_COMMIT = "03caa31b8a6abfe6e112a0544071618c689bb11f"
FROZEN_AUTHORITY_PROFILE_SHA256 = "d32b809931644617d763bad597b62904bba3273e7fa62f9afa963fc74387ac40"
FROZEN_EXECUTION_PROFILE_SHA256 = "518ba0f6de62fc4eb53ae6afb1fcf40177c6483c090a63be37cbbb119a4fcce9"
FROZEN_EXISTING_RESULT_SHA256 = "337b92bd88f11a51fe88d6dc2c47896c2bed731470016cd503b7fc2b44b4b8f9"
FROZEN_EXISTING_CANONICAL_SHA256 = "a54037abf452d7308f13a27a287b19a3797b5e9ab77bd62efbb48c1a81672360"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _project_existing_database_snapshot_to_nx(snapshot: ValidatedSnapshot, validation: SnapshotValidation) -> list[dict[str, Any]]:
    """Apply N[X] recursion to the frozen Database Core snapshot only."""
    if validation.snapshot_id != snapshot.snapshot_id:
        raise ValueError("Database validation does not belong to snapshot")
    tables = snapshot.tables
    sources = {row["source_information_id"]: row for row in tables.source_information_records}
    generated = {row["generated_origin_id"]: row for row in tables.generated_origins}
    supports = {row["support_id"]: row for row in tables.perceptual_support_records}
    bindings_by_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            bindings_by_support[outcome["support_id"]].append(binding)
    consumed_supports = {
        row["origin_payload"]["prior_support_id"]
        for row in generated.values()
        if row.get("origin_payload", {}).get("bridge_kind") == "support_to_generated_origin"
    }
    terminal_supports = sorted(set(supports) - consumed_supports)
    memo: dict[str, NXPolynomial] = {}
    visiting: set[str] = set()

    def origin_polynomial(origin: dict[str, str]) -> NXPolynomial:
        if origin["kind"] == "registered_source":
            source = sources[origin["source_information_id"]]
            return NXPolynomial.variable(variable_for_source(source["source_identity"]))
        if origin["kind"] == "generated_origin":
            record = generated[origin["generated_origin_id"]]
            payload = record["origin_payload"]
            if payload.get("bridge_kind") != "support_to_generated_origin":
                raise ValueError("existing Database GeneratedOrigin lacks prior support bridge")
            return support_polynomial(payload["prior_support_id"])
        raise ValueError("unknown Database Core origin")

    def support_polynomial(support_id: str) -> NXPolynomial:
        if support_id in memo:
            return memo[support_id]
        if support_id in visiting:
            raise ValueError("cycle in existing Database support graph")
        visiting.add(support_id)
        bindings = bindings_by_support.get(support_id, [])
        if not bindings or len({row["generation_occurrence_id"] for row in bindings}) != 1:
            raise ValueError("existing Database support has invalid inbound occurrence closure")
        result = NXPolynomial.product(origin_polynomial(row["origin_reference"]) for row in bindings)
        visiting.remove(support_id)
        memo[support_id] = result
        return result

    projected = []
    for support_id in terminal_supports:
        payload = supports[support_id]["support_payload"]
        tuple_identity = payload["tuple_identity"]
        polynomial = support_polynomial(support_id)
        source_by_variable = {
            variable_for_source(row["source_identity"]): row["source_identity"]
            for row in sources.values()
        }
        projected.append(
            {
                "output_tuple_id": tuple_identity,
                "polynomial": polynomial.to_document(),
                "variables": polynomial.variables(),
                "source_tuple_ids_from_variables": sorted(source_by_variable[variable] for variable in polynomial.variables()),
            }
        )
    return projected


def evaluate_existing_database_which_bridge(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = EXISTING_PROFILE_ROOT / "database_which_lineage_v1.json"
    authority_profile_path = repo_root / "experiments" / "operational_projection_proof_v2" / "profiles" / "database_which_lineage_v1.json"
    existing_result_path = repo_root / "experiments" / "operational_projection_proof_v2" / "artifacts" / "projection_equivalence_database.json"
    execution_profile_sha = _sha256_bytes(profile_path.read_bytes())
    authority_profile_sha = _sha256_bytes(authority_profile_path.read_bytes())
    existing_result_sha = _sha256_bytes(existing_result_path.read_bytes())
    profile = load_profile(profile_path)
    _adapter, _rows, business_snapshot, business_validation, business_candidate = _business_projection(
        profile=profile, run_id="provenance-semiring-existing-database-bridge"
    )
    _m2m_adapter, m2m_snapshot, m2m_validation, m2m_candidate = _many_to_many_projection(profile=profile)
    existing_candidate = combine_results(profile, business_candidate, m2m_candidate)
    existing_native = combine_results(profile, business_oracle_result(profile), many_to_many_oracle_result(profile))
    existing_exact = canonical_bytes(existing_candidate) == canonical_bytes(existing_native)
    existing_candidate_sha = _canonical_sha(existing_candidate)
    existing_native_sha = _canonical_sha(existing_native)
    nx_outputs = _project_existing_database_snapshot_to_nx(business_snapshot, business_validation)
    nx_outputs.extend(_project_existing_database_snapshot_to_nx(m2m_snapshot, m2m_validation))
    nx_by_output = {item["output_tuple_id"]: item for item in nx_outputs}
    native_backward = {row["output_tuple_id"]: row for row in existing_native["records"]["backward_lineage"]}
    candidate_backward = {row["output_tuple_id"]: row for row in existing_candidate["records"]["backward_lineage"]}
    compared_outputs = []
    for output_tuple_id in sorted(set(native_backward) | set(candidate_backward)):
        native_sources = native_backward.get(output_tuple_id, {}).get("source_tuple_ids")
        candidate_sources = candidate_backward.get(output_tuple_id, {}).get("source_tuple_ids")
        nx_record = nx_by_output.get(output_tuple_id)
        nx_sources = None if nx_record is None else nx_record["source_tuple_ids_from_variables"]
        exact = native_sources == candidate_sources == nx_sources
        compared_outputs.append(
            {
                "output_tuple_id": output_tuple_id,
                "native_existing_source_tuple_ids": native_sources,
                "core_candidate_existing_source_tuple_ids": candidate_sources,
                "nx_variables": None if nx_record is None else nx_record["variables"],
                "nx_source_tuple_ids": nx_sources,
                "three_way_exact": exact,
            }
        )
    three_way_exact = bool(compared_outputs) and all(item["three_way_exact"] for item in compared_outputs)
    frozen_authority_exact = (
        authority_profile_sha == FROZEN_AUTHORITY_PROFILE_SHA256
        and execution_profile_sha == FROZEN_EXECUTION_PROFILE_SHA256
        and existing_result_sha == FROZEN_EXISTING_RESULT_SHA256
    )
    bridge = {
        "schema_version": "nx-to-existing-which-lineage-v1",
        "status": "THREE_WAY_EXACT_SUPPORTED" if existing_exact and three_way_exact and frozen_authority_exact else "NOT_ESTABLISHED",
        "frozen_database_source_commit": FROZEN_DATABASE_COMMIT,
        "frozen_authority_profile_sha256": authority_profile_sha,
        "frozen_authority_profile_sha256_exact": authority_profile_sha == FROZEN_AUTHORITY_PROFILE_SHA256,
        "frozen_execution_profile_sha256": execution_profile_sha,
        "frozen_execution_profile_sha256_exact": execution_profile_sha == FROZEN_EXECUTION_PROFILE_SHA256,
        "frozen_existing_result_sha256": existing_result_sha,
        "frozen_existing_result_sha256_exact": existing_result_sha == FROZEN_EXISTING_RESULT_SHA256,
        "existing_native_candidate_all_records_exact": existing_exact,
        "existing_native_canonical_sha256": existing_native_sha,
        "existing_candidate_canonical_sha256": existing_candidate_sha,
        "frozen_expected_canonical_sha256": FROZEN_EXISTING_CANONICAL_SHA256,
        "existing_candidate_record_count": sum(len(rows) for rows in existing_candidate["records"].values()),
        "existing_native_record_count": sum(len(rows) for rows in existing_native["records"].values()),
        "nx_terminal_output_count": len(nx_outputs),
        "which_output_comparison_count": len(compared_outputs),
        "three_way_exact": three_way_exact,
        "comparisons": compared_outputs,
    }
    hierarchy = {
        "schema_version": "database-lineage-hierarchy-v1",
        "status": "DATABASE_WHICH_AS_NX_VARIABLE_PROJECTION_SUPPORTED" if bridge["status"] == "THREE_WAY_EXACT_SUPPORTED" else "NOT_ESTABLISHED",
        "hierarchy": [
            "Core v3 complete generation snapshot",
            "N[X] polynomial with coefficients and exponents",
            "Vars(N[X])",
            "existing Database tuple-level which-lineage source_tuple_ids",
        ],
        "existing_native_path": "frozen Database synthetic oracle and database_reference",
        "existing_core_candidate_path": "actual Database executor -> unmodified Core v3 -> project_database_snapshot",
        "nx_path": "same validated Database Core snapshot -> GeneratedOrigin recursion -> N[X] -> Vars",
        "three_way_exact": three_way_exact,
        "compared_output_count": len(compared_outputs),
        "scope_note": "The existing frozen many-to-many case has no backward_lineage final-output row; it remains covered by the existing 112-record exact comparison, while the three-way Vars comparison applies to every declared backward_lineage row.",
    }
    return bridge, hierarchy
