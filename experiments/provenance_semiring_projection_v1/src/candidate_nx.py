from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .nx_polynomial import NXPolynomial
from .profile_runtime import load_profile
from .structural import variable_for_source


class CandidateProjectionError(ValueError):
    pass


def project_snapshot_to_nx(snapshot: ValidatedSnapshot, validation: SnapshotValidation) -> dict[str, object]:
    """Project a validated Core snapshot without fixture, AST, Native, or answer access."""
    nx_profile = load_profile("provenance_semiring_nx_profile_v1.json")
    crosswalk = load_profile("core_to_nx_crosswalk_v1.json")
    if validation.snapshot_id != snapshot.snapshot_id:
        raise CandidateProjectionError("validation does not belong to snapshot")
    tables = snapshot.tables
    sources = {row["source_information_id"]: row for row in tables.source_information_records}
    generated = {row["generated_origin_id"]: row for row in tables.generated_origins}
    supports = {row["support_id"]: row for row in tables.perceptual_support_records}
    bindings_by_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            bindings_by_support[outcome["support_id"]].append(binding)

    memo: dict[str, NXPolynomial] = {}
    visiting: set[str] = set()

    def origin_polynomial(origin: dict[str, str]) -> NXPolynomial:
        kind = origin.get("kind")
        if kind == "registered_source":
            source = sources.get(origin.get("source_information_id", ""))
            if source is None:
                raise CandidateProjectionError("missing registered source")
            return NXPolynomial.variable(variable_for_source(source["source_identity"]))
        if kind == "generated_origin":
            record = generated.get(origin.get("generated_origin_id", ""))
            if record is None:
                raise CandidateProjectionError("missing GeneratedOrigin")
            payload = record.get("origin_payload")
            if not isinstance(payload, dict) or payload.get("bridge_kind") != "support_to_generated_origin":
                raise CandidateProjectionError("missing GeneratedOrigin support bridge")
            prior_support_id = payload.get("prior_support_id")
            if not isinstance(prior_support_id, str) or prior_support_id not in supports:
                raise CandidateProjectionError("missing prior support")
            return support_polynomial(prior_support_id)
        raise CandidateProjectionError(f"unknown origin kind: {kind!r}")

    def support_polynomial(support_id: str) -> NXPolynomial:
        if support_id in memo:
            return memo[support_id]
        if support_id in visiting:
            raise CandidateProjectionError("cycle in GeneratedOrigin recursion")
        visiting.add(support_id)
        bindings = bindings_by_support.get(support_id, [])
        if not bindings:
            raise CandidateProjectionError("support has zero inbound bindings")
        occurrence_ids = {binding["generation_occurrence_id"] for binding in bindings}
        if len(occurrence_ids) != 1:
            raise CandidateProjectionError("support has bindings from more than one occurrence")
        polynomial = NXPolynomial.product(origin_polynomial(binding["origin_reference"]) for binding in bindings)
        visiting.remove(support_id)
        memo[support_id] = polynomial
        return polynomial

    by_logical_output: dict[str, list[NXPolynomial]] = defaultdict(list)
    values_by_key: dict[str, dict[str, Any]] = {}
    for support_id, support in supports.items():
        payload = support["support_payload"]
        if not payload.get("terminal"):
            continue
        key = payload.get("logical_output_key")
        if not isinstance(key, str) or not key:
            raise CandidateProjectionError("noncanonical terminal logical output key")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise CandidateProjectionError("terminal support has no output values")
        prior_values = values_by_key.setdefault(key, values)
        if prior_values != values:
            raise CandidateProjectionError("logical output key aliases different values")
        by_logical_output[key].append(support_polynomial(support_id))
    if not by_logical_output:
        raise CandidateProjectionError("snapshot has no terminal support")

    source_variables = sorted(
        (
            {"variable": variable_for_source(row["source_identity"]), "source_identity": row["source_identity"]}
            for row in sources.values()
        ),
        key=lambda item: item["variable"],
    )
    return {
        "schema_version": "core-projected-nx-result-v1",
        "snapshot_id": snapshot.snapshot_id,
        "profile_id": nx_profile["profile_id"],
        "crosswalk_id": crosswalk["profile_id"],
        "source_variables": source_variables,
        "outputs": [
            {
                "logical_output_key": key,
                "values": values_by_key[key],
                "polynomial": NXPolynomial.sum(by_logical_output[key]).to_document(),
            }
            for key in sorted(by_logical_output)
        ],
    }
