from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .projection_errors import ProjectionError
from .snapshot_access import require_validated_snapshot


@dataclass(frozen=True)
class DatabaseSourceTuple:
    source_information_id: str
    tuple_identity: str
    table_identity: str
    source_payload_sha256: str


@dataclass(frozen=True)
class DatabaseProducedTuple:
    support_id: str
    tuple_identity: str
    operator_stage: str


@dataclass(frozen=True)
class DatabaseExclusion:
    disposition_id: str
    tuple_identity: str
    operator_stage: str
    operator_type: str
    reason_code: str


@dataclass(frozen=True)
class DatabaseGeneratedBridge:
    generated_origin_id: str
    prior_support_id: str
    tuple_identity: str


@dataclass(frozen=True)
class DatabaseOccurrence:
    generation_occurrence_id: str
    run_id: str
    occurrence_index: int
    occurrence_stage: str
    occurrence_type: str
    stable_instance_key: str
    transform_operator_type: str
    transform_stage: str


@dataclass(frozen=True)
class DatabaseBinding:
    generation_binding_id: str
    origin_kind: str
    origin_id: str
    occurrence_id: str
    outcome_kind: str
    outcome_id: str
    relation_role: str


@dataclass(frozen=True)
class DatabaseDomainProjection:
    run_id: str
    sources: tuple[DatabaseSourceTuple, ...]
    produced_tuples: tuple[DatabaseProducedTuple, ...]
    exclusions: tuple[DatabaseExclusion, ...]
    generated_bridges: tuple[DatabaseGeneratedBridge, ...]
    occurrences: tuple[DatabaseOccurrence, ...]
    bindings: tuple[DatabaseBinding, ...]


def _required_text(value: Any, detail: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectionError("DATABASE_PROJECTION_INVALID", detail)
    return value


def project_core_to_database(
    snapshot: ValidatedSnapshot,
    validation: SnapshotValidation,
) -> DatabaseDomainProjection:
    """Build an immutable, ephemeral database-domain view of Core facts."""

    require_validated_snapshot(snapshot, validation)
    tables = snapshot.tables
    occurrences: list[DatabaseOccurrence] = []
    run_ids: set[str] = set()
    for row in tables.generation_occurrences:
        transform = row["transform_reference"]
        payload = row["occurrence_payload"]
        run_id = _required_text(payload.get("run_id"), "RUN_ID")
        run_ids.add(run_id)
        occurrences.append(
            DatabaseOccurrence(
                generation_occurrence_id=row["generation_occurrence_id"],
                run_id=run_id,
                occurrence_index=row["occurrence_index"],
                occurrence_stage=row["occurrence_stage"],
                occurrence_type=row["occurrence_type"],
                stable_instance_key=row["stable_instance_key"],
                transform_operator_type=_required_text(
                    transform.get("operator_type"), "OPERATOR_TYPE"
                ),
                transform_stage=_required_text(transform.get("stage"), "STAGE"),
            )
        )
    if len(run_ids) != 1:
        raise ProjectionError("DATABASE_PROJECTION_INVALID", "RUN_ID_CARDINALITY")
    bindings: list[DatabaseBinding] = []
    for row in tables.generation_bindings:
        origin = row["origin_reference"]
        outcome = row["outcome_reference"]
        origin_id = (
            origin["source_information_id"]
            if origin["kind"] == "registered_source"
            else origin["generated_origin_id"]
        )
        outcome_id = (
            outcome["support_id"]
            if outcome["kind"] == "support"
            else outcome["disposition_id"]
        )
        bindings.append(
            DatabaseBinding(
                generation_binding_id=row["generation_binding_id"],
                origin_kind=origin["kind"],
                origin_id=origin_id,
                occurrence_id=row["generation_occurrence_id"],
                outcome_kind=outcome["kind"],
                outcome_id=outcome_id,
                relation_role=row["relation_role"],
            )
        )
    return DatabaseDomainProjection(
        run_id=next(iter(run_ids)),
        sources=tuple(
            DatabaseSourceTuple(
                source_information_id=row["source_information_id"],
                tuple_identity=row["source_identity"],
                table_identity=row["source_payload"]["table_identity"],
                source_payload_sha256=row["source_payload_sha256"],
            )
            for row in tables.source_information_records
        ),
        produced_tuples=tuple(
            DatabaseProducedTuple(
                support_id=row["support_id"],
                tuple_identity=row["support_payload"]["tuple_identity"],
                operator_stage=row["support_payload"]["operator_stage"],
            )
            for row in tables.perceptual_support_records
        ),
        exclusions=tuple(
            DatabaseExclusion(
                disposition_id=row["disposition_id"],
                tuple_identity=row["disposition_payload"]["tuple_identity"],
                operator_stage=row["disposition_payload"]["operator_stage"],
                operator_type=row["disposition_payload"]["operator_type"],
                reason_code=row["domain_reason_code"],
            )
            for row in tables.explicit_dispositions
        ),
        generated_bridges=tuple(
            DatabaseGeneratedBridge(
                generated_origin_id=row["generated_origin_id"],
                prior_support_id=_required_text(
                    row["origin_payload"].get("prior_support_id"), "PRIOR_SUPPORT"
                ),
                tuple_identity=_required_text(
                    row["origin_payload"].get("tuple_identity"), "TUPLE_IDENTITY"
                ),
            )
            for row in tables.generated_origins
        ),
        occurrences=tuple(occurrences),
        bindings=tuple(bindings),
    )
