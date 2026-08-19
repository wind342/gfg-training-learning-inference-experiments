from __future__ import annotations

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .projection_errors import ProjectionError


def require_validated_snapshot(
    snapshot: ValidatedSnapshot, validation: SnapshotValidation
) -> None:
    """Require the Core validator's proof without reimplementing validation."""

    if validation.snapshot_id != snapshot.snapshot_id:
        raise ProjectionError("SNAPSHOT_VALIDATION_REQUIRED", "SNAPSHOT_ID")
    binding_ids = {
        row["generation_binding_id"] for row in snapshot.tables.generation_bindings
    }
    if set(validation.relation_evidence) != binding_ids:
        raise ProjectionError("SNAPSHOT_VALIDATION_REQUIRED", "BINDING_COVERAGE")
