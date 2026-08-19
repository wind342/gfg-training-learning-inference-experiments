"""Sidecar Core v3: authoritative generation-relation identities."""

from .canonical import canonical_bytes, finalize_entity, table_hash, verify_entity
from .errors import CoreV3Error
from .snapshots import CoreV3Tables, ValidatedSnapshot, build_snapshot, validate_snapshot

__all__ = [
    "CoreV3Error",
    "CoreV3Tables",
    "ValidatedSnapshot",
    "build_snapshot",
    "canonical_bytes",
    "finalize_entity",
    "table_hash",
    "validate_snapshot",
    "verify_entity",
]
