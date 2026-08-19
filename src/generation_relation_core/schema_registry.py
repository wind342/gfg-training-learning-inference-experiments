from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .errors import CoreV3Error


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = REPOSITORY_ROOT / "protocol" / "core_v3"


ENTITY_TYPES = frozenset({
    "EnvironmentRecord",
    "EvidenceLink",
    "EvidenceRecord",
    "ExplicitDisposition",
    "GeneratedOrigin",
    "GenerationBinding",
    "GenerationOccurrence",
    "GeneratorManifest",
    "GeneratorOperationResult",
    "HierarchyRecord",
    "LegacyOccurrenceBindingProjection",
    "LegacySourceBindingProjection",
    "MigrationRecord",
    "PerceptualSupportRecord",
    "PredicateProfile",
    "QueryHit",
    "QueryRequest",
    "QueryResult",
    "SourceInformationRecord",
    "SupportSpaceRecord",
    "ValidatedSnapshot",
})


@lru_cache(maxsize=None)
def load_json(name: str) -> dict[str, Any]:
    path = PROTOCOL_ROOT / name
    if not path.is_file():
        raise CoreV3Error("EXTERNAL_KEY_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def schema_for(entity_type: str) -> dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise CoreV3Error("UNKNOWN_ENTITY_TYPE", entity_type)
    root = load_json("core_v3_entities.schema.json")
    return {
        "$schema": root["$schema"],
        "$id": f"{root['$id']}:{entity_type}",
        "$ref": f"#/$defs/{entity_type}",
        "$defs": root["$defs"],
    }


def validate(entity_type: str, value: Any) -> None:
    try:
        validator_for(entity_type).validate(value)
    except ValidationError as exc:
        raise CoreV3Error("HASH_OR_ID_MISMATCH", f"SCHEMA:{entity_type}:{exc.json_path}") from exc


@lru_cache(maxsize=None)
def validator_for(entity_type: str) -> Draft202012Validator:
    return Draft202012Validator(schema_for(entity_type))


@lru_cache(maxsize=1)
def projection_registry() -> dict[str, dict[str, Any]]:
    return load_json("canonical_serialization_v3.json")["projection_registry"]


@lru_cache(maxsize=1)
def protocol() -> dict[str, Any]:
    return load_json("core_v3_protocol.json")
