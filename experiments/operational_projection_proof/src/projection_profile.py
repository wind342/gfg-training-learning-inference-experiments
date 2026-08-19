from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ProjectionProofError


EVALUATION_STATUSES = frozenset({"EVALUATED", "NOT_EVALUATED"})
PROOF_STATUSES = frozenset(
    {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "NOT_SUPPORTED",
        "NOT_ESTABLISHED",
        "NOT_EVALUATED",
    }
)


@dataclass(frozen=True)
class ProjectionProfile:
    profile_id: str
    domain: str
    evaluation_status: str
    claim_scope: str
    prerequisite: str
    sections: dict[str, tuple[str, ...]]
    multiplicity_fields: frozenset[str]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ProjectionProfile":
        required = {
            "profile_id",
            "domain",
            "evaluation_status",
            "claim_scope",
            "prerequisite",
            "sections",
            "multiplicity_fields",
        }
        if set(value) != required:
            raise ProjectionProofError(
                "PROFILE_SCHEMA_INVALID", str(sorted(set(value) ^ required))
            )
        status = value["evaluation_status"]
        if status not in EVALUATION_STATUSES:
            raise ProjectionProofError(
                "PROFILE_SCHEMA_INVALID", f"evaluation_status={status}"
            )
        sections = value["sections"]
        if not isinstance(sections, dict) or not sections:
            raise ProjectionProofError("PROFILE_SCHEMA_INVALID", "sections")
        normalized_sections: dict[str, tuple[str, ...]] = {}
        for name, keys in sections.items():
            if not isinstance(name, str) or not isinstance(keys, list) or not keys:
                raise ProjectionProofError(
                    "PROFILE_SCHEMA_INVALID", f"section={name!r}"
                )
            if len(keys) != len(set(keys)) or not all(
                isinstance(item, str) and item for item in keys
            ):
                raise ProjectionProofError(
                    "PROFILE_SCHEMA_INVALID", f"identity_fields={name}"
                )
            normalized_sections[name] = tuple(keys)
        return cls(
            profile_id=value["profile_id"],
            domain=value["domain"],
            evaluation_status=status,
            claim_scope=value["claim_scope"],
            prerequisite=value["prerequisite"],
            sections=normalized_sections,
            multiplicity_fields=frozenset(value["multiplicity_fields"]),
        )


def load_profile(path: Path) -> ProjectionProfile:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionProofError("PROFILE_SCHEMA_INVALID", path.as_posix()) from exc
    if not isinstance(value, dict):
        raise ProjectionProofError("PROFILE_SCHEMA_INVALID", path.as_posix())
    return ProjectionProfile.from_mapping(value)


def require_truthful_status(
    *, prerequisite_satisfied: bool, requested_status: str
) -> None:
    if requested_status not in PROOF_STATUSES:
        raise ProjectionProofError("STATUS_INVALID", requested_status)
    if not prerequisite_satisfied and requested_status in {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
    }:
        raise ProjectionProofError("UNSUPPORTED_STATUS_ESCALATION", requested_status)
