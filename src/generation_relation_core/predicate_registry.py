from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, SchemaError

from .canonical import canonical_set, verify_entity
from .errors import CoreV3Error


Predicate = Callable[[dict, dict, str], bool]


def implementation_sha256(function: Predicate) -> str:
    source = inspect.getsource(function).replace("\r\n", "\n").replace("\r", "\n")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1:
        raise CoreV3Error("IMPLEMENTATION_HASH_MISMATCH", function.__name__)
    segment = ast.get_source_segment(source, functions[0])
    if segment is None:
        raise CoreV3Error("IMPLEMENTATION_HASH_MISMATCH", function.__name__)
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegisteredPredicate:
    profile: dict
    support_space: dict
    function: Predicate
    support_validator: Draft202012Validator
    query_validator: Draft202012Validator


class PredicateRegistry:
    """Explicit profile-ID dispatch. It has no default and no adapter/domain name branches."""

    def __init__(
        self,
        support_spaces: list[dict],
        profiles: list[dict],
        implementations: dict[str, Predicate],
    ) -> None:
        spaces = {}
        for row in support_spaces:
            verify_entity("SupportSpaceRecord", row)
            if row["support_space_id"] in spaces:
                raise CoreV3Error("DUPLICATE_ENTITY_ID", "SupportSpaceRecord")
            try:
                Draft202012Validator.check_schema(row["support_payload_schema"])
                Draft202012Validator.check_schema(row["query_payload_schema"])
            except SchemaError as exc:
                raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", row["support_space_id"]) from exc
            spaces[row["support_space_id"]] = row
        if set(implementations) != {row["predicate_profile_id"] for row in profiles}:
            raise CoreV3Error("PREDICATE_PROFILE_UNKNOWN", "IMPLEMENTATION_SET_MISMATCH")
        self._registered: dict[str, RegisteredPredicate] = {}
        self.invocation_counts: dict[str, int] = {}
        for profile in profiles:
            verify_entity("PredicateProfile", profile)
            profile_id = profile["predicate_profile_id"]
            if profile_id in self._registered:
                raise CoreV3Error("DUPLICATE_ENTITY_ID", "PredicateProfile")
            if not profile["authorized"]:
                raise CoreV3Error("PREDICATE_PROFILE_UNAUTHORIZED", profile_id)
            space = spaces.get(profile["support_space_id"])
            if space is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", profile["support_space_id"])
            function = implementations[profile_id]
            if function.__name__ != profile["implementation_symbol"]:
                raise CoreV3Error("IMPLEMENTATION_HASH_MISMATCH", profile_id)
            if implementation_sha256(function) != profile["predicate_implementation_sha256"]:
                raise CoreV3Error("IMPLEMENTATION_HASH_MISMATCH", profile_id)
            self._registered[profile_id] = RegisteredPredicate(
                profile, space, function,
                Draft202012Validator(space["support_payload_schema"]),
                Draft202012Validator(space["query_payload_schema"]),
            )
            self.invocation_counts[profile_id] = 0

    def resolve(self, profile_id: str) -> RegisteredPredicate:
        try:
            return self._registered[profile_id]
        except KeyError as exc:
            raise CoreV3Error("PREDICATE_PROFILE_UNKNOWN", profile_id) from exc

    def validate_support(self, profile_id: str, support: dict) -> None:
        registered = self.resolve(profile_id)
        if support.get("support_space_id") != registered.support_space["support_space_id"]:
            raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", profile_id)
        try:
            registered.support_validator.validate(support["support_payload"])
        except ValidationError as exc:
            raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", profile_id) from exc
        self._validate_registered_normalization(
            registered.support_space["support_payload_schema"], support["support_payload"], profile_id,
        )

    def validate_query(self, profile_id: str, predicate: str, query_payload: dict) -> None:
        registered = self.resolve(profile_id)
        if predicate not in registered.profile["supported_predicates"]:
            raise CoreV3Error("PREDICATE_PROFILE_UNAUTHORIZED", f"{profile_id}:{predicate}")
        try:
            registered.query_validator.validate(query_payload)
        except ValidationError as exc:
            raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", profile_id) from exc
        self._validate_registered_normalization(
            registered.support_space["query_payload_schema"], query_payload, profile_id,
        )

    def evaluate(self, profile_id: str, support: dict, query_payload: dict, predicate: str) -> bool:
        self.validate_support(profile_id, support)
        self.validate_query(profile_id, predicate, query_payload)
        registered = self.resolve(profile_id)
        self.invocation_counts[profile_id] += 1
        return bool(registered.function(support["support_payload"], query_payload, predicate))

    @property
    def profile_ids(self) -> frozenset[str]:
        return frozenset(self._registered)

    @classmethod
    def _validate_registered_normalization(cls, schema: dict, value: object, profile_id: str) -> None:
        if schema.get("x-sidecar-array-kind") == "set":
            if not isinstance(value, list) or value != canonical_set(value):
                raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", f"NON_CANONICAL_SET:{profile_id}")
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            for key, child in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    cls._validate_registered_normalization(child_schema, child, profile_id)
        elif isinstance(value, list) and isinstance(schema.get("items"), dict):
            for child in value:
                cls._validate_registered_normalization(schema["items"], child, profile_id)
