from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Iterable

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    read_json,
    require,
)


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    module_path: str
    parameter_prefix: str


@dataclass(frozen=True)
class ComponentRegistry:
    registry_id: str
    components: tuple[ComponentSpec, ...]
    source_sha256: str

    @classmethod
    def load(cls, path: Path) -> "ComponentRegistry":
        raw = read_json(path)
        require(raw["schema"] == "nanogpt-component-registry-v1", "SST_COMPONENT_REGISTRY_SCHEMA_INVALID")
        require(raw["status"].startswith("FROZEN_"), "SST_COMPONENT_REGISTRY_NOT_FROZEN")
        components = tuple(
            ComponentSpec(
                component_id=str(row["component_id"]),
                module_path=str(row["module_path"]),
                parameter_prefix=str(row["parameter_prefix"]),
            )
            for row in raw["components"]
        )
        require(bool(components), "SST_COMPONENT_REGISTRY_EMPTY")
        require(len({row.component_id for row in components}) == len(components), "SST_COMPONENT_ID_DUPLICATE")
        require(len({row.module_path for row in components}) == len(components), "SST_COMPONENT_MODULE_DUPLICATE")
        require(len({row.parameter_prefix for row in components}) == len(components), "SST_COMPONENT_PREFIX_DUPLICATE")
        return cls(str(raw["registry_id"]), components, file_sha256(path))

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(row.component_id for row in self.components)


@dataclass(frozen=True)
class ProbeContract:
    probe_contract_id: str
    component_registry_id: str
    baseline_repetitions: int
    derivation_profile: str
    gate_sets: tuple[tuple[str, ...], ...]
    target_group_count: int
    required_outputs: tuple[str, ...]
    source_sha256: str

    @classmethod
    def load(cls, path: Path, registry: ComponentRegistry) -> "ProbeContract":
        raw = read_json(path)
        require(raw["schema"] == "nanogpt-support-probe-contract-v1", "SST_PROBE_CONTRACT_SCHEMA_INVALID")
        require(raw["status"].startswith("FROZEN_"), "SST_PROBE_CONTRACT_NOT_FROZEN")
        require(str(raw["component_registry_id"]) == registry.registry_id, "SST_PROBE_REGISTRY_MISMATCH")
        require(re.fullmatch(r"[A-Za-z0-9._-]+", str(raw["probe_contract_id"])) is not None, "SST_PROBE_ID_UNSAFE")
        require(int(raw["baseline_repetitions"]) >= 2, "SST_PROBE_BASELINE_REPETITIONS_INSUFFICIENT")
        gate_sets = tuple(tuple(str(value) for value in row) for row in raw["gate_sets"])
        admitted = set(registry.component_ids)
        require(all(bool(row) and set(row) <= admitted for row in gate_sets), "SST_PROBE_GATE_UNKNOWN_COMPONENT")
        require(all(len(set(row)) == len(row) for row in gate_sets), "SST_PROBE_GATE_DUPLICATE_COMPONENT")
        require(len(set(gate_sets)) == len(gate_sets), "SST_PROBE_GATE_SET_DUPLICATE")
        derivation_profile = str(raw["derivation_profile"])
        if derivation_profile == "CSRG-support-metrics-v1":
            require(
                {row for row in gate_sets if len(row) == 1}
                == {(value,) for value in registry.component_ids},
                "SST_CSRG_SINGLE_GATES_INCOMPLETE",
            )
            require(
                {frozenset(row) for row in gate_sets if len(row) == 2}
                == {frozenset(row) for row in combinations(registry.component_ids, 2)},
                "SST_CSRG_PAIR_GATES_INCOMPLETE",
            )
            require(all(len(row) <= 2 for row in gate_sets), "SST_CSRG_UNSUPPORTED_GATE_ARITY")
        else:
            require(False, f"SST_PROBE_DERIVATION_PROFILE_UNSUPPORTED:{derivation_profile}")
        return cls(
            probe_contract_id=str(raw["probe_contract_id"]),
            component_registry_id=str(raw["component_registry_id"]),
            baseline_repetitions=int(raw["baseline_repetitions"]),
            derivation_profile=derivation_profile,
            gate_sets=gate_sets,
            target_group_count=int(raw["target_group_count"]),
            required_outputs=tuple(str(value) for value in raw["required_outputs"]),
            source_sha256=file_sha256(path),
        )

    def single_gates(self) -> tuple[tuple[str, ...], ...]:
        return tuple(row for row in self.gate_sets if len(row) == 1)

    def pair_gates(self) -> tuple[tuple[str, ...], ...]:
        return tuple(row for row in self.gate_sets if len(row) == 2)

    def validate_csrg4_compatibility(self, registry: ComponentRegistry) -> None:
        """Freeze the first contract without imposing four components on later contracts."""
        require(self.probe_contract_id == "CSRG-4C-v1", "SST_INITIAL_PROBE_ID_INVALID")
        require(self.baseline_repetitions == 2, "SST_INITIAL_BASELINE_COUNT_INVALID")
        require(set(self.single_gates()) == {(value,) for value in registry.component_ids}, "SST_INITIAL_SINGLE_GATES_INCOMPLETE")
        require(set(self.pair_gates()) == set(combinations(registry.component_ids, 2)), "SST_INITIAL_PAIR_GATES_INCOMPLETE")
        require(self.target_group_count == 23, "SST_INITIAL_TARGET_GROUP_COUNT_INVALID")


def resolve_module(root: Any, dotted_path: str) -> Any:
    current = root
    for token in dotted_path.split("."):
        current = current[int(token)] if token.isdigit() else getattr(current, token)
    return current


def component_parameter_names(
    named_parameters: Iterable[str], registry: ComponentRegistry
) -> dict[str, tuple[str, ...]]:
    names = tuple(named_parameters)
    result = {
        row.component_id: tuple(name for name in names if name.startswith(row.parameter_prefix))
        for row in registry.components
    }
    require(all(result.values()), "SST_COMPONENT_PARAMETER_SET_EMPTY")
    return result
