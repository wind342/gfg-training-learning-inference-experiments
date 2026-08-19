from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable
import weakref

import torch


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().tolist()
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor.detach()).all()),
        "shape": list(tensor.shape),
        "value": value,
        "value_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
    }


def gradient_descriptors(
    gradients: tuple[torch.Tensor | None, ...],
) -> list[dict[str, Any]]:
    return [
        {"slot": slot, "tensor": None if value is None else tensor_descriptor(value)}
        for slot, value in enumerate(gradients)
    ]


def frozen_perturbation(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    value = tensor.detach().clone()
    if kind == "additive_pattern":
        if value.numel() == 0:
            return value
        pattern = torch.linspace(
            0.125,
            0.375,
            value.numel(),
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape)
        return value + pattern
    if kind == "multiplicative_scaling":
        return value * 1.75
    if kind == "sign_mask_crossing":
        return -value + torch.full_like(value, 0.125)
    raise ValueError(f"UNKNOWN_FROZEN_PERTURBATION:{kind}")


FROZEN_PERTURBATIONS = (
    "additive_pattern",
    "multiplicative_scaling",
    "sign_mask_crossing",
)


@dataclass(frozen=True)
class PackedTensor:
    token_key: str
    value: torch.Tensor


@dataclass
class _PackEntry:
    token_key: str
    pack_ordinal: int
    occurrence: int
    stage: str
    descriptor: dict[str, Any]
    stable_tensor_ref: str | None
    live_tensor: weakref.ReferenceType[torch.Tensor] | None
    unpack_count: int = 0


class EventClock:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, event: str, **payload: Any) -> int:
        ordinal = len(self.rows)
        self.rows.append({"event": event, "ordinal": ordinal, **payload})
        return ordinal


class SavedTensorObserver:
    """Observe and optionally intervene through the public saved-tensor hooks."""

    def __init__(
        self,
        clock: EventClock,
        *,
        intervention_token: str | None = None,
        perturbation: str | None = None,
        source_replay_ref: str | None = None,
        baseline_token_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.clock = clock
        self.intervention_token = intervention_token
        self.perturbation = perturbation
        self.source_replay_ref = source_replay_ref
        self.baseline_token_values = baseline_token_values
        self._stage_getter: Callable[[], str] | None = None
        self._stage_counts: dict[str, int] = {}
        self._entries: list[_PackEntry] = []
        self._registered: list[tuple[weakref.ReferenceType[torch.Tensor], str]] = []
        self._sources: dict[str, dict[str, Any]] = {}
        self._tensor_registrations: list[dict[str, Any]] = []
        self._current_node_id: str | None = None
        self._current_node_type: str | None = None
        self._unpacks: list[dict[str, Any]] = []
        self._intervention_applications: list[dict[str, Any]] = []

    def context(self, stage_getter: Callable[[], str]) -> AbstractContextManager[None]:
        self._stage_getter = stage_getter
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def _lookup_ref(self, tensor: torch.Tensor) -> str | None:
        for live, stable_ref in self._registered:
            if live() is tensor:
                return stable_ref
        return None

    def tensor_registered(self, tensor: torch.Tensor, stable_ref: str, stage: str) -> None:
        existing = self._lookup_ref(tensor)
        if existing is not None and existing != stable_ref:
            raise RuntimeError("SAVED_OBSERVER_TENSOR_REF_CONFLICT")
        if existing is None:
            self._registered.append((weakref.ref(tensor), stable_ref))
        descriptor = tensor_descriptor(tensor)
        registration = {
            "stable_tensor_ref": stable_ref,
            "stage": stage,
            "tensor": descriptor,
        }
        if registration not in self._tensor_registrations:
            self._tensor_registrations.append(registration)
        for entry in self._entries:
            if entry.stable_tensor_ref is None and entry.live_tensor is not None:
                if entry.live_tensor() is tensor:
                    entry.stable_tensor_ref = stable_ref
                    entry.live_tensor = None

    def stable_ref_for_tensor(self, tensor: torch.Tensor) -> str | None:
        return self._lookup_ref(tensor)

    def source_registered(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        *,
        source_identity: str,
        source_role: str,
        version: str,
        selected: bool,
    ) -> None:
        self._sources[stable_ref] = {
            "selected": selected,
            "source_identity": source_identity,
            "source_ref": stable_ref,
            "source_role": source_role,
            "tensor": tensor_descriptor(tensor),
            "version": version,
        }

    def enter_node(self, node_id: str, node_type: str) -> None:
        if self._current_node_id is not None:
            raise RuntimeError("NESTED_NATIVE_NODE_EXECUTION_UNEXPECTED")
        self._current_node_id = node_id
        self._current_node_type = node_type

    def exit_node(self, node_id: str) -> None:
        if self._current_node_id != node_id:
            raise RuntimeError("NATIVE_NODE_EXECUTION_CONTEXT_MISMATCH")
        self._current_node_id = None
        self._current_node_type = None

    def _pack(self, tensor: torch.Tensor) -> PackedTensor:
        if self._stage_getter is None:
            raise RuntimeError("SAVED_TENSOR_STAGE_GETTER_MISSING")
        stage = self._stage_getter()
        occurrence = self._stage_counts.get(stage, 0)
        self._stage_counts[stage] = occurrence + 1
        token_key = f"saved:{stage}:occurrence:{occurrence}"
        stable_ref = self._lookup_ref(tensor)
        entry = _PackEntry(
            token_key=token_key,
            pack_ordinal=len(self._entries),
            occurrence=occurrence,
            stage=stage,
            descriptor=tensor_descriptor(tensor),
            stable_tensor_ref=stable_ref,
            live_tensor=None if stable_ref is not None else weakref.ref(tensor),
        )
        self._entries.append(entry)
        self.clock.record(
            "saved_tensor_pack",
            stable_tensor_ref=stable_ref,
            stage=stage,
            token_key=token_key,
        )
        return PackedTensor(token_key, tensor.detach().clone())

    def _entry(self, token_key: str) -> _PackEntry:
        matches = [entry for entry in self._entries if entry.token_key == token_key]
        if len(matches) != 1:
            raise RuntimeError(f"SAVED_TENSOR_TOKEN_CARDINALITY:{token_key}:{len(matches)}")
        return matches[0]

    def _unpack(self, packed: PackedTensor) -> torch.Tensor:
        entry = self._entry(packed.token_key)
        entry.unpack_count += 1
        result = packed.value.detach().clone()
        intervention_kind = None
        if self.intervention_token == packed.token_key:
            if self.perturbation is None:
                raise RuntimeError("SAVED_TENSOR_PERTURBATION_MISSING")
            result = frozen_perturbation(result, self.perturbation)
            intervention_kind = self.perturbation
        elif self.source_replay_ref is not None:
            if self.baseline_token_values is None:
                raise RuntimeError("SOURCE_REPLAY_BASELINE_TOKENS_MISSING")
            if entry.stable_tensor_ref != self.source_replay_ref:
                if packed.token_key not in self.baseline_token_values:
                    raise RuntimeError(f"SOURCE_REPLAY_TOKEN_MISSING:{packed.token_key}")
                result = self.baseline_token_values[packed.token_key].detach().clone()
            else:
                intervention_kind = "registered_source_replay"
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("SAVED_TENSOR_INTERVENTION_NONFINITE")
        if entry.stable_tensor_ref is not None:
            self.tensor_registered(result, entry.stable_tensor_ref, entry.stage)
        unpack_ordinal = self.clock.record(
            "saved_tensor_unpack",
            native_node_id=self._current_node_id,
            token_key=packed.token_key,
        )
        row = {
            "assigned": self._current_node_id is not None,
            "intervention_kind": intervention_kind,
            "native_node_id": self._current_node_id,
            "native_node_type": self._current_node_type,
            "result": tensor_descriptor(result),
            "stable_tensor_ref": entry.stable_tensor_ref,
            "token_key": packed.token_key,
            "unpack_ordinal": unpack_ordinal,
            "unpack_sequence_for_token": entry.unpack_count - 1,
        }
        self._unpacks.append(row)
        if intervention_kind is not None:
            self._intervention_applications.append(row)
        return result

    def packed_values(self) -> dict[str, torch.Tensor]:
        """Return detached copies reconstructed from the immutable pack descriptors."""
        result: dict[str, torch.Tensor] = {}
        for entry in self._entries:
            value = torch.tensor(entry.descriptor["value"], dtype=torch.float64)
            value = value.reshape(entry.descriptor["shape"])
            result[entry.token_key] = value
        return result

    def export(self) -> dict[str, Any]:
        packs = []
        for entry in self._entries:
            stable_ref = entry.stable_tensor_ref or f"support:{entry.token_key}"
            packs.append({
                "occurrence": entry.occurrence,
                "pack_ordinal": entry.pack_ordinal,
                "stable_tensor_ref": stable_ref,
                "stage": entry.stage,
                "tensor": entry.descriptor,
                "token_key": entry.token_key,
                "unpack_count": entry.unpack_count,
            })
        return {
            "intervention_applications": self._intervention_applications,
            "pack_trace": packs,
            "registered_sources": [self._sources[key] for key in sorted(self._sources)],
            "tensor_registrations": self._tensor_registrations,
            "unpack_trace": self._unpacks,
        }
