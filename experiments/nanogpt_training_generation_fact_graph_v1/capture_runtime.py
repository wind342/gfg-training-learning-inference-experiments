from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterator
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "requires_grad": bool(tensor.requires_grad),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "version": int(tensor._version),
    }


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        result.append(value)
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.extend(_flatten_tensors(child))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            result.extend(_flatten_tensors(value[key]))
    return result


def _scalar_summary(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return {"summary": type(value).__name__}
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_descriptor(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        text = value if not isinstance(value, str) else value[:240]
        return text
    if isinstance(value, slice):
        return {
            "slice": [
                _scalar_summary(value.start, depth + 1),
                _scalar_summary(value.stop, depth + 1),
                _scalar_summary(value.step, depth + 1),
            ]
        }
    if isinstance(value, (tuple, list)):
        rows = [_scalar_summary(child, depth + 1) for child in value[:24]]
        if len(value) > 24:
            rows.append({"truncated_count": len(value) - 24})
        return rows
    if isinstance(value, dict):
        return {
            str(key): _scalar_summary(child, depth + 1)
            for key, child in list(sorted(value.items(), key=lambda item: str(item[0])))[:24]
        }
    return {"type": type(value).__name__, "repr": repr(value)[:240]}


@dataclass
class _TensorReference:
    live: weakref.ReferenceType[torch.Tensor]
    stable_ref: str


class TrainingFactRecorder(TorchDispatchMode):
    """Write-only synchronous recorder around actual ATen dispatch calls."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self._references: dict[int, _TensorReference] = {}
        self._ordinal = 0
        self._external_count = 0
        self.step = 0
        self.micro_step: int | None = None
        self.phase = "initialization"
        self.active = True

    @contextmanager
    def stage(
        self,
        *,
        step: int,
        micro_step: int | None,
        phase: str,
    ) -> Iterator[None]:
        previous = (self.step, self.micro_step, self.phase)
        self.step, self.micro_step, self.phase = step, micro_step, phase
        try:
            yield
        finally:
            self.step, self.micro_step, self.phase = previous

    @contextmanager
    def paused(self) -> Iterator[None]:
        previous = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = previous

    def _set_reference(self, tensor: torch.Tensor, stable_ref: str) -> None:
        self._references[id(tensor)] = _TensorReference(weakref.ref(tensor), stable_ref)

    def reference(self, tensor: torch.Tensor) -> str:
        entry = self._references.get(id(tensor))
        if entry is not None and entry.live() is tensor:
            return entry.stable_ref
        stable_ref = f"source:runtime_external:{self._external_count:06d}"
        self._external_count += 1
        self.register_source(
            tensor,
            stable_ref,
            {
                "source_kind": "runtime_tensor_first_observed",
                "tensor": tensor_descriptor(tensor),
            },
        )
        return stable_ref

    def register_source(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        source_payload: dict[str, Any],
    ) -> None:
        existing = self.sources.get(stable_ref)
        row = {
            "source_ref": stable_ref,
            "source_payload": source_payload,
        }
        if existing is not None and existing != row:
            raise RuntimeError(f"SOURCE_REF_CONFLICT:{stable_ref}")
        self.sources[stable_ref] = row
        self._set_reference(tensor, stable_ref)

    def register_literal_source(
        self,
        stable_ref: str,
        source_payload: dict[str, Any],
    ) -> None:
        row = {
            "source_ref": stable_ref,
            "source_payload": source_payload,
        }
        existing = self.sources.get(stable_ref)
        if existing is not None and existing != row:
            raise RuntimeError(f"SOURCE_REF_CONFLICT:{stable_ref}")
        self.sources[stable_ref] = row

    def _new_output(
        self,
        tensor: torch.Tensor,
        ordinal: int,
        output_index: int,
        prefix: str = "tensor",
    ) -> dict[str, Any]:
        stable_ref = (
            f"{prefix}:step:{self.step:03d}:micro:"
            f"{-1 if self.micro_step is None else self.micro_step:02d}:"
            f"event:{ordinal:06d}:out:{output_index:03d}"
        )
        self._set_reference(tensor, stable_ref)
        return {
            "output_ref": stable_ref,
            "tensor": tensor_descriptor(tensor),
        }

    def _append_event(
        self,
        *,
        ordinal: int,
        event_kind: str,
        transform_reference: dict[str, Any],
        input_refs: list[str],
        input_roles: list[str],
        outputs: list[dict[str, Any]],
        receipt_payload: dict[str, Any],
    ) -> None:
        if not outputs:
            return
        if not input_refs:
            literal_ref = f"source:literal:event:{ordinal:06d}"
            self.register_literal_source(
                literal_ref,
                {
                    "source_kind": "declared_non_tensor_operands",
                    "transform_reference": transform_reference,
                    "receipt_payload": receipt_payload,
                },
            )
            input_refs = [literal_ref]
            input_roles = ["declared_non_tensor_operand"]
        if len(input_refs) != len(input_roles):
            raise RuntimeError("INPUT_ROLE_CARDINALITY_MISMATCH")
        self.events.append(
            {
                "event_kind": event_kind,
                "input_refs": input_refs,
                "input_roles": input_roles,
                "micro_step": self.micro_step,
                "ordinal": ordinal,
                "outputs": outputs,
                "phase": self.phase,
                "receipt_payload": receipt_payload,
                "step": self.step,
                "transform_reference": transform_reference,
            }
        )

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if not self.active:
            return func(*args, **kwargs)

        input_tensors = _flatten_tensors(args) + _flatten_tensors(kwargs)
        unique_inputs: list[torch.Tensor] = []
        seen_input_ids: set[int] = set()
        for tensor in input_tensors:
            if id(tensor) not in seen_input_ids:
                seen_input_ids.add(id(tensor))
                unique_inputs.append(tensor)
        input_refs = [self.reference(tensor) for tensor in unique_inputs]
        input_versions = {id(tensor): int(tensor._version) for tensor in unique_inputs}
        input_descriptors = [tensor_descriptor(tensor) for tensor in unique_inputs]

        result = func(*args, **kwargs)

        output_tensors = _flatten_tensors(result)
        for tensor in unique_inputs:
            if int(tensor._version) > input_versions[id(tensor)]:
                output_tensors.append(tensor)
        unique_outputs: list[torch.Tensor] = []
        seen_output_ids: set[int] = set()
        for tensor in output_tensors:
            if id(tensor) not in seen_output_ids:
                seen_output_ids.add(id(tensor))
                unique_outputs.append(tensor)

        if not unique_outputs:
            return result
        ordinal = self._ordinal
        self._ordinal += 1
        outputs = [
            self._new_output(tensor, ordinal, index)
            for index, tensor in enumerate(unique_outputs)
        ]
        self._append_event(
            ordinal=ordinal,
            event_kind="aten_dispatch",
            transform_reference={
                "framework": "PyTorch",
                "operator": str(func),
                "schema": str(getattr(func, "_schema", "")),
            },
            input_refs=input_refs,
            input_roles=["tensor_operand"] * len(input_refs),
            outputs=outputs,
            receipt_payload={
                "input_tensors_before": input_descriptors,
                "non_tensor_args": _scalar_summary(args),
                "non_tensor_kwargs": _scalar_summary(kwargs),
            },
        )
        return result

    def emit_manual(
        self,
        *,
        transform_reference: dict[str, Any],
        inputs: list[tuple[str, str]],
        output_tensors: list[tuple[torch.Tensor, str]],
        receipt_payload: dict[str, Any],
    ) -> list[str]:
        ordinal = self._ordinal
        self._ordinal += 1
        outputs: list[dict[str, Any]] = []
        for index, (tensor, output_kind) in enumerate(output_tensors):
            row = self._new_output(
                tensor,
                ordinal,
                index,
                prefix=output_kind,
            )
            with self.paused():
                row["content_sha256"] = tensor_content_sha256(tensor)
            outputs.append(row)
        self._append_event(
            ordinal=ordinal,
            event_kind="synchronous_training_receipt",
            transform_reference=transform_reference,
            input_refs=[row[0] for row in inputs],
            input_roles=[row[1] for row in inputs],
            outputs=outputs,
            receipt_payload=receipt_payload,
        )
        return [row["output_ref"] for row in outputs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_contract": {
                "capture_level": "every_tensor-producing_ATen_dispatch_plus_training_boundaries",
                "non_tensor_arguments": "stored_in_transform_or_receipt_payload",
                "pairing_rule": "actual_input_tensor_refs_cross_actual_output_tensor_refs_per_completed_call",
                "runtime_callback": "__torch_dispatch__ returns ordinary result before receipt append",
            },
            "events": self.events,
            "run_id": self.run_id,
            "sources": [self.sources[key] for key in sorted(self.sources)],
        }

