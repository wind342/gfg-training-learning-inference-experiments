from __future__ import annotations

from typing import Any

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.relational_executor import RelationTuple

from .native_otel_capture import NativeOtelCapture


class ProjectionCaptureContract:
    """Compose optional Core and native OTel capture at executor callback time."""

    def __init__(
        self,
        *,
        core: CoreAdapter | None,
        native: NativeOtelCapture | None,
    ) -> None:
        if core is None and native is None:
            raise ValueError("at least one capture target is required")
        self.core = core
        self.native = native

    def capture_output(
        self,
        *,
        stage: str,
        operator_type: str,
        output: RelationTuple,
        inputs: list[RelationTuple],
        roles: list[str],
        occurrence_payload: dict[str, Any],
    ) -> str:
        support_id = None
        if self.core is not None:
            support_id = self.core.capture_output(
                stage=stage,
                operator_type=operator_type,
                output=output,
                inputs=inputs,
                roles=roles,
                occurrence_payload=occurrence_payload,
            )
        if self.native is not None:
            self.native.record_output(
                stage=stage,
                operator_type=operator_type,
                output=output,
                inputs=inputs,
            )
        return support_id or f"native-only:{output.tuple_id}"

    def capture_disposition(
        self,
        *,
        stage: str,
        operator_type: str,
        input_tuple: RelationTuple,
        reason: str,
        occurrence_payload: dict[str, Any],
    ) -> str:
        disposition_id = None
        if self.core is not None:
            disposition_id = self.core.capture_disposition(
                stage=stage,
                operator_type=operator_type,
                input_tuple=input_tuple,
                reason=reason,
                occurrence_payload=occurrence_payload,
            )
        if self.native is not None:
            self.native.record_disposition(
                stage=stage,
                operator_type=operator_type,
                input_tuple=input_tuple,
            )
        return disposition_id or f"native-only:disposition:{input_tuple.tuple_id}"
