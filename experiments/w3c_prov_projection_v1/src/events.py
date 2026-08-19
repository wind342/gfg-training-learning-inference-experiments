from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentEvent:
    name: str
    version: str
    code_identity: str


@dataclass(frozen=True)
class SourceEvent:
    key: str
    source_identity: str
    source_granularity: str
    domain_type: str
    stable_domain_identity: str
    value: Any


@dataclass(frozen=True)
class ActivityEvent:
    key: str
    stage: str
    occurrence_type: str
    stable_instance_key: str
    occurrence_index: int
    operation_type: str
    transform_reference: dict[str, Any]
    diagnostic_context: dict[str, Any]


@dataclass(frozen=True)
class OutcomeEvent:
    key: str
    kind: str
    result_category: str | None = None
    result_identity: str | None = None
    payload: Any = None
    disposition_category: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class BridgeEvent:
    key: str
    prior_outcome_key: str
    profile_external_detail: str


@dataclass(frozen=True)
class BindingEvent:
    origin_kind: str
    origin_key: str
    activity_key: str
    outcome_key: str
    role: str
    ordinal: int


class GenerationSink(Protocol):
    def on_agent(self, event: AgentEvent) -> None: ...
    def on_source(self, event: SourceEvent) -> None: ...
    def on_activity(self, event: ActivityEvent) -> None: ...
    def on_outcome(self, event: OutcomeEvent) -> None: ...
    def on_bridge(self, event: BridgeEvent) -> None: ...
    def on_binding(self, event: BindingEvent) -> None: ...


@dataclass(frozen=True)
class GeneratorVariant:
    evidence_detail: str = "evidence-base"
    environment_detail: str = "environment-base"
    operation_name: str = "generate_fixture_base"
    bridge_detail: str = "bridge-base"
    transform_variant: str = "left_associative"


@dataclass(frozen=True)
class TransformExecutionReceipt:
    transform_variant: str
    executed_branch_id: str
    executed_function_or_code_path: str
    input_values: dict[str, int]
    intermediate_values: tuple[int, ...]
    output_value: int
    transform_reference_sha256: str
    occurrence_payload_sha256: str


class TransformReceiptSink(Protocol):
    def on_transform_execution(self, receipt: TransformExecutionReceipt) -> None: ...


@dataclass(frozen=True)
class GeneratedOutput:
    files: dict[str, bytes]
    media_types: dict[str, str]

    def metadata(self) -> list[dict[str, Any]]:
        import hashlib

        return [
            {
                "name": name,
                "media_type": self.media_types[name],
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for name, value in sorted(self.files.items())
        ]
