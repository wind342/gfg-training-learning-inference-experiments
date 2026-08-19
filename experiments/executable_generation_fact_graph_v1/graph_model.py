from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphFactNode:
    graph_node_id: str
    execution_run_id: str
    snapshot_id: str
    generation_binding_id: str
    domain_scope_id: str
    source_reference: dict[str, Any]
    realized_transformation: Any
    concrete_occurrence: Any
    outcome_reference: dict[str, Any]
    relation_role: str
    occurrence_identity: str
    outcome_identity: str
    evidence_refs: list[str]
    fact_content_hash: str
    node_instance_hash: str
    native_fact_identity: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphRelationEdge:
    graph_edge_id: str
    execution_run_id: str
    source_graph_node_id: str
    target_graph_node_id: str
    relation_type: str
    relation_semantics: str
    relation_payload: dict[str, Any]
    establishment_source: str
    authority_id: str
    evidence_refs: list[str]
    rule_id: str | None
    input_relation_refs: list[str]
    primitive_or_derived: str
    relation_instance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphMetadata:
    graph_schema_version: str
    graph_id: str
    execution_run_id: str
    domain_scope_id: str
    source_snapshot_ids: list[str]
    source_relation_store_ids: list[str]
    capture_audit_id: str
    graph_profile_id: str
    compiler_code_hash: str
    validator_code_hash: str
    relation_contract_hash: str
    node_count: int
    edge_count: int
    primitive_edge_count: int
    derived_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutableGenerationFactGraph:
    metadata: GraphMetadata
    nodes: tuple[GraphFactNode, ...]
    edges: tuple[GraphRelationEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "nodes": [row.to_dict() for row in self.nodes],
            "edges": [row.to_dict() for row in self.edges],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutableGenerationFactGraph":
        return cls(
            metadata=GraphMetadata(**value["metadata"]),
            nodes=tuple(GraphFactNode(**row) for row in value["nodes"]),
            edges=tuple(GraphRelationEdge(**row) for row in value["edges"]),
        )


@dataclass(frozen=True)
class GraphValidation:
    graph_id: str
    status: str
    gates: dict[str, bool]
    counts: dict[str, int]
    validation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedGenerationFactGraph:
    graph: ExecutableGenerationFactGraph
    validation: GraphValidation
    capture_audit: dict[str, Any]

    @property
    def graph_id(self) -> str:
        return self.graph.metadata.graph_id

