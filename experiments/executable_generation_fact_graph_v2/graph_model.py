from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FactNode:
    node_kind: str
    graph_node_id: str
    execution_run_id: str
    snapshot_id: str
    generation_binding_id: str
    domain_scope_id: str
    u: dict[str, Any]
    tau: Any
    omega_bar: dict[str, Any]
    z: dict[str, Any]
    rho: str
    generation_occurrence_id: str
    concrete_occurrence_instance_id: str
    outcome_identity: str
    evidence_refs: list[str]
    native_fact_id: str | None
    native_fact: dict[str, Any] | None
    fact_content_hash: str
    fact_instance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OccurrenceNode:
    node_kind: str
    graph_node_id: str
    execution_run_id: str
    concrete_occurrence_instance_id: str
    generation_occurrence_id: str | None
    occurrence_type: str
    occurrence_stage: str
    stable_instance_key: str
    occurrence_index: int | None
    transform_reference: Any
    occurrence_payload: dict[str, Any]
    generator_manifest_id: str | None
    evidence_refs: list[str]
    catalog_authority: str
    occurrence_content_hash: str
    occurrence_instance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IncidenceEdge:
    edge_kind: str
    relation_type: str
    graph_edge_id: str
    execution_run_id: str
    source_occurrence_node_id: str
    target_fact_node_id: str
    source_concrete_occurrence_instance_id: str
    target_generation_binding_id: str
    derivation: str
    incidence_instance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationEdge:
    edge_kind: str
    graph_edge_id: str
    execution_run_id: str
    source_node_id: str
    source_node_kind: str
    target_node_id: str
    target_node_kind: str
    native_source_id: str
    native_target_id: str
    relation_type: str
    relation_semantics: str
    relation_payload: dict[str, Any]
    primitive_or_derived: str
    establishment_source: str
    authority_id: str
    evidence_refs: list[str]
    rule_id: str | None
    input_relation_refs: list[str]
    original_relation_id: str
    source_relation_store_id: str
    native_relation: dict[str, Any]
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
    relation_store_envelopes: list[dict[str, Any]]
    relation_evidence_records: list[dict[str, Any]]
    occurrence_catalog_id: str
    capture_audit_id: str
    graph_profile_id: str
    relation_registry_id: str
    compiler_code_hash: str
    validator_code_hash: str
    fact_node_count: int
    occurrence_node_count: int
    incidence_edge_count: int
    primitive_relation_edge_count: int
    derived_relation_edge_count: int
    global_transitive_closure_materialized: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutableGenerationFactGraphV2:
    metadata: GraphMetadata
    fact_nodes: tuple[FactNode, ...]
    occurrence_nodes: tuple[OccurrenceNode, ...]
    incidence_edges: tuple[IncidenceEdge, ...]
    relation_edges: tuple[RelationEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "fact_nodes": [row.to_dict() for row in self.fact_nodes],
            "occurrence_nodes": [
                row.to_dict() for row in self.occurrence_nodes
            ],
            "incidence_edges": [
                row.to_dict() for row in self.incidence_edges
            ],
            "relation_edges": [
                row.to_dict() for row in self.relation_edges
            ],
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any]
    ) -> "ExecutableGenerationFactGraphV2":
        return cls(
            metadata=GraphMetadata(**value["metadata"]),
            fact_nodes=tuple(
                FactNode(**row) for row in value["fact_nodes"]
            ),
            occurrence_nodes=tuple(
                OccurrenceNode(**row)
                for row in value["occurrence_nodes"]
            ),
            incidence_edges=tuple(
                IncidenceEdge(**row)
                for row in value["incidence_edges"]
            ),
            relation_edges=tuple(
                RelationEdge(**row)
                for row in value["relation_edges"]
            ),
        )


@dataclass(frozen=True)
class GraphValidationV2:
    graph_id: str
    status: str
    gates: dict[str, bool]
    counts: dict[str, int]
    validation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedGenerationFactGraphV2:
    graph: ExecutableGenerationFactGraphV2
    validation: GraphValidationV2
    capture_audit: dict[str, Any]

    @property
    def graph_id(self) -> str:
        return self.graph.metadata.graph_id
