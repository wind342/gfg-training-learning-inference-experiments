from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.signed_generation_algebra_v1.collector import (
    SignedEffectCollector,
)
from experiments.signed_generation_algebra_v1.generator import (
    execute_native,
)
from experiments.signed_generation_algebra_v1.query import (
    RegisteredSignedEffectQuery,
)

from ..endpoint_registry import build_core_occurrence_catalog
from ..graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from ..graph_projections import project_signed_algebra
from ..graph_validator import (
    load_contracts,
    validate_executable_generation_fact_graph_v2,
)
from .common import complete_capture_audit, empty_relation_store


CONTRACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "signed_generation_algebra_v1"
    / "contracts"
)


def _load(name: str) -> dict[str, Any]:
    return json.loads(
        (CONTRACT_ROOT / name).read_text(encoding="utf-8")
    )


def run_signed_projection() -> tuple[dict[str, Any], dict[str, Any]]:
    operation_contract = _load("operation_contract.json")
    signed_contract = _load("signed_effect_contract.json")
    query_contract = _load("query_contract.json")
    execution_ids = [
        row["execution_id"]
        for row in operation_contract["executions"]
    ]
    contracts = load_contracts()
    comparisons = []
    contexts = []
    for execution in operation_contract["executions"]:
        execution_id = execution["execution_id"]
        collector = SignedEffectCollector(execution_id, execution_ids)
        native = execute_native(execution, collector.capture)
        collected = collector.finalize(native)
        snapshot_inputs = [
            {
                "snapshot": collected.snapshot,
                "execution_run_id": execution_id,
            }
        ]
        store = empty_relation_store(execution_id)
        catalog = build_core_occurrence_catalog(snapshot_inputs)
        audit = complete_capture_audit(
            execution_id, domain="signed_generation_algebra"
        )
        graph = compile_executable_generation_fact_graph_v2(
            snapshot_inputs,
            store,
            catalog,
            audit,
            contracts["graph_profile"],
            contracts["relation_type_registry"],
        )
        validated = validate_executable_generation_fact_graph_v2(
            graph,
            snapshot_inputs,
            store,
            catalog,
            audit,
            contracts,
        )
        candidate = project_signed_algebra(
            validated, signed_contract
        )
        reference = RegisteredSignedEffectQuery(
            collected.snapshot,
            collected.validation,
            collected.predicate_registry,
            signed_contract,
            query_contract,
        ).interpret()
        comparisons.append(
            {
                "execution_id": execution_id,
                "binding_count": len(graph.fact_nodes),
                "occurrence_count": len(graph.occurrence_nodes),
                "incidence_count": len(graph.incidence_edges),
                "graph_id": graph.metadata.graph_id,
                "signed_pair_exact": (
                    candidate["signed_pair"]
                    == reference["signed_pair"]
                ),
                "net_projection_exact": (
                    candidate["net_projection"]
                    == reference["net_projection"]
                ),
                "contributions_exact": (
                    candidate["algebraic_contributions"]
                    == reference["algebraic_contributions"]
                ),
                "complete_facts_exact": (
                    candidate["complete_facts"]
                    == reference["complete_facts"]
                ),
                "explicit_disposition_boundary_exact": (
                    candidate["neutral_fact_ids"]
                    == reference["neutral_fact_ids"]
                ),
                "candidate": candidate,
                "reference": reference,
            }
        )
        contexts.append(
            {
                "validated_graph": validated,
                "snapshot_inputs": snapshot_inputs,
                "relation_store": store,
                "occurrence_catalog": catalog,
                "capture_audit": audit,
            }
        )
    gates = {
        "all_execution_signed_pairs_exact": all(
            row["signed_pair_exact"] for row in comparisons
        ),
        "all_execution_net_projections_exact": all(
            row["net_projection_exact"] for row in comparisons
        ),
        "all_execution_contributions_exact": all(
            row["contributions_exact"] for row in comparisons
        ),
        "all_complete_facts_exact": all(
            row["complete_facts_exact"] for row in comparisons
        ),
        "explicit_disposition_not_automatically_negative": all(
            row["explicit_disposition_boundary_exact"]
            for row in comparisons
        ),
        "occurrence_nodes_not_used_as_signed_terms": all(
            row["candidate"]["algebraic_contributions"]
            == row["reference"]["algebraic_contributions"]
            for row in comparisons
        ),
    }
    result = {
        "schema_version": "graph-signed-projection-comparison-v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "execution_count": len(comparisons),
        "comparisons": comparisons,
        "gates": gates,
    }
    return result, {
        "contexts": contexts,
        "signed_effect_contract": signed_contract,
        "query_contract": query_contract,
    }
