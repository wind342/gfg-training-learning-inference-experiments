from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def refresh_manifest(
    output_root: Path,
    *,
    scientific_sha256: str,
) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.iterdir(), key=lambda row: row.name):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        rows.append(
            {
                "path": path.name,
                "byte_count": path.stat().st_size,
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "diagnostic_only": path.name
                in {
                    "scale_performance_diagnostics.json",
                    "test_results.json",
                },
            }
        )
    manifest = {
        "schema_version": (
            "executable-generation-fact-graph-manifest-v2"
        ),
        "scientific_sha256": scientific_sha256,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    _write_json(output_root / "artifact_manifest.json", manifest)
    return manifest


def _result_markdown(result: dict[str, Any]) -> str:
    census = result["endpoint_census"]
    order = result["order"]
    signal = result["signal"]
    scale = result["scale"]
    return f"""# Executable Generation-Fact Graph v2 — result

Final status: **{result["final_status"]}**

Scientific SHA-256: `{result["scientific_sha256"]}`

## Preserved v1 falsification

v1 remains `{result["v1_final_status"]}` because exactly
{result["v1_unmappable_relation_count"]} mandatory primitive relations could
not be represented when every vertex was required to be a
`GenerationBinding` fact node.

## v2 object

`G_e=(V_F,V_O,E_I,E_R;Sigma)`

- `V_F`: one complete five-coordinate fact node per `GenerationBinding`.
- `V_O`: one execution occurrence node per referenced concrete occurrence.
- `E_I`: exact `OccurrenceNode --realizes_fact--> FactNode` incidence.
- `E_R`: typed native-endpoint execution and generation relations.
- `Sigma`: schemas, endpoint signatures, evidence, capture and canonical rules.

Occurrence nodes are graph execution skeleton nodes. They are not a sixth
coordinate of the atomic generation fact.

## Exact machine results

- Signal: {signal["fact_node_count"]} facts,
  {signal["occurrence_node_count"]} occurrences,
  {signal["relation_edge_count"]} adjacent-stage relations,
  {signal["path_count"]} exact paths and
  {signal["raw_source_count"]} exact raw sources.
- Order: {order["compiled_primitive_relation_count"]}/
  {order["native_primitive_relation_count"]} primitive relations preserved;
  direct graph queries {order["direct_graph_query_count"]}/56 exact;
  compensation targets {order["compensation_query_count"]}/4 exact;
  projection compatibility
  {order["projection_compatibility_query_count"]}/56 exact;
  FP={order["false_positive_count"]},
  FN={order["false_negative_count"]}.
- Order zero-fact occurrences:
  {census["order_zero_fact_occurrence_count"]}.
- Scale: {scale["occurrence_node_count"]} occurrences,
  {scale["fact_node_count"]} facts,
  {scale["incidence_edge_count"]} incidence edges and
  {scale["primitive_relation_edge_count"]} primitive relation edges.
- Signed projection: {result["signed_projection"]["execution_count"]}
  executions exact.
- Negative controls:
  {result["negative_controls"]["detected_count"]}/48 detected.

## Projection boundary

`pi_Gamma`, `pi_occ`, `pi_R` and `pi_signed` passed their frozen exactness
checks. `pi_fact_graph` is explicitly lossy for occurrence-level primitive
relations and does not claim complete sidecar recovery.

## Limits

`shortest_path` means minimum admitted edge count under a supplied relation
policy. Weighted cost, critical path and algorithmic optimization remain
`NOT_EVALUATED` / `NOT_ESTABLISHED`. Recompilation validation shares the
compiler implementation, so semantic query claims are separately checked
against independent profile references. Scale timings are diagnostic only.
"""


def materialize(
    result: dict[str, Any],
    auxiliary: dict[str, Any],
    *,
    output_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    gates = result["mandatory_gates"]
    files = {
        "endpoint_type_census.json": result["endpoint_census"],
        "graph_schema_validation.json": {
            "schema_version": "graph-schema-validation-v2",
            "status": (
                "PASS" if gates["graph_schema_valid"] else "FAIL"
            ),
            "graph_schema_valid": gates["graph_schema_valid"],
            "canonical_graph_exact": gates[
                "canonical_graph_exact"
            ],
        },
        "fact_node_exactness.json": {
            "schema_version": "fact-node-exactness-v2",
            "every_binding_exactly_one_fact_node": gates[
                "every_binding_exactly_one_fact_node"
            ],
            "fact_content_exact": gates["fact_content_exact"],
            "fact_identity_preserved": gates[
                "fact_identity_preserved"
            ],
            "fact_multiplicity_preserved": gates[
                "fact_multiplicity_preserved"
            ],
        },
        "occurrence_node_exactness.json": {
            "schema_version": "occurrence-node-exactness-v2",
            "every_referenced_occurrence_exactly_one_node": gates[
                "every_referenced_occurrence_exactly_one_node"
            ],
            "occurrence_content_exact": gates[
                "occurrence_content_exact"
            ],
            "occurrence_identity_preserved": gates[
                "occurrence_identity_preserved"
            ],
            "zero_fact_occurrences_preserved": gates[
                "zero_fact_occurrences_preserved"
            ],
            "multi_fact_occurrences_preserved": gates[
                "multi_fact_occurrences_preserved"
            ],
        },
        "incidence_exactness.json": {
            "schema_version": "incidence-exactness-v2",
            "every_fact_exactly_one_incidence": gates[
                "every_fact_exactly_one_incidence"
            ],
            "incidence_exact": gates["incidence_exact"],
            "no_fake_incidence": gates["no_fake_incidence"],
        },
        "primitive_relation_exactness.json": {
            "schema_version": "primitive-relation-exactness-v2",
            **{
                key: value
                for key, value in gates.items()
                if key.startswith("primitive_")
                or key
                in {
                    "every_primitive_relation_exactly_once",
                    "no_relation_drop",
                    "no_relation_fabrication",
                    "no_forced_lifting",
                    "no_cartesian_expansion",
                }
            },
        },
        "signal_result.json": result["signal"],
        "order_result.json": result["order"],
        "scale_result.json": result["scale"],
        "projection_exactness.json": result["projections"],
        "signed_projection.json": result["signed_projection"],
        "negative_controls.json": result["negative_controls"],
        "determinism.json": {
            "schema_version": "graph-determinism-v2",
            "signal_two_run_graph_exact": result["signal"][
                "graph_two_run_deterministic"
            ],
            "scale_two_run_scientific_exact": result["scale"][
                "two_run_scientific_exact"
            ],
            "canonical_materializer": True,
        },
        "protected_path_audit.json": auxiliary[
            "protected_path_audit"
        ],
        "scale_performance_diagnostics.json": result["diagnostics"][
            "scale"
        ],
        "v2_final_result.json": result,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        _write_json(output_root / name, value)
    (output_root / "RESULT.md").write_text(
        _result_markdown(result),
        encoding="utf-8",
        newline="\n",
    )
    return refresh_manifest(
        output_root, scientific_sha256=result["scientific_sha256"]
    )
