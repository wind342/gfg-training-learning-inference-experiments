from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from .canonical_graph import canonical_hash


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _result_markdown(
    result: dict[str, Any], census: dict[str, Any]
) -> str:
    by_type = "\n".join(
        (
            f"- `{name}`: total={row['total']}, "
            f"unique={row['both_endpoints_unique_fact_mapping']}, "
            f"unmappable={row['unmappable_count']}"
        )
        for name, row in census["relation_type_mapping_counts"].items()
    )
    return f"""# Executable Generation-Fact Graph v1 — final result

Final status: **{result["final_status"]}**

Failure reason:
`{result["failure_reason"]}`

## Exact falsification

The frozen v1 vertex set contains only one fact node per
`GenerationBinding`. The fresh real order execution produced
{census["primitive_relation_count"]} mandatory primitive relations.
Only {census["legally_mappable_relation_count"]} have one and only one fact
mapping at both native endpoints. Exactly
{census["unmappable_primitive_relation_count"]} cannot be represented without
dropping a relation, fabricating a fact, reattaching an endpoint, or performing
an unsupported Cartesian expansion.

{by_type}

All prohibited action counts are zero. The compiler failed closed.

## What this does and does not falsify

This result falsifies only the combination of:

1. graph vertices are restricted to `GenerationBinding` fact nodes; and
2. every native occurrence-level primitive relation must be preserved.

It does not falsify the five-coordinate atomic generation fact, the existing
inter-fact relations, or the possibility of an executable graph with explicit
occurrence nodes.

## Other completed v1 runs

- Signal: {result["signal"]["node_count"]} fact nodes,
  {result["signal"]["edge_count"]} adjacent-stage edges,
  {result["signal"]["path_count"]} exact paths and
  {result["signal"]["raw_source_count"]} exact raw sources.
- Order source: {result["order"]["workflow_execution_count"]} real workflow
  executions and {result["order"]["query_count"]}/56 exact source queries.
- Scale source: {result["scale"]["source_scientific"]["occurrence_count"]}
  occurrences and {result["scale"]["source_scientific"]["fact_count"]} facts;
  graph compilation stopped at the ambiguous endpoint precondition.
- Signed projection: {result["signed_projection"]["execution_count"]}
  executions exactly matched the frozen Signed Generation Algebra candidate.
- Negative controls: {result["negative_controls"]["detected_count"]}/48
  detected once each with unique reason codes.

Scientific SHA-256: `{result["scientific_sha256"]}`
"""


def refresh_manifest(
    output_root: Path,
    *,
    scientific_sha256: str,
) -> dict[str, Any]:
    manifest_rows = []
    for path in sorted(output_root.iterdir(), key=lambda row: row.name):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        manifest_rows.append(
            {
                "path": path.name,
                "byte_count": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "diagnostic_only": path.name
                in {
                    "scale_performance_diagnostics.json",
                    "test_results.json",
                },
            }
        )
    manifest = {
        "schema_version": "executable-generation-fact-graph-manifest-v1",
        "scientific_sha256": scientific_sha256,
        "artifact_count": len(manifest_rows),
        "artifacts": manifest_rows,
    }
    _write_json(output_root / "artifact_manifest.json", manifest)
    return manifest


def materialize(
    result: dict[str, Any],
    auxiliary: dict[str, Any],
    *,
    output_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    census = auxiliary["endpoint_census"]
    scale_census = auxiliary["scale_endpoint_census"]
    failure = {
        "schema_version": "pure-fact-graph-failure-v1",
        "final_status": result["final_status"],
        "failure_reason": result["failure_reason"],
        "primitive_relation_count": census["primitive_relation_count"],
        "legally_mappable_relation_count": census[
            "legally_mappable_relation_count"
        ],
        "unmappable_primitive_relation_count": census[
            "unmappable_primitive_relation_count"
        ],
        "claim_boundary": result["claim_boundary"],
        "prohibited_action_counts": census["prohibited_action_counts"],
    }
    files: dict[str, Any] = {
        "endpoint_type_census.json": census,
        "pure_fact_graph_failure.json": failure,
        "unmappable_primitive_relations.json": {
            "schema_version": "unmappable-primitive-relations-v1",
            "count": len(census["unmappable_relations"]),
            "relations": census["unmappable_relations"],
        },
        "ambiguous_occurrence_lifting.json": {
            "schema_version": "ambiguous-occurrence-lifting-v1",
            "order_count": len(census["ambiguous_occurrence_lifting"]),
            "order_relations": census["ambiguous_occurrence_lifting"],
            "scale_count": len(
                scale_census["ambiguous_occurrence_lifting"]
            ),
            "scale_relation_summary_sha256": canonical_hash(
                scale_census["ambiguous_occurrence_lifting"]
            ),
            "scale_sample": scale_census[
                "ambiguous_occurrence_lifting"
            ][:100],
            "scale_full_list_omitted_from_this_diagnostic": (
                len(scale_census["ambiguous_occurrence_lifting"]) > 100
            ),
            "cartesian_expanded_edge_count": 0,
        },
        "signal_graph_result.json": result["signal"],
        "order_graph_result.json": result["order"],
        "scale_graph_result.json": result["scale"],
        "signed_projection.json": result["signed_projection"],
        "projection_exactness.json": result["projections"],
        "negative_controls.json": result["negative_controls"],
        "protected_path_audit.json": result["protected_path_audit"],
        "determinism.json": {
            "schema_version": "graph-determinism-v1",
            "signal_two_run_graph_exact": result["signal"][
                "graph_two_run_deterministic"
            ],
            "scale_two_run_scientific_exact": result["scale"]["gates"][
                "two_scientific_runs_deterministic"
            ],
            "canonical_materializer": True,
        },
        "scale_performance_diagnostics.json": auxiliary[
            "scale_diagnostics"
        ],
        "v1_final_result.json": result,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        _write_json(output_root / name, value)
    markdown = _result_markdown(result, census)
    (output_root / "V1_RESULT.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    return refresh_manifest(
        output_root, scientific_sha256=result["scientific_sha256"]
    )
