from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from generation_relation_core.canonical import canonical_bytes

from ..graph_model import (
    ExecutableGenerationFactGraphV2,
    GraphValidationV2,
    ValidatedGenerationFactGraphV2,
)
from ..graph_query import ExecutableGenerationFactGraphQueryEngineV2


def _path_signatures(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    formation: dict,
) -> list[str]:
    signatures = []
    for path in formation["path_instances"]:
        final_to_source = [
            query.fact_nodes[node_id]
            for node_id in reversed(path["node_ids"])
        ]
        support_keys = [
            node.z["entity"]["support_payload"]["native_support_key"]
            for node in final_to_source
        ]
        roles = [node.rho for node in final_to_source]
        occurrence_keys = [
            node.omega_bar["generation_occurrence"][
                "stable_instance_key"
            ]
            for node in final_to_source
        ]
        source = query.fact_nodes[path["node_ids"][0]].u["entity"]
        signatures.append(
            "|".join(
                [
                    *support_keys,
                    *roles,
                    *occurrence_keys,
                    source["source_identity"],
                ]
            )
        )
    return sorted(signatures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        payload = json.loads(
            Path(args.input).read_text(encoding="utf-8")
        )
        allowed = {
            "execution_run_id",
            "graph",
            "validation",
            "capture_audit",
            "query_window",
            "traversal_policy",
            "schema_version",
        }
        if set(payload) != allowed:
            raise ValueError("SIGNAL_CANDIDATE_INPUT_SCOPE_INVALID")
        graph = ExecutableGenerationFactGraphV2.from_dict(
            payload["graph"]
        )
        validated = ValidatedGenerationFactGraphV2(
            graph=graph,
            validation=GraphValidationV2(**payload["validation"]),
            capture_audit=payload["capture_audit"],
        )
        query = ExecutableGenerationFactGraphQueryEngineV2(validated)
        formation = query.formation_subgraph(
            payload["query_window"], payload["traversal_policy"]
        )
        signatures = _path_signatures(query, formation)
        raw_sources = sorted(
            {
                query.fact_nodes[path["node_ids"][0]]
                .u["entity"]["source_identity"]
                for path in formation["path_instances"]
            }
        )
        selected_supports = sorted(
            query.fact_nodes[node_id]
            .z["entity"]["support_payload"]["native_support_key"]
            for node_id in formation["selected_result_nodes"]
        )
        result = {
            "status": "PASS",
            "process_role": "candidate",
            "execution_run_id": payload["execution_run_id"],
            "answer": {
                "selected_final_support_keys": selected_supports,
                "raw_source_identities": raw_sources,
                "path_count": len(signatures),
                "path_signature_multiset_sha256": hashlib.sha256(
                    canonical_bytes(signatures)
                ).hexdigest(),
            },
            "schema_version": "signal-graph-candidate-output-v2",
        }
        output.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        output.write_text(
            json.dumps(
                {
                    "status": "FAIL",
                    "process_role": "candidate",
                    "reason_code": str(exc),
                    "partial_success": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
