from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..graph_model import (
    ExecutableGenerationFactGraphV2,
    GraphValidationV2,
    ValidatedGenerationFactGraphV2,
)
from ..scale_query import ScaleGraphQueryResolver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    allowed = {
        "execution_run_id",
        "graph",
        "validation",
        "capture_audit",
        "queries",
        "schema_version",
    }
    try:
        if set(payload) != allowed:
            raise ValueError("SCALE_GRAPH_CANDIDATE_INPUT_SCOPE_INVALID")
        graph = ExecutableGenerationFactGraphV2.from_dict(
            payload["graph"]
        )
        validation = GraphValidationV2(**payload["validation"])
        validated = ValidatedGenerationFactGraphV2(
            graph=graph,
            validation=validation,
            capture_audit=payload["capture_audit"],
        )
        if (
            validation.status != "PASS"
            or graph.metadata.execution_run_id
            != payload["execution_run_id"]
        ):
            raise ValueError("SCALE_GRAPH_VALIDATION_REQUIRED")
        resolver = ScaleGraphQueryResolver(
            validated, payload["capture_audit"]
        )
        result = {
            "status": "PASS",
            "process_role": "candidate",
            "execution_run_id": payload["execution_run_id"],
            "answers": [
                resolver.answer(query) for query in payload["queries"]
            ],
            "metrics": resolver.metrics(),
            "schema_version": "candidate-query-output-v2",
        }
        output_path.write_text(
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
        output_path.write_text(
            json.dumps(
                {
                    "status": "FAIL",
                    "reason_code": str(exc),
                    "partial_success": False,
                    "process_role": "candidate",
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
