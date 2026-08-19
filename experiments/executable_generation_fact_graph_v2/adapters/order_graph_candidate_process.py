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
from .order_graph_query_adapter import resolve_order_graph_queries


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_graph(context: dict) -> ValidatedGenerationFactGraphV2:
    if set(context) != {"graph", "validation", "capture_audit"}:
        raise ValueError("ORDER_GRAPH_CONTEXT_SCOPE_INVALID")
    return ValidatedGenerationFactGraphV2(
        graph=ExecutableGenerationFactGraphV2.from_dict(
            context["graph"]
        ),
        validation=GraphValidationV2(**context["validation"]),
        capture_audit=context["capture_audit"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    runtime_read_count = 0

    def audit(event: str, arguments: tuple) -> None:
        nonlocal runtime_read_count
        if event != "open" or not arguments:
            return
        raw_path = arguments[0]
        mode = arguments[1] if len(arguments) > 1 else "r"
        if not isinstance(raw_path, (str, bytes)):
            return
        if isinstance(mode, int) or "r" not in mode:
            return
        opened = Path(raw_path).resolve()
        if opened != input_path:
            raise RuntimeError(
                "ORDER_GRAPH_CANDIDATE_FILE_READ_FORBIDDEN"
            )
        runtime_read_count += 1

    sys.addaudithook(audit)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if set(payload) != {"contexts", "queries", "schema_version"}:
            raise ValueError("ORDER_GRAPH_CANDIDATE_INPUT_SCOPE_INVALID")
        if payload["schema_version"] != "order-graph-candidate-input-v2":
            raise ValueError("ORDER_GRAPH_CANDIDATE_INPUT_SCHEMA_INVALID")
        result = resolve_order_graph_queries(
            [_validated_graph(row) for row in payload["contexts"]],
            payload["queries"],
        )
        result["process_role"] = "candidate"
        result["runtime_file_read_audit"] = {
            "input_file_only": True,
            "read_count": runtime_read_count,
        }
        _write(output_path, result)
        return 0
    except Exception as exc:
        _write(
            output_path,
            {
                "status": "FAIL",
                "process_role": "candidate",
                "reason_code": str(exc),
                "partial_success": False,
                "runtime_file_read_audit": {
                    "input_file_only": True,
                    "read_count": runtime_read_count,
                },
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
