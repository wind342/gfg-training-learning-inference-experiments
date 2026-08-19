from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.signal_multistage_generated_origin_v1.data import (
    load_signal_window,
)
from experiments.signal_multistage_generated_origin_v1.reference import (
    compute_reference,
)


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
        if set(payload) != {
            "execution_run_id",
            "data_root",
            "schema_version",
        }:
            raise ValueError("SIGNAL_REFERENCE_INPUT_SCOPE_INVALID")
        answer = compute_reference(
            load_signal_window(Path(payload["data_root"]))
        ).answer
        result = {
            "status": "PASS",
            "process_role": "reference",
            "execution_run_id": payload["execution_run_id"],
            "answer": {
                key: answer[key]
                for key in (
                    "selected_final_support_keys",
                    "raw_source_identities",
                    "path_count",
                    "path_signature_multiset_sha256",
                )
            },
            "schema_version": "signal-reference-output-v2",
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
                    "process_role": "reference",
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
