from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import ExperimentError, load_json, write_json
from .indexed_candidate_resolver import resolve_candidate_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    try:
        result = resolve_candidate_payload(load_json(Path(args.input)))
    except ExperimentError as error:
        write_json(
            output_path,
            {
                "status": "FAIL",
                "reason_code": str(error),
                "partial_success": False,
                "process_role": "candidate",
            },
        )
        return 2
    write_json(output_path, {"status": "PASS", **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
