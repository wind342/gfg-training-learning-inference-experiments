from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.semiring_homomorphisms import (
    derive_lower_domains_from_nx,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated N[X]-derived lower projections")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.input.read_text(encoding="utf-8"))
    if native.get("schema_version") != "native-nx-corpus-v1":
        raise ValueError("unexpected Native N[X] corpus")
    document = {
        "schema_version": "nx-derived-lower-domain-corpus-v2",
        "process_role": "derive classified lower results from canonical N[X] documents",
        "results": [derive_lower_domains_from_nx(item) for item in native["results"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
