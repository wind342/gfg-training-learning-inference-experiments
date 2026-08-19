from __future__ import annotations

import argparse
import json

from ..adapters.scale_adapter import run_scale_graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        choices=("small", "medium", "large"),
        default="large",
    )
    args = parser.parse_args()
    result, _ = run_scale_graph(args.scale)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
