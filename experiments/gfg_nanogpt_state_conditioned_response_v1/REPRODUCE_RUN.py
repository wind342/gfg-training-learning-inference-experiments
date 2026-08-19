from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.gfg_nanogpt_state_conditioned_response_v1.INDEPENDENT_CHECKER import check
from experiments.gfg_nanogpt_state_conditioned_response_v1.runner import DEFAULT_OUTPUT, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-existing", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_existing:
        result = check(arguments.output_root)
        print(json.dumps({"mode": "check-existing", "status": result["status"]}, sort_keys=True))
        return
    manifest = run(arguments.output_root)
    result = check(arguments.output_root)
    print(
        json.dumps(
            {"mode": "full-reproduction", "manifest_status": manifest["status"], "independent_check": result["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
