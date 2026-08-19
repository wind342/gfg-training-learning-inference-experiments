from __future__ import annotations

import json

from ..runner import run_all_scientific


def main() -> int:
    result, _ = run_all_scientific()
    controls = result["negative_controls"]
    print(json.dumps(controls, sort_keys=True, indent=2))
    return 0 if controls["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
