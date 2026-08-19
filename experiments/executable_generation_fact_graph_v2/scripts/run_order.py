from __future__ import annotations

import json

from ..adapters.order_adapter import run_order_graph


def main() -> int:
    result, _ = run_order_graph()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
