from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
FILES = ("MODEL_CONTRACT.json", "PROTOCOL_FREEZE.md", "runtime.py", "runner.py")


def main() -> None:
    payload = {
        "schema": "rl-e06-contract-freeze-v1",
        "status": "FROZEN_BEFORE_FORMAL_EXECUTION",
        "files": {
            name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

