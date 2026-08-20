from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).parent
FROZEN_FILES = (
    "MODEL_CONTRACT.json",
    "PROTOCOL_FREEZE.md",
    "runtime.py",
    "runner.py",
    "independent_checker.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = PACKAGE / "CONTRACT_FREEZE.json"
    if target.exists():
        raise RuntimeError("RL_E05_CONTRACT_ALREADY_FROZEN")
    payload = {
        "schema": "rl-e05-formal-contract-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: sha256(PACKAGE / name) for name in FROZEN_FILES},
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
