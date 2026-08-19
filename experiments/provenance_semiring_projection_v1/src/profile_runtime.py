from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_ROOT = EXPERIMENT_ROOT / "profiles"


def support_membership_predicate(support: dict, query: dict, predicate: str) -> bool:
    return predicate == "membership" and support["tuple_identity"] == query["tuple_identity"]


def load_profile(filename: str) -> dict[str, Any]:
    path = PROFILES_ROOT / filename
    if path.parent != PROFILES_ROOT or not path.is_file():
        raise ValueError(f"unknown frozen profile: {filename!r}")
    return json.loads(path.read_text(encoding="utf-8"))

