from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture
def artifact():
    def load(name: str):
        path = ARTIFACTS / name
        assert path.is_file(), f"mandatory v2 artifact missing: {name}"
        return json.loads(path.read_text(encoding="utf-8"))

    return load

