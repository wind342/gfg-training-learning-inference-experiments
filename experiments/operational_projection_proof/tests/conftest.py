from __future__ import annotations

import pytest

from experiments.operational_projection_proof.scripts.run_all import build_reports


@pytest.fixture(scope="session")
def proof_reports():
    return build_reports()
