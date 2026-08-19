from __future__ import annotations

import pytest

from tests.fixtures.runtime import CoreFixture, build_core_fixture


@pytest.fixture(scope="session")
def core_fixture() -> CoreFixture:
    return build_core_fixture()
