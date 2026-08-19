from __future__ import annotations

import pytest

from experiments.executable_generation_fact_graph_v2.adapters.order_adapter import (
    run_order_graph,
)


@pytest.fixture(scope="session")
def order_run():
    return run_order_graph()
