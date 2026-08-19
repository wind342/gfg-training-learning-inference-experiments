from __future__ import annotations

import pytest

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.synthetic_cases import execute_business_query


@pytest.fixture
def business_run():
    adapter = CoreAdapter(run_id="synthetic-deterministic")
    rows, _executor = execute_business_query(adapter)
    snapshot = adapter.validated_snapshot()
    return adapter, rows, snapshot, CoreLineageReader(snapshot, adapter.registry)
