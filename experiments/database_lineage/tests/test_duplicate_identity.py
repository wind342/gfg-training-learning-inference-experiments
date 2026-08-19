from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.synthetic_cases import execute_many_to_many_case


def test_identical_business_values_keep_distinct_tuple_identity() -> None:
    adapter = CoreAdapter(run_id="duplicates")
    outputs, _executor = execute_many_to_many_case(adapter)
    snapshot = adapter.validated_snapshot()
    reader = CoreLineageReader(snapshot, adapter.registry)
    lineage = [set(reader.backward(row.tuple_id).tuple_ids) for row in outputs]
    assert {"m2m:left:1", "products:p1a"} in lineage
    assert {"m2m:left:1", "products:p1b"} in lineage
    assert outputs[0].values == outputs[1].values
    assert outputs[0].tuple_id != outputs[1].tuple_id
