from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.synthetic_cases import execute_many_to_many_case
from experiments.database_lineage.src.synthetic_oracle import MANY_TO_MANY_DIRECT_PAIRS


def test_no_post_hoc_or_fabricated_pairing() -> None:
    adapter = CoreAdapter(run_id="pairing")
    _outputs, _executor = execute_many_to_many_case(adapter)
    reader = CoreLineageReader(adapter.validated_snapshot(), adapter.registry)
    actual = {
        (row["input_tuple_id"], row["output_tuple_id"])
        for row in reader.direct_relations()
        if row["outcome_kind"] == "support"
    }
    assert actual == MANY_TO_MANY_DIRECT_PAIRS
    assert len(actual - MANY_TO_MANY_DIRECT_PAIRS) == 0
    assert len(MANY_TO_MANY_DIRECT_PAIRS - actual) == 0
