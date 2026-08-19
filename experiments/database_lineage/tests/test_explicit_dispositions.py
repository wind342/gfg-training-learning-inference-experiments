from experiments.database_lineage.src.synthetic_oracle import BUSINESS_DISPOSITIONS


def test_all_contract_scope_losses_are_explicit(business_run) -> None:
    _adapter, _rows, _snapshot, reader = business_run
    actual = {
        (row["input_tuple_id"], row["role"])
        for row in reader.direct_relations()
        if row["outcome_kind"] == "disposition"
    }
    assert actual == BUSINESS_DISPOSITIONS
