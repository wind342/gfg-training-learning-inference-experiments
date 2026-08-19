from experiments.database_lineage.src.synthetic_oracle import (
    BUSINESS_BACKWARD,
    BUSINESS_DIRECT_PAIRS,
    BUSINESS_FORWARD,
    BUSINESS_OUTPUT,
)


def test_business_query_output_backward_forward_and_direct_edges(business_run) -> None:
    _adapter, rows, _snapshot, reader = business_run
    assert [row.values for row in rows] == BUSINESS_OUTPUT
    final_ids = {row.tuple_id for row in rows}
    for output_id, expected in BUSINESS_BACKWARD.items():
        assert set(reader.backward(output_id).tuple_ids) == expected
    for source_id, expected in BUSINESS_FORWARD.items():
        assert set(reader.forward(source_id, final_ids).tuple_ids) == expected
    actual_pairs = {
        (row["input_tuple_id"], row["output_tuple_id"])
        for row in reader.direct_relations()
        if row["outcome_kind"] == "support"
    }
    assert actual_pairs == BUSINESS_DIRECT_PAIRS
    assert reader.backward(rows[0].tuple_id).derivation_path_count == 20


def test_every_consumed_intermediate_is_a_generated_origin(business_run) -> None:
    _adapter, _rows, snapshot, _reader = business_run
    support_ids = {
        row["support_id"] for row in snapshot.tables.perceptual_support_records
    }
    bridged = {
        row["origin_payload"]["prior_support_id"]
        for row in snapshot.tables.generated_origins
    }
    final_supports = {
        row["support_id"]
        for row in snapshot.tables.perceptual_support_records
        if row["support_payload"]["operator_stage"] == "customer_top_1"
    }
    disposed_at_limit = {
        row["origin_payload"]["prior_support_id"]
        for row in snapshot.tables.generated_origins
        if row["origin_payload"]["table_identity"] == "customer_rank"
    }
    assert support_ids - final_supports <= bridged | disposed_at_limit
