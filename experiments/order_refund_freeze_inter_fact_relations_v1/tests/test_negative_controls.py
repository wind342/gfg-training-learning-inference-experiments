from experiments.order_refund_freeze_inter_fact_relations_v1.src.negative_controls import (
    run_negative_controls,
)


def test_thirty_preregistered_controls_fail_closed_once() -> None:
    result = run_negative_controls()
    assert result["status"] == "PASS"
    assert result["control_count"] == 30
    assert result["passed_count"] == 30
    assert all(row["execution_count"] == 1 for row in result["controls"])
    assert all(row["auto_repaired"] is False for row in result["controls"])
    assert all(row["partial_success"] is False for row in result["controls"])
    assert all(
        row["partial_scientific_result_emitted"] is False
        for row in result["controls"]
    )
