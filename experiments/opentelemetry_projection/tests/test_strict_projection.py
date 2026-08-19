from experiments.opentelemetry_projection.src.strict_projection import (
    run_strict_projection_counterexamples,
)


def test_two_distinct_core_histories_have_identical_otel_projection() -> None:
    reports = run_strict_projection_counterexamples()
    assert len(reports) == 2
    assert all(report["normalized_otel_equal"] for report in reports)
    assert all(report["native_otel_equal"] for report in reports)
    assert all(not report["snapshot_ids_equal"] for report in reports)
    assert all(not report["source_sets_equal"] for report in reports)
    assert all(not report["binding_sets_equal"] for report in reports)
    assert all(not report["backward_lineage_equal"] for report in reports)
    assert all(not report["direct_relation_sets_equal"] for report in reports)
