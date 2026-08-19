import pytest

from experiments.operational_projection_proof.src.errors import ProjectionProofError
from experiments.operational_projection_proof.src.hierarchical_consistency import (
    assert_acyclic_edges,
    compare_hierarchical,
    require_hierarchical_equality,
)


def test_generic_hierarchical_comparator_preserves_multiplicity() -> None:
    records = [{"id": "a"}, {"id": "a"}, {"id": "b"}]
    report = compare_hierarchical(
        profile_id="generic-test", direct=records, hierarchical=list(reversed(records))
    )
    assert report["exact_equal"] is True
    assert report["status"] == "SUPPORTED"


def test_hierarchical_mismatch_and_cycle_fail_closed() -> None:
    report = compare_hierarchical(
        profile_id="generic-test", direct=[{"id": "a"}], hierarchical=[]
    )
    with pytest.raises(ProjectionProofError, match="HIERARCHICAL_MISMATCH"):
        require_hierarchical_equality(report)
    with pytest.raises(ProjectionProofError, match="HIERARCHY_CYCLE"):
        assert_acyclic_edges([("a", "b"), ("b", "a")])
