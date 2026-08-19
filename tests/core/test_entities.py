from __future__ import annotations

import copy
import math

import pytest

from generation_relation_core.entities import (
    generation_binding,
    generation_occurrence,
    source_information,
)
from generation_relation_core.errors import CoreV3Error
from generation_relation_core.snapshots import validate_tables


def test_tagged_origin_rejects_source_fields() -> None:
    with pytest.raises(CoreV3Error):
        generation_binding(
            domain_scope_id="d",
            origin_reference={"kind": "generated_origin", "source_information_id": "si3_" + "1" * 64},
            generation_occurrence_id="gocc3_" + "2" * 64,
            outcome_reference={"kind": "support", "support_id": "ps3_" + "3" * 64},
            relation_role="direct",
            evidence_ids=["ev3_" + "4" * 64],
        )


def test_occurrence_rejects_nonfinite_transform(core_fixture) -> None:
    manifest_id = core_fixture.snapshot.tables.generator_manifests[0]["generator_manifest_id"]
    with pytest.raises(CoreV3Error) as exc:
        generation_occurrence(
            domain_scope_id="d",
            generator_manifest_id=manifest_id,
            occurrence_stage="s",
            occurrence_type="t",
            stable_instance_key="k",
            occurrence_index=0,
            transform_reference={"scale": math.inf},
            occurrence_payload={},
        )
    assert "NON_FINITE_NUMBER" in str(exc.value)


def test_source_hierarchy_cycle_is_rejected(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    first, second = tables.source_information_records[:2]
    tables.source_information_records[:2] = [
        source_information(
            domain_scope_id=first["domain_scope_id"],
            source_identity=first["source_identity"],
            source_parent_id=second["source_identity"],
            source_granularity=first["source_granularity"],
            source_payload=first["source_payload"],
        ),
        source_information(
            domain_scope_id=second["domain_scope_id"],
            source_identity=second["source_identity"],
            source_parent_id=first["source_identity"],
            source_granularity=second["source_granularity"],
            source_payload=second["source_payload"],
        ),
    ]
    with pytest.raises(CoreV3Error) as exc:
        validate_tables(tables, core_fixture.registry)
    assert "HIERARCHY_CYCLE" in str(exc.value)
