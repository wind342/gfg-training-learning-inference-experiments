from __future__ import annotations

from typing import Any

from ..common import canonical_sha256
from .common import RuntimeScenarioBuilder


SCALE_CONFIG = {
    "small": {
        "actor_count": 10,
        "events_per_actor": 10,
        "message_count": 20,
        "sync_count": 10,
        "generated_origin_count": 30,
        "reads_from_count": 30,
        "conflict_count": 10,
    },
    "medium": {
        "actor_count": 100,
        "events_per_actor": 10,
        "message_count": 600,
        "sync_count": 300,
        "generated_origin_count": 500,
        "reads_from_count": 500,
        "conflict_count": 200,
    },
    "large": {
        "actor_count": 100,
        "events_per_actor": 100,
        "message_count": 4000,
        "sync_count": 1000,
        "generated_origin_count": 5000,
        "reads_from_count": 5000,
        "conflict_count": 3000,
    },
}


def _query(
    rows: list[dict[str, Any]], scale: str, query_type: str, **values: Any
) -> None:
    rows.append(
        {
            "query_id": f"{scale}-query-{len(rows):06d}",
            "query_type": query_type,
            **values,
        }
    )


def _small_queries(
    occurrence_ids: list[str], facts_by_occurrence: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for left_index, left in enumerate(occurrence_ids):
        for right_index, right in enumerate(occurrence_ids):
            if left_index == right_index:
                continue
            _query(
                queries,
                "small",
                "happens_before",
                source_id=left,
                target_id=right,
            )
        for right in occurrence_ids[left_index + 1 :]:
            _query(
                queries,
                "small",
                "concurrent_with",
                source_id=left,
                target_id=right,
            )
    _query(
        queries,
        "small",
        "relation_path",
        source_id=occurrence_ids[0],
        target_id=occurrence_ids[-1],
    )
    _query(queries, "small", "successors", source_id=occurrence_ids[0])
    _query(queries, "small", "predecessors", target_id=occurrence_ids[-1])
    _query(
        queries,
        "small",
        "fact_to_occurrence",
        fact_id=facts_by_occurrence[0][0]["fact_id"],
    )
    _query(
        queries,
        "small",
        "occurrence_to_selected_facts",
        occurrence_id=occurrence_ids[0],
        selected_fact_ids=[facts_by_occurrence[0][1]["fact_id"]],
    )
    _query(
        queries,
        "small",
        "conflicts",
        fact_id=facts_by_occurrence[0][0]["fact_id"],
    )
    return queries


def _medium_queries(
    occurrence_ids: list[str], facts_by_occurrence: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for index in range(1500):
        source_index = index % 500
        target_index = source_index + 1 + index // 500
        source = occurrence_ids[source_index]
        target = occurrence_ids[target_index]
        _query(
            queries,
            "medium",
            "happens_before",
            source_id=source,
            target_id=target,
        )
        _query(
            queries,
            "medium",
            "happens_before",
            source_id=target,
            target_id=source,
        )
    for index in range(1000):
        left = occurrence_ids[800 + index % 100]
        right = occurrence_ids[900 + (index * 7) % 100]
        _query(
            queries,
            "medium",
            "concurrent_with",
            source_id=left,
            target_id=right,
        )
    for index in range(100):
        _query(
            queries,
            "medium",
            "relation_path",
            source_id=occurrence_ids[index],
            target_id=occurrence_ids[500 + index],
        )
        _query(
            queries,
            "medium",
            "conflicts",
            fact_id=facts_by_occurrence[index][0]["fact_id"],
        )
    return queries


def _large_queries(
    occurrence_ids: list[str], facts_by_occurrence: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    adjacent_pairs: list[tuple[int, int]] = []
    for actor_index in range(80):
        base = actor_index * 100
        adjacent_pairs.extend((base + offset, base + offset + 1) for offset in range(99))
    for source_index, target_index in adjacent_pairs[:5000]:
        source = occurrence_ids[source_index]
        target = occurrence_ids[target_index]
        _query(
            queries,
            "large",
            "happens_before",
            source_id=source,
            target_id=target,
        )
        _query(
            queries,
            "large",
            "happens_before",
            source_id=target,
            target_id=source,
        )
    for index in range(5000):
        complete_index = 8000 + index % 1000
        incomplete_index = 9000 + (
            (index % 1000) + (index // 1000) * 137
        ) % 1000
        left = occurrence_ids[complete_index]
        right = occurrence_ids[incomplete_index]
        _query(
            queries,
            "large",
            "happens_before",
            source_id=left,
            target_id=right,
        )
        _query(
            queries,
            "large",
            "happens_before",
            source_id=right,
            target_id=left,
        )
    for index in range(1000):
        left = occurrence_ids[8000 + index]
        right = occurrence_ids[8000 + (index + 100) % 1000]
        _query(
            queries,
            "large",
            "concurrent_with",
            source_id=left,
            target_id=right,
        )
    for index in range(1000):
        left = occurrence_ids[8000 + index]
        right = occurrence_ids[9000 + index]
        _query(
            queries,
            "large",
            "concurrent_with",
            source_id=left,
            target_id=right,
        )
    for index in range(100):
        _query(
            queries,
            "large",
            "conflicts",
            fact_id=facts_by_occurrence[index][0]["fact_id"],
        )
        _query(
            queries,
            "large",
            "fact_to_occurrence",
            fact_id=facts_by_occurrence[index][1]["fact_id"],
        )
    for index in range(20):
        _query(
            queries,
            "large",
            "relation_path",
            source_id=occurrence_ids[index],
            target_id=occurrence_ids[index + 1],
        )
        _query(
            queries,
            "large",
            "occurrence_to_selected_facts",
            occurrence_id=occurrence_ids[index],
            selected_fact_ids=[facts_by_occurrence[index][2]["fact_id"]],
        )
    return queries


def build_mixed_dag(scale: str) -> dict[str, Any]:
    config = SCALE_CONFIG[scale]
    builder = RuntimeScenarioBuilder(label=f"mixed-dag-{scale}-v1")
    occurrence_ids: list[str] = []
    facts_by_occurrence: list[list[dict[str, Any]]] = []
    for actor_index in range(config["actor_count"]):
        actor_id = f"actor-{actor_index:03d}"
        scope_id = (
            "capture-incomplete"
            if scale == "large" and actor_index >= 90
            else "capture-complete"
        )
        for sequence_index in range(config["events_per_actor"]):
            occurrence, facts = builder.add_occurrence(
                actor_id=actor_id,
                sequence_index=sequence_index,
                operation="mixed-controlled-operation",
                semantic_slot=actor_index * config["events_per_actor"]
                + sequence_index,
                scope_id=scope_id,
                fact_count=3,
            )
            occurrence_ids.append(
                occurrence["concrete_occurrence_instance_id"]
            )
            facts_by_occurrence.append(facts)
    builder.add_all_program_order()

    if scale == "small":
        active_count = 100
        message_target_offset = 50
        sync_release_base = 70
        generated_target_offset = 40
        reads_target_offset = 70
        conflict_target_offset = 80
    elif scale == "medium":
        active_count = 1000
        message_target_offset = 200
        sync_release_base = 600
        generated_target_offset = 400
        reads_target_offset = 500
        conflict_target_offset = 700
    else:
        active_count = 8000
        message_target_offset = 4000
        sync_release_base = 5000
        generated_target_offset = 2000
        reads_target_offset = 2500
        conflict_target_offset = 3000

    for index in range(config["message_count"]):
        builder.add_message(
            occurrence_ids[index],
            occurrence_ids[index + message_target_offset],
            channel_id=f"queue-{index % 17:02d}",
            payload={"batch_index": index, "scale": scale},
        )
    for index in range(config["sync_count"]):
        builder.add_synchronization(
            [occurrence_ids[2 * index], occurrence_ids[2 * index + 1]],
            occurrence_ids[sync_release_base + index],
            generation=index,
        )
    for index in range(config["generated_origin_count"]):
        builder.add_generated_origin(
            facts_by_occurrence[index][1],
            facts_by_occurrence[index + generated_target_offset][0],
        )
    for index in range(config["reads_from_count"]):
        builder.add_reads_from(
            facts_by_occurrence[index][2],
            facts_by_occurrence[index + reads_target_offset][2],
            resource_id=f"versioned-resource-{index:06d}",
            version_id=f"version-{index:06d}",
        )
    for index in range(config["conflict_count"]):
        builder.add_conflict(
            facts_by_occurrence[index][0],
            facts_by_occurrence[index + conflict_target_offset][1],
            resource_id=f"conflict-hotspot-{index % 23:02d}",
            version_id=f"hot-version-{index // 23:05d}",
        )
    if scale == "large":
        builder.add_unknown_edge(
            "capture-incomplete", "intentionally-unclassified-external-edge"
        )

    if scale == "small":
        queries = _small_queries(occurrence_ids, facts_by_occurrence)
    elif scale == "medium":
        queries = _medium_queries(occurrence_ids, facts_by_occurrence)
    else:
        queries = _large_queries(occurrence_ids, facts_by_occurrence)
    ordinary_output = {
        "scale": scale,
        "value": sum(range(active_count)),
        "ordered_labels": [f"result-{index:03d}" for index in range(5)],
    }
    return {
        "scale": scale,
        "builder": builder,
        "queries": queries,
        "query_manifest_sha256": canonical_sha256(queries),
        "ordinary_output": ordinary_output,
        "ordinary_output_bytes_sha256": canonical_sha256(ordinary_output),
    }
