from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

import numpy as np

from .common import canonical_bytes, file_sha256, payload_sha256, write_json
from .training_capture import decode_block


def expanded_fact(
    run_id: str,
    fact_block: dict[str, Any],
    source: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    material = {
        "domain_scope_id": run_id,
        "occurrence": fact_block["occurrence_id"],
        "origin": source["object_id"],
        "outcome": outcome["object_id"],
        "relation_role": source["relation_role"],
    }
    fact_sha = payload_sha256(material)
    return {
        "evidence_id": "evidence_"
        + payload_sha256(
            {
                "authority": ("actual-synchronous-nanogpt-training-capture-v2"),
                "fact_sha256": fact_sha,
            }
        ),
        "fact_id": "fact_" + fact_sha,
        "fact_sha256": fact_sha,
        "fact_block_id": fact_block["fact_block_id"],
        "occurrence_id": fact_block["occurrence_id"],
        "outcome_object_id": outcome["object_id"],
        "outcome_role": outcome.get("outcome_role"),
        "relation_role": source["relation_role"],
        "source_object_id": source["object_id"],
    }


class TrainingGFG:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row
        self.run_id = json.loads(
            self.connection.execute(
                "SELECT value_json FROM metadata WHERE key='run_id'"
            ).fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()

    def blocks(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
        stage: str | None = None,
    ) -> Iterator[tuple[sqlite3.Row, dict[str, Any]]]:
        clauses: list[str] = []
        values: list[Any] = []
        if min_step is not None:
            clauses.append("optimizer_step>=?")
            values.append(min_step)
        if max_step is not None:
            clauses.append("optimizer_step<=?")
            values.append(max_step)
        if stage is not None:
            clauses.append("stage=?")
            values.append(stage)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        for row in self.connection.execute(
            "SELECT * FROM graph_blocks" + where + " ORDER BY block_ordinal",
            values,
        ):
            yield row, decode_block(row["payload_zlib"])

    def objects(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
        name_contains: str | None = None,
        role: str | None = None,
        materialized: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        for _row, block in self.blocks(max_step=max_step, min_step=min_step):
            for obj in block["objects"]:
                if role is not None and obj["role"] != role:
                    continue
                if name_contains is not None and name_contains not in obj["name"]:
                    continue
                if (
                    materialized is not None
                    and bool(obj["materialized"]) != materialized
                ):
                    continue
                yield obj

    def occurrences(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
        occurrence_type: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for _row, block in self.blocks(max_step=max_step, min_step=min_step):
            for occurrence in block["occurrences"]:
                if (
                    occurrence_type is not None
                    and occurrence["occurrence_type"] != occurrence_type
                ):
                    continue
                yield occurrence

    def fact_blocks(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        for _row, block in self.blocks(max_step=max_step, min_step=min_step):
            yield from block["fact_blocks"]

    def facts(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
        outcome_object_id: str | None = None,
        source_object_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for block in self.fact_blocks(max_step=max_step, min_step=min_step):
            for source in block["sources"]:
                if (
                    source_object_id is not None
                    and source["object_id"] != source_object_id
                ):
                    continue
                for outcome in block["outcomes"]:
                    if (
                        outcome_object_id is not None
                        and outcome["object_id"] != outcome_object_id
                    ):
                        continue
                    yield expanded_fact(self.run_id, block, source, outcome)

    def edges(
        self,
        *,
        max_step: int | None = None,
        min_step: int | None = None,
        relation_type: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for _row, block in self.blocks(max_step=max_step, min_step=min_step):
            for edge in block["edges"]:
                if relation_type is None or edge["relation_type"] == relation_type:
                    yield edge
            for fact_block in block["fact_blocks"]:
                for source in fact_block["sources"]:
                    for outcome in fact_block["outcomes"]:
                        fact = expanded_fact(
                            self.run_id,
                            fact_block,
                            source,
                            outcome,
                        )
                        derived = (
                            (
                                "realizes_fact",
                                fact["occurrence_id"],
                                fact["fact_id"],
                            ),
                            (
                                "origin_incidence",
                                fact["source_object_id"],
                                fact["fact_id"],
                            ),
                            (
                                "outcome_incidence",
                                fact["fact_id"],
                                fact["outcome_object_id"],
                            ),
                            (
                                "reads_from",
                                fact["source_object_id"],
                                fact["outcome_object_id"],
                            ),
                        )
                        for kind, source_id, target_id in derived:
                            if relation_type is not None and kind != relation_type:
                                continue
                            material = {
                                "fact_id": fact["fact_id"],
                                "primitive_or_derived": "primitive",
                                "relation_type": kind,
                                "source_id": source_id,
                                "target_id": target_id,
                            }
                            yield {
                                "edge_id": "edge_" + payload_sha256(material),
                                **material,
                            }

    def evaluations(self, *, max_step: int | None = None) -> list[dict[str, Any]]:
        where = " WHERE optimizer_step<=?" if max_step is not None else ""
        args = (max_step,) if max_step is not None else ()
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM evaluations" + where + " ORDER BY optimizer_step",
                args,
            )
        ]

    def object_by_id(self, object_id: str) -> dict[str, Any]:
        for obj in self.objects():
            if obj["object_id"] == object_id:
                return obj
        raise KeyError(object_id)

    def load_materialized_tensor(self, object_id: str) -> np.ndarray:
        obj = self.object_by_id(object_id)
        if not obj["materialized"]:
            raise ValueError("TENSOR_REQUIRES_DETERMINISTIC_REPLAY")
        prefix = "objects://"
        if not obj["locator"].startswith(prefix):
            raise ValueError("OBJECT_LOCATOR_NOT_TENSOR_FILE")
        return np.load(
            self.database_path.parent
            / "tensor-objects"
            / obj["locator"][len(prefix) :],
            allow_pickle=False,
        )

    def summary(self, *, max_step: int | None = None) -> dict[str, Any]:
        if max_step is None:
            row = self.connection.execute(
                """
                SELECT COUNT(*) blocks,COALESCE(SUM(object_count),0) objects,
                       COALESCE(SUM(occurrence_count),0) occurrences,
                       COALESCE(SUM(fact_count),0) facts,
                       COALESCE(SUM(explicit_edge_count),0) explicit_edges
                FROM graph_blocks
                """
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT COUNT(*) blocks,COALESCE(SUM(object_count),0) objects,
                       COALESCE(SUM(occurrence_count),0) occurrences,
                       COALESCE(SUM(fact_count),0) facts,
                       COALESCE(SUM(explicit_edge_count),0) explicit_edges
                FROM graph_blocks WHERE optimizer_step<=?
                """,
                (max_step,),
            ).fetchone()
        return {
            **dict(row),
            "database_sha256": file_sha256(self.database_path),
            "evaluation_count": len(self.evaluations(max_step=max_step)),
            "max_step": max_step,
            "run_id": self.run_id,
            "schema": "participant-training-gfg-summary-v1",
        }


def _semantic_binding_valid(
    *,
    occurrence_type: str,
    outcome_role: str,
    source_roles: list[str],
    binding_payload: dict[str, Any],
) -> bool:
    roles = Counter(source_roles)
    if occurrence_type == "batch_materialization":
        expected = {
            "training_batch_inputs": Counter(
                ["selected_dataset_inputs", "selection_order"]
            ),
            "training_batch_targets": Counter(
                ["selected_dataset_targets", "selection_order"]
            ),
        }
        return outcome_role in expected and roles == expected[outcome_role]
    if occurrence_type == "layer_forward":
        return bool(source_roles) and all(
            role.startswith("layer_input_") or role == "layer_parameter"
            for role in source_roles
        )
    if occurrence_type == "training_forward":
        return (
            outcome_role == "training_logits"
            and roles["input_tokens"] == 1
            and roles["parameter_version"] > 0
            and sum(roles.values()) == 1 + roles["parameter_version"]
        )
    if occurrence_type == "training_loss":
        return outcome_role == "training_loss" and roles == Counter(
            ["forward_logits", "target_tokens"]
        )
    if occurrence_type == "autograd_backward":
        return outcome_role == "parameter_gradient" and roles == Counter(
            ["loss_seed", "differentiated_parameter"]
        )
    if occurrence_type == "gradient_global_norm":
        return (
            outcome_role == "gradient_total_norm"
            and bool(source_roles)
            and set(source_roles) == {"norm_input_gradient"}
        )
    if occurrence_type == "gradient_clip_application":
        if outcome_role == "clipped_parameter_gradient":
            return roles == Counter(
                [
                    "unclipped_gradient",
                    "global_gradient_norm",
                    "clip_configuration",
                ]
            )
        if outcome_role == "explicit_disposition":
            return roles == Counter(["unclipped_gradient", "clip_configuration"])
        return False
    if occurrence_type == "optimizer_parameter_update":
        component = binding_payload.get("outcome_component")
        gradient_count = roles["clipped_gradient"] + roles["gradient_disposition"]
        if component == "parameter":
            return (
                outcome_role == "parameter_version"
                and roles["parameter_before_update"] == 1
                and gradient_count == 1
                and roles["optimizer_configuration_for_parameter_update"] == 1
                and all(
                    role
                    in {
                        "parameter_before_update",
                        "clipped_gradient",
                        "gradient_disposition",
                        "optimizer_configuration_for_parameter_update",
                    }
                    or role.endswith("_before_parameter_update")
                    for role in source_roles
                )
            )
        state_name = binding_payload.get("optimizer_state")
        if component != "optimizer_state" or outcome_role != "optimizer_state":
            return False
        configuration_role = f"optimizer_configuration_for_{state_name}"
        if roles[configuration_role] != 1:
            return False
        if state_name == "step":
            return roles["optimizer_state_owner"] == 1 and all(
                role
                in {
                    configuration_role,
                    "optimizer_state_owner",
                    "step_before_update",
                }
                for role in source_roles
            )
        if gradient_count != 1:
            return False
        allowed = {
            configuration_role,
            "clipped_gradient",
            "gradient_disposition",
            f"{state_name}_before_update",
        }
        if state_name == "max_exp_avg_sq":
            allowed.add("exp_avg_sq_before_update")
        return all(role in allowed for role in source_roles)
    if occurrence_type == "evaluation_forward":
        return (
            outcome_role in {"train_logits", "validation_logits"}
            and roles["evaluation_inputs"] == 1
            and roles["evaluated_parameter_version"] > 0
            and sum(roles.values()) == 1 + roles["evaluated_parameter_version"]
        )
    if occurrence_type == "evaluation_prediction":
        return outcome_role in {
            "train_predictions",
            "validation_predictions",
        } and roles == Counter(["evaluation_logits"])
    if occurrence_type == "capability_evaluation":
        return outcome_role == "capability_evaluation" and roles == Counter(
            [
                "train_predictions",
                "train_targets",
                "validation_predictions",
                "validation_targets",
                "latest_training_loss",
            ]
        )
    return False


def validate_training_gfg(
    database_path: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    gfg = TrainingGFG(database_path)
    prior_block: str | None = None
    prior_occurrence: str | None = None
    object_ids: set[str] = set()
    object_hashes: dict[str, str] = {}
    object_names: dict[str, str] = {}
    object_roles: dict[str, str] = {}
    occurrence_ids: set[str] = set()
    occurrence_types: dict[str, str] = {}
    fact_ids: set[str] = set()
    edge_ids: set[str] = set()
    parameter_origins: dict[tuple[str, int], tuple[str, str]] = {}
    gates = {
        "block_hash_chain": True,
        "content_addressed_objects": True,
        "exact_fact_expansion": True,
        "exact_object_references": True,
        "exact_parameter_version_chain": True,
        "exact_program_order": True,
        "evaluation_parameter_binding": True,
        "no_approximate_temporal_join": True,
        "one_outcome_per_binding_block": True,
        "semantic_source_outcome_profiles": True,
    }
    try:
        for row, block in gfg.blocks():
            raw = canonical_bytes(block)
            if payload_sha256(block) != row["payload_sha256"]:
                gates["block_hash_chain"] = False
            expected_block = payload_sha256(
                {
                    "block_ordinal": row["block_ordinal"],
                    "optimizer_step": row["optimizer_step"],
                    "payload_sha256": row["payload_sha256"],
                    "prior_block_sha256": prior_block,
                    "run_id": gfg.run_id,
                    "stage": row["stage"],
                }
            )
            if (
                hashlib_sha256(raw) != row["payload_sha256"]
                or expected_block != row["block_sha256"]
                or row["prior_block_sha256"] != prior_block
            ):
                gates["block_hash_chain"] = False
            prior_block = row["block_sha256"]
            current_objects = {obj["object_id"]: obj for obj in block["objects"]}
            for object_id, obj in current_objects.items():
                if object_id in object_ids:
                    gates["content_addressed_objects"] = False
                expected_id = "obj_" + payload_sha256(
                    {
                        "content_sha256": obj["content_sha256"],
                        "dtype": obj["dtype"],
                        "run_id": gfg.run_id,
                        "semantic_key": obj["semantic_key"],
                        "shape": obj["shape"],
                    }
                )
                if object_id != expected_id:
                    gates["content_addressed_objects"] = False
                object_ids.add(object_id)
                object_hashes[object_id] = obj["content_sha256"]
                object_names[object_id] = obj["name"]
                object_roles[object_id] = obj["role"]
            for occurrence in block["occurrences"]:
                occurrence_id = occurrence["occurrence_id"]
                if occurrence_id in occurrence_ids:
                    gates["exact_program_order"] = False
                occurrence_ids.add(occurrence_id)
                occurrence_types[occurrence_id] = occurrence["occurrence_type"]
            for edge in block["edges"]:
                if edge["edge_id"] in edge_ids:
                    gates["exact_object_references"] = False
                edge_ids.add(edge["edge_id"])
                if edge["relation_type"] == "program_order":
                    if (
                        prior_occurrence is not None
                        and edge["source_id"] != prior_occurrence
                    ):
                        gates["exact_program_order"] = False
                    prior_occurrence = edge["target_id"]
                if edge["relation_type"] == "GeneratedOrigin":
                    payload = edge["payload"]
                    if "parameter_name" in payload and "optimizer_state" not in payload:
                        parameter_origins[
                            (
                                payload["parameter_name"],
                                payload["result_version"],
                            )
                        ] = (edge["source_id"], edge["target_id"])
            if block["occurrences"]:
                prior_occurrence = block["occurrences"][-1]["occurrence_id"]
            available = object_ids
            expanded_count = sum(
                len(fact_block["sources"])
                for fact_block in block["fact_blocks"]
            )
            if expanded_count != row["fact_count"]:
                gates["exact_fact_expansion"] = False
            for fact_block in block["fact_blocks"]:
                if fact_block["occurrence_id"] not in occurrence_ids:
                    gates["exact_object_references"] = False
                for source in fact_block["sources"]:
                    if source["object_id"] not in available:
                        gates["exact_object_references"] = False
                    elif (
                        source["content_sha256"]
                        != object_hashes[source["object_id"]]
                    ):
                        gates["exact_object_references"] = False
                for outcome in fact_block["outcomes"]:
                    if outcome["object_id"] not in available:
                        gates["exact_object_references"] = False
                    elif (
                        outcome["content_sha256"]
                        != object_hashes[outcome["object_id"]]
                        or outcome.get("outcome_role")
                        != object_roles[outcome["object_id"]]
                    ):
                        gates["exact_object_references"] = False
                if len(fact_block["outcomes"]) != 1:
                    gates["one_outcome_per_binding_block"] = False
                    gates["semantic_source_outcome_profiles"] = False
                elif not _semantic_binding_valid(
                    occurrence_type=occurrence_types.get(
                        fact_block["occurrence_id"], ""
                    ),
                    outcome_role=fact_block["outcomes"][0].get("outcome_role", ""),
                    source_roles=[
                        source["relation_role"] for source in fact_block["sources"]
                    ],
                    binding_payload=fact_block.get("payload", {}),
                ):
                    gates["semantic_source_outcome_profiles"] = False
                for source in fact_block["sources"]:
                    for outcome in fact_block["outcomes"]:
                        fact = expanded_fact(
                            gfg.run_id,
                            fact_block,
                            source,
                            outcome,
                        )
                        if fact["fact_id"] in fact_ids:
                            gates["exact_fact_expansion"] = False
                        fact_ids.add(fact["fact_id"])

        parameter_versions: dict[str, dict[int, str]] = {}
        for obj in gfg.objects(role="parameter_version"):
            version = int(obj["payload"]["parameter_version"])
            parameter_versions.setdefault(obj["name"], {})[version] = obj["object_id"]
        for name, versions in parameter_versions.items():
            first_version = min(versions)
            for version in range(first_version + 1, max(versions) + 1):
                edge = parameter_origins.get((name, version))
                if (
                    version not in versions
                    or version - 1 not in versions
                    or edge != (versions[version - 1], versions[version])
                ):
                    gates["exact_parameter_version_chain"] = False
                    break
        for evaluation in gfg.evaluations():
            bound_splits: set[str] = set()
            for block in gfg.fact_blocks(
                min_step=evaluation["optimizer_step"],
                max_step=evaluation["optimizer_step"],
            ):
                occurrence_type = occurrence_types.get(block["occurrence_id"])
                if occurrence_type != "evaluation_forward":
                    continue
                bound_names = {
                    object_names[source["object_id"]]
                    for source in block["sources"]
                    if source["relation_role"] == "evaluated_parameter_version"
                }
                if bound_names == set(parameter_versions):
                    outcome_role = block["outcomes"][0].get("outcome_role")
                    if outcome_role == "train_logits":
                        bound_splits.add("train")
                    elif outcome_role == "validation_logits":
                        bound_splits.add("validation")
            if bound_splits != {"train", "validation"}:
                gates["evaluation_parameter_binding"] = False
                break
        report = {
            "counts": gfg.summary(),
            "database_sha256": file_sha256(database_path),
            "gates": gates,
            "schema": "validated-participant-training-gfg-v1",
            "status": ("PASS" if all(gates.values()) else "GFG_CAPTURE_FAILURE"),
        }
        report["validation_sha256"] = payload_sha256(report)
        if report_path is not None:
            write_json(report_path, report)
        return report
    finally:
        gfg.close()


def hashlib_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _emit_rows(rows: Iterator[dict[str, Any]], limit: int) -> None:
    for index, row in enumerate(rows):
        if index >= limit:
            break
        print(_json_line(row))


def _json_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "command",
        choices=[
            "summary",
            "evaluations",
            "objects",
            "occurrences",
            "facts",
            "edges",
            "validate",
        ],
    )
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--role")
    parser.add_argument("--occurrence-type")
    parser.add_argument("--relation-type")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.command == "validate":
        print(_json_line(validate_training_gfg(args.database)))
        return
    gfg = TrainingGFG(args.database)
    try:
        if args.command == "summary":
            print(_json_line(gfg.summary(max_step=args.max_step)))
        elif args.command == "evaluations":
            print(_json_line(gfg.evaluations(max_step=args.max_step)))
        elif args.command == "objects":
            _emit_rows(
                gfg.objects(
                    max_step=args.max_step,
                    min_step=args.min_step,
                    role=args.role,
                ),
                args.limit,
            )
        elif args.command == "occurrences":
            _emit_rows(
                gfg.occurrences(
                    max_step=args.max_step,
                    min_step=args.min_step,
                    occurrence_type=args.occurrence_type,
                ),
                args.limit,
            )
        elif args.command == "facts":
            _emit_rows(
                gfg.facts(
                    max_step=args.max_step,
                    min_step=args.min_step,
                ),
                args.limit,
            )
        elif args.command == "edges":
            _emit_rows(
                gfg.edges(
                    max_step=args.max_step,
                    min_step=args.min_step,
                    relation_type=args.relation_type,
                ),
                args.limit,
            )
    finally:
        gfg.close()


if __name__ == "__main__":
    main()
