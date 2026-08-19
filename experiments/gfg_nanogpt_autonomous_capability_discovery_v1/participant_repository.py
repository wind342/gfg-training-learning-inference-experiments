from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from .common import file_sha256, write_json


CLIENT_SOURCE = r'''from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class GFG:
    def __init__(self, root=None, max_step=None):
        self.root = Path(root or os.environ.get("NANOGPT_GFG_ROOT", "/evidence"))
        self.max_step = max_step
        self.database = self.root / "participant_gfg.sqlite3"
        self.connection = sqlite3.connect(
            "file:" + self.database.as_posix() + "?mode=ro&immutable=1", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.log_path = Path(os.environ.get(
            "GFG_QUERY_LOG", "submission/query_log.jsonl"))

    def _log(self, operation, arguments, count):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"operation": operation, "arguments": arguments, "count": count}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(row) + "\n")

    def _blocks(self, min_step=None, max_step=None, stage=None):
        upper = self.max_step if max_step is None else max_step
        clauses, values = [], []
        if min_step is not None:
            clauses.append("optimizer_step>=?"); values.append(min_step)
        if upper is not None:
            clauses.append("optimizer_step<=?"); values.append(upper)
        if stage is not None:
            clauses.append("stage=?"); values.append(stage)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        for row in self.connection.execute(
                "SELECT * FROM graph_blocks" + where +
                " ORDER BY block_ordinal", values):
            yield row, json.loads(zlib.decompress(
                row["payload_zlib"]).decode("utf-8"))

    def summary(self):
        where, values = ("", ())
        if self.max_step is not None:
            where, values = (" WHERE optimizer_step<=?", (self.max_step,))
        row = self.connection.execute(
            """SELECT COUNT(*) blocks,COALESCE(SUM(object_count),0) objects,
               COALESCE(SUM(occurrence_count),0) occurrences,
               COALESCE(SUM(fact_count),0) facts,
               COALESCE(SUM(explicit_edge_count),0) explicit_edges
               FROM graph_blocks""" + where, values).fetchone()
        result = dict(row)
        self._log("summary", {"max_step": self.max_step}, 1)
        return result

    def evaluations(self):
        where, values = ("", ())
        if self.max_step is not None:
            where, values = (" WHERE optimizer_step<=?", (self.max_step,))
        rows = [dict(row) for row in self.connection.execute(
            "SELECT * FROM evaluations" + where +
            " ORDER BY optimizer_step", values)]
        self._log("evaluations", {"max_step": self.max_step}, len(rows))
        return rows

    def objects(self, role=None, name_contains=None, min_step=None,
                max_step=None, materialized=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["objects"]:
                if role is not None and row["role"] != role: continue
                if name_contains is not None and name_contains not in row["name"]: continue
                if materialized is not None and bool(row["materialized"]) != materialized: continue
                rows.append(row)
        self._log("objects", {
            "role": role, "name_contains": name_contains,
            "min_step": min_step, "max_step": max_step,
            "materialized": materialized}, len(rows))
        return rows

    def occurrences(self, occurrence_type=None, min_step=None, max_step=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["occurrences"]:
                if occurrence_type is None or row["occurrence_type"] == occurrence_type:
                    rows.append(row)
        self._log("occurrences", {
            "occurrence_type": occurrence_type,
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def fact_blocks(self, min_step=None, max_step=None):
        rows = [row for _header, block in self._blocks(min_step, max_step)
                for row in block["fact_blocks"]]
        self._log("fact_blocks", {
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def edges(self, relation_type=None, min_step=None, max_step=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["edges"]:
                if relation_type is None or row["relation_type"] == relation_type:
                    rows.append(row)
        self._log("edges", {
            "relation_type": relation_type,
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def load_tensor(self, object_row):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TENSOR_LOADING_REQUIRES_NUMPY; graph queries remain available"
            ) from exc
        if not object_row["materialized"]:
            raise ValueError("TENSOR_REQUIRES_DETERMINISTIC_REPLAY")
        prefix = "objects://"
        if not object_row["locator"].startswith(prefix):
            raise ValueError("NOT_A_MATERIALIZED_TENSOR")
        value = np.load(self.root / "tensor-objects" /
                        object_row["locator"][len(prefix):],
                        allow_pickle=False)
        digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if digest != object_row["content_sha256"]:
            raise ValueError("TENSOR_CONTENT_HASH_MISMATCH")
        self._log("load_tensor", {
            "object_id": object_row["object_id"]}, int(value.size))
        return value

    def close(self):
        self.connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "summary", "evaluations", "objects", "occurrences",
        "fact-blocks", "edges"])
    parser.add_argument("--root")
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--role")
    parser.add_argument("--name-contains")
    parser.add_argument("--occurrence-type")
    parser.add_argument("--relation-type")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    gfg = GFG(args.root, args.max_step)
    try:
        if args.command == "summary": result = gfg.summary()
        elif args.command == "evaluations": result = gfg.evaluations()
        elif args.command == "objects": result = gfg.objects(
            args.role, args.name_contains, args.min_step, args.max_step)
        elif args.command == "occurrences": result = gfg.occurrences(
            args.occurrence_type, args.min_step, args.max_step)
        elif args.command == "fact-blocks": result = gfg.fact_blocks(
            args.min_step, args.max_step)
        else: result = gfg.edges(
            args.relation_type, args.min_step, args.max_step)
        if isinstance(result, list): result = result[:args.limit]
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    finally:
        gfg.close()

if __name__ == "__main__":
    main()
'''


TASK_TEXT = """You may read one complete, real, validated participant-safe
nanoGPT training generation-fact graph mounted at /evidence. This graph is
your only scientific input.

This executable mechanism-discovery task family is new. The five-minute
orientation manual defines its workflow. The sealed executable mechanism is
the scientific claim; discovery_report.md is supporting explanation. Every
variable described as causally necessary for a required prediction must be
represented in executable state or update code, unless a recorded
falsification test shows that it can be safely eliminated. Test state closure
on multiple prefix-only replays before sealing, and audit every state that can
continue to evolve under the proposed intervention.

Before sealing, perform an executable-use audit. Merely reading, serializing
or reporting a proposed state variable does not make it part of the operative
mechanism. For each variable claimed to control a required result, identify
and test the exact code path through which it can change that result. Reject
an executable that reduces a rich state explanation to an unsupported
single-run clock regularity when the GFG exposes prefix state capable of
conditioning the future.

Autonomously discover one unified executable finite-state theory containing
two explicit submodels: `FormationDynamics`, which explains how this training
execution changed from training-sample memorization to rule-level
generalization, and `StabilityDynamics`, which explains the subsequent
retention, degradation and recovery of that capability. You are not being
asked to optimize source code or merely describe an accuracy curve.

In addition to capability formation, determine whether the formed capability
is dynamically stable. Test, rather than assume, whether the graph supports
stable generalization, transient degradation followed by recovery, persistent
degradation, or an undetermined result. If instability is supported, identify
its observable preconditions and recovery mechanism from generation facts.
Treat this as an independent state-closure problem, not as an auxiliary label
or diagnostic appended after the formation forecast. At multiple historical
cuts, actively search for comparable compressed stability states with
different subsequent behavior. Such a counterexample means that the proposed
state is insufficient: retain additional future-distinguishing state and make
it operational in the executable update law before sealing. A fixed time
difference, growth rate, state increment, threshold, cycle number or cycle
parity from one discovery execution cannot by itself carry a cross-run
stability claim unless multiple prefix-only tests support its prospective use.
Derive the predicted intervention stability effect from the same closed state
and update law. The stability submodel must recursively predict the complete
future horizon, including the location, duration, amplitude and recovery of
all degradation it predicts. Predicting only the next instability event,
returning only an interval list, or replaying a precomputed cycle calendar is
not a complete stability model. These requirements reveal no target variable,
event time, threshold or answer.

Your sealed mechanism must initialize from a previously unseen run's complete
GFG prefix and forecast, before the future occurs:
1. whether a capability transition will occur by step 10000;
2. two independently frozen transition intervals: a high-precision interval
   no wider than 200 optimizer steps and a primary success interval no wider
   than 500 optimizer steps;
3. predicted_formation_curve on the 100-step evaluation grid, with rows
   `{"step": integer, "capability": number}`;
4. predicted_stability_degradation_curve on the same grid, with rows
   `{"step": integer, "degradation": nonnegative number}`;
5. predicted_validation_curve on the same grid, with each accuracy exactly
   equal to clamp(capability - degradation, 0, 1);
6. mechanism_state containing formation_state, stability_state and a
   state_trajectory row for every forecast step; each trajectory row must
   contain step, formation_state and stability_state;
7. post_formation_stability, using exactly STABLE,
   TRANSIENT_DEGRADATION_RECOVERY, PERSISTENT_DEGRADATION or UNDETERMINED; and
8. predicted_instability_intervals as a list of zero or more inclusive
   optimizer-step intervals, each with integer step_low and step_high.

`mechanism.py` must define `FormationDynamics`, `StabilityDynamics` and
`CapabilityDynamicsMechanism`. Each submodel must define initialize, step and
output methods. The unified mechanism must define initialize and forecast,
must return initial formation_state and stability_state, and must actually
execute both submodels to produce the component curves and state trajectory.
After formation, do not encode a predicted degradation by lowering only the
formation curve: every predicted deficit below 0.90 must be caused by a
positive stability-degradation output.

Submit one executable intervention that acts only through the frozen training
hook API and predicts ADVANCE or DELAY, a transition-step shift interval, and
the corresponding change in your mechanism state. Also declare the predicted
post-formation stability effect as IMPROVE, WORSEN or NO_CHANGE and explain
which mechanism variable should change. The same sealed candidate
will be evaluated on an unseen task, token map, split, initialization and data
order. It must not identify runs or dates and must not contain a trajectory
answer table.

Use the complete graph freely: sources, batch membership, tensors, layer
activations, logits, gradients, clipping, optimizer states, parameter
versions, exact occurrence identities, roles, reads_from, GeneratedOrigin,
program_order and realizes_fact. You may derive matrices and temporary files.
The helper gfg_client.py is optional; direct read-only SQLite access is also
allowed. Do not use the network, another AI, external literature, or hidden
material.

Use transition_step_low_200 and transition_step_high_200 for the 200-step
high-precision interval. Use transition_step_low_500 and
transition_step_high_500 for the 500-step primary interval. The 200-step
interval is reported as an additional precision diagnostic; the 500-step
interval is the interval-containment gate for forecast success. Both are
sealed before the unseen future is generated.

After the external runner releases /evidence, the formal-work budget is
exactly 120 minutes. Stop expanding analysis by formal-work minute 110 and
use the final ten minutes only to complete and check every required file in
submission/. You may submit earlier as soon as the required work is ready.
The runner does not provide an interactive last-minute message.

The sealed mechanism and intervention are replayed in a restricted runtime.
They may import ordinary numerical modules and gfg_client, but must not
import filesystem, process, dynamic-import, system or network modules; call
open; retain an absolute evidence path in serialized state; or perform
run-identity, date or wall-clock lookup. Temporary analysis scripts used
before submission are not subject to this executable-candidate restriction.

Before submitting, propose and attempt to falsify at least three mechanisms,
including the selected mechanism and at least two rejected alternatives.
Record the GFG evidence for each in discovery_report.md. At least one tested
hypothesis must concern post-formation stability. Do not assume that a
transient degradation exists merely because this task asks you to test for it.

Write exactly the required candidate files under submission/ as specified by
candidate_interface.json and intervention_api.json. Do not modify /evidence.
After all candidate files have been completely written and checked, create
`submission/FINAL_SUBMISSION_READY.json` last with the exact JSON object
`{"status":"READY"}`. The external runner may then terminate an otherwise
idle Codex process and preserve the completed submission; this marker is not
part of the scientific answer.

Before creating the READY marker, run
`python DUAL_DYNAMICS_SUBMISSION_CHECKER.py submission /evidence`. It checks
only the frozen executable interface and curve composition, supplies no
scientific answer, and must report PASS.
"""


INTERFACE_TEXT = """# Participant GFG interface

The evidence database contains append-only, hash-chained `graph_blocks`.
Each zlib-compressed canonical JSON payload contains:

- `objects`: exact tensor/literal identities, content hashes, dtype, shape,
  role, optimizer step and either an object locator or exact replay locator;
- `occurrences`: concrete transformations and occurrence identities;
- `fact_blocks`: a reversible encoding of atomic generation facts;
- `edges`: explicit `GeneratedOrigin` and `program_order` relations.

A fact block contains exactly one outcome and only the ordered
`(source, relation_role)` entries that actually participated in forming that
outcome. It is not a source-by-outcome Cartesian template. The atomic identity
is the canonical hash of scope, source object, occurrence, outcome object and
relation role. `realizes_fact`,
`origin_incidence`, `outcome_incidence` and `reads_from` follow directly.
Expansion preserves multiplicity; it never joins by step, value or time.

`gfg_client.GFG` supplies summary, evaluation, object, occurrence, fact-block,
edge and materialized-tensor access. It logs helper calls, but helper usage is
not a success criterion. Nonmaterialized objects retain their exact content
hash and deterministic replay locator. Evaluation-grid, boundary and batch
objects are directly materialized.
"""


def prepare_participant_repository(
    *,
    repository: Path,
    evidence_directory: Path,
    instance_id: str,
    contracts_directory: Path,
) -> dict[str, Any]:
    if repository.exists():
        raise RuntimeError("PARTICIPANT_REPOSITORY_ALREADY_EXISTS")
    repository.mkdir(parents=True)
    submission = repository / "submission"
    submission.mkdir()
    (repository / "TASK.txt").write_text(TASK_TEXT, encoding="utf-8", newline="\n")
    (repository / "GFG_INTERFACE.md").write_text(
        INTERFACE_TEXT, encoding="utf-8", newline="\n"
    )
    (repository / "gfg_client.py").write_text(
        CLIENT_SOURCE, encoding="utf-8", newline="\n"
    )
    orientation_directory = Path(__file__).resolve().parent / "orientation"
    shutil.copy2(
        orientation_directory / "GFG_MACHINE_SEMANTICS.md",
        repository / "GFG_MACHINE_SEMANTICS.md",
    )
    shutil.copy2(
        orientation_directory / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md",
        repository / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md",
    )
    shutil.copy2(
        orientation_directory / "DUAL_DYNAMICS_SUBMISSION_CHECKER.py",
        repository / "DUAL_DYNAMICS_SUBMISSION_CHECKER.py",
    )
    shutil.copy2(
        orientation_directory / "ORIENTATION_RECEIPT_CHECKER.py",
        repository / "ORIENTATION_RECEIPT_CHECKER.py",
    )
    shutil.copy2(
        orientation_directory / "unrelated_example.json",
        repository / "unrelated_example.json",
    )
    for name in (
        "candidate_interface.json",
        "intervention_api.json",
        "forecast_validation.json",
        "stability_validation.json",
        "causal_validation.json",
        "final_decision_rule.json",
        "scientific_protocol.json",
        "capability_transition.json",
        "training_time_alignment.json",
        "gfg_orientation.json",
    ):
        shutil.copy2(contracts_directory / name, repository / name)
    write_json(
        repository / "participant_manifest.json",
        {
            "evidence_database_sha256": file_sha256(
                evidence_directory / "participant_gfg.sqlite3"
            ),
            "evidence_manifest_sha256": file_sha256(
                evidence_directory / "capture_manifest.json"
            ),
            "instance_id": instance_id,
            "mechanism_discovery_guide_sha256": file_sha256(
                repository / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
            ),
            "orientation_receipt_checker_sha256": file_sha256(
                repository / "ORIENTATION_RECEIPT_CHECKER.py"
            ),
            "scientific_input": ("complete validated participant-safe GFG only"),
        },
    )
    subprocess.run(
        ["git", "init", "-q"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "GFG Experiment"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "gfg-experiment@invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Freeze participant interface"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    return {
        "evidence_directory": str(evidence_directory.resolve()),
        "instance_id": instance_id,
        "participant_repository_commit": commit,
        "repository": str(repository.resolve()),
        "orientation_receipt_checker_sha256": file_sha256(
            repository / "ORIENTATION_RECEIPT_CHECKER.py"
        ),
        "task_sha256": file_sha256(repository / "TASK.txt"),
        "task_guide_sha256": file_sha256(
            repository / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
        ),
    }
