from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


HORIZON = 64
CUE_BITS = 6
ACTION_COUNT = 2
FUNCTIONAL_CODES = tuple(range(1, 7))
PASSENGER_CODES = tuple(range(7, 10))
DISTRACTOR_CODE = 0
EVENT_CODE_COUNT = 10
METHODS = (
    "gfg_forks",
    "trace_decomposition_forks",
    "temporal_recency_forks",
    "rewired_gfg_forks",
    "gfg_ancestry_only",
    "terminal_all_actions",
    "oracle_forks",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _bit(cue: tuple[int, ...], *indices: int) -> int:
    value = 0
    for index in indices:
        value ^= int(cue[index])
    return value


def target_actions(cue: tuple[int, ...]) -> tuple[int, ...]:
    """Hidden evaluator rule.  It is never written to participant GFG payloads."""
    return (
        _bit(cue, 0, 3),
        _bit(cue, 1, 4),
        _bit(cue, 2, 5),
        _bit(cue, 0, 2, 4),
        _bit(cue, 1, 3, 5),
        _bit(cue, 0, 1, 5),
    )


def deterministic_schedule(seed: int, episode_id: int) -> dict[int, int]:
    """Map opaque event codes to variable, non-overlapping occurrence positions."""
    rng = random.Random(seed * 1_000_003 + episode_id * 97 + 29)
    windows = {
        1: (2, 8),
        7: (7, 12),
        2: (11, 18),
        3: (19, 27),
        8: (24, 32),
        4: (29, 37),
        5: (38, 48),
        9: (45, 54),
        6: (53, 61),
    }
    schedule: dict[int, int] = {}
    used: set[int] = set()
    for code, (low, high) in windows.items():
        choices = [position for position in range(low, high + 1) if position not in used]
        position = choices[rng.randrange(len(choices))]
        schedule[code] = position
        used.add(position)
    return schedule


def event_codes_from_schedule(schedule: dict[int, int]) -> tuple[int, ...]:
    codes = [DISTRACTOR_CODE] * HORIZON
    for code, position in schedule.items():
        if codes[position] != DISTRACTOR_CODE:
            raise RuntimeError("TCD_SCHEDULE_COLLISION")
        codes[position] = code
    return tuple(codes)


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    episode_id: int
    cue: tuple[int, ...]
    schedule: dict[int, int]
    event_codes: tuple[int, ...]

    @property
    def functional_positions(self) -> tuple[int, ...]:
        return tuple(self.schedule[code] for code in FUNCTIONAL_CODES)

    @property
    def passenger_positions(self) -> tuple[int, ...]:
        return tuple(self.schedule[code] for code in PASSENGER_CODES)

    @property
    def ancestry_positions(self) -> tuple[int, ...]:
        return tuple(sorted((*self.functional_positions, *self.passenger_positions)))


@dataclass(frozen=True)
class EpisodeResult:
    consequence: float
    success: bool
    criteria: tuple[bool, bool, bool, bool]
    report: dict[str, int]
    event_outputs: tuple[dict[str, Any], ...]


def make_episode_spec(seed: int, episode_id: int) -> EpisodeSpec:
    rng = random.Random(seed * 2_000_033 + episode_id * 193 + 11)
    cue = tuple(rng.randrange(2) for _ in range(CUE_BITS))
    schedule = deterministic_schedule(seed, episode_id)
    return EpisodeSpec(
        seed=seed,
        episode_id=episode_id,
        cue=cue,
        schedule=schedule,
        event_codes=event_codes_from_schedule(schedule),
    )


def execute_episode(spec: EpisodeSpec, actions: Iterable[int]) -> EpisodeResult:
    action_values = tuple(int(value) for value in actions)
    if len(action_values) != HORIZON or any(value not in (0, 1) for value in action_values):
        raise ValueError("TCD_ACTION_LEDGER_INVALID")
    slots: dict[int, int] = {}
    outputs: list[dict[str, Any]] = []
    for step, (code, action) in enumerate(zip(spec.event_codes, action_values)):
        if code == DISTRACTOR_CODE:
            outputs.append({"step": step, "event_code": code, "formed_slot": None})
            continue
        slots[code] = action
        outputs.append({
            "step": step,
            "event_code": code,
            "formed_slot": f"slot-{code}",
            "slot_value": action,
        })
    expected = target_actions(spec.cue)
    route_ok = slots[1] == expected[0]
    source_ok = slots[2] == expected[1] or slots[3] == expected[2]
    catalyst_ok = slots[4] == expected[3] and slots[5] == expected[4]
    finish_ok = slots[6] == expected[5]
    criteria = (route_ok, source_ok, catalyst_ok, finish_ok)
    success = all(criteria)
    # Only this scalar is delivered as environmental consequence.  Component
    # criteria remain evaluator truth and are not available to credit methods.
    consequence = float(sum(criteria) / len(criteria) + (1.0 if success else 0.0))
    report = {f"slot-{code}": slots[code] for code in (*FUNCTIONAL_CODES, *PASSENGER_CODES)}
    return EpisodeResult(
        consequence=consequence,
        success=success,
        criteria=criteria,
        report=report,
        event_outputs=tuple(outputs),
    )


def deterministic_behavior_actions(seed: int, episode_id: int) -> tuple[int, ...]:
    rng = random.Random(seed * 3_000_017 + episode_id * 389 + 101)
    return tuple(rng.randrange(ACTION_COUNT) for _ in range(HORIZON))


def terminal_consequence_only(spec: EpisodeSpec, actions: tuple[int, ...]) -> float:
    """Semantics-preserving terminal projection used by high-volume replays."""
    expected = target_actions(spec.cue)
    at = lambda code: actions[spec.schedule[code]]
    route_ok = at(1) == expected[0]
    source_ok = at(2) == expected[1] or at(3) == expected[2]
    catalyst_ok = at(4) == expected[3] and at(5) == expected[4]
    finish_ok = at(6) == expected[5]
    criteria = (route_ok, source_ok, catalyst_ok, finish_ok)
    return float(sum(criteria) / len(criteria) + (1.0 if all(criteria) else 0.0))


def build_base_gfg(spec: EpisodeSpec, actions: tuple[int, ...], result: EpisodeResult) -> dict[str, Any]:
    """Build an identity-preserving execution graph without causal-credit labels."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for step, (code, action) in enumerate(zip(spec.event_codes, actions)):
        action_id = f"episode-{spec.episode_id}:action-{step}"
        occurrence_id = f"episode-{spec.episode_id}:occurrence-{step}"
        nodes.extend((
            {"node_id": action_id, "node_kind": "source", "payload": {"step": step, "action": action}},
            {"node_id": occurrence_id, "node_kind": "occurrence", "payload": {"step": step, "event_code": code}},
        ))
        edges.append({"edge_kind": "participates_in", "source": action_id, "target": occurrence_id})
        if step:
            edges.append({
                "edge_kind": "program_order",
                "source": f"episode-{spec.episode_id}:occurrence-{step - 1}",
                "target": occurrence_id,
            })
        if code != DISTRACTOR_CODE:
            slot_id = f"episode-{spec.episode_id}:slot-{code}"
            nodes.append({
                "node_id": slot_id,
                "node_kind": "fact",
                "payload": {"opaque_slot_identity": f"slot-{code}", "value": action},
            })
            edges.append({"edge_kind": "realizes_fact", "source": occurrence_id, "target": slot_id})
    terminal_occurrence = f"episode-{spec.episode_id}:terminal-occurrence"
    terminal_fact = f"episode-{spec.episode_id}:terminal-consequence"
    nodes.extend((
        {"node_id": terminal_occurrence, "node_kind": "occurrence", "payload": {"stage": "terminal"}},
        {"node_id": terminal_fact, "node_kind": "fact", "payload": {"consequence": result.consequence}},
    ))
    for code in (*FUNCTIONAL_CODES, *PASSENGER_CODES):
        edges.append({
            "edge_kind": "consumed_by",
            "source": f"episode-{spec.episode_id}:slot-{code}",
            "target": terminal_occurrence,
        })
    edges.append({"edge_kind": "realizes_fact", "source": terminal_occurrence, "target": terminal_fact})
    graph = {
        "schema": "temporal-credit-base-gfg-v1",
        "episode_id": spec.episode_id,
        "terminal_fact_node_id": terminal_fact,
        "nodes": nodes,
        "edges": edges,
        "forbidden_credit_edge_kinds_absent": True,
    }
    graph["graph_sha256"] = object_sha256(graph)
    return graph


def validate_base_gfg(graph: dict[str, Any]) -> dict[str, Any]:
    if graph.get("schema") != "temporal-credit-base-gfg-v1":
        raise RuntimeError("TCD_GFG_SCHEMA_INVALID")
    supplied_hash = graph.get("graph_sha256")
    body = {key: value for key, value in graph.items() if key != "graph_sha256"}
    if supplied_hash != object_sha256(body):
        raise RuntimeError("TCD_GFG_HASH_INVALID")
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise RuntimeError("TCD_GFG_DUPLICATE_NODE")
    allowed = {"participates_in", "program_order", "realizes_fact", "consumed_by"}
    for edge in graph["edges"]:
        if edge["edge_kind"] not in allowed:
            raise RuntimeError("TCD_GFG_EDGE_KIND_INVALID")
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise RuntimeError("TCD_GFG_EDGE_ENDPOINT_MISSING")
    forbidden = {"credited_to_action", "causes_consequence", "necessary", "synergy", "backup"}
    if any(edge["edge_kind"] in forbidden for edge in graph["edges"]):
        raise RuntimeError("TCD_GFG_CREDIT_LEAK")
    return {"status": "PASS", "node_count": len(nodes), "edge_count": len(graph["edges"])}


def retrieve_formation_candidates(graph: dict[str, Any]) -> tuple[int, ...]:
    """Traverse only native formation edges; program order is deliberately excluded."""
    validate_base_gfg(graph)
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    reverse: dict[str, list[str]] = {}
    forward: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge["edge_kind"] == "program_order":
            continue
        reverse.setdefault(edge["target"], []).append(edge["source"])
        forward.setdefault(edge["source"], []).append(edge["target"])
    terminal = graph["terminal_fact_node_id"]
    stack = [terminal]
    seen = {terminal}
    while stack:
        current = stack.pop()
        for prior in reverse.get(current, []):
            if prior not in seen:
                seen.add(prior)
                stack.append(prior)
    result = []
    for node_id in seen:
        node = nodes[node_id]
        if node["node_kind"] == "source" and node_id.split(":")[-1].startswith("action-"):
            result.append(int(node["payload"]["step"]))
    return tuple(sorted(result))


def rewire_graph(graph: dict[str, Any], seed: int) -> dict[str, Any]:
    """Degree- and type-preserving rewiring of fact-to-terminal relations."""
    value = json.loads(json.dumps(graph))
    rng = random.Random(seed)
    consumed = [edge for edge in value["edges"] if edge["edge_kind"] == "consumed_by"]
    fact_nodes = [row["node_id"] for row in value["nodes"] if row["node_kind"] == "fact" and ":slot-" in row["node_id"]]
    rng.shuffle(fact_nodes)
    # Replace some admitted slot ancestors with facts produced by randomly
    # chosen distractor occurrences, keeping the number of incoming edges.
    distractor_occurrences = [
        row["node_id"] for row in value["nodes"]
        if row["node_kind"] == "occurrence" and row.get("payload", {}).get("event_code") == DISTRACTOR_CODE
    ]
    rng.shuffle(distractor_occurrences)
    replace_count = min(len(consumed), len(distractor_occurrences))
    for index in range(replace_count):
        edge = consumed[index]
        synthetic_id = f"{distractor_occurrences[index]}:rewired-fact"
        value["nodes"].append({
            "node_id": synthetic_id,
            "node_kind": "fact",
            "payload": {"rewired_control": True},
        })
        value["edges"].append({
            "edge_kind": "realizes_fact",
            "source": distractor_occurrences[index],
            "target": synthetic_id,
        })
        edge["source"] = synthetic_id
    value.pop("graph_sha256", None)
    value["graph_sha256"] = object_sha256(value)
    return value


def replay_with_flips(spec: EpisodeSpec, actions: tuple[int, ...], flipped_steps: Iterable[int]) -> EpisodeResult:
    changed = list(actions)
    for step in flipped_steps:
        changed[int(step)] = 1 - changed[int(step)]
    consequence = terminal_consequence_only(spec, tuple(changed))
    # The compact object deliberately exposes only the field used in a replay.
    return EpisodeResult(consequence, False, (False, False, False, False), {}, ())


def exact_shapley_credits(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    candidates: tuple[int, ...],
) -> tuple[dict[int, float], int]:
    """Exact Shapley effects of retaining actual actions rather than flipping them."""
    ordered = tuple(sorted(set(int(value) for value in candidates)))
    n = len(ordered)
    if n == 0:
        return {}, 0
    values: dict[int, float] = {}
    all_mask = (1 << n) - 1
    for mask in range(1 << n):
        # Coalition members retain their actual action.  Non-members are flipped.
        flipped = [ordered[index] for index in range(n) if not (mask & (1 << index))]
        values[mask] = replay_with_flips(spec, actions, flipped).consequence
    factorial = [math.factorial(index) for index in range(n + 1)]
    denominator = math.factorial(n)
    credits: dict[int, float] = {}
    for index, step in enumerate(ordered):
        bit = 1 << index
        total = 0.0
        for mask in range(1 << n):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = factorial[size] * factorial[n - size - 1] / denominator
            total += weight * (values[mask | bit] - values[mask])
        credits[step] = float(total)
    if all_mask not in values:
        raise RuntimeError("TCD_SHAPLEY_LEDGER_INCOMPLETE")
    return credits, len(values)


def pair_interactions(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    candidates: tuple[int, ...],
) -> tuple[dict[str, float], int]:
    base = execute_episode(spec, actions).consequence
    singles = {
        step: replay_with_flips(spec, actions, [step]).consequence - base
        for step in candidates
    }
    interactions: dict[str, float] = {}
    for left, right in itertools.combinations(candidates, 2):
        joint = replay_with_flips(spec, actions, [left, right]).consequence - base
        interactions[f"{left}:{right}"] = float(joint - singles[left] - singles[right])
    return interactions, len(candidates) + math.comb(len(candidates), 2)


def trace_feature_vector(spec: EpisodeSpec, actions: tuple[int, ...]) -> np.ndarray:
    """Lossless chronological payload summarized without GFG relations."""
    features = np.zeros(EVENT_CODE_COUNT * (2 + CUE_BITS), dtype=np.float64)
    cue = np.asarray(spec.cue, dtype=np.float64)
    for code, action in zip(spec.event_codes, actions):
        base = code * (2 + CUE_BITS)
        features[base + action] += 1.0
        features[base + 2 : base + 2 + CUE_BITS] += cue * (1.0 if action else -1.0)
    return features


@dataclass(frozen=True)
class TraceDecompositionModel:
    coefficients: tuple[float, ...]
    intercept: float

    def step_scores(self, spec: EpisodeSpec, actions: tuple[int, ...]) -> tuple[float, ...]:
        coef = np.asarray(self.coefficients, dtype=np.float64)
        cue = np.asarray(spec.cue, dtype=np.float64)
        width = 2 + CUE_BITS
        scores = []
        for code, action in zip(spec.event_codes, actions):
            base = code * width
            contribution = coef[base + action]
            contribution += float(np.dot(coef[base + 2 : base + width], cue * (1.0 if action else -1.0)))
            scores.append(abs(float(contribution)))
        return tuple(scores)


def fit_trace_decomposition(
    rows: list[tuple[EpisodeSpec, tuple[int, ...], EpisodeResult]],
    ridge: float,
) -> TraceDecompositionModel:
    matrix = np.stack([trace_feature_vector(spec, actions) for spec, actions, _ in rows])
    targets = np.asarray([result.consequence for _, _, result in rows], dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    normalized = (matrix - means) / scales
    augmented = np.concatenate([normalized, np.ones((len(rows), 1))], axis=1)
    regularizer = np.eye(augmented.shape[1]) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + regularizer, augmented.T @ targets)
    coefficients = weights[:-1] / scales
    intercept = float(weights[-1] - np.dot(means, coefficients))
    return TraceDecompositionModel(tuple(float(value) for value in coefficients), intercept)


def top_k_trace_candidates(
    model: TraceDecompositionModel,
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    k: int,
) -> tuple[int, ...]:
    scores = model.step_scores(spec, actions)
    ranked = sorted(range(HORIZON), key=lambda step: (-scores[step], step))
    return tuple(sorted(ranked[:k]))


class CreditPolicy(nn.Module):
    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CUE_BITS + EVENT_CODE_COUNT, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, ACTION_COUNT),
        )

    def forward(self, cue: torch.Tensor, event_code: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(event_code.long(), EVENT_CODE_COUNT).float()
        return self.network(torch.cat((cue.float(), one_hot), dim=1))


def credit_training_examples(
    method: str,
    rows: list[tuple[EpisodeSpec, tuple[int, ...], EpisodeResult, dict[str, Any]]],
    trace_model: TraceDecompositionModel,
    candidate_budget: int,
    rewiring_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    fork_count = 0
    selected_counts: list[int] = []
    for row_index, (spec, actions, result, graph) in enumerate(rows):
        if method == "gfg_forks" or method == "gfg_ancestry_only":
            candidates = retrieve_formation_candidates(graph)
        elif method == "trace_decomposition_forks":
            candidates = top_k_trace_candidates(trace_model, spec, actions, candidate_budget)
        elif method == "temporal_recency_forks":
            candidates = tuple(range(HORIZON - candidate_budget, HORIZON))
        elif method == "rewired_gfg_forks":
            candidates = retrieve_formation_candidates(rewire_graph(graph, rewiring_seed + row_index))
            candidates = candidates[:candidate_budget]
        elif method == "oracle_forks":
            candidates = spec.functional_positions
        elif method == "terminal_all_actions":
            candidates = tuple(range(HORIZON))
        else:
            raise ValueError(f"unknown method {method}")
        selected_counts.append(len(candidates))
        if method.endswith("_forks"):
            credits, used = exact_shapley_credits(spec, actions, candidates)
            fork_count += used
            for step, credit in credits.items():
                if abs(credit) <= 1e-12:
                    continue
                actual = actions[step]
                target = actual if credit > 0 else 1 - actual
                examples.append({
                    "cue": spec.cue,
                    "event_code": spec.event_codes[step],
                    "target_action": target,
                    # Causal effect magnitude and learning weight are distinct.
                    # A validated nonzero relation supplies the action target;
                    # every such relation gets one learning unit so that a
                    # smaller terminal component is not silently treated as a
                    # less important capability.
                    "weight": 1.0,
                    "causal_effect": credit,
                    "episode_id": spec.episode_id,
                    "step": step,
                })
        else:
            centered = result.consequence - 0.75
            for step in candidates:
                actual = actions[step]
                target = actual if centered >= 0 else 1 - actual
                weight = max(abs(centered), 0.05)
                examples.append({
                    "cue": spec.cue,
                    "event_code": spec.event_codes[step],
                    "target_action": target,
                    "weight": weight,
                    "episode_id": spec.episode_id,
                    "step": step,
                })
    return examples, {
        "method": method,
        "example_count": len(examples),
        "counterfactual_replay_count": fork_count,
        "mean_candidate_count": float(np.mean(selected_counts)),
    }


def train_credit_policy(
    examples: list[dict[str, Any]],
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
) -> tuple[CreditPolicy, dict[str, Any]]:
    seed_everything(seed)
    model = CreditPolicy(hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if not examples:
        return model, {"epochs": 0, "final_loss": None}
    cue = torch.tensor([row["cue"] for row in examples], dtype=torch.float32)
    event_code = torch.tensor([row["event_code"] for row in examples], dtype=torch.long)
    targets = torch.tensor([row["target_action"] for row in examples], dtype=torch.long)
    weights = torch.tensor([row["weight"] for row in examples], dtype=torch.float32)
    weights = weights / weights.mean().clamp_min(1e-12)
    final_loss = 0.0
    for _ in range(epochs):
        logits = model(cue, event_code)
        per_example = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        loss = (per_example * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    return model, {"epochs": epochs, "final_loss": final_loss}


@torch.no_grad()
def evaluate_credit_policy(model: CreditPolicy, specs: list[EpisodeSpec]) -> dict[str, Any]:
    consequences: list[float] = []
    successes: list[bool] = []
    functional_correct = 0
    functional_total = 0
    for spec in specs:
        cue_rows = torch.tensor([spec.cue] * HORIZON, dtype=torch.float32)
        codes = torch.tensor(spec.event_codes, dtype=torch.long)
        actions = tuple(int(value) for value in model(cue_rows, codes).argmax(dim=1).tolist())
        result = execute_episode(spec, actions)
        consequences.append(result.consequence)
        successes.append(result.success)
        expected = target_actions(spec.cue)
        for code, target in zip(FUNCTIONAL_CODES, expected):
            functional_correct += actions[spec.schedule[code]] == target
            functional_total += 1
    return {
        "episode_count": len(specs),
        "mean_terminal_consequence": float(np.mean(consequences)),
        "terminal_success_rate": float(np.mean(successes)),
        "functional_action_accuracy": functional_correct / functional_total,
        "per_episode_consequence": consequences,
        "per_episode_success": successes,
    }


def candidate_metrics(selected: tuple[int, ...], truth: tuple[int, ...]) -> dict[str, float | int]:
    selected_set = set(selected)
    truth_set = set(truth)
    tp = len(selected_set & truth_set)
    fp = len(selected_set - truth_set)
    fn = len(truth_set - selected_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


__all__ = [
    "ACTION_COUNT",
    "CreditPolicy",
    "EpisodeResult",
    "EpisodeSpec",
    "FUNCTIONAL_CODES",
    "HORIZON",
    "METHODS",
    "PASSENGER_CODES",
    "TraceDecompositionModel",
    "build_base_gfg",
    "candidate_metrics",
    "credit_training_examples",
    "deterministic_behavior_actions",
    "evaluate_credit_policy",
    "exact_shapley_credits",
    "execute_episode",
    "fit_trace_decomposition",
    "make_episode_spec",
    "object_sha256",
    "pair_interactions",
    "retrieve_formation_candidates",
    "rewire_graph",
    "seed_everything",
    "target_actions",
    "top_k_trace_candidates",
    "train_credit_policy",
    "validate_base_gfg",
]
