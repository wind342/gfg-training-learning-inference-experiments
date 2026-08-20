from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.analysis import (
    _support_values as diffusion_support_values,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.boundary import (
    residual_boundary,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.data import (
    DiffusionSchedule,
    evaluation_pack,
    loaders as diffusion_loaders,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.model import (
    CifarDiffusionUNet,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.runner import (
    _seed_everything as seed_diffusion,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.analysis import (
    _support_values as resnet_support_values,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.data import (
    loaders as resnet_loaders,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.model import (
    CifarResNet18,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.numeric import (
    sha256_file,
    state_sha256,
    target_margins,
    write_json,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.runner import (
    _seed_everything as seed_resnet,
)

from .analysis import (
    clone_model_state,
    component_call_capture,
    hybrid_state,
    object_hash,
    support_diagnostics,
)
from .gfg import ProjectionGFG


PACKAGE = Path(__file__).resolve().parent
CONTRACT = json.loads((PACKAGE / "MODEL_CONTRACT.json").read_text(encoding="utf-8"))
RESNET_CONTRACT = json.loads(
    (
        PACKAGE.parent
        / "gfg_resnet_cifar_training_learning_generalization_v1"
        / "MODEL_CONTRACT.json"
    ).read_text(encoding="utf-8")
)
DIFFUSION_CONTRACT = json.loads(
    (
        PACKAGE.parent
        / "gfg_ddpm_cifar_training_learning_generalization_v1"
        / "MODEL_CONTRACT.json"
    ).read_text(encoding="utf-8")
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_integrity(checkpoint: Path) -> dict[str, Any]:
    manifest = _read(checkpoint.parent / "MANIFEST.json")
    expected = manifest["files"][checkpoint.name]
    actual = sha256_file(checkpoint)
    return {
        "checkpoint": checkpoint.as_posix(),
        "expected_sha256": expected["sha256"],
        "actual_sha256": actual,
        "byte_count": checkpoint.stat().st_size,
        "pass": actual == expected["sha256"] and checkpoint.stat().st_size == expected["bytes"],
    }


def _initial_resnet(
    seed: int, data_root: Path, device: torch.device
) -> tuple[CifarResNet18, dict[str, Tensor], Tensor, Tensor, Tensor]:
    seed_resnet(seed)
    training = RESNET_CONTRACT["training"]
    evaluation = RESNET_CONTRACT["evaluation"]
    _, anchor_loader, _ = resnet_loaders(
        data_root,
        seed,
        int(training["batch_size"]),
        int(evaluation["anchor_count"]),
        download=False,
    )
    images, labels, identities = next(iter(anchor_loader))
    images = images.to(device)
    labels = labels.to(device)
    identities = identities.to(device)
    model = CifarResNet18().to(device)
    return model, clone_model_state(model), images, labels, identities


def _initial_diffusion(
    seed: int, data_root: Path, device: torch.device
) -> tuple[CifarDiffusionUNet, dict[str, Tensor], dict[str, Tensor], DiffusionSchedule]:
    seed_diffusion(seed)
    profile = DIFFUSION_CONTRACT["formal"]
    system = DIFFUSION_CONTRACT["system"]
    _, test_loader = diffusion_loaders(
        data_root,
        batch_size=int(profile["batch_size"]),
        seed=seed,
        download=False,
        test_batch_size=max(int(profile["batch_size"]), int(profile["anchor_count"])),
    )
    schedule = DiffusionSchedule.linear(
        int(system["diffusion_steps"]),
        float(system["beta_schedule"][0]),
        float(system["beta_schedule"][1]),
        device,
    )
    evaluation = evaluation_pack(
        test_loader,
        count=int(profile["anchor_count"]),
        seed=seed,
        candidate_count=int(system["candidate_count"]),
        candidate_scale=float(system["candidate_scale"]),
        schedule=schedule,
        device=device,
    )
    model = CifarDiffusionUNet().to(device)
    return model, clone_model_state(model), evaluation, schedule


def _single_gate_effects_resnet(
    model: CifarResNet18, images: Tensor, baseline: Tensor
) -> dict[str, float]:
    result: dict[str, float] = {}
    with torch.no_grad():
        for index, name in enumerate(CifarResNet18.component_names):
            gates = [1.0] * 4
            gates[index] = 0.0
            value = model(images, tuple(gates))
            result[name] = float((value - baseline).abs().max())
    return result


def _single_gate_effects_diffusion(
    model: CifarDiffusionUNet,
    noisy: Tensor,
    timesteps: Tensor,
    baseline: Tensor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    with torch.no_grad():
        for index, name in enumerate(CifarDiffusionUNet.component_names):
            gates = [1.0] * 4
            gates[index] = 0.0
            value = model(noisy, timesteps, tuple(gates))
            result[name] = float((value - baseline).abs().max())
    return result


def _diffusion_sample(
    model: CifarDiffusionUNet,
    schedule: DiffusionSchedule,
    initial_noise: Tensor,
) -> Tensor:
    value = initial_noise.detach().clone()
    with torch.no_grad():
        for step in range(len(schedule.alpha_bar) - 1, -1, -1):
            timesteps = torch.full(
                (len(value),), step, dtype=torch.long, device=value.device
            )
            epsilon = model(value, timesteps)
            alpha_bar = schedule.alpha_bar[step]
            prior_alpha_bar = (
                schedule.alpha_bar[step - 1]
                if step > 0
                else torch.ones((), device=value.device)
            )
            clean = (
                value - torch.sqrt(1.0 - alpha_bar) * epsilon
            ) / torch.sqrt(alpha_bar)
            value = (
                torch.sqrt(prior_alpha_bar) * clean
                + torch.sqrt(1.0 - prior_alpha_bar) * epsilon
            )
    return value


def _tests(
    *,
    system: str,
    checkpoint_exact: bool,
    initial_exact: bool,
    final_exact: bool,
    model_unchanged: bool,
    optimizer_unchanged: bool,
    repeat_error: float,
    component_rms: dict[str, float],
    gate_effects: dict[str, float],
    support: dict[str, Any],
    rollback_effects: dict[str, dict[str, float]],
    restoration_error: float,
    restoration_hash_exact: bool,
) -> dict[str, bool]:
    thresholds = CONTRACT["tests"]
    interaction_threshold = float(
        thresholds[f"{system}_interaction_minimum"]
    )
    return {
        "source_version_identity": checkpoint_exact and initial_exact and final_exact,
        "persistent_state_frozen": model_unchanged and optimizer_unchanged,
        "repeat_inference_exact": repeat_error <= float(thresholds["repeat_output_tolerance"]),
        "components_called": all(
            value > float(thresholds["component_output_rms_minimum"])
            for value in component_rms.values()
        ),
        "gate_changes_output": max(gate_effects.values(), default=0.0)
        > float(thresholds["gate_effect_minimum"]),
        "query_conditioned_support": support["maximum_query_profile_l1"]
        > float(thresholds["query_profile_l1_minimum"]),
        "nonadditive_combination": support["maximum_absolute_pair_interaction"]
        > interaction_threshold,
        "learned_version_dependence": max(
            (max(values.values()) for values in rollback_effects.values()),
            default=0.0,
        )
        > float(thresholds["rollback_effect_minimum"]),
        "exact_restoration": restoration_hash_exact
        and restoration_error <= float(thresholds["restoration_tolerance"]),
    }


def run_resnet_seed(
    *,
    seed: int,
    data_root: Path,
    checkpoint_root: Path,
    device: torch.device,
    graph: ProjectionGFG,
) -> dict[str, Any]:
    model, initial, images, labels, identities = _initial_resnet(seed, data_root, device)
    source = checkpoint_root / f"seed_{seed}"
    source_events = _read(source / "EVENTS.json")
    source_summary = _read(source / "RUN_SUMMARY.json")
    checkpoint_path = source / "FINAL_CHECKPOINT.pt"
    checkpoint_integrity = _checkpoint_integrity(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    initial_hash = state_sha256(initial)
    expected_initial_hash = source_events[0]["analysis"]["pre_state_sha256"]
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    trained = clone_model_state(model)
    trained_hash = state_sha256(trained)
    expected_trained_hash = source_summary["final_state_sha256"]
    optimizer_before = object_hash(checkpoint["optimizer_state"])
    model_before = state_sha256(clone_model_state(model))
    query_count = int(CONTRACT["systems"]["resnet"]["query_count"])
    images = images[:query_count]
    labels = labels[:query_count]
    identities = identities[:query_count]
    components = [
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
    ]
    with torch.no_grad():
        baseline, component_rms = component_call_capture(
            components, lambda: model(images)
        )
        repeated = model(images)
        support_values = resnet_support_values(model, trained, images, labels)
    repeat_error = float((baseline - repeated).abs().max())
    support = support_diagnostics(support_values)
    gate_effects = _single_gate_effects_resnet(model, images, baseline)
    rollbacks: dict[str, dict[str, float]] = {}
    maximum_restoration_error = 0.0
    restorations_exact = True
    for name, _module in components:
        hybrid = hybrid_state(trained, initial, (name + ".",))
        model.load_state_dict(hybrid, strict=True)
        model.eval()
        with torch.no_grad():
            changed = model(images)
        rollbacks[name] = {
            "complete_logit_max_abs_change": float((changed - baseline).abs().max()),
            "correct_prediction_change_fraction": float(
                changed.argmax(dim=1).ne(baseline.argmax(dim=1)).float().mean()
            ),
        }
        model.load_state_dict(trained, strict=True)
        model.eval()
        with torch.no_grad():
            restored = model(images)
        error = float((restored - baseline).abs().max())
        maximum_restoration_error = max(maximum_restoration_error, error)
        restorations_exact = restorations_exact and state_sha256(
            clone_model_state(model)
        ) == trained_hash
        graph.add(
            occurrence_id=f"resnet:{seed}:rollback:{name}",
            kind="component_version_rollback",
            source=f"trained:{trained_hash}|pre_learning:{initial_hash}",
            transformation=f"replace:{name}:trained_to_pre_learning",
            outcome=f"logit_change:{rollbacks[name]['complete_logit_max_abs_change']}",
            role="causal_version_intervention",
            evidence=rollbacks[name],
        )
    model_after = state_sha256(clone_model_state(model))
    optimizer_after = object_hash(checkpoint["optimizer_state"])
    tests = _tests(
        system="resnet",
        checkpoint_exact=checkpoint_integrity["pass"],
        initial_exact=initial_hash == expected_initial_hash,
        final_exact=trained_hash == expected_trained_hash,
        model_unchanged=model_before == model_after,
        optimizer_unchanged=optimizer_before == optimizer_after,
        repeat_error=repeat_error,
        component_rms=component_rms,
        gate_effects=gate_effects,
        support=support,
        rollback_effects=rollbacks,
        restoration_error=maximum_restoration_error,
        restoration_hash_exact=restorations_exact,
    )
    result = {
        "system": "resnet",
        "seed": seed,
        "query_identities": identities.detach().cpu().tolist(),
        "source_identity": {
            "checkpoint": checkpoint_integrity,
            "initial_state_sha256": initial_hash,
            "expected_initial_state_sha256": expected_initial_hash,
            "trained_state_sha256": trained_hash,
            "expected_trained_state_sha256": expected_trained_hash,
        },
        "frozen_state": {
            "model_before": model_before,
            "model_after": model_after,
            "optimizer_before": optimizer_before,
            "optimizer_after": optimizer_after,
            "repeat_output_max_abs_error": repeat_error,
        },
        "component_output_rms": component_rms,
        "single_gate_complete_output_effects": gate_effects,
        "support": support,
        "rollbacks": rollbacks,
        "maximum_restoration_output_error": maximum_restoration_error,
        "tests": tests,
        "pass": all(tests.values()),
    }
    graph.add(
        occurrence_id=f"resnet:{seed}:frozen_inference",
        kind="frozen_inference",
        source=f"checkpoint:{checkpoint_integrity['actual_sha256']}",
        transformation="resnet_query_conditioned_projection",
        outcome=f"complete_logits:{object_hash(baseline)}",
        role="trained_support_projection",
        evidence={"tests": tests, "component_output_rms": component_rms},
    )
    graph.add(
        occurrence_id=f"resnet:{seed}:support_coalitions",
        kind="support_intervention",
        source=f"queries:{object_hash(identities)}",
        transformation="all_sixteen_stage_gate_coalitions",
        outcome=f"maximum_interaction:{support['maximum_absolute_pair_interaction']}",
        role="support_combination_adjudication",
        evidence={
            "maximum_query_profile_l1": support["maximum_query_profile_l1"],
            "maximum_absolute_pair_interaction": support["maximum_absolute_pair_interaction"],
        },
    )
    return result


def run_diffusion_seed(
    *,
    seed: int,
    data_root: Path,
    checkpoint_root: Path,
    device: torch.device,
    graph: ProjectionGFG,
) -> dict[str, Any]:
    model, initial, evaluation, schedule = _initial_diffusion(seed, data_root, device)
    source = checkpoint_root / f"seed_{seed}"
    source_events = _read(source / "EVENTS.json")
    checkpoint_path = source / "FINAL_CHECKPOINT.pt"
    checkpoint_integrity = _checkpoint_integrity(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    initial_hash = state_sha256(initial)
    expected_initial_hash = source_events[0]["analysis"]["pre_state_sha256"]
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    trained = clone_model_state(model)
    trained_hash = state_sha256(trained)
    optimizer_before = object_hash(checkpoint["optimizer"])
    model_before = state_sha256(clone_model_state(model))
    query_count = int(CONTRACT["systems"]["diffusion"]["query_count"])
    noisy = evaluation["noisy"][:query_count]
    timesteps = evaluation["timesteps"][:query_count]
    true_noise = evaluation["true_noise"][:query_count]
    candidates = evaluation["candidates"][:query_count]
    identities = evaluation["identities"][:query_count]
    components = [
        ("high_resolution_skip", model.encoder_high),
        ("low_resolution_skip", model.encoder_low),
        ("bottleneck", model.middle),
        ("decoder_refinement", model.refinement),
    ]
    with torch.no_grad():
        baseline, component_rms = component_call_capture(
            components, lambda: model(noisy, timesteps)
        )
        repeated = model(noisy, timesteps)
        support_values = diffusion_support_values(
            model, trained, noisy, timesteps, true_noise, candidates
        )
    repeat_error = float((baseline - repeated).abs().max())
    support = support_diagnostics(support_values)
    gate_effects = _single_gate_effects_diffusion(
        model, noisy, timesteps, baseline
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 500_000)
    sample_noise = torch.randn(
        (
            int(CONTRACT["systems"]["diffusion"]["sample_count"]),
            3,
            32,
            32,
        ),
        generator=sample_generator,
        device=device,
    )
    sample_baseline = _diffusion_sample(model, schedule, sample_noise)
    sample_repeated = _diffusion_sample(model, schedule, sample_noise)
    sample_repeat_error = float((sample_baseline - sample_repeated).abs().max())
    repeat_error = max(repeat_error, sample_repeat_error)
    prefix_map = {
        "high_resolution_skip": ("encoder_high.",),
        "low_resolution_skip": ("encoder_low.",),
        "bottleneck": ("middle.",),
        "decoder_refinement": ("refinement.",),
    }
    rollbacks: dict[str, dict[str, float]] = {}
    maximum_restoration_error = 0.0
    restorations_exact = True
    for name, _module in components:
        hybrid = hybrid_state(trained, initial, prefix_map[name])
        model.load_state_dict(hybrid, strict=True)
        model.eval()
        with torch.no_grad():
            changed = model(noisy, timesteps)
        changed_sample = _diffusion_sample(model, schedule, sample_noise)
        changed_boundary = residual_boundary(changed, true_noise, candidates)[0]
        baseline_boundary = residual_boundary(baseline, true_noise, candidates)[0]
        rollbacks[name] = {
            "epsilon_max_abs_change": float((changed - baseline).abs().max()),
            "boundary_change_rms": float(
                (changed_boundary - baseline_boundary).double().square().mean().sqrt()
            ),
            "complete_sample_max_abs_change": float(
                (changed_sample - sample_baseline).abs().max()
            ),
        }
        model.load_state_dict(trained, strict=True)
        model.eval()
        with torch.no_grad():
            restored = model(noisy, timesteps)
        restored_sample = _diffusion_sample(model, schedule, sample_noise)
        error = max(
            float((restored - baseline).abs().max()),
            float((restored_sample - sample_baseline).abs().max()),
        )
        maximum_restoration_error = max(maximum_restoration_error, error)
        restorations_exact = restorations_exact and state_sha256(
            clone_model_state(model)
        ) == trained_hash
        graph.add(
            occurrence_id=f"diffusion:{seed}:rollback:{name}",
            kind="component_version_rollback",
            source=f"trained:{trained_hash}|pre_learning:{initial_hash}",
            transformation=f"replace:{name}:trained_to_pre_learning",
            outcome=f"sample_change:{rollbacks[name]['complete_sample_max_abs_change']}",
            role="causal_version_intervention",
            evidence=rollbacks[name],
        )
    model_after = state_sha256(clone_model_state(model))
    optimizer_after = object_hash(checkpoint["optimizer"])
    tests = _tests(
        system="diffusion",
        checkpoint_exact=checkpoint_integrity["pass"],
        initial_exact=initial_hash == expected_initial_hash,
        final_exact=True,
        model_unchanged=model_before == model_after,
        optimizer_unchanged=optimizer_before == optimizer_after,
        repeat_error=repeat_error,
        component_rms=component_rms,
        gate_effects=gate_effects,
        support=support,
        rollback_effects=rollbacks,
        restoration_error=maximum_restoration_error,
        restoration_hash_exact=restorations_exact,
    )
    result = {
        "system": "diffusion",
        "seed": seed,
        "query_identities": identities.detach().cpu().tolist(),
        "source_identity": {
            "checkpoint": checkpoint_integrity,
            "initial_state_sha256": initial_hash,
            "expected_initial_state_sha256": expected_initial_hash,
            "trained_state_sha256": trained_hash,
        },
        "frozen_state": {
            "model_before": model_before,
            "model_after": model_after,
            "optimizer_before": optimizer_before,
            "optimizer_after": optimizer_after,
            "one_step_repeat_max_abs_error": float((baseline - repeated).abs().max()),
            "complete_sample_repeat_max_abs_error": sample_repeat_error,
        },
        "component_output_rms": component_rms,
        "single_gate_complete_output_effects": gate_effects,
        "support": support,
        "rollbacks": rollbacks,
        "maximum_restoration_output_error": maximum_restoration_error,
        "tests": tests,
        "pass": all(tests.values()),
    }
    graph.add(
        occurrence_id=f"diffusion:{seed}:frozen_inference",
        kind="frozen_inference",
        source=f"checkpoint:{checkpoint_integrity['actual_sha256']}",
        transformation="diffusion_query_conditioned_projection_and_sampling",
        outcome=f"complete_sample:{object_hash(sample_baseline)}",
        role="trained_support_projection",
        evidence={"tests": tests, "component_output_rms": component_rms},
    )
    graph.add(
        occurrence_id=f"diffusion:{seed}:support_coalitions",
        kind="support_intervention",
        source=f"queries:{object_hash(identities)}",
        transformation="all_sixteen_unet_route_gate_coalitions",
        outcome=f"maximum_interaction:{support['maximum_absolute_pair_interaction']}",
        role="support_combination_adjudication",
        evidence={
            "maximum_query_profile_l1": support["maximum_query_profile_l1"],
            "maximum_absolute_pair_interaction": support["maximum_absolute_pair_interaction"],
        },
    )
    return result


def aggregate(results: list[dict[str, Any]], graph_document: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, dict[str, Any]] = {}
    for system in ("resnet", "diffusion"):
        rows = [row for row in results if row["system"] == system]
        systems[system] = {
            "seed_count": len(rows),
            "passing_seeds": sum(bool(row["pass"]) for row in rows),
            "all_seeds_pass": bool(rows) and all(row["pass"] for row in rows),
            "maximum_query_profile_l1": max(
                row["support"]["maximum_query_profile_l1"] for row in rows
            ),
            "maximum_absolute_pair_interaction": max(
                row["support"]["maximum_absolute_pair_interaction"] for row in rows
            ),
            "maximum_rollback_effect": max(
                value
                for row in rows
                for effects in row["rollbacks"].values()
                for value in effects.values()
            ),
            "maximum_restoration_output_error": max(
                row["maximum_restoration_output_error"] for row in rows
            ),
        }
    integrity = graph_document["validation"]["status"] == "PASS"
    all_pass = integrity and all(row["pass"] for row in results)
    if not integrity:
        verdict = "INTEGRITY_FAILURE"
    elif all_pass:
        verdict = "CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED"
    elif any(row["pass"] for row in results):
        verdict = "CROSS_SYSTEM_FROZEN_PROJECTION_PARTIALLY_SUPPORTED"
    else:
        verdict = "CROSS_SYSTEM_FROZEN_PROJECTION_NOT_SUPPORTED"
    return {
        "schema": "gfg-cross-system-frozen-inference-projection-formal-results-v1",
        "verdict": verdict,
        "gfg_validation": graph_document["validation"],
        "systems": systems,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resnet-data-root",
        type=Path,
        default=Path(os.environ.get("GFG_CIFAR100_ROOT", "runtime/datasets/cifar100")),
    )
    parser.add_argument(
        "--diffusion-data-root",
        type=Path,
        default=Path(os.environ.get("GFG_CIFAR10_ROOT", "runtime/datasets/cifar10")),
    )
    parser.add_argument(
        "--resnet-checkpoint-root",
        type=Path,
        default=Path(CONTRACT["systems"]["resnet"]["checkpoint_root"]),
    )
    parser.add_argument(
        "--diffusion-checkpoint-root",
        type=Path,
        default=Path(CONTRACT["systems"]["diffusion"]["checkpoint_root"]),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime/gfg_cross_system_frozen_inference_projection_v1"),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"OUTPUT_ROOT_EXISTS:{args.output_root}")
    args.output_root.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    graph = ProjectionGFG()
    results: list[dict[str, Any]] = []
    for seed in CONTRACT["systems"]["resnet"]["seeds"]:
        row = run_resnet_seed(
            seed=int(seed),
            data_root=args.resnet_data_root,
            checkpoint_root=args.resnet_checkpoint_root,
            device=device,
            graph=graph,
        )
        results.append(row)
        print(json.dumps({"system": "resnet", "seed": seed, "pass": row["pass"]}), flush=True)
    for seed in CONTRACT["systems"]["diffusion"]["seeds"]:
        row = run_diffusion_seed(
            seed=int(seed),
            data_root=args.diffusion_data_root,
            checkpoint_root=args.diffusion_checkpoint_root,
            device=device,
            graph=graph,
        )
        results.append(row)
        print(json.dumps({"system": "diffusion", "seed": seed, "pass": row["pass"]}), flush=True)
    graph_document = graph.document()
    formal = aggregate(results, graph_document)
    write_json(args.output_root / "RUN_RESULTS.json", results)
    write_json(args.output_root / "FORMAL_GFG.json", graph_document)
    write_json(args.output_root / "FORMAL_RESULTS.json", formal)
    print(json.dumps({"verdict": formal["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
