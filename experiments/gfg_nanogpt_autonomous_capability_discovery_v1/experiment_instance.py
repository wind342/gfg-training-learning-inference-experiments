from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .checkpoint_fork import save_checkpoint
from .common import payload_sha256, read_json, require, write_json
from .evaluation_capture import capability_transition, is_prediction_cut
from .nanogpt_adapter import (
    TrainingCheckpoint,
    TrainingConfig,
    train_plain,
)
from .task_generator import TaskInstanceSpec, TokenizedTask, build_task
from .training_capture import TrainingGFGCapture
from .training_gfg import validate_training_gfg


def frozen_training_config(
    *,
    model_seed: int,
    data_order_seed: int,
) -> TrainingConfig:
    profile = read_json(
        Path(__file__).with_name("contracts") / "training_profile.json"
    )
    return TrainingConfig(
        n_layer=profile["model"]["n_layer"],
        n_head=profile["model"]["n_head"],
        n_embd=profile["model"]["n_embd"],
        dropout=profile["dropout"],
        bias=profile["model"]["bias"],
        learning_rate=profile["learning_rate"],
        weight_decay=profile["optimizer"]["weight_decay"],
        beta1=profile["optimizer"]["beta1"],
        beta2=profile["optimizer"]["beta2"],
        gradient_clip=profile["gradient_clip"],
        max_steps=profile["max_steps"],
        evaluation_interval=profile["evaluation_interval"],
        seed=model_seed,
        data_order_seed=data_order_seed,
        device="cuda",
    )


def build_formal_task(
    *,
    instance_id: str,
    token_seed: int,
    split_seed: int,
) -> TokenizedTask:
    return build_task(
        TaskInstanceSpec(
            instance_id=instance_id,
            modulus=23,
            train_fraction=0.6,
            token_permutation_seed=token_seed,
            split_seed=split_seed,
        )
    )


def run_captured_segment(
    *,
    trainer_root: Path,
    task: TokenizedTask,
    config: TrainingConfig,
    run_id: str,
    output_directory: Path,
    initial_checkpoint: TrainingCheckpoint | None = None,
    stop_at_prediction_cut: bool = False,
    intervention_hook: Callable[
        [str, dict[str, Any], dict[str, Any]], dict[str, Any] | None
    ]
    | None = None,
    intervention_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require(not output_directory.exists(), "RUN_DIRECTORY_ALREADY_EXISTS")
    output_directory.mkdir(parents=True)
    profile = read_json(
        Path(__file__).with_name("contracts") / "training_profile.json"
    )
    capture = TrainingGFGCapture(
        run_id=run_id,
        run_directory=output_directory,
        task=task,
        evaluation_interval=config.evaluation_interval,
        materialization_interval=profile["materialization_interval"],
        chunk_steps=profile["capture_chunk_steps"],
    )
    result = train_plain(
        trainer_root.resolve(),
        task,
        config,
        initial_checkpoint=initial_checkpoint,
        hook=capture,
        intervention_hook=intervention_hook,
        intervention_state=intervention_state,
        stop_when=is_prediction_cut if stop_at_prediction_cut else None,
    )
    manifest = capture.finalize(result["stop_step"])
    checkpoint_path = output_directory / "checkpoint.pt"
    checkpoint_receipt = save_checkpoint(
        checkpoint_path, result["checkpoint"]
    )
    validation = validate_training_gfg(
        output_directory / "participant_gfg.sqlite3",
        report_path=output_directory / "gfg_validation.json",
    )
    require(validation["status"] == "PASS", "GFG_CAPTURE_FAILURE")
    if stop_at_prediction_cut:
        require(
            is_prediction_cut(result["metrics"][-1], result["metrics"]),
            "PREDICTION_CUT_NOT_REACHED",
        )
    public_result = {
        "capture_manifest": manifest,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_receipt": checkpoint_receipt,
        "elapsed_seconds": result["elapsed_seconds"],
        "gfg_validation": validation,
        "metrics": result["metrics"],
        "model_parameter_count": result["model_parameter_count"],
        "model_parameter_tensor_count": result[
            "model_parameter_tensor_count"
        ],
        "run_id": run_id,
        "schema": "formal-nanogpt-training-segment-v1",
        "start_step": result["start_step"],
        "stop_step": result["stop_step"],
        "task_participant_commitment": (
            task.participant_task_commitment
        ),
        "training_config": asdict(config),
        "transition_step": capability_transition(result["metrics"]),
    }
    public_result["result_sha256"] = payload_sha256(public_result)
    write_json(output_directory / "segment_result.json", public_result)
    return {
        **public_result,
        "_checkpoint": result["checkpoint"],
        "_intervention_state": result["intervention_state"],
    }


def ordinary_training_equivalence(
    *,
    trainer_root: Path,
    task: TokenizedTask,
    config: TrainingConfig,
    output_directory: Path,
) -> dict[str, Any]:
    short = TrainingConfig(**{**asdict(config), "max_steps": 2})
    plain = train_plain(trainer_root, task, short)
    captured_dir = output_directory / "captured"
    captured = run_captured_segment(
        trainer_root=trainer_root,
        task=task,
        config=short,
        run_id="capture-equivalence",
        output_directory=captured_dir,
    )
    material = {
        "capture_checkpoint": captured["checkpoint_receipt"][
            "checkpoint_commitment"
        ],
        "capture_on_off_ordinary_training_equal": (
            plain["checkpoint_sha256"]
            == captured["checkpoint_receipt"]["checkpoint_commitment"]
            and plain["metrics"] == captured["metrics"]
        ),
        "plain_checkpoint": plain["checkpoint_sha256"],
        "schema": "capture-on-off-training-equivalence-v1",
    }
    material["status"] = (
        "PASS"
        if material["capture_on_off_ordinary_training_equal"]
        else "FAIL"
    )
    material["result_sha256"] = payload_sha256(material)
    write_json(output_directory / "result.json", material)
    return material
