"""Run, validate and materialize the real-data multi-stage signal experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.errors import CoreV3Error

from .candidate import answer_from_snapshot
from .collector import SignalGenerationCollector
from .data import DEFAULT_DATA_ROOT, SignalWindow, load_manifest, load_signal_window
from .pipeline import PipelineResult, execute_pipeline
from .reference import ReferenceResult, compute_reference


@dataclass(frozen=True)
class Execution:
    signal: SignalWindow
    output_only: PipelineResult
    captured: PipelineResult
    collector: SignalGenerationCollector
    snapshot: object
    candidate_answer: dict
    reference: ReferenceResult
    comparison: dict
    negative_controls: dict
    scientific_summary: dict


def snapshot_document(snapshot) -> dict:
    return {
        "snapshot": snapshot.record,
        "tables": {
            field: getattr(snapshot.tables, field)
            for field in snapshot.tables.__dataclass_fields__
        },
    }


def _query_pairs(answer: dict) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for signature in answer["path_signatures"]:
        fields = signature.split("|")
        pairs.add((fields[0], fields[-1]))
    return pairs


def _negative_controls(
    snapshot,
    collector: SignalGenerationCollector,
    candidate_answer: dict,
    svg_bytes: bytes,
) -> dict:
    visual_support_ids = {
        row["support_id"]
        for row in snapshot.tables.perceptual_support_records
        if row["support_space_id"] == collector.visual_space["support_space_id"]
    }
    direct_shortcuts = [
        row["generation_binding_id"]
        for row in snapshot.tables.generation_bindings
        if row["outcome_reference"]["kind"] == "support"
        and row["outcome_reference"]["support_id"] in visual_support_ids
        and row["origin_reference"]["kind"] == "registered_source"
    ]
    true_pairs = _query_pairs(candidate_answer)
    cartesian_pairs = {
        (support, source)
        for support in candidate_answer["selected_final_support_keys"]
        for source in candidate_answer["raw_source_identities"]
    }
    false_cartesian_pairs = cartesian_pairs - true_pairs
    tampered = copy.deepcopy(snapshot)
    bridge = tampered.tables.generated_origins[0]
    bridge["origin_payload"]["prior_support_id"] = "ps3_" + ("0" * 64)
    tamper_rejected = False
    tamper_reason = None
    try:
        answer_from_snapshot(tampered, collector.registry)
    except (CoreV3Error, ValueError) as exc:
        tamper_rejected = True
        tamper_reason = getattr(exc, "reason_code", str(exc))
    output_text = svg_bytes.decode("utf-8")
    return {
        "direct_raw_to_final_shortcut_count": len(direct_shortcuts),
        "direct_raw_to_final_shortcut_rejected": not direct_shortcuts,
        "cartesian_candidate_pair_count": len(cartesian_pairs),
        "actual_pair_count": len(true_pairs),
        "false_cartesian_pair_count": len(false_cartesian_pairs),
        "cartesian_expansion_rejected": bool(false_cartesian_pairs),
        "tampered_generated_origin_rejected": tamper_rejected,
        "tamper_reason": tamper_reason,
        "source_identity_leak_in_svg_count": output_text.count("physionet:"),
        "source_identity_leak_rejected": "physionet:" not in output_text,
    }


def execute_once(signal: SignalWindow) -> Execution:
    output_only = execute_pipeline(signal)
    collector = SignalGenerationCollector(signal)
    captured = execute_pipeline(signal, collector)
    snapshot = collector.validated_snapshot()
    candidate = answer_from_snapshot(snapshot, collector.registry)
    reference = compute_reference(signal)
    filtered_error = float(
        np.max(np.abs(captured.filtered - reference.filtered))
    )
    downsampled_error = float(
        np.max(np.abs(captured.downsampled - reference.downsampled))
    )
    spectrum_error = float(
        np.max(
            np.abs(
                captured.spectrum_magnitudes
                - reference.spectrum_magnitudes
            )
        )
    )
    answer_fields = [
        "selected_final_support_keys",
        "raw_source_identities",
        "path_count",
        "path_signature_multiset_sha256",
    ]
    answer_exact = all(
        candidate[field] == reference.answer[field]
        for field in answer_fields
    )
    comparison = {
        "ordinary_output_byte_identical": (
            output_only.svg_bytes == captured.svg_bytes
        ),
        "output_only_sha256": hashlib.sha256(
            output_only.svg_bytes
        ).hexdigest(),
        "captured_output_sha256": hashlib.sha256(
            captured.svg_bytes
        ).hexdigest(),
        "filtered_max_abs_error": filtered_error,
        "downsampled_max_abs_error": downsampled_error,
        "spectrum_max_abs_error": spectrum_error,
        "numeric_reference_exact_within_1e_10": max(
            filtered_error, downsampled_error, spectrum_error
        )
        <= 1e-10,
        "candidate_reference_answer_exact": answer_exact,
        "candidate_path_count": candidate["path_count"],
        "reference_path_count": reference.answer["path_count"],
        "candidate_raw_source_count": len(
            candidate["raw_source_identities"]
        ),
        "reference_raw_source_count": len(
            reference.answer["raw_source_identities"]
        ),
        "selected_final_support_count": len(
            candidate["selected_final_support_keys"]
        ),
    }
    negative_controls = _negative_controls(
        snapshot, collector, candidate, captured.svg_bytes
    )
    counts = snapshot.record["authoritative_table_counts"]
    all_controls_pass = all(
        [
            negative_controls["direct_raw_to_final_shortcut_rejected"],
            negative_controls["cartesian_expansion_rejected"],
            negative_controls["tampered_generated_origin_rejected"],
            negative_controls["source_identity_leak_rejected"],
        ]
    )
    status = (
        "MULTISTAGE_SIGNAL_GENERATION_FACTS_SUPPORTED"
        if comparison["ordinary_output_byte_identical"]
        and comparison["numeric_reference_exact_within_1e_10"]
        and comparison["candidate_reference_answer_exact"]
        and all_controls_pass
        else "MULTISTAGE_SIGNAL_GENERATION_FACTS_NOT_SUPPORTED"
    )
    summary = {
        "status": status,
        "dataset": "MIT-BIH Arrhythmia Database v1.0.0",
        "record": signal.record,
        "channel": signal.channel,
        "sample_rate_hz": signal.sample_rate_hz,
        "window_start": signal.absolute_start,
        "window_length": len(signal.digital_samples),
        "input_sha256": signal.input_sha256,
        "stages": [
            "fir_filter",
            "downsample",
            "fft",
            "svg_render",
        ],
        "snapshot_id": snapshot.snapshot_id,
        "authoritative_table_counts": counts,
        "comparison": comparison,
        "negative_controls": negative_controls,
        "query_result": {
            "rectangle": candidate["query_rectangle"],
            "selected_final_support_count": len(
                candidate["selected_final_support_keys"]
            ),
            "raw_source_count": len(
                candidate["raw_source_identities"]
            ),
            "path_count": candidate["path_count"],
            "path_signature_multiset_sha256": candidate[
                "path_signature_multiset_sha256"
            ],
            "traversed_binding_count": candidate[
                "traversed_binding_count"
            ],
            "traversed_generated_origin_count": candidate[
                "traversed_generated_origin_count"
            ],
        },
    }
    return Execution(
        signal=signal,
        output_only=output_only,
        captured=captured,
        collector=collector,
        snapshot=snapshot,
        candidate_answer=candidate,
        reference=reference,
        comparison=comparison,
        negative_controls=negative_controls,
        scientific_summary=summary,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def materialize_execution(execution: Execution, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output_only.svg").write_bytes(
        execution.output_only.svg_bytes
    )
    (output_dir / "continuity.svg").write_bytes(
        execution.captured.svg_bytes
    )
    _write_json(output_dir / "input_manifest.json", load_manifest())
    _write_json(
        output_dir / "snapshot.json",
        snapshot_document(execution.snapshot),
    )
    _write_json(
        output_dir / "candidate_answer.json",
        execution.candidate_answer,
    )
    _write_json(
        output_dir / "reference_answer.json",
        execution.reference.answer,
    )
    _write_json(output_dir / "comparison.json", execution.comparison)
    _write_json(
        output_dir / "negative_controls.json",
        execution.negative_controls,
    )
    _write_json(
        output_dir / "scientific_summary.json",
        execution.scientific_summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results")
        / "signal_multistage_generated_origin_v1",
    )
    args = parser.parse_args()
    signal = load_signal_window(args.data_root)
    first = execute_once(signal)
    second = execute_once(signal)
    materialize_execution(first, args.output_root / "run_1")
    materialize_execution(second, args.output_root / "run_2")
    first_bytes = canonical_bytes(first.scientific_summary)
    second_bytes = canonical_bytes(second.scientific_summary)
    replay = {
        "run_1_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "run_2_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "scientific_summaries_byte_identical": first_bytes
        == second_bytes,
        "output_svg_byte_identical_across_runs": (
            first.captured.svg_bytes == second.captured.svg_bytes
        ),
    }
    _write_json(args.output_root / "replay_comparison.json", replay)
    final = {
        **first.scientific_summary,
        "replay": replay,
    }
    _write_json(args.output_root / "final_report.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if (
        first.scientific_summary["status"]
        == "MULTISTAGE_SIGNAL_GENERATION_FACTS_SUPPORTED"
        and replay["scientific_summaries_byte_identical"]
        and replay["output_svg_byte_identical_across_runs"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
