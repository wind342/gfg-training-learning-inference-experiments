"""Synchronous Core v3 collection for the four-stage signal pipeline."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Any

import numpy as np

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    evidence_record,
    explicit_disposition,
    generated_origin,
    generation_binding,
    generation_occurrence,
    generator_manifest,
    generator_operation_result,
    perceptual_support,
    predicate_profile,
    relation_material,
    source_information,
    support_space,
)
from generation_relation_core.predicate_registry import (
    PredicateRegistry,
    implementation_sha256,
)
from generation_relation_core.snapshots import (
    CoreV3Tables,
    ValidatedSnapshot,
    build_snapshot,
)

from .contract import DOMAIN_SCOPE_ID
from .data import SignalWindow
from .pipeline import (
    CELL_HEIGHT,
    CELL_WIDTH,
    DOWNSAMPLE_FACTOR,
    FFT_HOP,
    FFT_WINDOW,
    FIR_TAPS,
)
from .predicates import (
    native_key_membership,
    positive_area_rectangle_intersection,
)

EVIDENCE_AUTHORITY = "synchronous_multistage_signal_generation_v1"


def experiment_code_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.suffix == ".py"
            or (
                path.suffix == ".json"
                and "contracts" in path.relative_to(root).parts
            )
        )
    ]
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _native_key_schema(extra: dict[str, dict], required: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["native_support_key", *required],
        "properties": {
            "native_support_key": {"type": "string", "minLength": 1},
            **extra,
        },
    }


def _membership_query_schema() -> dict:
    return _native_key_schema({}, [])


class SignalGenerationCollector:
    """Create facts inside the native filter/downsample/FFT/render loops."""

    def __init__(self, signal: SignalWindow) -> None:
        self.signal = signal
        self.code_hash = experiment_code_hash()
        self.scalar_space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="signal_scalar_stage_support",
            support_payload_schema=_native_key_schema(
                {
                    "stage": {
                        "enum": ["fir_filter", "downsample"],
                    },
                    "sample_index": {"type": "integer", "minimum": 0},
                    "time_seconds": {"type": "number"},
                    "value_mv": {"type": "number"},
                    "sample_rate_hz": {"type": "number", "exclusiveMinimum": 0},
                },
                [
                    "stage",
                    "sample_index",
                    "time_seconds",
                    "value_mv",
                    "sample_rate_hz",
                ],
            ),
            query_payload_schema=_membership_query_schema(),
            normalization_rule="finite binary64 values and exact native key membership",
        )
        self.spectrum_space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="signal_time_frequency_cell",
            support_payload_schema=_native_key_schema(
                {
                    "frame_index": {"type": "integer", "minimum": 0},
                    "bin_index": {"type": "integer", "minimum": 0},
                    "time_start_seconds": {"type": "number"},
                    "time_end_seconds": {"type": "number"},
                    "frequency_hz": {"type": "number", "minimum": 0},
                    "magnitude": {"type": "number", "minimum": 0},
                },
                [
                    "frame_index",
                    "bin_index",
                    "time_start_seconds",
                    "time_end_seconds",
                    "frequency_hz",
                    "magnitude",
                ],
            ),
            query_payload_schema=_membership_query_schema(),
            normalization_rule="rectangular FFT window with exact cell identity",
        )
        self.visual_space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="signal_svg_spectrogram_css_rectangle",
            support_payload_schema=_native_key_schema(
                {
                    "frame_index": {"type": "integer", "minimum": 0},
                    "bin_index": {"type": "integer", "minimum": 0},
                    "magnitude": {"type": "number", "minimum": 0},
                    "fill": {
                        "type": "string",
                        "pattern": "^#[0-9a-f]{6}$",
                    },
                    "rectangle": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y", "width", "height"],
                        "properties": {
                            "x": {"type": "number", "minimum": 0},
                            "y": {"type": "number", "minimum": 0},
                            "width": {"type": "number", "exclusiveMinimum": 0},
                            "height": {"type": "number", "exclusiveMinimum": 0},
                        },
                    },
                },
                [
                    "frame_index",
                    "bin_index",
                    "magnitude",
                    "fill",
                    "rectangle",
                ],
            ),
            query_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["rectangle"],
                "properties": {
                    "rectangle": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y", "width", "height"],
                        "properties": {
                            "x": {"type": "number", "minimum": 0},
                            "y": {"type": "number", "minimum": 0},
                            "width": {"type": "number", "exclusiveMinimum": 0},
                            "height": {"type": "number", "exclusiveMinimum": 0},
                        },
                    }
                },
            },
            normalization_rule="positive-area intersection in deterministic SVG user coordinates",
        )
        self.scalar_profile = self._profile(
            self.scalar_space,
            "native_key_membership",
            ["membership"],
            native_key_membership,
        )
        self.spectrum_profile = self._profile(
            self.spectrum_space,
            "native_key_membership",
            ["membership"],
            native_key_membership,
        )
        self.visual_profile = self._profile(
            self.visual_space,
            "positive_area_rectangle_intersection",
            ["intersection"],
            positive_area_rectangle_intersection,
        )
        input_dependency = hashlib.sha256(
            f"mitdb-100.dat:{signal.input_sha256}".encode("utf-8")
        ).hexdigest()
        numpy_dependency = hashlib.sha256(
            f"numpy:{np.__version__}".encode("utf-8")
        ).hexdigest()
        self.environment = environment_record(
            runtime_name="CPython and NumPy",
            runtime_version=platform.python_version(),
            operating_system=platform.platform(),
            dependency_hashes={
                "mitdb-record-100": input_dependency,
                "numpy": numpy_dependency,
            },
        )
        spaces = [self.scalar_space, self.spectrum_space, self.visual_space]
        profiles = [
            self.scalar_profile,
            self.spectrum_profile,
            self.visual_profile,
        ]
        self.manifest = generator_manifest(
            generator_name="mitdb-multistage-signal-generator",
            generator_version="1",
            generator_code_hash=self.code_hash,
            supported_support_space_ids=[
                row["support_space_id"] for row in spaces
            ],
            supported_predicate_profile_ids=[
                row["predicate_profile_id"] for row in profiles
            ],
            supported_operations=[
                "collect_fir_filter_generation_facts",
                "collect_downsample_generation_facts",
                "collect_fft_generation_facts",
                "collect_svg_render_generation_facts",
            ],
            authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
            dependency_hashes=[input_dependency, numpy_dependency],
        )
        self.registry = PredicateRegistry(
            spaces,
            profiles,
            {
                self.scalar_profile["predicate_profile_id"]: native_key_membership,
                self.spectrum_profile["predicate_profile_id"]: native_key_membership,
                self.visual_profile[
                    "predicate_profile_id"
                ]: positive_area_rectangle_intersection,
            },
        )
        self.tables = CoreV3Tables(
            support_space_records=spaces,
            predicate_profiles=profiles,
            generator_manifests=[self.manifest],
            environment_records=[self.environment],
        )
        self._stage_produced: dict[str, set[str]] = {}
        self._stage_evidence: dict[str, set[str]] = {}
        self._stage_operations: dict[str, dict] = {}
        self._sources_by_absolute_index: dict[int, dict] = {}
        self._filtered_supports: dict[int, dict] = {}
        self._downsampled_supports: dict[int, dict] = {}
        self._spectrum_supports: dict[tuple[int, int], dict] = {}
        self._generated_by_support: dict[str, dict] = {}
        self._occurrences: dict[str, dict] = {}
        self._occurrence_index = 0

    def _profile(
        self,
        space: dict,
        kind: str,
        supported: list[str],
        implementation,
    ) -> dict:
        return predicate_profile(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=space["support_space_id"],
            predicate_kind=kind,
            supported_predicates=supported,
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=implementation.__module__,
            implementation_symbol=implementation.__name__,
            predicate_implementation_sha256=implementation_sha256(implementation),
            normalization_rule=space["normalization_rule"],
            result_ordering_rule="ascending content-addressed support_id",
        )

    def _produced(self, stage: str, *entity_ids: str) -> None:
        self._stage_produced.setdefault(stage, set()).update(entity_ids)

    def _source(self, absolute_index: int, stage: str) -> dict:
        existing = self._sources_by_absolute_index.get(absolute_index)
        if existing is not None:
            return existing
        local_index = absolute_index - self.signal.absolute_start
        digital = int(self.signal.digital_samples[local_index])
        physical = float(self.signal.physical_samples_mv[local_index])
        row = source_information(
            domain_scope_id=DOMAIN_SCOPE_ID,
            source_identity=(
                f"physionet:mitdb:1.0.0:100:MLII:sample:{absolute_index}"
            ),
            source_parent_id="physionet:mitdb:1.0.0:100:MLII",
            source_granularity="one_ecg_sample",
            source_payload={
                "dataset": "MIT-BIH Arrhythmia Database",
                "record": self.signal.record,
                "channel": self.signal.channel,
                "absolute_sample_index": absolute_index,
                "sample_rate_hz": self.signal.sample_rate_hz,
                "digital_value": digital,
                "physical_value_mv": physical,
                "input_sha256": self.signal.input_sha256,
            },
        )
        self._sources_by_absolute_index[absolute_index] = row
        self.tables.source_information_records.append(row)
        self._produced(stage, row["source_information_id"])
        return row

    def _occurrence(
        self,
        stage: str,
        stable_key: str,
        occurrence_type: str,
        transform_reference: dict,
        payload: dict,
    ) -> dict:
        existing = self._occurrences.get(stable_key)
        if existing is not None:
            return existing
        row = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage=stage,
            occurrence_type=occurrence_type,
            stable_instance_key=stable_key,
            occurrence_index=self._occurrence_index,
            transform_reference=transform_reference,
            occurrence_payload={
                **payload,
                "capture_timing": "synchronous_inside_native_stage",
            },
        )
        self._occurrence_index += 1
        self._occurrences[stable_key] = row
        self.tables.generation_occurrences.append(row)
        self._produced(stage, row["generation_occurrence_id"])
        return row

    def _support(
        self, stage: str, space: dict, profile: dict, payload: dict
    ) -> dict:
        row = perceptual_support(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=space["support_space_id"],
            support_payload=payload,
            predicate_profile_id=profile["predicate_profile_id"],
        )
        self.tables.perceptual_support_records.append(row)
        self._produced(stage, row["support_id"])
        return row

    def _disposition(
        self, stage: str, reason: str, payload: dict
    ) -> dict:
        row = explicit_disposition(
            domain_scope_id=DOMAIN_SCOPE_ID,
            core_disposition_category="suppressed",
            domain_reason_code=reason,
            disposition_payload=payload,
        )
        self.tables.explicit_dispositions.append(row)
        self._produced(stage, row["disposition_id"])
        return row

    def _generated(self, stage: str, prior_support: dict) -> dict:
        prior_id = prior_support["support_id"]
        existing = self._generated_by_support.get(prior_id)
        if existing is not None:
            return existing
        prior_stage = prior_support["support_payload"].get("stage", "fft")
        producer_operation = self._stage_operations.get(prior_stage)
        if producer_operation is None:
            raise ValueError(
                f"GENERATED_ORIGIN_PRODUCER_OPERATION_MISSING:{prior_stage}"
            )
        row = generated_origin(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            origin_type="prior_stage_outcome_reintroduced_as_input",
            origin_payload={
                "bridge_kind": "support_to_generated_origin",
                "prior_support_id": prior_id,
                "prior_support_key": prior_support["support_payload"][
                    "native_support_key"
                ],
                "prior_stage": prior_stage,
                "consumer_stage": stage,
                "producer_operation_result_id": producer_operation[
                    "operation_result_id"
                ],
            },
        )
        self._generated_by_support[prior_id] = row
        self.tables.generated_origins.append(row)
        self._produced(stage, row["generated_origin_id"])
        return row

    def _bind(
        self,
        stage: str,
        origin_reference: dict,
        origin_id: str,
        occurrence: dict,
        outcome_reference: dict,
        outcome_id: str,
        role: str,
    ) -> dict:
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        material_bytes = canonical_bytes(material)
        digest = hashlib.sha256(material_bytes).hexdigest()
        evidence = evidence_record(
            artifact_locator=(
                f"candidate://relation_materials.jsonl#sha256={digest}"
            ),
            artifact_role="generation_relation_material",
            artifact_bytes=material_bytes,
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method=(
                "synchronous native-stage capture with explicit origin and outcome"
            ),
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=sorted(
                [
                    origin_id,
                    occurrence["generation_occurrence_id"],
                    outcome_id,
                ]
            ),
        )
        binding = generation_binding(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
            evidence_ids=[evidence["evidence_id"]],
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        )
        self.tables.evidence_records.append(evidence)
        self.tables.generation_bindings.append(binding)
        self.tables.evidence_links.append(link)
        self._produced(
            stage,
            binding["generation_binding_id"],
            evidence["evidence_id"],
            link["evidence_link_id"],
        )
        self._stage_evidence.setdefault(stage, set()).add(
            evidence["evidence_id"]
        )
        return binding

    def _finish_stage(self, stage: str) -> None:
        if stage in self._stage_operations:
            raise ValueError(f"STAGE_ALREADY_FINISHED:{stage}")
        operation = generator_operation_result(
            generator_manifest_id=self.manifest["generator_manifest_id"],
            operation_name=f"collect_{stage}_generation_facts",
            produced_entity_ids=sorted(self._stage_produced.get(stage, set())),
            evidence_ids=sorted(self._stage_evidence.get(stage, set())),
        )
        self.tables.generator_operation_results.append(operation)
        self._stage_operations[stage] = operation

    def capture_filter_sample(
        self, output_index: int, value: float, raw_indices: list[int]
    ) -> None:
        stage = "fir_filter"
        support = self._support(
            stage,
            self.scalar_space,
            self.scalar_profile,
            {
                "native_support_key": f"filtered:{output_index}",
                "stage": stage,
                "sample_index": output_index,
                "time_seconds": float(
                    (
                        self.signal.absolute_start
                        + output_index
                        + (len(FIR_TAPS) - 1) / 2
                    )
                    / self.signal.sample_rate_hz
                ),
                "value_mv": value,
                "sample_rate_hz": float(self.signal.sample_rate_hz),
            },
        )
        self._filtered_supports[output_index] = support
        occurrence = self._occurrence(
            stage,
            f"occurrence:fir:{output_index}",
            "nine_tap_fir_filter_output",
            {
                "filter_kind": "symmetric_binomial_fir",
                "taps": [float(item) for item in FIR_TAPS],
                "boundary": "valid",
            },
            {"output_index": output_index, "output_value_mv": value},
        )
        outcome = {"kind": "support", "support_id": support["support_id"]}
        for tap_index, absolute_index in enumerate(raw_indices):
            source = self._source(absolute_index, stage)
            self._bind(
                stage,
                {
                    "kind": "registered_source",
                    "source_information_id": source["source_information_id"],
                },
                source["source_information_id"],
                occurrence,
                outcome,
                support["support_id"],
                f"fir_input_tap_{tap_index}",
            )

    def finish_filter_stage(self) -> None:
        self._finish_stage("fir_filter")

    def capture_downsample_decision(
        self, filtered_index: int, retained_index: int | None, value: float
    ) -> None:
        stage = "downsample"
        prior = self._filtered_supports[filtered_index]
        origin = self._generated(stage, prior)
        occurrence = self._occurrence(
            stage,
            f"occurrence:downsample:{filtered_index}",
            "factor_four_downsample_decision",
            {
                "factor": DOWNSAMPLE_FACTOR,
                "phase": 0,
                "decision": "retain" if retained_index is not None else "dispose",
            },
            {
                "filtered_index": filtered_index,
                "retained_index": retained_index,
            },
        )
        if retained_index is None:
            outcome_row = self._disposition(
                stage,
                "FILTERED_SAMPLE_NOT_ON_DOWNSAMPLE_PHASE",
                {
                    "filtered_index": filtered_index,
                    "factor": DOWNSAMPLE_FACTOR,
                    "phase": 0,
                },
            )
            outcome_ref = {
                "kind": "disposition",
                "disposition_id": outcome_row["disposition_id"],
            }
            outcome_id = outcome_row["disposition_id"]
            role = "downsample_explicitly_disposed"
        else:
            output_rate = self.signal.sample_rate_hz / DOWNSAMPLE_FACTOR
            outcome_row = self._support(
                stage,
                self.scalar_space,
                self.scalar_profile,
                {
                    "native_support_key": f"downsampled:{retained_index}",
                    "stage": stage,
                    "sample_index": retained_index,
                    "time_seconds": float(
                        (
                            self.signal.absolute_start
                            + filtered_index
                            + (len(FIR_TAPS) - 1) / 2
                        )
                        / self.signal.sample_rate_hz
                    ),
                    "value_mv": value,
                    "sample_rate_hz": float(output_rate),
                },
            )
            self._downsampled_supports[retained_index] = outcome_row
            outcome_ref = {
                "kind": "support",
                "support_id": outcome_row["support_id"],
            }
            outcome_id = outcome_row["support_id"]
            role = "downsample_retained_phase_zero"
        self._bind(
            stage,
            {
                "kind": "generated_origin",
                "generated_origin_id": origin["generated_origin_id"],
            },
            origin["generated_origin_id"],
            occurrence,
            outcome_ref,
            outcome_id,
            role,
        )

    def finish_downsample_stage(self) -> None:
        self._finish_stage("downsample")

    def capture_spectrum_cell(
        self,
        frame_index: int,
        bin_index: int,
        magnitude: float,
        downsampled_indices: list[int],
    ) -> None:
        stage = "fft"
        output_rate = self.signal.sample_rate_hz / DOWNSAMPLE_FACTOR
        start = downsampled_indices[0]
        support = self._support(
            stage,
            self.spectrum_space,
            self.spectrum_profile,
            {
                "native_support_key": (
                    f"spectrum:frame:{frame_index}:bin:{bin_index}"
                ),
                "frame_index": frame_index,
                "bin_index": bin_index,
                "time_start_seconds": float(
                    self._downsampled_supports[start]["support_payload"][
                        "time_seconds"
                    ]
                ),
                "time_end_seconds": float(
                    self._downsampled_supports[
                        downsampled_indices[-1]
                    ]["support_payload"]["time_seconds"]
                    + 1.0 / output_rate
                ),
                "frequency_hz": float(
                    bin_index * output_rate / FFT_WINDOW
                ),
                "magnitude": magnitude,
            },
        )
        self._spectrum_supports[(frame_index, bin_index)] = support
        occurrence = self._occurrence(
            stage,
            f"occurrence:fft:frame:{frame_index}",
            "rectangular_window_real_fft",
            {
                "fft_window": FFT_WINDOW,
                "fft_hop": FFT_HOP,
                "window_function": "rectangular",
                "sample_rate_hz": float(output_rate),
            },
            {
                "frame_index": frame_index,
                "downsampled_start_index": start,
            },
        )
        outcome = {"kind": "support", "support_id": support["support_id"]}
        for input_index in downsampled_indices:
            prior = self._downsampled_supports[input_index]
            origin = self._generated(stage, prior)
            self._bind(
                stage,
                {
                    "kind": "generated_origin",
                    "generated_origin_id": origin["generated_origin_id"],
                },
                origin["generated_origin_id"],
                occurrence,
                outcome,
                support["support_id"],
                "fft_window_sample",
            )

    def capture_fft_tail(self, downsampled_index: int, value: float) -> None:
        stage = "fft"
        prior = self._downsampled_supports[downsampled_index]
        origin = self._generated(stage, prior)
        occurrence = self._occurrence(
            stage,
            "occurrence:fft:incomplete-tail",
            "incomplete_fft_window_disposition",
            {
                "fft_window": FFT_WINDOW,
                "fft_hop": FFT_HOP,
                "disposition": "no_complete_window",
            },
            {"first_tail_index": downsampled_index},
        )
        disposition = self._disposition(
            stage,
            "NO_COMPLETE_FFT_WINDOW",
            {
                "downsampled_index": downsampled_index,
                "value_mv": value,
                "fft_window": FFT_WINDOW,
            },
        )
        self._bind(
            stage,
            {
                "kind": "generated_origin",
                "generated_origin_id": origin["generated_origin_id"],
            },
            origin["generated_origin_id"],
            occurrence,
            {
                "kind": "disposition",
                "disposition_id": disposition["disposition_id"],
            },
            disposition["disposition_id"],
            "fft_incomplete_tail_disposed",
        )

    def finish_fft_stage(self) -> None:
        self._finish_stage("fft")

    def capture_render_cell(
        self,
        frame_index: int,
        bin_index: int,
        magnitude: float,
        rectangle: dict,
        fill: str,
    ) -> None:
        stage = "svg_render"
        prior = self._spectrum_supports[(frame_index, bin_index)]
        origin = self._generated(stage, prior)
        support = self._support(
            stage,
            self.visual_space,
            self.visual_profile,
            {
                "native_support_key": (
                    f"svg:frame:{frame_index}:bin:{bin_index}"
                ),
                "frame_index": frame_index,
                "bin_index": bin_index,
                "magnitude": magnitude,
                "fill": fill,
                "rectangle": rectangle,
            },
        )
        occurrence = self._occurrence(
            stage,
            f"occurrence:svg:frame:{frame_index}:bin:{bin_index}",
            "svg_spectrogram_cell_render",
            {
                "cell_width": CELL_WIDTH,
                "cell_height": CELL_HEIGHT,
                "fill_mapping": "fixed_logarithmic_rgb",
            },
            {
                "frame_index": frame_index,
                "bin_index": bin_index,
                "rectangle": rectangle,
                "fill": fill,
            },
        )
        self._bind(
            stage,
            {
                "kind": "generated_origin",
                "generated_origin_id": origin["generated_origin_id"],
            },
            origin["generated_origin_id"],
            occurrence,
            {"kind": "support", "support_id": support["support_id"]},
            support["support_id"],
            "spectrum_cell_rendered_as_svg_rectangle",
        )

    def finish_render_stage(self) -> None:
        self._finish_stage("svg_render")

    def validated_snapshot(self) -> ValidatedSnapshot:
        expected = {"fir_filter", "downsample", "fft", "svg_render"}
        if set(self._stage_operations) != expected:
            raise ValueError("INCOMPLETE_STAGE_CLOSURE")
        source_rows, occurrence_rows = derive_legacy_projections(
            self.tables.source_information_records,
            self.tables.generation_occurrences,
            self.tables.generation_bindings,
            validate_schema=False,
        )
        self.tables.legacy_source_binding_projections = source_rows
        self.tables.legacy_occurrence_binding_projections = occurrence_rows
        return build_snapshot(self.tables, self.registry)
