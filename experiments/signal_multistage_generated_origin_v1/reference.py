"""Independent mathematical reference for values, geometry and path multiplicity."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from generation_relation_core.canonical import canonical_bytes

from .contract import QUERY_RECTANGLE
from .data import SignalWindow


REFERENCE_TAPS = np.asarray(
    [1, 8, 28, 56, 70, 56, 28, 8, 1], dtype=np.float64
) / 256.0
REFERENCE_DOWNSAMPLE = 4
REFERENCE_WINDOW = 32
REFERENCE_HOP = 16
REFERENCE_CELL_WIDTH = 20.0
REFERENCE_CELL_HEIGHT = 10.0


@dataclass(frozen=True)
class ReferenceResult:
    filtered: np.ndarray
    downsampled: np.ndarray
    spectrum_magnitudes: np.ndarray
    answer: dict


def _intersects(left: dict, right: dict) -> bool:
    return (
        left["x"] < right["x"] + right["width"]
        and right["x"] < left["x"] + left["width"]
        and left["y"] < right["y"] + right["height"]
        and right["y"] < left["y"] + left["height"]
    )


def _dft_magnitudes(values: np.ndarray) -> np.ndarray:
    size = len(values)
    rows = []
    for bin_index in range(size // 2 + 1):
        total = 0.0 + 0.0j
        for sample_index, value in enumerate(values):
            angle = -2.0 * math.pi * bin_index * sample_index / size
            total += float(value) * complex(math.cos(angle), math.sin(angle))
        rows.append(abs(total))
    return np.asarray(rows, dtype=np.float64)


def _path_signature(
    frame_index: int,
    bin_index: int,
    downsampled_index: int,
    filtered_index: int,
    tap_index: int,
    raw_identity: str,
) -> str:
    return "|".join(
        [
            f"svg:frame:{frame_index}:bin:{bin_index}",
            f"spectrum:frame:{frame_index}:bin:{bin_index}",
            f"downsampled:{downsampled_index}",
            f"filtered:{filtered_index}",
            "spectrum_cell_rendered_as_svg_rectangle",
            "fft_window_sample",
            "downsample_retained_phase_zero",
            f"fir_input_tap_{tap_index}",
            f"occurrence:svg:frame:{frame_index}:bin:{bin_index}",
            f"occurrence:fft:frame:{frame_index}",
            f"occurrence:downsample:{filtered_index}",
            f"occurrence:fir:{filtered_index}",
            raw_identity,
        ]
    )


def compute_reference(signal: SignalWindow) -> ReferenceResult:
    filtered = np.convolve(
        signal.physical_samples_mv, REFERENCE_TAPS, mode="valid"
    )
    downsampled = filtered[::REFERENCE_DOWNSAMPLE].copy()
    frame_starts = list(
        range(
            0,
            len(downsampled) - REFERENCE_WINDOW + 1,
            REFERENCE_HOP,
        )
    )
    spectra = np.vstack(
        [
            _dft_magnitudes(
                downsampled[start : start + REFERENCE_WINDOW]
            )
            for start in frame_starts
        ]
    )
    bin_count = spectra.shape[1]
    selected: list[tuple[int, int]] = []
    for frame_index in range(spectra.shape[0]):
        for bin_index in range(bin_count):
            rectangle = {
                "x": frame_index * REFERENCE_CELL_WIDTH,
                "y": (
                    bin_count - 1 - bin_index
                )
                * REFERENCE_CELL_HEIGHT,
                "width": REFERENCE_CELL_WIDTH,
                "height": REFERENCE_CELL_HEIGHT,
            }
            if _intersects(rectangle, QUERY_RECTANGLE):
                selected.append((frame_index, bin_index))
    signatures: list[str] = []
    raw_identities: set[str] = set()
    for frame_index, bin_index in selected:
        start = frame_starts[frame_index]
        for downsampled_index in range(
            start, start + REFERENCE_WINDOW
        ):
            filtered_index = (
                downsampled_index * REFERENCE_DOWNSAMPLE
            )
            for tap_index in range(len(REFERENCE_TAPS)):
                absolute_index = (
                    signal.absolute_start + filtered_index + tap_index
                )
                raw_identity = (
                    "physionet:mitdb:1.0.0:100:MLII:"
                    f"sample:{absolute_index}"
                )
                raw_identities.add(raw_identity)
                signatures.append(
                    _path_signature(
                        frame_index,
                        bin_index,
                        downsampled_index,
                        filtered_index,
                        tap_index,
                        raw_identity,
                    )
                )
    signatures.sort()
    answer = {
        "query_rectangle": QUERY_RECTANGLE,
        "selected_final_support_keys": sorted(
            f"svg:frame:{frame}:bin:{bin_index}"
            for frame, bin_index in selected
        ),
        "raw_source_identities": sorted(raw_identities),
        "path_count": len(signatures),
        "path_signature_multiset_sha256": hashlib.sha256(
            canonical_bytes(signatures)
        ).hexdigest(),
        "path_signatures": signatures,
    }
    return ReferenceResult(
        filtered=filtered,
        downsampled=downsampled,
        spectrum_magnitudes=spectra,
        answer=answer,
    )
