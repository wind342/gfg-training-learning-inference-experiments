"""Deterministic real-data filtering, downsampling, FFT and SVG generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import SignalWindow


FIR_TAPS = np.asarray([1, 8, 28, 56, 70, 56, 28, 8, 1], dtype=np.float64) / 256.0
DOWNSAMPLE_FACTOR = 4
FFT_WINDOW = 32
FFT_HOP = 16
CELL_WIDTH = 20.0
CELL_HEIGHT = 10.0


class CaptureSink(Protocol):
    def capture_filter_sample(
        self, output_index: int, value: float, raw_indices: list[int]
    ) -> None: ...

    def finish_filter_stage(self) -> None: ...

    def capture_downsample_decision(
        self, filtered_index: int, retained_index: int | None, value: float
    ) -> None: ...

    def finish_downsample_stage(self) -> None: ...

    def capture_spectrum_cell(
        self,
        frame_index: int,
        bin_index: int,
        magnitude: float,
        downsampled_indices: list[int],
    ) -> None: ...

    def capture_fft_tail(self, downsampled_index: int, value: float) -> None: ...

    def finish_fft_stage(self) -> None: ...

    def capture_render_cell(
        self,
        frame_index: int,
        bin_index: int,
        magnitude: float,
        rectangle: dict,
        fill: str,
    ) -> None: ...

    def finish_render_stage(self) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    filtered: np.ndarray
    downsampled: np.ndarray
    spectrum_magnitudes: np.ndarray
    svg_bytes: bytes
    rendered_cells: tuple[dict, ...]


def _render_fill(magnitude: float) -> str:
    level = int(np.clip(round(20.0 * np.log10(1.0 + magnitude) * 12.0), 0, 255))
    return f"#{level:02x}{(255 - level):02x}80"


def execute_pipeline(
    signal: SignalWindow, capture: CaptureSink | None = None
) -> PipelineResult:
    raw = signal.physical_samples_mv
    filtered_values: list[float] = []
    for output_index in range(len(raw) - len(FIR_TAPS) + 1):
        raw_slice = raw[output_index : output_index + len(FIR_TAPS)]
        value = float(sum(float(raw_slice[index]) * float(FIR_TAPS[index]) for index in range(len(FIR_TAPS))))
        filtered_values.append(value)
        if capture is not None:
            capture.capture_filter_sample(
                output_index,
                value,
                [
                    signal.absolute_start + output_index + tap_index
                    for tap_index in range(len(FIR_TAPS))
                ],
            )
    if capture is not None:
        capture.finish_filter_stage()
    filtered = np.asarray(filtered_values, dtype=np.float64)

    downsampled_values: list[float] = []
    for filtered_index, value in enumerate(filtered):
        retained_index = (
            len(downsampled_values)
            if filtered_index % DOWNSAMPLE_FACTOR == 0
            else None
        )
        if retained_index is not None:
            downsampled_values.append(float(value))
        if capture is not None:
            capture.capture_downsample_decision(
                filtered_index, retained_index, float(value)
            )
    if capture is not None:
        capture.finish_downsample_stage()
    downsampled = np.asarray(downsampled_values, dtype=np.float64)

    frame_starts = list(
        range(0, len(downsampled) - FFT_WINDOW + 1, FFT_HOP)
    )
    spectrum_rows: list[np.ndarray] = []
    used_downsampled: set[int] = set()
    for frame_index, start in enumerate(frame_starts):
        indices = list(range(start, start + FFT_WINDOW))
        used_downsampled.update(indices)
        magnitudes = np.abs(np.fft.rfft(downsampled[start : start + FFT_WINDOW]))
        spectrum_rows.append(magnitudes)
        if capture is not None:
            for bin_index, magnitude in enumerate(magnitudes):
                capture.capture_spectrum_cell(
                    frame_index,
                    bin_index,
                    float(magnitude),
                    indices,
                )
    if capture is not None:
        for index, value in enumerate(downsampled):
            if index not in used_downsampled:
                capture.capture_fft_tail(index, float(value))
        capture.finish_fft_stage()
    spectrum = np.vstack(spectrum_rows)

    bin_count = spectrum.shape[1]
    cells: list[dict] = []
    body: list[str] = []
    for frame_index in range(spectrum.shape[0]):
        for bin_index in range(bin_count):
            magnitude = float(spectrum[frame_index, bin_index])
            rectangle = {
                "x": float(frame_index * CELL_WIDTH),
                "y": float((bin_count - 1 - bin_index) * CELL_HEIGHT),
                "width": CELL_WIDTH,
                "height": CELL_HEIGHT,
            }
            fill = _render_fill(magnitude)
            cell = {
                "frame_index": frame_index,
                "bin_index": bin_index,
                "magnitude": magnitude,
                "rectangle": rectangle,
                "fill": fill,
            }
            cells.append(cell)
            body.append(
                '<rect '
                f'x="{rectangle["x"]:.1f}" y="{rectangle["y"]:.1f}" '
                f'width="{rectangle["width"]:.1f}" height="{rectangle["height"]:.1f}" '
                f'fill="{fill}"/>'
            )
            if capture is not None:
                capture.capture_render_cell(
                    frame_index, bin_index, magnitude, rectangle, fill
                )
    if capture is not None:
        capture.finish_render_stage()
    width = spectrum.shape[0] * CELL_WIDTH
    height = bin_count * CELL_HEIGHT
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">'
        + "".join(body)
        + "</svg>\n"
    ).encode("utf-8")
    return PipelineResult(
        filtered=filtered,
        downsampled=downsampled,
        spectrum_magnitudes=spectrum,
        svg_bytes=svg,
        rendered_cells=tuple(cells),
    )

