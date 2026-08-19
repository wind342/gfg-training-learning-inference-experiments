from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


POSITIVE_ALPHAS = np.asarray([0.125, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
FULL_ALPHAS = np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
BASE_SEED = 20260806
RANDOM_FEATURE_WIDTH = 96
RIDGE = 1e-2


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{BASE_SEED}:{label}".encode("utf-8")).digest()[:8], "big") % (2**32 - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass
class RandomFeatureCurveRegressor:
    scheme: str
    seed: int
    feature_names: list[str]
    width: int = RANDOM_FEATURE_WIDTH
    ridge: float = RIDGE
    median: np.ndarray | None = None
    iqr: np.ndarray | None = None
    random_weight: np.ndarray | None = None
    random_bias: np.ndarray | None = None
    coefficient: np.ndarray | None = None
    response_mean: np.ndarray | None = None
    response_basis: np.ndarray | None = None
    fitted: bool = False

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        require(self.median is not None and self.iqr is not None, "MODEL_STANDARDIZATION_NOT_FITTED")
        standardized = (np.asarray(values, dtype=np.float64) - self.median) / self.iqr
        return np.clip(standardized, -10.0, 10.0)

    def _design(self, values: np.ndarray) -> np.ndarray:
        standardized = self._standardize(values)
        require(self.random_weight is not None and self.random_bias is not None, "MODEL_RANDOM_FEATURES_NOT_FITTED")
        nonlinear = np.tanh(standardized @ self.random_weight + self.random_bias)
        return np.concatenate(
            [np.ones((len(standardized), 1), dtype=np.float64), standardized, nonlinear],
            axis=1,
        )

    def _encode_response(self, displacement: np.ndarray) -> np.ndarray:
        displacement = np.asarray(displacement, dtype=np.float64)
        require(displacement.ndim == 2 and displacement.shape[1] == 5, "RESPONSE_SHAPE_INVALID")
        if self.scheme == "A_DIRECT":
            return displacement
        if self.scheme == "B_AMPLITUDE_SHAPE":
            amplitude = np.max(np.abs(displacement), axis=1)
            shape = displacement / (amplitude[:, None] + 1e-12)
            return np.concatenate([np.log1p(amplitude)[:, None], shape], axis=1)
        if self.scheme == "C_PCA3":
            self.response_mean = np.mean(displacement, axis=0)
            centered = displacement - self.response_mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            self.response_basis = vt[:3]
            return centered @ self.response_basis.T
        raise RuntimeError(f"UNKNOWN_RESPONSE_SCHEME:{self.scheme}")

    def _decode_response(self, encoded: np.ndarray) -> np.ndarray:
        encoded = np.asarray(encoded, dtype=np.float64)
        if self.scheme == "A_DIRECT":
            return encoded
        if self.scheme == "B_AMPLITUDE_SHAPE":
            amplitude = np.maximum(np.expm1(encoded[:, 0]), 0.0)
            return amplitude[:, None] * encoded[:, 1:]
        if self.scheme == "C_PCA3":
            require(self.response_mean is not None and self.response_basis is not None, "PCA_RESPONSE_BASIS_MISSING")
            return self.response_mean + encoded @ self.response_basis
        raise RuntimeError(f"UNKNOWN_RESPONSE_SCHEME:{self.scheme}")

    def fit(self, values: np.ndarray, displacement: np.ndarray) -> "RandomFeatureCurveRegressor":
        values = np.asarray(values, dtype=np.float64)
        require(values.ndim == 2 and values.shape[1] == len(self.feature_names), "FEATURE_MATRIX_SHAPE_INVALID")
        require(np.all(np.isfinite(values)), "NONFINITE_MODEL_INPUT")
        require(np.all(np.isfinite(displacement)), "NONFINITE_MODEL_TARGET")
        self.median = np.median(values, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        raw_iqr = q75 - q25
        self.iqr = np.where(raw_iqr > 1e-10, raw_iqr, 1.0)
        generator = np.random.default_rng(self.seed)
        scale = 1.0 / math.sqrt(max(values.shape[1], 1))
        self.random_weight = generator.normal(0.0, scale, size=(values.shape[1], self.width))
        self.random_bias = generator.uniform(-math.pi, math.pi, size=self.width)
        design = self._design(values)
        encoded = self._encode_response(displacement)
        gram = design.T @ design
        penalty = np.eye(gram.shape[0], dtype=np.float64) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficient = np.linalg.solve(gram + penalty, design.T @ encoded)
        self.fitted = True
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        require(self.fitted and self.coefficient is not None, "MODEL_NOT_FITTED")
        encoded = self._design(values) @ self.coefficient
        prediction = self._decode_response(encoded)
        require(prediction.shape == (len(values), 5), "PREDICTION_SHAPE_INVALID")
        return prediction

    def save(self, path: Path, metadata_path: Path) -> None:
        require(self.fitted, "MODEL_NOT_FITTED")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            median=self.median,
            iqr=self.iqr,
            random_weight=self.random_weight,
            random_bias=self.random_bias,
            coefficient=self.coefficient,
            response_mean=np.asarray([]) if self.response_mean is None else self.response_mean,
            response_basis=np.asarray([[]]) if self.response_basis is None else self.response_basis,
        )
        metadata = {
            "schema": "nanogpt-state-conditioned-response-model-v1",
            "status": "FITTED_ON_ALL_DEVELOPMENT_RUNS_NOT_UNBIASED_EVALUATION",
            "scheme": self.scheme,
            "seed": self.seed,
            "feature_names": self.feature_names,
            "random_feature_width": self.width,
            "ridge": self.ridge,
            "artifact": path.name,
            "artifact_sha256": file_sha256(path),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path, metadata_path: Path) -> "RandomFeatureCurveRegressor":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(file_sha256(path) == metadata["artifact_sha256"], "MODEL_ARTIFACT_HASH_MISMATCH")
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                scheme=str(metadata["scheme"]),
                seed=int(metadata["seed"]),
                feature_names=list(metadata["feature_names"]),
                width=int(metadata["random_feature_width"]),
                ridge=float(metadata["ridge"]),
            )
            model.median = np.asarray(data["median"], dtype=np.float64)
            model.iqr = np.asarray(data["iqr"], dtype=np.float64)
            model.random_weight = np.asarray(data["random_weight"], dtype=np.float64)
            model.random_bias = np.asarray(data["random_bias"], dtype=np.float64)
            model.coefficient = np.asarray(data["coefficient"], dtype=np.float64)
            response_mean = np.asarray(data["response_mean"], dtype=np.float64)
            response_basis = np.asarray(data["response_basis"], dtype=np.float64)
            model.response_mean = None if response_mean.size == 0 else response_mean
            model.response_basis = None if response_basis.size == 0 else response_basis
            model.fitted = True
            return model


def normalized_shape(displacement: np.ndarray) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=np.float64)
    scale = np.max(np.abs(displacement), axis=1, keepdims=True)
    return displacement / (scale + 1e-12)


def response_type(displacement: np.ndarray) -> str:
    values = np.asarray(displacement, dtype=np.float64)
    require(values.shape == (5,), "RESPONSE_TYPE_CURVE_SHAPE_INVALID")
    scale = float(np.max(np.abs(values)))
    if scale <= 1e-10:
        return "NEAR_LINEAR"
    linear = POSITIVE_ALPHAS * float(values[-1])
    residual = float(np.sqrt(np.mean(np.square((values - linear) / (scale + 1e-12)))))
    slopes = np.diff(np.concatenate([[0.0], values])) / np.diff(FULL_ALPHAS)
    active = np.abs(slopes) > 0.05 * scale
    signs = np.sign(slopes[active])
    turnback = bool(signs.size >= 2 and np.any(signs[1:] != signs[:-1]))
    first_sign = float(np.sign(values[0]))
    endpoint_sign = float(np.sign(values[-1]))
    sign_reversal = bool(first_sign != 0 and endpoint_sign != 0 and first_sign != endpoint_sign)
    early = float(np.mean(np.abs(slopes[:2])))
    late = float(np.mean(np.abs(slopes[-2:])))
    if turnback:
        return "TURNBACK"
    if sign_reversal:
        return "SIGN_REVERSAL"
    if residual <= 0.10:
        return "NEAR_LINEAR"
    if late <= 0.50 * (early + 1e-12):
        return "SATURATING"
    if late >= 2.0 * (early + 1e-12):
        return "ACCELERATING"
    return "OTHER"


def boundary_class(margin0: float, displacement: np.ndarray) -> str:
    start = float(margin0) >= 0.0
    end = float(margin0 + float(np.asarray(displacement)[-1])) >= 0.0
    if start and end:
        return "MAINTAIN_CORRECT"
    if start and not end:
        return "CORRECT_TO_WRONG"
    if not start and end:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


__all__ = [
    "BASE_SEED",
    "FULL_ALPHAS",
    "POSITIVE_ALPHAS",
    "RandomFeatureCurveRegressor",
    "boundary_class",
    "canonical_json",
    "file_sha256",
    "normalized_shape",
    "require",
    "response_type",
    "stable_seed",
]
