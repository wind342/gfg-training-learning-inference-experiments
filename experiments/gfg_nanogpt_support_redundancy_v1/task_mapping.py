from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _nullspace_mod_prime(
    matrix: np.ndarray,
    modulus: int,
) -> tuple[list[np.ndarray], int]:
    rows = np.asarray(matrix, dtype=np.int64).copy() % modulus
    row_count, column_count = rows.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        candidate = next(
            (
                row
                for row in range(pivot_row, row_count)
                if int(rows[row, column]) % modulus != 0
            ),
            None,
        )
        if candidate is None:
            continue
        rows[[pivot_row, candidate]] = rows[[candidate, pivot_row]]
        inverse = pow(int(rows[pivot_row, column]), -1, modulus)
        rows[pivot_row] = (rows[pivot_row] * inverse) % modulus
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = int(rows[row, column]) % modulus
            if factor:
                rows[row] = (rows[row] - factor * rows[pivot_row]) % modulus
        pivot_columns.append(column)
        pivot_row += 1

    free_columns = [
        column for column in range(column_count) if column not in pivot_columns
    ]
    basis: list[np.ndarray] = []
    for free_column in free_columns:
        vector = np.zeros(column_count, dtype=np.int64)
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -rows[row, free_column] % modulus
        basis.append(vector)
    return basis, len(pivot_columns)


def recover_cyclic_target_mapping(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    *,
    modulus: int = 23,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the complete opaque-token operation from observed training facts.

    The task generator relabels the cyclic group of prime order ``modulus``.
    Every observed row therefore gives a homogeneous linear constraint
    ``x[left] + x[right] - x[target] = 0`` over the finite field.  A rank of
    ``modulus - 1`` leaves one automorphism degree of freedom.  Every non-zero
    scalar multiple induces the same opaque-token operation table, so the
    validation targets are uniquely determined even though the latent residue
    labels are not.
    """

    train_inputs = np.asarray(train_inputs, dtype=np.int64)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    validation_inputs = np.asarray(validation_inputs, dtype=np.int64)
    if train_inputs.ndim != 2 or train_inputs.shape[1] != 3:
        raise RuntimeError("CSRG_TRAIN_INPUT_SHAPE_INVALID")
    if train_targets.shape != train_inputs.shape:
        raise RuntimeError("CSRG_TRAIN_TARGET_SHAPE_INVALID")
    if validation_inputs.ndim != 2 or validation_inputs.shape[1] != 3:
        raise RuntimeError("CSRG_VALIDATION_INPUT_SHAPE_INVALID")
    if modulus < 3:
        raise RuntimeError("CSRG_MODULUS_INVALID")
    operator_tokens = set(int(value) for value in train_inputs[:, 1])
    if len(operator_tokens) != 1 or next(iter(operator_tokens)) != modulus:
        raise RuntimeError("CSRG_OPERATOR_TOKEN_INVALID")
    if not np.all(validation_inputs[:, 1] == modulus):
        raise RuntimeError("CSRG_VALIDATION_OPERATOR_TOKEN_INVALID")

    equations = np.zeros((len(train_inputs), modulus), dtype=np.int64)
    observed_targets = train_targets[:, -1]
    if np.any(observed_targets < 0) or np.any(observed_targets >= modulus):
        raise RuntimeError("CSRG_OBSERVED_TARGET_INVALID")
    for index, ((left, _operator, right), target) in enumerate(
        zip(train_inputs, observed_targets, strict=True)
    ):
        for token in (left, right, target):
            if int(token) < 0 or int(token) >= modulus:
                raise RuntimeError("CSRG_TASK_TOKEN_OUT_OF_RANGE")
        equations[index, int(left)] += 1
        equations[index, int(right)] += 1
        equations[index, int(target)] -= 1

    basis, rank = _nullspace_mod_prime(equations, modulus)
    if rank != modulus - 1 or len(basis) != 1:
        raise RuntimeError(
            f"CSRG_TARGET_MAPPING_NOT_IDENTIFIED:RANK_{rank}:NULLITY_{len(basis)}"
        )
    residue_by_token = basis[0] % modulus
    if sorted(int(value) for value in residue_by_token) != list(range(modulus)):
        raise RuntimeError("CSRG_TARGET_MAPPING_NOT_BIJECTIVE")

    token_by_residue = {
        int(residue): token for token, residue in enumerate(residue_by_token)
    }

    def complete(inputs: np.ndarray, scale: int = 1) -> np.ndarray:
        scaled = (residue_by_token * scale) % modulus
        inverse = {int(residue): token for token, residue in enumerate(scaled)}
        targets = np.full_like(inputs, -1)
        targets[:, -1] = [
            inverse[(int(scaled[int(left)]) + int(scaled[int(right)])) % modulus]
            for left, _operator, right in inputs
        ]
        return targets

    recovered_train_targets = complete(train_inputs)
    if not np.array_equal(recovered_train_targets[:, -1], observed_targets):
        raise RuntimeError("CSRG_TARGET_MAPPING_OBSERVED_CONSTRAINT_MISMATCH")
    validation_targets = complete(validation_inputs)
    if not all(
        np.array_equal(validation_targets, complete(validation_inputs, scale))
        for scale in range(1, modulus)
    ):
        raise RuntimeError("CSRG_TARGET_COMPLETION_NOT_AUTOMORPHISM_INVARIANT")

    certificate = {
        "completion_unique_up_to_nonzero_field_scalar": True,
        "derived_validation_target_sha256": _array_sha256(validation_targets),
        "field_modulus": modulus,
        "latent_residue_by_token": [int(value) for value in residue_by_token],
        "linear_rank": rank,
        "mapping_basis_sha256": _array_sha256(residue_by_token),
        "nullity": len(basis),
        "observed_equation_count": len(equations),
        "observed_training_targets_reproduced": True,
        "operator_token": modulus,
        "schema": "opaque-cyclic-target-mapping-certificate-v1",
        "target_table_invariant_under_all_nonzero_field_scalars": True,
        "token_by_residue_for_selected_basis": [
            int(token_by_residue[residue]) for residue in range(modulus)
        ],
        "train_input_sha256": _array_sha256(train_inputs),
        "train_target_sha256": _array_sha256(train_targets),
        "validation_input_sha256": _array_sha256(validation_inputs),
    }
    return validation_targets, certificate
