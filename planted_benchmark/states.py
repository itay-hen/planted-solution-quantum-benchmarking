"""
Random pure states, Haar-random unitaries, and orthonormal basis
completion.

All randomness flows through an explicit ``numpy.random.Generator`` so
that instances are reproducible from a seed.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def normalize(v: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > tol else v


def sample_haar_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random pure state: a normalised complex Gaussian vector."""
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return normalize(v)


def sample_haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    A Haar-random unitary via the QR construction (Mezzadri 2007): take a
    complex-Gaussian matrix, QR-decompose it, and fix the phases of ``R``'s
    diagonal so the distribution is exactly Haar.
    """
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph[np.newaxis, :]


def complete_to_orthonormal_basis(
    seeds: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """
    Return a ``dim x dim`` unitary whose first ``len(seeds)`` columns equal
    the (orthonormalised) ``seeds`` exactly, with the remaining columns a
    random orthonormal completion.

    Uses two-pass modified Gram-Schmidt for numerical orthogonality (the
    leading seed columns are preserved exactly, which matters because
    column 0 is the planted vector).
    """
    seeds = [normalize(np.asarray(s, dtype=complex)) for s in seeds]
    dim = seeds[0].size
    k = len(seeds)

    basis = np.zeros((dim, dim), dtype=complex)
    col = 0

    def orthonormalise_against(v: np.ndarray, upto: int) -> np.ndarray:
        for _ in range(2):  # two passes -> robust orthogonality
            for m in range(upto):
                v = v - np.vdot(basis[:, m], v) * basis[:, m]
        return v

    # Place the (mutually orthogonalised) seeds first.
    for s in seeds:
        v = orthonormalise_against(s.copy(), col)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            raise ValueError("seed vectors are linearly dependent")
        basis[:, col] = v / nv
        col += 1

    # Fill the remaining columns from a Haar unitary.
    U = sample_haar_unitary(dim, rng)
    j = 0
    while col < dim and j < dim:
        v = orthonormalise_against(U[:, j].copy(), col)
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            basis[:, col] = v / nv
            col += 1
        j += 1

    if col != dim:  # pragma: no cover - extremely unlikely
        raise RuntimeError("failed to complete an orthonormal basis")
    return basis


def computational_basis_index(bits: Sequence[int]) -> int:
    """Decode a bit list (qubit 0 most significant) to a basis index."""
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


def computational_basis_state(bits: Sequence[int]) -> np.ndarray:
    """The computational-basis state vector for ``bits`` (qubit 0 first)."""
    dim = 2 ** len(bits)
    v = np.zeros(dim, dtype=complex)
    v[computational_basis_index(bits)] = 1.0
    return v
