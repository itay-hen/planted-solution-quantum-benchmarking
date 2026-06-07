"""
Spectral diagnostics.

``ground_gap`` computes the manuscript's gap quantity correctly in the
presence of degeneracy: it reports both the raw ``E_1 - E_0`` (the
paper's literal definition, which is ~0 on a degenerate ground space and
is what Fig. 2 counts) and the gap *above the whole ground manifold*,
together with the ground degeneracy.

``adiabatic_min_gap`` computes the minimum gap along a transverse-field
annealing path.  This is **not** the manuscript's spectral gap; it is an
adiabatic-hardness probe.  It is the natural quantity for the diagonal
planted-SAT channel (where the transverse field is the standard driver)
but for Haar-random or Clifford-scrambled instances the driver is
arbitrary, so interpret with care.  The problem Hamiltonian is centered
(its identity component removed) and rescaled by its spectral width by
default, so the ``(1-s)``/``s`` mixing is not distorted by the large
constant energy offset that accumulates when many clauses are summed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import pauli as _pl
from .reconstruct import build_hamiltonian_dense, build_hamiltonian_sparse


def _gap_above_manifold(evals: np.ndarray, eps: float) -> Tuple[float, float, int]:
    """Return (raw_gap, gap_above_manifold, ground_degeneracy)."""
    E0 = evals[0]
    deg = int(np.sum(evals - E0 < eps))
    raw = float(evals[1] - E0) if len(evals) > 1 else float("nan")
    if deg < len(evals):
        above = float(evals[deg] - E0)
    else:
        above = 0.0
    return raw, above, deg


def ground_gap(
    instance: Dict, eps_degeneracy: float = 1e-9, dense_threshold_qubits: int = 12, k_lowest: int = 16
) -> Dict:
    """
    Lowest-spectrum diagnostics: ``E0``, raw ``E1-E0``, gap above the
    degenerate ground manifold, and the ground degeneracy.

    Dense exact spectrum for ``n <= dense_threshold_qubits``; otherwise a
    sparse k-lowest solve (requires scipy), growing ``k`` until the ground
    manifold is resolved.
    """
    n = instance["n_qubits"]
    if n <= dense_threshold_qubits:
        evals = np.linalg.eigvalsh(build_hamiltonian_dense(instance))
    elif _pl.HAVE_SCIPY:
        from scipy.sparse.linalg import eigsh

        H = build_hamiltonian_sparse(instance)
        k = k_lowest
        while True:
            ev = np.sort(eigsh(H, k=min(k, H.shape[0] - 1), which="SA", return_eigenvectors=False))
            if int(np.sum(ev - ev[0] < eps_degeneracy)) < len(ev) or k >= H.shape[0] - 1:
                evals = ev
                break
            k *= 2
    else:
        raise RuntimeError(
            f"n={n} exceeds dense_threshold_qubits={dense_threshold_qubits} and scipy is "
            "unavailable; install scipy for the sparse k-lowest path, or lower n. "
            "(Refusing to build a dense 2**n x 2**n matrix.)"
        )
    raw, above, deg = _gap_above_manifold(evals, eps_degeneracy)
    return {
        "n_qubits": n,
        "E0": float(evals[0]),
        "raw_gap_E1_minus_E0": raw,
        "gap_above_ground_manifold": above,
        "ground_degeneracy": deg,
    }


def _transverse_field_dense(n: int) -> np.ndarray:
    dim = 2 ** n
    Hd = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        Hd -= _pl.pauli_string_to_matrix("I" * i + "X" + "I" * (n - i - 1))
    return Hd  # ground state |+...+>, gap 2


def _golden_section(f, a, b, tol=1e-7, max_iter=80):
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    s = 0.5 * (a + b)
    return s, f(s)


def adiabatic_min_gap(
    instance: Dict,
    grid_points: int = 201,
    normalize_problem: bool = True,
    eps_degeneracy: float = 1e-9,
    max_n: int = 12,
) -> Dict:
    """
    Minimum gap of ``H(s) = (1-s) H_drive + s H_problem`` with
    ``H_drive = -sum_i X_i`` (ground state ``|+...+>``).

    Returns the located minimum (coarse grid + golden-section refinement),
    its classification (interior / endpoint / closing), the gap at ``s=1``
    (the problem gap above its ground manifold), and the full ``(s, gap)``
    trace.  See the module docstring for interpretation caveats.

    This is a heuristic diagnostic, not a certified global minimum-gap
    finder: an avoided crossing narrower than the grid spacing can fall
    between grid points and be missed.  Increase ``grid_points`` to reduce
    that risk.
    """
    n = instance["n_qubits"]
    if n > max_n:
        raise ValueError(f"adiabatic_min_gap is dense and small-n only (n={n} > {max_n})")

    Hp = build_hamiltonian_dense(instance)
    if normalize_problem:
        # Remove the identity (constant) component, which inflates a
        # radius-based scale without affecting gaps, then normalize by the
        # spectral width. Planted Hamiltonians carry a large negative
        # ground-energy offset from summing many clauses.
        evals_p = np.linalg.eigvalsh(Hp)
        width = float(evals_p[-1] - evals_p[0])
        if width > 0:
            Hp = (Hp - np.eye(Hp.shape[0]) * float(np.mean(evals_p))) / width
    Hd = _transverse_field_dense(n)

    def gap_at(s: float) -> float:
        ev = np.linalg.eigvalsh((1 - s) * Hd + s * Hp)
        _, above, _ = _gap_above_manifold(ev, eps_degeneracy)
        return above

    s_grid = np.linspace(0.0, 1.0, grid_points)
    gaps = np.array([gap_at(s) for s in s_grid])
    i_min = int(np.argmin(gaps))

    if i_min in (0, len(s_grid) - 1):
        s_star, gap_star, where = float(s_grid[i_min]), float(gaps[i_min]), "endpoint"
    else:
        a, b = s_grid[i_min - 1], s_grid[i_min + 1]
        s_star, gap_star = _golden_section(gap_at, a, b, tol=1e-7)
        where = "closing" if gap_star < 1e-6 else "interior"

    return {
        "n_qubits": n,
        "s_min": float(s_star),
        "gap_min": float(gap_star),
        "gap_at_s1": float(gap_at(1.0)),
        "location": where,
        "problem_normalized": bool(normalize_problem),
        "trace": [(float(s), float(g)) for s, g in zip(s_grid, gaps)],
    }
