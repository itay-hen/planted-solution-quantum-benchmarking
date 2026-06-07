"""
Local clause Hamiltonians.

Each builder returns a dict with the fields the certification key stores:

    H_local        dense local clause matrix (source of truth)
    eigenbasis     unitary whose columns are the clause eigenvectors
    eigenvalues    spectrum sorted ascending (for inspection; min first)
    l0             minimum eigenvalue
    planted_local  the planted restriction (an eigenvector at l0)

The planted restriction always lies in the minimum-eigenvalue space, so
the global planted product state minimises every clause and is a ground
state of ``H = sum_k alpha_k H^{(k)}`` with ``alpha_k > 0``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .states import complete_to_orthonormal_basis, computational_basis_state, normalize

VALID_SCHEMES = ("bimodal", "golf_course", "uniform", "linear")


def clause_eigenvalues(
    dim: int,
    n_ground: int,
    scheme: str,
    rng: np.random.Generator,
    lmin: float = -1.0,
    lmax: float = 1.0,
    gap: float = 0.5,
) -> np.ndarray:
    """
    Build a length-``dim`` spectrum with the first ``n_ground`` entries at
    ``lmin`` (the planted/ground subspace) and the rest set per ``scheme``.

    Note on ``bimodal`` (the manuscript's default): half of the *excited*
    levels are placed at ``lmin`` as well, so the clause's minimum-energy
    space is **larger** than the planted span -- this broadening is
    intentional and is what the manuscript studies.  If you want the
    minimum-energy space to be *exactly* the planted span (e.g. for clean
    multi-planted benchmarks), use ``golf_course``.
    """
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"unknown eigenvalue scheme {scheme!r}; choose from {VALID_SCHEMES}")
    n_exc = dim - n_ground
    ground = np.full(n_ground, lmin)

    if scheme == "bimodal":
        n_low = n_exc // 2
        exc = np.concatenate([np.full(n_low, lmin), np.full(n_exc - n_low, lmax)])
    elif scheme == "golf_course":
        exc = np.zeros(n_exc)
    elif scheme == "uniform":
        lo = lmin + gap if gap and gap > 0 else lmin + 1e-6
        exc = rng.uniform(lo, lmax, size=n_exc)
    else:  # linear
        exc = np.linspace(lmin + gap, lmax, n_exc) if n_exc > 0 else np.array([])

    return np.concatenate([ground, exc])


def _from_eigendata(eigenbasis: np.ndarray, eigs: np.ndarray, planted_locals) -> Dict:
    H = eigenbasis @ np.diag(eigs) @ eigenbasis.conj().T
    H = 0.5 * (H + H.conj().T)
    pls = [normalize(np.asarray(p, dtype=complex)) for p in planted_locals]
    return {
        "H_local": H,
        "eigenbasis": eigenbasis,
        "eigenvalues": np.sort(eigs.real),
        "l0": float(np.min(eigs.real)),
        "planted_local": pls[0],
        "planted_locals": pls,
    }


def build_generic_clause(
    planted_locals: Sequence[np.ndarray],
    rng: np.random.Generator,
    scheme: str = "bimodal",
    lmin: float = -1.0,
    lmax: float = 1.0,
    gap: float = 0.5,
) -> Dict:
    """
    A maximally-generic Hermitian clause whose minimum-energy space
    contains the (mutually orthogonal) ``planted_locals``.

    For a single planted state pass a length-1 list.
    """
    planted_locals = [normalize(np.asarray(p, dtype=complex)) for p in planted_locals]
    m = len(planted_locals)
    dim = planted_locals[0].size
    eigenbasis = complete_to_orthonormal_basis(planted_locals, rng)
    eigs = clause_eigenvalues(dim, m, scheme, rng, lmin, lmax, gap)
    return _from_eigendata(eigenbasis, eigs, [eigenbasis[:, a] for a in range(m)])


def build_diagonal_sat_clause(support_size: int, violating_index: int, planted_index: int) -> Dict:
    """
    A diagonal (classical) clause: ``+1`` on the single violating
    computational-basis assignment, ``0`` elsewhere.

    This is the manuscript's diagonal embedding channel.  The eigenbasis
    is the computational basis (identity), so a sum of such clauses is a
    diagonal Pauli Hamiltonian whose ground state is the planted bitstring.
    """
    dim = 2 ** support_size
    diag = np.zeros(dim)
    diag[violating_index] = 1.0
    H = np.diag(diag).astype(complex)
    planted = computational_basis_state(
        [(planted_index >> (support_size - 1 - i)) & 1 for i in range(support_size)]
    )
    return {
        "H_local": H,
        "eigenbasis": np.eye(dim, dtype=complex),
        "eigenvalues": np.sort(diag),
        "l0": 0.0,
        "planted_local": planted,
        "planted_locals": [planted],
    }
