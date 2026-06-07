"""
Pauli-string utilities.

Conventions
-----------
* A Pauli string is a string over the alphabet ``{I, X, Y, Z}``.
* **Qubit 0 is the left-most character** and the most-significant tensor
  factor, i.e. ``pauli_string_to_matrix("XZ") == kron(X, Z)`` acts as
  ``X`` on qubit 0 and ``Z`` on qubit 1.  A computational-basis index
  ``b`` decodes to bits with qubit 0 in the most-significant position:
  ``bit(q) = (b >> (n - 1 - q)) & 1``.  Every module in this package
  uses this single convention.

A clause acting on ``len(support) = O(1)`` qubits admits an exact Pauli
expansion with at most ``4**len(support)`` terms, so the whole
Hamiltonian is a polynomial-size Pauli sum (this is the manuscript's
``|Lambda_k| = O(1)`` property).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import scipy.sparse as _sp

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is optional
    _sp = None
    HAVE_SCIPY = False

PAULI: Dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
PAULI_LABELS = ("I", "X", "Y", "Z")


def kron_sequence(matrices: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for m in matrices:
        out = np.kron(out, m)
    return out


@lru_cache(maxsize=4096)
def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Dense matrix of a Pauli string (cached). Returns a read-only view."""
    mat = kron_sequence([PAULI[ch] for ch in pauli_str])
    mat.setflags(write=False)
    return mat


def all_pauli_strings(n_qubits: int) -> Iterable[str]:
    if n_qubits == 0:
        yield ""
        return
    for rest in all_pauli_strings(n_qubits - 1):
        for p in PAULI_LABELS:
            yield p + rest


def pauli_decompose(
    H: np.ndarray, tol: float = 1e-12, imag_tol: float = 1e-9
) -> Dict[str, float]:
    """
    Expand a Hermitian operator ``H`` over Pauli strings.

    Returns ``{pauli_string: real_coefficient}`` keeping only coefficients
    with ``abs(.) > tol``.

    For a Hermitian ``H`` every coefficient ``Tr(P H)/2**n`` is real.  We
    therefore **assert** that the largest imaginary part is below
    ``imag_tol`` and raise otherwise: a large imaginary coefficient is the
    fingerprint of a tensor-ordering bug or a non-Hermitian input, and
    silently discarding it (as the original code did) hides exactly those
    bugs.
    """
    dim = H.shape[0]
    n = int(round(math.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(f"operator dimension {dim} is not a power of two")

    coeffs: Dict[str, float] = {}
    max_imag = 0.0
    for ps in all_pauli_strings(n):
        c = np.trace(H @ pauli_string_to_matrix(ps)) / dim
        max_imag = max(max_imag, abs(c.imag))
        if abs(c.real) > tol:
            coeffs[ps] = float(c.real)
    if max_imag > imag_tol:
        raise ValueError(
            f"non-real Pauli coefficient (max |Im|={max_imag:.2e} > {imag_tol:.0e}); "
            "input is not Hermitian or qubit ordering is inconsistent"
        )
    return coeffs


def embed_local_pauli(local_pauli: str, support_qubits: Sequence[int], n_total: int) -> str:
    """
    Embed a local Pauli string into the full ``n_total``-qubit string.

    ``support_qubits[i]`` is the global index of the qubit carrying
    ``local_pauli[i]``; ``support_qubits`` must be given in the *same*
    tensor order used to build the local operator.
    """
    chars = ["I"] * n_total
    for i, ch in enumerate(local_pauli):
        chars[support_qubits[i]] = ch
    return "".join(chars)


def accumulate_term(acc: Dict[str, float], pauli_str: str, coeff: float, tol: float = 1e-12) -> None:
    """Add ``coeff`` to ``acc[pauli_str]`` and prune to-zero entries."""
    acc[pauli_str] = acc.get(pauli_str, 0.0) + coeff
    if abs(acc[pauli_str]) <= tol:
        del acc[pauli_str]


def pauli_sum_to_dense(terms: List[dict], n: int) -> np.ndarray:
    """
    Build the dense ``2**n x 2**n`` matrix of a Pauli sum.

    ``terms`` is a list of ``{"pauli": str, "coeff": float|[re, im]}``.
    The result is explicitly Hermitised to remove round-off asymmetry.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for t in terms:
        c = t["coeff"]
        if isinstance(c, (list, tuple)):
            c = c[0] + 1j * c[1]
        H += c * pauli_string_to_matrix(t["pauli"])
    return 0.5 * (H + H.conj().T)


# Single-qubit sparse Paulis, built lazily.
def _sparse_single():
    return {k: _sp.csr_matrix(v) for k, v in PAULI.items()}


def pauli_sum_to_sparse(terms: List[dict], n: int):
    """
    Build a scipy CSR matrix for a Pauli sum (requires scipy).

    Preferred for k-lowest eigenvalue problems on larger ``n`` where the
    full dense matrix would be wasteful; raises if scipy is unavailable.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for the sparse builder")
    sp1 = _sparse_single()
    dim = 2 ** n
    H = _sp.csr_matrix((dim, dim), dtype=complex)
    for t in terms:
        c = t["coeff"]
        if isinstance(c, (list, tuple)):
            c = c[0] + 1j * c[1]
        op = _sp.identity(1, dtype=complex, format="csr")
        for ch in t["pauli"]:
            op = _sp.kron(op, sp1[ch], format="csr")
        H = H + c * op
    H = H.tocsr()
    return 0.5 * (H + H.conj().T)
