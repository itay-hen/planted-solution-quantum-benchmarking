"""
Reconstruction utilities.

Everything here reads the certification key directly -- there is **no RNG
replay**.  This is the central correctness fix relative to the original
``qb_assess.replay_clause_hamiltonians``, which re-ran the generator's
random stream and silently reproduced only its default code path.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import pauli as _pl
from .serial import dict_to_mat, dict_to_vec


def reconstruct_clauses_from_key(key: Dict) -> List[Dict]:
    """
    Return the exact clause data, read from the key.

    Each entry: ``{k, alpha, l0, eigenvalues, H_local, support_qubits, Sk}``.
    Works identically for any coefficient range, eigenvalue scheme, subset
    sampling, or block layout, because none of those are re-derived.
    """
    out = []
    for c in key["clauses"]:
        out.append(
            {
                "k": c["k"],
                "alpha": float(c["alpha"]),
                "l0": float(c["l0"]),
                "eigenvalues": np.asarray(c["eigenvalues"], dtype=float),
                "H_local": dict_to_mat(c["H_local"]),
                "support_qubits": list(c["support_qubits"]),
                "Sk": list(c.get("Sk", [])),
            }
        )
    return out


def _block_states(key: Dict, which: int) -> List[np.ndarray]:
    bs = key["planted_structure"]["block_states"]
    states = []
    for b in bs:
        vecs = b["states"]
        states.append(dict_to_vec(vecs[which]))
    return states


def reconstruct_planted_state(key: Dict, which: int = 0) -> np.ndarray:
    """
    Reconstruct the *hidden* (unscrambled) planted global state ``|Psi>`` in
    standard qubit order.  ``which`` selects among multiple planted states.

    For a Clifford-scrambled instance the public ground state is
    ``U|Psi>``; obtain it with
    ``clifford.apply_clifford(gates, reconstruct_planted_state(key))``.
    """
    blocks = key["planted_structure"]["blocks"]
    block_states = _block_states(key, which)
    n = sum(len(b) for b in blocks)

    state_block = block_states[0]
    for s in block_states[1:]:
        state_block = np.kron(state_block, s)

    block_order_qubits = [q for blk in blocks for q in blk]
    dim = 2 ** n
    state_std = np.zeros(dim, dtype=complex)
    for std_idx in range(dim):
        std_bits = [(std_idx >> (n - 1 - q)) & 1 for q in range(n)]
        block_bits = [std_bits[q] for q in block_order_qubits]
        block_idx = 0
        for b in block_bits:
            block_idx = (block_idx << 1) | b
        state_std[std_idx] = state_block[block_idx]
    return state_std


def build_hamiltonian_dense(instance: Dict) -> np.ndarray:
    """Dense Hermitised Hamiltonian from the public Pauli sum."""
    return _pl.pauli_sum_to_dense(instance["hamiltonian"]["terms"], instance["n_qubits"])


def build_hamiltonian_sparse(instance: Dict):
    """Sparse (scipy CSR) Hamiltonian from the public Pauli sum."""
    return _pl.pauli_sum_to_sparse(instance["hamiltonian"]["terms"], instance["n_qubits"])
