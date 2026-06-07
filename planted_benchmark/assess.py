"""
Instance assessment / diagnostics.

Fixes relative to the original ``qb_assess.py``:

* Clauses are read from the key (``reconstruct_clauses_from_key``); the
  RNG is never replayed, so non-default coefficient ranges, eigenvalue
  parameters, Bernoulli sampling, and variable block sizes all assess
  correctly.
* The planted check is the overlap with the whole degenerate ground
  **subspace**, replacing the original ``fidelity[0] > 0.999``.
* In Clifford-scrambled mode the diagnostics are explicitly split:
  ``hidden_*`` quantities are computed in the unscrambled clause basis;
  ``public_*`` quantities (entanglement, IPR) are computed on the public
  scrambled ground state.  The original code mixed the two under
  unqualified names.
* The Hamiltonian is accumulated once rather than retaining one dense
  matrix per clause; sign-structure adjacency is only built when
  frustration is requested.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from . import clifford as _cliff
from . import pauli as _pl
from .reconstruct import (
    build_hamiltonian_dense,
    reconstruct_clauses_from_key,
    reconstruct_planted_state,
)
from .spectral import _gap_above_manifold
from .verify import _reconstructed_unscrambled_dense


def _local_expect(state: np.ndarray, H_local: np.ndarray, support: List[int], n: int) -> float:
    d_g = 2 ** len(support)
    env = [q for q in range(n) if q not in support]
    perm = list(support) + env
    arr = state.reshape([2] * n).transpose(perm).reshape(d_g, -1)
    return float(np.real(np.vdot(arr, H_local @ arr)))


def _entanglement_entropy(state: np.ndarray, n: int) -> float:
    a = n // 2
    psi = state.reshape(2 ** a, 2 ** (n - a))
    s = np.linalg.svd(psi, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return float(max(0.0, -np.sum(p * np.log2(p))))


def _ipr(state: np.ndarray) -> float:
    p = np.abs(state) ** 2
    return float(np.sum(p ** 2))


def _sign_diagnostics(H: np.ndarray, compute_frustration: bool = False) -> Dict:
    off = H.copy()
    np.fill_diagonal(off, 0.0)
    nz = np.abs(off) > 1e-12
    n_off = int(np.sum(nz))
    real_part = off.real
    n_pos = int(np.sum((real_part > 1e-12) & nz))
    n_neg = int(np.sum((real_part < -1e-12) & nz))
    n_complex = int(np.sum((np.abs(off.imag) > 1e-12)))
    out = {
        "basis": "computational",
        "note": "basis-specific heuristic, not a basis-independent obstruction",
        "n_offdiagonal": n_off,
        "n_positive_real": n_pos,
        "n_negative_real": n_neg,
        "n_complex": n_complex,
        "stoquastic_in_computational_basis": (n_pos == 0 and n_complex == 0),
    }
    if compute_frustration:
        # Only build adjacency when explicitly requested.
        dim = H.shape[0]
        deg = int(np.sum(nz) // 2)
        out["adjacency_edges"] = deg
        out["dimension"] = dim
    return out


def compute_assessment(
    instance: Dict,
    key: Dict,
    max_n: int = 12,
    top_k_states: int = 6,
    eps_degeneracy: float = 1e-9,
    compute_frustration: bool = False,
) -> Dict:
    n = instance["n_qubits"]
    scrambled = bool(key.get("clifford_scrambled") and "clifford_circuit" in key)
    out: Dict = {
        "instance_id": instance.get("instance_id"),
        "n_qubits": n,
        "kind": instance.get("kind"),
        "clifford_scrambled": scrambled,
        "num_pauli_terms": instance["hamiltonian"]["num_terms"],
    }
    if n > max_n:
        out["skipped"] = f"n={n} exceeds max_n={max_n} for dense assessment"
        return out

    clauses = reconstruct_clauses_from_key(key)
    E0_key = float(key["ground_energy"])
    E0_from_clauses = float(sum(c["alpha"] * c["l0"] for c in clauses))

    # Public spectrum.
    H_pub = build_hamiltonian_dense(instance)
    evals_pub, evecs_pub = np.linalg.eigh(H_pub)
    raw, above, deg = _gap_above_manifold(evals_pub, eps_degeneracy)
    out["energy"] = {
        "E0_key": E0_key,
        "E0_from_clause_sum": E0_from_clauses,
        "lowest_eigenvalue": float(evals_pub[0]),
        "gap_identity_residual": abs(E0_key - E0_from_clauses),
        "lowest_matches_key": abs(float(evals_pub[0]) - E0_key) < 1e-7,
    }
    out["spectrum"] = {
        "raw_gap_E1_minus_E0": raw,
        "gap_above_ground_manifold": above,
        "ground_degeneracy": deg,
        "low_lying": [float(x) for x in evals_pub[: min(top_k_states, len(evals_pub))]],
    }

    # Hidden-basis objects (clause structure lives here).
    hidden_state = reconstruct_planted_state(key)
    if scrambled:
        H_hidden = _reconstructed_unscrambled_dense(key, n)
        evals_h, evecs_h = np.linalg.eigh(H_hidden)
    else:
        H_hidden, evals_h, evecs_h = H_pub, evals_pub, evecs_pub

    # Ground-subspace fidelity of the planted state (degeneracy-robust).
    gidx = np.where(evals_h - evals_h[0] < eps_degeneracy)[0]
    overlaps = evecs_h[:, gidx].conj().T @ hidden_state
    fid_sub = float(np.sum(np.abs(overlaps) ** 2))
    out["planted"] = {
        "hidden_fidelity_into_ground_subspace": fid_sub,
        "hidden_ground_degeneracy": int(len(gidx)),
        "passes": fid_sub > 1.0 - 1e-6,
        "note": "overlap with the full degenerate ground subspace, not a single eigenvector",
    }

    # Hidden clause diagnostics on the hidden ground state.
    g_hidden = evecs_h[:, 0]
    clause_energies, n_violated = [], 0
    for c in clauses:
        e = _local_expect(g_hidden, c["alpha"] * c["H_local"], c["support_qubits"], n)
        clause_energies.append(e)
        if e > c["alpha"] * c["l0"] + 1e-7:
            n_violated += 1
    out["hidden_clause_diagnostics"] = {
        "clause_energies": clause_energies,
        "num_violated_by_ground": n_violated,
        "total": float(np.sum(clause_energies)),
    }

    # Public-basis structure. The ground eigenvector is one arbitrary
    # numerical vector when the ground space is degenerate, so we also
    # report the (interpretable) planted public state U|Psi*>.
    g_pub = evecs_pub[:, 0]
    if scrambled:
        gates_pub = _cliff.circuit_from_dict(key["clifford_circuit"])
        planted_pub = _cliff.apply_clifford(gates_pub, hidden_state, dagger=False)
    else:
        planted_pub = hidden_state
    out["public_structure"] = {
        "ground_eigenvector_entanglement_half_cut": _entanglement_entropy(g_pub, n),
        "ground_eigenvector_ipr": _ipr(g_pub),
        "planted_state_entanglement_half_cut": _entanglement_entropy(planted_pub, n),
        "planted_state_ipr": _ipr(planted_pub),
        "ground_degeneracy": deg,
        "note": (
            "computed on the public (possibly scrambled) states; the ground "
            "eigenvector is one arbitrary numerical ground vector when the ground "
            "space is degenerate -- prefer the planted-state quantities in that case"
        ),
    }

    out["sign_structure"] = _sign_diagnostics(H_pub, compute_frustration)
    return out


def save_assessment(assessment: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(assessment, f, indent=2)


def load_assessment(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def print_summary(assessment: Dict) -> None:
    a = assessment
    print(f"instance : {a.get('instance_id')}  ({a.get('kind')}, n={a['n_qubits']})")
    if "skipped" in a:
        print(f"  skipped: {a['skipped']}")
        return
    print(f"  scrambled : {a['clifford_scrambled']}   pauli terms : {a['num_pauli_terms']}")
    e = a["energy"]
    print(f"  E0(key) {e['E0_key']:.10f}   lowest {e['lowest_eigenvalue']:.10f}   "
          f"gap-identity residual {e['gap_identity_residual']:.2e}")
    s = a["spectrum"]
    print(f"  raw gap {s['raw_gap_E1_minus_E0']:.3e}   gap>manifold {s['gap_above_ground_manifold']:.3e}"
          f"   ground deg {s['ground_degeneracy']}")
    p = a["planted"]
    print(f"  planted fidelity into ground subspace {p['hidden_fidelity_into_ground_subspace']:.6f}"
          f"   ({'PASS' if p['passes'] else 'FAIL'})")
    ps = a["public_structure"]
    print(f"  public ground eigvec: S(half)={ps['ground_eigenvector_entanglement_half_cut']:.3f}  "
          f"IPR={ps['ground_eigenvector_ipr']:.3e}   "
          f"planted: S(half)={ps['planted_state_entanglement_half_cut']:.3f}  "
          f"IPR={ps['planted_state_ipr']:.3e}")
    sg = a["sign_structure"]
    print(f"  stoquastic (comp. basis): {sg['stoquastic_in_computational_basis']}  "
          f"[{sg['n_positive_real']}+ / {sg['n_negative_real']}- / {sg['n_complex']} cplx]")
