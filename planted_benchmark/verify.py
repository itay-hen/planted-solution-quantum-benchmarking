"""
Verification of a planted instance against its key.

The certificate is the *energy* of the planted state together with the
fact that the planted state lies in the numerically-degenerate ground
**subspace** -- not its overlap with a single ground eigenvector.  The
original code used ``fidelity[0] > 0.999``, which false-fails on
degenerate instances (the low-clause-count regime the manuscript
explicitly treats as legitimate) because the diagonaliser returns an
arbitrary basis of the degenerate space.

For a Clifford-scrambled instance the public ground state is ``U|Psi>``;
verifying its energy against ``H'`` exercises the Clifford convention
end-to-end.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import clifford as _cliff
from . import pauli as _pl
from .reconstruct import (
    build_hamiltonian_dense,
    reconstruct_clauses_from_key,
    reconstruct_planted_state,
)


def _reconstructed_unscrambled_dense(key: Dict, n: int) -> np.ndarray:
    terms: Dict[str, float] = {}
    for c in reconstruct_clauses_from_key(key):
        for lp, coeff in _pl.pauli_decompose(c["H_local"]).items():
            gp = _pl.embed_local_pauli(lp, c["support_qubits"], n)
            _pl.accumulate_term(terms, gp, c["alpha"] * coeff)
    return _pl.pauli_sum_to_dense([{"pauli": p, "coeff": v} for p, v in terms.items()], n)


def verify_instance(instance: Dict, key: Dict, tol: float = 1e-8, max_n: int = 14) -> Dict:
    report = {"checks": [], "all_passed": True, "n_qubits": instance["n_qubits"]}

    def add(name, passed, **info):
        report["checks"].append({"name": name, "passed": bool(passed), **info})
        if not passed:
            report["all_passed"] = False

    n = instance["n_qubits"]

    # Clause-spectrum consistency: eig(H_local) must match the stored spectrum.
    max_clause_err = 0.0
    for c in reconstruct_clauses_from_key(key):
        ev = np.linalg.eigvalsh(c["H_local"])
        max_clause_err = max(max_clause_err, float(np.max(np.abs(np.sort(ev) - np.sort(c["eigenvalues"])))))
    add("clause_spectrum_consistency", max_clause_err < tol, max_err=max_clause_err)

    if n > max_n:
        add("dense_checks_skipped", True, reason=f"n={n} > max_n={max_n}")
        report["status"] = "failed" if not report["all_passed"] else "partial_dense_checks_skipped"
        report["fully_verified"] = report["status"] == "verified"
        return report

    H = build_hamiltonian_dense(instance)

    # Public planted state: U|Psi> if scrambled, else |Psi>.
    hidden = reconstruct_planted_state(key)
    if key.get("clifford_scrambled") and "clifford_circuit" in key:
        gates = _cliff.circuit_from_dict(key["clifford_circuit"])
        public = _cliff.apply_clifford(gates, hidden, dagger=False)
    else:
        public = hidden

    E0 = float(key["ground_energy"])
    E_public = float(np.real(public.conj() @ H @ public))
    evals, evecs = np.linalg.eigh(H)
    E_lowest = float(evals[0])
    add("planted_energy", abs(E_public - E0) < tol, expected=E0, planted_energy=E_public,
        diff=abs(E_public - E0))
    add("ground_energy_is_minimum", abs(E_lowest - E0) < tol, lowest_eigenvalue=E_lowest,
        key_energy=E0, diff=abs(E_lowest - E0))

    # Ground-subspace fidelity (degeneracy-robust).
    eps_deg = max(tol, 1e-8)
    ground_idx = np.where(evals - evals[0] < eps_deg)[0]
    overlaps = evecs[:, ground_idx].conj().T @ public
    fid = float(np.sum(np.abs(overlaps) ** 2))
    add("planted_in_ground_subspace", fid > 1.0 - 1e-6, fidelity_into_ground_subspace=fid,
        ground_degeneracy=int(len(ground_idx)))

    # Spectrum reconstruction (validates stored clauses + Clifford invariance).
    H_un = _reconstructed_unscrambled_dense(key, n)
    spec_recon = np.sort(np.linalg.eigvalsh(H_un))
    spec_public = np.sort(evals)
    spec_err = float(np.max(np.abs(spec_recon - spec_public)))
    add("spectrum_reconstruction", spec_err < tol, max_err=spec_err)

    report["status"] = "failed" if not report["all_passed"] else "verified"
    report["fully_verified"] = report["status"] == "verified"
    return report
