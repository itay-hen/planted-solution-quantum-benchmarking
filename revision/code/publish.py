"""
Hardened publication of a benchmark instance (Sec. III.A of the manuscript).

Identity-offset handling and the limits of hardening:

1. *Identity coefficient.* c_I = sum_k alpha_k tr(H^(k))/d_k is public and invariant
   under Clifford conjugation.  For the projector ensemble tr(H^(k)) = -1, so with a
   uniform local dimension d one has E0 = d c_I exactly, and a scalar energy score can
   be produced without any computation.  `publish` writes the traceless part
   H - c_I I and stores the offset in the key, so that the public energy scale is
   E0_public = E0 - c_I.

2. *Concentration of E0.*  With a fixed local ground energy lambda_0 = -1, E0 = -sum_k
   alpha_k, which a generator-aware solver can guess to within ~sigma_alpha sqrt(K).

   Randomising lambda_0 per clause does NOT help, and the self-test below demonstrates
   this: for a clause with spectrum {lambda_0, lambda_0 + gamma} the shift enters only
   the identity component, so the traceless public Hamiltonian is unchanged up to rounding and the
    public ground energy in the half-rank ensemble is -(gamma/2) sum_k alpha_k either way.  What does change the
   public operator is the per-clause *scale* gamma_k (`random_scale_spectrum`), since
   the public Hamiltonian is sum_k alpha_k gamma_k (Pi_k - I/2).  That is equivalent to
   widening the distribution of the weights: the public ground energy is still a sum of
   K independent contributions, so its relative fluctuation still falls as 1/sqrt(K).
   No choice of local spectra removes this concentration, and a scalar energy score
   should therefore not be the only test applied to these instances.

Usage:
    inst = pb.generate(...)
    pub, key = publish(inst)
    json.dump(pub, open("instance.json", "w"));  json.dump(key, open("key.json", "w"))
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb  # noqa: E402


def random_scale_spectrum(d: int, rng, scheme: str = "bimodal", l0: float = -1.0,
                          scale=(0.5, 2.0)):
    """Eigenvalues of a clause whose local *gap* gamma is drawn from `scale`.

    Unlike a shift of lambda_0 (which is absorbed into the identity component and
    therefore invisible in the traceless public Hamiltonian), gamma_k changes the
    published operator.  It does not defeat concentration of the public ground energy;
    see the module docstring."""
    gamma = rng.uniform(*scale)
    if scheme == "bimodal":
        n_low = d // 2
        return np.concatenate([np.full(n_low, l0), np.full(d - n_low, l0 + gamma)])
    if scheme == "golf_course":
        e = np.full(d, l0 + gamma)
        e[0] = l0
        return e
    raise ValueError(scheme)


def publish(inst: pb.Instance, traceless: bool = True, tol: float = 1e-14):
    """Returns (public_instance, key) dicts.  The public object carries only the Pauli
    list; the key carries E0, the identity offset, the partition, the block states, the
    clause data and the Clifford gate list."""
    terms = pb.pauli_sum(inst, tol=tol)
    ident = "I" * inst.n
    c_id = float(terms.get(ident, 0.0))
    if traceless and ident in terms:
        del terms[ident]
    pub = {
        "schema": "planted-benchmark/public/v3",
        "n_qubits": inst.n,
        "hamiltonian": {"format": "pauli_sum", "num_terms": len(terms),
                        "terms": [{"pauli": p, "coeff": c} for p, c in sorted(terms.items())]},
        "metadata": {"traceless": bool(traceless), "clifford_scrambled": inst.clifford is not None,
                     "coefficient_tolerance": tol},
    }
    key = {
        "schema": "planted-benchmark-key/v3",
        "n_qubits": inst.n,
        "ground_energy": inst.E0,
        "identity_offset": c_id,
        "ground_energy_public": inst.E0 - (c_id if traceless else 0.0),
        "scheme": inst.scheme,
        "num_planted": inst.num_planted,
        "blocks": inst.blocks,
        "block_states": [[{"re": s.real.tolist(), "im": s.imag.tolist()} for s in bs]
                         for bs in inst.block_states],
        "clauses": [{"Sk": c.Sk, "qubits": c.qubits, "alpha": c.alpha, "l0": c.l0,
                     "eigenvalues": c.eigs.tolist(),
                     "H_local": {"re": c.h.real.tolist(), "im": c.h.imag.tolist()}}
                    for c in inst.clauses],
        "clifford": inst.clifford.to_dict() if inst.clifford is not None else None,
    }
    return pub, key


def reconstruct(key):
    """Reconstruct the ideal instance solely from a JSON-round-tripped v3 key.

    Local matrices and the gate list are authoritative; no RNG is replayed.
    Older incomplete v3 keys without H_local must be re-exported.
    """
    if key.get("schema") != "planted-benchmark-key/v3":
        raise ValueError("expected a planted-benchmark-key/v3 key")
    def array(obj):
        return np.asarray(obj["re"]) + 1j * np.asarray(obj["im"])
    clauses = []
    for c in key["clauses"]:
        if "H_local" not in c:
            raise ValueError("incomplete key: missing H_local; re-export the instance")
        h = array(c["H_local"])
        clauses.append(pb.Clause(c["Sk"], c["qubits"], h, np.asarray(c["eigenvalues"]),
                                 c["l0"], c["alpha"], pb._local_kernel(h, c["l0"])))
    circ = None
    if key["clifford"] is not None:
        circ = pb.v5.CliffordCircuit(key["n_qubits"])
        constructors = {"H": pb.v5.H_gate, "S": pb.v5.S_gate, "Sdg": pb.v5.Sdg_gate,
                        "CNOT": pb.v5.CNOT_gate, "CZ": pb.v5.CZ_gate}
        for gate in key["clifford"]["gates"]:
            name = gate["name"].split("(", 1)[0]
            if name not in constructors:
                raise ValueError(f"unsupported Clifford gate: {name}")
            circ.gates.append(constructors[name](*gate["qubits"]))
    return pb.Instance(key["n_qubits"], key["blocks"],
                       [[array(s) for s in bs] for bs in key["block_states"]],
                       clauses, 0, key["scheme"], key["num_planted"], circ)


def verify_public(pub, key, atol=1e-10):
    """Check local certificates and the public Pauli sum without full-system ED.

    Raises ValueError on a mismatch; checks all planted restrictions and the
    smallest local eigenvalues, then reconstructs the public sum from the key.
    """
    inst = reconstruct(key)
    if sorted(q for b in inst.blocks for q in b) != list(range(inst.n)):
        raise ValueError("blocks must partition all qubits exactly once")
    for block, states in zip(inst.blocks, inst.block_states):
        basis = np.column_stack(states)
        if (basis.shape != (2 ** len(block), inst.num_planted)
                or not np.allclose(basis.conj().T @ basis, np.eye(inst.num_planted), atol=atol, rtol=0)):
            raise ValueError("block states must be an orthonormal planted frame")
    if pub.get("schema") != "planted-benchmark/public/v3" or pub["n_qubits"] != inst.n:
        raise ValueError("public schema or qubit count does not match key")
    for c in inst.clauses:
        if c.alpha <= 0 or not np.allclose(c.h, c.h.conj().T, atol=atol, rtol=0):
            raise ValueError("invalid weight or non-Hermitian clause")
        if not np.allclose(np.linalg.eigvalsh(c.h), np.sort(c.eigs), atol=atol, rtol=0):
            raise ValueError("local spectrum does not match matrix")
        if abs(float(np.min(c.eigs)) - c.l0) > atol:
            raise ValueError("local ground energy is not minimal")
        if c.qubits != [q for b in c.Sk for q in inst.blocks[b]]:
            raise ValueError("clause qubit order does not match block support")
        for a in range(inst.num_planted):
            v = np.array([1.0 + 0j])
            for b in c.Sk:
                v = np.kron(v, inst.block_states[b][a])
            if abs(np.linalg.norm(v) - 1) > atol or np.linalg.norm(c.h @ v - c.l0 * v) > atol:
                raise ValueError("local planted restriction is not a normalized ground vector")
    rebuilt, rebuilt_key = publish(inst, traceless=pub["metadata"]["traceless"],
                                   tol=pub["metadata"].get("coefficient_tolerance", 1e-14))
    items = pub["hamiltonian"]["terms"]
    actual = {t["pauli"]: t["coeff"] for t in items}
    expected = {t["pauli"]: t["coeff"] for t in rebuilt["hamiltonian"]["terms"]}
    if len(actual) != len(items) or pub["hamiltonian"]["num_terms"] != len(items):
        raise ValueError("duplicate terms or incorrect term count")
    if max((abs(actual.get(p, 0) - expected.get(p, 0))
            for p in actual.keys() | expected.keys()), default=0) > atol:
        raise ValueError("public Hamiltonian does not match key")
    for field in ("ground_energy", "identity_offset", "ground_energy_public"):
        if abs(key[field] - rebuilt_key[field]) > atol:
            raise ValueError(f"inconsistent {field}")
    return True


def _selftest():
    for scheme in ("bimodal", "golf_course"):
        for cl in (False, True):
            inst = pb.generate(8, 12, 5, scheme=scheme, clifford=cl)
            pub, key = publish(inst)
            n = inst.n
            H = pb.pauli_dense_fast([(t["pauli"], t["coeff"]) for t in pub["hamiltonian"]["terms"]], n)
            w = np.linalg.eigvalsh(H)
            assert abs(w[0] - key["ground_energy_public"]) < 1e-10, (scheme, cl, w[0], key)
            assert not any(set(t["pauli"]) == {"I"} for t in pub["hamiltonian"]["terms"])
            assert abs(np.trace(H)) < 1e-9
    # a per-clause lambda_0 shift leaves the traceless public Hamiltonian unchanged
    inst = pb.generate(8, 10, 3, scheme="bimodal")
    ident = "I" * inst.n
    before = {p: c for p, c in pb.pauli_sum(inst).items() if p != ident}
    rng = np.random.default_rng(0)
    for c in inst.clauses:
        delta = rng.uniform(-1, 1)
        c.h = c.h + delta * np.eye(c.h.shape[0])
        c.eigs = c.eigs + delta
        c.l0 = c.l0 + delta
    after = {p: c for p, c in pb.pauli_sum(inst).items() if p != ident}
    assert set(before) == set(after)
    assert max(abs(before[p] - after[p]) for p in before) < 1e-12
    _, key = publish(inst)
    assert abs(key["ground_energy_public"] + sum(c.alpha for c in inst.clauses)) < 1e-10

    # a per-clause gap does change the published operator
    rng = np.random.default_rng(1)
    e = random_scale_spectrum(8, rng)
    assert abs(e.min() - e[0]) < 1e-15 and 0.5 <= (e.max() - e.min()) <= 2.0
    print("publish.py self-test OK: traceless publication reproduces the public ground energy;"
          " identity term removed; trace zero; lambda_0 randomisation verified to leave the"
          " traceless public Hamiltonian unchanged (so it is not a mitigation)")


if __name__ == "__main__":
    _selftest()
