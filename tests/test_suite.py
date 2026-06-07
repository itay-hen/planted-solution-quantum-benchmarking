#!/usr/bin/env python3
"""
Test suite for the planted_benchmark package.

Runs without pytest:  ``python tests/test_suite.py``  (also pytest-compatible).

The tests deliberately target the behaviours the original code got wrong:
reconstruction under non-default parameters, degeneracy-robust fidelity,
the diagonal SAT channel, the Clifford convention, and loud input
validation -- alongside the basic energy / gap-identity guarantees.
"""

import os

# Pin BLAS/OMP thread pools to 1 BEFORE importing numpy: some containers
# oversubscribe threads on small complex eigh and stall on teardown at exit.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planted_benchmark import (  # noqa: E402
    CommutingConfig,
    GeneralConfig,
    PlantedSATConfig,
    compute_assessment,
    generate_commuting_instance,
    generate_general_instance,
    generate_planted_sat_instance,
    ground_gap,
    reconstruct_planted_state,
    verify_instance,
)
from planted_benchmark import clifford as cf  # noqa: E402
from planted_benchmark import pauli as pl  # noqa: E402
from planted_benchmark.generator import commuting_dynamics_reference  # noqa: E402
from planted_benchmark.reconstruct import build_hamiltonian_dense  # noqa: E402

TOL = 1e-8


def _assert_verified(instance, key, msg=""):
    rep = verify_instance(instance, key, tol=TOL)
    if not rep["all_passed"]:
        failed = [c for c in rep["checks"] if not c["passed"]]
        raise AssertionError(f"verification failed {msg}: {failed}")


# --- basic correctness ------------------------------------------------------
def test_default_energy_and_gap_identity():
    inst, key = generate_general_instance(GeneralConfig(n_qubits=8, num_clauses=10, seed=1))
    _assert_verified(inst, key, "(default general)")
    a = compute_assessment(inst, key)
    assert a["energy"]["gap_identity_residual"] < 1e-9
    assert a["energy"]["lowest_matches_key"]


def test_pauli_roundtrip_and_hermiticity_assertion():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    H = A + A.conj().T
    terms = [{"pauli": p, "coeff": c} for p, c in pl.pauli_decompose(H).items()]
    H2 = pl.pauli_sum_to_dense(terms, 3)
    assert np.allclose(H, H2, atol=1e-10)
    # a non-Hermitian operator must raise rather than silently drop the imag part
    raised = False
    try:
        pl.pauli_decompose(np.array([[0, 1], [0, 0]], dtype=complex))
    except ValueError:
        raised = True
    assert raised, "pauli_decompose should reject non-Hermitian input"


# --- the reconstruction bug class (non-default params) ----------------------
def test_reconstruction_nondefault_alpha():
    # The original replay hard-coded alpha in [0.5, 1.5]; here alpha in [2, 3].
    cfg = GeneralConfig(n_qubits=8, num_clauses=10, alpha_min=2.0, alpha_max=3.0, seed=3)
    inst, key = generate_general_instance(cfg)
    _assert_verified(inst, key, "(alpha in [2,3])")


def test_reconstruction_nondefault_lambda_and_scheme():
    cfg = GeneralConfig(n_qubits=8, num_clauses=10, eigenvalue_scheme="uniform",
                        lambda_min=-3.0, lambda_max=2.0, spectral_gap=0.7, seed=4)
    inst, key = generate_general_instance(cfg)
    _assert_verified(inst, key, "(uniform scheme, non-default lambda)")


def test_reconstruction_bernoulli_subset():
    # The original replay desynced its RNG on bernoulli sampling.
    cfg = GeneralConfig(n_qubits=8, num_clauses=12, subset_sampling="bernoulli",
                        bernoulli_prob=0.35, seed=5)
    inst, key = generate_general_instance(cfg)
    _assert_verified(inst, key, "(bernoulli subset sampling)")


def test_reconstruction_variable_block_sizes():
    # The original replay asserted-out on variable block sizes.
    cfg = GeneralConfig(n_qubits=6, variable_block_sizes=[1, 2, 3], blocks_per_clause=3,
                        num_clauses=8, seed=6)
    inst, key = generate_general_instance(cfg)
    _assert_verified(inst, key, "(variable block sizes)")


# --- degeneracy -------------------------------------------------------------
def test_degenerate_ground_subspace_fidelity():
    # Low clause count -> degenerate ground space (the manuscript's regime).
    found_degenerate = False
    for seed in range(8):
        cfg = GeneralConfig(n_qubits=6, num_clauses=2, blocks_per_clause=3, seed=seed)
        inst, key = generate_general_instance(cfg)
        a = compute_assessment(inst, key)
        _assert_verified(inst, key, f"(degenerate, seed={seed})")
        # planted state must register in the ground subspace regardless of degeneracy
        assert a["planted"]["passes"], f"planted not in ground subspace (seed={seed})"
        if a["planted"]["hidden_ground_degeneracy"] > 1:
            found_degenerate = True
    assert found_degenerate, "expected at least one degenerate instance at K=2"


# --- diagonal planted-SAT channel ------------------------------------------
def test_planted_sat_is_diagonal_and_solved():
    cfg = PlantedSATConfig(n_qubits=10, k=3, num_clauses=25, seed=2)
    inst, key = generate_planted_sat_instance(cfg)
    # ground energy must be 0 (z* satisfies all clauses)
    assert abs(key["ground_energy"]) < 1e-12
    H = build_hamiltonian_dense(inst)
    off = H.copy()
    np.fill_diagonal(off, 0.0)
    assert np.max(np.abs(off)) < 1e-12, "unscrambled SAT Hamiltonian must be diagonal"
    # planted bitstring is a ground state and energy matches
    _assert_verified(inst, key, "(planted SAT)")
    z = key["planted_assignment"]
    idx = 0
    for b in z:
        idx = (idx << 1) | int(b)
    assert abs(H[idx, idx].real - key["ground_energy"]) < 1e-12


def test_planted_sat_with_clifford_preserves_spectrum():
    plain = generate_planted_sat_instance(PlantedSATConfig(n_qubits=8, k=3, num_clauses=16, seed=9))
    scrambled = generate_planted_sat_instance(
        PlantedSATConfig(n_qubits=8, k=3, num_clauses=16, clifford=True, seed=9)
    )
    ep = np.sort(np.linalg.eigvalsh(build_hamiltonian_dense(plain[0])))
    es = np.sort(np.linalg.eigvalsh(build_hamiltonian_dense(scrambled[0])))
    assert np.allclose(ep, es, atol=1e-9), "Clifford must preserve the spectrum"
    _assert_verified(*scrambled, "(scrambled SAT)")


# --- Clifford convention ----------------------------------------------------
def _dense_unitary(gates, n):
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        U[:, j] = cf.apply_clifford(gates, e, dagger=False)
    return U


def test_clifford_symbolic_matches_dense():
    rng = np.random.default_rng(11)
    n = 3
    gates = cf.random_circuit(n, depth=4, rng=rng)
    U = _dense_unitary(gates, n)
    assert np.allclose(U @ U.conj().T, np.eye(2 ** n), atol=1e-10), "U must be unitary"
    for _ in range(10):
        p = "".join(rng.choice(list("IXYZ")) for _ in range(n))
        q, sign = cf.conjugate_pauli_string(gates, p, dagger=False)
        lhs = U @ pl.pauli_string_to_matrix(p) @ U.conj().T
        rhs = sign * pl.pauli_string_to_matrix(q)
        assert np.allclose(lhs, rhs, atol=1e-10), f"conjugation mismatch on {p}"


def test_scrambled_ground_state_is_U_psi():
    cfg = GeneralConfig(n_qubits=6, num_clauses=10, clifford=True, seed=12)
    inst, key = generate_general_instance(cfg)
    _assert_verified(inst, key, "(scrambled general)")
    H = build_hamiltonian_dense(inst)
    gates = cf.circuit_from_dict(key["clifford_circuit"])
    public = cf.apply_clifford(gates, reconstruct_planted_state(key), dagger=False)
    E = float(np.real(public.conj() @ H @ public))
    assert abs(E - key["ground_energy"]) < 1e-8


# --- input validation (loud failure) ---------------------------------------
def _expect_valueerror(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}")


def test_validation_errors():
    _expect_valueerror(
        lambda: generate_general_instance(GeneralConfig(n_qubits=4, block_size=1, num_planted=3)),
        "num_planted > block dimension",
    )
    _expect_valueerror(
        lambda: generate_general_instance(GeneralConfig(n_qubits=7, block_size=2)),
        "n not divisible by block_size",
    )
    _expect_valueerror(
        lambda: generate_general_instance(
            GeneralConfig(n_qubits=6, block_size=1, blocks_per_clause=99)
        ),
        "blocks_per_clause > num_blocks",
    )


def test_validation_rejects_bad_parameters():
    bad = [
        (lambda: generate_general_instance(GeneralConfig(n_qubits=6, alpha_min=-0.5)),
         "negative alpha_min"),
        (lambda: generate_general_instance(GeneralConfig(n_qubits=6, alpha_min=2.0, alpha_max=1.0)),
         "alpha_min > alpha_max"),
        (lambda: generate_general_instance(GeneralConfig(n_qubits=6, lambda_min=1.0, lambda_max=-1.0)),
         "lambda_max <= lambda_min"),
        (lambda: generate_general_instance(
            GeneralConfig(n_qubits=6, eigenvalue_scheme="golf_course", lambda_min=0.5)),
         "golf_course with positive lambda_min"),
        (lambda: generate_general_instance(GeneralConfig(n_qubits=6, spectral_gap=-0.1)),
         "negative spectral_gap"),
        (lambda: generate_planted_sat_instance(PlantedSATConfig(n_qubits=6, weight=0.0)),
         "non-positive SAT weight"),
        (lambda: generate_commuting_instance(CommutingConfig(n_qubits=6, weight_min=-1.0)),
         "negative projector weight"),
    ]
    for fn, label in bad:
        _expect_valueerror(fn, label)


def test_instance_ids_differ_across_parameters():
    a = generate_general_instance(GeneralConfig(n_qubits=6, num_clauses=8, seed=1))[1]["instance_id"]
    b = generate_general_instance(
        GeneralConfig(n_qubits=6, num_clauses=8, seed=1, eigenvalue_scheme="golf_course")
    )[1]["instance_id"]
    c = generate_general_instance(
        GeneralConfig(n_qubits=6, num_clauses=8, seed=1, clifford=True)
    )[1]["instance_id"]
    assert len({a, b, c}) == 3, "instance_id must encode parameter differences"


# --- multi-planted ----------------------------------------------------------
def test_multi_planted_states_in_ground_subspace():
    cfg = GeneralConfig(n_qubits=6, block_size=2, num_planted=2, num_clauses=8,
                        blocks_per_clause=2, eigenvalue_scheme="golf_course", seed=20)
    inst, key = generate_general_instance(cfg)
    H = build_hamiltonian_dense(inst)
    evals, evecs = np.linalg.eigh(H)
    gidx = np.where(evals - evals[0] < 1e-8)[0]
    for which in range(2):
        psi = reconstruct_planted_state(key, which=which)
        fid = float(np.sum(np.abs(evecs[:, gidx].conj().T @ psi) ** 2))
        assert fid > 1 - 1e-6, f"planted state {which} not in ground subspace (fid={fid})"


def test_validation_structural_and_probability():
    bad = [
        lambda: generate_general_instance(GeneralConfig(n_qubits=0)),
        lambda: generate_general_instance(GeneralConfig(n_qubits=6, block_size=0)),
        lambda: generate_general_instance(GeneralConfig(n_qubits=6, num_clauses=-1)),
        lambda: generate_general_instance(GeneralConfig(n_qubits=6, blocks_per_clause=0)),
        lambda: generate_general_instance(GeneralConfig(n_qubits=6, max_support_qubits=0)),
        lambda: generate_general_instance(GeneralConfig(n_qubits=6, clifford_depth=-1)),
        lambda: generate_general_instance(
            GeneralConfig(n_qubits=6, subset_sampling="bernoulli", bernoulli_prob=1.5)),
        lambda: generate_general_instance(
            GeneralConfig(n_qubits=6, subset_sampling="bernoulli", bernoulli_prob=-0.1)),
        lambda: generate_general_instance(
            GeneralConfig(n_qubits=4, variable_block_sizes=[1.5, 2.5])),
        lambda: generate_planted_sat_instance(
            PlantedSATConfig(n_qubits=4, planted_assignment=[0, 2, 1, 0])),
    ]
    for fn in bad:
        _expect_valueerror(fn, "structural/probability validation")


def test_multi_planted_clause_stores_planted_locals_list():
    cfg = GeneralConfig(n_qubits=6, block_size=2, num_planted=2, num_clauses=6,
                        blocks_per_clause=2, eigenvalue_scheme="golf_course", seed=20)
    inst, key = generate_general_instance(cfg)
    assert key["num_planted"] == 2
    for c in key["clauses"]:
        assert "planted_locals" in c
        assert len(c["planted_locals"]) == 2


# --- commuting + experimental dynamics --------------------------------------
def test_commuting_and_dynamics_reference():
    inst, key = generate_commuting_instance(CommutingConfig(n_qubits=6, block_size=2, seed=30))
    _assert_verified(inst, key, "(commuting)")
    dyn = commuting_dynamics_reference(inst, key, num_observables=3, times=[0.0, 1.0, 2.0], seed=1)
    # at t=0 the value is <psi0|P|psi0>; dynamics must be finite and real
    for obs in dyn["observables"]:
        for v in obs["values"]:
            assert np.isfinite(v["value"])


def test_verify_status_field():
    inst, key = generate_general_instance(GeneralConfig(n_qubits=8, num_clauses=10, seed=1))
    full = verify_instance(inst, key)
    assert full["status"] == "verified" and full["all_passed"]
    partial = verify_instance(inst, key, max_n=4)  # n=8 > 4 -> dense checks skipped
    assert partial["status"] == "partial_dense_checks_skipped"


def test_strict_sat_bit_validation():
    for bad in ([0, 0.5, 1, 0], [0, "1", 1, 0], [0, 1.0, 1, 0]):
        _expect_valueerror(
            lambda b=bad: generate_planted_sat_instance(
                PlantedSATConfig(n_qubits=4, planted_assignment=b)),
            f"non-integer planted bit {bad}",
        )
    # a clean integer assignment is accepted
    generate_planted_sat_instance(PlantedSATConfig(n_qubits=4, planted_assignment=[0, 1, 1, 0]))


def test_assessment_reports_planted_public_structure():
    inst, key = generate_general_instance(
        GeneralConfig(n_qubits=6, num_clauses=2, clifford=True, seed=4))
    ps = compute_assessment(inst, key)["public_structure"]
    for field in ("ground_eigenvector_ipr", "planted_state_ipr",
                  "planted_state_entanglement_half_cut", "ground_degeneracy"):
        assert field in ps


def test_sat_instance_ids_depend_on_planted_assignment():
    a = generate_planted_sat_instance(
        PlantedSATConfig(n_qubits=4, k=2, num_clauses=5, seed=1, planted_assignment=[0, 0, 0, 0]))[1]
    b = generate_planted_sat_instance(
        PlantedSATConfig(n_qubits=4, k=2, num_clauses=5, seed=1, planted_assignment=[1, 1, 1, 1]))[1]
    assert a["instance_id"] != b["instance_id"], "SAT IDs must depend on the planted assignment"


def test_commuting_instance_ids_depend_on_block_sizes():
    c = generate_commuting_instance(CommutingConfig(n_qubits=6, block_size=2, seed=1))[1]
    d = generate_commuting_instance(CommutingConfig(n_qubits=6, variable_block_sizes=[1, 2, 3], seed=1))[1]
    assert c["instance_id"] != d["instance_id"], "commuting IDs must depend on block layout"


def test_fully_verified_flag():
    inst, key = generate_general_instance(GeneralConfig(n_qubits=8, num_clauses=10, seed=1))
    assert verify_instance(inst, key)["fully_verified"] is True
    assert verify_instance(inst, key, max_n=4)["fully_verified"] is False


def test_dynamics_locality_validation():
    inst, key = generate_commuting_instance(CommutingConfig(n_qubits=4, block_size=2, seed=1))
    _expect_valueerror(
        lambda: commuting_dynamics_reference(inst, key, observable_locality=9),
        "observable_locality > n_qubits",
    )


# --- runner -----------------------------------------------------------------
def _all_tests():
    return [(name, obj) for name, obj in sorted(globals().items())
            if name.startswith("test_") and callable(obj)]


def main():
    passed, failed = 0, 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return 1 if failed else 0


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit avoids a slow BLAS/thread-pool teardown that can otherwise
    # hang the interpreter at exit in some containers. (Not reached under
    # pytest, which imports this module rather than running __main__.)
    os._exit(_code)
