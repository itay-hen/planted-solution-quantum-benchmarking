"""
Automated checks required by the revision plan (Sec. O of the instruction file).
Run:  python tests_pbcore.py        (exits nonzero on any failure)
"""
import os
import random
import sys
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb  # noqa: E402
import qb_planted_v5 as v5  # noqa: E402

RESULTS = []


def check(name):
    def deco(f):
        def run():
            try:
                msg = f()
                RESULTS.append((name, True, msg or ""))
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        run.__name__ = f.__name__
        return run
    return deco


# 0. generator equivalence with the code used for the submitted manuscript
@check("T0 pbcore.generate reproduces qb_planted_v5 Pauli sums (incl. Clifford)")
def t0():
    worst = 0.0
    cases = [(6, 6, 11, "bimodal", False, 1), (8, 10, 12, "bimodal", True, 1),
             (8, 7, 13, "golf_course", True, 1), (6, 5, 14, "golf_course", False, 2),
             (9, 8, 15, "uniform", True, 1)]
    for n, K, seed, sch, cl, m in cases:
        cfg = v5.GeneralConfig(n_qubits=n, block_size=1, num_planted=m, num_clauses=K,
                               blocks_per_clause=3, eigenvalue_scheme=sch, clifford=cl, seed=seed)
        inst_v5, key_v5 = v5.generate_general_instance(cfg)
        ref = {t["pauli"]: t["coeff"] for t in inst_v5["hamiltonian"]["terms"]}
        mine = pb.pauli_sum(pb.generate(n, K, seed, scheme=sch, clifford=cl, num_planted=m))
        assert set(ref) == set(mine), f"term sets differ for {(n, K, seed, sch)}"
        worst = max(worst, max(abs(ref[p] - mine[p]) for p in ref))
        assert abs(key_v5["ground_energy"] - pb.generate(n, K, seed, scheme=sch, num_planted=m).E0) < 1e-12
    assert worst < 1e-12
    return f"max coefficient deviation {worst:.1e} over {len(cases)} configurations"


# 0b. nested prefixes
@check("T0b clause sequences are nested across K for fixed seed")
def t0b():
    big = pb.generate(10, 30, 99)
    for K in (5, 12, 29):
        small = pb.generate(10, K, 99)
        for a, b in zip(small.clauses, big.clauses[:K]):
            assert a.Sk == b.Sk and np.allclose(a.h, b.h) and a.alpha == b.alpha
    return "prefix property holds for K=5,12,29 of K'=30"


# 1. Hermiticity
@check("T1 every clause and total Hamiltonian Hermitian")
def t1():
    inst = pb.generate(8, 12, 3, clifford=True)
    for c in inst.clauses:
        assert np.allclose(c.h, c.h.conj().T, atol=1e-14)
    terms = pb.pauli_sum(inst)
    H = pb.pauli_dense(terms, 8)
    assert np.allclose(H, H.conj().T, atol=1e-13)
    return "8 qubits, 12 clauses, Clifford on"


# 2. planted energy
@check("T2 <Psi*|H|Psi*> = E0 (unscrambled) within 1e-12 relative")
def t2():
    worst = 0
    for seed in range(5):
        inst = pb.generate(10, 16, seed)
        H = pb.pauli_dense(pb.pauli_sum(inst), 10)
        v = pb.planted_vector(inst)
        e = np.vdot(v, H @ v).real
        worst = max(worst, abs(e - inst.E0) / abs(inst.E0))
    assert worst < 1e-12
    return f"max rel. error {worst:.1e}"


# 3. PSD after local shift
@check("T3 A_k = alpha_k (H^(k) - lambda_0 I) is PSD and annihilates planted restriction")
def t3():
    inst = pb.generate(9, 20, 4, scheme="bimodal")
    for c in inst.clauses:
        A = c.alpha * (c.h - c.l0 * np.eye(c.h.shape[0]))
        w = np.linalg.eigvalsh(A)
        assert w.min() > -1e-12
    B = pb.dense_B(inst)
    v = pb.planted_vector(inst)
    assert np.linalg.norm(B @ v) < 1e-12
    return "min local eigenvalue >= -1e-12; ||B Psi*|| < 1e-12"


# 4. spectra before/after Clifford
@check("T4 spectrum invariant under Clifford conjugation; U maps planted state to ground state")
def t4():
    for seed in (5, 6):
        inst = pb.generate(8, 12, seed, clifford=True, clifford_depth=5)
        H0 = pb.pauli_dense(pb.pauli_sum(inst, apply_clifford=False), 8)
        H1 = pb.pauli_dense(pb.pauli_sum(inst, apply_clifford=True), 8)
        assert np.allclose(np.linalg.eigvalsh(H0), np.linalg.eigvalsh(H1), atol=1e-11)
    return "eigenvalues agree to 1e-11"


# 4b. gate tables are correct conjugations
@check("T4b Clifford gate conjugation tables match matrix conjugation")
def t4b():
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.diag([1, 1j])
    CN = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    CZ = np.diag([1, 1, 1, -1])
    mats = {"H": Hm, "S": S, "Sdg": S.conj().T}
    for name, G in mats.items():
        gate = {"H": v5.H_gate, "S": v5.S_gate, "Sdg": v5.Sdg_gate}[name](0)
        for p in "XYZ":
            q, ph = gate.conjugate(p)
            P, Q = pb._P1[p], pb._P1[q]
            # table implements either G P G^dag or G^dag P G consistently; record which
            fwd = np.allclose(G @ P @ G.conj().T, ph * Q)
            bwd = np.allclose(G.conj().T @ P @ G, ph * Q)
            assert fwd or bwd, (name, p)
    for name, G, fn in (("CNOT", CN, v5.CNOT_gate), ("CZ", CZ, v5.CZ_gate)):
        gate = fn(0, 1)
        for a in "IXYZ":
            for b in "IXYZ":
                q, ph = gate.conjugate(a + b)
                P = np.kron(pb._P1[a], pb._P1[b])
                Q = np.kron(pb._P1[q[0]], pb._P1[q[1]])
                assert np.allclose(G @ P @ G.conj().T, ph * Q) or np.allclose(G.conj().T @ P @ G, ph * Q), (name, a + b)
    return "all 1- and 2-qubit tables verified"


@check("T4c code convention: H' = V H V^dag with V = g_1 g_2 ... g_d (gate list order); target V|Psi*>")
def t4c():
    n = 6
    inst = pb.generate(n, 8, 21, clifford=True, clifford_depth=3)
    H = pb.pauli_dense(pb.pauli_sum(inst, apply_clifford=False), n)
    Hp = pb.pauli_dense(pb.pauli_sum(inst), n)
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.diag([1, 1j])

    def emb(g):
        nm, q = g.name.split("(")[0], g.qubits
        if nm in ("H", "S", "Sdg"):
            m = {"H": Hm, "S": S, "Sdg": S.conj().T}[nm]
            out = np.array([[1]])
            for j in range(n):
                out = np.kron(out, m if j == q[0] else np.eye(2))
            return out
        D = 2 ** n
        out = np.zeros((D, D), complex)
        for b in range(D):
            bits = [(b >> (n - 1 - i)) & 1 for i in range(n)]
            if nm == "CNOT":
                nb = bits.copy()
                nb[q[1]] ^= bits[q[0]]
                out[int("".join(map(str, nb)), 2), b] = 1
            else:
                out[b, b] = -1 if bits[q[0]] and bits[q[1]] else 1
        return out
    V = np.eye(2 ** n)
    for g in inst.clifford.gates:
        V = V @ emb(g)
    assert np.allclose(Hp, V @ H @ V.conj().T, atol=1e-12)
    t = V @ pb.planted_vector(inst)
    assert abs(np.vdot(t, Hp @ t).real - inst.E0) < 1e-12
    return f"{len(inst.clifford.gates)} gates; convention confirmed"


# 5. term count bound
@check("T5 L <= sum_k 4^|Lambda_k| <= K 4^r; Clifford: L(V H V^dag) = L(H), equal one-norm")
def t5():
    for seed in range(4):
        inst = pb.generate(12, 24, seed, clifford=True)
        t0 = pb.pauli_sum(inst, apply_clifford=False)
        t1 = pb.pauli_sum(inst, apply_clifford=True)
        bound = sum(4 ** len(c.qubits) for c in inst.clauses)
        assert len(t0) <= bound <= inst.K * 4 ** 3
        assert len(t1) == len(t0)
        s0 = pb.pauli_stats(t0)
        s1 = pb.pauli_stats(t1)
        assert abs(s1["one_norm"] - s0["one_norm"]) < 1e-9
        assert sorted(abs(c) for c in t0.values()) == sorted(abs(c) for c in t1.values())
    return f"example: L={len(t0)}, L'={len(t1)}, bound={bound}, max weight {s0['max_weight']}->{s1['max_weight']}"


# 6. structural dimension vs exact diagonalisation
@check("T6 structural ground-space dimension equals ED multiplicity (bimodal + golf_course)")
def t6():
    count = 0
    mism = []
    for scheme in ("bimodal", "golf_course"):
        for n in (6, 7, 8, 9):
            for K in (2, 3, 4, 5, 6, 8, 10):
                for seed in range(12):
                    inst = pb.generate(n, K, 1000 * n + 10 * K + seed, scheme=scheme)
                    sd = pb.structural_dimension(inst)
                    w = np.linalg.eigvalsh(pb.dense_B(inst))
                    scale = sum(c.alpha * (c.eigs.max() - c.l0) for c in inst.clauses)
                    tol = 1e3 * np.finfo(float).eps * scale * 2 ** n
                    g = int(np.sum(w < tol))
                    count += 1
                    if g != sd["dim"]:
                        mism.append((scheme, n, K, seed, g, sd))
    assert not mism, mism[:3]
    return f"{count} instances, 0 mismatches"


# 6b. deflated gap agrees with dense gap above manifold
@check("T6b matrix-free deflated gap equals ED gap above ground manifold")
def t6b():
    worst = 0
    for seed in range(8):
        for scheme in ("bimodal", "golf_course"):
            inst = pb.generate(9, 10 + seed, 300 + seed, scheme=scheme)
            sd = pb.structural_dimension(inst)
            w = np.linalg.eigvalsh(pb.dense_B(inst))
            exact = w[sd["dim"]]
            gap = pb.excitation_gap(inst)
            worst = max(worst, abs(gap - exact))
    assert worst < 1e-8
    return f"max |Delta_Lanczos - Delta_ED| = {worst:.1e}"


# 7. coverage formula vs Monte Carlo
@check("T7 E[N_unc] and P(full coverage) formulas vs Monte Carlo")
def t7():
    rng = random.Random(1)
    out = []
    for M, K in ((10, 8), (20, 20), (30, 40), (60, 90)):
        T = 20000
        unc, full = 0, 0
        for _ in range(T):
            cov = set()
            for _ in range(K):
                cov.update(rng.sample(range(M), 3))
            u = M - len(cov)
            unc += u
            full += (u == 0)
        e_mc, p_mc = unc / T, full / T
        e_th, p_th = pb.expected_uncovered(M, 3, K), pb.prob_full_coverage(M, 3, K)
        se = np.sqrt(p_th * (1 - p_th) / T)
        assert abs(p_mc - p_th) < 5 * se + 1e-9, (M, K, p_mc, p_th)
        assert abs(e_mc - e_th) / max(e_th, 1e-3) < 0.05, (M, K, e_mc, e_th)
        out.append(f"M={M},K={K}: P {p_mc:.4f}/{p_th:.4f}")
    return "; ".join(out)


# 8. multi-planted dimension = m^c * prod 2^{|A_i|}
@check("T8 multi-planted ground dimension = m^c x uncovered factor (golf_course, exact local spaces)")
def t8():
    count = 0
    for n in (6, 8):
        for K in (1, 2, 3, 4, 6):
            for seed in range(10):
                inst = pb.generate(n, K, 5000 + 100 * n + 10 * K + seed, scheme="golf_course",
                                   num_planted=2)
                sd = pb.structural_dimension(inst)
                unc, comps = pb.hypergraph_components(inst.M, [c.Sk for c in inst.clauses])
                pred = 2 ** len(comps) * 2 ** len(unc)
                w = np.linalg.eigvalsh(pb.dense_B(inst))
                g = int(np.sum(w < 1e-9))
                assert g == pred == sd["dim"], (n, K, seed, g, pred)
                # planted global states are ground states
                for a in range(2):
                    v = pb.planted_vector(inst, a)
                    assert np.linalg.norm(pb.dense_B(inst) @ v) < 1e-10
                count += 1
    # blocks of size 2 with m = 3 (m <= 2^{|A_i|})
    for seed in range(6):
        inst = pb.generate(8, 3, 7000 + seed, block_size=2, blocks_per_clause=2,
                           scheme="golf_course", num_planted=3)
        unc, comps = pb.hypergraph_components(inst.M, [c.Sk for c in inst.clauses])
        pred = 3 ** len(comps) * 4 ** len(unc)
        g = int(np.sum(np.linalg.eigvalsh(pb.dense_B(inst)) < 1e-9))
        assert g == pred, (seed, g, pred)
        count += 1
    return f"{count} instances, 0 mismatches"


@check("T8b bimodal multi-planted: local kernel strictly larger than planted span (theorem hypothesis fails)")
def t8b():
    inst = pb.generate(6, 3, 17, scheme="bimodal", num_planted=2)
    dims = {c.kernel.shape[1] for c in inst.clauses}
    assert dims == {2 + (8 - 2) // 2}, dims
    return f"local kernel dims {dims} (m=2, d=8)"


@check("T9 product-state baseline: analytic gradient vs finite differences; fast energy vs dense expectation")
def t9():
    from scipy.optimize import check_grad
    import run_product_baseline as rb
    worst_g = worst_e = 0.0
    for n, K, seed, cl in ((6, 8, 3, True), (8, 10, 4, False)):
        inst = pb.generate(n, K, seed, clifford=cl)
        terms = pb.pauli_sum(inst)
        q, a, cs = rb.compile_terms(terms, n)
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 3, 2 * n)
        worst_g = max(worst_g, check_grad(lambda z: rb.energy_grad(z, q, a, cs, n)[0],
                                          lambda z: rb.energy_grad(z, q, a, cs, n)[1], x))
        H = pb.pauli_dense_fast(terms, n)
        th, ph = x[:n], x[n:]
        v = np.array([1.0 + 0j])
        for j in range(n):
            v = np.kron(v, np.array([np.cos(th[j] / 2), np.exp(1j * ph[j]) * np.sin(th[j] / 2)]))
        worst_e = max(worst_e, abs(np.vdot(v, H @ v).real - rb.energy_grad(x, q, a, cs, n)[0]))
    assert worst_g < 1e-6 and worst_e < 1e-12
    return f"max grad error {worst_g:.1e}; max energy error {worst_e:.1e}"


@check("T10 publish: traceless publication; lambda_0 randomisation leaves the public Hamiltonian unchanged")
def t10():
    import publish
    publish._selftest()
    return "identity removed, trace zero, public E0 reproduced; lambda_0 shifts invisible"


@check("T11 coordinate witnesses cannot certify half-rank uniqueness; Haar draws give dim 1")
def t11():
    import witness_search as ws
    sup = [[0, 1, 2], [2, 3, 4]]
    best = ws.min_coordinate_witness(5, sup)
    haar = ws.haar_dimension(5, sup, trials=5)
    assert best == 4, best
    assert haar == {1}, haar
    return f"min coordinate witness = {best}; Haar draws = {sorted(haar)}"


@check("T12 prefixes preserve the Clifford frame and reject invalid lengths")
def t12():
    inst = pb.generate(6, 8, 7, clifford=True)
    assert pb.pauli_sum(inst) == pb.pauli_sum(inst.prefix(inst.K))
    small = inst.prefix(4)
    assert small.clifford is inst.clifford and small.K == 4
    bare = pb.pauli_sum(small, apply_clifford=False)
    expected = dict(inst.clifford.conjugate_pauli(p, c) for p, c in bare.items())
    assert pb.pauli_sum(small) == expected
    for k in (-1, 9, 1.5):
        try:
            inst.prefix(k)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid prefix {k}")
    return "full and shorter prefixes retain the same circuit"


@check("T13 non-divisible block sizes fail explicitly; valid blocks cover all qubits")
def t13():
    for n, bs in ((7, 2), (7, 0), (0, 1)):
        try:
            pb.generate(n, 5, 3, block_size=bs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid n={n}, block_size={bs}")
    inst = pb.generate(6, 5, 3, block_size=2)
    assert sorted(q for b in inst.blocks for q in b) == list(range(6))
    assert pb.planted_vector(inst).shape == (64,)
    return "unsupported remainders rejected before generation"


@check("T14 publication keys reconstruct after JSON serialization and reject tampering")
def t14():
    import copy
    import json
    import publish
    for scheme, cl, m in (("bimodal", False, 1), ("bimodal", True, 1),
                          ("golf_course", True, 2)):
        inst = pb.generate(6, 6, 42, scheme=scheme, clifford=cl, num_planted=m)
        for traceless in (False, True):
            pub, key = json.loads(json.dumps(publish.publish(inst, traceless=traceless)))
            assert publish.verify_public(pub, key)
            rebuilt = publish.reconstruct(key)
            assert pb.pauli_sum(rebuilt) == pb.pauli_sum(inst)
            for corrupt in ("public", "matrix", "energy", "missing_matrix"):
                p, k = copy.deepcopy(pub), copy.deepcopy(key)
                if corrupt == "public":
                    p["hamiltonian"]["terms"][0]["coeff"] += 0.1
                elif corrupt == "matrix":
                    k["clauses"][0]["H_local"]["re"][0][0] += 0.1
                elif corrupt == "energy":
                    k["ground_energy_public"] += 0.1
                else:
                    del k["clauses"][0]["H_local"]
                try:
                    publish.verify_public(p, k)
                except ValueError:
                    continue
                raise AssertionError(f"accepted {corrupt} tampering")
    return "six round trips verified without RNG replay; altered records rejected"


if __name__ == "__main__":
    for f in (t0, t0b, t1, t2, t3, t4, t4b, t4c, t5, t6, t6b, t7, t8, t8b, t9, t10, t11,
              t12, t13, t14):
        f()
    ok = True
    for name, passed, msg in RESULTS:
        print(("PASS " if passed else "FAIL ") + name + (" -- " + msg if msg else ""))
        ok &= passed
    sys.exit(0 if ok else 1)
