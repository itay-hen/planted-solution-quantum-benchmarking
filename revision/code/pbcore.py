"""
pbcore.py -- core routines for the PRA revision numerics.

Instance generation mirrors python/qb_planted_v5.py::generate_general_instance
call-for-call (same RNG streams, same order), so that a (config, seed) pair
produces the same Hamiltonian as the generator used for the submitted
manuscript.  tests_pbcore.py asserts this equality against qb_planted_v5.

Two consequences of the v5 RNG order that the numerics rely on:
  * block partition and block states are drawn before any clause, and each
    clause consumes (subset, basis, weight) in sequence, so the instance with
    K clauses is exactly the K-clause prefix of the instance with K' > K
    clauses and the same seed ("nested" clause sequences);
  * the optional Clifford circuit is drawn after all clauses.

Conventions
-----------
Qubit j <-> tensor axis j; the first character of a Pauli string is qubit 0
and is the most significant factor in kron order (as in v5).
A_k := alpha_k (H^(k) - lambda_0^(k) I) >= 0, so H - E0 = sum_k A_k.
"""
from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the version shipped with this revision, independently of sibling folders.
for _cand in (os.path.abspath(os.path.join(HERE, "..", "vendor")),):
    if os.path.exists(os.path.join(_cand, "qb_planted_v5.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import qb_planted_v5 as v5  # noqa: E402

# ---------------------------------------------------------------------------
# Instance generation (mirrors v5)
# ---------------------------------------------------------------------------


@dataclass
class Clause:
    Sk: List[int]            # block indices (ordered as sampled)
    qubits: List[int]        # local_to_global qubit order (kron order)
    h: np.ndarray            # local clause matrix H^(k) on 2^r dims
    eigs: np.ndarray         # local spectrum as generated
    l0: float                # lambda_0^(k)
    alpha: float
    kernel: np.ndarray       # orthonormal basis (columns) of the local ground space


@dataclass
class Instance:
    n: int
    blocks: List[List[int]]
    block_states: List[List[np.ndarray]]   # [block][a] -> state vector
    clauses: List[Clause]
    seed: int
    scheme: str
    num_planted: int
    clifford: Optional[object] = None      # v5.CliffordCircuit or None

    @property
    def K(self) -> int:
        return len(self.clauses)

    @property
    def M(self) -> int:
        return len(self.blocks)

    @property
    def E0(self) -> float:
        return float(sum(c.alpha * c.l0 for c in self.clauses))

    def prefix(self, K: int) -> "Instance":
        """Keep the first K clauses and the same Clifford frame.

        Independently generating K clauses can draw a different Clifford circuit,
        because the circuit is sampled after the clauses in the legacy generator.
        """
        if not isinstance(K, int) or not 0 <= K <= self.K:
            raise ValueError("prefix length must be an integer between 0 and K")
        return Instance(self.n, self.blocks, self.block_states, self.clauses[:K],
                        self.seed, self.scheme, self.num_planted, self.clifford)


def _local_kernel(h: np.ndarray, l0: float, tol: float = 1e-9) -> np.ndarray:
    w, V = np.linalg.eigh(h)
    return V[:, np.abs(w - l0) < tol]


def generate(n: int, K: int, seed: int, block_size: int = 1, blocks_per_clause: int = 3,
             scheme: str = "bimodal", num_planted: int = 1, alpha_min: float = 0.5,
             alpha_max: float = 1.5, lambda_min: float = -1.0, lambda_max: float = 1.0,
             spectral_gap: float = 0.5, clifford: bool = False, clifford_depth: int = 5,
             clifford_gate_set: str = "full") -> Instance:
    """Replicates v5.generate_general_instance (uniform subset sampling, fixed
    block size) but keeps the local clause matrices."""
    if n <= 0 or block_size <= 0 or n % block_size:
        raise ValueError("n must be positive and divisible by the positive block_size")
    if K < 1 or blocks_per_clause < 1:
        raise ValueError("K and blocks_per_clause must be positive")
    if not 1 <= num_planted <= 2 ** block_size:
        raise ValueError("num_planted must be between 1 and the block dimension")
    if not 0 < alpha_min <= alpha_max:
        raise ValueError("weights require 0 < alpha_min <= alpha_max")
    rng_py, rng_np = random.Random(seed), np.random.default_rng(seed)
    qubits = list(range(n))
    rng_py.shuffle(qubits)
    blocks = [sorted(qubits[i:i + block_size]) for i in range(0, n, block_size)
              if i + block_size <= n]
    M = len(blocks)
    if M < blocks_per_clause:
        raise ValueError("too few blocks")
    block_states: List[List[np.ndarray]] = []
    for bq in blocks:
        bdim = 2 ** len(bq)
        if num_planted == 1:
            block_states.append([v5.sample_haar_state(bdim, rng_np)])
        else:
            U = v5.sample_haar_unitary(bdim, rng_np)
            block_states.append([U[:, a].copy() for a in range(num_planted)])
    clauses: List[Clause] = []
    for _ in range(K):
        Sk = rng_py.sample(range(M), blocks_per_clause)
        l2g: List[int] = []
        for bi in Sk:
            l2g.extend(blocks[bi])
        if num_planted == 1:
            planted = np.array([1.0], dtype=complex)
            for bi in Sk:
                planted = np.kron(planted, block_states[bi][0])
            h, _, eigs = v5.construct_generic_clause(planted, rng_np, scheme=scheme,
                                                     lmin=lambda_min, lmax=lambda_max,
                                                     gap=spectral_gap)
        else:
            plist = []
            for a in range(num_planted):
                st = np.array([1.0], dtype=complex)
                for bi in Sk:
                    st = np.kron(st, block_states[bi][a])
                plist.append(st)
            h, _, eigs = v5.construct_multi_planted_clause(plist, rng_np, scheme=scheme,
                                                           lmin=lambda_min, lmax=lambda_max,
                                                           gap=spectral_gap)
        l0 = float(eigs[0])
        alpha = rng_py.uniform(alpha_min, alpha_max)
        clauses.append(Clause(list(Sk), l2g, h, np.asarray(eigs), l0, alpha,
                              _local_kernel(h, l0)))
    circ = None
    if clifford:
        circ = v5.CliffordCircuit.random_circuit(n, clifford_depth, rng_py, clifford_gate_set)
    return Instance(n, blocks, block_states, clauses, seed, scheme, num_planted, circ)


# ---------------------------------------------------------------------------
# Pauli representation
# ---------------------------------------------------------------------------

_P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
       "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
       "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
_PAULI_CACHE: Dict[int, List[Tuple[str, np.ndarray]]] = {}


def _local_paulis(r: int):
    if r not in _PAULI_CACHE:
        out = []
        for labels in product("IXYZ", repeat=r):
            P = np.array([[1.0]], dtype=complex)
            for c in labels:
                P = np.kron(P, _P1[c])
            out.append(("".join(labels), P))
        _PAULI_CACHE[r] = out
    return _PAULI_CACHE[r]


def pauli_sum(inst: Instance, tol: float = 1e-14, apply_clifford: bool = True) -> Dict[str, float]:
    """Aggregated Pauli expansion of H = sum_k alpha_k H^(k) (optionally U H U^dag)."""
    acc: Dict[str, float] = {}
    for c in inst.clauses:
        r = len(c.qubits)
        d = 2 ** r
        for lab, P in _local_paulis(r):
            coef = np.trace(c.h @ P).real / d
            if abs(coef) <= tol:
                continue
            g = ["I"] * inst.n
            for i, q in enumerate(c.qubits):
                g[q] = lab[i]
            v5.accumulate_pauli_term(acc, "".join(g), c.alpha * coef, tol)
    if apply_clifford and inst.clifford is not None:
        out: Dict[str, float] = {}
        for p, cf in acc.items():
            np_, nc = inst.clifford.conjugate_pauli(p, cf)
            v5.accumulate_pauli_term(out, np_, nc, tol)
        acc = out
    return acc


def pauli_stats(terms: Dict[str, float]) -> Dict[str, float]:
    ws = [sum(ch != "I" for ch in p) for p in terms]
    nonid = {p: c for p, c in terms.items() if set(p) != {"I"}}
    return {"L": len(terms), "L_nonid": len(nonid),
            "one_norm": float(sum(abs(c) for c in terms.values())),
            "one_norm_nonid": float(sum(abs(c) for c in nonid.values())),
            "max_weight": int(max(ws) if ws else 0),
            "mean_weight": float(np.mean(ws) if ws else 0.0)}


def pauli_dense_fast(terms, n: int) -> np.ndarray:
    """Dense matrix of sum_l c_l P_l via bit masks (qubit 0 = most significant bit).
    `terms` is a dict or an iterable of (pauli_string, coeff)."""
    D = 2 ** n
    x = np.arange(D)
    H = np.zeros((D, D), dtype=complex)
    items = terms.items() if isinstance(terms, dict) else terms
    for p, c in items:
        xm = zm = 0
        ny = 0
        for q, ch in enumerate(p):
            bit = 1 << (n - 1 - q)
            if ch in "XY":
                xm |= bit
            if ch in "ZY":
                zm |= bit
            if ch == "Y":
                ny += 1
        # P|x> = i^{ny} (-1)^{popcount(x & zm)} |x ^ xm>
        par = np.array([bin(v).count("1") & 1 for v in (x & zm)]) if zm else np.zeros(D, dtype=int)
        phase = (1j) ** ny * (1 - 2 * par)
        H[x ^ xm, x] += c * phase
    return H


def pauli_dense(terms: Dict[str, float], n: int) -> np.ndarray:
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for p, c in terms.items():
        P = np.array([[1.0]], dtype=complex)
        for ch in p:
            P = np.kron(P, _P1[ch])
        H += c * P
    return H


# ---------------------------------------------------------------------------
# Matrix-free application of B = H - E0 = sum_k A_k
# ---------------------------------------------------------------------------


def apply_local(psi: np.ndarray, qubits: Sequence[int], mat: np.ndarray, n: int) -> np.ndarray:
    """Apply mat (acting on `qubits` in kron order) to a state tensor of shape (2,)*n."""
    r = len(qubits)
    t = np.moveaxis(psi, list(qubits), list(range(r)))
    sh = t.shape
    t = (mat @ t.reshape(2 ** r, -1)).reshape(sh)
    return np.moveaxis(t, list(range(r)), list(qubits))


def shifted_locals(inst: Instance) -> List[Tuple[List[int], np.ndarray]]:
    out = []
    for c in inst.clauses:
        d = c.h.shape[0]
        out.append((c.qubits, c.alpha * (c.h - c.l0 * np.eye(d))))
    return out


def apply_B(psi_flat: np.ndarray, locs, n: int) -> np.ndarray:
    psi = psi_flat.reshape((2,) * n)
    out = np.zeros_like(psi)
    for q, A in locs:
        out += apply_local(psi, q, A, n)
    return out.reshape(-1)


def dense_B(inst: Instance) -> np.ndarray:
    n = inst.n
    D = 2 ** n
    locs = shifted_locals(inst)
    B = np.zeros((D, D), dtype=complex)
    I = np.eye(D, dtype=complex)
    for j in range(D):
        B[:, j] = apply_B(I[:, j], locs, n)
    return B


def planted_vector(inst: Instance, a: int = 0) -> np.ndarray:
    """|Psi_star^(a)> in the computational basis (qubit 0 most significant)."""
    psi = np.ones((1,) * 0, dtype=complex)
    psi = np.array(1.0, dtype=complex)
    order: List[int] = []
    for bq, st in zip(inst.blocks, inst.block_states):
        psi = np.multiply.outer(psi, st[a].reshape((2,) * len(bq)))
        order.extend(bq)
    # axes currently in `order`; permute to 0..n-1
    psi = np.transpose(psi, np.argsort(order))
    return psi.reshape(-1)


# ---------------------------------------------------------------------------
# Structural analysis (Theorems in Sec. IV / V of the revision)
# ---------------------------------------------------------------------------


def hypergraph_components(M: int, supports: Sequence[Sequence[int]]):
    """Union-find over covered blocks.  Returns (uncovered, components) where each
    component is (sorted block list, list of clause indices)."""
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    covered = set()
    for S in supports:
        covered.update(S)
        for b in S[1:]:
            ra, rb = find(S[0]), find(b)
            if ra != rb:
                parent[ra] = rb
    comps: Dict[int, List[int]] = {}
    for b in covered:
        comps.setdefault(find(b), []).append(b)
    cl: Dict[int, List[int]] = {}
    for k, S in enumerate(supports):
        cl.setdefault(find(S[0]), []).append(k)
    comp_list = [(sorted(v), cl[r]) for r, v in comps.items()]
    uncovered = sorted(set(range(M)) - covered)
    return uncovered, comp_list


def structural_dimension(inst: Instance) -> Dict[str, object]:
    """Exact ground-space dimension from the support hypergraph.

    exact=True when the prediction is a theorem for this instance:
      * local ground spaces exactly equal the span of planted restrictions
        (schemes golf_course / uniform / linear): dim = m^c * prod_{uncovered} 2^{|A_i|};
      * bimodal (local kernel strictly larger): the factorization over components is
        exact, uncovered blocks give 2^{|A_i|}, and a component consisting of clauses
        on a single identical support contributes dim of the intersection of those
        local kernels (computed exactly); larger components are *predicted* to
        contribute m (verified numerically, not proven) -> exact=False if any exist.
    """
    supports = [c.Sk for c in inst.clauses]
    unc, comps = hypergraph_components(inst.M, supports)
    m = inst.num_planted
    dim = 1
    for b in unc:
        dim *= 2 ** len(inst.blocks[b])
    exact_local = all(c.kernel.shape[1] == m for c in inst.clauses)
    exact = True
    single_support_comps = 0
    for blocks_c, cls in comps:
        sup = {tuple(sorted(inst.clauses[k].Sk)) for k in cls}
        if exact_local:
            dim *= m
        elif len(sup) == 1:
            # all clauses of the component act on the same block set: intersect kernels
            single_support_comps += 1
            ref = inst.clauses[cls[0]]
            perm_ref = ref.qubits
            Ps = []
            for k in cls:
                c = inst.clauses[k]
                Kmat = c.kernel
                P = Kmat @ Kmat.conj().T
                if c.qubits != perm_ref:  # reorder to reference qubit order
                    r = len(c.qubits)
                    perm = [c.qubits.index(q) for q in perm_ref]
                    P = P.reshape((2,) * (2 * r)).transpose(perm + [r + p for p in perm]).reshape(2 ** r, 2 ** r)
                Ps.append(np.eye(P.shape[0]) - P)
            S = sum(Ps)
            w = np.linalg.eigvalsh(S)
            dim *= int(np.sum(w < 1e-9))
        else:
            dim *= m
            exact = False
    return {"dim": int(dim), "uncovered": unc, "n_components": len(comps),
            "single_support_components": single_support_comps, "exact": exact,
            "exact_local": exact_local}


def ground_projector_factors(inst: Instance):
    """Factors of the projector onto the *predicted* ground space, as a list of
    (qubits, matrix) acting on disjoint qubit sets; qubits of uncovered blocks carry
    identity and are omitted.  Single-planted only."""
    assert inst.num_planted == 1
    supports = [c.Sk for c in inst.clauses]
    unc, comps = hypergraph_components(inst.M, supports)
    exact_local = all(c.kernel.shape[1] == 1 for c in inst.clauses)
    facs = []
    for blocks_c, cls in comps:
        sup = {tuple(sorted(inst.clauses[k].Sk)) for k in cls}
        if (not exact_local) and len(sup) == 1:
            ref = inst.clauses[cls[0]]
            # projector onto intersection of kernels on ref.qubits
            r = len(ref.qubits)
            S = np.zeros((2 ** r, 2 ** r), dtype=complex)
            for k in cls:
                c = inst.clauses[k]
                P = c.kernel @ c.kernel.conj().T
                if c.qubits != ref.qubits:
                    perm = [c.qubits.index(q) for q in ref.qubits]
                    P = P.reshape((2,) * (2 * r)).transpose(perm + [r + p for p in perm]).reshape(2 ** r, 2 ** r)
                S += np.eye(2 ** r) - P
            w, V = np.linalg.eigh(S)
            Kc = V[:, w < 1e-9]
            facs.append((list(ref.qubits), Kc @ Kc.conj().T))
        else:
            for b in blocks_c:
                v = inst.block_states[b][0]
                facs.append((list(inst.blocks[b]), np.outer(v, v.conj())))
    return facs


def apply_projector(psi_flat: np.ndarray, facs, n: int) -> np.ndarray:
    psi = psi_flat.reshape((2,) * n)
    for q, P in facs:
        psi = apply_local(psi, q, P, n)
    return psi.reshape(-1)


def excitation_gap(inst: Instance, tol: float = 1e-10, maxiter: int = 20000,
                   return_info: bool = False):
    """Gap above the predicted ground manifold, Delta_exc = E_g - E_0.

    Computes lambda_min(B + sigma * P_pred) with B = H - E0 >= 0 applied matrix-free and
    P_pred the projector onto the structurally predicted ground space.  If the
    prediction is correct the result is Delta_exc (provided sigma > Delta_exc);
    if the true kernel is larger than predicted the result is ~0, which is flagged.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    n = inst.n
    D = 2 ** n
    locs = shifted_locals(inst)
    facs = ground_projector_factors(inst)
    sigma = float(sum(c.alpha * (c.eigs.max() - c.l0) for c in inst.clauses))  # >= ||B||

    def mv(x):
        x = np.asarray(x).reshape(-1)
        return apply_B(x, locs, n) + sigma * apply_projector(x, facs, n)

    op = LinearOperator((D, D), matvec=mv, dtype=complex)
    rng = np.random.default_rng(inst.seed + 7919)
    v0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    # smallest eigenvalue of op == sigma - largest eigenvalue of (sigma I - op); use SA directly
    vals = eigsh(op, k=2, which="SA", tol=tol, maxiter=maxiter, v0=v0,
                 return_eigenvectors=False)
    vals = np.sort(vals)
    if return_info:
        return float(vals[0]), {"sigma": sigma, "second": float(vals[1])}
    return float(vals[0])


# ---------------------------------------------------------------------------
# Coverage probabilities (Theorem / Eq. in Sec. V)
# ---------------------------------------------------------------------------


def expected_uncovered(M: int, q: int, K: int) -> float:
    return M * (1.0 - q / M) ** K


def prob_full_coverage(M: int, q: int, K: int) -> float:
    """Inclusion-exclusion: sum_j (-1)^j C(M,j) [C(M-j,q)/C(M,q)]^K.
    Evaluated in exact rational arithmetic (the alternating sum cancels badly in floats)."""
    from fractions import Fraction
    tot = Fraction(0)
    den = math.comb(M, q)
    for j in range(M + 1):
        num = math.comb(M - j, q) if M - j >= q else 0
        if num == 0:
            break
        tot += (-1) ** j * math.comb(M, j) * Fraction(num, den) ** K
    return float(tot)


def coverage_scale_K(M: int, q: int, c: float) -> int:
    return int(math.ceil((M / q) * (math.log(M) + c)))
