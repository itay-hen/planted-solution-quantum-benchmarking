"""
Why the uniqueness of the half-rank ground space is not proved here (Sec. IV A remark).

Lemma 1 (almost-sure constancy) reduces the claim g_Gamma = 1 to exhibiting ONE choice of
clause ground spaces whose intersection is one-dimensional.  The natural explicit family is
the set of *coordinate* subspaces: pick, for each clause, a set T_k of product basis vectors
(in the basis adapted to the planted block states) with |T_k| = d_k/2 and 0...0 in T_k.  For
such a choice the intersection is spanned by the product basis vectors whose label restricts
into T_k on every clause, so its dimension is the number of satisfying assignments of a
constraint satisfaction problem.

`min_coordinate_witness` enumerates that family exhaustively for a given hypergraph.  For the
component S1 = {0,1,2}, S2 = {2,3,4} the minimum is 4, never 1: coordinate witnesses cannot
certify uniqueness, and any witness must use clause ground spaces entangled across blocks.
`haar_dimension` shows that random (entangled) draws do achieve 1 for the same hypergraph.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb  # noqa: E402


def min_coordinate_witness(M: int, supports, a: int = 1):
    """Smallest intersection dimension achievable by coordinate ground spaces."""
    dim_block = 2 ** a
    labels = list(range(dim_block))
    best = None
    per_clause = []
    for S in supports:
        tups = list(itertools.product(labels, repeat=len(S)))
        zero = tuple([0] * len(S))
        half = len(tups) // 2
        per_clause.append([T for T in itertools.combinations(tups, half) if zero in T])
    for choice in itertools.product(*per_clause):
        sets = [set(T) for T in choice]
        cnt = 0
        for b in itertools.product(labels, repeat=M):
            if all(tuple(b[i] for i in S) in s for S, s in zip(supports, sets)):
                cnt += 1
        if best is None or cnt < best:
            best = cnt
            if best == 1:
                break
    return best


def _embed(P, S, n):
    D = 2 ** n
    out = np.zeros((D, D), dtype=complex)
    I = np.eye(D, dtype=complex)
    for j in range(D):
        out[:, j] = pb.apply_local(I[:, j].reshape((2,) * n), S, P, n).reshape(-1)
    return out


def haar_dimension(M: int, supports, trials: int = 20, seed: int = 0):
    """Kernel dimension for Haar half-rank clause ground spaces containing the planted state."""
    dims = set()
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        n = M
        B = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for S in supports:
            d = 2 ** len(S)
            psi = np.zeros(d, complex)
            psi[0] = 1.0
            Z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(Z)
            cols = []
            for j in range(d):
                v = Q[:, j] - np.vdot(psi, Q[:, j]) * psi
                for c in cols:
                    v = v - np.vdot(c, v) * c
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    cols.append(v / nv)
            excited = cols[d // 2 - 1:]                      # kernel = psi + (d/2 - 1) vectors
            P = sum(np.outer(v, v.conj()) for v in excited)  # A_k propto projector onto excited
            B += _embed(P, S, n)
        dims.add(int(np.sum(np.linalg.eigvalsh(B) < 1e-9)))
    return dims


if __name__ == "__main__":
    sup = [[0, 1, 2], [2, 3, 4]]
    print("hypergraph S1={0,1,2}, S2={2,3,4} (connected, two distinct supports)")
    print("  min over all coordinate witnesses :", min_coordinate_witness(5, sup))
    print("  Haar (entangled) draws            :", sorted(haar_dimension(5, sup)))
    print("=> coordinate witnesses cannot certify g_Gamma = 1; random entangled draws do.")
