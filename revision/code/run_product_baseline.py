"""
Generator-aware classical baseline: variational product-state minimisation.

A solver that knows the generator (but not the key) knows that for block size 1 and no
Clifford scrambling the target is a product state.  It can therefore minimise
    E(r_1..r_n) = sum_P c_P prod_{j in supp P} r_j[P_j],   r_j on the Bloch sphere,
directly from the public Pauli list (cost O(L) per evaluation, independent of 2^n).
We run this with R random restarts (L-BFGS-B, analytic gradient) on
  (i) unscrambled, block size 1; (ii) Clifford-scrambled (depth-5 brickwork), block size 1;
  (iii) unscrambled, block size 2 (planted state entangled within blocks), n <= 20 only.
Restart initialisations use default_rng([20260923, n, idx, block_size, clifford]) so that the
run is resumable and each instance's restarts are reproducible independently of run order.
Success := min_restarts E - E0 <= 1e-8 * sum_k alpha_k.
Output: ../data/product_baseline.jsonl
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbcore as pb  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "product_baseline.jsonl")
PIDX = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def compile_terms(terms, n):
    items = [(p, c) for p, c in terms.items()]
    w = max(1, max(sum(ch != "I" for ch in p) for p, _ in items))
    T = len(items)
    q = np.full((T, w), n, dtype=np.int64)       # dummy qubit n (all ones)
    a = np.zeros((T, w), dtype=np.int64)
    cs = np.zeros(T)
    for t, (p, c) in enumerate(items):
        cs[t] = c
        j = 0
        for qi, ch in enumerate(p):
            if ch != "I":
                q[t, j], a[t, j] = qi, PIDX[ch]
                j += 1
    return q, a, cs


def energy_grad(x, q, a, cs, n):
    th, ph = x[:n], x[n:]
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    R = np.ones((n + 1, 4))
    R[:n, 1], R[:n, 2], R[:n, 3] = st * cp, st * sp, ct
    V = R[q, a]                                   # T x w
    T, w = V.shape
    pre = np.ones((T, w + 1))
    suf = np.ones((T, w + 1))
    for j in range(w):
        pre[:, j + 1] = pre[:, j] * V[:, j]
        suf[:, w - 1 - j] = suf[:, w - j] * V[:, w - 1 - j]
    E = float(cs @ pre[:, w])
    loo = pre[:, :w] * suf[:, 1:]                 # leave-one-out products
    G = np.zeros((n + 1, 4))
    np.add.at(G, (q, a), cs[:, None] * loo)
    dth = G[:n, 1] * ct * cp + G[:n, 2] * ct * sp - G[:n, 3] * st
    dph = -G[:n, 1] * st * sp + G[:n, 2] * st * cp
    return E, np.concatenate([dth, dph])


def product_min(terms, n, restarts, rng):
    q, a, cs = compile_terms(terms, n)
    best = np.inf
    energies = []
    for _ in range(restarts):
        x0 = np.concatenate([np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)])
        res = minimize(energy_grad, x0, args=(q, a, cs, n), jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000, "gtol": 1e-12, "ftol": 1e-15})
        energies.append(float(res.fun))
        best = min(best, res.fun)
    return float(best), energies


def main():
    ns = [int(v) for v in os.environ.get("PB_NS", "10,20,50,100").split(",")]
    per = int(os.environ.get("PB_PER", "20"))
    restarts = int(os.environ.get("PB_RESTARTS", "20"))

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["n"], r.get("block_size", 1), r["clifford"], r["idx"]))
    with open(OUT, "a") as f:
        for n in ns:
            for idx in range(per):
                seed = 777000 + 1000 * n + idx
                # arms: (block_size, clifford); the |A_i|=2 arm has L <= K 4^6 terms and is
                # restricted to n <= 20 for cost reasons
                for bs, cl in ((1, False), (1, True), (2, False)):
                    if (n, bs, cl, idx) in done or (bs == 2 and n > 20):
                        continue
                    # the scrambled arm never succeeds and is far more expensive per
                    # optimisation; it is run with fewer instances and restarts at large n
                    n_rest = restarts
                    if cl and n > 20:
                        if idx >= 10:
                            continue
                        n_rest = 5
                    rng = np.random.default_rng([20260923, n, idx, bs, int(cl)])
                    M = n // bs
                    K = pb.coverage_scale_K(M, 3, 2.0)
                    inst = pb.generate(n, K, seed, block_size=bs, scheme="bimodal", clifford=cl,
                                       clifford_depth=5)
                    terms = pb.pauli_sum(inst)
                    st = pb.pauli_stats(terms)
                    t0 = time.time()
                    best, es = product_min(terms, n, n_rest, rng)
                    scale = sum(c.alpha for c in inst.clauses)
                    tol = 1e-8 * scale
                    row = {"n": n, "block_size": bs, "M": M, "K": K, "idx": idx, "seed": seed, "clifford": cl,
                           "E0": inst.E0, "E_prod_best": best, "excess": best - inst.E0,
                           "rel_excess": (best - inst.E0) / scale,
                           "success": bool(best - inst.E0 <= tol),
                           "restart_success_frac": float(np.mean([e - inst.E0 <= tol for e in es])),
                           "L": st["L"], "max_weight": st["max_weight"], "restarts": n_rest,
                           "t": time.time() - t0}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            print("done n =", n, flush=True)


if __name__ == "__main__":
    main()
