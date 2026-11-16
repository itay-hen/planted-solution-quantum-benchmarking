"""
Coverage statistics for the three clause-count regimes (Sec. IV B of the revision).

For q = 3 single-qubit blocks (M = n) and clause supports drawn as in the generator
(each clause: q distinct blocks uniformly at random; clauses i.i.d.):
  * exact P(N_unc = 0) (inclusion-exclusion, exact rationals) and E[N_unc] = M (1-q/M)^K;
  * Monte Carlo of N_unc and of a graph proxy: full coverage and at least two
    distinct supports per connected component. This proxy is not verified uniqueness.
    Repeated clauses on a single support can already have a one-dimensional kernel;
    this graph-only calculation does not evaluate their local intersections.
Output: ../data/coverage.json
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbcore as pb  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "coverage.json")
Q = 3
MS = [6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 200]
REGIMES = {
    "fixed_K_20": lambda M: 20,
    "fixed_K_60": lambda M: 60,
    "density_1": lambda M: M,
    "density_2": lambda M: 2 * M,
    "coverage_c0": lambda M: pb.coverage_scale_K(M, Q, 0.0),
    "coverage_c2": lambda M: pb.coverage_scale_K(M, Q, 2.0),
}
TRIALS = 20000


def mc(M, K, rng):
    unc_sum = 0
    full = 0
    uniq_half = 0
    for _ in range(TRIALS):
        sup = [tuple(sorted(rng.sample(range(M), Q))) for _ in range(K)]
        cov = set(b for s in sup for b in s)
        u = M - len(cov)
        unc_sum += u
        if u == 0:
            full += 1
            _, comps = pb.hypergraph_components(M, sup)
            if all(len({sup[k] for k in cls}) > 1 for _, cls in comps):
                uniq_half += 1
    return unc_sum / TRIALS, full / TRIALS, uniq_half / TRIALS


def main():
    rng = random.Random(4242)
    res = {"q": Q, "trials": TRIALS, "Ms": MS, "regimes": {}}
    for name, fK in REGIMES.items():
        rows = []
        for M in MS:
            K = fK(M)
            if K < 1:
                continue
            e_mc, p_mc, u_mc = mc(M, K, rng)
            rows.append({"M": M, "K": K, "E_unc_theory": pb.expected_uncovered(M, Q, K),
                         "P_full_theory": pb.prob_full_coverage(M, Q, K),
                         "E_unc_mc": e_mc, "P_full_mc": p_mc, "P_graph_proxy_mc": u_mc})
        res["regimes"][name] = rows
        print(name, [(r["M"], r["K"], round(r["P_full_theory"], 4), r["P_full_mc"]) for r in rows[:4]], flush=True)
    res["poisson_limit"] = {"c0": math.exp(-1.0), "c2": math.exp(-math.exp(-2.0))}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
