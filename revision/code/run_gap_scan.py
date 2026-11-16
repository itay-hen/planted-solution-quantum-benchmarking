"""
Coverage-scale gap scan (new numerics for the PRA revision).

For each (scheme, n, instance) a single nested clause sequence of length
K_max = K_c(c=C_MAX) is generated (same seed => prefixes are the smaller-K instances).
For every c in C_VALUES the prefix K_c = ceil((M/q)(ln M + c)) is analysed:
  * structural ground-space dimension and uncovered blocks (exact, combinatorial);
  * Delta_exc = gap above the (structurally predicted) ground manifold, by matrix-free
    Lanczos on B + sigma P_pred (see pbcore.excitation_gap);
  * Pauli-representation statistics L, ||c||_1, max weight.
Output: one JSON object per line in ../data/gap_scan.jsonl (resumable).

Usage: python run_gap_scan.py [--procs 15] [--smoke]
"""
import argparse
import json
import os
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "data", "gap_scan.jsonl")

Q = 3
C_VALUES = [0, 1, 2, 3, 4]
C_MAX = max(C_VALUES)
BASE_SEED = 20260922
PLAN = {  # n -> instances per scheme
    6: 100, 8: 100, 10: 100, 12: 100, 14: 100, 16: 100, 18: 60, 20: 40,
}
SCHEMES = ["bimodal", "golf_course"]


def task(args):
    scheme, n, idx = args
    import numpy as np  # noqa: F401
    import pbcore as pb
    seed = BASE_SEED + 1000 * n + idx + (0 if scheme == "bimodal" else 500)
    M = n  # block size 1
    Kmax = pb.coverage_scale_K(M, Q, C_MAX)
    full = pb.generate(n, Kmax, seed, scheme=scheme, blocks_per_clause=Q)
    rows = []
    for c in C_VALUES:
        K = pb.coverage_scale_K(M, Q, c)
        inst = full.prefix(K)
        sd = pb.structural_dimension(inst)
        t0 = time.time()
        gap, info = pb.excitation_gap(inst, return_info=True)
        dt = time.time() - t0
        st = pb.pauli_stats(pb.pauli_sum(inst))
        rows.append({
            "scheme": scheme, "n": n, "M": M, "q": Q, "idx": idx, "seed": seed, "c": c, "K": K,
            "E0": inst.E0, "struct_dim": sd["dim"], "n_uncovered": len(sd["uncovered"]),
            "n_components": sd["n_components"], "struct_exact": sd["exact"],
            "single_support_components": sd["single_support_components"],
            "gap_exc": gap, "second": info["second"], "sigma": info["sigma"],
            "flag_kernel_larger_than_predicted": bool(gap < 1e-8 * info["sigma"]),
            "alpha_sum": float(sum(cl.alpha for cl in inst.clauses)),
            "L": st["L"], "one_norm": st["one_norm"], "one_norm_nonid": st["one_norm_nonid"],
            "max_weight": st["max_weight"], "t_eig": dt,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["scheme"], r["n"], r["idx"]))
    tasks = []
    plan = {6: 3, 8: 3} if a.smoke else PLAN
    # large n first so the long jobs start early
    for n in sorted(plan, reverse=True):
        for scheme in SCHEMES:
            for idx in range(plan[n]):
                if (scheme, n, idx) not in done:
                    tasks.append((scheme, n, idx))
    print(f"{len(tasks)} tasks ({len(done)} already done)", flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.procs) as pool, open(OUT, "a") as f:
        for i, rows in enumerate(pool.imap_unordered(task, tasks)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
