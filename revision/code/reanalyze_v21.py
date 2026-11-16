"""
Structural re-analysis of the numerical data underlying the submitted manuscript (v21).

Inputs (unchanged originals):
  python/benchmarks_bimodal/<stem>_key.json                     generator keys (qb_planted_v5)
  python/benchmarks_bimodal/npz results/npz2/output_<stem>.npz  full dense spectra
Output:
  ../data/v21_structural.csv   one row per instance
  ../data/v21_structural_summary.csv   per (n,K)
Prints tolerance-stability table for the numerical degeneracy count.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbcore as pb  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BB = os.path.join(ROOT, "qb gs before revision", "python", "benchmarks_bimodal")
NPZ = os.path.join(BB, "npz results", "npz2")
OUT = os.path.join(HERE, "..", "data")
EPS = np.finfo(float).eps
TOL_FACTORS = [1e2, 1e3, 1e4, 1e5, 1e6]   # tol = f * eps * ||H||_bound


COMPLETED = os.path.join(HERE, "..", "data", "v21_completed_spectra")


def spectrum_for(stem, inst):
    """Original dense spectrum if the submitted run produced one; otherwise compute it
    now (dense ED of the same instance) and cache it, labelled as newly computed."""
    f = os.path.join(NPZ, f"output_{stem}.npz")
    if os.path.exists(f):
        return np.load(f)["eigenvalues"], "original"
    os.makedirs(COMPLETED, exist_ok=True)
    g = os.path.join(COMPLETED, f"{stem}.npy")
    if not os.path.exists(g):
        # diagonalise the *serialized* public instance (Pauli list in the instance JSON),
        # exactly as the original spectrum_analysis.py did
        inst_json = json.load(open(os.path.join(BB, stem + "_instance.json")))
        terms = [(t["pauli"], t["coeff"]) for t in inst_json["hamiltonian"]["terms"]]
        ev = np.linalg.eigvalsh(pb.pauli_dense_fast(terms, inst_json["n_qubits"]))
        np.save(g, ev)
    return np.load(g), "completed_in_revision"


def main():
    global BB, NPZ
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-data", default=os.environ.get("PB_ORIGINAL_DATA", BB),
                    help="directory containing n*_K*_idx*_key.json and original instance files")
    args = ap.parse_args()
    BB = os.path.abspath(args.original_data)
    NPZ = os.path.join(BB, "npz results", "npz2")
    keys = sorted(glob.glob(os.path.join(BB, "n*_K*_idx*_key.json")))
    if not keys:
        ap.error("original archive not found; pass --original-data PATH to benchmarks_bimodal. "
                 "To regenerate tables from bundled processed data, run make_figures.py instead.")
    rows = []
    for kf in keys:
        stem = os.path.basename(kf)[:-len("_key.json")]
        key = json.load(open(kf))
        p = key["params"]
        if p["n_qubits"] > 12:  # submitted manuscript analysed n = 6..12 only
            continue
        n, K, seed = p["n_qubits"], p["num_clauses"], p["seed"]
        # regenerate the instance from its seed (T0 proves equality with v5) to get local kernels
        inst = pb.generate(n, K, seed, block_size=p["block_size"],
                           blocks_per_clause=p["blocks_per_clause"], scheme=p["eigenvalue_scheme"])
        assert [c.Sk for c in inst.clauses] == [c["Sk"] for c in key["clauses"]]
        assert abs(inst.E0 - key["ground_energy"]) < 1e-12
        sd = pb.structural_dimension(inst)
        ev, source = spectrum_for(stem, inst)
        hnorm = float(sum(c.alpha * (np.max(np.abs(c.eigs))) for c in inst.clauses))
        gs = {fac: int(np.sum(ev - ev[0] < fac * EPS * hnorm)) for fac in TOL_FACTORS}
        g_abs = int(np.sum(ev - ev[0] < 1e-10))
        d = sd["dim"]
        rows.append({
            "stem": stem, "source": source, "n": n, "K": K, "seed": seed, "struct_dim": d,
            "n_uncovered": len(sd["uncovered"]), "single_support_components": sd["single_support_components"],
            "n_components": sd["n_components"], "struct_exact": sd["exact"],
            "g_abs1e-10": g_abs, **{f"g_tol{fac:.0e}": gs[fac] for fac in TOL_FACTORS},
            "E0_ed_minus_key": float(ev[0] - key["ground_energy"]), "H_bound": hnorm,
            "delta1_raw": float(ev[1] - ev[0]), "delta_exc": float(ev[d] - ev[0]) if d < len(ev) else float("nan"),
        })
    if not rows:
        ap.error("no original instances with n <= 12 were found; no outputs written")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "v21_structural.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # summary
    agg = defaultdict(list)
    for r in rows:
        agg[(r["n"], r["K"])].append(r)
    with open(os.path.join(OUT, "v21_structural_summary.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["n", "K", "N", "struct_degenerate", "uncovered_any", "isolated_clause_any",
                    "E_Nunc_theory", "mean_Nunc_observed", "P_fullcov_theory", "frac_fullcov_observed"])
        for (n, K), rs in sorted(agg.items()):
            N = len(rs)
            w.writerow([n, K, N, sum(r["struct_dim"] > 1 for r in rs),
                        sum(r["n_uncovered"] > 0 for r in rs),
                        sum(r["single_support_components"] > 0 for r in rs),
                        f"{pb.expected_uncovered(n, 3, K):.3f}",
                        f"{np.mean([r['n_uncovered'] for r in rs]):.3f}",
                        f"{pb.prob_full_coverage(n, 3, K):.3f}",
                        f"{np.mean([r['n_uncovered'] == 0 for r in rs]):.3f}"])
    N = len(rows)
    print(f"instances: {N}")
    print("structural dim > 1:", sum(r["struct_dim"] > 1 for r in rows))
    print("all structural predictions exact (theorem, no unproven components)?",
          all(r["struct_exact"] for r in rows), "| non-exact count:", sum(not r["struct_exact"] for r in rows))
    print("max |E0_ED - E0_key| =", max(abs(r["E0_ed_minus_key"]) for r in rows))
    for fac in TOL_FACTORS:
        agree = sum(r[f"g_tol{fac:.0e}"] == r["struct_dim"] for r in rows)
        print(f"tol = {fac:.0e} eps ||H||: numerical multiplicity == structural in {agree}/{N}")
    agree = sum(r["g_abs1e-10"] == r["struct_dim"] for r in rows)
    print(f"absolute 1e-10 rule (submitted): == structural in {agree}/{N}")
    unique = [r for r in rows if r["struct_dim"] == 1]
    print("min Delta_1 among numerically verified unique:", min(r["delta1_raw"] for r in unique))
    print("spectra: original", sum(r["source"] == "original" for r in rows),
          "| completed in revision", sum(r["source"] != "original" for r in rows))


if __name__ == "__main__":
    main()
