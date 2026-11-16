"""Writes ../REPRODUCIBILITY.md: environment, versions, input hashes, seeds, solver
settings and the exact commands that regenerate every figure and table."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy
import scipy
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(REV, ".."))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def count_lines(p):
    try:
        return sum(1 for _ in open(p))
    except FileNotFoundError:
        return 0


def main():
    files = {"vendor/qb_planted_v5.py": os.path.join(REV, "vendor", "qb_planted_v5.py"),
             "requirements.txt": os.path.join(REV, "requirements.txt")}
    files.update({"code/" + name: os.path.join(HERE, name)
                  for name in sorted(os.listdir(HERE)) if name.endswith(".py")})
    data = {
        "data/gap_scan.jsonl": count_lines(os.path.join(REV, "data", "gap_scan.jsonl")),
        "data/product_baseline.jsonl": count_lines(os.path.join(REV, "data", "product_baseline.jsonl")),
        "data/v21_structural.csv": count_lines(os.path.join(REV, "data", "v21_structural.csv")) - 1,
    }
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    L = []
    L.append("# Reproducibility report — qb_gs_v22 revision\n")
    L.append(f"Generated {datetime.now().isoformat(timespec='seconds')} by `code/make_repro_report.py`.\n")
    L.append("This records the current validation environment. Original numerical results were generated "
             "with Python 3.13.4, NumPy 2.2.6, SciPy 1.16.3 and Matplotlib 3.10.3 on an Apple M3 Max; "
             "the full spectral and optimization campaigns have not been rerun by this report generator.\n")
    L.append("## Environment\n")
    L.append(f"- OS: {platform.platform()}")
    L.append(f"- CPU: {cpu}, {os.cpu_count()} logical cores")
    L.append(f"- Python {sys.version.split()[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, "
             f"Matplotlib {matplotlib.__version__}")
    L.append("- Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` when running numerical scripts.")
    L.append("- LaTeX: TeX Live (pdflatex, bibtex, latexdiff), REVTeX 4.2\n")
    L.append("## Code versions (SHA-256, first 16 hex digits)\n")
    L.append("The project folder is not under version control; file hashes identify the exact code.\n")
    for k, p in files.items():
        L.append(f"- `{k}`: `{sha(p)}`")
    L.append("\n## Random seeds\n")
    L.append("- Instance seed `s` drives two streams exactly as in `qb_planted_v5.generate_general_instance`: "
             "`random.Random(s)` (partition, supports, weights, Clifford) and `numpy.random.default_rng(s)` "
             "(block states, Haar completions).")
    L.append("- Original (v21) data: seeds in the separate original archive's `*_key.json` (`params.seed`); "
             "pass its directory via `reanalyze_v21.py --original-data PATH`.")
    L.append("- Coverage-scale scan: `20260922 + 1000 n + i` (half-rank), `+500` (projector), "
             "i = instance index; Lanczos start vector from `default_rng(s + 7919)`.")
    L.append("- Coverage Monte Carlo: `random.Random(4242)`; 20000 trials per point.")
    L.append("- Product baseline: instance seeds `777000 + 1000 n + i`; restarts from `default_rng([20260923, n, i, block_size, clifford])`; the |A_i|=2 arm is run for n <= 20 only.")
    L.append("\n## Solver settings\n")
    L.append("- Dense spectra (v21 data, n ≤ 12): `numpy.linalg.eigvalsh` (LAPACK zheevd).")
    L.append("- Gaps: `scipy.sparse.linalg.eigsh(k=2, which='SA', tol=1e-10, maxiter=20000)` on the "
             "matrix-free operator `H - E0 + sigma P_pred`; sigma = sum_k alpha_k (max lambda - lambda_0).")
    L.append("- Structural kernel tolerance for local clause kernels: |lambda - lambda_0| < 1e-9 "
             "(local spectra are exactly -1/+1/0 by construction).")
    L.append("- Numerical multiplicity cross-check: tau = f * eps * sum_k alpha_k ||H^(k)||, f = 1e2..1e6.")
    L.append("- Product baseline: L-BFGS-B, maxiter 5000, gtol 1e-12, ftol 1e-15, 20 restarts; "
             "success iff E - E0 <= 1e-8 sum_k alpha_k.")
    L.append("- Failed / non-converged instances: none recorded (eigsh raises on non-convergence; "
             "no exceptions occurred). See the validation report.")
    L.append("\n## Data files\n")
    for k, v in data.items():
        L.append(f"- `{k}`: {v} records")
    L.append("- `data/v21_completed_spectra/*.npy`: spectra computed in the revision for the v21 instances "
             "whose runs had not completed (labelled `completed_in_revision` in `v21_structural.csv`).")
    L.append("- `data/coverage.json`: coverage theory + Monte Carlo.")
    L.append("\n## Commands (run from `revision/code/`)\n")
    L.append("```bash")
    L.append("python3 -m pip install -r ../requirements.txt")
    L.append("python3 tests_pbcore.py                 # all validation and regression checks")
    L.append("python3 make_figures.py                 # regenerate tables/figures from bundled data")
    L.append("```\n\nOptional full numerical regeneration (can be expensive):\n\n```bash")
    L.append("python3 reanalyze_v21.py --original-data /path/to/benchmarks_bimodal  # separate archive")
    L.append("python3 run_coverage.py                 # Fig. 2 data")
    L.append("python3 run_gap_scan.py --procs 15      # Fig. 3 / Table III data (resumable)")
    L.append("python3 run_product_baseline.py         # Table I data")
    L.append("python3 make_fig_pipeline.py            # Fig. 1")
    L.append("python3 make_figures.py                 # Figs. 2-3, table bodies, data/numbers.json")
    L.append("python3 make_repro_report.py           # current environment and source hashes")
    L.append("python3 package_release.py --prepare   # pin corrected software/data in the citation")
    L.append("python3 build_submission.py            # regenerate marked source and build all 3 PDFs")
    L.append("python3 package_release.py --archive   # prepare the upload bundle; no remote publication")
    L.append("```")
    with open(os.path.join(REV, "REPRODUCIBILITY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote REPRODUCIBILITY.md")


if __name__ == "__main__":
    main()
