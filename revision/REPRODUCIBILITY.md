# Reproducibility report — qb_gs_v22 revision

Generated 2026-11-15T13:32:50 by `code/make_repro_report.py`.

This records the current validation environment. Original numerical results were generated with Python 3.13.4, NumPy 2.2.6, SciPy 1.16.3 and Matplotlib 3.10.3 on an Apple M3 Max; the full spectral and optimization campaigns have not been rerun by this report generator.

## Environment

- OS: macOS-27.2.1-arm64-arm-64bit-Mach-O
- CPU: , 16 logical cores
- Python 3.13.4, NumPy 2.2.6, SciPy 1.16.3, Matplotlib 3.10.3
- Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` when running numerical scripts.
- LaTeX: TeX Live (pdflatex, bibtex, latexdiff), REVTeX 4.2

## Code versions (SHA-256, first 16 hex digits)

The project folder is not under version control; file hashes identify the exact code.

- `vendor/qb_planted_v5.py`: `ffc95a43458810ad`
- `requirements.txt`: `e02adfbdc4ad271b`
- `code/build_submission.py`: `8b133e5a794ddc18`
- `code/make_fig_pipeline.py`: `855a9d4fc1f62c73`
- `code/make_figures.py`: `715968599fbe8c5e`
- `code/make_repro_report.py`: `8cc213482031da9f`
- `code/package_release.py`: `3d39ad65f33178f2`
- `code/pbcore.py`: `220abc3d453aa7ee`
- `code/publish.py`: `d34dcd06e42b095a`
- `code/reanalyze_v21.py`: `047c27dab34f4461`
- `code/run_coverage.py`: `d3d3ccdd27b130fd`
- `code/run_gap_scan.py`: `babfd60838533933`
- `code/run_product_baseline.py`: `ac6f1dea2bf72617`
- `code/tests_pbcore.py`: `66b555945ceb3c6e`
- `code/witness_search.py`: `845d7290ada3af95`

## Random seeds

- Instance seed `s` drives two streams exactly as in `qb_planted_v5.generate_general_instance`: `random.Random(s)` (partition, supports, weights, Clifford) and `numpy.random.default_rng(s)` (block states, Haar completions).
- Original (v21) data: seeds in the separate original archive's `*_key.json` (`params.seed`); pass its directory via `reanalyze_v21.py --original-data PATH`.
- Coverage-scale scan: `20260922 + 1000 n + i` (half-rank), `+500` (projector), i = instance index; Lanczos start vector from `default_rng(s + 7919)`.
- Coverage Monte Carlo: `random.Random(4242)`; 20000 trials per point.
- Product baseline: instance seeds `777000 + 1000 n + i`; restarts from `default_rng([20260923, n, i, block_size, clifford])`; the |A_i|=2 arm is run for n <= 20 only.

## Solver settings

- Dense spectra (v21 data, n ≤ 12): `numpy.linalg.eigvalsh` (LAPACK zheevd).
- Gaps: `scipy.sparse.linalg.eigsh(k=2, which='SA', tol=1e-10, maxiter=20000)` on the matrix-free operator `H - E0 + sigma P_pred`; sigma = sum_k alpha_k (max lambda - lambda_0).
- Structural kernel tolerance for local clause kernels: |lambda - lambda_0| < 1e-9 (local spectra are exactly -1/+1/0 by construction).
- Numerical multiplicity cross-check: tau = f * eps * sum_k alpha_k ||H^(k)||, f = 1e2..1e6.
- Product baseline: L-BFGS-B, maxiter 5000, gtol 1e-12, ftol 1e-15, 20 restarts; success iff E - E0 <= 1e-8 sum_k alpha_k.
- Failed / non-converged instances: none recorded (eigsh raises on non-convergence; no exceptions occurred). See the validation report.

## Data files

- `data/gap_scan.jsonl`: 7000 records
- `data/product_baseline.jsonl`: 180 records
- `data/v21_structural.csv`: 1400 records
- `data/v21_completed_spectra/*.npy`: spectra computed in the revision for the v21 instances whose runs had not completed (labelled `completed_in_revision` in `v21_structural.csv`).
- `data/coverage.json`: coverage theory + Monte Carlo.

## Commands for this public code/data release

Run from `revision/code/`:

```bash
python3 -m pip install -r ../requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 tests_pbcore.py
python3 make_figures.py
python3 make_repro_report.py
```

Full experiment reruns are optional and expensive:

```bash
python3 run_coverage.py
python3 run_gap_scan.py --procs 15
python3 run_product_baseline.py
python3 reanalyze_v21.py --original-data /path/to/benchmarks_bimodal
```

The original earlier-ensemble archive is a separate input. The included processed
records suffice to regenerate all tables and numerical summaries without it.
The document-build and submission-packaging helpers in `code/` are included for
source provenance; they require the companion manuscript sources and are not part
of the standalone code/data workflow. No referee correspondence is included.
