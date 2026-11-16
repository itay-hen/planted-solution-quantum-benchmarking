# Corrected v22 revision code and data

This is the validated implementation and numerical data for *Certified
Planted-Solution Pauli Hamiltonians for Ground-State Benchmarking* (Kalev & Hen).
The source/data snapshot is `v22-r1`, SHA-256
`0bd8a3597f20c5e59120794a861e61d7923b6643d23d5f4b63a96a5fdbe7735b`.
Every code, vendor, dependency and data file is pinned in [MANIFEST.json](MANIFEST.json).

This directory is independent of the older `planted_benchmark` package at the
repository root. Its v3 publication keys are a separate schema; no compatibility
with the older package's v2 keys is assumed. The older API is retained for existing
users and is documented in [README-legacy.md](../README-legacy.md).

## Install and validate

Python 3.13.4 and the pinned versions below were used for validation. From the
repository root:

```bash
python3 -m pip install -r revision/requirements.txt
cd revision/code
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 tests_pbcore.py
python3 make_figures.py
```

All 20 checks pass, including 672 structural-dimension versus exact-diagonalization
cases, 106 multi-planted cases, and regression checks for Clifford prefixes,
block partitions, and serialized certification keys. The three table bodies and
`numbers.json` reproduce from the bundled records without rerunning the full
numerical campaign. `make_figures.py` writes figures under `revision/figs/`.

## Generate and validate a public instance

Run from `revision/code/`:

```python
import json
import pbcore as pb
from publish import publish, reconstruct, verify_public

instance = pb.generate(8, 12, seed=7, clifford=True)
public, key = publish(instance)
# Keep the key separate from public benchmark instances.
with open("instance.json", "w") as f:
    json.dump(public, f)
with open("key.json", "w") as f:
    json.dump(key, f)

with open("instance.json") as f:
    public_loaded = json.load(f)
with open("key.json") as f:
    key_loaded = json.load(f)
assert verify_public(public_loaded, key_loaded)
rebuilt = reconstruct(key_loaded)  # uses stored local matrices; no RNG replay
```

`generate` requires a positive fixed block size dividing the qubit count; it rejects
unsupported remainders. `instance.prefix(K)` retains the same Clifford frame.
Independent calls with different clause counts can draw different Clifford circuits
because the legacy RNG order samples the circuit after the clauses.

The hardened publication key stores local matrices, spectra, planted block states,
weights, identity offset and the Clifford gate list. `verify_public` checks local
certificates and reconstructs the public Pauli sum without full-system diagonalization.
Older incomplete v3 keys without local matrices must be re-exported.

## Data and experiment scripts

| Script | Purpose |
|---|---|
| `pbcore.py` | Generation, Pauli expansion, structural analysis, coverage and deflated gaps |
| `publish.py` | Traceless public format, complete v3 keys, reconstruction and validation |
| `run_gap_scan.py` | Resumable nested coverage-scale spectral scan |
| `run_coverage.py` | Exact coverage theory and Monte Carlo support sampling |
| `run_product_baseline.py` | Generator-aware product-state optimization |
| `reanalyze_v21.py` | Earlier-ensemble re-analysis; requires a separate original archive |
| `make_figures.py` | Figures, three tables and numerical summaries from bundled records |
| `witness_search.py` | Coordinate-witness obstruction and small Haar examples |
| `make_repro_report.py` | Current environment and source hashes |

The dataset includes 7,000 gap evaluations over 1,400 nested sequences, 180
product-baseline records, 1,400 earlier-ensemble structural records, and 129 spectra
completed during the revision. The original earlier-ensemble instance/key archive
and previously completed spectra are separate inputs. To rerun that re-analysis:

```bash
python3 reanalyze_v21.py --original-data /path/to/benchmarks_bimodal
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for seeds and solver settings. Full
spectral and optimization reruns are expensive and were not repeated for this
release; the tests and processed-data consistency checks were rerun.

## Scientific scope

- No computational-hardness or general GSEE-representativeness claim is made.
- Clifford conjugation preserves the spectrum, term count and coefficient
  magnitudes. A Clifford-dressed Haar block-product state is generally not a
  stabilizer state.
- Half-rank uniqueness on general connected components remains unproved. The
  spectral scan uses a numerical deflation check; coverage certification is a
  theorem for unique local ground spaces.
- Figure 2's open markers are a graph proxy, not verified uniqueness: full coverage
  and at least two distinct supports per connected component. Repeated clauses on
  one support can already have a unique common ground state and are excluded by
  that proxy. Its data field is `P_graph_proxy_mc`.
- Unscrambled single-qubit-block instances succumb to the tested product-state
  baseline. Identity coefficients can leak energy; random local energy shifts do
  not harden a traceless public Hamiltonian.
- The fixed-clause-count exponential-gap-closing claim from the earlier manuscript
  is withdrawn. The gap results characterize finite sizes up to 20 qubits.

The `build_submission.py` and `package_release.py` helpers are retained unchanged
for snapshot provenance. They require companion manuscript/submission sources,
which are not part of this public code/data release. Referee reports and private
correspondence are deliberately excluded.
