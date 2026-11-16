# Planted-solution quantum benchmarking

The corrected code and numerical data for **Certified Planted-Solution Pauli
Hamiltonians for Ground-State Benchmarking** (Kalev & Hen) are in
**[revision/](revision/README.md)**.

## Start with the corrected revision

```bash
python3 -m pip install -r revision/requirements.txt
cd revision/code
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 tests_pbcore.py
python3 make_figures.py
```

The revision has 20 validation/regression checks, complete JSON certification keys,
and the data needed to regenerate its numerical figures and tables. It fixes
Clifford prefixes losing their circuit, rejects non-divisible fixed block sizes,
and validates serialized keys without replaying the generator's random choices.
See [revision/REPRODUCIBILITY.md](revision/REPRODUCIBILITY.md) for versions and seeds,
and [revision/MANIFEST.json](revision/MANIFEST.json) for the exact source/data hashes.

The construction plants Clifford-dressed block-product ground states of
frustration-free Hamiltonians. It does **not** establish computational hardness.
Clifford-dressed Haar states are generally **not** stabilizer states, and the
earlier fixed-clause-count exponential-gap-closing claim has been withdrawn.
The revision documentation explains the generator-aware baseline, energy leaks,
and distinction between coverage-certified, predicted and numerically verified
uniqueness.

## Existing package users

The original `planted_benchmark` package and `qbench` CLI remain available at the
repository root for compatibility. Their documentation is in
[README-legacy.md](README-legacy.md). `pip install -e .` installs that older API;
it does not install the revision scripts. The root-level `qb_gs_v18.pdf` is a
historical manuscript, not the revised scientific description.

The revision's v3 key schema and older package's v2 schema are separate formats.
New reproduction work should use `revision/`.
