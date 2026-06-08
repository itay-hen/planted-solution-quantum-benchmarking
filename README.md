# planted_benchmark

A reference implementation of the planted-solution Pauli-Hamiltonian
benchmarking construction (Kalev & Hen, *Planted-Solution Pauli
Hamiltonians as a Quantum Benchmarking Primitive*). The manuscript is
included in this repository as [`qb_gs_v18.pdf`](qb_gs_v18.pdf).

This is a documented, tested rewrite of an earlier script collection
(`qb_planted_v5.py` + `qb_assess.py` + spectral/adiabatic helpers). It
fixes a set of correctness and robustness problems in the original code;
`IMPROVEMENTS.md` maps each fix to the issue it addresses.

## What the construction does

Plant a product state across disjoint `O(1)`-size qubit blocks,

    |Psi*> = (x)_i |psi_{A_i}> .

Build `K` clauses on `O(1)`-size supports whose minimum-eigenvalue space
contains the planted restriction, expose each clause as an exact
(polynomial-size) Pauli expansion, and sum them with positive weights:

    H = sum_k alpha_k H^{(k)} ,   E0 = sum_k alpha_k lambda_0^{(k)} (known exactly).

The public object is only the Pauli sum. A private **key** certifies the
planted ground state and energy. An optional bounded-depth Clifford
`H' = U H U^dagger` preserves the spectrum and maps the ground state to
the stabilizer state `U|Psi*>`.

Two specializations:

* **Diagonal planted-SAT channel** (`PlantedSATConfig`): `block_size=1`,
  computational-basis block states, `0/+1` diagonal clauses. This is the
  route through which classical hardness is inherited; it was missing
  from the original code.
* **Commuting block-projector subclass** (`CommutingConfig`), with an
  optional small-n dynamical reference (experimental).

## Install

```bash
pip install -e .          # installs the package + the `qbench` console script
pip install -e .[sparse]  # also pulls scipy (enables the sparse k-lowest path)
```

scipy is optional; everything (including all tests) runs on numpy alone.
Alternatively, with no install, put the `planted_benchmark/` directory on
your path or run from this directory. Tested on Python 3.10+, NumPy
1.24+/2.x. After an editable install, `qbench ...` works in place of
`python qbench.py ...`.

**Batch runs.** On some containers OpenBLAS oversubscribes threads on
small complex `eigh` calls and becomes very slow (and can be slow to tear
down at process exit). If diagonalization crawls, set
`OPENBLAS_NUM_THREADS=1` (also `OMP_NUM_THREADS=1`) before launching batch
jobs. The bundled test runner sets these itself.

## Quick start (library)

```python
from planted_benchmark import (
    GeneralConfig, generate_general_instance,
    PlantedSATConfig, generate_planted_sat_instance,
    verify_instance, compute_assessment, print_summary, ground_gap,
)

inst, key = generate_general_instance(GeneralConfig(n_qubits=10, num_clauses=14, seed=7))
print(verify_instance(inst, key)["all_passed"])     # True
print_summary(compute_assessment(inst, key))
print(ground_gap(inst))                              # E0, raw gap, gap-above-manifold, degeneracy

# classical diagonal channel
sat_inst, sat_key = generate_planted_sat_instance(PlantedSATConfig(n_qubits=12, k=3, num_clauses=30))
```

## Quick start (CLI)

```bash
python qbench.py general --n 10 --clauses 14 --seed 7 --out inst10
python qbench.py sat     --n 12 --k 3 --clauses 30 --seed 1 --out sat12
python qbench.py verify  inst10
python qbench.py assess  inst10
python qbench.py spectrum inst10           # Delta_1, degeneracy-aware
python qbench.py mingap   sat12            # adiabatic minimum gap
python qbench.py batch   --out suite --n 6 8 10 --clauses 6 14 22 --per-config 10
```

`batch` / `generate_benchmark_suite` is a paper-suite helper: it exposes a
subset of options and uses the manuscript defaults. For the full parameter
surface or graceful per-instance error handling, call
`generate_general_instance` directly.

Pre-generated instance/key pairs for all channels are in
[`examples/`](examples/) — `python qbench.py verify examples/general_single_n8`
works out of the box. See [`examples/README.md`](examples/README.md).

Drop-in single-instance scripts (replacements for the originals) are also
provided: `spectrum_single.py`, `mingap_single.py`, `make_filelist.py`.
They mirror the old file-name I/O contract but report degeneracy-aware
gaps and write `error_<stem>.log` on failure instead of exiting silently.

## The certification key (schema v2)

The single most important change. The **public instance** holds only the
Pauli sum and metadata. The **private key** stores, per clause, the exact
local Hamiltonian matrix `H_local`, its spectrum, its planted restriction,
the weight `alpha`, and the qubit order the clause was built in:

```
key["clauses"][k] = {
    "k", "Sk", "support_qubits", "alpha", "l0", "eigenvalues",
    "H_local":      {"re": [[...]], "im": [[...]]},
    "planted_local":{"re": [...],  "im": [...]},
}
key["planted_structure"] = {"blocks": [...], "block_states": [...]}
```

Down-stream code (`reconstruct`, `verify`, `assess`) reconstructs clauses
by **reading these fields**, never by replaying the generator's RNG. Any
coefficient range, eigenvalue scheme, subset-sampling mode, or block
layout therefore reconstructs exactly.

**Key size is bounded but support-dependent.** Each clause stores a
`2^s x 2^s` complex matrix for support size `s`, so the JSON key grows as
`4^s` per clause. This is the deliberate cost of self-containment (a
seed-plus-hash scheme would be compact but would reintroduce the
RNG-replay reconstruction this rewrite removed). The default
`max_support_qubits = 6` keeps keys small; the paper ensemble uses `s = 3`.
For genuinely large support, store the matrices in a binary array backend
(`.npz`) rather than JSON — do **not** revert to seed-based reconstruction.

## Conventions

* **Qubit order.** Qubit 0 is the left-most Pauli character and the
  most-significant tensor factor; computational index bit of qubit `q` is
  `(b >> (n-1-q)) & 1`. One convention everywhere.
* **Clifford.** A circuit `[G_1, ..., G_m]` means `U = G_1 @ ... @ G_m`,
  the scrambled Hamiltonian is `H' = U H U^dagger`, and the scrambled
  ground state is `U|Psi*>` (prepared by applying the gate list in
  **reverse** order to `|Psi*>`). See `clifford.py`; `apply_clifford`
  prepares `U|Psi*>` correctly and `verify_instance` checks it.

## Degeneracy

Low clause counts give legitimately degenerate ground spaces. The planted
state is certified by its **energy** and by its overlap with the whole
numerically-degenerate ground **subspace**, never by overlap with a single
(arbitrarily-chosen) ground eigenvector. `ground_gap` reports both the raw
`E1 - E0` (the manuscript's literal Delta_1) and the gap above the ground
manifold, plus the degeneracy.

## Layout

```
planted_benchmark/
  pauli.py        Pauli matrices, decomposition (Hermiticity-checked), dense/sparse build
  states.py       Haar states/unitaries, robust orthonormal basis completion
  clifford.py     Clifford gates/circuits, single conjugation, state preparation
  clauses.py      eigenvalue schemes and local clause builders (Haar / diagonal-SAT)
  generator.py    GeneralConfig / PlantedSATConfig / CommutingConfig, suites, validation
  reconstruct.py  key -> clauses / planted state / Hamiltonian   (no RNG)
  verify.py       degeneracy-robust verification
  assess.py       diagnostics with explicit hidden/public-basis labeling
  spectral.py     ground_gap (Delta_1) and adiabatic_min_gap
qbench.py         unified CLI
spectrum_single.py, mingap_single.py, make_filelist.py    drop-in scripts
tests/test_suite.py    dependency-free tests (python tests/test_suite.py)
```

## Tests

```bash
python tests/test_suite.py
```

No pytest required (the file is also pytest-compatible). The tests target
the behaviours the original code got wrong: reconstruction under
non-default `alpha`/`lambda`/Bernoulli/variable-block configurations,
degeneracy-robust fidelity, the diagonal SAT channel, the Clifford
convention, and loud input validation.

## Scope notes

* Dense exact diagonalization is the default analysis path and is intended
  for small `n` (defaults cap at `n = 12`). The sparse k-lowest path
  (scipy) extends `ground_gap` somewhat further. Generation itself is
  polynomial and works at larger `n`.
* The dynamical reference (`commuting_dynamics_reference`) is experimental,
  small-`n`, and computed by direct dense evolution; dynamical benchmarks
  are future work in the manuscript.
* `adiabatic_min_gap` is a heuristic adiabatic-hardness diagnostic, not part
  of the manuscript's main spectral characterization and not a certified
  global minimum-gap finder (a crossing narrower than the grid spacing can
  be missed). Use `ground_gap` for the manuscript's Delta_1.
* The self-contained JSON key is intended for **bounded local support**
  (default `max_support_qubits = 6`); Pauli decomposition is brute-force
  over `4^s` local strings and the key stores `4^s`-sized matrices, both
  fine for the `O(1)`-support construction. For larger support, use a
  binary (`.npz`) key backend rather than JSON or seed-replay.
