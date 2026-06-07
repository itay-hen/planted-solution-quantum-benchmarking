# Improvements over the original code

Each item names the problem in the original script collection
(`qb_planted_v5.py`, `qb_assess.py`, `spectrum_analysis_single.py`,
`minGapFind_v3_single.py`, `filelistgenerator.py`) and how the rewrite
addresses it. Both code reviews converged on items 1 and 2 as the most
important; the rest follow the union of the two reviews.

## High-priority correctness

**1. Assessment reconstructed clauses by replaying the RNG.**
The original `qb_assess.replay_clause_hamiltonians` re-ran the generator's
random stream and hard-coded the default code path: `alpha in [0.5, 1.5]`,
default `lambda` bounds, uniform subset sampling, fixed block size. Any
non-default configuration produced silently wrong clause diagnostics
(non-default `alpha`/`lambda` failed the internal gap-identity check;
`variable_block_sizes` asserted out; `bernoulli` desynced the RNG and ran
with wrong numbers).
*Fix:* the key stores each clause's exact `H_local`, spectrum, planted
restriction, weight, and qubit order. `reconstruct.reconstruct_clauses_from_key`
reads these directly — no RNG. Covered by
`test_reconstruction_nondefault_alpha`,
`test_reconstruction_nondefault_lambda_and_scheme`,
`test_reconstruction_bernoulli_subset`,
`test_reconstruction_variable_block_sizes`.

**2. `F0_pass = fidelity[0] > 0.999` false-failed on degenerate instances.**
In a degenerate ground space the diagonalizer returns an arbitrary basis,
so the planted state's overlap with the *first* eigenvector can be small
even when it lies entirely in the ground space — exactly the low-clause
regime the manuscript treats as legitimate.
*Fix:* `verify_instance` and `compute_assessment` report the overlap with
the whole numerically-degenerate ground **subspace**,
`sum_{j: E_j - E_0 < eps} |<psi_j|Psi*>|^2`, and report the degeneracy.
Covered by `test_degenerate_ground_subspace_fidelity`.

**3. The diagonal planted-SAT channel was not implemented.**
The original generator only produced Haar-random planted clauses and
commuting projectors; it could not ingest a planted bitstring + classical
clauses and emit the corresponding diagonal Pauli Hamiltonian — the
manuscript's route to inherited hardness.
*Fix:* `PlantedSATConfig` / `generate_planted_sat_instance` plant `z*`,
draw clauses satisfied by `z*`, and compile each to a rank-1 diagonal
penalty; optional Clifford scrambling preserves the spectrum. Covered by
`test_planted_sat_is_diagonal_and_solved`,
`test_planted_sat_with_clifford_preserves_spectrum`.

**4. Clifford convention was undocumented and easy to misuse.**
The round-trip was internally consistent, but the stored gate list's
meaning (which product, `UHU^dagger` vs `U^dagger H U`, and how to rebuild
the scrambled state) was unspecified; reading the list and applying it in
"listed order" reconstructs the wrong state.
*Fix:* one documented convention (`U = G_1...G_m`, `H' = UHU^dagger`,
ground `= U|Psi*>` via the reversed gate list); a single conjugation
function with the inverse rules *derived* from the forward rules (the
original kept two hand-written copies); `apply_clifford` builds `U|Psi*>`
and `verify_instance` checks its energy. Covered by
`test_clifford_symbolic_matches_dense`, `test_scrambled_ground_state_is_U_psi`.

**5. Unsupported configurations failed silently or produced wrong output.**
*Fix:* `_validate_general` raises on `num_planted` exceeding the block
dimension, `n` not divisible by `block_size` (which would leave free
qubits and an artificial degeneracy), `blocks_per_clause` exceeding the
number of blocks, and unknown schemes. Bernoulli sampling raises if a
clause support exceeds `max_support_qubits` (Bernoulli at fixed
probability grows the support like `O(n)` and breaks the `O(1)`-locality
that the polynomial-size claim depends on). Covered by
`test_validation_errors`.

## Diagnostics clarity

**6. Scrambled-mode diagnostics mixed hidden- and public-basis quantities.**
The original computed clause expectations and planted fidelity on the
*unscrambled* eigenvectors while computing entanglement/IPR on the
*scrambled* eigenvectors, under unqualified names.
*Fix:* `compute_assessment` labels them explicitly — `hidden_*`
(clause structure, in the unscrambled basis) vs `public_*` (entanglement,
IPR, on the public scrambled ground state).

**7. Sign diagnostics presented as basis-independent.**
*Fix:* the sign-structure output is labeled a computational-basis
heuristic and notes it is not a basis-independent obstruction; complex
off-diagonal entries are counted separately.

## Spectral / adiabatic analysis

**8. `minGapFind` and `spectrum_analysis` reported `E1 - E0` only.**
For `spectrum_analysis` this matches the manuscript's Delta_1 definition
and is fine — but it is uninformative on a degenerate ground space, and in
`minGapFind` the resulting zero is an interpretive trap.
*Fix:* `ground_gap` reports raw `E1 - E0`, the gap above the ground
manifold, and the degeneracy. The adiabatic probe reports the gap above
the manifold along the path.

**9. `minGapFind` could miss narrow avoided crossings.**
The original only refined minima already visible on the coarse grid.
*Fix:* `adiabatic_min_gap` uses a dense grid plus golden-section
refinement around the discrete minimum, classifies the location
(interior / endpoint / closing), and centers the problem Hamiltonian and
rescales it by spectral width so the `(1-s)/s` mixing is not distorted by a
large constant energy offset (see item 21). It is documented as an
adiabatic-hardness probe, distinct from Delta_1.

## Performance and robustness

**10. Memory/compute.** Reconstruction no longer retains `K` dense
`2^n x 2^n` matrices — the Hamiltonian is accumulated once as a Pauli sum.
Clause expectations use local-support contraction. `ground_gap` uses a
sparse k-lowest solve (scipy) above a configurable qubit threshold. Dense
analysis defaults cap at `n = 12` (the original allowed `n > 18`, where a
dense complex matrix is terabytes). Sign-structure adjacency is built only
when frustration is explicitly requested.

**11. `operator_to_pauli_expansion` silently dropped imaginary parts.**
*Fix:* `pauli.pauli_decompose` asserts the largest imaginary coefficient
is below tolerance and raises otherwise (a large imaginary part signals a
tensor-ordering bug or non-Hermitian input). Covered by
`test_pauli_roundtrip_and_hermiticity_assertion`.

**12. Silent batch scripts.** The drop-in scripts write `error_<stem>.log`
and exit non-zero on failure instead of a bare `exit(1)`.

**13. `filelistgenerator` was brittle.** `make_filelist.py` takes the
target script and directory as arguments and shell-quotes file names.

**14. Default mismatch.** The original single-instance `general` default
used `block_size=2, blocks_per_clause=2`, silently differing from the
paper. `GeneralConfig` now defaults to the manuscript's single-planted
ensemble (`block_size=1, blocks_per_clause=3`); `generate_paper_suite`
reproduces the paper parameters.

**15. Dynamics frame.** The commuting module's dynamical reference now
states its frame explicitly — the initial state is `U|init>` (a stabilizer
state) and the observable is measured on `H'` — and is computed by direct
dense evolution (small `n`), marked experimental, with no factorization to
get wrong.

## v2.1 — validation and scaling hygiene

Follow-up fixes from a third review of the v2.0 package.

**16. Parameters are validated at generation time, not only at
verification.** `GeneralConfig` now rejects non-positive clause weights
(`alpha_min`, `alpha_max > 0`), `alpha_min > alpha_max`, negative
`spectral_gap`, `lambda_max <= lambda_min`, `golf_course` with positive
`lambda_min` (which would put the planted level above the excited band),
and a `uniform`/`linear` excited band that would fall below the planted
level. `PlantedSATConfig` rejects non-positive `weight`; `CommutingConfig`
rejects non-positive or mis-ordered weights. Covered by
`test_validation_rejects_bad_parameters`.

**17. Key-size bound made explicit.** `max_support_qubits` default lowered
from 12 to 6 (JSON key grows as `4^s` per clause); README documents the
bound and points to a binary array backend for large support rather than
seed-based reconstruction.

**18. `ground_gap` no longer falls back to a giant dense solve.** If
`n > dense_threshold_qubits` and scipy is unavailable it now raises,
instead of attempting to build a dense `2^n x 2^n` matrix.

**19. CLI `--max-n` wired through.** `qbench.py spectrum` now passes
`--max-n` to `ground_gap(dense_threshold_qubits=...)` (it was previously
accepted but ignored).

**20. Collision-resistant instance IDs.** Instance IDs now include a short
hash of the full parameter set, so instances sharing `n`/`K`/`seed` but
differing in scheme, eigenvalue bounds, weights, Clifford flag, or block
layout get distinct internal IDs. Covered by
`test_instance_ids_differ_across_parameters`.

**21. Adiabatic-gap normalization centers and scales by width.**
`adiabatic_min_gap` removes the identity (constant) component and
normalizes by spectral width, rather than dividing by spectral radius — a
large ground-energy offset from summing many clauses no longer distorts
the `(1-s)`/`s` interpolation.

## v2.2 — input-validation and documentation polish

Follow-up from a fourth review; no core-logic changes.

**22. Range / structural validation.** Generators now reject out-of-range
and malformed parameters at construction: `bernoulli_prob` outside
`[0, 1]`; `num_clauses < 0`; `blocks_per_clause < 1`; `n_qubits < 1`;
`block_size < 1`; `max_support_qubits < 1`; `clifford_depth < 0`; variable
block sizes `< 1`. For the SAT channel, `planted_assignment` entries that
are not `0` or `1` are rejected rather than silently bit-masked. Covered by
`test_validation_structural_and_probability`.

**23. Multi-planted clauses store `planted_locals` as a list.** Each clause
key now carries the full list of planted local vectors (not just the
first); `planted_local` is retained as a convenience alias for the first.
Reconstruction still uses the block states, so this is a clarity change.
Covered by `test_multi_planted_clause_stores_planted_locals_list`.

**24. Documentation.** The README states explicitly that the JSON
self-contained key targets bounded local support (with `.npz` as the
large-support path), that Pauli decomposition is `O(4^s)`, and that
`adiabatic_min_gap` is a heuristic diagnostic rather than a certified
global minimum-gap finder or part of the manuscript's main spectral
characterization.

## v2.3 — release hygiene

Follow-up from a fifth review; no core-logic changes.

**25. Verification reports a status.** `verify_instance` now sets
`report["status"]` to `verified`, `partial_dense_checks_skipped`
(when `n > max_n`, only clause-spectrum consistency is checked), or
`failed`. The CLI prints the status, so a skipped dense check no longer
reads as full verification.

**26. Strict SAT bitstring validation.** `planted_assignment` entries are
validated by type — integers/booleans equal to 0 or 1 — so floats
(`0.5`, `1.0`) and strings (`"1"`) are rejected rather than cast.

**27. Degenerate-space diagnostics labeled.** `compute_assessment`
reports entanglement/IPR for both the (arbitrary) public ground
eigenvector and the interpretable planted public state `U|Psi*>`, with a
note that the eigenvector is arbitrary when the ground space is
degenerate.

**28. Dynamics reference flagged as non-protocol.** Its docstring now
states that the `U|init>` initial state makes the public/private split
unsettled, so it is a designer-side tool only, explicitly not part of the
main benchmark claim.

**29. Stale docstring fixed.** `mingap_single.py` now describes the
centered / width-normalized interpolation actually used.

**30. CLI exposes the full parameter surface.** The `general` subcommand
gains `--alpha-min/--alpha-max`, `--lambda-min/--lambda-max`,
`--spectral-gap`, `--subset-sampling`, `--bernoulli-prob`,
`--max-support-qubits`, `--variable-block-sizes`, and `--clifford-depth`;
`sat` gains `--weight`, `--planted`, `--clifford-depth`; `commuting` gains
`--weight-min/--weight-max` and `--clifford-depth`.

**31. Packaging.** A `pyproject.toml` is included; `pip install -e .`
installs the package and a `qbench` console script, with `scipy` as the
optional `[sparse]` extra.

**Test runner.** `tests/test_suite.py` pins BLAS/OMP threads to 1 before
importing NumPy and flushes/`os._exit`s after the loop, so it terminates
promptly in containers where BLAS thread teardown otherwise stalls.

## v2.3.1 — instance-ID collision fix

Follow-up from a sixth review.

**32. SAT instance IDs include the planted assignment (bug fix).** The
planted-SAT parameter hash omitted the (user-supplied or drawn) planted
assignment, so two SAT instances differing only in `planted_assignment`
received the same `instance_id` despite different Hamiltonians. The hash
now includes the actual assignment used. Covered by
`test_sat_instance_ids_depend_on_planted_assignment`.

**33. Commuting instance IDs include `variable_block_sizes` (bug fix).**
Same class of collision for `CommutingConfig` differing only in
`variable_block_sizes`; that field is now part of the hash. Covered by
`test_commuting_instance_ids_depend_on_block_sizes`.

**34. `verify_instance` exposes `fully_verified`.** A boolean
(`True` only when `status == "verified"`) distinguishes full verification
from `partial_dense_checks_skipped`. The CLI gains `verify --require-dense`,
which exits non-zero when dense checks were skipped, for automated
pipelines.

**35. Dynamics locality validated.** `commuting_dynamics_reference` rejects
`observable_locality` outside `[1, n_qubits]` instead of failing later
inside `rng.choice`.

**36. Entropy roundoff clamped.** Half-cut entanglement entropy clamps tiny
negative roundoff to 0 (no more `-0.000`).

**37. Stale doc line fixed.** `IMPROVEMENTS.md` item 9 no longer says the
adiabatic path normalizes by "unit spectral radius"; it centers and scales
by spectral width (matching the code and README).

## v2.3.2 — freeze-candidate polish

Follow-up from a seventh review (which recommended freezing after these).

**38. `mingap_single.py` docstring no longer overclaims.** It now says the
grid-plus-refinement "reduces the chance of missing" a narrow avoided
crossing and states plainly that it is a heuristic, not a certified global
minimizer — matching `spectral.py`.

**39. Strict `variable_block_sizes` typing.** Non-integer block sizes
(e.g. `[1.5, 2.5]`) are rejected at the API level with a clear message,
instead of failing later inside the partition slicing. Covered by
`test_validation_structural_and_probability`.

**40. `generate_benchmark_suite` documented as a paper-suite helper.** Its
docstring now states that it exposes only a subset of `GeneralConfig`, uses
the paper defaults, and raises mid-suite on invalid `(n, block_size,
blocks_per_clause)` combinations — pointing to `generate_general_instance`
for full control or graceful per-instance handling.

**Deliberately not in this version (future feature).** The diagonal
planted-SAT channel is implemented and certified, but the bundled SAT
generator draws random satisfied clauses; it does not by itself guarantee
hard instances. Importing an external planted CNF or implementing known
hard planted distributions is the natural next feature for a stronger
"hardness inherited through planted CSPs" claim, and is left as future
work rather than folded into the frozen companion code.
