# Example instances

Pre-generated `*_instance.json` (public) and `*_key.json` (private
certificate) pairs so you can exercise the tools immediately, without
generating anything first. All five verify cleanly. They are small
(`n = 6`–`10`) and diagonalize in well under a second.

| basename | what it shows | n | E0 | ground degeneracy |
|---|---|---|---|---|
| `general_single_n8` | clean single-planted Haar instance | 8 | −13.71 | 1 (non-degenerate) |
| `general_degenerate_n6_seed0` | legitimate degenerate ground space at low clause count | 6 | −2.42 | 4 |
| `general_scrambled_n6` | bounded-depth Clifford scrambling (`H' = U H U†`) | 6 | −10.32 | 1 |
| `general_multiplanted_n6` | two planted states, exact 2-dim ground space (`golf_course`) | 6 | −5.20 | 2 |
| `planted_sat_n10_k3` | diagonal classical planted-SAT channel | 10 | 0.00 | 25 |

Each is reproducible from the seed in its key's `params` via `qbench.py`;
they are committed only for convenience.

## Try them

From the repository root (commands take a basename and read both
`<base>_instance.json` and `<base>_key.json`):

```bash
# certify the planted ground state against the public Hamiltonian
python qbench.py verify   examples/general_single_n8

# diagnostics: energy, gap above the ground manifold, planted fidelity,
# hidden/public-basis structure
python qbench.py assess   examples/general_degenerate_n6_seed0

# ground gap (E1-E0 and the gap above the full ground manifold, degeneracy)
python qbench.py spectrum examples/general_multiplanted_n6

# adiabatic minimum-gap probe (heuristic; natural for the SAT channel)
python qbench.py mingap   examples/planted_sat_n10_k3
```

The degenerate example is the one to look at for the manuscript's
low-clause-count regime: `verify` still reports `verified` because the
planted state is certified by its energy and its overlap with the whole
degenerate ground subspace, not with a single arbitrary eigenvector.

The SAT example is diagonal in the computational basis (its public
Hamiltonian is a sum of rank-1 penalties); its planted assignment is a
ground state at energy 0, and the ground space is degenerate because the
random clause set has several satisfying assignments.
