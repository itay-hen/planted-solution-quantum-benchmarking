"""
planted_benchmark
=================

A reference implementation of the planted-solution Pauli-Hamiltonian
benchmarking construction (Kalev & Hen, "Planted-Solution Pauli
Hamiltonians as a Quantum Benchmarking Primitive").

This package is a from-scratch, documented rewrite of an earlier
script collection (``qb_planted_v5.py`` + ``qb_assess.py`` + helpers).
The rewrite fixes a set of correctness and robustness problems in the
original code; see ``IMPROVEMENTS.md`` for the full mapping from each
fix to the issue it addresses.

The two structural decisions that drive the rewrite:

1.  **The certification key is self-contained.**  Every clause stores
    its exact local Hamiltonian matrix, its spectrum, its planted local
    restriction, and the qubit order it was built in.  Down-stream
    analysis reconstructs clauses by *reading the key*, never by
    replaying the generator's random-number stream.  This removes an
    entire class of silent failures in which the analyzer reproduced
    only the generator's default code path.

2.  **Degeneracy is a first-class citizen.**  The planted state is
    certified by its *energy* and by its overlap with the full
    numerically-degenerate ground subspace, never by its overlap with a
    single (arbitrarily-chosen) ground eigenvector.

Public entry points
--------------------
Generation:
    GeneralConfig, generate_general_instance
    PlantedSATConfig, generate_planted_sat_instance
    CommutingConfig, generate_commuting_instance        (experimental)
    generate_benchmark_suite, generate_paper_suite

Reconstruction (key -> objects, no RNG):
    reconstruct_clauses_from_key, reconstruct_planted_state,
    build_hamiltonian_dense, build_hamiltonian_sparse

Verification & analysis:
    verify_instance
    compute_assessment, save_assessment, load_assessment
    ground_gap, adiabatic_min_gap
"""

from .generator import (
    GeneralConfig,
    generate_general_instance,
    PlantedSATConfig,
    generate_planted_sat_instance,
    CommutingConfig,
    generate_commuting_instance,
    generate_benchmark_suite,
    generate_paper_suite,
)
from .reconstruct import (
    reconstruct_clauses_from_key,
    reconstruct_planted_state,
    build_hamiltonian_dense,
    build_hamiltonian_sparse,
)
from .verify import verify_instance
from .assess import compute_assessment, save_assessment, load_assessment, print_summary
from .spectral import ground_gap, adiabatic_min_gap

__version__ = "2.3.2"

__all__ = [
    "GeneralConfig",
    "generate_general_instance",
    "PlantedSATConfig",
    "generate_planted_sat_instance",
    "CommutingConfig",
    "generate_commuting_instance",
    "generate_benchmark_suite",
    "generate_paper_suite",
    "reconstruct_clauses_from_key",
    "reconstruct_planted_state",
    "build_hamiltonian_dense",
    "build_hamiltonian_sparse",
    "verify_instance",
    "compute_assessment",
    "save_assessment",
    "load_assessment",
    "print_summary",
    "ground_gap",
    "adiabatic_min_gap",
    "__version__",
]
