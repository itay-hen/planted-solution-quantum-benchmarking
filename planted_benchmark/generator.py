"""
Instance generators.

Three channels, all emitting the same self-contained key schema (v2):

* ``generate_general_instance``      -- Haar-random planted block-product
  ensemble (manuscript Sec. II; multi-planted is Sec. IV).
* ``generate_planted_sat_instance``  -- the diagonal classical planted-SAT
  embedding channel (manuscript Sec. II), the route through which
  classical hardness is inherited.  This was absent from the original code.
* ``generate_commuting_instance``    -- a commuting block-projector
  subclass with an optional, **experimental** small-n dynamical reference.

Key schema v2 (the important change)
-------------------------------------
The public ``instance`` contains only the Pauli-sum Hamiltonian and
metadata.  The private ``key`` stores, per clause, the exact local
Hamiltonian matrix ``H_local``, its spectrum, its planted restriction,
the clause weight ``alpha``, and the qubit order the clause was built in.
Down-stream code reconstructs clauses by reading these fields; it never
replays the generator's RNG.  Non-default coefficient ranges, eigenvalue
parameters, Bernoulli subset sampling, and variable block sizes are
therefore all reconstructed exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import clauses as _cl
from . import clifford as _cliff
from . import pauli as _pl
from .serial import mat_to_dict, vec_to_dict
from .states import (
    computational_basis_index,
    computational_basis_state,
    sample_haar_state,
    sample_haar_unitary,
)

SCHEMA_INSTANCE = "planted-benchmark/instance/v2"
SCHEMA_KEY = "planted-benchmark-key/v2"


def _param_hash(params: dict, length: int = 8) -> str:
    """Short stable hash of a parameter dict, used to disambiguate
    instance IDs that would otherwise collide across parameter choices
    (e.g. same n, K, seed but different scheme/lambda/alpha/clifford)."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Shared assembly
# ---------------------------------------------------------------------------
def _partition_blocks(
    n: int, block_size: int, variable_block_sizes: Optional[Sequence[int]], rng_py: np.random.Generator
) -> List[List[int]]:
    qubits = list(range(n))
    rng_py.shuffle(qubits)
    if variable_block_sizes:
        if sum(variable_block_sizes) != n:
            raise ValueError(
                f"variable_block_sizes sum to {sum(variable_block_sizes)} != n={n}"
            )
        blocks, idx = [], 0
        for sz in variable_block_sizes:
            blocks.append(sorted(qubits[idx : idx + sz]))
            idx += sz
        return blocks
    if n % block_size != 0:
        raise ValueError(
            f"n={n} is not divisible by block_size={block_size}; "
            "every qubit must belong to a block (silently dropping qubits "
            "would leave free qubits and an artificially degenerate ground space)"
        )
    return [sorted(qubits[i : i + block_size]) for i in range(0, n, block_size)]


def _assemble(
    n: int,
    blocks: List[List[int]],
    block_states_data: List[dict],
    clause_records: List[dict],
    params: dict,
    kind: str,
    num_planted: int,
    extra_key: Optional[dict] = None,
    clifford_gates: Optional[Sequence] = None,
) -> Tuple[Dict, Dict]:
    """
    Build the public instance and private key from pre-built clause records.

    Each record: ``{k, Sk, support_qubits, alpha, clause}`` where ``clause``
    is a dict from :mod:`planted_benchmark.clauses`.
    """
    H_terms: Dict[str, float] = {}
    E0 = 0.0
    for rec in clause_records:
        alpha, clause = rec["alpha"], rec["clause"]
        E0 += alpha * clause["l0"]
        local_terms = _pl.pauli_decompose(clause["H_local"])
        for lp, c in local_terms.items():
            gp = _pl.embed_local_pauli(lp, rec["support_qubits"], n)
            _pl.accumulate_term(H_terms, gp, alpha * c)

    clifford_dict = None
    if clifford_gates:
        H_terms = _cliff.conjugate_pauli_sum(H_terms, clifford_gates, dagger=False)
        clifford_dict = _cliff.circuit_to_dict(clifford_gates, n)

    iid = params.get("instance_id", f"{kind}_{n}q_seed{params.get('seed')}")
    instance = {
        "schema": SCHEMA_INSTANCE,
        "instance_id": iid,
        "kind": kind,
        "n_qubits": n,
        "hamiltonian": {
            "format": "pauli_sum",
            "num_terms": len(H_terms),
            "terms": [{"pauli": p, "coeff": float(c)} for p, c in sorted(H_terms.items())],
        },
        "metadata": {"clifford_scrambled": bool(clifford_gates), "num_planted": num_planted},
    }

    clause_data = []
    for rec in clause_records:
        clause = rec["clause"]
        clause_data.append(
            {
                "k": rec["k"],
                "Sk": [int(x) for x in rec["Sk"]],
                "support_qubits": [int(x) for x in rec["support_qubits"]],
                "alpha": float(rec["alpha"]),
                "l0": float(clause["l0"]),
                "eigenvalues": [float(x) for x in clause["eigenvalues"]],
                "H_local": mat_to_dict(clause["H_local"]),
                "planted_local": vec_to_dict(clause["planted_local"]),
                "planted_locals": [vec_to_dict(v) for v in clause["planted_locals"]],
            }
        )

    key = {
        "schema": SCHEMA_KEY,
        "instance_id": iid,
        "kind": kind,
        "ground_energy": float(E0),
        "num_planted": num_planted,
        "clifford_scrambled": bool(clifford_gates),
        "planted_structure": {"blocks": blocks, "block_states": block_states_data},
        "clauses": clause_data,
        "params": params,
    }
    if clifford_dict:
        key["clifford_circuit"] = clifford_dict
    if extra_key:
        key.update(extra_key)
    return instance, key


# ---------------------------------------------------------------------------
# General Haar-random planted ensemble
# ---------------------------------------------------------------------------
@dataclass
class GeneralConfig:
    """
    Configuration for the Haar-random planted ensemble.

    Defaults reproduce the manuscript's single-planted ensemble
    (``block_size=1``, ``blocks_per_clause=3``).  (The original code's
    single-instance default used ``block_size=2, blocks_per_clause=2``,
    which silently differed from the paper.)
    """

    n_qubits: int = 12
    block_size: int = 1
    variable_block_sizes: Optional[List[int]] = None
    num_planted: int = 1
    num_clauses: int = 10
    blocks_per_clause: int = 3
    eigenvalue_scheme: str = "bimodal"
    lambda_min: float = -1.0
    lambda_max: float = 1.0
    spectral_gap: float = 0.5
    alpha_min: float = 0.5
    alpha_max: float = 1.5
    clifford: bool = False
    clifford_depth: int = 5
    subset_sampling: str = "uniform"  # "uniform" | "bernoulli"
    bernoulli_prob: float = 0.5
    max_support_qubits: int = 6
    seed: int = 42


def _validate_general_basic(cfg: GeneralConfig) -> None:
    """Structural / range checks that do not depend on the block partition."""
    if cfg.n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if cfg.block_size < 1:
        raise ValueError("block_size must be >= 1")
    if cfg.num_clauses < 0:
        raise ValueError("num_clauses must be >= 0")
    if cfg.blocks_per_clause < 1:
        raise ValueError("blocks_per_clause must be >= 1")
    if cfg.max_support_qubits < 1:
        raise ValueError("max_support_qubits must be >= 1")
    if cfg.clifford_depth < 0:
        raise ValueError("clifford_depth must be >= 0")
    if not (0.0 <= cfg.bernoulli_prob <= 1.0):
        raise ValueError("bernoulli_prob must lie in [0, 1]")
    if cfg.variable_block_sizes is not None:
        if any(not isinstance(s, (int, np.integer)) for s in cfg.variable_block_sizes):
            raise ValueError("variable block sizes must be integers")
        if any(int(s) < 1 for s in cfg.variable_block_sizes):
            raise ValueError("every variable block size must be >= 1")


def _validate_general(cfg: GeneralConfig, num_blocks: int, block_dims: Sequence[int]) -> None:
    if cfg.num_planted < 1:
        raise ValueError("num_planted must be >= 1")
    min_block_dim = min(block_dims)
    if cfg.num_planted > min_block_dim:
        raise ValueError(
            f"num_planted={cfg.num_planted} exceeds the smallest block dimension "
            f"{min_block_dim}; cannot have that many mutually orthogonal block states "
            f"(need block_size >= ceil(log2(num_planted)))"
        )
    if cfg.subset_sampling not in ("uniform", "bernoulli"):
        raise ValueError("subset_sampling must be 'uniform' or 'bernoulli'")
    if cfg.subset_sampling == "uniform" and cfg.blocks_per_clause > num_blocks:
        raise ValueError(
            f"blocks_per_clause={cfg.blocks_per_clause} exceeds the number of blocks "
            f"{num_blocks}"
        )
    if cfg.eigenvalue_scheme not in _cl.VALID_SCHEMES:
        raise ValueError(f"unknown eigenvalue scheme {cfg.eigenvalue_scheme!r}")
    if cfg.alpha_min <= 0 or cfg.alpha_max <= 0:
        raise ValueError("clause weights must be positive: require alpha_min, alpha_max > 0")
    if cfg.alpha_min > cfg.alpha_max:
        raise ValueError("alpha_min cannot exceed alpha_max")
    if cfg.spectral_gap < 0:
        raise ValueError("spectral_gap must be nonnegative")
    if cfg.lambda_max <= cfg.lambda_min:
        raise ValueError("lambda_max must exceed lambda_min")
    if cfg.eigenvalue_scheme == "golf_course" and cfg.lambda_min > 0:
        raise ValueError(
            "golf_course places the excited levels at 0, so lambda_min must be <= 0 "
            "for the planted state to remain the minimum-energy state"
        )
    if cfg.eigenvalue_scheme in ("uniform", "linear") and (
        cfg.lambda_min + cfg.spectral_gap > cfg.lambda_max
    ):
        raise ValueError(
            "lambda_min + spectral_gap must be <= lambda_max so the excited band lies "
            "at or above the planted level"
        )


def generate_general_instance(cfg: GeneralConfig) -> Tuple[Dict, Dict]:
    _validate_general_basic(cfg)
    rng_py = np.random.default_rng(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed + 12345)
    rng_c = np.random.default_rng(cfg.seed + 777)

    blocks = _partition_blocks(cfg.n_qubits, cfg.block_size, cfg.variable_block_sizes, rng_py)
    num_blocks = len(blocks)
    block_dims = [2 ** len(b) for b in blocks]
    _validate_general(cfg, num_blocks, block_dims)

    # Block states.
    block_states: List[List[np.ndarray]] = []
    block_states_data: List[dict] = []
    for bi, bq in enumerate(blocks):
        bdim = 2 ** len(bq)
        if cfg.num_planted == 1:
            states = [sample_haar_state(bdim, rng_np)]
            btype = "haar"
        else:
            U = sample_haar_unitary(bdim, rng_np)
            states = [U[:, a].copy() for a in range(cfg.num_planted)]
            btype = "haar_orthogonal"
        block_states.append(states)
        block_states_data.append(
            {
                "block_index": bi,
                "qubits": bq,
                "type": btype,
                "states": [vec_to_dict(s) for s in states],
            }
        )

    # Clauses.
    clause_records = []
    for k in range(cfg.num_clauses):
        if cfg.subset_sampling == "bernoulli":
            Sk = [i for i in range(num_blocks) if rng_py.random() < cfg.bernoulli_prob]
            if not Sk:
                Sk = [int(rng_py.integers(0, num_blocks))]
        else:
            Sk = sorted(int(x) for x in rng_py.choice(num_blocks, cfg.blocks_per_clause, replace=False))

        support_qubits = [q for bi in Sk for q in blocks[bi]]
        if len(support_qubits) > cfg.max_support_qubits:
            raise ValueError(
                f"clause {k} has |support|={len(support_qubits)} > max_support_qubits="
                f"{cfg.max_support_qubits}. With subset_sampling='bernoulli' the support "
                "grows like O(n); lower bernoulli_prob so the expected support stays O(1), "
                "or raise max_support_qubits deliberately."
            )

        planted_list = []
        for a in range(cfg.num_planted):
            v = np.array([1.0], dtype=complex)
            for bi in Sk:
                v = np.kron(v, block_states[bi][a])
            planted_list.append(v)

        clause = _cl.build_generic_clause(
            planted_list, rng_np, scheme=cfg.eigenvalue_scheme,
            lmin=cfg.lambda_min, lmax=cfg.lambda_max, gap=cfg.spectral_gap,
        )
        alpha = float(rng_py.uniform(cfg.alpha_min, cfg.alpha_max))
        clause_records.append(
            {"k": k, "Sk": Sk, "support_qubits": support_qubits, "alpha": alpha, "clause": clause}
        )

    clifford_gates = (
        _cliff.random_circuit(cfg.n_qubits, cfg.clifford_depth, rng_c) if cfg.clifford else None
    )

    params = {
        "n_qubits": cfg.n_qubits, "block_size": cfg.block_size,
        "variable_block_sizes": cfg.variable_block_sizes,
        "num_blocks": num_blocks, "num_clauses": cfg.num_clauses,
        "blocks_per_clause": cfg.blocks_per_clause, "num_planted": cfg.num_planted,
        "eigenvalue_scheme": cfg.eigenvalue_scheme,
        "lambda_min": cfg.lambda_min, "lambda_max": cfg.lambda_max,
        "spectral_gap": cfg.spectral_gap, "alpha_min": cfg.alpha_min, "alpha_max": cfg.alpha_max,
        "subset_sampling": cfg.subset_sampling, "bernoulli_prob": cfg.bernoulli_prob,
        "clifford": cfg.clifford, "clifford_depth": cfg.clifford_depth, "seed": cfg.seed,
    }
    params["instance_id"] = (
        f"general_{cfg.n_qubits}q_{cfg.num_clauses}c_seed{cfg.seed}_{_param_hash(params)}"
    )
    return _assemble(
        cfg.n_qubits, blocks, block_states_data, clause_records, params,
        kind="general", num_planted=cfg.num_planted, clifford_gates=clifford_gates,
    )


# ---------------------------------------------------------------------------
# Diagonal planted-SAT channel
# ---------------------------------------------------------------------------
@dataclass
class PlantedSATConfig:
    """
    Planted random k-SAT compiled to a diagonal Pauli Hamiltonian.

    A planted assignment ``z*`` is fixed, clauses are drawn so that ``z*``
    satisfies all of them, and each clause becomes a rank-1 diagonal
    penalty (``+1`` on its unique violating assignment).  The ground energy
    is 0 and ``z*`` is a ground state.  Optional Clifford scrambling makes
    the Hamiltonian non-diagonal while preserving the spectrum.
    """

    n_qubits: int = 12
    k: int = 3
    num_clauses: int = 20
    weight: float = 1.0
    clifford: bool = False
    clifford_depth: int = 5
    planted_assignment: Optional[List[int]] = None
    seed: int = 42


def generate_planted_sat_instance(cfg: PlantedSATConfig) -> Tuple[Dict, Dict]:
    if cfg.n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if cfg.num_clauses < 0:
        raise ValueError("num_clauses must be >= 0")
    if cfg.clifford_depth < 0:
        raise ValueError("clifford_depth must be >= 0")
    if cfg.k > cfg.n_qubits:
        raise ValueError("k cannot exceed n_qubits")
    if cfg.k < 1:
        raise ValueError("k must be >= 1")
    if cfg.weight <= 0:
        raise ValueError("SAT clause weight must be positive")
    rng = np.random.default_rng(cfg.seed)
    rng_c = np.random.default_rng(cfg.seed + 777)

    if cfg.planted_assignment is not None:
        if len(cfg.planted_assignment) != cfg.n_qubits:
            raise ValueError("planted_assignment length must equal n_qubits")
        if not all(
            isinstance(b, (bool, int, np.integer)) and int(b) in (0, 1)
            for b in cfg.planted_assignment
        ):
            raise ValueError("planted_assignment entries must each be exactly 0 or 1 (integers)")
        z = np.array([int(b) for b in cfg.planted_assignment], dtype=int)
    else:
        z = rng.integers(0, 2, size=cfg.n_qubits)

    blocks = [[i] for i in range(cfg.n_qubits)]
    block_states_data = [
        {"block_index": i, "qubits": [i], "type": "computational",
         "states": [vec_to_dict(computational_basis_state([int(z[i])]))]}
        for i in range(cfg.n_qubits)
    ]

    clause_records = []
    cnf = []
    for k_idx in range(cfg.num_clauses):
        variables = sorted(int(v) for v in rng.choice(cfg.n_qubits, cfg.k, replace=False))
        signs = rng.integers(0, 2, size=cfg.k)  # 1 = positive literal, 0 = negated
        # A literal is false when x_v == (0 if positive else 1).
        violating = np.array([0 if s == 1 else 1 for s in signs], dtype=int)
        planted_bits = np.array([z[v] for v in variables], dtype=int)
        if np.array_equal(violating, planted_bits):
            # z* would violate this clause; flip one literal so z* satisfies it.
            j = int(rng.integers(0, cfg.k))
            signs[j] ^= 1
            violating[j] ^= 1
        v_idx = computational_basis_index([int(b) for b in violating])
        p_idx = computational_basis_index([int(b) for b in planted_bits])
        clause = _cl.build_diagonal_sat_clause(cfg.k, v_idx, p_idx)
        clause_records.append(
            {"k": k_idx, "Sk": variables, "support_qubits": variables,
             "alpha": float(cfg.weight), "clause": clause}
        )
        cnf.append([{"var": int(v), "positive": bool(s)} for v, s in zip(variables, signs)])

    clifford_gates = (
        _cliff.random_circuit(cfg.n_qubits, cfg.clifford_depth, rng_c) if cfg.clifford else None
    )
    params = {
        "n_qubits": cfg.n_qubits, "k": cfg.k, "num_clauses": cfg.num_clauses,
        "weight": cfg.weight, "clifford": cfg.clifford, "clifford_depth": cfg.clifford_depth,
        "seed": cfg.seed, "block_size": 1, "blocks_per_clause": cfg.k,
        "eigenvalue_scheme": "diagonal_sat", "num_blocks": cfg.n_qubits,
        "planted_assignment": [int(b) for b in z],
    }
    params["instance_id"] = (
        f"sat_{cfg.n_qubits}q_{cfg.k}sat_{cfg.num_clauses}c_seed{cfg.seed}_{_param_hash(params)}"
    )
    extra = {"planted_assignment": [int(b) for b in z], "cnf": cnf}
    return _assemble(
        cfg.n_qubits, blocks, block_states_data, clause_records, params,
        kind="planted_sat", num_planted=1, extra_key=extra, clifford_gates=clifford_gates,
    )


# ---------------------------------------------------------------------------
# Commuting block-projector subclass (+ experimental dynamics)
# ---------------------------------------------------------------------------
@dataclass
class CommutingConfig:
    n_qubits: int = 12
    block_size: int = 2
    variable_block_sizes: Optional[List[int]] = None
    weight_min: float = 0.5
    weight_max: float = 1.5
    clifford: bool = True
    clifford_depth: int = 5
    seed: int = 42


def generate_commuting_instance(cfg: CommutingConfig) -> Tuple[Dict, Dict]:
    if cfg.weight_min <= 0 or cfg.weight_max <= 0:
        raise ValueError("projector weights must be positive: require weight_min, weight_max > 0")
    if cfg.weight_min > cfg.weight_max:
        raise ValueError("weight_min cannot exceed weight_max")
    if cfg.n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if cfg.block_size < 1:
        raise ValueError("block_size must be >= 1")
    if cfg.clifford_depth < 0:
        raise ValueError("clifford_depth must be >= 0")
    rng_py = np.random.default_rng(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed + 12345)
    rng_c = np.random.default_rng(cfg.seed + 777)

    blocks = _partition_blocks(cfg.n_qubits, cfg.block_size, cfg.variable_block_sizes, rng_py)

    block_states_data, clause_records, weights = [], [], []
    for bi, bq in enumerate(blocks):
        bdim = 2 ** len(bq)
        psi = sample_haar_state(bdim, rng_np)
        w = float(rng_py.uniform(cfg.weight_min, cfg.weight_max))
        weights.append(w)
        # Ground = -w on the planted state, 0 elsewhere (golf_course with lmin=-w).
        clause = _cl.build_generic_clause([psi], rng_np, scheme="golf_course", lmin=-w)
        clause_records.append(
            {"k": bi, "Sk": [bi], "support_qubits": list(bq), "alpha": 1.0, "clause": clause}
        )
        block_states_data.append(
            {"block_index": bi, "qubits": bq, "type": "haar", "weight": w,
             "states": [vec_to_dict(psi)]}
        )

    clifford_gates = (
        _cliff.random_circuit(cfg.n_qubits, cfg.clifford_depth, rng_c) if cfg.clifford else None
    )
    params = {
        "n_qubits": cfg.n_qubits, "block_size": cfg.block_size,
        "variable_block_sizes": cfg.variable_block_sizes,
        "num_blocks": len(blocks), "seed": cfg.seed, "eigenvalue_scheme": "projector",
        "weight_min": cfg.weight_min, "weight_max": cfg.weight_max,
        "clifford": cfg.clifford, "clifford_depth": cfg.clifford_depth,
    }
    params["instance_id"] = f"commuting_{cfg.n_qubits}q_seed{cfg.seed}_{_param_hash(params)}"
    extra = {"spectral_gap": float(min(weights))}
    return _assemble(
        cfg.n_qubits, blocks, block_states_data, clause_records, params,
        kind="commuting", num_planted=1, extra_key=extra, clifford_gates=clifford_gates,
    )


def commuting_dynamics_reference(
    instance: Dict,
    key: Dict,
    num_observables: int = 5,
    observable_locality: int = 2,
    times: Optional[Sequence[float]] = None,
    seed: int = 0,
    max_n: int = 14,
) -> Dict:
    """
    EXPERIMENTAL, small-n only.  Certified dynamical reference data.

    Frame (made explicit): the public benchmark is the (possibly
    scrambled) Hamiltonian ``H'``.  The reference value for observable
    ``P`` and computational seed ``init`` is

        <psi(t)| P |psi(t)>,   |psi(t)> = e^{-i H' t} |psi_0>,
        |psi_0> = U |init>     (U from the key; |init> if unscrambled).

    i.e. the initial state is the *stabilizer state* ``U|init>``, which a
    solver would prepare from the public circuit description.  This is
    computed by direct dense evolution (small n) so there is no
    factorisation to get wrong; dynamical benchmarks are future work and
    this routine is provided as a designer-side reference only.

    It is NOT a settled public benchmark protocol: the initial state is
    ``U|init>``, so disclosing ``U`` to a solver partially reveals the
    scrambling, while withholding it leaves the public initial state
    operationally unspecified. Defining that public/private split is left
    to future work; do not present this as part of the main benchmark claim.
    """
    from .reconstruct import build_hamiltonian_dense

    n = instance["n_qubits"]
    if n > max_n:
        raise ValueError(f"dynamics reference is dense and small-n only (n={n} > {max_n})")
    if not (1 <= observable_locality <= n):
        raise ValueError("observable_locality must satisfy 1 <= observable_locality <= n_qubits")
    times = list(times) if times is not None else [0.0, 0.5, 1.0, 2.0, 5.0]
    rng = np.random.default_rng(seed)

    H = build_hamiltonian_dense(instance)
    E, W = np.linalg.eigh(H)
    gates = _cliff.circuit_from_dict(key["clifford_circuit"]) if key.get("clifford_circuit") else None

    obs_out = []
    for oi in range(num_observables):
        chars = ["I"] * n
        nnt = int(rng.integers(1, observable_locality + 1))
        for q in rng.choice(n, nnt, replace=False):
            chars[int(q)] = "XYZ"[int(rng.integers(0, 3))]
        obs_p = "".join(chars)
        P = _pl.pauli_string_to_matrix(obs_p)

        init_bits = [int(b) for b in rng.integers(0, 2, size=n)]
        psi0 = computational_basis_state(init_bits)
        if gates:
            psi0 = _cliff.apply_clifford(gates, psi0, dagger=False)

        c = W.conj().T @ psi0
        vals = []
        for t in times:
            ct = np.exp(-1j * E * t) * c
            psit = W @ ct
            vals.append({"t": float(t), "value": float(np.real(psit.conj() @ P @ psit))})
        obs_out.append({"index": oi, "pauli": obs_p, "initial_seed_bits": init_bits, "values": vals})

    return {
        "frame": "initial state is U|init> (stabilizer); observable measured on H'",
        "times": [float(t) for t in times],
        "observables": obs_out,
    }


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------
def generate_benchmark_suite(
    output_dir: str,
    n_values: Sequence[int],
    K_values: Sequence[int],
    instances_per_config: int = 10,
    block_size: int = 1,
    blocks_per_clause: int = 3,
    num_planted: int = 1,
    eigenvalue_scheme: str = "bimodal",
    clifford: bool = False,
    base_seed: int = 1000,
    verbose: bool = True,
) -> List[dict]:
    """
    Paper-style batch helper.

    This is a convenience routine for the manuscript's single-planted
    ensemble. It exposes only a subset of ``GeneralConfig`` (it always uses
    uniform subset sampling, the default coefficient and eigenvalue ranges,
    fixed block size, and one planted state), and an invalid
    ``(n, block_size, blocks_per_clause)`` combination raises mid-suite
    rather than being skipped per item. For the full configuration surface,
    or for graceful per-instance error handling, drive
    ``generate_general_instance`` directly. ``n`` values for which
    ``n // block_size < blocks_per_clause`` are skipped with a message.
    """
    os.makedirs(output_dir, exist_ok=True)
    results, seed = [], base_seed
    total = len(n_values) * len(K_values) * instances_per_config
    count = 0
    for n in n_values:
        if n // block_size < blocks_per_clause:
            if verbose:
                print(f"skip n={n}: only {n // block_size} blocks, need {blocks_per_clause}")
            continue
        for K in K_values:
            for idx in range(instances_per_config):
                count += 1
                cfg = GeneralConfig(
                    n_qubits=n, block_size=block_size, num_planted=num_planted,
                    num_clauses=K, blocks_per_clause=blocks_per_clause,
                    eigenvalue_scheme=eigenvalue_scheme, clifford=clifford, seed=seed,
                )
                inst, key = generate_general_instance(cfg)
                base = f"n{n}_K{K}_idx{idx:02d}_seed{seed}"
                with open(os.path.join(output_dir, f"{base}_instance.json"), "w") as f:
                    json.dump(inst, f)
                with open(os.path.join(output_dir, f"{base}_key.json"), "w") as f:
                    json.dump(key, f)
                results.append(
                    {"n": n, "K": K, "idx": idx, "seed": seed,
                     "ground_energy": key["ground_energy"],
                     "num_pauli_terms": inst["hamiltonian"]["num_terms"],
                     "instance_file": os.path.join(output_dir, f"{base}_instance.json"),
                     "key_file": os.path.join(output_dir, f"{base}_key.json")}
                )
                if verbose and (count % 25 == 0 or count == total):
                    print(f"[{count}/{total}] n={n} K={K} idx={idx}")
                seed += 1
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(
            {"description": "planted-solution Pauli Hamiltonian benchmark suite",
             "parameters": {"n_values": list(n_values), "K_values": list(K_values),
                            "instances_per_config": instances_per_config,
                            "block_size": block_size, "blocks_per_clause": blocks_per_clause,
                            "num_planted": num_planted, "eigenvalue_scheme": eigenvalue_scheme,
                            "clifford": clifford, "base_seed": base_seed},
             "num_instances": len(results), "instances": results},
            f, indent=2,
        )
    if verbose:
        print(f"generated {len(results)} instances in {output_dir}")
    return results


def generate_paper_suite(output_dir: str = "paper_benchmarks", base_seed: int = 1000) -> List[dict]:
    """The manuscript's single-planted ensemble (block_size=1, |S_k|=3,
    bimodal, no Clifford). ``n`` defaults to the exact-diagonalisation
    range studied in the paper."""
    return generate_benchmark_suite(
        output_dir=output_dir, n_values=[6, 8, 10, 12],
        K_values=[6, 10, 14, 18, 22, 26, 30], instances_per_config=50,
        block_size=1, blocks_per_clause=3, num_planted=1,
        eigenvalue_scheme="bimodal", clifford=False, base_seed=base_seed,
    )
