#!/usr/bin/env python3
"""
qb_planted_complete.py - Quantum Benchmarking via Planted-Solution Pauli Hamiltonians

Complete implementation covering:
- Section II: General planted block-product Hamiltonians
- Section III: Multiple planted solutions
- Section VII: Commuting-projector subclass for dynamical benchmarks
"""

from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# === PAULI UTILITIES ===
PAULI_MATRICES = {
    "I": np.array([[1,0],[0,1]], dtype=complex),
    "X": np.array([[0,1],[1,0]], dtype=complex),
    "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
    "Z": np.array([[1,0],[0,-1]], dtype=complex),
}
PAULI_LABELS = ("I", "X", "Y", "Z")

def kron_sequence(matrices: Sequence[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0]], dtype=complex)
    for mat in matrices:
        result = np.kron(result, mat)
    return result

def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    return kron_sequence([PAULI_MATRICES[p] for p in pauli_str])

def generate_all_pauli_strings(n_qubits: int):
    if n_qubits == 0:
        yield ""
        return
    for rest in generate_all_pauli_strings(n_qubits - 1):
        for p in PAULI_LABELS:
            yield rest + p

def operator_to_pauli_expansion(H: np.ndarray, tolerance: float = 1e-14) -> Dict[str, float]:
    dim = H.shape[0]
    n_qubits = int(round(math.log2(dim)))
    coefficients: Dict[str, float] = {}
    for pauli_str in generate_all_pauli_strings(n_qubits):
        P = pauli_string_to_matrix(pauli_str)
        coeff = np.trace(H @ P) / float(dim)
        if abs(coeff.real) > tolerance:
            coefficients[pauli_str] = float(coeff.real)
    return coefficients

def embed_local_pauli(local_pauli: str, local_to_global: List[int], n_total: int) -> str:
    global_pauli = ["I"] * n_total
    for local_idx, pauli_char in enumerate(local_pauli):
        global_pauli[local_to_global[local_idx]] = pauli_char
    return "".join(global_pauli)

def accumulate_pauli_term(acc: Dict[str, float], pauli_str: str, coeff: float, tol: float = 1e-14):
    acc[pauli_str] = acc.get(pauli_str, 0.0) + coeff
    if abs(acc[pauli_str]) <= tol:
        del acc[pauli_str]

# === CLIFFORD GATES ===
class CliffordGate:
    def __init__(self, name: str, qubits: List[int], rules: Dict[str, Tuple[str, complex]]):
        self.name, self.qubits, self.rules = name, qubits, rules
    def conjugate(self, pauli: str) -> Tuple[str, complex]:
        return self.rules[pauli]

def H_gate(q): return CliffordGate(f"H({q})", [q], {"I":("I",1),"X":("Z",1),"Y":("Y",-1),"Z":("X",1)})
def S_gate(q): return CliffordGate(f"S({q})", [q], {"I":("I",1),"X":("Y",1),"Y":("X",-1),"Z":("Z",1)})
def Sdg_gate(q): return CliffordGate(f"Sdg({q})", [q], {"I":("I",1),"X":("Y",-1),"Y":("X",1),"Z":("Z",1)})
def CNOT_gate(c,t): return CliffordGate(f"CNOT({c},{t})", [c,t], {
    "II":("II",1),"IX":("IX",1),"IY":("ZY",1),"IZ":("ZZ",1),
    "XI":("XX",1),"XX":("XI",1),"XY":("YZ",1),"XZ":("YY",-1),
    "YI":("YX",1),"YX":("YI",1),"YY":("XZ",-1),"YZ":("XY",1),
    "ZI":("ZI",1),"ZX":("ZX",1),"ZY":("IY",1),"ZZ":("IZ",1)})
def CZ_gate(a,b): return CliffordGate(f"CZ({a},{b})", [a,b], {
    "II":("II",1),"IX":("ZX",1),"IY":("ZY",1),"IZ":("IZ",1),
    "XI":("XZ",1),"XX":("YY",1),"XY":("YX",-1),"XZ":("XI",1),
    "YI":("YZ",1),"YX":("XY",-1),"YY":("XX",1),"YZ":("YI",1),
    "ZI":("ZI",1),"ZX":("IX",1),"ZY":("IY",1),"ZZ":("ZZ",1)})

class CliffordCircuit:
    def __init__(self, n_qubits: int):
        self.n_qubits, self.gates = n_qubits, []
    def h(self, q): self.gates.append(H_gate(q)); return self
    def s(self, q): self.gates.append(S_gate(q)); return self
    def sdg(self, q): self.gates.append(Sdg_gate(q)); return self
    def cnot(self, c, t): self.gates.append(CNOT_gate(c,t)); return self
    def cz(self, a, b): self.gates.append(CZ_gate(a,b)); return self
    
    def conjugate_pauli(self, pauli_str: str, coeff: float) -> Tuple[str, float]:
        current, phase = pauli_str, complex(1.0)
        for gate in reversed(self.gates):
            local = "".join(current[q] for q in gate.qubits)
            new_local, p = gate.conjugate(local)
            phase *= p
            lst = list(current)
            for i, q in enumerate(gate.qubits):
                lst[q] = new_local[i]
            current = "".join(lst)
        return current, coeff * phase.real
    
    def to_dict(self):
        return {"n_qubits": self.n_qubits, "depth": len(self.gates),
                "gates": [{"name": g.name, "qubits": g.qubits} for g in self.gates]}
    
    @classmethod
    def random_circuit(cls, n, depth, rng, gate_set="full"):
        circ = cls(n)
        for layer in range(depth):
            for q in range(n):
                c = rng.randint(0,5)
                if c==0: circ.h(q)
                elif c==1: circ.s(q)
                elif c==2: circ.sdg(q)
                elif c==3: circ.h(q); circ.s(q)
                elif c==4: circ.s(q); circ.h(q)
            if gate_set == "full":
                for i in range(layer%2, n-1, 2):
                    if rng.random() < 0.7:
                        if rng.random() < 0.5: circ.cnot(i, i+1)
                        else: circ.cz(i, i+1)
        return circ

# === RANDOM STATES ===
def normalize(v):
    n = np.linalg.norm(v)
    return v/n if n > 1e-15 else v

def sample_haar_state(dim, rng):
    v = rng.standard_normal(dim) + 1j*rng.standard_normal(dim)
    return normalize(v)

def sample_haar_unitary(dim, rng):
    Z = rng.standard_normal((dim,dim)) + 1j*rng.standard_normal((dim,dim))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q @ np.diag(d/np.abs(d))

def complete_to_basis(seeds, rng):
    dim, k = seeds[0].size, len(seeds)
    basis = np.zeros((dim,dim), dtype=complex)
    for i, v in enumerate(seeds):
        basis[:,i] = normalize(v)
    U = sample_haar_unitary(dim, rng)
    col = k
    for j in range(dim):
        if col >= dim: break
        v = U[:,j].copy()
        for m in range(col):
            v -= np.vdot(basis[:,m], v) * basis[:,m]
        if np.linalg.norm(v) > 1e-10:
            basis[:,col] = normalize(v)
            col += 1
    return basis

# === CLAUSE CONSTRUCTION ===

# Eigenvalue scheme options:
#   "bimodal"    : ground = -1, half of excited = -1, half = +1 (default)
#   "golf_course": ground = -1, all excited = 0
#   "uniform"    : ground = lmin, excited ~ Uniform[lmin+gap, lmax]
#   "linear"     : ground = -1, excited evenly spaced from -1+gap to +1

def generate_clause_eigenvalues(dim: int, scheme: str = "bimodal", 
                                 lmin: float = -1.0, lmax: float = 1.0,
                                 gap: float = 0.5, rng=None) -> np.ndarray:
    """
    Generate eigenvalues for a clause Hamiltonian.
    
    Args:
        dim: Hilbert space dimension (number of eigenvalues)
        scheme: Eigenvalue distribution scheme
        lmin: Minimum eigenvalue (ground state energy)
        lmax: Maximum eigenvalue (for uniform/linear schemes)
        gap: Spectral gap above ground (for uniform/linear schemes)
        rng: NumPy random generator (required for uniform scheme)
    
    Returns:
        Array of eigenvalues with eigs[0] = ground eigenvalue
    """
    if scheme == "bimodal":
        # Ground state and half of excited states at -1, other half at +1
        n_low = dim // 2  # includes ground state
        n_high = dim - n_low
        eigs = np.concatenate([np.full(n_low, -1.0), np.full(n_high, 1.0)])
        
    elif scheme == "golf_course":
        # Ground state at -1, all excited states at 0
        eigs = np.zeros(dim)
        eigs[0] = -1.0
        
    elif scheme == "uniform":
        # Ground at lmin, excited uniform in [lmin+gap, lmax]
        if rng is None:
            raise ValueError("uniform scheme requires rng")
        l0 = lmin
        if gap and gap > 0:
            excited = rng.uniform(lmin + gap, lmax, size=dim - 1)
        else:
            excited = rng.uniform(lmin + 1e-6, lmax, size=dim - 1)
        eigs = np.concatenate([[l0], excited])
        
    elif scheme == "linear":
        # Ground at -1, excited evenly spaced from -1+gap to +1
        eigs = np.zeros(dim)
        eigs[0] = -1.0
        if dim > 1:
            eigs[1:] = np.linspace(-1.0 + gap, 1.0, dim - 1)
            
    else:
        raise ValueError(f"Unknown eigenvalue scheme: {scheme}")
    
    return eigs


def construct_generic_clause(planted, rng, scheme: str = "bimodal",
                              lmin: float = -1.0, lmax: float = 1.0, 
                              gap: float = 0.5):
    """
    Construct a generic clause Hamiltonian with planted ground state.
    
    Args:
        planted: Planted ground state vector
        rng: NumPy random generator
        scheme: Eigenvalue distribution scheme
        lmin: Minimum eigenvalue (for uniform scheme)
        lmax: Maximum eigenvalue (for uniform scheme)
        gap: Spectral gap (for uniform/linear schemes)
    
    Returns:
        (H, eigenbasis, eigenvalues) where H is the clause Hamiltonian
    """
    dim = planted.size
    planted = normalize(planted)
    eigenbasis = complete_to_basis([planted], rng)
    eigs = generate_clause_eigenvalues(dim, scheme, lmin, lmax, gap, rng)
    H = eigenbasis @ np.diag(eigs) @ eigenbasis.conj().T
    return (H + H.conj().T) / 2, eigenbasis, eigs

def construct_multi_planted_clause(planted_list, rng, scheme: str = "bimodal",
                                    lmin: float = -1.0, lmax: float = 1.0,
                                    gap: float = 0.5):
    """
    Construct a clause Hamiltonian with multiple planted ground states.
    
    Args:
        planted_list: List of m planted ground state vectors (must be orthogonal)
        rng: NumPy random generator
        scheme: Eigenvalue distribution scheme
        lmin: Minimum eigenvalue (for uniform scheme)
        lmax: Maximum eigenvalue (for uniform scheme)
        gap: Spectral gap (for uniform/linear schemes)
    
    Returns:
        (H, eigenbasis, eigenvalues) where H is the clause Hamiltonian
    """
    m, dim = len(planted_list), planted_list[0].size
    mat = np.column_stack([normalize(p) for p in planted_list])
    Q, R = np.linalg.qr(mat)
    seeds = [Q[:, a] for a in range(m)]
    eigenbasis = complete_to_basis(seeds, rng)
    
    # Generate eigenvalues for excited states
    excited_dim = dim - m
    
    if scheme == "bimodal":
        # All m ground states at -1, half of excited at -1, half at +1
        n_low = excited_dim // 2
        n_high = excited_dim - n_low
        ground_eigs = np.full(m, -1.0)
        excited_eigs = np.concatenate([np.full(n_low, -1.0), np.full(n_high, 1.0)])
        
    elif scheme == "golf_course":
        # All m ground states at -1, all excited at 0
        ground_eigs = np.full(m, -1.0)
        excited_eigs = np.zeros(excited_dim)
        
    elif scheme == "uniform":
        ground_eigs = np.full(m, lmin)
        if gap and gap > 0:
            excited_eigs = rng.uniform(lmin + gap, lmax, size=excited_dim)
        else:
            excited_eigs = rng.uniform(lmin + 1e-6, lmax, size=excited_dim)
            
    elif scheme == "linear":
        ground_eigs = np.full(m, -1.0)
        if excited_dim > 0:
            excited_eigs = np.linspace(-1.0 + gap, 1.0, excited_dim)
        else:
            excited_eigs = np.array([])
    else:
        raise ValueError(f"Unknown eigenvalue scheme: {scheme}")
    
    eigs = np.concatenate([ground_eigs, excited_eigs])
    H = eigenbasis @ np.diag(eigs) @ eigenbasis.conj().T
    return (H + H.conj().T) / 2, eigenbasis, eigs

def construct_projector_clause(planted, weight=1.0):
    psi = normalize(planted)
    return -weight * np.outer(psi, psi.conj()), -weight

# === GENERAL CONSTRUCTION (Sec II-III) ===
@dataclass
class GeneralConfig:
    n_qubits: int = 12
    block_size: int = 2
    variable_block_sizes: Optional[List[int]] = None
    num_planted: int = 1
    num_clauses: int = 10
    blocks_per_clause: int = 2
    eigenvalue_scheme: str = "bimodal"  # "bimodal", "golf_course", "uniform", "linear"
    lambda_min: float = -1.0
    lambda_max: float = 1.0
    spectral_gap: Optional[float] = 0.5
    alpha_min: float = 0.5
    alpha_max: float = 1.5
    clifford: bool = False
    clifford_depth: int = 5
    clifford_gate_set: str = "full"
    subset_sampling: str = "uniform"
    bernoulli_prob: float = 0.5
    seed: int = 42

def generate_general_instance(cfg: GeneralConfig) -> Tuple[Dict, Dict]:
    rng_py, rng_np = random.Random(cfg.seed), np.random.default_rng(cfg.seed)
    
    # Block partition
    qubits = list(range(cfg.n_qubits))
    rng_py.shuffle(qubits)
    if cfg.variable_block_sizes:
        blocks, idx = [], 0
        for sz in cfg.variable_block_sizes:
            if idx + sz <= cfg.n_qubits:
                blocks.append(sorted(qubits[idx:idx+sz]))
                idx += sz
    else:
        blocks = [sorted(qubits[i:i+cfg.block_size]) 
                  for i in range(0, cfg.n_qubits, cfg.block_size)
                  if i + cfg.block_size <= cfg.n_qubits]
    
    num_blocks = len(blocks)
    if num_blocks < cfg.blocks_per_clause:
        raise ValueError(f"Need {cfg.blocks_per_clause} blocks, have {num_blocks}")
    
    # Block states
    block_states, block_data = [], []
    for bi, bq in enumerate(blocks):
        bdim = 2**len(bq)
        if cfg.num_planted == 1:
            psi = sample_haar_state(bdim, rng_np)
            block_states.append([psi])
            block_data.append({"block_index": bi, "block_qubits": bq, "type": "haar",
                "states": [{"re": psi.real.tolist(), "im": psi.imag.tolist()}]})
        else:
            U = sample_haar_unitary(bdim, rng_np)
            states = [U[:,a].copy() for a in range(cfg.num_planted)]
            block_states.append(states)
            block_data.append({"block_index": bi, "block_qubits": bq, "type": "haar_orthogonal",
                "states": [{"re": s.real.tolist(), "im": s.imag.tolist()} for s in states]})
    
    # Clauses
    H_terms: Dict[str, float] = {}
    E0 = 0.0
    clause_data = []
    
    for k in range(cfg.num_clauses):
        if cfg.subset_sampling == "bernoulli":
            Sk = [i for i in range(num_blocks) if rng_py.random() < cfg.bernoulli_prob]
            if not Sk: Sk = [rng_py.randint(0, num_blocks-1)]
        else:
            Sk = rng_py.sample(range(num_blocks), cfg.blocks_per_clause)
        
        local_to_global = []
        for bi in Sk:
            local_to_global.extend(blocks[bi])
        
        if cfg.num_planted == 1:
            planted = np.array([1.0], dtype=complex)
            for bi in Sk:
                planted = np.kron(planted, block_states[bi][0])
            Hc, _, eigs = construct_generic_clause(planted, rng_np, 
                                                    scheme=cfg.eigenvalue_scheme,
                                                    lmin=cfg.lambda_min, 
                                                    lmax=cfg.lambda_max, 
                                                    gap=cfg.spectral_gap)
            l0 = eigs[0]
        else:
            planted_list = []
            for a in range(cfg.num_planted):
                st = np.array([1.0], dtype=complex)
                for bi in Sk:
                    st = np.kron(st, block_states[bi][a])
                planted_list.append(st)
            Hc, _, eigs = construct_multi_planted_clause(planted_list, rng_np,
                                                          scheme=cfg.eigenvalue_scheme,
                                                          lmin=cfg.lambda_min,
                                                          lmax=cfg.lambda_max,
                                                          gap=cfg.spectral_gap)
            l0 = eigs[0]
        
        alpha = rng_py.uniform(cfg.alpha_min, cfg.alpha_max)
        E0 += alpha * l0
        
        for lp, c in operator_to_pauli_expansion(Hc).items():
            gp = embed_local_pauli(lp, local_to_global, cfg.n_qubits)
            accumulate_pauli_term(H_terms, gp, alpha * c)
        
        clause_data.append({"k": k, "Sk": Sk, "qubits": local_to_global, "alpha": alpha, 
                           "l0": l0, "local_spectrum": eigs.tolist()})
    
    # Clifford scrambling
    cliff_data = None
    if cfg.clifford:
        circ = CliffordCircuit.random_circuit(cfg.n_qubits, cfg.clifford_depth, rng_py, cfg.clifford_gate_set)
        scrambled: Dict[str, float] = {}
        for p, c in H_terms.items():
            np_, nc = circ.conjugate_pauli(p, c)
            accumulate_pauli_term(scrambled, np_, nc)
        H_terms = scrambled
        cliff_data = circ.to_dict()
    
    iid = f"general_{cfg.n_qubits}q_{cfg.num_clauses}c_seed{cfg.seed}"
    instance = {
        "schema": "planted-benchmark/general/v1", "instance_id": iid,
        "n_qubits": cfg.n_qubits,
        "hamiltonian": {"format": "pauli_sum", "num_terms": len(H_terms),
            "terms": [{"pauli": p, "coeff": c} for p,c in sorted(H_terms.items())]},
        "metadata": {"clifford_scrambled": cfg.clifford}
    }
    key = {
        "schema": "planted-benchmark-key/general/v1", "instance_id": iid,
        "ground_energy": E0, "num_planted": cfg.num_planted,
        "planted_structure": {"blocks": blocks, "block_states": block_data},
        "clauses": clause_data,
        "params": {"n_qubits": cfg.n_qubits, "block_size": cfg.block_size, 
                   "num_blocks": num_blocks, "num_clauses": cfg.num_clauses,
                   "blocks_per_clause": cfg.blocks_per_clause, 
                   "eigenvalue_scheme": cfg.eigenvalue_scheme,
                   "seed": cfg.seed}
    }
    if cliff_data:
        key["clifford_circuit"] = cliff_data
    return instance, key

# === COMMUTING PROJECTOR (Sec VII) ===
@dataclass
class CommutingConfig:
    n_qubits: int = 12
    block_size: int = 2
    variable_block_sizes: Optional[List[int]] = None
    weight_min: float = 0.5
    weight_max: float = 1.5
    clifford: bool = True
    clifford_depth: int = 5
    clifford_gate_set: str = "full"
    include_dynamics: bool = True
    num_observables: int = 5
    observable_locality: int = 2
    time_points: Optional[List[float]] = None
    seed: int = 42

@dataclass
class CommutingResult:
    instance: Dict
    key: Dict
    dynamics: Optional[Dict] = None

def generate_commuting_instance(cfg: CommutingConfig) -> CommutingResult:
    rng_py, rng_np = random.Random(cfg.seed), np.random.default_rng(cfg.seed)
    
    qubits = list(range(cfg.n_qubits))
    rng_py.shuffle(qubits)
    if cfg.variable_block_sizes:
        blocks, idx = [], 0
        for sz in cfg.variable_block_sizes:
            if idx + sz <= cfg.n_qubits:
                blocks.append(sorted(qubits[idx:idx+sz]))
                idx += sz
    else:
        blocks = [sorted(qubits[i:i+cfg.block_size]) 
                  for i in range(0, cfg.n_qubits, cfg.block_size)
                  if i + cfg.block_size <= cfg.n_qubits]
    
    num_blocks = len(blocks)
    block_states, block_weights, block_bases, block_data = [], [], [], []
    
    for bi, bq in enumerate(blocks):
        bdim = 2**len(bq)
        psi = sample_haar_state(bdim, rng_np)
        w = rng_py.uniform(cfg.weight_min, cfg.weight_max)
        basis = complete_to_basis([psi], rng_np)
        block_states.append(psi)
        block_weights.append(w)
        block_bases.append(basis)
        block_data.append({"block_index": bi, "block_qubits": bq, "weight": w,
            "planted_state": {"re": psi.real.tolist(), "im": psi.imag.tolist()},
            "local_spectrum": {"ground": -w, "excited": 0.0}})
    
    H_terms: Dict[str, float] = {}
    E0 = -sum(block_weights)
    gap = min(block_weights)
    
    for bi, bq in enumerate(blocks):
        Hb, _ = construct_projector_clause(block_states[bi], block_weights[bi])
        for lp, c in operator_to_pauli_expansion(Hb).items():
            gp = embed_local_pauli(lp, bq, cfg.n_qubits)
            accumulate_pauli_term(H_terms, gp, c)
    
    circ = None
    cliff_data = None
    if cfg.clifford:
        circ = CliffordCircuit.random_circuit(cfg.n_qubits, cfg.clifford_depth, rng_py, cfg.clifford_gate_set)
        scrambled: Dict[str, float] = {}
        for p, c in H_terms.items():
            np_, nc = circ.conjugate_pauli(p, c)
            accumulate_pauli_term(scrambled, np_, nc)
        H_terms = scrambled
        cliff_data = circ.to_dict()
    
    dynamics = None
    if cfg.include_dynamics:
        dynamics = _gen_dynamics(cfg, blocks, block_states, block_weights, block_bases, circ, rng_py)
    
    iid = f"commuting_{cfg.n_qubits}q_{num_blocks}b_seed{cfg.seed}"
    instance = {
        "schema": "planted-benchmark/commuting/v1", "instance_id": iid,
        "n_qubits": cfg.n_qubits,
        "hamiltonian": {"format": "pauli_sum", "num_terms": len(H_terms),
            "terms": [{"pauli": p, "coeff": c} for p,c in sorted(H_terms.items())]},
        "metadata": {"clifford_scrambled": cfg.clifford, "has_dynamics": cfg.include_dynamics}
    }
    key = {
        "schema": "planted-benchmark-key/commuting/v1", "instance_id": iid,
        "ground_energy": E0, "spectral_gap": gap,
        "planted_structure": {"blocks": blocks, "block_data": block_data},
        "params": {"n_qubits": cfg.n_qubits, "block_size": cfg.block_size, 
                   "num_blocks": num_blocks, "seed": cfg.seed}
    }
    if cliff_data:
        key["clifford_circuit"] = cliff_data
    return CommutingResult(instance=instance, key=key, dynamics=dynamics)

def _gen_dynamics(cfg, blocks, states, weights, bases, circ, rng):
    n = cfg.n_qubits
    times = cfg.time_points or [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    obs_data = []
    
    for oi in range(cfg.num_observables):
        pchars = ["I"]*n
        nnt = rng.randint(1, cfg.observable_locality)
        for q in rng.sample(range(n), nnt):
            pchars[q] = rng.choice(["X","Y","Z"])
        obs_p = "".join(pchars)
        
        if circ:
            back_p, phase = obs_p, 1.0
            for gate in circ.gates:
                lp = "".join(back_p[q] for q in gate.qubits)
                nlp, ph = gate.conjugate(lp)
                phase *= ph
                lst = list(back_p)
                for i,q in enumerate(gate.qubits): lst[q] = nlp[i]
                back_p = "".join(lst)
            back_p, phase = back_p, float(np.real(phase))
        else:
            back_p, phase = obs_p, 1.0
        
        init_bits = "".join(str(rng.randint(0,1)) for _ in range(n))
        
        exp_vals = []
        for t in times:
            val = _compute_exp(t, back_p, phase, init_bits, blocks, bases, weights)
            exp_vals.append({"t": t, "value": val})
        
        obs_data.append({"index": oi, "pauli": obs_p, "initial": init_bits, "values": exp_vals})
    
    return {"times": times, "observables": obs_data}

def _compute_exp(t, obs_p, phase, init_bits, blocks, bases, weights):
    num_blocks = len(blocks)
    init_states = []
    for bi, bq in enumerate(blocks):
        bdim = 2**len(bq)
        bits = [int(init_bits[q]) for q in bq]
        idx = sum(b*(2**(len(bits)-1-i)) for i,b in enumerate(bits))
        sv = np.zeros(bdim, dtype=complex)
        sv[idx] = 1.0
        init_states.append(sv)
    
    active = {}
    for bi, bq in enumerate(blocks):
        lps = "".join(obs_p[q] for q in bq)
        if lps != "I"*len(bq):
            active[bi] = lps
    
    result = complex(phase)
    for bi in range(num_blocks):
        basis, w = bases[bi], weights[bi]
        init_st = init_states[bi]
        bdim = basis.shape[0]
        coeffs = basis.conj().T @ init_st
        eigs = np.zeros(bdim)
        eigs[0] = -w
        
        if bi in active:
            Ploc = pauli_string_to_matrix(active[bi])
            local_sum = 0.0
            for r in range(bdim):
                for rp in range(bdim):
                    ph = np.exp(1j*t*(eigs[r]-eigs[rp]))
                    me = basis[:,r].conj() @ Ploc @ basis[:,rp]
                    local_sum += coeffs[r].conj() * coeffs[rp] * ph * me
            result *= local_sum
    
    return float(np.real(result))

# === VERIFICATION ===
def verify_energy(terms, state_vec, expected, tol=1e-10):
    computed = 0.0
    for t in terms:
        P = pauli_string_to_matrix(t["pauli"])
        computed += t["coeff"] * np.real(state_vec.conj() @ P @ state_vec)
    return abs(computed - expected) < tol, computed

def reconstruct_state(key):
    """Reconstruct planted state in standard qubit order."""
    blocks = key["planted_structure"]["blocks"]
    bd = key["planted_structure"].get("block_states") or key["planted_structure"].get("block_data")
    
    # First build state in block order
    block_states = []
    for b in bd:
        si = b["states"][0] if "states" in b else b["planted_state"]
        re = np.array(si.get("re") or si.get("amplitudes_re"))
        im = np.array(si.get("im") or si.get("amplitudes_im"))
        block_states.append(re + 1j*im)
    
    # Build mapping: position in block-order tensor -> position in standard order
    n_qubits = sum(len(blk) for blk in blocks)
    
    # block_order_qubits[i] = which global qubit is at position i in block-ordered tensor
    block_order_qubits = []
    for blk in blocks:
        block_order_qubits.extend(blk)
    
    # Compute state in block order
    state_block_order = block_states[0]
    for bs in block_states[1:]:
        state_block_order = np.kron(state_block_order, bs)
    
    # Permute to standard order
    # For each computational basis state |i⟩ in standard order,
    # find corresponding index in block order
    dim = 2**n_qubits
    state_std = np.zeros(dim, dtype=complex)
    
    for std_idx in range(dim):
        # Extract bits in standard order
        std_bits = [(std_idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        # Reorder to block order
        block_bits = [std_bits[q] for q in block_order_qubits]
        # Compute block-order index
        block_idx = sum(b * (2**(n_qubits - 1 - i)) for i, b in enumerate(block_bits))
        state_std[std_idx] = state_block_order[block_idx]
    
    return state_std

def verify_instance(instance, key, tol=1e-8):
    report = {"checks": [], "all_passed": True}
    
    n = instance.get("n_qubits")
    is_scrambled = instance.get("metadata", {}).get("clifford_scrambled", False)
    
    if n and n <= 14:
        try:
            state = reconstruct_state(key)
            
            if is_scrambled and "clifford_circuit" in key:
                # For U† H' U where H' = U H U†, we get back H
                # U† P U: iterate gates forward, use inverse gate rules
                circ_data = key["clifford_circuit"]
                terms = instance["hamiltonian"]["terms"]
                
                # Build inverse conjugation rules
                # H† = H, CNOT† = CNOT, CZ† = CZ (self-inverse)
                # S† = Sdg, Sdg† = S
                def get_inverse_rules(name):
                    if name.startswith("H("):
                        return {"I":("I",1),"X":("Z",1),"Y":("Y",-1),"Z":("X",1)}
                    elif name.startswith("Sdg("):  # Sdg† = S
                        return {"I":("I",1),"X":("Y",1),"Y":("X",-1),"Z":("Z",1)}
                    elif name.startswith("S("):  # S† = Sdg
                        return {"I":("I",1),"X":("Y",-1),"Y":("X",1),"Z":("Z",1)}
                    elif name.startswith("CNOT("):
                        return {"II":("II",1),"IX":("IX",1),"IY":("ZY",1),"IZ":("ZZ",1),
                                "XI":("XX",1),"XX":("XI",1),"XY":("YZ",1),"XZ":("YY",-1),
                                "YI":("YX",1),"YX":("YI",1),"YY":("XZ",-1),"YZ":("XY",1),
                                "ZI":("ZI",1),"ZX":("ZX",1),"ZY":("IY",1),"ZZ":("IZ",1)}
                    elif name.startswith("CZ("):
                        return {"II":("II",1),"IX":("ZX",1),"IY":("ZY",1),"IZ":("IZ",1),
                                "XI":("XZ",1),"XX":("YY",1),"XY":("YX",-1),"XZ":("XI",1),
                                "YI":("YZ",1),"YX":("XY",-1),"YY":("XX",1),"YZ":("YI",1),
                                "ZI":("ZI",1),"ZX":("IX",1),"ZY":("IY",1),"ZZ":("ZZ",1)}
                
                unscrambled_terms = []
                for t in terms:
                    p, c = t["pauli"], t["coeff"]
                    # Iterate gates FORWARD for U† P U
                    for g in circ_data["gates"]:
                        rules = get_inverse_rules(g["name"])
                        qs = g["qubits"]
                        lp = "".join(p[q] for q in qs)
                        nlp, ph = rules[lp]
                        c *= float(np.real(ph))
                        lst = list(p)
                        for i, q in enumerate(qs):
                            lst[q] = nlp[i]
                        p = "".join(lst)
                    unscrambled_terms.append({"pauli": p, "coeff": c})
                terms = unscrambled_terms
            else:
                terms = instance["hamiltonian"]["terms"]
            
            E0 = key["ground_energy"]
            ok, computed = verify_energy(terms, state, E0, tol)
            report["checks"].append({"name": "ground_energy", "passed": ok, 
                "expected": E0, "computed": computed, "diff": abs(computed-E0)})
            if not ok: report["all_passed"] = False
        except Exception as e:
            import traceback
            report["checks"].append({"name": "ground_energy", "passed": False, 
                "error": str(e), "traceback": traceback.format_exc()})
            report["all_passed"] = False
    
    return report

# === CLI ===
def generate_benchmark_suite(
    output_dir: str = "benchmarks",
    n_values: List[int] = [6, 8, 10, 12, 14, 16],
    K_values: List[int] = [6, 8, 10, 12, 14, 16, 18, 20],
    instances_per_config: int = 10,
    block_size: int = 1,
    blocks_per_clause: int = 3,
    num_planted: int = 1,
    eigenvalue_scheme: str = "bimodal",
    clifford: bool = False,
    lambda_min: float = -1.0,
    lambda_max: float = 1.0,
    spectral_gap: float = 0.5,
    base_seed: int = 1000,
):
    """
    Generate a grid of benchmark instances.
    
    Args:
        output_dir: Directory to save instances
        n_values: List of qubit counts
        K_values: List of clause counts
        instances_per_config: Number of instances per (n, K) pair
        block_size: Size of each block |A_i|
        blocks_per_clause: Number of blocks per clause |S_k|
        num_planted: Number of planted solutions (m)
        eigenvalue_scheme: Clause eigenvalue scheme ("bimodal", "golf_course", "uniform", "linear")
        clifford: Whether to apply Clifford scrambling
        lambda_min, lambda_max: Clause eigenvalue range (for uniform scheme)
        spectral_gap: Minimum gap above ground state (for uniform/linear schemes)
        base_seed: Starting seed (incremented for each instance)
        
    Returns:
        List of dicts with instance metadata
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    seed = base_seed
    
    total = len(n_values) * len(K_values) * instances_per_config
    count = 0
    
    for n in n_values:
        # Check feasibility: need at least blocks_per_clause blocks
        num_blocks = n // block_size
        if num_blocks < blocks_per_clause:
            print(f"Skipping n={n}: only {num_blocks} blocks, need {blocks_per_clause}")
            continue
            
        for K in K_values:
            for inst_idx in range(instances_per_config):
                count += 1
                
                cfg = GeneralConfig(
                    n_qubits=n,
                    block_size=block_size,
                    num_planted=num_planted,
                    num_clauses=K,
                    blocks_per_clause=blocks_per_clause,
                    eigenvalue_scheme=eigenvalue_scheme,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                    spectral_gap=spectral_gap,
                    clifford=clifford,
                    seed=seed,
                )
                
                try:
                    instance, key = generate_general_instance(cfg)
                    
                    # File naming: n{n}_K{K}_idx{idx}_seed{seed}
                    basename = f"n{n}_K{K}_idx{inst_idx:02d}_seed{seed}"
                    inst_path = os.path.join(output_dir, f"{basename}_instance.json")
                    key_path = os.path.join(output_dir, f"{basename}_key.json")
                    
                    with open(inst_path, "w") as f:
                        json.dump(instance, f, indent=2)
                    with open(key_path, "w") as f:
                        json.dump(key, f, indent=2)
                    
                    meta = {
                        "n": n,
                        "K": K,
                        "instance_idx": inst_idx,
                        "seed": seed,
                        "ground_energy": key["ground_energy"],
                        "num_pauli_terms": instance["hamiltonian"]["num_terms"],
                        "instance_file": inst_path,
                        "key_file": key_path,
                    }
                    results.append(meta)
                    
                    if count % 20 == 0 or count == total:
                        print(f"[{count}/{total}] Generated n={n}, K={K}, idx={inst_idx}")
                        
                except Exception as e:
                    print(f"Error at n={n}, K={K}, idx={inst_idx}: {e}")
                
                seed += 1
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "description": "Planted-solution Pauli Hamiltonian benchmark suite",
        "parameters": {
            "n_values": n_values,
            "K_values": K_values,
            "instances_per_config": instances_per_config,
            "block_size": block_size,
            "blocks_per_clause": blocks_per_clause,
            "num_planted": num_planted,
            "eigenvalue_scheme": eigenvalue_scheme,
            "clifford": clifford,
            "lambda_range": [lambda_min, lambda_max],
            "spectral_gap": spectral_gap,
            "base_seed": base_seed,
        },
        "num_instances": len(results),
        "instances": results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nGenerated {len(results)} instances")
    print(f"Manifest: {manifest_path}")
    
    return results


def generate_paper_benchmark_suite(output_dir: str = "paper_benchmarks", base_seed: int = 1000,
                                    eigenvalue_scheme: str = "bimodal"):
    """
    Generate the benchmark suite specified for the paper:
    - n = 6, 8, 10, 12, 14, 16 qubits
    - K = 6 -- 30  clauses
    - 50 instances per (n, K) configuration
    - block_size = 1, blocks_per_clause = 3
    - m = 1 (single planted solution)
    - No Clifford scrambling
    - Eigenvalue scheme: bimodal (default)
    """
    return generate_benchmark_suite(
        output_dir=output_dir,
        n_values=[6, 8, 10, 12, 14, 16],
        K_values=[6, 10, 14, 18, 22, 26, 30],
        instances_per_config=50,
        block_size=1,
        blocks_per_clause=3,
        num_planted=1,
        eigenvalue_scheme=eigenvalue_scheme,
        clifford=False,
        base_seed=base_seed,
    )
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Planted-solution quantum benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    g = sub.add_parser("general", help="General construction (Sec II-III)")
    g.add_argument("--n-qubits", type=int, default=12)
    g.add_argument("--block-size", type=int, default=2)
    g.add_argument("--num-planted", type=int, default=1)
    g.add_argument("--num-clauses", type=int, default=10)
    g.add_argument("--blocks-per-clause", type=int, default=2)
    g.add_argument("--eigenvalue-scheme", type=str, default="bimodal",
                   choices=["bimodal", "golf_course", "uniform", "linear"],
                   help="Clause eigenvalue distribution scheme (default: bimodal)")
    g.add_argument("--lambda-min", type=float, default=-1.0)
    g.add_argument("--lambda-max", type=float, default=1.0)
    g.add_argument("--spectral-gap", type=float, default=0.5)
    g.add_argument("--clifford", action="store_true")
    g.add_argument("--clifford-depth", type=int, default=5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", type=str, default="benchmark")
    g.add_argument("--verify", action="store_true")
    
    c = sub.add_parser("commuting", help="Commuting projector (Sec VII)")
    c.add_argument("--n-qubits", type=int, default=12)
    c.add_argument("--block-size", type=int, default=2)
    c.add_argument("--weight-min", type=float, default=0.5)
    c.add_argument("--weight-max", type=float, default=1.5)
    c.add_argument("--clifford", action="store_true")
    c.add_argument("--clifford-depth", type=int, default=5)
    c.add_argument("--include-dynamics", action="store_true")
    c.add_argument("--num-observables", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--output", type=str, default="benchmark")
    c.add_argument("--verify", action="store_true")
    
    b = sub.add_parser("batch", help="Generate benchmark suite")
    b.add_argument("--output-dir", type=str, default="benchmarks")
    b.add_argument("--n-values", type=str, default="6,8,10,12,14,16", 
                   help="Comma-separated qubit counts")
    b.add_argument("--K-values", type=str, default="6,8,10,12,14,16,18,20",
                   help="Comma-separated clause counts")
    b.add_argument("--instances", type=int, default=10)
    b.add_argument("--block-size", type=int, default=1)
    b.add_argument("--blocks-per-clause", type=int, default=3)
    b.add_argument("--eigenvalue-scheme", type=str, default="bimodal",
                   choices=["bimodal", "golf_course", "uniform", "linear"],
                   help="Clause eigenvalue distribution scheme (default: bimodal)")
    b.add_argument("--base-seed", type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.cmd == "general":
        cfg = GeneralConfig(
            n_qubits=args.n_qubits, block_size=args.block_size, num_planted=args.num_planted,
            num_clauses=args.num_clauses, blocks_per_clause=args.blocks_per_clause,
            eigenvalue_scheme=args.eigenvalue_scheme,
            lambda_min=args.lambda_min, lambda_max=args.lambda_max, spectral_gap=args.spectral_gap,
            clifford=args.clifford, clifford_depth=args.clifford_depth, seed=args.seed)
        
        print(f"Generating general instance: {cfg.n_qubits}q, {cfg.num_clauses} clauses, {cfg.num_planted} planted, scheme={cfg.eigenvalue_scheme}")
        inst, key = generate_general_instance(cfg)
        
        with open(f"{args.output}_instance.json", "w") as f: json.dump(inst, f, indent=2)
        with open(f"{args.output}_key.json", "w") as f: json.dump(key, f, indent=2)
        
        print(f"Ground energy: {key['ground_energy']:.10f}")
        print(f"Pauli terms: {inst['hamiltonian']['num_terms']}")
        
        if args.verify:
            r = verify_instance(inst, key)
            for ch in r["checks"]:
                print(f"  {'PASS' if ch['passed'] else 'FAIL'}: {ch['name']} (diff={ch.get('diff',0):.2e})")
    
    elif args.cmd == "commuting":
        cfg = CommutingConfig(
            n_qubits=args.n_qubits, block_size=args.block_size,
            weight_min=args.weight_min, weight_max=args.weight_max,
            clifford=args.clifford, clifford_depth=args.clifford_depth,
            include_dynamics=args.include_dynamics, num_observables=args.num_observables, seed=args.seed)
        
        print(f"Generating commuting instance: {cfg.n_qubits}q, clifford={cfg.clifford}")
        res = generate_commuting_instance(cfg)
        
        with open(f"{args.output}_instance.json", "w") as f: json.dump(res.instance, f, indent=2)
        with open(f"{args.output}_key.json", "w") as f: json.dump(res.key, f, indent=2)
        if res.dynamics:
            with open(f"{args.output}_dynamics.json", "w") as f: json.dump(res.dynamics, f, indent=2)
        
        print(f"Ground energy: {res.key['ground_energy']:.10f}")
        print(f"Spectral gap: {res.key['spectral_gap']:.10f}")
        print(f"Pauli terms: {res.instance['hamiltonian']['num_terms']}")
        
        if args.verify:
            r = verify_instance(res.instance, res.key)
            for ch in r["checks"]:
                print(f"  {'PASS' if ch['passed'] else 'FAIL'}: {ch['name']} (diff={ch.get('diff',0):.2e})")
    
    elif args.cmd == "batch":
        n_vals = [int(x) for x in args.n_values.split(",")]
        K_vals = [int(x) for x in args.K_values.split(",")]
        
        print(f"Generating benchmark suite:")
        print(f"  n values: {n_vals}")
        print(f"  K values: {K_vals}")
        print(f"  instances per config: {args.instances}")
        print(f"  block_size={args.block_size}, blocks_per_clause={args.blocks_per_clause}")
        print(f"  eigenvalue_scheme={args.eigenvalue_scheme}")
        print()
        
        generate_benchmark_suite(
            output_dir=args.output_dir,
            n_values=n_vals,
            K_values=K_vals,
            instances_per_config=args.instances,
            block_size=args.block_size,
            blocks_per_clause=args.blocks_per_clause,
            eigenvalue_scheme=args.eigenvalue_scheme,
            base_seed=args.base_seed,
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command-line usage
        main()
    else:
        # Interactive usage (IDLE, Jupyter, etc.)
        print("Interactive mode - use the functions directly:")
        print()
        print("  # Generate a single instance:")
        print("  cfg = GeneralConfig(n_qubits=8, num_clauses=5, eigenvalue_scheme='bimodal')")
        print("  instance, key = generate_general_instance(cfg)")
        print()
        print("  # Or use the quick function:")
        print("  instance, key = quick_instance(n_qubits=8, num_clauses=5, scheme='golf_course')")
        print()
        print("Available eigenvalue schemes: 'bimodal' (default), 'golf_course', 'uniform', 'linear'")


# === INTERACTIVE CONVENIENCE FUNCTIONS ===

def quick_instance(
    n_qubits: int = 12,
    num_clauses: int = 10,
    block_size: int = 2,
    blocks_per_clause: int = 2,
    scheme: str = "bimodal",
    num_planted: int = 1,
    clifford: bool = False,
    clifford_depth: int = 5,
    seed: int = 42,
    verify: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Quick generation of a benchmark instance for interactive use.
    
    Args:
        n_qubits: Number of qubits
        num_clauses: Number of clauses (K)
        block_size: Size of each block |A_i|
        blocks_per_clause: Number of blocks per clause |S_k|
        scheme: Eigenvalue scheme ("bimodal", "golf_course", "uniform", "linear")
        num_planted: Number of planted ground states
        clifford: Whether to apply Clifford scrambling
        clifford_depth: Depth of Clifford circuit
        seed: Random seed
        verify: Whether to verify the instance (for small systems)
    
    Returns:
        (instance, key) tuple
    
    Example:
        >>> instance, key = quick_instance(n_qubits=8, num_clauses=5, scheme='golf_course')
        >>> print(f"Ground energy: {key['ground_energy']}")
    """
    cfg = GeneralConfig(
        n_qubits=n_qubits,
        block_size=block_size,
        num_planted=num_planted,
        num_clauses=num_clauses,
        blocks_per_clause=blocks_per_clause,
        eigenvalue_scheme=scheme,
        clifford=clifford,
        clifford_depth=clifford_depth,
        seed=seed,
    )
    
    print(f"Generating: {n_qubits}q, {num_clauses} clauses, scheme={scheme}, seed={seed}")
    instance, key = generate_general_instance(cfg)
    
    print(f"Ground energy: {key['ground_energy']:.10f}")
    print(f"Pauli terms: {instance['hamiltonian']['num_terms']}")
    
    if verify and n_qubits <= 14:
        result = verify_instance(instance, key)
        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {status}: {check['name']} (diff={check.get('diff', 0):.2e})")
    
    return instance, key


def quick_commuting(
    n_qubits: int = 12,
    block_size: int = 2,
    clifford: bool = True,
    clifford_depth: int = 5,
    include_dynamics: bool = False,
    seed: int = 42,
    verify: bool = True,
) -> CommutingResult:
    """
    Quick generation of a commuting-projector instance for interactive use.
    
    Args:
        n_qubits: Number of qubits
        block_size: Size of each block
        clifford: Whether to apply Clifford scrambling
        clifford_depth: Depth of Clifford circuit
        include_dynamics: Whether to compute dynamical reference data
        seed: Random seed
        verify: Whether to verify the instance
    
    Returns:
        CommutingResult with instance, key, and optional dynamics
    """
    cfg = CommutingConfig(
        n_qubits=n_qubits,
        block_size=block_size,
        clifford=clifford,
        clifford_depth=clifford_depth,
        include_dynamics=include_dynamics,
        seed=seed,
    )
    
    print(f"Generating commuting: {n_qubits}q, clifford={clifford}, seed={seed}")
    result = generate_commuting_instance(cfg)
    
    print(f"Ground energy: {result.key['ground_energy']:.10f}")
    print(f"Spectral gap: {result.key['spectral_gap']:.10f}")
    print(f"Pauli terms: {result.instance['hamiltonian']['num_terms']}")
    
    if verify and n_qubits <= 14:
        vresult = verify_instance(result.instance, result.key)
        for check in vresult["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {status}: {check['name']} (diff={check.get('diff', 0):.2e})")
    
    return result


def show_clause_spectra(key: Dict):
    """Display the local spectrum of each clause."""
    scheme = key.get('params', {}).get('eigenvalue_scheme', 'unknown')
    print(f"Eigenvalue scheme: {scheme}")
    print()
    for clause in key.get('clauses', []):
        k = clause['k']
        spec = clause.get('local_spectrum', [])
        print(f"Clause {k}: support={clause['qubits']}, alpha={clause['alpha']:.4f}")
        print(f"  Spectrum: {spec[:8]}{'...' if len(spec) > 8 else ''}")
        print(f"  Ground: {spec[0] if spec else 'N/A'}, dim={len(spec)}")
        print()

