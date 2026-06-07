"""
Clifford circuits used as a structural obfuscation layer.

Convention (this is the single source of truth for the whole package)
---------------------------------------------------------------------
A circuit is a list of gates ``[G_1, G_2, ..., G_m]``.  The unitary it
represents is the **left-to-right matrix product**

        U = G_1 @ G_2 @ ... @ G_m .

A Hamiltonian is scrambled as ``H' = U H U^\\dagger``.  Consequently the
ground state of ``H'`` is ``U |Psi>``, and -- because applying a circuit
to a state means the *right-most* gate acts first -- ``U |Psi>`` is
prepared by applying the gate list **in reverse order** to ``|Psi>``.
``apply_clifford(gates, psi)`` does exactly this.

Pauli conjugation is implemented once, in ``conjugate_pauli_string``:

* ``dagger=False`` returns ``U P U^\\dagger`` (iterate gates in reverse,
  applying the forward single-gate rule ``g P g^\\dagger`` at each step);
* ``dagger=True``  returns ``U^\\dagger P U``.

The inverse single-gate rules are **derived** from the forward rules by
inverting the (signed) permutation, rather than being typed out a second
time -- the original code kept two hand-written copies and they could
drift apart.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Forward single-gate Heisenberg rules: P -> (g P g^dagger as (string, sign)).
# Signs are real (+/-1) for Clifford conjugation of Paulis.
_H_RULES = {"I": ("I", 1), "X": ("Z", 1), "Y": ("Y", -1), "Z": ("X", 1)}
_S_RULES = {"I": ("I", 1), "X": ("Y", 1), "Y": ("X", -1), "Z": ("Z", 1)}
_SDG_RULES = {"I": ("I", 1), "X": ("Y", -1), "Y": ("X", 1), "Z": ("Z", 1)}
_CNOT_RULES = {
    "II": ("II", 1), "IX": ("IX", 1), "IY": ("ZY", 1), "IZ": ("ZZ", 1),
    "XI": ("XX", 1), "XX": ("XI", 1), "XY": ("YZ", 1), "XZ": ("YY", -1),
    "YI": ("YX", 1), "YX": ("YI", 1), "YY": ("XZ", -1), "YZ": ("XY", 1),
    "ZI": ("ZI", 1), "ZX": ("ZX", 1), "ZY": ("IY", 1), "ZZ": ("IZ", 1),
}
_CZ_RULES = {
    "II": ("II", 1), "IX": ("ZX", 1), "IY": ("ZY", 1), "IZ": ("IZ", 1),
    "XI": ("XZ", 1), "XX": ("YY", 1), "XY": ("YX", -1), "XZ": ("XI", 1),
    "YI": ("YZ", 1), "YX": ("XY", -1), "YY": ("XX", 1), "YZ": ("YI", 1),
    "ZI": ("ZI", 1), "ZX": ("IX", 1), "ZY": ("IY", 1), "ZZ": ("ZZ", 1),
}

# Dense matrices (for state preparation and dense cross-checks).
_SQRT1_2 = 1.0 / np.sqrt(2.0)
_H_MAT = _SQRT1_2 * np.array([[1, 1], [1, -1]], dtype=complex)
_S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)
_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ_MAT = np.diag([1, 1, 1, -1]).astype(complex)


def _invert_rules(fwd: Dict[str, Tuple[str, int]]) -> Dict[str, Tuple[str, int]]:
    """Given forward rules P->(Q,s), build inverse rules Q->(P,s)."""
    inv: Dict[str, Tuple[str, int]] = {}
    for p, (q, s) in fwd.items():
        if q in inv:
            raise ValueError("forward rule is not a bijection")
        inv[q] = (p, s)
    return inv


class Gate:
    """A Clifford gate: name, the qubits it acts on, its rules and matrix."""

    __slots__ = ("name", "qubits", "rules", "matrix")

    def __init__(self, name, qubits, rules, matrix):
        self.name = name
        self.qubits = list(qubits)
        self.rules = rules
        self.matrix = matrix

    def to_dict(self):
        return {"name": self.name, "qubits": list(self.qubits)}


def H(q):
    return Gate(f"H({q})", [q], _H_RULES, _H_MAT)


def S(q):
    return Gate(f"S({q})", [q], _S_RULES, _S_MAT)


def Sdg(q):
    return Gate(f"Sdg({q})", [q], _SDG_RULES, _SDG_MAT)


def CNOT(c, t):
    return Gate(f"CNOT({c},{t})", [c, t], _CNOT_RULES, _CNOT_MAT)


def CZ(a, b):
    return Gate(f"CZ({a},{b})", [a, b], _CZ_RULES, _CZ_MAT)


_NAME_TO_FACTORY = {"H": H, "S": S, "Sdg": Sdg, "CNOT": CNOT, "CZ": CZ}


def gate_from_dict(d) -> Gate:
    name = d["name"]
    head = name.split("(")[0]
    return _NAME_TO_FACTORY[head](*d["qubits"])


def random_circuit(n: int, depth: int, rng, two_qubit: bool = True) -> List[Gate]:
    """A random Clifford circuit: per layer a random 1q gate on each qubit
    plus a brick-wall of CNOT/CZ on neighbouring qubits."""
    gates: List[Gate] = []
    for layer in range(depth):
        for q in range(n):
            c = rng.integers(0, 5)
            if c == 0:
                gates.append(H(q))
            elif c == 1:
                gates.append(S(q))
            elif c == 2:
                gates.append(Sdg(q))
            elif c == 3:
                gates.append(H(q)); gates.append(S(q))
            else:
                gates.append(S(q)); gates.append(H(q))
        if two_qubit:
            for i in range(layer % 2, n - 1, 2):
                if rng.random() < 0.7:
                    gates.append(CNOT(i, i + 1) if rng.random() < 0.5 else CZ(i, i + 1))
    return gates


def conjugate_pauli_string(
    gates: Sequence[Gate], pauli_str: str, coeff: float = 1.0, dagger: bool = False
) -> Tuple[str, float]:
    """
    Conjugate a Pauli string by the circuit.

    ``dagger=False`` -> ``U P U^\\dagger`` ; ``dagger=True`` -> ``U^\\dagger P U``.
    Returns ``(new_pauli_str, coeff * sign)``.
    """
    current = list(pauli_str)
    sign = 1
    order = list(gates) if dagger else list(reversed(gates))
    for g in order:
        rules = _invert_rules(g.rules) if dagger else g.rules
        local = "".join(current[q] for q in g.qubits)
        new_local, s = rules[local]
        sign *= s
        for i, q in enumerate(g.qubits):
            current[q] = new_local[i]
    return "".join(current), coeff * sign


def conjugate_pauli_sum(
    terms: Dict[str, float], gates: Sequence[Gate], dagger: bool = False, tol: float = 1e-12
) -> Dict[str, float]:
    """Conjugate a whole Pauli sum ``{string: coeff}`` by the circuit."""
    out: Dict[str, float] = {}
    for p, c in terms.items():
        np_, nc = conjugate_pauli_string(gates, p, c, dagger=dagger)
        out[np_] = out.get(np_, 0.0) + nc
        if abs(out[np_]) <= tol:
            del out[np_]
    return out


def _apply_gate_to_state(psi: np.ndarray, G: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    d_g = 2 ** len(qubits)
    env = [q for q in range(n) if q not in qubits]
    perm = list(qubits) + env
    inv_perm = list(np.argsort(perm))
    arr = psi.reshape([2] * n).transpose(perm).reshape(d_g, -1)
    arr = (G @ arr).reshape([2] * n).transpose(inv_perm)
    return arr.reshape(-1)


def apply_clifford(gates: Sequence[Gate], psi: np.ndarray, dagger: bool = False) -> np.ndarray:
    """
    Apply ``U`` (or ``U^\\dagger``) to a state vector, where
    ``U = G_1 @ ... @ G_m``.

    ``U |psi>`` applies gates right-to-left, i.e. the gate list in reverse;
    ``U^\\dagger |psi>`` applies the conjugate-transpose gates in forward order.
    """
    n = int(round(np.log2(psi.size)))
    if dagger:
        for g in gates:
            psi = _apply_gate_to_state(psi, g.matrix.conj().T, g.qubits, n)
    else:
        for g in reversed(gates):
            psi = _apply_gate_to_state(psi, g.matrix, g.qubits, n)
    return psi


def circuit_to_dict(gates: Sequence[Gate], n: int) -> dict:
    return {
        "convention": "U = product(gates, left_to_right); H' = U H U^dagger; ground state = U|psi>",
        "n_qubits": n,
        "depth": len(gates),
        "gates": [g.to_dict() for g in gates],
    }


def circuit_from_dict(d: dict) -> List[Gate]:
    return [gate_from_dict(g) for g in d["gates"]]
