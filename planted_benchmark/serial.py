"""(De)serialization of complex-valued arrays for JSON certification keys."""

from __future__ import annotations

import numpy as np


def vec_to_dict(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=complex)
    return {"re": v.real.tolist(), "im": v.imag.tolist()}


def dict_to_vec(d: dict) -> np.ndarray:
    re = np.asarray(d.get("re", d.get("amplitudes_re")), dtype=float)
    im = np.asarray(d.get("im", d.get("amplitudes_im")), dtype=float)
    return re + 1j * im


def mat_to_dict(M: np.ndarray) -> dict:
    M = np.asarray(M, dtype=complex)
    return {"re": M.real.tolist(), "im": M.imag.tolist()}


def dict_to_mat(d: dict) -> np.ndarray:
    re = np.asarray(d["re"], dtype=float)
    im = np.asarray(d["im"], dtype=float)
    return re + 1j * im
