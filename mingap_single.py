#!/usr/bin/env python3
"""
mingap_single.py -- adiabatic minimum-gap probe for one instance.

Replacement for the original ``minGapFind_v3_single.py``.  Improvements:

* The minimum gap is taken over a dense grid AND refined by golden-section
  search around the discrete minimum, which reduces the chance of missing a
  narrow avoided crossing. This is a heuristic, not a certified global
  minimizer: a crossing narrower than the grid spacing can still be missed.
  Increase the grid resolution to lower that risk.
* The located minimum is classified (interior / endpoint / closing) and
  the gap is measured above the ground manifold (degeneracy-aware).
* The problem Hamiltonian is centered (its identity/constant component
  removed) and rescaled by its spectral width by default, so the (1-s)/s
  interpolation is not distorted by the large constant energy offset that
  accumulates when many clauses are summed.

This is the gap of ``H(s) = (1-s)(-sum X_i) + s H_problem``; it is an
adiabatic-hardness probe, NOT the manuscript's spectral gap Delta_1 (use
``spectrum_single.py`` for that).

On failure it writes ``error_<stem>.log`` and exits non-zero.

Usage:
    python mingap_single.py BASENAME
    python mingap_single.py path/to_instance.json
"""

import json
import os
import sys
import traceback


def _resolve(arg):
    if arg.endswith("_instance.json") or arg.endswith(".json"):
        return arg, os.path.basename(arg).replace("_instance.json", "").replace(".json", "")
    return f"{arg}_instance.json", os.path.basename(arg)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path, stem = _resolve(argv[1])
    try:
        from planted_benchmark import adiabatic_min_gap

        with open(path) as f:
            instance = json.load(f)
        result = adiabatic_min_gap(instance)
        out_path = f"{stem}_mingap.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{stem}: gap_min={result['gap_min']:.6e} at s={result['s_min']:.4f} "
              f"({result['location']})  gap(s=1)={result['gap_at_s1']:.6e}  -> {out_path}")
        return 0
    except Exception:
        with open(f"error_{stem}.log", "w") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(f"FAILED {stem}; see error_{stem}.log\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
