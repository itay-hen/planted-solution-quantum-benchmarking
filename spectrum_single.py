#!/usr/bin/env python3
"""
spectrum_single.py -- degeneracy-aware ground-gap analysis for one instance.

Replacement for the original ``spectrum_analysis_single.py``.  It reports
both the raw ``E1 - E0`` (the manuscript's Delta_1, which is ~0 on a
degenerate ground space) and the gap above the whole ground manifold,
plus the ground degeneracy -- so a degenerate instance is no longer
silently reported as gapless without context.

On failure it writes ``error_<stem>.log`` and exits non-zero, instead of
exiting silently.

Usage:
    python spectrum_single.py BASENAME      # reads BASENAME_instance.json
    python spectrum_single.py path/to_instance.json
"""

import json
import os
import sys
import traceback


def _resolve(arg):
    if arg.endswith("_instance.json") or arg.endswith(".json"):
        path = arg
        stem = os.path.basename(arg).replace("_instance.json", "").replace(".json", "")
    else:
        path = f"{arg}_instance.json"
        stem = os.path.basename(arg)
    return path, stem


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path, stem = _resolve(argv[1])
    try:
        from planted_benchmark import ground_gap

        with open(path) as f:
            instance = json.load(f)
        result = ground_gap(instance)
        out_path = f"{stem}_spectrum.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{stem}: E0={result['E0']:.8f}  raw_gap={result['raw_gap_E1_minus_E0']:.3e}  "
              f"gap>manifold={result['gap_above_ground_manifold']:.3e}  "
              f"deg={result['ground_degeneracy']}  -> {out_path}")
        return 0
    except Exception:
        with open(f"error_{stem}.log", "w") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(f"FAILED {stem}; see error_{stem}.log\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
