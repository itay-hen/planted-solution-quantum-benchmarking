#!/usr/bin/env python3
"""
qbench -- command-line interface to the planted_benchmark package.

Examples
--------
    # generate a single-planted paper-style instance
    python qbench.py general --n 10 --clauses 14 --seed 7 --out inst10

    # the classical diagonal planted-SAT channel
    python qbench.py sat --n 12 --k 3 --clauses 30 --seed 1 --out sat12

    # verify and assess
    python qbench.py verify inst10
    python qbench.py assess inst10

    # spectral gap (Delta_1, degeneracy-aware) and adiabatic min-gap
    python qbench.py spectrum inst10
    python qbench.py mingap sat12

    # a whole benchmark suite
    python qbench.py batch --out suite --n 6 8 10 --clauses 6 14 22 --per-config 10

The CLI exposes the commonly-used parameters (including non-default alpha
and lambda ranges, Bernoulli sampling, support caps, and Clifford depth on
the `general` subcommand). The Python API in `planted_benchmark` exposes
the complete configuration surface; see the dataclasses in
`planted_benchmark.generator`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from planted_benchmark import (
    CommutingConfig,
    GeneralConfig,
    PlantedSATConfig,
    compute_assessment,
    generate_benchmark_suite,
    generate_commuting_instance,
    generate_general_instance,
    generate_planted_sat_instance,
    ground_gap,
    adiabatic_min_gap,
    print_summary,
    verify_instance,
)


def _write_pair(instance, key, out):
    with open(f"{out}_instance.json", "w") as f:
        json.dump(instance, f, indent=2)
    with open(f"{out}_key.json", "w") as f:
        json.dump(key, f, indent=2)
    print(f"wrote {out}_instance.json and {out}_key.json")
    print(f"  ground energy : {key['ground_energy']:.10f}")
    print(f"  pauli terms   : {instance['hamiltonian']['num_terms']}")


def _load_pair(args):
    if args.instance and args.key:
        inst_path, key_path = args.instance, args.key
    else:
        base = args.basename
        inst_path, key_path = f"{base}_instance.json", f"{base}_key.json"
    with open(inst_path) as f:
        instance = json.load(f)
    with open(key_path) as f:
        key = json.load(f)
    return instance, key


def main(argv=None):
    p = argparse.ArgumentParser(prog="qbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("general", help="generate a Haar-random planted instance")
    g.add_argument("--n", type=int, default=12)
    g.add_argument("--block-size", type=int, default=1)
    g.add_argument("--clauses", type=int, default=10)
    g.add_argument("--blocks-per-clause", type=int, default=3)
    g.add_argument("--num-planted", type=int, default=1)
    g.add_argument("--scheme", default="bimodal", choices=["bimodal", "golf_course", "uniform", "linear"])
    g.add_argument("--alpha-min", type=float, default=0.5)
    g.add_argument("--alpha-max", type=float, default=1.5)
    g.add_argument("--lambda-min", type=float, default=-1.0)
    g.add_argument("--lambda-max", type=float, default=1.0)
    g.add_argument("--spectral-gap", type=float, default=0.5)
    g.add_argument("--subset-sampling", default="uniform", choices=["uniform", "bernoulli"])
    g.add_argument("--bernoulli-prob", type=float, default=0.5)
    g.add_argument("--max-support-qubits", type=int, default=6)
    g.add_argument("--variable-block-sizes", type=int, nargs="*", default=None)
    g.add_argument("--clifford", action="store_true")
    g.add_argument("--clifford-depth", type=int, default=5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", required=True)

    s = sub.add_parser("sat", help="generate a diagonal planted-SAT instance")
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--clauses", type=int, default=20)
    s.add_argument("--weight", type=float, default=1.0)
    s.add_argument("--planted", type=int, nargs="*", default=None, help="planted assignment bits")
    s.add_argument("--clifford", action="store_true")
    s.add_argument("--clifford-depth", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)

    c = sub.add_parser("commuting", help="generate a commuting block-projector instance")
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--block-size", type=int, default=2)
    c.add_argument("--weight-min", type=float, default=0.5)
    c.add_argument("--weight-max", type=float, default=1.5)
    c.add_argument("--no-clifford", action="store_true")
    c.add_argument("--clifford-depth", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", required=True)

    b = sub.add_parser("batch", help="generate a benchmark suite")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12])
    b.add_argument("--clauses", type=int, nargs="+", default=[6, 10, 14, 18, 22, 26, 30])
    b.add_argument("--per-config", type=int, default=10)
    b.add_argument("--block-size", type=int, default=1)
    b.add_argument("--blocks-per-clause", type=int, default=3)
    b.add_argument("--scheme", default="bimodal")
    b.add_argument("--clifford", action="store_true")
    b.add_argument("--seed", type=int, default=1000)

    for name, helptext in [("verify", "verify an instance against its key"),
                           ("assess", "compute diagnostics"),
                           ("spectrum", "ground gap (Delta_1, degeneracy-aware)"),
                           ("mingap", "adiabatic minimum gap")]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("basename", nargs="?", help="basename -> <base>_instance.json/_key.json")
        sp.add_argument("--instance")
        sp.add_argument("--key")
        if name in ("assess", "spectrum", "mingap", "verify"):
            sp.add_argument("--max-n", type=int, default=12 if name != "verify" else 14)
        if name == "assess":
            sp.add_argument("--frustration", action="store_true")
        if name == "verify":
            sp.add_argument("--require-dense", action="store_true",
                            help="exit non-zero if dense checks were skipped (n > max-n)")
        sp.add_argument("--json-out")

    args = p.parse_args(argv)

    if args.cmd == "general":
        cfg = GeneralConfig(n_qubits=args.n, block_size=args.block_size, num_clauses=args.clauses,
                            blocks_per_clause=args.blocks_per_clause, num_planted=args.num_planted,
                            eigenvalue_scheme=args.scheme, alpha_min=args.alpha_min,
                            alpha_max=args.alpha_max, lambda_min=args.lambda_min,
                            lambda_max=args.lambda_max, spectral_gap=args.spectral_gap,
                            subset_sampling=args.subset_sampling, bernoulli_prob=args.bernoulli_prob,
                            max_support_qubits=args.max_support_qubits,
                            variable_block_sizes=args.variable_block_sizes,
                            clifford=args.clifford, clifford_depth=args.clifford_depth, seed=args.seed)
        _write_pair(*generate_general_instance(cfg), args.out)
    elif args.cmd == "sat":
        cfg = PlantedSATConfig(n_qubits=args.n, k=args.k, num_clauses=args.clauses,
                               weight=args.weight, planted_assignment=args.planted,
                               clifford=args.clifford, clifford_depth=args.clifford_depth,
                               seed=args.seed)
        _write_pair(*generate_planted_sat_instance(cfg), args.out)
    elif args.cmd == "commuting":
        cfg = CommutingConfig(n_qubits=args.n, block_size=args.block_size,
                              weight_min=args.weight_min, weight_max=args.weight_max,
                              clifford=not args.no_clifford, clifford_depth=args.clifford_depth,
                              seed=args.seed)
        _write_pair(*generate_commuting_instance(cfg), args.out)
    elif args.cmd == "batch":
        generate_benchmark_suite(output_dir=args.out, n_values=args.n, K_values=args.clauses,
                                 instances_per_config=args.per_config, block_size=args.block_size,
                                 blocks_per_clause=args.blocks_per_clause, eigenvalue_scheme=args.scheme,
                                 clifford=args.clifford, base_seed=args.seed)
    elif args.cmd == "verify":
        instance, key = _load_pair(args)
        rep = verify_instance(instance, key, max_n=args.max_n)
        print(f"instance {instance.get('instance_id')}  ->  {rep['status'].upper()}")
        for ch in rep["checks"]:
            print(f"  [{'ok' if ch['passed'] else 'XX'}] {ch['name']}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(rep, f, indent=2)
        if getattr(args, "require_dense", False) and not rep.get("fully_verified", False):
            print("  --require-dense set and dense checks were skipped -> failing")
            return 2
        return 0 if rep["all_passed"] else 1
    elif args.cmd == "assess":
        instance, key = _load_pair(args)
        a = compute_assessment(instance, key, max_n=args.max_n, compute_frustration=args.frustration)
        print_summary(a)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(a, f, indent=2)
    elif args.cmd == "spectrum":
        instance, _ = _load_pair(args)
        r = ground_gap(instance, dense_threshold_qubits=args.max_n)
        print(json.dumps(r, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(r, f, indent=2)
    elif args.cmd == "mingap":
        instance, _ = _load_pair(args)
        r = adiabatic_min_gap(instance, max_n=args.max_n)
        summary = {k: v for k, v in r.items() if k != "trace"}
        print(json.dumps(summary, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(r, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
