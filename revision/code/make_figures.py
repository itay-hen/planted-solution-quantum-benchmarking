"""
Figures and tables for the revision (all regenerated from ../data).
  figs/fig_coverage.pdf   <- data/coverage.json
  figs/fig_gap.pdf        <- data/gap_scan.jsonl
  data/tab_*.tex          <- LaTeX table bodies used in qb_gs_v22.tex
  data/numbers.json       <- scalar numbers quoted in the text
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbcore as pb  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
    "axes.labelsize": 8.5, "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#2B6CB0", "#DD6B20", "#2F855A", "#C53030", "#6B46C1", "#4A5568"
NUMBERS = {}


def load_numbers():
    p = os.path.join(DATA, "numbers.json")
    if os.path.exists(p):
        NUMBERS.update(json.load(open(p)))


def save_numbers():
    with open(os.path.join(DATA, "numbers.json"), "w") as f:
        json.dump(NUMBERS, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------
def fig_coverage():
    d = json.load(open(os.path.join(DATA, "coverage.json")))
    R = d["regimes"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    spec = [("fixed_K_20", r"$K=20$", GREY, "s", "-"),
            ("fixed_K_60", r"$K=60$", "#718096", "D", "-"),
            ("density_1", r"$K=M$", ORANGE, "v", "--"),
            ("density_2", r"$K=2M$", RED, "^", "--"),
            ("coverage_c0", r"$c=0$", BLUE, "o", "-."),
            ("coverage_c2", r"$c=2$", GREEN, "o", "-.")]
    maxdev = 0.0
    for key, lab, col, mk, ls in spec:
        rows = R[key]
        M = np.array([r["M"] for r in rows])
        th = np.array([r["P_full_theory"] for r in rows])
        mc = np.array([r["P_full_mc"] for r in rows])
        uh = np.array([r["P_graph_proxy_mc"] for r in rows])
        se = np.sqrt(np.maximum(th * (1 - th), 1e-12) / d["trials"])
        maxdev = max(maxdev, float(np.max(np.abs(mc - th) / np.maximum(se, 1.0 / d["trials"]))))
        ax.plot(M, th, ls=ls, color=col, lw=1.0, label=lab)
        ax.plot(M, mc, mk, color=col, ms=3.0, mew=0)
        ax.plot(M, uh, mk, color=col, ms=4.2, mfc="none", mew=0.6)
    for c, col in ((0, BLUE), (2, GREEN)):
        ax.axhline(math.exp(-math.exp(-c)), color=col, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"number of blocks $M$")
    ax.set_ylabel(r"$\Pr(N_{\rm unc}=0)$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(5.5, 215)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=2.0,
              columnspacing=1.0, borderaxespad=0.2)
    fig.savefig(os.path.join(FIGS, "fig_coverage.pdf"))
    fig.savefig(os.path.join(FIGS, "fig_coverage.png"), dpi=200)
    plt.close(fig)
    NUMBERS["coverage_max_dev_in_se"] = maxdev
    NUMBERS["coverage_trials"] = d["trials"]
    # Poisson check at largest M
    for key in ("coverage_c0", "coverage_c2"):
        r = R[key][-1]
        NUMBERS[f"{key}_M{r['M']}_theory"] = r["P_full_theory"]
    print("fig_coverage: max |MC-theory| / SE =", round(maxdev, 2))


# ---------------------------------------------------------------------------
def load_scan():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "gap_scan.jsonl"))]
    return rows


def fig_gap():
    rows = load_scan()
    by = defaultdict(list)
    for r in rows:
        by[(r["scheme"], r["n"], r["c"])].append(r)
    ns = sorted({r["n"] for r in rows})
    cs = sorted({r["c"] for r in rows})
    NUMBERS["scan_flags"] = int(sum(r["flag_kernel_larger_than_predicted"] for r in rows))
    NUMBERS["scan_rows"] = len(rows)
    NUMBERS["scan_instances"] = len({(r["scheme"], r["n"], r["idx"]) for r in rows})
    # monotonicity along nested sequences (restricted to consecutive verified-unique prefixes)
    seqs = defaultdict(list)
    for r in rows:
        seqs[(r["scheme"], r["n"], r["idx"])].append(r)
    viol, pairs = 0, 0
    worst = 0.0
    for k, s in seqs.items():
        s.sort(key=lambda r: r["K"])
        for a, b in zip(s, s[1:]):
            if a["struct_dim"] == 1 and b["struct_dim"] == 1:
                pairs += 1
                if b["gap_exc"] < a["gap_exc"] - 1e-8:
                    viol += 1
                    worst = max(worst, a["gap_exc"] - b["gap_exc"])
    NUMBERS["monotone_pairs"] = pairs
    NUMBERS["monotone_violations"] = viol
    NUMBERS["monotone_worst"] = worst

    fig, axs = plt.subplots(1, 3, figsize=(7.05, 2.35))
    cols = {0: BLUE, 1: "#63B3ED", 2: GREEN, 3: "#9AE6B4", 4: PURPLE}
    show_c = [0, 2, 4]
    counts = []
    for ax, scheme, title in ((axs[0], "bimodal", "(a) half-rank ensemble"),
                              (axs[1], "golf_course", "(b) projector ensemble")):
        for c in show_c:
            med, q1, q3, xs = [], [], [], []
            for n in ns:
                rs = [r for r in by[(scheme, n, c)] if r["struct_dim"] == 1]
                if not rs:
                    continue
                g = np.array([r["gap_exc"] for r in rs])
                xs.append(n)
                med.append(np.median(g))
                q1.append(np.percentile(g, 25))
                q3.append(np.percentile(g, 75))
                jit = (np.random.default_rng(n * 10 + c).random(len(g)) - 0.5) * 0.5
                ax.plot(n + 0.35 * (c - 2) / 2 + jit * 0.3, g, ".", ms=1.4, color=cols[c], alpha=0.35,
                        rasterized=True)
            ax.fill_between(xs, q1, q3, color=cols[c], alpha=0.18, lw=0)
            ax.plot(xs, med, "-o", color=cols[c], ms=2.6, lw=1.0, label=rf"$c={c}$")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$n$")
        ax.set_yscale("log")
        ax.set_xticks(ns)
    axs[0].set_ylabel(r"$\Delta_{\rm exc}$ (unique instances)")
    axs[0].legend(frameon=False, loc="lower left")
    # (c) fraction unique vs theory, and lambda per clause
    ax = axs[2]
    for c in (0, 2, 4):
        fr_b, fr_g, th = [], [], []
        for n in ns:
            rb = by[("bimodal", n, c)]
            rg = by[("golf_course", n, c)]
            fr_b.append(np.mean([r["struct_dim"] == 1 for r in rb]) if rb else np.nan)
            fr_g.append(np.mean([r["struct_dim"] == 1 for r in rg]) if rg else np.nan)
            th.append(pb.prob_full_coverage(n, 3, pb.coverage_scale_K(n, 3, c)))
        ax.plot(ns, th, "-", color=cols[c], lw=0.9)
        ax.plot(ns, fr_g, "o", color=cols[c], ms=2.8, mew=0)
        ax.plot(ns, fr_b, "o", color=cols[c], ms=4.0, mfc="none", mew=0.6)
        ax.axhline(math.exp(-math.exp(-c)), color=cols[c], lw=0.5, ls=":")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"fraction verified unique")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ns)
    ax.set_title("(c) uniqueness and one-norm", fontsize=8)
    ax2 = ax.twinx()
    for scheme, mk, col, lab in (("bimodal", "^", RED, "half-rank"), ("golf_course", "v", "#A0AEC0", "projector")):
        lam = [np.median([r["one_norm_nonid"] / r["K"] for r in rows
                          if r["scheme"] == scheme and r["n"] == n]) for n in ns]
        ax2.plot(ns, lam, mk + "--", color=col, ms=3.0, lw=0.9, label=lab)
    ax2.set_ylabel(r"$\lambda/K$", fontsize=8)
    ax2.set_ylim(0, 16)
    ax2.tick_params(labelsize=7)
    ax2.legend(frameon=False, loc="center", bbox_to_anchor=(0.52, 0.70), fontsize=6.2,
               title=r"$\lambda/K$ (right axis)", title_fontsize=6.2, handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGS, "fig_gap.pdf"), dpi=300)
    fig.savefig(os.path.join(FIGS, "fig_gap.png"), dpi=200)
    plt.close(fig)

    # table of counts and summary statistics
    lines = []
    for scheme in ("bimodal", "golf_course"):
        for n in ns:
            cells = []
            for c in cs:
                rs = by[(scheme, n, c)]
                u = [r for r in rs if r["struct_dim"] == 1]
                if not rs:
                    cells.append("--")
                    continue
                g = np.array([r["gap_exc"] for r in u]) if u else np.array([np.nan])
                cells.append(f"{len(u)}/{len(rs)}; {np.median(g):.3g}")
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_median"] = float(np.median(g))
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_q1"] = float(np.percentile(g, 25))
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_q3"] = float(np.percentile(g, 75))
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_min"] = float(np.min(g))
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_nunique"] = len(u)
                NUMBERS[f"gap_{scheme}_n{n}_c{c}_N"] = len(rs)
            K_list = [pb.coverage_scale_K(n, 3, c) for c in cs]
            lab = "half-rank" if scheme == "bimodal" else "projector"
            lines.append(f"{lab} & {n} & " + " & ".join(
                f"{K} & {cell}" for K, cell in zip(K_list, cells)) + r" \\")
        lines.append(r"\hline")
    with open(os.path.join(DATA, "tab_gapcounts.tex"), "w") as f:
        f.write("\n".join(lines[:-1]) + "\n")
    for scheme in ("bimodal", "golf_course"):
        lamK = [r["one_norm_nonid"] / r["K"] for r in rows if r["scheme"] == scheme]
        NUMBERS[f"lambda_per_clause_{scheme}_median"] = float(np.median(lamK))
        NUMBERS[f"lambda_per_clause_{scheme}_range"] = [float(np.min(lamK)), float(np.max(lamK))]
        NUMBERS[f"t_eig_max_{scheme}"] = float(max(r["t_eig"] for r in rows if r["scheme"] == scheme))
    print("fig_gap:", NUMBERS["scan_instances"], "instances;", pairs, "nested unique pairs,", viol, "violations;",
          NUMBERS["scan_flags"], "flags")


# ---------------------------------------------------------------------------
def tab_baseline():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "product_baseline.jsonl"))]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("block_size", 1), r["clifford"], r["n"])].append(r)
    lines = []
    names = {(1, False): r"$|A_i|=1$, no Clifford", (1, True): r"$|A_i|=1$, Clifford $D=5$",
             (2, False): r"$|A_i|=2$, no Clifford"}
    for key in ((1, False), (1, True), (2, False)):
        for n in sorted({k[2] for k in by if k[:2] == key}):
            rs = by[key + (n,)]
            solved = sum(r["success"] for r in rs)
            exc = np.median([r["rel_excess"] for r in rs])
            restart = np.mean([r["restart_success_frac"] for r in rs])
            tmed = np.median([r["t"] for r in rs])
            rst = sorted({r.get("restarts", 20) for r in rs})
            rst_s = str(rst[0]) if len(rst) == 1 else f"{rst[0]}--{rst[-1]}"
            lines.append(f"{names[key]} & {n} & {rs[0]['M']} & {rs[0]['K']} & {rst_s} & "
                         f"{solved}/{len(rs)} & {max(exc, 0):.2g} & {restart:.2f} & {tmed:.2g} \\\\")
            NUMBERS[f"baseline_bs{key[0]}_cl{int(key[1])}_n{n}_solved"] = [solved, len(rs)]
            NUMBERS[f"baseline_bs{key[0]}_cl{int(key[1])}_n{n}_excess"] = float(exc)
            NUMBERS[f"baseline_bs{key[0]}_cl{int(key[1])}_n{n}_tmed"] = float(tmed)
    with open(os.path.join(DATA, "tab_baseline.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("tab_baseline:", len(rows), "rows")


# ---------------------------------------------------------------------------
def tab_v21():
    rows = list(csv.DictReader(open(os.path.join(DATA, "v21_structural.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[(int(r["n"]), int(r["K"]))].append(r)
    lines = []
    for (n, K), rs in sorted(by.items()):
        N = len(rs)
        deg = sum(int(r["struct_dim"]) > 1 for r in rs)
        unc = sum(int(r["n_uncovered"]) > 0 for r in rs)
        iso = sum(int(r["single_support_components"]) > 0 and int(r["n_uncovered"]) == 0 for r in rs)
        new = sum(r["source"] != "original" for r in rs)
        agree = all(int(r["g_tol1e+03"]) == int(r["struct_dim"]) for r in rs)
        lines.append(f"{n} & {K} & {N} & {new} & {deg} & {unc} & {iso} & "
                     f"{pb.prob_full_coverage(n, 3, K):.3f} & {np.mean([int(r['n_uncovered']) == 0 for r in rs]):.2f} & "
                     f"{'yes' if agree else 'NO'} \\\\")
    with open(os.path.join(DATA, "tab_v21.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    NUMBERS["v21_N"] = len(rows)
    NUMBERS["v21_completed"] = sum(r["source"] != "original" for r in rows)
    for fac in ("1e+02", "1e+03", "1e+04", "1e+05", "1e+06"):
        NUMBERS[f"v21_agree_{fac}"] = sum(int(r[f"g_tol{fac}"]) == int(r["struct_dim"]) for r in rows)
    NUMBERS["v21_agree_abs"] = sum(int(r["g_abs1e-10"]) == int(r["struct_dim"]) for r in rows)
    NUMBERS["v21_max_E0_err"] = max(abs(float(r["E0_ed_minus_key"])) for r in rows)
    NUMBERS["v21_min_gap_unique"] = min(float(r["delta1_raw"]) for r in rows if int(r["struct_dim"]) == 1)
    NUMBERS["v21_degenerate"] = sum(int(r["struct_dim"]) > 1 for r in rows)
    NUMBERS["v21_isolated_only"] = sum(int(r["struct_dim"]) > 1 and int(r["n_uncovered"]) == 0 for r in rows)
    print("tab_v21:", len(rows), "instances")


if __name__ == "__main__":
    load_numbers()
    what = sys.argv[1:] or ["coverage", "gap", "baseline", "v21"]
    for w in what:
        try:
            {"coverage": fig_coverage, "gap": fig_gap, "baseline": tab_baseline, "v21": tab_v21}[w]()
        except FileNotFoundError as e:
            print("skip", w, e)
    save_numbers()
