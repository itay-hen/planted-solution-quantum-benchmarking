"""Figure 1 (construction pipeline), rebuilt for the revision.
Output: ../figs/fig_construction_pipeline.pdf (+ .png preview)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7.5})
C_BLOCK = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
C_CL = ["#CC6677", "#44AA99", "#882255", "#999933"]
TXT, SUB = "#222222", "#555555"

W, Hh = 7.05, 2.05
fig = plt.figure(figsize=(W, Hh))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 29)
ax.axis("off")

PW, GAP, X0, Y0, PH = 17.5, 2.6, 0.7, 0.6, 27.6
xs = [X0 + i * (PW + GAP) for i in range(5)]
titles = ["(a) Partition\nand plant", "(b) Clause\nsupports", "(c) Local\nclauses",
          "(d) Assemble\nPauli sum", "(e) Optional\nClifford"]
for x, t in zip(xs, titles):
    ax.add_patch(FancyBboxPatch((x, Y0), PW, PH, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#F5F5F5", ec="#BBBBBB", lw=0.6))
    ax.text(x + PW / 2, Y0 + PH - 2.6, t, ha="center", va="center", fontsize=7.4, linespacing=1.05,
            fontweight="bold", color=TXT)
for i in range(4):
    ax.add_patch(FancyArrowPatch((xs[i] + PW + 0.3, 15), (xs[i + 1] - 0.3, 15),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#555555"))

# (a) blocks with qubits
x = xs[0]
sizes = [2, 1, 2, 2, 2]
bx = x + 1.0
for i, s in enumerate(sizes):
    w = 0.9 + 1.05 * s
    ax.add_patch(FancyBboxPatch((bx, 14.6), w, 4.0, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=C_BLOCK[i], alpha=0.25, ec=C_BLOCK[i], lw=0.6))
    for j in range(s):
        ax.plot(bx + 0.97 + 1.05 * j, 17.3, "o", ms=2.3, color=C_BLOCK[i])
    ax.text(bx + w / 2, 15.5, rf"$A_{i + 1}$", ha="center", va="center", fontsize=6.8, color=C_BLOCK[i])
    bx += w + 0.3
ax.text(x + PW / 2, 20.3, r"disjoint blocks, $|A_i|\leq a$", ha="center", fontsize=6.6, color=SUB)
ax.text(x + PW / 2, 11.6, r"$|\psi_{A_i}\rangle\in\mathcal{H}_{A_i}$ (Haar)", ha="center", fontsize=7.2, color=TXT)
ax.text(x + PW / 2, 6.6, r"$|\Psi_\star\rangle=\bigotimes_{i=1}^{M}|\psi_{A_i}\rangle$", ha="center",
        fontsize=7.6, color=TXT)

# (b) incidence grid: rows = clauses k, columns = blocks i
x = xs[1]
S = [[1, 2, 4], [2, 3, 5], [1, 3, 5], [3, 4, 5]]
cw, ch = 2.1, 1.95
gx0, gy0 = x + 4.6, 19.2
for i in range(5):
    ax.text(gx0 + i * cw + cw / 2, gy0 + 0.9, rf"${i + 1}$", ha="center", fontsize=6.4, color=C_BLOCK[i])
for k, Sk in enumerate(S):
    y = gy0 - (k + 1) * ch
    ax.text(gx0 - 0.5, y + ch / 2, rf"$S_{k + 1}$", ha="right", va="center", fontsize=6.8, color=C_CL[k])
    for i in range(5):
        on = (i + 1) in Sk
        ax.add_patch(Rectangle((gx0 + i * cw + 0.12, y + 0.12), cw - 0.24, ch - 0.24,
                               fc=C_CL[k] if on else "white", alpha=0.75 if on else 1, ec="#BBBBBB", lw=0.4))
ax.text(gx0 - 0.5, gy0 + 0.9, r"$i\!:$", ha="right", fontsize=6.4, color=SUB)
ax.text(x + PW / 2, 8.2, r"$S_k\subseteq\{1,\dots,M\}$, $|S_k|\leq q$", ha="center", fontsize=7.0, color=TXT)
ax.text(x + PW / 2, 4.2, r"$\Lambda_k=\bigcup_{i\in S_k}A_i$, $|\Lambda_k|\leq r$", ha="center",
        fontsize=7.0, color=TXT)

# (c) local spectra
x = xs[2]
for k in range(4):
    cx = x + 2.2 + k * 3.85
    ax.plot([cx, cx + 2.6], [11.9, 11.9], color=C_CL[k], lw=2.0, solid_capstyle="butt")
    for yy in (13.5, 14.3, 15.0, 15.9, 16.6, 17.4):
        ax.plot([cx, cx + 2.6], [yy, yy], color=C_CL[k], lw=0.5, alpha=0.6)
    ax.text(cx + 1.3, 19.0, rf"$H^{{({k + 1})}}$", ha="center", fontsize=6.8, color=C_CL[k])
ax.text(x + PW / 2, 9.6, r"ground: $|\psi_{\Lambda_k}\rangle=\otimes_{i\in S_k}|\psi_{A_i}\rangle$",
        ha="center", fontsize=6.4, color=TXT)
ax.text(x + PW / 2, 6.5, r"$H^{(k)}=\Sigma_r\,\lambda^{(k)}_r|\phi^{(k)}_r\rangle\langle\phi^{(k)}_r|$",
        ha="center", fontsize=6.6, color=TXT)
ax.text(x + PW / 2, 3.1, r"$\lambda^{(k)}_0=\mathrm{min}_r\,\lambda^{(k)}_r$",
        ha="center", fontsize=6.8, color=TXT)

# (d) assemble
x = xs[3]
ax.text(x + PW / 2, 18.4, r"$H=\Sigma_{k=1}^{K}\,\alpha_k H^{(k)}$", ha="center", fontsize=7.8, color=TXT)
ax.text(x + PW / 2, 14.6, r"$=\Sigma_{\ell=1}^{L}\,c_\ell P_\ell$", ha="center", fontsize=7.8, color=TXT)
ax.text(x + PW / 2, 10.6, r"$\alpha_k>0,\ \ K\leq\mathrm{poly}(n)$", ha="center", fontsize=6.9, color=TXT)
ax.text(x + PW / 2, 6.8, r"$L\leq K\,4^{r}=\mathrm{poly}(n)$", ha="center", fontsize=6.9, color=TXT)
ax.text(x + PW / 2, 3.1, r"$E_0=\Sigma_k\,\alpha_k\lambda^{(k)}_0$ (key)", ha="center", fontsize=6.9, color=TXT)

# (e) Clifford
x = xs[4]
ax.text(x + PW / 2, 18.4, r"$H'=VHV^\dagger$", ha="center", fontsize=8.0, color=TXT)
ax.text(x + PW / 2, 15.2, r"support-scrambled", ha="center", fontsize=6.9, color=TXT)
ax.text(x + PW / 2, 12.6, r"Pauli sum", ha="center", fontsize=6.9, color=TXT)
ax.text(x + PW / 2, 8.9, r"same spectrum and", ha="center", fontsize=6.6, color=SUB)
ax.text(x + PW / 2, 6.5, r"term-count bound", ha="center", fontsize=6.6, color=SUB)
ax.text(x + PW / 2, 3.1, r"target $V|\Psi_\star\rangle$ (key)", ha="center", fontsize=6.9, color=TXT)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_construction_pipeline.pdf"))
fig.savefig(os.path.join(OUT, "fig_construction_pipeline.png"), dpi=220)
print("saved")
