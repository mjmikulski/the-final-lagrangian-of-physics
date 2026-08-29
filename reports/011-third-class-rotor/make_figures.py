"""Report 011 figures from committed artifacts.

fig_statics.png    : (a) axial line-tension excess vs box size;
                     (b) the delta comparison of the axial cost.
fig_survival.png   : (a) inertia excess before/after relaxation for
                     the two core-deformation types (symlog: the
                     collapse and its sign are visible);
                     (b) localization (PR) of the excess density.
fig_centrifugal.png: (a) the threshold jump of I(J);
                     (b) the localization of the rebuilt deformation.
Readability: one Opus round applied (labels inside axes, axisbelow,
symlog for the collapse, deuteranopia-safe line styles, margins).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
an = json.load(open(os.path.join(R, "analysis.json")))
ra = json.load(open(os.path.join(R, "relax_all.json")))
tw = json.load(open(os.path.join(R, "frame_twist.json")))
rs = json.load(open(os.path.join(R, "rot_stabilization.json")))


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


Ls = [24, 36, 48]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax1.semilogy(Ls, an["lam_excess_EQ"], color="#2166ac", marker="D",
             ms=7, mfc="none", mew=1.6, lw=1.4, ls="-", label="EQ")
ax1.semilogy(Ls, an["lam_excess_CB"], color="#7b3294", marker="^",
             ms=6, lw=1.4, ls="--", label="CB")
ax1.set_xticks(Ls)
ax1.margins(y=0.15)
ax1.set_xlabel(r"box size $L$  [lattice units]")
ax1.set_ylabel(r"$\lambda_z - \lambda_x$  [lattice units / length]")
ax1.set_title("(a) axial line-tension excess SHRINKS with $L$\n"
              "(relative to background: 21% $\\to$ 14% $\\to$ 2%)",
              fontsize=9.5)
ax1.legend(fontsize=8.5, framealpha=1.0)
ax1.grid(alpha=0.25, which="both")
ax1.set_axisbelow(True)

dd = an["axial_cost_delta"]
vals = [dd["d0300"], dd["d0125"]]
bars = ax2.bar([0, 1], vals, color=["#2166ac", "#1b7837"], width=0.5)
ax2.axhline(0, color="0.4", lw=0.9)
ax2.set_xticks([0, 1])
ax2.set_xticklabels([r"$\delta = 0.3$", r"$\delta = 1/8$"])
ax2.set_ylim(-3.4e-3, 1.1e-3)
for b, v in zip(bars, vals):
    if v >= 0:
        ax2.text(b.get_x() + b.get_width() / 2, v + 1e-4, sci(v),
                 ha="center", va="bottom", fontsize=8.5, color="0.25")
    else:
        ax2.text(b.get_x() + b.get_width() / 2, v - 1e-4, sci(v),
                 ha="center", va="top", fontsize=8.5, color="0.25")
ax2.set_ylabel(r"$\lambda_z - \lambda_x$  at $L = 48$")
ax2.set_title("(b) the axial cost falls with $\\delta$\n"
              "(negative at $1/8$: the axis is cheaper than the "
              "radial background)", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_statics.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_statics.png")

fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
cases = [("CB spectral\n($N{=}24$)",
          ra["cases"]["CB_N24"]["pre"]["I_comb"]
          - ra["cases"]["EQ_N24"]["pre"]["I_comb"],
          ra["cases"]["CB_N24"]["post"]["I_comb"]
          - ra["cases"]["EQ_N24"]["post"]["I_comb"]),
         ("CB2 frame twist\n($N{=}24$)",
          tw["cases"]["CB2"]["pre"]["I_comb"]
          - tw["cases"]["EQ"]["pre"]["I_comb"],
          tw["I_diff_raw"])]
x = range(len(cases))
w = 0.35
b1 = bx1.bar([i - w / 2 for i in x], [c[1] for c in cases], w,
             color="#999999", label="seed (before relax)")
b2 = bx1.bar([i + w / 2 for i in x], [c[2] for c in cases], w,
             color="#1b7837", label="after relax")
bx1.set_yscale("symlog", linthresh=10)
bx1.set_ylim(-12, 900)
bx1.axhline(0, color="0.4", lw=0.9)
for bar, v in zip(list(b1) + list(b2),
                  [c[1] for c in cases] + [c[2] for c in cases]):
    off = 0.12 if v >= 0 else -0.12
    bx1.text(bar.get_x() + bar.get_width() / 2,
             v * 1.25 if v > 10 else v + off * 10,
             f"{v:+.1f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=8,
             color="0.2")
bx1.set_xticks(list(x))
bx1.set_xticklabels([c[0] for c in cases], fontsize=9)
bx1.set_ylabel(r"$I_{\rm comb} - I_{\rm comb}^{\rm EQ}$  [lattice units]")
bx1.set_title("(a) the core deformations collapse under statics\n"
              "(symlog; sign visible)", fontsize=9.5)
bx1.legend(fontsize=8, framealpha=1.0, loc="lower left")
bx1.grid(alpha=0.25, axis="y", which="both")
bx1.set_axisbelow(True)

pr_pre = [ra["cases"]["CB_N24"]["pre"]["PR"],
          tw["cases"]["CB2"]["pre"]["PR"]]
pr_post = [ra["cases"]["CB_N24"]["post"]["PR"], tw["PR_excess"]]
p1 = bx2.bar([i - w / 2 for i in x], pr_pre, w, color="#999999",
             label="seed")
p2 = bx2.bar([i + w / 2 for i in x], pr_post, w, color="#7b3294",
             label="after relax (excess density)")
bx2.set_yscale("log")
bx2.set_ylim(20, 6e4)
for bar, v in zip(list(p1) + list(p2), pr_pre + pr_post):
    bx2.text(bar.get_x() + bar.get_width() / 2, v * 1.25, f"{v:.0f}",
             ha="center", fontsize=8, color="0.2")
bx2.set_xticks(list(x))
bx2.set_xticklabels([c[0] for c in cases], fontsize=9)
bx2.set_ylabel("PR  [sites]")
bx2.set_title("(b) the surviving excess is diffuse\n"
              "(seed vs post-relax)", fontsize=9.5)
bx2.legend(fontsize=8, framealpha=1.0, loc="upper right")
bx2.grid(alpha=0.25, axis="y", which="both")
bx2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_survival.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_survival.png")

rows = {r["J"]: r for r in rs["rows"]}
Js = [0.0, 0.4, 0.8, 4.0]
fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
cx1.plot(Js, [rows[j]["I"] for j in Js], color="#1b7837", marker="o",
         ms=7, lw=1.6)
cx1.axvline(4.0, color="0.5", ls=":", lw=1.0)
cx1.annotate("threshold estimate\n$J_{\\rm thr}\\sim\\sqrt{2I\\,"
             "\\Delta E_{\\rm def}}\\approx 4$", xy=(3.95, 300),
             xytext=(0.6, 470), fontsize=8.5, color="0.3",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9,
                             shrinkA=6, shrinkB=4))
cx1.set_xlim(-0.15, 4.3)
cx1.margins(y=0.12)
cx1.set_xlabel(r"prescribed $J$  [lattice units]")
cx1.set_ylabel(r"$I$ after minimization  [lattice units]")
cx1.set_title("(a) centrifugal stabilization:\nthe inertia jumps at "
              "threshold", fontsize=10)
cx1.grid(alpha=0.25)
cx1.set_axisbelow(True)

exJ = [0.4, 0.8, 4.0]
exPR = [rs["excess_J0.4"]["PR_excess"], rs["excess_J0.8"]["PR_excess"],
        rs["excess_J4.0"]["PR_excess"]]
cx2.semilogy(exJ, exPR, color="#7b3294", marker="^", ms=7, lw=1.6)
for j, v in zip(exJ, exPR):
    cx2.text(j, v * 1.3, f"{v:.0f}", ha="center", fontsize=8.5,
             color="0.2")
cx2.axhline(200, color="0.6", ls="--", lw=1.0)
cx2.text(4.25, 210, "PR = 200", fontsize=8, color="0.35", ha="right",
         va="bottom")
cx2.text(0.45, 230, "localized below this line", fontsize=8.5,
         color="0.35")
cx2.set_xlim(-0.15, 4.3)
cx2.set_ylim(60, 3500)
cx2.minorticks_off()
cx2.set_xlabel(r"prescribed $J$  [lattice units]")
cx2.set_ylabel("PR of excess density  [sites]")
cx2.set_title("(b) the rebuilt deformation is core-localized\n"
              "(diffuse noise at small $J$, PR = 137 at $J = 4$)",
              fontsize=10)
cx2.grid(alpha=0.25, which="major")
cx2.set_axisbelow(True)
fig3.tight_layout()
fig3.savefig(os.path.join(R, "fig_centrifugal.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_centrifugal.png")
