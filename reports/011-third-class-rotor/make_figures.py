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
lp = json.load(open(os.path.join(R, "lambda_plateau.json")))
lp2 = json.load(open(os.path.join(R, "lambda_plateau2.json")))
cbj = json.load(open(os.path.join(R, "centrifugal_branches.json")))


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


Ls = [24, 36, 48]
plateau = [lp["cases"]["EQ_N16"]["trajectory"][-1]["excess"],
           lp2["cases"]["EQ_N24"]["final_excess"],
           lp2["cases"]["EQ_N32"]["final_excess"]]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax1.semilogy(Ls, plateau, color="#2166ac", marker="D", ms=7,
             mfc="none", mew=1.6, lw=1.4, ls="-",
             label="EQ, deep-continued plateau")
ax1.semilogy([48], [an["lam_excess_EQ"][2]], color="#c0392b",
             marker="x", ms=9, mew=2, ls="none",
             label="round-1 endpoint (unconverged)")
ax1.annotate("", xy=(48, plateau[2] * 0.85),
             xytext=(48, an["lam_excess_EQ"][2] * 1.5),
             arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
ax1.axhline(plateau[1], color="0.6", ls=":", lw=1.0)
ax1.text(24.5, plateau[1] * 1.15,
         r"constant line tension $\approx 7.6\cdot10^{-4}$",
         fontsize=8, color="0.35")
ax1.set_xticks(Ls)
ax1.margins(y=0.2)
ax1.set_xlabel(r"box size $L$  [lattice units]")
ax1.set_ylabel(r"$\lambda_z - \lambda_x$  [lattice units / length]")
ax1.set_title("(a) the axial line tension is CONSTANT for "
              "$L \\geq 36$\n(the round-1 shrinking claim was an "
              "artifact)", fontsize=9.5)
ax1.legend(fontsize=8, framealpha=1.0, loc="upper right")
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

B = cbj["branches"]
Js = [0.0, 2.0, 4.0, 6.0]
fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
cx1.plot(Js, [B[f"EQ_J{j}"]["I"] for j in Js], color="#2166ac",
         marker="D", ms=7, mfc="none", mew=1.6, lw=1.5, ls="-",
         label="EQ-start branch")
cx1.plot(Js, [B[f"CB_J{j}"]["I"] for j in Js], color="#7b3294",
         marker="^", ms=6, lw=1.4, ls="--", label="CB-start branch")
cx1.annotate("continuation $J{=}4 \\to 0$:\nthe inertia scalar returns",
             xy=(0.15, cbj["hysteresis"]["I"]),
             xytext=(1.35, 460), fontsize=8, color="0.3",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
cx1.plot([0.0], [cbj["hysteresis"]["I"]], marker="v", ms=7,
         color="#b8860b")
cx1.set_xlabel(r"prescribed $J$  [lattice units]")
cx1.set_ylabel(r"$I$ after minimization  [lattice units]")
cx1.set_title("(a) both branches grow inertia spontaneously\n"
              "(qualitative record; no branch-selection claim)",
              fontsize=9)
cx1.legend(fontsize=8, framealpha=1.0, loc="upper left")
cx1.grid(alpha=0.25)
cx1.set_axisbelow(True)

edges = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5]
prof = B["EQ_J4.0"]["excess_shell_profile"]
cx2.bar(edges, prof, width=2.5, color="#1b7837")
cx2.axvline(B["EQ_J4.0"]["excess_centroid_r"], color="0.35", ls=":",
            lw=1.2)
cx2.text(9.6, max(prof) * 0.86, "centroid\n$r = 14.7$",
         fontsize=8, color="0.3", ha="right")
cx2.annotate("", xy=(B["EQ_J4.0"]["excess_centroid_r"] - 0.15,
                     max(prof) * 0.84),
             xytext=(9.8, max(prof) * 0.84),
             arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))
cx2.axvline(18.0, color="#c0392b", ls="--", lw=1.1)
cx2.text(17.6, max(prof) * 0.55, "boundary", fontsize=8,
         color="#c0392b", ha="right", rotation=90)
cx2.set_xlabel(r"shell radius $r$  [lattice units]")
cx2.set_ylabel("inertia-excess per shell  [lattice units]")
cx2.set_title("(b) the excess lives at the PERIPHERY\n"
              "(EQ-start, $J = 4$; core shells empty)", fontsize=9.5)
cx2.grid(alpha=0.25, axis="y")
cx2.set_axisbelow(True)
fig3.tight_layout()
fig3.savefig(os.path.join(R, "fig_centrifugal.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_centrifugal.png")
