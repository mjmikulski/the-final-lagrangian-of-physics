"""Figure for APPENDIX-L-ladder: the (I1^G)^2 energy-functional well at three box sizes, from the committed JSON.

Left: E(omega) - E(0) per box (final protocol level), with the frozen-profile prediction omega_E of each box as a
tick on the axis. Right: the depth of the well relative to omega = 0 at each protocol level (Adam, then four
L-BFGS cycles) per box, i.e. the plateau check of report 008 section 6. Boxes are an ordered set, so one hue,
light to dark with L; each series is direct-labelled.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "L_ladder")
COLORS = {32: "#86b6ef", 48: "#2a78d6", 64: "#104281"}     # ordinal blue ramp (dataviz reference palette)
INK, MUTED = "#1a1a19", "#6b6a63"

ladders = {}
for N in (32, 48, 64):
    f = os.path.join(RES, f"ladder_N{N}.json")
    if os.path.exists(f):
        ladders[N] = json.load(open(f))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
for N, lad in ladders.items():
    L = lad["L"]
    om = [r["omega"] for r in lad["rungs"]]
    E0 = lad["rungs"][0]["E_total"]
    dE = [(r["E_total"] - E0) * 1e5 for r in lad["rungs"]]
    ax.plot(om, dE, "-", color=COLORS[N], lw=2, marker="o", ms=5, mec="white", mew=1.2, label=f"L = {L:g} (N = {N})")
    k = min(range(len(dE)), key=lambda i: dE[i])
    ax.annotate(f"L = {L:g}", (om[k], dE[k]), xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
    ax.axvline(lad["profile"]["omega_pred_E"], color=COLORS[N], lw=1, ls=":", ymin=0, ymax=0.08)
    levels = lad["depth_per_level"]
    bx.plot(range(len(levels)), [d * 1e5 for d in levels], "-", color=COLORS[N], lw=2, marker="o", ms=5,
            mec="white", mew=1.2, label=f"L = {L:g}")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.03, 0.55)
ymin = min(min((r["E_total"] - lad["rungs"][0]["E_total"]) * 1e5 for r in lad["rungs"] if r["omega"] <= 0.5) for lad in ladders.values())
ax.set_ylim(ymin * 1.6, 12)
ax.set_xlabel("ω (rung)")
ax.set_ylabel("E(ω) − E(0)  [×10⁻⁵]")
ax.set_title("JG_E ladder at fixed h = 1.5, γ = 70.6: the well per box", fontsize=10, loc="left")
ax.text(0.02, 0.04, "dotted ticks: frozen-profile ω_E per box", transform=ax.transAxes, color=MUTED, fontsize=8)
ax.legend(frameon=False, fontsize=9, loc="upper right")
bx.set_ylim(0, 1.15 * max(max(d for d in lad["depth_per_level"]) for lad in ladders.values()) * 1e5)
bx.set_xticks(range(5))
bx.set_xticklabels(["Adam", "L-BFGS 1", "L-BFGS 2", "L-BFGS 3", "L-BFGS 4"], fontsize=8)
bx.set_ylabel("well depth vs ω = 0  [×10⁻⁵]")
bx.set_title("depth of the well at each protocol level (plateau check)", fontsize=10, loc="left")
bx.legend(frameon=False, fontsize=9)
for a in (ax, bx):
    a.grid(True, color="#e6e5df", lw=0.6)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(RES, "fig_L_ladder.png")
fig.savefig(out)
print("written", out, "boxes:", sorted(ladders))
