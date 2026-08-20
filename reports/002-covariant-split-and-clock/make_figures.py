"""Figures for report 002 (data: proto_results.json, clock_results.json).

Layout follows the measured readability review (2026-08-20): fig A is
cropped to the Mexican-hat core with verdict-carrying direct labels and
no legend; fig B keeps annotations strictly below all bars and the
legend above the axes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, aqua
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "figure.dpi": 160,
})

clock = json.load(open(os.path.join(HERE, "results", "clock_results.json")))
proto = json.load(open(os.path.join(HERE, "results", "proto_results.json")))

# ---- Fig A: E(omega) for the three functionals on one clock texture ----
fam = clock["family_curves"]
om = np.array(fam["omega"])
ce = clock["B3_counterexample"]

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(om, fam["condensate"], color=C3, lw=2.6, zorder=2)
ax.plot(om, fam["G"], color=C2, lw=2, zorder=2)
ax.plot(om, fam["eta"], color=C1, lw=2, zorder=3)
ax.plot([ce["omega_star"]], [ce["E_min"]], "o", color=C3, ms=9, zorder=4)

ax.text(1.02, -7.2, "current ($\\eta$): runaway", color=C1, fontsize=9,
        ha="right", va="center")
ax.text(1.55, -12.25, "$\\downarrow$ unbounded", color=C1, fontsize=8,
        ha="right", va="center")
ax.text(1.15, 9.8, "all-$G$: clock dies ($\\omega^*=0$)", color=C2,
        fontsize=9, ha="right", va="center")
ax.text(2.42, 6.5, "boost condensate:\nfinite clock", color=C3,
        fontsize=9, ha="right", va="center")
ax.text(1.70, -5.8, "$\\omega^*=1.60$ (theory 1.58)\n$E(\\omega^*)<E(0)$",
        fontsize=9, ha="left", va="top", color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(0, 2.8)
ax.set_ylim(-13, 13)
ax.set_xlabel("$\\omega$  (clock speed)")
ax.set_ylabel("energy density  [model units]")
ax.set_title("One clock texture, three kinetic terms", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "figA_clock_family.png"))
plt.close(fig)

# ---- Fig B: kin channel table, eta vs G --------------------------------
kt = proto["kin_table"]
names = ["rot_xy", "rot_xz", "rot_yz", "boost_x", "boost_y", "boost_z"]
labels = ["rot xy", "rot xz", "rot yz", "boost x", "boost y", "boost z"]
v_eta = [kt[n]["eta"] for n in names]
v_G = [kt[n]["G"] for n in names]

fig, ax = plt.subplots(figsize=(6.8, 4.0))
x = np.arange(6)
w = 0.36
b1 = ax.bar(x - w / 2 - 0.015, v_eta, w, color=C1,
            label="current metric $\\eta$")
b2 = ax.bar(x + w / 2 + 0.015, v_G, w, color=C2,
            label="Euclideanizer $G$ (covariant)")
# rotations: one centered label per pair (identical numbers -- saying it
# once IS the message); boosts: per-bar, the sign is the point there
for i in range(3):
    ax.text(i, v_eta[i] + 1.8, f"{v_eta[i]:+.1f}", ha="center",
            va="bottom", fontsize=8, color=INK)
for bars, vals in ((b1, v_eta), (b2, v_G)):
    for i, (r, v) in enumerate(zip(bars, vals)):
        if i < 3:
            continue
        ax.text(r.get_x() + r.get_width() / 2, v + (1.8 if v > 0 else -1.8),
                f"{v:+.1f}".replace("-", "−"), ha="center",
                va="bottom" if v > 0 else "top", fontsize=8, color=INK)
ax.axhline(0, color=INK, lw=1.0)
ax.axvline(2.62, color=GRID, lw=1.0)
ax.text(1.0, -68, "rotations: identical under $\\eta$ and $G$\n"
        "(Coulomb sector untouched)", ha="center", va="top",
        fontsize=8.5, color=INK2)
ax.text(4.35, -68, "boosts: sign flipped, magnitude kept\n"
        "(runaway channel closed)", ha="center", va="top",
        fontsize=8.5, color=INK2)
ax.set_xticks(x, labels, fontsize=9)
ax.set_ylim(-84, 62)
ax.set_ylabel("kin($M$; $a_0$)  [$\\omega^2$ coefficient]")
ax.set_title("Clock-channel table: the one-line covariant boundedness fix",
             fontsize=11, pad=30)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005),
          ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "figB_kin_channels.png"))
plt.close(fig)

print("written: figA_clock_family.png, figB_kin_channels.png")
