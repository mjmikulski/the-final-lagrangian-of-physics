"""Report 013 figure: the bracket differences vs continued cycle for
both window couplings -- the record of the non-saturating drift."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
W = json.load(open(os.path.join(R, "window_deep.json")))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
for ax, (tag, label, extra) in zip(axes, (
        ("x14_continued", "coupling ×14 (continued from the persisted "
         "six-cycle fields)", "cycles shown are +continued"),
        ("x10_fresh", "coupling ×10 (fresh start)", ""))):
    h = W["arms"][tag]["history"]
    n = len(h["0.15"]["E"])
    cyc = range(n)
    for om, col, ls, mk in (("0.1", "#2166ac", "-", "D"),
                            ("0.2", "#1b7837", "--", "o"),
                            ("0.28", "#c0392b", ":", "^")):
        d = [1e3 * (h[om]["E"][i] - h["0.15"]["E"][i])
             for i in range(n)]
        ax.plot(cyc, d, color=col, ls=ls, marker=mk, ms=3.5, lw=1.3,
                mec="white", mew=0.4,
                label=f"$E({om}) - E(0.15)$")
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xlabel("continued L-BFGS cycle" if "continued" in tag
                  else "L-BFGS cycle")
    ax.set_title(label, fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"bracket difference  [$10^{-3}$ lattice units]")
axes[0].legend(fontsize=8, loc="lower left", framealpha=0.95)
axes[1].legend(fontsize=8, loc="lower left", framealpha=0.95)
fig.suptitle("No observable-level plateau in 24 cycles:\n"
             "an interior well needs every difference positive",
             fontsize=12, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_drift.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_drift.png")
