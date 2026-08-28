"""Appendix figure, generated from the committed result JSON only.

fig_appendix_connections.png :
  (a) the time sector opened by the index-mixing connection on the static
      ansatz, |F(D)_0i| vs dressing amplitude m at lam = 0.1, full vs the
      vacuum-powered part (spatial legs frozen to their vacuum value) --
      the small-m sector rides on the vacuum-breaking term;
  (b) the price: uniform vacuum field strength |F(D)_ij| = lam^2, measured
      points against the exact law.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
res = json.load(open(os.path.join(R, "appendix_no_connection.json")))
cx = res["index_mixing"]["crossover"]
vac = res["index_mixing"]["vacuum"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.4))

ms = np.array(cx["ms"])
full = np.array(cx["F_0i"])
vp = np.array(cx["vacuum_powered"])
ax1.loglog(ms, full, "o-", color="#2166ac", ms=5, label="full $|F(D)_{0i}|$")
ax1.loglog(ms, vp, "s--", color="#c0392b", ms=5, mfc="none", mew=1.5,
           label="vacuum-powered part")
g1 = 0.55 * full[0] * (ms / ms[0])
ax1.loglog(ms[:5], g1[:5], ":", color="0.45", lw=1)
ax1.text(ms[1] * 1.05, g1[1] * 0.52, r"$\propto m$", color="0.35", fontsize=10)
m2 = np.geomspace(0.12, 0.28, 20)
g2 = 0.21 * full[-1] * (m2 / ms[-1]) ** 2
ax1.loglog(m2, g2, ":", color="0.45", lw=1)
ax1.text(0.262, g2[-1] * 0.92, r"$\propto m^2$", color="0.35", fontsize=10,
         ha="left", va="center")
ax1.axvline(cx["lam"], color="0.75", lw=0.8, ls="--", zorder=0)
ax1.text(cx["lam"], 1.4e-4, r"$m=\lambda$", color="0.45", fontsize=9,
         rotation=90, va="center", ha="center",
         bbox=dict(fc="white", ec="none", pad=1.5))
ax1.set_xlabel(r"dressing amplitude $m$")
ax1.set_ylabel(r"$\max|F(D)_{0i}|$  (static ansatz, $\lambda=0.1$)")
ax1.set_title("(a) opened time sector is vacuum-powered", fontsize=11)
ax1.legend(loc="upper left", fontsize=9, framealpha=1.0)
ax1.xaxis.set_major_locator(FixedLocator([0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]))
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.set_xticklabels(["0.0125", "0.025", "0.05", "0.1", "0.2", "0.3", "0.4"],
                    rotation=45, ha="right")

lams = np.array(vac["lams"][1:])           # drop lam = 0 on the log axis
Fv = np.array(vac["F_ij"][1:])
ax2.loglog(lams, lams ** 2, "-", color="0.45", lw=1.5,
           label=r"$\lambda^2$ (exact law)")
ax2.loglog(lams, Fv, "o", color="#1a9850", ms=7, mfc="none", mew=1.8,
           label="measured")
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel(r"$\max|F(D)_{ij}|$  in the vacuum")
ax2.set_title(r"(b) the price: vacuum $|F| = \lambda^2$", fontsize=11)
ax2.legend(loc="upper left", fontsize=9, framealpha=1.0)
ax2.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.3, 0.4]))
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.4"])
ax2.margins(x=0.08, y=0.12)

fig.tight_layout()
out = os.path.join(R, "fig_appendix_connections.png")
fig.savefig(out, dpi=200)
print("wrote", out)
