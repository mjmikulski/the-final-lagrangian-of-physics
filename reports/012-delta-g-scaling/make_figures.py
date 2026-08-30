"""Report 012 figures from committed artifacts.

fig_grid.png    : (a) the two time-part contractions and the predicted
                  well position vs g, with all three delta values
                  overplotted per g (their coincidence IS the
                  delta-flatness result); (b) per-observable verdict
                  summary: maximal delta-spread vs g-change.
fig_ladders.png : the three extended clock ladders (delta = 1/8),
                  energies vs omega/om_pred, with the well-position
                  drift and the PR values annotated.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10.5})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
G = json.load(open(os.path.join(R, "grid.json")))
V = json.load(open(os.path.join(R, "scaling_verdicts.json")))
E = json.load(open(os.path.join(R, "extended_ladders_all.json")))
pts = {(round(p["delta"], 6), p["g"]): p for p in G["points"]}
DS = [0.125, 0.015625, 0.001953]
GS = [8, 64, 512]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
JIT = {DS[0]: 1.0, DS[1]: 0.87, DS[2]: 1.15}
for ob, col, mk, ls, lw, lab in (
        ("time_part_G", "#2166ac", "D", "-", 2.2,
         r"time part, $G$ contraction ($\times$-1, drive)"),
        ("time_part_eta", "#e67e22", "^", "--", 1.0,
         r"time part, $\eta$ contraction (inert)"),
        ("om_pred", "#1b7837", "o", "-", 1.3,
         r"predicted well position $\omega_{\rm pred}$")):
    for d in DS:
        vals = [abs(pts[(d, g)][ob]) for g in GS]
        xs = [g * JIT[d] for g in GS]
        ax1.loglog(xs, vals, color=col, marker=mk, ms=5,
                   mfc="none" if d != DS[0] else col, mew=1.2,
                   ls=ls if d == DS[0] else "none",
                   lw=lw, alpha=1.0 if d == DS[0] else 0.8,
                   label=lab if d == DS[0] else None)
sl = V["observables"]
ax1.text(70, 0.055,
         f"$\\sim g^{{{sl['time_part_G']['g_loglog_slope']:.2f}}}$"
         " (both)", fontsize=9, color="0.15",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax1.text(24, 1.62, f"$\\sim g^{{{sl['om_pred']['g_loglog_slope']:.2f}}}$",
         fontsize=9, color="#1b7837",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax1.set_xticks(GS)
ax1.set_xticklabels(["8", "64", "512"])
ax1.minorticks_off()
ax1.set_xlabel(r"timelike vacuum eigenvalue $g$")
ax1.set_ylabel("observable magnitude  [lattice units]")
ax1.margins(y=0.15)
ax1.set_title("(a) $g$ moves the clock observables;\nthe three "
              r"$\delta$ values coincide (jittered open markers)",
              fontsize=9.5)
ax1.legend(fontsize=7.5, loc="center left", framealpha=1.0)
ax1.grid(alpha=0.25, which="major")
ax1.set_axisbelow(True)

obs = ["time_part_G", "om_pred", "C1", "C2", "depth", "I_pure",
       "I_comb", "mix34_curv", "E_stat"]
labels = ["time part G", r"$\omega_{\rm pred}$", "$C_1$", "$C_2$",
          "well depth", r"$I_{\rm pure}$", r"$I_{\rm comb}$",
          "mix-3/4 curv.", r"$E_{\rm stat}$"]
dspread = [max(V["observables"][o]["delta_spread_per_g"].values())
           for o in obs]
gchange = [V["observables"][o]["g_change_rel"] for o in obs]
y = np.arange(len(obs))
cols_d = ["#c0392b" if 100 * v > 2.0 else "#999999" for v in dspread]
ax2.barh(y - 0.2, [max(100 * v, 0.035) for v in dspread], height=0.36,
         color=cols_d, label=r"max spread along $\delta$ (3 octaves)")
for yi, v in zip(y, dspread):
    if 100 * v < 0.04:
        ax2.text(0.042, yi - 0.2, "<0.03%", fontsize=6.5,
                 va="center", color="0.35")
    if 100 * v > 2.0:
        ax2.text(100 * v * 1.15, yi - 0.2, f"{100*v:.1f}%",
                 fontsize=7, va="center", color="#c0392b")
ax2.barh(y + 0.2, [100 * v for v in gchange], height=0.36,
         color="#7b3294", label="change along $g$ (2 octaves)")
ax2.axvline(2.0, color="0.4", ls=":", lw=1.0)
ax2.text(2.2, len(obs) - 0.45, "2%", fontsize=8, color="0.35")
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=8.5)
ax2.invert_yaxis()
ax2.set_xscale("log")
ax2.set_xlim(0.03, 300)
ax2.set_xlabel("relative variation  [%]")
ax2.set_title(r"(b) verdicts: $\delta \lesssim 2\%$ over 3 octaves"
              " (red: at/over the 2% line),\n$g$ is live",
              fontsize=9.5)
ax2.legend(fontsize=7, loc="lower right", handlelength=1.2,
           borderaxespad=0.3, framealpha=0.95)
ax2.grid(alpha=0.25, axis="x", which="major")
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_grid.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_grid.png")

fig2, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=False)
for ax, c in zip(axes, E["cases"]):
    g = c["g"]
    om = [r["omega"] / c["om_pred"] for r in c["rows"]]
    dE = [1e4 * (r["E_total"] - c["rows"][0]["E_total"])
          for r in c["rows"]]
    ax.plot(om, dE, color="#1b7837", marker="o", ms=5, lw=1.5)
    ax.margins(y=0.10)
    k = min(range(len(dE)), key=lambda i: dE[i])
    ax.plot([om[k]], [dE[k]], marker="*", ms=13, color="#b8860b")
    ax.axvline(1.0, color="0.5", ls=":", lw=1.0)
    pr = c["rows"][k]["PR"]
    ax.set_title(f"$g = {g}$: min at "
                 f"{c['min_omega']/c['om_pred']:.1f}$\\times$pred, "
                 f"PR {pr:.0f}", fontsize=9.5)
    ax.set_xlabel(r"$\omega \,/\, \omega_{\rm pred}$")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"$E(\omega)-E(0)$  [$10^{-4}$ lattice units]")
fig2.suptitle("Extended clock ladders ($\\delta = 1/8$): interior "
              "wells at every $g$; the frozen-profile prediction "
              "degrades with $g$ (dotted line = prediction)",
              fontsize=9.5, y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_ladders.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_ladders.png")
