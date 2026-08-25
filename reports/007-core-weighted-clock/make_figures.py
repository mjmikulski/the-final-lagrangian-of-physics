"""Report figures, generated from the committed result JSONs only (no
lattice fields needed -- runs on a clean checkout). Layout follows a
readability review (annotation collisions, 800-px README scaling,
color-vision-safe linestyles, log-axis range).

fig_ladders.png : (a) E_total(omega) for the ladder series with a zoom
                  inset on the interior wells; (b) participation ratio
                  of the boost density (log scale, omega > 0).
fig_kinetic_window.png : full-core sweep -- (a) distribution of the
                  first-mode threshold c_clock; (b) per-cell comparison
                  with the Rayleigh threshold of the a0 channel.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
lad = json.load(open(os.path.join(R, "ladder_series.json")))
kin = json.load(open(os.path.join(R, "kinetic_forms.json")))

STYLE = {
    "L1_dynamic_local": dict(color="#c0392b", marker="o", ls="-",
                             label="L1 local, dyn. weight"),
    "L2_frozen_local": dict(color="#e67e22", marker="s", ls="--",
                            label="L2 local, frozen mask"),
    "L4_intensive_fresh": dict(color="#2166ac", marker="D", ls="-",
                               label="L4 intensive, frozen"),
    "L5_intensive_dynamic": dict(color="#1a9850", marker="^", ls="-.",
                                 label="L5 intensive, dyn. $c(M)$"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.4))
for key, st in STYLE.items():
    if key not in lad:
        continue
    rungs = lad[key]["rungs"]
    om = [r["omega"] for r in rungs]
    ax1.plot(om, [r["E_total"] for r in rungs], ms=3.5, lw=1.3, **st)
    om_p = [r["omega"] for r in rungs if r["omega"] > 0]
    ax2.semilogy(om_p, [r["PR_bk_sites"] for r in rungs
                        if r["omega"] > 0], ms=3.5, lw=1.3, **st)
ax1.set_xlabel(r"$\omega$  [lattice units]")
ax1.set_ylabel(r"$E_{\rm total}$  [lattice units]")
ax1.set_title("(a) ladder energies")
ax1.legend(fontsize=7, loc="lower left", framealpha=1.0,
           borderpad=0.35, handlelength=1.6,
           handletextpad=0.5)
ax1.grid(alpha=0.25)

# zoom inset on the interior wells (L2/L4/L5 band)
axi = ax1.inset_axes([0.40, 0.46, 0.57, 0.50])
for key in ("L2_frozen_local", "L4_intensive_fresh",
            "L5_intensive_dynamic"):
    if key not in lad:
        continue
    rungs = lad[key]["rungs"]
    axi.plot([r["omega"] for r in rungs],
             [r["E_total"] for r in rungs], ms=2.5, lw=1.1,
             **{k: v for k, v in STYLE[key].items() if k != "label"})
axi.set_ylim(4.805, 4.93)
axi.set_xlim(-0.05, 2.9)
axi.set_yticks([4.85, 4.90])
axi.tick_params(axis="y", direction="in", pad=-24, labelsize=6.5)
# x labels of the inset would render outside its frame, straight onto
# the descending L1 curve -- same omega range as the host, so drop them
axi.tick_params(axis="x", labelbottom=False)
axi.grid(alpha=0.2)
r5 = {x["omega"]: x["E_total"] for x in lad["L5_intensive_dynamic"]["rungs"]}
axi.annotate(r"$\omega_*=0.8$ (L4, L5)", xy=(0.8, r5[0.8]),
             xytext=(0.42, 4.912), fontsize=8, zorder=10,
             arrowprops=dict(arrowstyle="->", color="0.2", lw=0.9))
ax1.indicate_inset_zoom(axi, edgecolor="0.5")

ax2.axhline(1962, color="0.45", ls=":", lw=1.1)
ax2.text(0.25, 2050, "report-004 delocalized floor (1962)",
         fontsize=8, color="0.3", va="bottom")
ax2.set_ylim(50, 3000)
ax2.set_xlabel(r"$\omega$  [lattice units]")
ax2.set_ylabel(r"participation ratio of $b_k$  [sites]")
ax2.set_title(r"(b) localization of the ticking ($\omega>0$)")
ax2.grid(alpha=0.25)
ax2.grid(alpha=0.10, which="minor")
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_ladders.png"), dpi=160)
print("written: results/fig_ladders.png")

if "core_sweep" in kin:
    cs = kin["core_sweep"]
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(7.8, 3.1))
    bx1.hist(cs["c_clock_all"], bins=24, color="#2166ac", alpha=0.85)
    bx1.set_ylim(top=bx1.get_ylim()[1] * 1.15)
    bx1.axvline(cs["c_clock_median"], color="k", ls="--", lw=1)
    med_txt = f"median {cs['c_clock_median']:.3f}".replace("-", "\u2212")
    bx1.text(cs["c_clock_median"] - 0.012, bx1.get_ylim()[1] * 0.95,
             med_txt + " ", fontsize=9,
             ha="right", va="top", clip_on=True)
    bx1.set_xlabel(r"first-mode threshold $c_{\rm clock}$")
    bx1.set_ylabel("core cells")
    bx1.set_title(f"(a) $c_{{\\rm clock}}$, all {cs['n_cells']} core cells")
    bx1.grid(alpha=0.25)
    bx2.scatter(cs["c_clock_all"], cs["c_a0_all"], s=9, alpha=0.35,
                color="#1a9850", edgecolors="white", linewidths=0.3)
    bx2.axhline(cs["c_a0_median"], color="0.4", ls="--", lw=1)
    bx2.axvline(cs["c_clock_median"], color="k", ls="--", lw=0.9)
    x0, x1 = bx2.get_xlim()
    a0_txt = (f"$a_0$-channel median {cs['c_a0_median']:.2f}"
              .replace(" -1", " \u22121"))
    bx2.text(x0 + 0.02 * (x1 - x0), cs["c_a0_median"] + 0.02,
             a0_txt, fontsize=8, va="bottom", color="0.25")
    bx2.text(x0 + 0.012 * (x1 - x0), -1.975,
             "accumulation at $-2.0$", fontsize=8, va="bottom",
             color="0.25")
    bx2.set_xlabel(r"first-mode threshold $c_{\rm clock}$")
    bx2.set_ylabel(r"$a_0$-channel threshold $c_{a_0}$")
    bx2.set_title("(b) two threshold scales per cell")
    bx2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(os.path.join(R, "fig_kinetic_window.png"), dpi=160)
    print("written: results/fig_kinetic_window.png")
else:
    print("core_sweep absent from kinetic_forms.json -- figure 2 skipped")
