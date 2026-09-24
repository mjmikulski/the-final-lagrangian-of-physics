"""Figure for APPENDIX-contraction-release, from the committed JSON records (no lattice work).

Left: E(omega) - E(0) at the final protocol level for the three matrix-slot contractions of the report's box:
the working metric G (the report's JG_E bracket, rerun record rungs_N32.json), the Frobenius contraction (this
appendix) and eta (the report's J_ETA ladder). Right: the depth E(0) - E(omega) at every protocol level for the
frozen tangent and for the tangent recomputed from the field (released), eight L-BFGS cycles.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "contraction_release")
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#6b6a63"
MAGENTA, YELLOW, GREEN = "#e87ba4", "#c98500", "#008300"      # right panel: its own hues (unrelated quantities)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

g = json.load(open(os.path.join(HERE, "results", "L_ladder", "rungs_N32.json")))["rungs"]
e = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))["J_ETA"]["rungs"]
f = json.load(open(os.path.join(RES, "frob.json")))["rungs"]
rel = json.load(open(os.path.join(RES, "release.json")))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
for rows, col, mk, lab in ((g, BLUE, "o", "G (report)"), (f, AQUA, "s", "Frobenius (appendix; overlaps G)"),
                           (e, ORANGE, "^", "η (report)")):
    E0 = [r for r in rows if r["omega"] == 0.0][0]["E_total"]
    rr = sorted(rows, key=lambda r: r["omega"])
    ax.plot([r["omega"] for r in rr], [(r["E_total"] - E0) * 1e5 for r in rr], "-", color=col, marker=mk, ms=6,
            mec="white", mew=1.0, lw=2, label=lab)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set(xlabel="clock frequency ω", ylabel="E(ω) − E(0)  (10⁻⁵, model units)", ylim=(-13, 16), xlim=(-0.03, 0.56),
       title="(I₁)² with G, Frobenius or η: the well needs a positive norm")
ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=3, columnspacing=1.0, handlelength=1.6)
ax.title.set_fontsize(11)

levels = list(range(len(rel["frozen8"][0]["E_levels"])))
for key, col, mk, lab in (("frozen 0.35", MAGENTA, "o", "frozen tangent, ω = 0.35"),
                          ("released 0.35", YELLOW, "s", "released tangent, ω = 0.35"),
                          ("released 0.2", GREEN, "^", "released tangent, ω = 0.2")):
    d = rel["depth_per_level"][key]
    bx.plot(levels, [x * 1e5 for x in d], "-", color=col, marker=mk, ms=6, mec="white", mew=1.0, lw=2, label=lab)
bx.set(xlabel="protocol level (0 = after Adam, then one per L-BFGS cycle)", ylabel="depth E(0) − E(ω)  (10⁻⁵)",
       title="frozen vs released clock tangent: the depth stays flat")
bx.set_ylim(2.5, 7.5)
bx.legend(frameon=False, fontsize=10, loc="center right")
bx.title.set_fontsize(11)
fig.tight_layout()
tmp = os.path.join(RES, "fig_contraction_release.tmp.png")
fig.savefig(tmp)
os.replace(tmp, os.path.join(RES, "fig_contraction_release.png"))
