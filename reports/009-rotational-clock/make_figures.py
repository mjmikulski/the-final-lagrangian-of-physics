"""Report figures from committed artifacts only (this report's JSON
plus report 008's committed JSON for the boost-channel comparison).
Readability rules follow the 007/008 review lessons.

fig_rot_ladders.png : (a) Delta E(omega) for the rotational ladder and
                      its sign control, main view on the well, inset
                      full range; (b) well depth vs relaxation level,
                      including the deep-endpoint reference.
fig_rot_channel.png : (a) localization PR(omega) rotation vs boost
                      (008); (b) channel angular momentum J = I_R*omega
                      with the sampled minimum marked.
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
rot = json.load(open(os.path.join(R, "rot_ladders.json")))
b08 = json.load(open(os.path.join(HERE, "..", "008-i1-squared-clock",
                                  "results", "i1sq_ladders.json")))

om_R = rot["omega_pred_R"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.3))


def dE(v):
    rungs = v["rungs"]
    E0 = rungs[0]["E_total"]
    return ([r["omega"] for r in rungs],
            [1e4 * (r["E_total"] - E0) for r in rungs])


for key, col, mk, ls, lab in (
        ("JR_E", "#1b7837", "v", "-", "rotation, energy reading"),
        ("JR0", "#e67e22", "o", "--", "sign control")):
    om, y = dE(rot[key])
    ax1.plot(om, y, color=col, marker=mk, ls=ls, ms=4, lw=1.4,
             label=lab)
ax1.set_xlim(-0.02, 0.46)
ax1.set_ylim(-0.55, 0.35)
ax1.axvline(om_R, color="0.35", ls=":", lw=1.1)
ax1.text(0.012, -0.50, "predicted\n$\\omega_R=%.3f$" % om_R,
         fontsize=8.5, color="0.25", ha="left", va="bottom",
         bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.2))
ax1.axhline(0, color="0.75", lw=0.8)
ax1.set_xlabel(r"$\omega$  [lattice units]")
ax1.set_ylabel(r"$E(\omega)-E(0)$  [$10^{-4}$ lattice units]")
ax1.set_title("(a) rotational well")
ax1.legend(fontsize=7.5, loc="upper left",
           bbox_to_anchor=(0.01, 0.87), framealpha=1.0,
           handlelength=3.0)
ax1.grid(alpha=0.25)
axi = ax1.inset_axes([0.62, 0.09, 0.35, 0.34])
for key, col, ls in (("JR_E", "#1b7837", "-"), ("JR0", "#e67e22", "--")):
    om, y = dE(rot[key])
    axi.plot(om, y, color=col, ls=ls, lw=1.0)
axi.tick_params(labelsize=6, direction="in", pad=2,
                length=2)
axi.set_title(r"full range [$10^{-4}$]", fontsize=6.5, pad=2)
axi.grid(alpha=0.2)

dpl = rot["JR_E"]["depth_per_level"]
dde = rot["JR_E"]["depth_deep_endpoint"]
ax2.plot(range(len(dpl)), [1e4 * d for d in dpl], color="#1b7837",
         marker="o", ms=10, mfc="none", mew=1.6, lw=1.5,
         label="vs min at same level")
ax2.plot(range(len(dde)), [1e4 * d for d in dde], color="#7b3294",
         marker="D", ms=4, ls="--", lw=1.3,
         label="deep endpoint vs converged min")
ax2.set_xticks(range(len(dde)))
ax2.set_xticklabels(["Adam"] + [f"+{i}" for i in range(1, len(dde))],
                    fontsize=9)
ax2.set_xlim(-0.3, len(dde) - 0.7)
ax2.set_ylim(0, max(1e4 * d for d in dpl) * 1.3)
ax2.set_xlabel("relaxation level (L-BFGS restarts)")
ax2.set_ylabel(r"well depth  [$10^{-4}$ lattice units]")
ax2.set_title("(b) well depth vs relaxation level",
              fontsize=11)
ax2.legend(fontsize=7.5, loc="lower left", framealpha=1.0,
           handlelength=3.0)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_rot_ladders.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_rot_ladders.png")

fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(7.8, 3.2))
for src, key, col, mk, ls, lab in (
        (rot, "JR_E", "#1b7837", "v", "-", "rotation (this report)"),
        (b08, "JG_E", "#2166ac", "D", "-", "boost (report 008)")):
    rungs = src[key]["rungs"]
    om = [r["omega"] for r in rungs if r["omega"] > 0]
    pr = [r["PR_k_sites"] for r in rungs if r["omega"] > 0]
    bx1.plot(om, pr, color=col, marker=mk, ls=ls, ms=4, lw=1.3,
             label=lab)
bx1.set_ylim(40, 200)
bx1.set_xlabel(r"$\omega$  [lattice units]")
bx1.set_ylabel(r"PR of $k$  [sites]")
bx1.set_title("(a) localization: rotation vs boost")
bx1.legend(fontsize=8, loc="upper left", framealpha=1.0)
bx1.grid(alpha=0.25)

I_R = rot["I_R"]
oms = [r["omega"] for r in rot["JR_E"]["rungs"]]
bx2.plot([0, max(oms)], [0, I_R * max(oms)], color="#1b7837",
         lw=1.5)
om_min = rot["JR_E"]["min_omega"]
bx2.axvline(om_min, color="0.35", ls=":", lw=1.0)
bx2.plot([om_min], [rot["J_at_min"]], marker="*", ms=14,
         color="#b8860b")
bx2.annotate(f"$J_* = I_R\\,\\omega_* = {rot['J_at_min']:.3f}$",
             xy=(om_min, rot["J_at_min"]),
             xytext=(om_min + 0.05, rot["J_at_min"] - 0.022),
             fontsize=9, color="0.25",
             arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
bx2.set_xlabel(r"$\omega$  [lattice units]")
bx2.set_ylabel(r"$J = I_R\,\omega$  [lattice units]")
bx2.set_title("(b) channel angular momentum")
bx2.text(0.03, 0.92, f"$I_R = {I_R:.3f}$", transform=bx2.transAxes,
         fontsize=9, color="0.25")
bx2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_rot_channel.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_rot_channel.png")
