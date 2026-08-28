"""Report figures from committed artifacts only (result JSONs plus the
committed rung field) -- runs on a clean checkout, no 004 fields
needed. Layout follows the readability rules learned in reports 007/008
review rounds (annotation collisions, 800-px README scaling,
color-vision-safe linestyles).

fig_i1sq_ladders.png : (a) energy reading -- Delta E(omega) for the
                       G form at gamma and 4*gamma, the faithful eta
                       form and the sign control, with the omega_E
                       prediction; (b) depth plateau of the JG_E well
                       across relaxation-protocol levels (round-2
                       convergence criterion).
fig_mechanism.png    : (a) radial template/ticking profiles;
                       (b) localization PR for all ladders;
                       (c) the intensive variant zeroing its integral.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
lad = json.load(open(os.path.join(R, "i1sq_ladders.json")))
gsc = json.load(open(os.path.join(R, "gamma_scaling.json")))

SERIES_E = [
    ("JG_E", lad, "#2166ac", "D", "-", r"$G$ form, $\gamma$"),
    (None, gsc, "#1a1a1a", "^", "-.", r"$G$ form, $4\gamma$"),
    ("J_ETA", lad, "#e67e22", "s", ":",
     r"faithful $\eta$ form (inert)"),
    ("J0", lad, "#c0392b", "o", "--", "sign control"),
]


def dE(key, src):
    rungs = src[key]["rungs"] if key else src["rows"]
    E0 = rungs[0]["E_total"]
    return ([r["omega"] for r in rungs],
            [1e4 * (r["E_total"] - E0) for r in rungs])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.3))
for key, src, col, mk, ls, lab in SERIES_E:
    om, y = dE(key, src)
    ax1.plot(om, y, color=col, marker=mk, ls=ls, ms=4, lw=1.4,
             label=lab)
ax1.set_xlim(-0.03, 0.88)
ax1.set_ylim(-3.6, 2.0)
ax1.axvline(lad["omega_pred_E"], color="0.35", ls=":", lw=1.1)
ax1.text(0.02, -3.2, "predicted\n$\\omega_E=%.3f$"
         % lad["omega_pred_E"], fontsize=8.5, color="0.25",
         ha="left", va="bottom",
         bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.2))
ax1.axhline(0, color="0.75", lw=0.8)
ax1.set_xlabel(r"$\omega$  [lattice units]")
ax1.set_ylabel(r"$E(\omega)-E(0)$  [$10^{-4}$ lattice units]")
ax1.set_title("(a) energy reading")
ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
           borderpad=0.35, handlelength=1.8, handletextpad=0.5)
ax1.grid(alpha=0.25)
axi = ax1.inset_axes([0.60, 0.08, 0.38, 0.40])
for key, src, col, mk, ls, lab in SERIES_E:
    om, y = dE(key, src)
    axi.plot(om, y, color=col, ls=ls, lw=1.0)
axi.set_xlim(-0.05, 1.25)
axi.tick_params(labelsize=6, direction="in", pad=2)
axi.set_title("full range", fontsize=6.5, pad=2)
axi.grid(alpha=0.2)

dpl = lad["JG_E"]["depth_per_level"]
levels = list(range(len(dpl)))
ax2.plot(levels, [1e4 * d for d in dpl], color="#2166ac", marker="D",
         ms=5, lw=1.5)
ax2.set_xticks(levels)
ax2.set_xticklabels(["Adam"] + [f"+{i}" for i in range(1, len(dpl))],
                    fontsize=9)
ax2.set_xlim(-0.3, 4.3)
ax2.set_ylim(0, max(1e4 * d for d in dpl) * 1.25)
ax2.set_xlabel("relaxation level (L-BFGS restarts)")
ax2.set_ylabel(r"well depth  [$10^{-4}$ lattice units]")
ax2.set_title("(b) well depth $E(0)-E(0.35)$ plateaus", fontsize=11)
for i, d in enumerate(dpl[1:], start=1):
    ch = 1e4 * (dpl[i] - dpl[i - 1])
    ax2.annotate(f"{ch:+.3f}", (i, 1e4 * dpl[i]),
                 textcoords="offset points", xytext=(0, -14),
                 ha="center", fontsize=7.5, color="0.35")
ax2.text(0.98, 0.05, "labels: step-to-step change, panel units",
         transform=ax2.transAxes, fontsize=7.5, color="0.35",
         ha="right")
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_i1sq_ladders.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_i1sq_ladders.png")

# ---- mechanism figure --------------------------------------------------
N, Lbox = 32, 48.0
Hh = Lbox / N
SG, DELTA = 8.0, 0.3
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def d1(f, ax, st):
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim
    lo, hi = [slice(None)] * f.ndim, [slice(None)] * f.ndim
    lo[ax], hi[ax] = slice(0, -1), slice(1, None)
    if st == "fwd":
        sl[ax] = slice(0, -1)
        out[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / Hh
    else:
        sl[ax] = slice(1, None)
        out[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / Hh
    return out


def comm(A, B):
    return A @ ETA @ B - B @ ETA @ A


def G_of(M):
    x = np.einsum("ab,...bc->...ac", ETA, M)
    I4 = np.broadcast_to(np.eye(4), M.shape)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def inner_pc(F, X):
    return np.einsum("...ab,...ac,...bd,...cd->...", F, X, X, F)


plt.rcParams.update({"font.size": 13, "axes.titlesize": 13,
                     "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12})
fig2, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(10.6, 3.4))
rung_fp = os.path.join(R, "jge_rung_om035.npz")
a0_fp = os.path.join(R, "a0_frozen.npz")
if os.path.exists(rung_fp) and os.path.exists(a0_fp):
    M = np.load(rung_fp)["M"]
    a0 = np.load(a0_fp)["a0"]
    G = G_of(M)
    om_min = lad["JG_E"]["min_omega"]
    V = om_min * a0
    i1s, k = 0.0, 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            k = k + 0.5 * 4.0 * inner_pc(comm(V, A[i]), G)
            for j in range(i + 1, 3):
                i1s = i1s + 0.5 * 4.0 * inner_pc(comm(A[i], A[j]), G)
    c = N // 2
    rr = np.sqrt(((np.indices((N, N, N)) - c) ** 2).sum(axis=0)) * Hh
    rbins = np.arange(0.0, 16.0, 1.0)
    for dens, ls, col, lab in (
            (i1s, "-", "#2166ac", r"static $i_1$ density (template)"),
            (k, "--", "#7b3294",
             rf"ticking density $k$ at $\omega={om_min}$")):
        m = np.array([dens[(rr >= r0) & (rr < r0 + 1.0)].mean()
                      for r0 in rbins[:-1]])
        bx1.plot(rbins[:-1] + 0.5, m / m.max(), ls=ls, color=col,
                 lw=1.5, label=lab)
    bx1.set_xlabel(r"$r$  [lattice units]")
    bx1.set_ylabel("shell-averaged density (norm.)")
    bx1.set_title("(a) template and ticking")
    bx1.set_ylim(-0.05, 1.45)
    bx1.legend(fontsize=9, loc="upper center", framealpha=1.0)
    bx1.grid(alpha=0.25)
else:
    bx1.text(0.5, 0.5, "committed rung field absent",
             ha="center", va="center", transform=bx1.transAxes)

for key, col, mk, ls, lab in (
        ("JG_E", "#2166ac", "D", "-", r"$G$, energy reading"),
        ("J_ETA", "#e67e22", "s", ":", r"$\eta$ (inert)")):
    rungs = lad[key]["rungs"]
    om = [r["omega"] for r in rungs if r["omega"] > 0]
    pr = [r["PR_k_sites"] for r in rungs if r["omega"] > 0]
    bx2.plot(om, pr, color=col, marker=mk, ls=ls, ms=4, lw=1.3,
             label=lab)
bx2.set_ylim(95, 205)
bx2.text(0.5, 0.955, "delocalized floor (004): 1962, off scale",
         fontsize=9, color="0.25", ha="center", va="top",
         transform=bx2.transAxes)
bx2.set_xlabel(r"$\omega$  [lattice units]")
bx2.set_ylabel(r"PR of $k$  [sites]")
bx2.set_title(r"(b) localization ($\omega>0$)")
bx2.legend(fontsize=9, loc="upper left",
           bbox_to_anchor=(0.01, 0.88), framealpha=1.0)
bx2.grid(alpha=0.25)

rj2 = lad["J2_intensive"]["rungs"]
bx3.plot([r["omega"] for r in rj2],
         [1e3 * r["E_extra"] for r in rj2], marker="P", ms=5,
         color="#8c510a", lw=1.3)
bx3.axvline(5.0, color="0.35", ls=":", lw=1.0)
bx3.text(4.85, 9.5, "min: $\\omega=5$", fontsize=9, color="0.25",
         ha="right")
bx3.axhline(0, color="0.75", lw=0.8)
bx3.set_xlabel(r"$\omega$  [lattice units] (intensive scan)")
bx3.set_ylabel(r"$E_{\rm extra}$  [$10^{-3}$ lattice units]")
bx3.set_title("(c) intensive variant:\nminimum by zeroed integral",
              fontsize=11.5)
bx3.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_mechanism.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_mechanism.png")
