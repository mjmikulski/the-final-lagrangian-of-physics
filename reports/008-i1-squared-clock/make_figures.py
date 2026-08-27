"""Report figures from committed artifacts only (result JSONs plus the
committed J1 rung field) -- runs on a clean checkout, no 004 fields
needed. Layout follows the readability rules learned in report 007
(annotation collisions, 800-px README scaling, color-vision-safe
linestyles).

fig_i1sq_ladders.png : (a) Delta E(omega) for J1 at gamma and 4*gamma
                       and the J0 sign control, with the frozen-profile
                       prediction line; (b) localization PR(omega).
fig_mechanism.png    : (a) radial profiles of the static I1 density
                       (the template) and the ticking density at
                       omega_* -- template matching, with the honest
                       point that i1_stat is nearly flat; (b) the
                       intensive variant zeroing its integral -- why
                       the local form is the physical one.
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.3))

series = [
    ("J1_local_covariant", lad, "#2166ac", "D", "-",
     r"J1: local $(I_1)^2$, $\gamma$"),
    (None, gsc, "#7b3294", "^", "-.",
     r"J1 at $4\gamma$ (depth $\times$4 check)"),
    ("J0_local_control", lad, "#c0392b", "o", "--",
     "J0: flipped cross sign (control)"),
]


def dE_series(key, src):
    rungs = src[key]["rungs"] if key else src["rows"]
    om = [r["omega"] for r in rungs]
    E0 = rungs[0]["E_total"]
    return om, [1e4 * (r["E_total"] - E0) for r in rungs]


# main view: the wells themselves (the figure's thesis); full range in
# an inset (report-007 lesson: don't let one steep curve crush the story)
for key, src, col, mk, ls, lab in series:
    om, dE = dE_series(key, src)
    ax1.plot(om, dE, color=col, marker=mk, ls=ls, ms=4, lw=1.4,
             label=lab)
ax1.set_xlim(-0.03, 0.88)
ax1.set_ylim(-3.6, 2.0)
ax1.axvline(lad["omega_pred_frozen"], color="0.35", ls=":", lw=1.1)
ax1.text(0.02, -3.2, "predicted\n$\\omega_*=%.3f$"
         % lad["omega_pred_frozen"], fontsize=8.5, color="0.25",
         ha="left", va="bottom",
         bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.2))
ax1.axhline(0, color="0.75", lw=0.8)
ax1.set_xlabel(r"$\omega$  [lattice units]")
ax1.set_ylabel(r"$E(\omega)-E(0)$  [$10^{-4}$ lattice units]")
ax1.set_title("(a) the $(I_1)^2$ well and its controls")
ax1.legend(fontsize=8, loc="upper right", framealpha=1.0,
           borderpad=0.35, handlelength=1.8, handletextpad=0.5)
ax1.grid(alpha=0.25)

axi = ax1.inset_axes([0.60, 0.08, 0.38, 0.40])
for key, src, col, mk, ls, lab in series:
    om, dE = dE_series(key, src)
    axi.plot(om, dE, color=col, ls=ls, lw=1.0)
axi.set_xlim(-0.05, 1.25)
axi.tick_params(labelsize=6, direction="in", pad=2)
axi.set_title("full range", fontsize=6.5, pad=2)
axi.grid(alpha=0.2)

# localization: linear axis around the measured band; the delocalized
# floor is off scale and quoted as text
for key, src, col, mk, ls, lab in series:
    rungs = src[key]["rungs"] if key else src["rows"]
    om = [r["omega"] for r in rungs if r["omega"] > 0]
    pr = [r["PR_bk_sites"] for r in rungs if r["omega"] > 0]
    ax2.plot(om, pr, color=col, marker=mk, ls=ls, ms=4, lw=1.4,
             label=lab)
ax2.set_ylim(85, 300)
ax2.text(0.5, 0.955, "delocalized floor (004): 1962 sites, off scale",
         fontsize=8, color="0.25", ha="center", va="top",
         transform=ax2.transAxes)
ax2.set_xlabel(r"$\omega$  [lattice units]")
ax2.set_ylabel(r"participation ratio of $b_k$  [sites]")
ax2.set_title(r"(b) localization ($\omega>0$)")
ax2.legend(fontsize=7, loc="center left", framealpha=1.0,
           borderpad=0.3, handlelength=1.8)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_i1sq_ladders.png"), dpi=160,
            bbox_inches="tight")
print("written: results/fig_i1sq_ladders.png")

# ---- mechanism figure: needs only committed artifacts -----------------
# (numpy definitions re-declared here; verify_energies.py cannot be
# imported for them because it runs its assertions on import)
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


def inner_const(F, X):
    return np.einsum("...ab,ac,bd,...cd->...", F, X, X, F)


fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(7.8, 3.1))
rung_fp = os.path.join(R, "j1_rung_om035.npz")
a0_fp = os.path.join(R, "a0_frozen.npz")
if os.path.exists(rung_fp) and os.path.exists(a0_fp):
    M = np.load(rung_fp)["M"]
    a0 = np.load(a0_fp)["a0"]
    G = G_of(M)
    i1s = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                i1s = i1s + 0.5 * 4.0 * inner_pc(comm(A[i], A[j]), G)
    om_min = lad["J1_local_covariant"]["min_omega"]
    Vv = om_min * a0
    bk = 0.0
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F = comm(Vv, A[i])
            bk = bk + 0.5 * 4.0 * (inner_pc(F, G)
                                   - inner_const(F, ETA)) / 2
    c = N // 2
    ii = np.indices((N, N, N))
    rr = np.sqrt(((ii - c) ** 2).sum(axis=0)) * Hh
    rbins = np.arange(0.0, 16.0, 1.0)
    prof = {}
    for name, dens in (("i1_stat", i1s), ("b_k", bk)):
        m = [dens[(rr >= r0) & (rr < r0 + 1.0)].mean()
             for r0 in rbins[:-1]]
        prof[name] = np.array(m)
    for (name, ls, col, lab) in (
            ("i1_stat", "-", "#2166ac",
             r"static $i_1$ density (the template)"),
            ("b_k", "--", "#7b3294",
             rf"ticking density $b_k$ at $\omega={om_min}$")):
        y = prof[name] / prof[name].max()
        bx1.plot(rbins[:-1] + 0.5, y, ls=ls, color=col, lw=1.5,
                 label=lab)
    bx1.set_xlabel(r"$r$  [lattice units]")
    bx1.set_ylabel("shell-averaged density (normalized)")
    bx1.set_title("(a) template and ticking profiles")
    bx1.set_ylim(-0.05, 1.32)
    bx1.legend(fontsize=8, loc="upper center", framealpha=1.0)
    bx1.grid(alpha=0.25)
else:
    bx1.text(0.5, 0.5, "committed rung field absent",
             ha="center", va="center", transform=bx1.transAxes)

rj2 = lad["J2_intensive"]["rungs"]
bx2.plot([r["omega"] for r in rj2],
         [1e3 * r["E_extra"] for r in rj2], marker="s", ms=3.5,
         color="#e67e22", lw=1.3)
bx2.axhline(0, color="0.75", lw=0.8)
bx2.set_xlabel(r"$\omega$  (intensive-variant scan)")
bx2.set_ylabel(r"$E_{\rm extra}$  [$10^{-3}$ lattice units]")
bx2.set_title(r"(b) intensive variant (own $\omega$ scale):"
              "\nminimum by zeroed integral", fontsize=10)
bx2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_mechanism.png"), dpi=160,
             bbox_inches="tight")
print("written: results/fig_mechanism.png")
