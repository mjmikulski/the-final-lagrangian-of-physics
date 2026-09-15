"""M5.21.2b: the well-posed 3D instrument (hedgehog + charged ring).

WHY (the M5.21.2 instrument finding, findings/m5_21_2_census.md § 5):
at toy parameters the bare quartic commutator functional has NO
stencil-consistent minimizer in reach: each stencil's deep descent
hides curvature in that stencil's blind directions (2h checkerboard;
fwd aligned-gradient soft modes), cross-stencil E_u disagreement
x7-128. This script builds the candidate fixes and the consistency
instruments to select among them, h-aware throughout.

THE FUNCTIONAL (3x3 symmetric M(x) on a cubic grid, spacing h, box
L = N*h fixed across the refinement ladder):
    A_i   = d_i M / h            (stencil: fwd | bwd | 2h | sym)
    C_ij  = [A_i, A_j]
    u     = 4 * sum_{i<j} tr(C_ij^T C_ij)          (curvature)
    D     = sum_i tr(A_i A_i^T)                    (Dirichlet)
    V     = per term set below
    E     = h^3 * sum_cells (u + eps*D + V)
    "sym" stencil: E = 1/2 (E[fwd] + E[bwd]); gradient = the same
    average of the two exact adjoints (kills the odd-even null family
    of 2h AND the parity bias of fwd alone).
    Derrick (3D, boundary-free): dE/dlambda at 1 = -E_u + E_D + 3E_V,
    reported per endpoint as the virial residual.

TERM SETS (the Q25 discrimination arms; s = the T3 spectrum shift):
    T1 trace-target:  V = W1 * sum_{p=1..3} (tr(M^p) - c_p)^2,
                      c_p = 1 + delta^p            (vacuum (1,delta,0))
    T2 eig-penalty:   V = W2 * sum_i (lam_i - v_i)^2, lam ascending,
                      v = sorted(vacuum spectrum)   (Eq-12 form)
    T3 shifted+det:   vacuum (1+s, delta+s, s) (all seeds get +s*I);
                      V = W1 * sum_p (tr(M^p) - c_p')^2
                          + Wd * (det M - D0)^2,  D0 = (1+s)(delta+s)s
                      det M   = (t1^3 - 3 t1 t2 + 2 t3)/6   (traces)
                      d(det)/dM = adj(M) = M^2 - t1 M
                                  + 0.5 (t1^2 - t2) I   (Cayley-Ham.)
    (T1, T3 complex-step checkable end-to-end; T2 gradient checked by
    central finite differences, eigh is not complex-step safe.)

GATES (pre-registered, cap 3 tries): G0 gradient check per
(term x stencil x eps) combo; Gadj adjoint identity per stencil;
G1 SO(3) conjugation invariance + negative control; G2 vacuum == 0
per term set; G3 seed far spectra.

CONSISTENCY READ (every endpoint row): E_u re-read under fwd / bwd /
2h at the run h + factor-2 subsample probes (both parities, h -> 2h);
xstencil_ratio = max/min of the re-reads. The I1 PASS bar: <= 1.5.

MODES: calib | gates | relax key=val... | collect | status
    relax keys: seed=A|B|C|R term=T1|T2|T3 stencil=sym|fwd|bwd|2h
        eps=0 n=32 L=48 delta=0.3 bc=pinned|free maxit=6000 aring=4.0
        shift=0.3 w2=... wdet=... snaps=0|1 tag=...
Out: ../data/m5_21_2b_row_<tag>.json (+ endpoint npz), race-free per
run; collect merges rows -> ../data/m5_21_2b_all.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "L_ladder", "seeds")
PLOTS = os.path.join(HERE, "..", "plots")
JPATH = os.path.join(DATA, "m5_21_2b_all.json")

W1 = 0.000724023879              # the M5.21.2 WSCALE, carried
DELTA0 = 0.3


# ================= grid + seeds (PHYSICAL coordinates) =================
def coords(n, h):
    x = (np.arange(n) - (n - 1) / 2.0) * h
    return np.meshgrid(x, x, x, indexing="ij")


def frame(n, h):
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    rs = np.where(r < 1e-12, 1e-12, r)
    rhat = np.stack([X / rs, Y / rs, Z / rs], axis=-1)
    rho = np.sqrt(X * X + Y * Y)
    rhos = np.where(rho < 1e-12, 1e-12, rho)
    phihat = np.stack([-Y / rhos, X / rhos, np.zeros_like(Z)], axis=-1)
    near = rho < 1e-9
    if np.any(near):
        phihat[near] = np.array([0.0, 1.0, 0.0])
    that = np.cross(phihat, rhat)
    return r, rhat, phihat, that


SEEDS = {"A": (1.0, None, 0.0), "B": (None, 0.0, 1.0),
         "C": (0.0, 1.0, None)}


def seed3(n, h, delta, which, r_c=4.0):
    lam = [delta if v is None else v for v in SEEDS[which]]
    r, rhat, phihat, that = frame(n, h)
    S = (lam[0] * rhat[..., :, None] * rhat[..., None, :]
         + lam[1] * phihat[..., :, None] * phihat[..., None, :]
         + lam[2] * that[..., :, None] * that[..., None, :])
    a = (1.0 + delta) / 3.0
    w = (1.0 - np.exp(-((r / r_c) ** 2)))[..., None, None]
    return w * S + (1.0 - w) * (a * np.eye(3))


def seed_ring(n, h, delta, a=4.0, r_c=2.5):
    """the charged disclination ring, seed-A far field (M5.21.2
    construction, physical units): psi = 0.5*(arg pair), escaped
    interior, delta on phihat."""
    X, Y, Z = coords(n, h)
    rho = np.sqrt(X * X + Y * Y)
    rhos = np.where(rho < 1e-12, 1e-12, rho)
    phihat = np.stack([-Y / rhos, X / rhos, np.zeros_like(Z)], axis=-1)
    rhohat = np.stack([X / rhos, Y / rhos, np.zeros_like(Z)], axis=-1)
    psi = 0.5 * (np.arctan2(rho - a, Z) + np.arctan2(rho + a, Z))
    zhat = np.array([0.0, 0.0, 1.0])
    nvec = np.sin(psi)[..., None] * rhohat + np.cos(psi)[..., None] * zhat
    S = (nvec[..., :, None] * nvec[..., None, :]
         + delta * phihat[..., :, None] * phihat[..., None, :])
    d_ring = np.sqrt((rho - a) ** 2 + Z ** 2)
    av = (1.0 + delta) / 3.0
    w = (1.0 - np.exp(-((d_ring / r_c) ** 2)))[..., None, None]
    return w * S + (1.0 - w) * (av * np.eye(3))


def make_seed(cfg):
    n, h, delta = cfg["n"], cfg["h"], cfg["delta"]
    M = seed_ring(n, h, delta, a=cfg["aring"]) if cfg["seed"] == "R" \
        else seed3(n, h, delta, cfg["seed"])
    if cfg["term"] == "T3":
        M = M + cfg["shift"] * np.eye(3)
    return M


def vac_diag(cfg):
    d, s = cfg["delta"], (cfg["shift"] if cfg["term"] == "T3" else 0.0)
    return np.array([1.0 + s, d + s, s])


def pin_shell(n, h, depth=1.6):
    """outer shell of PHYSICAL depth (>= depth units) pinned."""
    wc = max(1, int(np.ceil(depth / h)))
    P = np.zeros((n, n, n), dtype=bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, wc); P[tuple(sl)] = True
        sl[ax] = slice(n - wc, n); P[tuple(sl)] = True
    return P


# ================= stencils (h-aware, exact adjoints) =================
def d1(f, ax, h, st):
    """single-stencil derivative: fwd | bwd | 2h (central + one-sided
    edges)."""
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim

    def at(i):
        s = list(sl); s[ax] = i; return tuple(s)
    if st == "fwd":
        out[at(slice(0, -1))] = (f[at(slice(1, None))]
                                 - f[at(slice(0, -1))]) / h
    elif st == "bwd":
        out[at(slice(1, None))] = (f[at(slice(1, None))]
                                   - f[at(slice(0, -1))]) / h
    else:                                              # 2h
        out[at(slice(1, -1))] = (f[at(slice(2, None))]
                                 - f[at(slice(0, -2))]) / (2 * h)
        out[at(0)] = (f[at(1)] - f[at(0)]) / h
        out[at(-1)] = (f[at(-1)] - f[at(-2)]) / h
    return out


def d1_adj(g, ax, h, st):
    out = np.zeros_like(g)
    sl = [slice(None)] * g.ndim

    def at(i):
        s = list(sl); s[ax] = i; return tuple(s)
    if st == "fwd":
        out[at(slice(1, None))] += g[at(slice(0, -1))] / h
        out[at(slice(0, -1))] -= g[at(slice(0, -1))] / h
    elif st == "bwd":
        out[at(slice(1, None))] += g[at(slice(1, None))] / h
        out[at(slice(0, -1))] -= g[at(slice(1, None))] / h
    else:
        out[at(slice(2, None))] += g[at(slice(1, -1))] / (2 * h)
        out[at(slice(0, -2))] -= g[at(slice(1, -1))] / (2 * h)
        out[at(1)] += g[at(0)] / h
        out[at(0)] -= g[at(0)] / h
        out[at(-1)] += g[at(-1)] / h
        out[at(-2)] -= g[at(-1)] / h
    return out


def branches(st):
    """the stencil branches averaged: sym = (fwd + bwd)/2."""
    return [("fwd", 0.5), ("bwd", 0.5)] if st == "sym" else [(st, 1.0)]


# ================= energy pieces =================
def tr_pows(M):
    M2 = M @ M
    t1 = np.einsum("...kk->...", M)
    t2 = np.einsum("...kk->...", M2)
    t3 = np.einsum("...kk->...", M2 @ M)
    return M2, t1, t2, t3


def det3(t1, t2, t3):
    return (t1 ** 3 - 3.0 * t1 * t2 + 2.0 * t3) / 6.0


def adj3(M, M2, t1, t2):
    return M2 - t1[..., None, None] * M \
        + (0.5 * (t1 ** 2 - t2))[..., None, None] * np.eye(3)


def v_density(M, cfg):
    """potential density per cell (complex-safe for T1/T3)."""
    d = cfg["delta"]
    if cfg["term"] == "T2":
        lam = np.linalg.eigvalsh(M)
        v = np.sort(vac_diag(cfg))
        return cfg["w2"] * np.sum((lam - v) ** 2, axis=-1)
    s = cfg["shift"] if cfg["term"] == "T3" else 0.0
    vd = vac_diag(cfg)
    cp = [float(np.sum(vd ** p)) for p in (1, 2, 3)]
    M2, t1, t2, t3 = tr_pows(M)
    out = W1 * ((t1 - cp[0]) ** 2 + (t2 - cp[1]) ** 2
                + (t3 - cp[2]) ** 2)
    if cfg["term"] == "T3":
        D0 = float(np.prod(vd))
        out = out + cfg["wdet"] * (det3(t1, t2, t3) - D0) ** 2
    return out


def v_grad(M, cfg):
    if cfg["term"] == "T2":
        lam, V = np.linalg.eigh(M)
        v = np.sort(vac_diag(cfg))
        w = 2.0 * cfg["w2"] * (lam - v)
        return np.einsum("...k,...ik,...jk->...ij", w, V, V)
    vd = vac_diag(cfg)
    cp = [float(np.sum(vd ** p)) for p in (1, 2, 3)]
    M2, t1, t2, t3 = tr_pows(M)
    eye = np.eye(3)
    G = W1 * (2.0 * (t1 - cp[0])[..., None, None] * eye
              + 4.0 * (t2 - cp[1])[..., None, None] * M
              + 6.0 * (t3 - cp[2])[..., None, None] * M2)
    if cfg["term"] == "T3":
        D0 = float(np.prod(vd))
        G = G + (2.0 * cfg["wdet"]
                 * (det3(t1, t2, t3) - D0))[..., None, None] \
            * adj3(M, M2, t1, t2)
    return G


def e_parts(M, cfg, st=None):
    """(E_u, E_D, E_V) under stencil st (default cfg), h^3-weighted."""
    st = st or cfg["stencil"]
    h = cfg["h"]
    e_u = 0.0
    e_d = 0.0
    for br, wt in branches(st):
        A = [d1(M, ax, h, br) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                C = A[i] @ A[j] - A[j] @ A[i]
                e_u = e_u + wt * 4.0 * np.sum(
                    np.einsum("...kl,...kl->...", C, C))
        for i in range(3):
            e_d = e_d + wt * np.sum(
                np.einsum("...kl,...kl->...", A[i], A[i]))
    e_v = np.sum(v_density(M, cfg))
    h3 = h ** 3
    return h3 * e_u, h3 * cfg["eps"] * e_d, h3 * e_v


def e_total(M, cfg, st=None):
    a, b, c = e_parts(M, cfg, st)
    return a + b + c


def grad(M, cfg):
    h = cfg["h"]
    G = np.zeros_like(M)
    for br, wt in branches(cfg["stencil"]):
        A = [d1(M, ax, h, br) for ax in range(3)]
        dA = [np.zeros_like(M) for _ in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                C = A[i] @ A[j] - A[j] @ A[i]
                dA[i] += 8.0 * (C @ A[j] - A[j] @ C)
                dA[j] -= 8.0 * (C @ A[i] - A[i] @ C)
        for i in range(3):
            dA[i] += 2.0 * cfg["eps"] * A[i]
        for ax in range(3):
            G += wt * d1_adj(dA[ax], ax, h, br)
    G += v_grad(M, cfg)
    return (h ** 3) * G


# ================= diagnostics =================
def eig3(M):
    return np.linalg.eigh(M)


def retention(M, cfg, r_lo=8.0, r_hi=16.0):
    which = "A" if cfg["seed"] == "R" else cfg["seed"]
    n, h, delta = cfg["n"], cfg["h"], cfg["delta"]
    r, rhat, phihat, that = frame(n, h)
    lam, V = eig3(M)
    lam_seed = [delta if v is None else v for v in SEEDS[which]]
    order = np.argsort(lam_seed)
    refs = [rhat, phihat, that]
    sel = (r >= r_lo) & (r <= r_hi)
    vals = []
    for k in range(3):
        l = int(order[k])
        vk = V[sel][..., :, k]
        vals.append(float(np.mean(np.sum(vk * refs[l][sel],
                                         axis=-1) ** 2)))
    return {"per_eig": vals, "mean": float(np.mean(vals))}


def r_half(M, cfg):
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    dv = v_density(M, cfg)
    order = np.argsort(r.ravel())
    cum = np.cumsum(dv.ravel()[order])
    if cum[-1] <= 0:
        return float("nan")
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(r.ravel()[order][min(k, len(order) - 1)])


def ring_read(M, cfg):
    """cord-radius read (Q30): potential-excess-weighted geometry of
    the high-excess set (> 0.5 max), + the peak-cell location."""
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = coords(n, h)
    rho = np.sqrt(X * X + Y * Y)
    r = np.sqrt(rho * rho + Z * Z)
    vd = v_density(M, cfg)
    i = np.unravel_index(int(np.argmax(vd)), vd.shape)
    hi = vd > 0.5 * vd.max()
    wsum = float(vd[hi].sum())
    out = {"vmax_rho": float(rho[i]), "vmax_z": float(Z[i]),
           "vmax_r": float(r[i]), "n_hi": int(hi.sum())}
    if wsum > 0:
        out["rho_w"] = float((vd[hi] * rho[hi]).sum() / wsum)
        out["z_rms_w"] = float(np.sqrt((vd[hi] * Z[hi] ** 2).sum()
                                       / wsum))
    return out


def consistency(M, cfg):
    """the endpoint cross-instrument read: E_u under fwd/bwd/2h at h,
    + factor-2 subsample probes (h -> 2h) both parities."""
    reads = {}
    for st in ("fwd", "bwd", "2h"):
        reads[f"E_u_{st}"] = float(e_parts(M, cfg, st)[0])
    for par in (0, 1):
        sub = M[par::2, par::2, par::2]
        c2 = dict(cfg); c2["h"] = 2 * cfg["h"]; c2["n"] = sub.shape[0]
        reads[f"E_u_sub{par}"] = float(e_parts(sub, c2, "fwd")[0])
    vals = [v for v in reads.values() if v > 0]
    reads["xstencil_ratio"] = float(max(vals) / max(min(vals), 1e-300))
    return reads


def min_gap(M):
    lam = np.linalg.eigvalsh(M)
    g = np.minimum(lam[..., 1] - lam[..., 0], lam[..., 2] - lam[..., 1])
    return float(g.min())


# ================= FIRE =================
def fire(M0, cfg, free_mask, max_iter, log_every=500, snaps=(),
         tag="", f_tol=1e-8, plateau=(2000, 1e-10), dt0=0.02,
         dt_max=0.2):
    M = M0.copy()
    free = free_mask[..., None, None].astype(float)
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    hist, states = [], [{"it": 0, "M": M0.copy()}]
    F = -grad(M, cfg) * free
    t0 = time.time()
    stop = "max_iter"
    for it in range(1, max_iter + 1):
        P = float(np.sum(F * v))
        if P > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_up = 0
        v += dt * F
        M += dt * v
        F = -grad(M, cfg) * free
        fmax = float(np.max(np.abs(F)))
        if not np.isfinite(fmax):
            stop = "non-finite"
            break
        if it % log_every == 0 or it == max_iter:
            e_u, e_d, e_v = e_parts(M, cfg)
            E = e_u + e_d + e_v
            hist.append({"it": it, "E": float(E), "E_u": float(e_u),
                         "E_d": float(e_d), "E_v": float(e_v),
                         "fmax": fmax, "dt": dt})
            print(f"  {tag} it {it:6d} E {E:12.6f} "
                  f"u {e_u:10.5f} d {e_d:9.5f} v {e_v:9.5f} "
                  f"fmax {fmax:.3e} [{time.time() - t0:.0f}s]",
                  flush=True)
            back = plateau[0] // max(log_every, 1)
            if len(hist) > back and \
                    abs(E - hist[-1 - back]["E"]) < plateau[1]:
                stop = "plateau"
                break
        if fmax < f_tol:
            stop = "f_tol"
            break
        if it in snaps:
            states.append({"it": it, "M": M.copy()})
    states.append({"it": -1, "M": M.copy()})
    return M, states, {"stop": stop, "trace": hist,
                       "wall_s": time.time() - t0}


# ================= gates =================
def base_cfg(**kw):
    cfg = {"seed": "A", "term": "T1", "stencil": "sym", "eps": 0.0,
           "n": 32, "L": 48.0, "delta": DELTA0, "bc": "pinned",
           "maxit": 6000, "aring": 4.0, "shift": 0.3, "w2": W1,
           "wdet": W1, "snaps": 0, "tag": ""}
    cfg.update(kw)
    cfg["h"] = cfg["L"] / cfg["n"]
    return cfg


def gates():
    rng = np.random.default_rng(11)
    out = {}
    # Gadj: adjoint identity per stencil branch
    f = rng.normal(size=(9, 9, 9, 3, 3))
    g = rng.normal(size=f.shape)
    for st in ("fwd", "bwd", "2h"):
        errs = []
        for ax in range(3):
            a = float(np.sum(d1(f, ax, 1.3, st) * g))
            b = float(np.sum(f * d1_adj(g, ax, 1.3, st)))
            errs.append(abs(a - b) / max(abs(a), 1e-300))
        out[f"adj_{st}"] = float(np.max(errs))
    # G0 gradient checks per (term x stencil x eps)
    for term in ("T1", "T2", "T3"):
        for st in ("sym", "fwd", "2h"):
            for eps in (0.0, 0.01):
                cfg = base_cfg(term=term, stencil=st, eps=eps, n=10,
                               L=15.0)
                M = rng.normal(size=(10, 10, 10, 3, 3))
                M = 0.5 * (M + M.swapaxes(-1, -2))
                G = grad(M, cfg)
                errs = []
                for _ in range(3):
                    Vv = rng.normal(size=M.shape)
                    Vv = 0.5 * (Vv + Vv.swapaxes(-1, -2))
                    de_an = float(np.sum(G * Vv))
                    if term == "T2":            # eigh: central FD
                        t = 1e-5
                        de = (e_total(M + t * Vv, cfg)
                              - e_total(M - t * Vv, cfg)) / (2 * t)
                        errs.append(abs(de - de_an)
                                    / max(abs(de), 1e-300))
                    else:                        # complex step
                        t = 1e-30
                        de = e_total(M + 1j * t * Vv, cfg).imag / t
                        errs.append(abs(de - de_an)
                                    / max(abs(de), 1e-300))
                out[f"g0_{term}_{st}_e{eps}"] = float(np.max(errs))
    # G1 SO(3) conjugation invariance + negative control (per term)
    for term in ("T1", "T2", "T3"):
        cfg = base_cfg(term=term, n=20, L=30.0, eps=0.01)
        M = make_seed(cfg)
        E0 = float(e_total(M, cfg))
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        MR = np.einsum("ab,...bc,dc->...ad", Q, M, Q)
        out[f"g1_{term}_so3"] = abs(float(e_total(MR, cfg)) - E0) \
            / abs(E0)
        Qb = Q + 0.05 * rng.normal(size=(3, 3))
        MB = np.einsum("ab,...bc,dc->...ad", Qb, M, Qb)
        out[f"g1_{term}_negctrl"] = abs(float(e_total(MB, cfg)) - E0) \
            / abs(E0)
    # G2 vacuum == 0 per term set (shifted vacuum for T3)
    for term in ("T1", "T2", "T3"):
        cfg = base_cfg(term=term, n=12, L=18.0, eps=0.01)
        Mv = np.zeros((12, 12, 12, 3, 3))
        Mv[:] = np.diag(vac_diag(cfg))
        out[f"g2_{term}_vacE"] = float(e_total(Mv, cfg))
    # G3 far spectra (physical box, two h rungs must agree)
    for n in (32, 48):
        cfg = base_cfg(n=n)
        M = make_seed(cfg)
        lam = np.linalg.eigvalsh(M)
        X, Y, Z = coords(n, cfg["h"])
        r = np.sqrt(X * X + Y * Y + Z * Z)
        far = (r > 14) & (r < 20)
        out[f"g3_far_A_n{n}"] = lam[far].mean(axis=0).tolist()
    out["g0_pass"] = max(v for k, v in out.items()
                         if k.startswith("g0_")) < 5e-6
    out["adj_pass"] = max(v for k, v in out.items()
                          if k.startswith("adj_")) < 1e-12
    out["g1_pass"] = (max(v for k, v in out.items()
                          if k.endswith("_so3")) < 1e-10
                      and min(v for k, v in out.items()
                              if k.endswith("_negctrl")) > 1e-6)
    out["g2_pass"] = max(abs(v) for k, v in out.items()
                         if k.startswith("g2_")) < 1e-12
    with open(os.path.join(DATA, "m5_21_2b_gates.json"), "w") as f2:
        json.dump(out, f2, indent=1)
    print(json.dumps(out, indent=1))
    return out


def calib():
    """seed-A energy split at the working point: sets the eps grid +
    the T2/T3 weight normalizations (recorded, then passed explicitly
    to every relax run)."""
    out = {}
    for n in (32, 48):
        cfg = base_cfg(n=n, eps=1.0)      # eps=1 -> E_D raw
        M = make_seed(cfg)
        e_u, e_d, e_v = e_parts(M, cfg)
        out[f"n{n}"] = {"E_u": e_u, "E_D_raw": e_d, "E_V_T1": e_v,
                        "eps_for_5pct": 0.05 * e_u / e_d,
                        "eps_for_1pct": 0.01 * e_u / e_d}
        # T2 raw (w2 = 1)
        c2 = base_cfg(n=n, term="T2", w2=1.0)
        out[f"n{n}"]["V_T2_raw"] = float(np.sum(v_density(M, c2))
                                         * cfg["h"] ** 3)
        out[f"n{n}"]["w2_match"] = e_v / out[f"n{n}"]["V_T2_raw"]
        # T3 det part raw (wdet = 1) on the shifted seed
        c3 = base_cfg(n=n, term="T3", wdet=1.0)
        M3 = make_seed(c3)
        vd_full = float(np.sum(v_density(M3, c3)) * c3["h"] ** 3)
        c3b = dict(c3); c3b["wdet"] = 0.0
        vd_tr = float(np.sum(v_density(M3, c3b)) * c3["h"] ** 3)
        out[f"n{n}"]["V_T3_trace"] = vd_tr
        out[f"n{n}"]["V_T3_det_raw"] = vd_full - vd_tr
        out[f"n{n}"]["wdet_match"] = vd_tr / max(vd_full - vd_tr,
                                                 1e-300)
    with open(os.path.join(DATA, "m5_21_2b_calib.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


# ================= the relax workhorse =================
def relax(cfg):
    tag = cfg["tag"] or (f"{cfg['seed']}_{cfg['term']}_{cfg['stencil']}"
                         f"_e{cfg['eps']:g}_n{cfg['n']}_d{cfg['delta']:g}"
                         f"_{cfg['bc']}")
    M0 = make_seed(cfg)
    n = cfg["n"]
    free = ~pin_shell(n, cfg["h"]) if cfg["bc"] == "pinned" else \
        np.ones((n, n, n), dtype=bool)
    e0 = e_parts(M0, cfg)
    snaps = (500, 1000, 2000, 4000, 8000) if cfg["snaps"] else ()
    M, states, info = fire(M0, cfg, free, max_iter=cfg["maxit"],
                           log_every=500, snaps=snaps, tag=tag)
    e_u, e_d, e_v = e_parts(M, cfg)
    row = {k: cfg[k] for k in ("seed", "term", "stencil", "eps", "n",
                               "L", "h", "delta", "bc", "maxit",
                               "aring", "shift", "w2", "wdet")}
    row.update({
        "tag": tag, "E_end": float(e_u + e_d + e_v),
        "E_u": float(e_u), "E_d": float(e_d), "E_v": float(e_v),
        "E_seed": float(sum(e0)),
        "virial_resid": float((-e_u + e_d + 3 * e_v)
                              / max(e_u + e_d + e_v, 1e-300)),
        "u_over_3v": float(e_u / max(3 * e_v, 1e-300)),
        "r_half": r_half(M, cfg),
        "retention": retention(M, cfg),
        "ring": ring_read(M, cfg),
        "ring_seed": ring_read(M0, cfg),
        "consistency": consistency(M, cfg),
        "min_gap_end": min_gap(M),
        "stop": info["stop"], "trace": info["trace"][-6:],
        "wall_s": info["wall_s"]})
    os.makedirs(DATA, exist_ok=True)
    snap_arrays = {f"M_it{st['it']}": st["M"].astype(np.float32)
                   for st in states if st["it"] > 0}
    np.savez_compressed(
        os.path.join(DATA, f"m5_21_2b_end_{tag}.npz"),
        M=M.astype(np.float32), delta=cfg["delta"], h=cfg["h"],
        term=cfg["term"], eps=cfg["eps"], maxit=cfg["maxit"],
        **snap_arrays)
    with open(os.path.join(DATA, f"m5_21_2b_row_{tag}.json"),
              "w") as f:
        json.dump(row, f, indent=1)
    print(json.dumps({k: row[k] for k in
                      ("tag", "E_end", "E_u", "E_d", "E_v",
                       "u_over_3v", "virial_resid", "r_half", "stop")}
                     | {"xratio":
                        row["consistency"]["xstencil_ratio"]}))
    return row


# ================= collect / status =================
def collect():
    import glob
    rows = {}
    for p in sorted(glob.glob(os.path.join(DATA,
                                           "m5_21_2b_row_*.json"))):
        key = os.path.basename(p)[len("m5_21_2b_row_"):-len(".json")]
        with open(p) as f:
            rows[key] = json.load(f)
    with open(JPATH, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"collected {len(rows)} rows -> {os.path.basename(JPATH)}")


def status():
    if os.path.exists(JPATH):
        with open(JPATH) as f:
            J = json.load(f)
        for k, r in sorted(J.items()):
            print(f"{k:44s} E {r['E_end']:10.5f} xr "
                  f"{r['consistency']['xstencil_ratio']:8.2f} "
                  f"u/3V {r['u_over_3v']:7.3f} stop {r['stop']}")
    else:
        print("no collected json yet")


def parse_kv(argv):
    kw = {}
    casts = {"eps": float, "n": int, "L": float, "delta": float,
             "maxit": int, "aring": float, "shift": float,
             "w2": float, "wdet": float, "snaps": int}
    for a in argv:
        k, v = a.split("=", 1)
        kw[k] = casts[k](v) if k in casts else v
    return kw


if __name__ == "__main__":
    ARGV = sys.argv[1:]
    mode = ARGV[0] if ARGV else "status"
    if mode == "gates":
        gates()
    elif mode == "calib":
        calib()
    elif mode == "relax":
        relax(base_cfg(**parse_kv(ARGV[1:])))
    elif mode == "collect":
        collect()
    else:
        status()
