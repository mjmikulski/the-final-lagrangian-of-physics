"""Report 011 shared definitions: the equivariant (EQ) and core-broken
(CB) analytic seeds, stack loading with patched (N, L, delta), the
relaxation protocol (deep from the start -- the 009 lesson), and the
measurement kit (zeta residual, inertia, tube line-tensions).

Seeds (spatial 3x3 block; time row fixed at -8):
  EQ: M = r^ r^^T + delta * g(rho) th^ th^^T,  g = 1 - exp(-(rho/rho0)^2)
      (spherical transverse frame; g regularizes the z-axis frame
      singularity -- the transverse amplitude escapes to zero on the
      axis, which is precisely the axial-defect core)
  CB: EQ + eps * chi(r) * D,  D = x^x^T - y^y^T,  chi = exp(-(r/r0)^2)
      (the round-4 third class: asymptotically equivariant,
      non-equivariant core)
The shell is frozen at the seed formula (equivariant boundary for both:
CB's deformation is compactly supported away from the shell).
"""
import os
import runpy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
SRC = open(os.path.join(R004, "lattice.py")).read()

RHO0, EPS, R0 = 3.0, 0.3, 6.0


def load_stack(N=32, Lbox=None, delta=0.3):
    Lbox = 1.5 * N if Lbox is None else Lbox
    src = SRC.replace("N, L = 32, 48.0", f"N, L = {N}, {float(Lbox)}")
    src = src.replace("SG, DELTA, W1 = 8.0, 0.3, 0.000724023879",
                      f"SG, DELTA, W1 = 8.0, {delta}, 0.000724023879")
    ns = {"__name__": "not_main",
          "__file__": os.path.join(R004, "lattice.py")}
    exec(compile(src, "lattice_patched", "exec"), ns)
    return ns


def geometry(L):
    N, H = L["N"], L["H"]
    DT, DEV = L["DT"], L["DEV"]
    ax = (torch.arange(N, dtype=DT, device=DEV) - (N - 1) / 2) * H
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    R3 = torch.stack([X, Y, Z], dim=-1)
    r = R3.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    rh = R3 / r
    rho = torch.sqrt(X ** 2 + Y ** 2).clamp_min(1e-9)
    ph = torch.stack([-Y / rho, X / rho, torch.zeros_like(X)], dim=-1)
    th = torch.cross(ph, rh, dim=-1)
    return X, Y, Z, r.squeeze(-1), rho, rh, th


def seed(L, kind, delta=0.3):
    N = L["N"]
    DT, DEV = L["DT"], L["DEV"]
    X, Y, Z, r, rho, rh, th = geometry(L)
    g = 1.0 - torch.exp(-(rho / RHO0) ** 2)
    S3 = (rh[..., :, None] * rh[..., None, :]
          + delta * g[..., None, None]
          * (th[..., :, None] * th[..., None, :]))
    if kind == "CB":
        D3 = torch.zeros(3, 3, dtype=DT, device=DEV)
        D3[0, 0], D3[1, 1] = 1.0, -1.0
        chi = torch.exp(-(r / R0) ** 2)
        S3 = S3 + EPS * chi[..., None, None] * D3
    M = torch.zeros(N, N, N, 4, 4, dtype=DT, device=DEV)
    M[..., 1:4, 1:4] = S3
    M[..., 0, 0] = -8.0
    return M


def install_seed(L, M4):
    L["seed_embedded"] = lambda: M4
    L["SHELL_VALS"] = None


def relax(L, cycles=6, adam=1000):
    field, e_static = L["field"], L["e_static"]
    M_raw = L["seed_embedded"]().clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(adam):
        opt.zero_grad()
        e_static(field(M_raw), "G").backward()
        opt.step()
    E_levels = [float(e_static(field(M_raw), "G").detach())]
    for cyc in range(cycles):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                                 tolerance_grad=1e-9, tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_static(field(M_raw), "G")
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_static(field(M_raw), "G").detach()))
    g = torch.autograd.grad(e_static(field(M_raw), "G"), M_raw)[0]
    return field(M_raw.detach()), E_levels, float(g.abs().max())


def zeta_of(L, M):
    d1, DT = L["d1"], L["DT"]
    W = L["gen_catalog"]()["rot_xy"]
    interior = (1.0 - L["SHELL"].to(DT))[..., None, None]
    X, Y, _, _, _, _, _ = geometry(L)
    dMy, dMx = d1(M, 1, "fwd"), d1(M, 0, "fwd")
    orbital = -(X[..., None, None] * dMy - Y[..., None, None] * dMx)
    internal = (torch.einsum("ab,...bc->...ac", W, M)
                - torch.einsum("...ab,bc->...ac", M, W))
    return interior * (orbital + internal)


def kin_density(L, M, tan):
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV = L["DT"], L["DEV"]
    G = G_of(M)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(tan, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
    return k


def measures(L, M):
    """zeta residual profile, combined inertia + PR, static-energy
    density and the two line-tension tubes (z axis vs x axis)."""
    H = L["H"]
    z = zeta_of(L, M)
    kz = kin_density(L, M, z)
    I = float(2.0 * H ** 3 * kz.sum())
    pr = float((kz.sum() ** 2) / (kz ** 2).sum().clamp_min(1e-30))
    X, Y, Z, r, rho, _, _ = geometry(L)
    zn = z.norm(dim=(-2, -1))
    prof = []
    rmax = float(r.max()) * 0.85
    edges = np.arange(3.0, rmax, 3.0)
    off = rho > 3.0
    for r0_ in edges:
        sel = (r >= r0_) & (r < r0_ + 3.0) & off
        prof.append(float(zn[sel].mean()) if sel.any() else None)

    # static energy density (u + V4), per site, via e_static parts:
    # recompute densities the way e_static does, but keep them local
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV = L["DT"], L["DEV"]
    G = G_of(M)
    dens = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                dens = dens + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F, G, G, F)
    Me = M @ L["ETA"]
    C_P = L["C_P"]
    P, v4 = Me, 0.0
    for pw in range(4):
        if pw:
            P = P @ Me
        v4 = v4 + (torch.einsum("...kk->...", P) - C_P[pw]) ** 2
    dens = dens + L["W1"] * v4

    # tube line tensions: energy per unit length in a rho < 3 tube
    # around the z axis vs the x axis, away from the center (|axis|>6)
    rho_z = torch.sqrt(X ** 2 + Y ** 2)
    rho_x = torch.sqrt(Y ** 2 + Z ** 2)
    zone_z = (rho_z < 3.0) & (Z.abs() > 6.0)
    zone_x = (rho_x < 3.0) & (X.abs() > 6.0)
    len_z = float(2 * (Z.abs().max() - 6.0))
    lam_z = float(H ** 3 * dens[zone_z].sum()) / len_z
    lam_x = float(H ** 3 * dens[zone_x].sum()) / len_z
    return {"I_comb": I, "PR": pr, "zeta_profile": prof,
            "zeta_edges": [float(e) + 1.5 for e in edges],
            "lambda_z": lam_z, "lambda_x": lam_x,
            "lambda_axis_excess": lam_z - lam_x}
