"""Review round 2, finding 2 (constructive part): the combined
space-internal generator zeta = -(x d_y - y d_x) + [W, .] measured on
(1) the working polished field, and, as ANALYTIC controls,
(2) the uniaxial hedgehog M = -8 e0e0^T + r^ r^^T and
(3) the spherical-frame biaxial hedgehog
    M = -8 e0e0^T + r^ r^^T + delta th^ th^^T  (delta = 0.3),
whose eigenframe is the spherical basis (r^, th^, ph^) -- an axially
EQUIVARIANT texture in the reviewer's counterexample class. Off-axis
shell profiles of |zeta M| separate texture choice from biaxiality:
the equivariant textures leave only a 1/r discretization residual at
ANY delta, while the working radial texture carries an O(1) flat
residual. Out: results/combined_texture.json
"""
import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
R008 = os.path.join(HERE, "..", "008-i1-squared-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "texture_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("combined_texture: NOT REPRODUCED HERE -- needs report "
          f"004's stack in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

lad = runpy.run_path(os.path.join(R008, "ladder_i1sq_defs.py"))
field, M_pol = lad["field"], lad["M_pol"]
L004 = runpy.run_path(os.path.join(R004, "lattice.py"),
                      run_name="not_main")
d1, comm, G_of = L004["d1"], L004["comm"], L004["G_of"]
DT, DEV, H, N = L004["DT"], L004["DEV"], L004["H"], L004["N"]
W = L004["gen_catalog"]()["rot_xy"]
interior = (1.0 - L004["SHELL"].to(DT))[..., None, None]
X, Y, Z = L004["coords"]()

eps = 1e-9
R3 = torch.stack([X, Y, Z], dim=-1)
r = R3.norm(dim=-1, keepdim=True).clamp_min(eps)
rh = R3 / r
rho = torch.sqrt(X ** 2 + Y ** 2).clamp_min(eps)
ph = torch.stack([-Y / rho, X / rho, torch.zeros_like(X)], dim=-1)
th = torch.cross(ph, rh, dim=-1)


def spatial_outer(v):
    return v[..., :, None] * v[..., None, :]


def embed(S3):
    M = torch.zeros(N, N, N, 4, 4, dtype=DT, device=DEV)
    M[..., 1:4, 1:4] = S3
    M[..., 0, 0] = -8.0
    return M


def measure(M):
    dMy, dMx = d1(M, 1, "fwd"), d1(M, 0, "fwd")
    orbital = -(X[..., None, None] * dMy - Y[..., None, None] * dMx)
    internal = (torch.einsum("ab,...bc->...ac", W, M)
                - torch.einsum("...ab,bc->...ac", M, W))
    z = interior * (orbital + internal)
    G = G_of(M)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(z, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
    I = float(2.0 * H ** 3 * k.sum())
    zn = z.norm(dim=(-2, -1))
    c = N // 2
    idx = torch.stack(torch.meshgrid(
        torch.arange(N), torch.arange(N), torch.arange(N),
        indexing="ij")).to(DT).to(DEV)
    rr = torch.sqrt(((idx - c) ** 2).sum(dim=0)) * H
    off = rho > 3.0
    prof, rmids = [], []
    for r0 in (3.0, 6.0, 9.0, 12.0, 15.0, 18.0):
        sel = (rr >= r0) & (rr < r0 + 3.0) & off
        prof.append(float(zn[sel].mean()))
        rmids.append(r0 + 1.5)
    return {"I_comb": I, "r_mid": rmids, "zeta_profile": prof}


out = {}
for tag, M in (("working", field(M_pol)),
               ("uniaxial", embed(spatial_outer(rh))),
               ("spherical_biax",
                embed(spatial_outer(rh) + 0.3 * spatial_outer(th)))):
    m = measure(M)
    out[tag] = m
    print(f"{tag}: I_comb {m['I_comb']:.4e}; |zeta|(r) "
          + " ".join(f"{p:.4f}" for p in m["zeta_profile"]),
          flush=True)

# --- the third class (review round 4's counterexample, measured) ---
# asymptotically equivariant, non-equivariant CORE:
#   M = M_sph + eps * chi(r) * D,  D = x^x^T - y^y^T (constant, so the
# orbital part of zeta annihilates it and zeta M = eps chi [W, D]_c is
# compactly supported), chi = exp(-(r/r0)^2). Measured on its own
# analytic lattice at three box sizes (fixed spacing) to test the
# L-independence of the inertia.
EPS, R0 = 0.3, 6.0
D3 = torch.zeros(3, 3, dtype=DT, device=DEV)
D3[0, 0], D3[1, 1] = 1.0, -1.0


def build_core_deformed(n):
    hh = 1.5
    ax = (torch.arange(n, dtype=DT, device=DEV) - (n - 1) / 2) * hh
    Xn, Yn, Zn = torch.meshgrid(ax, ax, ax, indexing="ij")
    R3n = torch.stack([Xn, Yn, Zn], dim=-1)
    rn = R3n.norm(dim=-1, keepdim=True).clamp_min(eps)
    rhn = R3n / rn
    rhon = torch.sqrt(Xn ** 2 + Yn ** 2).clamp_min(eps)
    phn = torch.stack([-Yn / rhon, Xn / rhon,
                       torch.zeros_like(Xn)], dim=-1)
    thn = torch.cross(phn, rhn, dim=-1)
    chi = torch.exp(-(rn.squeeze(-1) / R0) ** 2)
    S3 = (rhn[..., :, None] * rhn[..., None, :]
          + 0.3 * thn[..., :, None] * thn[..., None, :]
          + EPS * chi[..., None, None] * D3)
    M = torch.zeros(n, n, n, 4, 4, dtype=DT, device=DEV)
    M[..., 1:4, 1:4] = S3
    M[..., 0, 0] = -8.0
    return M, Xn, Yn, rhon


def measure_on(M, n, Xn, Yn, rhon):
    """Same measurement as measure(), on an n^3 analytic lattice."""
    hh = 1.5
    shell = torch.zeros(n, n, n, dtype=torch.bool, device=DEV)
    for ax_ in range(3):
        sl = [slice(None)] * 3
        sl[ax_] = 0
        shell[tuple(sl)] = True
        sl[ax_] = n - 1
        shell[tuple(sl)] = True
    inter = (~shell).to(DT)[..., None, None]

    def d1n(f, ax_, st):
        o = torch.zeros_like(f)
        lo = [slice(None)] * f.dim()
        hi = [slice(None)] * f.dim()
        sl = [slice(None)] * f.dim()
        lo[ax_], hi[ax_] = slice(0, -1), slice(1, None)
        if st == "fwd":
            sl[ax_] = slice(0, -1)
        else:
            sl[ax_] = slice(1, None)
        o[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / hh
        return o

    dMy, dMx = d1n(M, 1, "fwd"), d1n(M, 0, "fwd")
    orbital = -(Xn[..., None, None] * dMy - Yn[..., None, None] * dMx)
    internal = (torch.einsum("ab,...bc->...ac", W, M)
                - torch.einsum("...ab,bc->...ac", M, W))
    z = inter * (orbital + internal)
    G = G_of(M)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1n(M, ax_, st) for ax_ in range(3)]
        for i in range(3):
            F0 = comm(z, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
    I = float(2.0 * hh ** 3 * k.sum())
    pr = float((k.sum() ** 2) / (k ** 2).sum().clamp_min(1e-30))
    zn = z.norm(dim=(-2, -1))
    c = (n - 1) / 2
    idx = torch.stack(torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n),
        indexing="ij")).to(DT).to(DEV)
    rr = torch.sqrt(((idx - c) ** 2).sum(dim=0)) * hh
    off = rhon > 3.0
    prof = []
    for r0_ in (3.0, 6.0, 9.0, 12.0, 15.0, 18.0):
        sel = (rr >= r0_) & (rr < r0_ + 3.0) & off
        prof.append(float(zn[sel].mean()) if sel.any() else None)
    return {"I_comb": I, "PR": pr, "zeta_profile": prof}


cd = {"eps": EPS, "r0": R0, "boxes": []}
for n in (16, 24, 32):
    M, Xn, Yn, rhon = build_core_deformed(n)
    m = measure_on(M, n, Xn, Yn, rhon)
    m["N"], m["Lbox"] = n, n * 1.5
    cd["boxes"].append(m)
    zp = " ".join("None" if v is None else f"{v:.4f}"
                  for v in m["zeta_profile"])
    print(f"core_deformed N {n}: I_comb {m['I_comb']:.4e} "
          f"(PR {m['PR']:.0f}); |zeta|(r) {zp}", flush=True)
Is = [b["I_comb"] for b in cd["boxes"]]
Ls = [b["Lbox"] for b in cd["boxes"]]
expo = float(np.polyfit(np.log(Ls), np.log(Is), 1)[0])
cd["exponent"] = expo
print(f"core_deformed inertia ~ L^{expo:.3f} (finite: exponent ~ 0)")
out["core_deformed"] = cd

json.dump(out, open(os.path.join(HERE, "results",
                                 "combined_texture.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("combined texture test computed in this run\n")
print("written: results/combined_texture.json")
