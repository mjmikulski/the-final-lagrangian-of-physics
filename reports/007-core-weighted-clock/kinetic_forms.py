"""Pointwise kinetic forms: where does c open the clock direction?

The velocity-quadratic form of the working kinetic term (G metric of
report 002, PSD on the hedgehog per report 004) plus c times the
eta-based I4 channel, as a 10x10 form on the symmetric velocity Mdot:
- vacuum: both forms vanish identically (quartic theory is soft in
  vacuum) -- safe for every c;
- hedgehog core cells: PSD at c = 0; the first negative direction opens
  at a configuration-dependent threshold c_clock, scanned downward from 0.
  Per review round 1: the FULL core region (every cell of the frozen mask
  cw > 0.5) is swept and the negative-eigenvalue count just past threshold
  is asserted to be exactly ONE. Two clock-direction diagnostics: (i) the
  Rayleigh quotient of the form ON the boost tangent a0 -- the threshold
  c_a0 where the a0 channel itself turns negative (the physically loaded
  question for a c*I4 realization of the condensate) -- and (ii) the raw
  eigenvector overlap with a0 (reported honestly; the lowest mode of the
  pointwise form need not align with a0 even when the a0 channel opens).

Routes: torch autograd Hessian (route 1) and central finite differences
on the 10 dof (route 2), agreement asserted on the five sample cells.

Needs report 004's polished field (or M5_FIELDS_DIR); NOT-REPRODUCED
notice otherwise (committed results carry the recorded values).
Out: results/kinetic_forms.json
"""
import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
MP = os.path.join(FIELDS, "M_G_polished.npz")
if not os.path.exists(MP):
    flag = os.path.join(HERE, "results", "kinetic_ran.flag")
    if os.path.exists(flag):
        os.remove(flag)
    print("kinetic_forms: NOT REPRODUCED HERE -- needs report 004's "
          f"polished field in {FIELDS} (or M5_FIELDS_DIR). Committed "
          "results carry the recorded values (c_clock ~ -0.53..-0.65).")
    sys.exit(0)

L = runpy.run_path(os.path.join(R004, "lattice.py"), run_name="not_main")
d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
DT, DEV, ETA, NLAT = L["DT"], L["DEV"], L["ETA"], L["N"]

ETA_T = ETA
IDX = [(a, b) for a in range(4) for b in range(a, 4)]


def unpack(v):
    A0 = torch.zeros(4, 4, dtype=DT, device=DEV)
    for k, (a, b) in enumerate(IDX):
        A0[a, b] = A0[a, b] + v[k]
        if a != b:
            A0[b, a] = A0[b, a] + v[k]
    return A0


def dens_G(v, Asp, G):
    A0 = unpack(v)
    s = 0.0
    for i in range(3):
        F = A0 @ ETA_T @ Asp[i] - Asp[i] @ ETA_T @ A0
        s = s + torch.einsum("ab,ac,bd,cd->", F, G, G, F)
    return s


def dens_I4(v, Asp):
    A0 = unpack(v)
    F0 = [A0 @ ETA_T @ Asp[i] - Asp[i] @ ETA_T @ A0 for i in range(3)]
    Phi = torch.zeros(4, 4, dtype=DT, device=DEV)
    for i in range(3):
        Phi[i + 1, :] = Phi[i + 1, :] + (-1.0) * F0[i][0, :]
        Phi[0, :] = Phi[0, :] - F0[i][i + 1, :]
    return torch.einsum("nb,nm,bc,mc->", Phi, ETA_T, ETA_T, Phi)


def hessian_auto(fn):
    return torch.autograd.functional.hessian(
        fn, torch.zeros(10, dtype=DT, device=DEV))


def hessian_fd(fn, h=1e-4):
    Hm = torch.zeros(10, 10, dtype=DT, device=DEV)
    for i in range(10):
        for j in range(10):
            v = torch.zeros(10, dtype=DT, device=DEV)
            vpp = v.clone(); vpp[i] += h; vpp[j] += h
            vpm = v.clone(); vpm[i] += h; vpm[j] -= h
            vmp = v.clone(); vmp[i] -= h; vmp[j] += h
            vmm = v.clone(); vmm[i] -= h; vmm[j] -= h
            Hm[i, j] = (fn(vpp) - fn(vpm) - fn(vmp) + fn(vmm)) / (4 * h * h)
    return Hm


field, a0_of, gen_catalog = L["field"], L["a0_of"], L["gen_catalog"]
C_P = L["C_P"]
M = torch.tensor(np.load(MP)["M"], dtype=DT, device=DEV)
A_all = [d1(M, ax, "fwd") for ax in range(3)]
cells = [(NLAT // 2 + dx, NLAT // 2 + dy, NLAT // 2 + dz)
         for (dx, dy, dz) in ((0, 0, 0), (2, 0, 0), (0, 3, 0), (4, 4, 4),
                              (8, 0, 0))]
results = {"cells": []}

# vacuum check: zero spatial derivatives -> both forms vanish
Z = torch.zeros(3, 4, 4, dtype=DT, device=DEV)
Gv = G_of(M[0, 0, 0][None, None, None])[0, 0, 0]
hv = hessian_auto(lambda v: dens_G(v, Z, Gv)).abs().max().item()
h4 = hessian_auto(lambda v: dens_I4(v, Z)).abs().max().item()
print(f"vacuum forms: |K_G| = {hv:.1e}, |K4| = {h4:.1e} (soft for all c)")
assert hv < 1e-12 and h4 < 1e-12
results["vacuum_max"] = max(hv, h4)

worst_fd = 0.0
for cell in cells:
    Asp = torch.stack([A_all[ax][cell] for ax in range(3)])
    G = G_of(M[cell][None, None, None])[0, 0, 0]
    HG = hessian_auto(lambda v: dens_G(v, Asp, G))
    H4 = hessian_auto(lambda v: dens_I4(v, Asp))
    # route 2: finite differences
    HGf = hessian_fd(lambda v: dens_G(v, Asp, G))
    H4f = hessian_fd(lambda v: dens_I4(v, Asp))
    worst_fd = max(worst_fd,
                   ((HG - HGf).abs().max() / HG.abs().max()).item(),
                   ((H4 - H4f).abs().max()
                    / H4.abs().max().clamp_min(1e-30)).item())
    negs0 = int((torch.linalg.eigvalsh(HG) < -1e-10).sum())
    c_clock = None
    for cval in np.linspace(0, -4, 801):
        if int((torch.linalg.eigvalsh(HG + float(cval) * H4)
                < -1e-10).sum()) > 0:
            c_clock = round(float(cval), 3)
            break
    off = tuple(c - NLAT // 2 for c in cell)
    results["cells"].append({"offset": off, "negs_c0": negs0,
                             "c_clock": c_clock})
    print(f"cell {off}: PSD at c=0: {negs0 == 0}, c_clock = {c_clock}")
    assert negs0 == 0 and c_clock is not None and -1.0 < c_clock < -0.3

results["fd_route_rel"] = worst_fd
print(f"route-2 (finite differences) agreement: rel {worst_fd:.1e}")
assert worst_fd < 1e-6

# ---- full-core sweep: threshold, multiplicity, clock-direction overlap ----
Me = M @ ETA_T
Pp, v4 = Me, 0.0
for pw in range(4):
    if pw:
        Pp = Pp @ Me
    v4 = v4 + (torch.einsum("...kk->...", Pp) - C_P[pw]) ** 2
v0 = 0.5 * v4.max()
core = (v4 / (v4 + v0) > 0.5).nonzero()
a0_full = a0_of(gen_catalog()["boost_x"], M)      # frozen boost tangent
print(f"full-core sweep: {core.shape[0]} cells (mask cw > 0.5)")
cs_grid = torch.linspace(0, -4, 801, dtype=DT, device=DEV)
sweep = {"c_clock": [], "overlap": [], "n_neg_past": []}
for cell in core:
    cell = tuple(int(x) for x in cell)
    Asp = torch.stack([A_all[ax][cell] for ax in range(3)])
    HG = hessian_auto(lambda v: dens_G(v, Asp,
                                       G_of(M[cell][None, None, None]
                                            )[0, 0, 0]))
    H4 = hessian_auto(lambda v: dens_I4(v, Asp))
    batch = HG[None] + cs_grid[:, None, None] * H4[None]
    negs = (torch.linalg.eigvalsh(batch) < -1e-10).sum(dim=1)
    idx = int((negs > 0).to(torch.int8).argmax())
    assert negs[idx] > 0 and idx > 0, "core cell not PSD at c = 0"
    c_cl = float(cs_grid[idx])
    c_star = c_cl - 0.05
    w, V = torch.linalg.eigh(HG + c_star * H4)
    n_neg = int((w < -1e-10).sum())
    vneg = V[:, 0]
    v_clock = torch.stack([a0_full[cell][a, b]
                           for (a, b) in IDX])
    ov = float((vneg @ v_clock).abs()
               / (vneg.norm() * v_clock.norm()))
    # Rayleigh threshold of the a0 channel itself: dens is a homogeneous
    # quadratic in the chart, so R_X = v_a0^T H_X v_a0 and with c < 0 the
    # channel R_G + c R_4 turns negative at c_a0 = -R_G/R_4, which lies
    # on the physical (negative) side iff R_4 > 0
    RG = float(v_clock @ HG @ v_clock)
    R4 = float(v_clock @ H4 @ v_clock)
    c_a0 = (-RG / R4) if R4 > 0 else None
    sweep["c_clock"].append(c_cl)
    sweep["overlap"].append(ov)
    sweep["n_neg_past"].append(n_neg)
    sweep.setdefault("c_a0", []).append(c_a0)
cc = np.array(sweep["c_clock"])
ovs = np.array(sweep["overlap"])
nn = np.array(sweep["n_neg_past"])
ca0_ok = [x for x in sweep["c_a0"] if x is not None]
ca = np.array(ca0_ok)
results["core_sweep"] = {
    "n_cells": int(core.shape[0]),
    "c_clock_min": float(cc.min()), "c_clock_max": float(cc.max()),
    "c_clock_median": float(np.median(cc)),
    "all_exactly_one_negative": bool((nn == 1).all()),
    "overlap_min": float(ovs.min()),
    "overlap_median": float(np.median(ovs)),
    "a0_channel_defined_cells": len(ca0_ok),
    "c_a0_min": float(ca.min()) if len(ca) else None,
    "c_a0_max": float(ca.max()) if len(ca) else None,
    "c_a0_median": float(np.median(ca)) if len(ca) else None,
    "c_clock_all": sweep["c_clock"], "overlap_all": sweep["overlap"],
    "c_a0_all": sweep["c_a0"]}
print(f"  c_clock over the core: [{cc.min():.3f}, {cc.max():.3f}], "
      f"median {np.median(cc):.3f}")
print(f"  negatives just past threshold: exactly one on all cells: "
      f"{bool((nn == 1).all())}")
print(f"  a0-channel (Rayleigh) threshold c_a0 defined on "
      f"{len(ca0_ok)}/{core.shape[0]} cells; range "
      f"[{ca.min():.3f}, {ca.max():.3f}], median {np.median(ca):.3f}")
print(f"  raw eigenvector |overlap| with a0: min {ovs.min():.3f}, "
      f"median {np.median(ovs):.3f}  (the lowest pointwise mode need "
      f"not align with a0; the loaded question is c_a0)")
assert (nn == 1).all()
assert len(ca0_ok) == core.shape[0], "a0 channel must open (R4 > 0)"

results["provenance"] = {"fields_dir": os.path.abspath(FIELDS)}
with open(os.path.join(HERE, "results", "kinetic_forms.json"), "w") as f:
    json.dump(results, f, indent=1)
open(os.path.join(HERE, "results", "kinetic_ran.flag"), "w").write("ran\n")
print("written: results/kinetic_forms.json")
