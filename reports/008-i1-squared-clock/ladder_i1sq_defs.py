"""Shared definitions for the auxiliary checks (gamma scaling, deep
well, runaway): the same densities, fixed-depth relaxation protocol
(Adam + L-BFGS cycles; no gradient tolerance is reached on this
landscape -- see README section 6) and G-form functionals as
ladder_i1sq.py, parameterized by gamma. Loaded via runpy by the check
scripts so the physics is defined once.
"""
import json
import os
import runpy

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_R004 = os.path.join(_HERE, "..", "004-lattice-clock")
_FIELDS = os.environ.get("M5_FIELDS_DIR",
                         os.path.join(_R004, "results"))
_L = runpy.run_path(os.path.join(_R004, "lattice.py"),
                    run_name="not_main")
field, e_static = _L["field"], _L["e_static"]
a0_of, gen_catalog = _L["a0_of"], _L["gen_catalog"]
d1, comm, G_of = _L["d1"], _L["comm"], _L["G_of"]
H, DT, DEV, ETA = _L["H"], _L["DT"], _L["DEV"], _L["ETA"]

M_pol = torch.tensor(
    np.load(os.path.join(_FIELDS, "M_G_polished.npz"))["M"],
    dtype=DT, device=DEV)


def densities(M, a0, om, metric):
    G = G_of(M) if metric == "G" else None
    V = om * a0
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    k = torch.zeros(M.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(V, A[i])
            if metric == "G":
                k = k + 0.5 * 4.0 * torch.einsum(
                    "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
            else:
                k = k - 0.5 * 4.0 * torch.einsum(
                    "...ab,ac,bd,...cd->...", F0, ETA, ETA, F0)
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                if metric == "G":
                    i1s = i1s + 0.5 * 4.0 * torch.einsum(
                        "...ab,...ac,...bd,...cd->...", F, G, G, F)
                else:
                    i1s = i1s + 0.5 * 4.0 * torch.einsum(
                        "...ab,ac,bd,...cd->...", F, ETA, ETA, F)
    return i1s, k


_Mg = field(M_pol)
A0 = a0_of(gen_catalog()["boost_x"], _Mg)


def relax(e_total_fn):
    """Same fixed-depth protocol as ladder_i1sq.py: Adam 500 + two
    L-BFGS cycles; returns (field, E_levels, ginf)."""
    M_raw = M_pol.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        e_total_fn(M_raw).backward()
        opt.step()
    E_levels = [float(e_total_fn(M_raw).detach())]
    for cycle in range(2):
        opt2 = torch.optim.LBFGS([M_raw], max_iter=200,
                                 history_size=25, tolerance_grad=1e-9,
                                 tolerance_change=0,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            E = e_total_fn(M_raw)
            E.backward()
            return E
        opt2.step(closure)
        E_levels.append(float(e_total_fn(M_raw).detach()))
    g = torch.autograd.grad(e_total_fn(M_raw), M_raw)[0]
    return M_raw.detach(), E_levels, float(g.abs().max())


def load_base(here):
    return json.load(open(os.path.join(here, "results",
                                       "i1sq_ladders.json")))


def run_rungs(tag, gamma, reading, omegas):
    """G-form rungs at the given gamma; reading in {'energy',
    'fundamental'}; returns rows with energies, PR and residuals."""
    def e_extra(Mf, om):
        i1s, k = densities(Mf, A0, om, "G")
        if reading == "energy":
            dens = (i1s - k) ** 2
        else:
            dens = i1s ** 2 - 2.0 * i1s * k + 3.0 * k ** 2
        return gamma * H ** 3 * dens.sum()

    rows = []
    for om in omegas:
        M_raw, E_levels, ginf = relax(
            lambda Mr, om=om: e_static(field(Mr), "G")
            + e_extra(field(Mr), om))
        Mf = field(M_raw)
        E = (e_static(Mf, "G") + e_extra(Mf, om)).item()
        _, kd = densities(Mf, A0, max(om, 1e-9), "G")
        pr = ((kd.sum() ** 2) / (kd ** 2).sum().clamp_min(1e-30)).item()
        rows.append({"omega": om, "E_total": E, "PR_k_sites": pr,
                     "grad_inf": ginf, "E_levels": E_levels})
        print(f"  [{tag}] omega {om}: E {E:.6f}, PR {pr:.0f}, "
              f"|g|inf {ginf:.1e}", flush=True)
    return rows
