"""Review round 2, finding 2: the box-size scaling of the channel
inertia. Three boxes at fixed spacing H = 1.5 (N = 16, 24, 32; the
smaller boxes are central crops of the 004 seed, same texture pinned
at a smaller radius), short static relax per box (Adam 1000 + one
L-BFGS cycle -- the inertia is a profile property), then the inertias
of the pure internal tangent and the combined space-internal tangent,
with log-log fitted exponents. Volume extensivity means I ~ L^3.
Out: results/inertia_scaling.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R004 = os.path.join(HERE, "..", "004-lattice-clock")
FIELDS = os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results"))
FLAG = os.path.join(HERE, "results", "scaling_ran.flag")
if not os.path.exists(os.path.join(FIELDS, "M_G_polished.npz")):
    if os.path.exists(FLAG):
        os.remove(FLAG)
    print("inertia_scaling: NOT REPRODUCED HERE -- needs report 004's "
          f"stack in {FIELDS} (or M5_FIELDS_DIR).")
    sys.exit(0)

SRC = open(os.path.join(R004, "lattice.py")).read()


def load_stack(N):
    src = SRC.replace("N, L = 32, 48.0", f"N, L = {N}, {N * 1.5}")
    ns = {"__name__": "not_main",
          "__file__": os.path.join(R004, "lattice.py")}
    exec(compile(src, "lattice_patched", "exec"), ns)
    return ns


FULL = load_stack(32)["seed_embedded"]()


def seed_for(L, N):
    M4 = FULL
    if N != 32:
        c0 = (32 - N) // 2
        M4 = M4[c0:c0 + N, c0:c0 + N, c0:c0 + N].clone()
    L["seed_embedded"] = lambda: M4
    return M4


def short_relax(L, seed):
    field, e_static = L["field"], L["e_static"]
    M_raw = seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=1e-3)
    for it in range(1000):
        opt.zero_grad()
        e_static(field(M_raw), "G").backward()
        opt.step()
    opt2 = torch.optim.LBFGS([M_raw], max_iter=200, history_size=25,
                             tolerance_grad=1e-9, tolerance_change=0,
                             line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        E = e_static(field(M_raw), "G")
        E.backward()
        return E
    opt2.step(closure)
    return field(M_raw.detach())


def tangents(L, Mf):
    d1 = L["d1"]
    DT = L["DT"]
    W = L["gen_catalog"]()["rot_xy"]
    interior = (1.0 - L["SHELL"].to(DT))[..., None, None]
    X, Y, _ = L["coords"]()
    internal = (torch.einsum("ab,...bc->...ac", W, Mf)
                - torch.einsum("...ab,bc->...ac", Mf, W))
    dMy, dMx = d1(Mf, 1, "fwd"), d1(Mf, 0, "fwd")
    orbital = -(X[..., None, None] * dMy - Y[..., None, None] * dMx)
    return interior * internal, interior * (orbital + internal)


def inertia_of(L, Mf, tan):
    d1, comm, G_of = L["d1"], L["comm"], L["G_of"]
    DT, DEV, H = L["DT"], L["DEV"], L["H"]
    G = G_of(Mf)
    k = torch.zeros(Mf.shape[:3], dtype=DT, device=DEV)
    for st in ("fwd", "bwd"):
        A = [d1(Mf, ax, st) for ax in range(3)]
        for i in range(3):
            F0 = comm(tan, A[i])
            k = k + 0.5 * 4.0 * torch.einsum(
                "...ab,...ac,...bd,...cd->...", F0, G, G, F0)
    I = float(2.0 * H ** 3 * k.sum())
    pr = float((k.sum() ** 2) / (k ** 2).sum().clamp_min(1e-30))
    return I, pr


rows = []
for N in (16, 24, 32):
    L = load_stack(N)
    Mf = short_relax(L, seed_for(L, N))
    g_pure, g_comb = tangents(L, Mf)
    Ip, PRp = inertia_of(L, Mf, g_pure)
    Ic, PRc = inertia_of(L, Mf, g_comb)
    rows.append({"N": N, "Lbox": N * 1.5, "I_pure": Ip, "PR_pure": PRp,
                 "I_comb": Ic, "PR_comb": PRc})
    print(f"N {N} (L {N*1.5}): I_pure {Ip:.4e} (PR {PRp:.0f}), "
          f"I_comb {Ic:.4e} (PR {PRc:.0f})", flush=True)

out = {"H": 1.5, "rows": rows}
for q in ("I_pure", "I_comb"):
    p = float(np.polyfit([np.log(x["Lbox"]) for x in rows],
                         [np.log(x[q]) for x in rows], 1)[0])
    out[f"exponent_{q}"] = p
    print(f"{q} ~ L^{p:.2f}")
json.dump(out, open(os.path.join(HERE, "results",
                                 "inertia_scaling.json"), "w"),
          indent=1)
with open(FLAG, "w") as f:
    f.write("inertia scaling computed in this run\n")
print("written: results/inertia_scaling.json")
