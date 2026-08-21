"""PR #3 review response (P1a+P1b): gradient-gated polish of the G-statics
endpoint, then a residual-certified smallest-eigenvalue solve (scipy eigsh
on the autograd HVP). Records achieved gradient residual and Ritz residual;
claims in the report are conditioned on both.
"""
import json
import os

import numpy as np
import runpy
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
L = runpy.run_path(os.path.join(HERE, "lattice.py"), run_name="not_main")
field, e_static, sym4 = L["field"], L["e_static"], L["sym4"]
FREE, DT, DEV, H = L["FREE"], L["DT"], L["DEV"], L["H"]
mask = FREE[..., None, None].to(DT)

Mr = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G.npz")))["M"],
                  dtype=DT, device=DEV)

def grad_inf(m):
    m = m.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(e_static(field(m), "G"), m)
    g = sym4(g) * mask
    return g.abs().max().item(), g.norm().item()

gi0, gn0 = grad_inf(Mr)
E0 = e_static(field(Mr), "G").item()
print(f"before polish: E {E0:.6f}, |g|_inf {gi0:.4f}, ||g|| {gn0:.4f}",
      flush=True)

# stage 1: Adam annealed; stage 2: LBFGS polish
M = Mr.clone().requires_grad_(True)
for lr, steps in ((1e-3, 3000), (2e-4, 3000), (5e-5, 2000)):
    opt = torch.optim.Adam([M], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        e_static(field(M), "G").backward()
        M.grad.mul_(mask)
        opt.step()
    gi, _ = grad_inf(M.detach())
    print(f"adam lr {lr}: |g|_inf {gi:.5f}", flush=True)

opt = torch.optim.LBFGS([M], max_iter=40, history_size=25,
                        tolerance_grad=1e-9, tolerance_change=1e-14)
for outer in range(12):
    def closure():
        opt.zero_grad()
        E = e_static(field(M), "G")
        E.backward()
        M.grad.mul_(mask)
        return E
    opt.step(closure)
    gi, _ = grad_inf(M.detach())
    print(f"lbfgs outer {outer}: |g|_inf {gi:.6f}", flush=True)
    if gi < 1e-4:
        break

Mp = M.detach()
gi1, gn1 = grad_inf(Mp)
E1 = e_static(field(Mp), "G").item()
offb = field(Mp)[..., 0, 1:].abs().max().item()
np.savez_compressed(os.path.join(HERE, os.path.join("results", "M_G_polished.npz")),
                    M=Mp.cpu().numpy())
print(f"after polish: E {E1:.6f}, |g|_inf {gi1:.6f}, ||g|| {gn1:.4f}, "
      f"offblock {offb:.2e}", flush=True)

# certified smallest eigenvalue: scipy eigsh over the autograd HVP
Mh = Mp.clone().requires_grad_(True)
(g,) = torch.autograd.grad(e_static(field(Mh), "G"), Mh, create_graph=True)
n = Mh.numel()

def matvec(x):
    v = sym4(torch.tensor(x.reshape(Mh.shape), dtype=DT, device=DEV)) * mask
    (Hv,) = torch.autograd.grad(g, Mh, grad_outputs=v, retain_graph=True)
    return (sym4(Hv) * mask).cpu().numpy().ravel()

op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
vals, vecs = eigsh(op, k=3, which="SA", tol=1e-8, maxiter=3000)
v0 = vecs[:, 0]
res_ritz = np.linalg.norm(matvec(v0) - vals[0] * v0)
print(f"eigsh SA: lam = {vals.tolist()}, ritz residual {res_ritz:.3e}",
      flush=True)

res = json.load(open(os.path.join(HERE, "results", "lattice_results.json")))
res["polish"] = {"E_before": E0, "E_after": E1, "grad_inf_before": gi0,
                 "grad_inf_after": gi1, "grad_norm_after": gn1,
                 "offblock": offb}
res["hessian_q1_v2"] = {"lam_smallest_3": vals.tolist(),
                        "ritz_residual": float(res_ritz),
                        "grad_inf_at_point": gi1,
                        "method": "scipy eigsh SA on autograd HVP"}
json.dump(res, open(os.path.join(HERE, "results", "lattice_results.json"), "w"),
          indent=1)
print("written: lattice_results.json", flush=True)
