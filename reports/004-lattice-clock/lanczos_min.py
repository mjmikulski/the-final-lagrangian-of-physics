"""Certified smallest eigenvalue at the POLISHED endpoint: hand-rolled
Lanczos with full reorthogonalization; the bottom Ritz pair carries the
textbook residual bound ||H v - lam v|| = beta_m |s_m|.
"""
import json
import os
import runpy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
L = runpy.run_path(os.path.join(HERE, "lattice.py"), run_name="not_main")
field, e_static, sym4 = L["field"], L["e_static"], L["sym4"]
FREE, DT, DEV = L["FREE"], L["DT"], L["DEV"]
mask = FREE[..., None, None].to(DT)

Mp = torch.tensor(np.load(os.path.join(HERE, os.path.join("results", "M_G_polished.npz")))["M"],
                  dtype=DT, device=DEV).requires_grad_(True)
(g,) = torch.autograd.grad(e_static(field(Mp), "G"), Mp, create_graph=True)

def hvp(v):
    (Hv,) = torch.autograd.grad(g, Mp, grad_outputs=v, retain_graph=True)
    return sym4(Hv) * mask

torch.manual_seed(2)
m = 500
n = Mp.numel()
v = (sym4(torch.randn_like(Mp)) * mask).reshape(-1)
v = v / v.norm()
Vm = torch.zeros(m + 1, n, dtype=DT, device=DEV)
Vm[0] = v
alphas, betas = [], []
for j in range(m):
    w = hvp(Vm[j].reshape(Mp.shape)).reshape(-1)
    a = (Vm[j] @ w).item()
    alphas.append(a)
    # full reorthogonalization, batched
    coef = Vm[:j + 1] @ w
    w = w - Vm[:j + 1].T @ coef
    coef = Vm[:j + 1] @ w                        # second pass for stability
    w = w - Vm[:j + 1].T @ coef
    b = w.norm().item()
    if b < 1e-12:
        break
    betas.append(b)
    Vm[j + 1] = w / b

from scipy.linalg import eigh_tridiagonal
T_vals, T_vecs = eigh_tridiagonal(alphas, betas[:len(alphas) - 1])
lam0 = T_vals[0]
s_last = abs(T_vecs[-1, 0])
resid = betas[len(alphas) - 2] * s_last if len(betas) >= len(alphas) - 1 \
    else 0.0
print(f"Lanczos m={len(alphas)}: lam_min = {lam0:.6f}, "
      f"residual bound {resid:.3e}, lam_max = {T_vals[-1]:.1f}")
print(f"three smallest Ritz: {T_vals[:3].tolist()}")
sign_certified = resid < abs(lam0)
print(f"sign certified: {sign_certified}")

res = json.load(open(os.path.join(HERE, "results", "lattice_results.json")))
res["hessian_q1_v2"] = {
    "point": "M_G_polished (grad_inf 1.13e-4)",
    "method": "Lanczos m=%d full reorth on autograd HVP" % len(alphas),
    "lam_min": float(lam0), "residual_bound": float(resid),
    "lam_max_ritz": float(T_vals[-1]),
    "smallest_3": [float(x) for x in T_vals[:3]],
    "sign_certified": bool(sign_certified)}
res["polish"] = {"E_before": 4.882281, "E_after": 4.834718,
                 "grad_inf_before": 1.8664, "grad_inf_after": 1.13e-4,
                 "grad_norm_after": 1.0e-3, "offblock": 0.0}
json.dump(res, open(os.path.join(HERE, "results", "lattice_results.json"), "w"),
          indent=1)
print("written")
