"""Direct lattice check of the pointwise condensation argument: energy along an
explicit frame-twist wave M(x) = R(theta(x)) M_vac R^T, theta_k(x) = t * c[i,k] x_i
(smoothly windowed to vanish on the pinned shell), for the most negative
direction c of the linear class's rotational form, both signs of lambda.
Usage: python twist_scan.py B|A  (CPU is fine: energy evaluations only)"""
import json, sys
import numpy as np, torch
ROUTE = sys.argv[1]
sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, M_VAC, N, H, field, e_static
import lattice_grid_defs as G
sys.path.insert(0, '.')
from linear_defs import eval_diagram_np, metrics_np
from u_family_defs import F_tensor_np
G.field.__globals__['SHELL_VALS'] = M_VAC.expand(N, N, N, 4, 4).clone()
Mv = np.diag([-8.0, 1.0, 0.3, 0.0]); mets = metrics_np(np.eye(4))
J = np.zeros((3, 4, 4))
for k, (i, j) in enumerate(((2, 3), (3, 1), (1, 2))):
    J[k, i, j], J[k, j, i] = -1.0, 1.0
def A_of(c):
    c = c.reshape(3, 3); A = [np.zeros((4, 4))]
    for i in range(3):
        w = np.einsum('k,kab->ab', c[i], J); A.append(w @ Mv + Mv @ w.T)
    return A
scan = json.load(open(L.RESULTS))
cname, (a, b) = list(L.CLASSES.items())[0]
ds = [('even', '02-13', 2, 3)] if ROUTE == 'A' else [('even', '02-13', 2, 3), ('even', '02-13', 2, 4)]   # P1P2 | F1Q = F12 + F13
Q = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        e = np.zeros(9); e[i] += 1; e[j] += 1; f = np.zeros(9); f[i] += 1; f[j] -= 1
        Q[i, j] = sum((eval_diagram_np(d, F_tensor_np(A_of(e)), mets) - eval_diagram_np(d, F_tensor_np(A_of(f)), mets)) / 4 for d in ds)
w, V = np.linalg.eigh(0.5 * (Q + Q.T)); c = V[:, 0].reshape(3, 3); print('most negative q direction, q =', w[0])
# lattice twist wave: theta_k(x) = t * sum_i c[i,k] x_i * window(x)
x = (torch.arange(N, dtype=DT) - (N - 1) / 2) * H
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij'); r = torch.sqrt(X**2 + Y**2 + Z**2)
win = torch.exp(-(r / 14.0) ** 6)                                    # vanishes well inside the shell
coords = torch.stack([X, Y, Z], -1)
Jt = torch.tensor(J, dtype=DT); ct = torch.tensor(c, dtype=DT)
lam5 = 0.05 * scan['E_stat_base'] / abs(scan['base_integrals'][cname])
def M_twist(t):
    theta = t * torch.einsum('...i,ik->...k', coords, ct) * win[..., None]   # (N,N,N,3)
    W = torch.einsum('...k,kab->...ab', theta, Jt)
    R = torch.matrix_exp(W)
    return R @ M_VAC.cpu() @ R.transpose(-1, -2)
out = {}
for t in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08):
    M = M_twist(t).to(DEV)
    Es = e_static(field(M), 'eta').item()
    lin = float(H ** 3 * L.dens_class(field(M), a, b).sum())
    out[t] = {'E_stat': Es, 'lin_integral': lin, 'E_plus': Es + lam5 * lin, 'E_minus': Es - lam5 * lin}
    print(f"t {t:6.3f}: E_stat {Es:+.4e}  lin {lin:+.4e}  E(+lam) {Es + lam5 * lin:+.4e}  E(-lam) {Es - lam5 * lin:+.4e}", flush=True)
json.dump(out, open(f'results/twist_scan_{ROUTE}.json', 'w'), indent=1)
