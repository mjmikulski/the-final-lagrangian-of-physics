"""Post-hoc restart gate (spatial-block perturbation) for the 20%-weight runs.

The in-run restart perturbed all components including M_{0i}; the eta statics
is unbounded below outside the 3x3 spatial sector (equations_of_record §2), so
that perturbation tests the known off-block instability, not the linear term.
Here the perturbation is restricted to the spatial block on free sites
(sigma = 1e-2), re-relaxed with Adam 500 + L-BFGS(100) from the persisted
relaxed field of the same run, and compared. Usage: python restart_check.py A|B
"""
import json, os, sys
import numpy as np
import torch
sys.argv = [sys.argv[0]] + sys.argv[1:]
ROUTE = sys.argv[1]
import importlib
sys.argv = ['lattice_linear_v2.py', ROUTE]
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, field, e_static, load_or_make_base, H
res = json.load(open(L.RESULTS))
out = {}
for tag, r in list(res.items()):
    if not (isinstance(r, dict) and 'status' in r and 'f0.2' in tag and r['status'] == 'ok'):
        continue
    a, b = r['class']; lam = r['lambda']
    M0 = torch.tensor(np.load(os.path.join(L.FIELDS, f'{ROUTE}_{tag}.npz'))['M'], dtype=DT, device=DEV)
    torch.manual_seed(2)
    P = torch.randn_like(M0); P = 0.5 * (P + P.transpose(-1, -2))
    P[..., 0, :] = 0; P[..., :, 0] = 0                     # spatial block only
    Mp = M0 + 1e-2 * P * FREE[..., None, None].to(DT)
    M_raw = Mp.clone().requires_grad_(True)
    opt = torch.optim.Adam([M_raw], lr=L.LR)
    for it in range(500):
        opt.zero_grad(); E = L.E_of(M_raw, lam, a, b); E.backward(); opt.step()
    opt2 = torch.optim.LBFGS([M_raw], max_iter=100, history_size=25, tolerance_grad=1e-9, tolerance_change=0, line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad(); E = L.E_of(M_raw, lam, a, b); E.backward(); return E
    opt2.step(closure)
    E_re = L.E_of(M_raw.detach(), lam, a, b).item()
    E_kick = L.E_of(Mp, lam, a, b).item()
    tail = L.tail_fit(field(M_raw.detach()), 'eta')['slope']
    out[tag] = {'E_main': r['E_total'], 'E_kicked_start': E_kick, 'E_restart': E_re,
                'dE_vs_main': E_re - r['E_total'], 'tail_main': r['tail_eta'], 'tail_restart': tail,
                'offblock_restart': L.offblock(field(M_raw.detach()))}
    print(f"[{ROUTE}:{tag}] main {r['E_total']:.6f} kicked {E_kick:.4f} restart {E_re:.6f} dE {E_re - r['E_total']:+.2e} tail {r['tail_eta']:.3f}->{tail:.3f}", flush=True)
    json.dump(out, open(f'results/restart_check_{ROUTE}.json', 'w'), indent=1)
print('restart check complete', flush=True)
