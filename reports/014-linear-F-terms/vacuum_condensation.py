"""Vacuum stability under a linear-in-F term (CPU).

Uniform vacuum M_vac on the 004 lattice, plus small spatial-block noise on the
free sites, relaxed under E_lambda = e_static(eta) + lambda H^3 sum dens for
lambda = 0 (control: must return to E = 0) and lambda of both signs at the 5%
setting of the electron scan. Pointwise theorem: along a vacuum-manifold twist
direction a, e(t) = lambda q(a) t^2 + p(a) t^4 with q indefinite, so for any
lambda != 0 the uniform vacuum is a saddle and the ground state carries a
gradient condensate (E < 0, extensive). Usage: python vacuum_condensation.py A|B
"""
import json, os, sys, time
import numpy as np
import torch
ROUTE = sys.argv[1]
sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, M_VAC, N, field, e_static, H, _L
scan = json.load(open(L.RESULTS))
out = {}
cls = list(L.CLASSES.items())[:2] if ROUTE == 'B' else list(L.CLASSES.items())[:1]
torch.manual_seed(5)
noise = torch.randn(N, N, N, 4, 4, dtype=DT, device=DEV); noise = 0.5 * (noise + noise.transpose(-1, -2))
noise[..., 0, :] = 0; noise[..., :, 0] = 0
Mvac = M_VAC.expand(N, N, N, 4, 4).clone()
import lattice_grid_defs as G
G._L['SHELL_VALS'] = None
# shell frozen at the VACUUM for this test (pin_shell BC with vacuum values)
G.field.__globals__['SHELL_VALS'] = Mvac.clone()
def run(lam, a, b, tag):
    M_raw = (Mvac + 1e-2 * noise * FREE[..., None, None].to(DT)).clone().requires_grad_(True)
    E0 = L.E_of(M_raw.detach(), lam, a, b).item()
    opt = torch.optim.Adam([M_raw], lr=L.LR)
    for it in range(1500):
        opt.zero_grad(); E = L.E_of(M_raw, lam, a, b); E.backward(); opt.step()
    opt2 = torch.optim.LBFGS([M_raw], max_iter=100, history_size=25, tolerance_grad=1e-9, tolerance_change=0, line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad(); E = L.E_of(M_raw, lam, a, b); E.backward(); return E
    opt2.step(closure)
    Mf = field(M_raw.detach())
    E = L.E_of(M_raw.detach(), lam, a, b).item()
    grad_amp = float(max(G.d1(Mf, ax, 'fwd').abs().max() for ax in range(3)))
    return {'E_start': E0, 'E_relaxed': E, 'E_stat_eta': e_static(Mf, 'eta').item(),
            'max_grad': grad_amp, 'offblock': G.offblock(Mf)}
for cname, (a, b) in cls:
    lam5 = 0.05 * scan['E_stat_base'] / abs(scan['base_integrals'][cname])
    for lam in (0.0, +lam5, -lam5):
        t0 = time.time()
        r = run(lam, a, b, cname)
        out[f'{cname}_lam{lam:+.3e}'] = r
        print(f"[{ROUTE}:{cname}] lam {lam:+.3e}: E start {r['E_start']:+.4e} -> relaxed {r['E_relaxed']:+.4e} "
              f"(eta part {r['E_stat_eta']:+.4e}) max|grad| {r['max_grad']:.3e} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(f'results/vacuum_condensation_{ROUTE}.json', 'w'), indent=1)
print('vacuum test complete', flush=True)
