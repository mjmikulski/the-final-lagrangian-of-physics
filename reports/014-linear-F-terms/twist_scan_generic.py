"""Review round 2, resolution 1: generic compact vacuum-orbit perturbations.

M_t(x) = R_t(x) M_vac R_t(x)^T with R_t = expm(t W(x)), W(x) three random
spatial-rotation fields (seeded), smoothed by nearest-neighbour averaging,
zero on the pinned shell -- no inversion symmetry. Both signs of t. Reports
the odd (cubic) and even (quartic) parts of the linear integral l(t) and of
the eta static energy: c3 = [l(t) - l(-t)] / (2 t^3), c4 = [E(t) + E(-t)] / (2 t^4).
Usage: python twist_scan_generic.py A|B"""
import json, sys
import numpy as np, torch
ROUTE = sys.argv[1]; sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, M_VAC, N, H, field, e_static
import lattice_grid_defs as G
G.field.__globals__['SHELL_VALS'] = M_VAC.expand(N, N, N, 4, 4).clone()
J = torch.zeros(3, 4, 4, dtype=DT, device=DEV)
for k, (i, j) in enumerate(((2, 3), (3, 1), (1, 2))):
    J[k, i, j], J[k, j, i] = -1.0, 1.0
scan = json.load(open(L.RESULTS))
out = {}
for seed in (456, 7, 99):
    g = torch.Generator(device='cpu').manual_seed(seed)
    W = torch.randn(N, N, N, 3, generator=g, dtype=DT).to(DEV)
    for _ in range(8):                                     # nearest-neighbour smoothing
        W = (W + torch.roll(W, 1, 0) + torch.roll(W, -1, 0) + torch.roll(W, 1, 1) + torch.roll(W, -1, 1)
             + torch.roll(W, 1, 2) + torch.roll(W, -1, 2)) / 7
    W = W * FREE[..., None].to(DT)                         # zero on the pinned shell
    W = W / W.abs().max()
    def M_of(t):
        R = torch.matrix_exp(t * torch.einsum('...k,kab->...ab', W, J))
        return R @ M_VAC @ R.transpose(-1, -2)
    for cname, (a, b) in L.CLASSES.items():
        if abs(scan['base_integrals'][cname]) < 1e-10:
            continue
        rec = {}
        for t in (5e-4, 1e-3, 2e-3, 4e-3):
            vals = {}
            for sg in (+1, -1):
                Mt = field(M_of(sg * t))
                vals[sg] = (e_static(Mt, 'eta').item(), float(H ** 3 * L.dens_class(Mt, a, b).sum()))
            c3 = (vals[1][1] - vals[-1][1]) / (2 * t ** 3)
            l4 = (vals[1][1] + vals[-1][1]) / (2 * t ** 4)
            c4 = (vals[1][0] + vals[-1][0]) / (2 * t ** 4)
            e3 = (vals[1][0] - vals[-1][0]) / (2 * t ** 3)
            rec[t] = {'lin_plus': vals[1][1], 'lin_minus': vals[-1][1], 'E_plus': vals[1][0], 'E_minus': vals[-1][0],
                      'c3_lin': c3, 'even_lin_over_t4': l4, 'c4_eta': c4, 'odd_eta_over_t3': e3}
            print(f"[{ROUTE}:{cname} seed {seed}] t {t:.0e}: lin odd/t^3 {c3:+.6f}  lin even/t^4 {l4:+.4f}  E_eta even/t^4 {c4:.4f}  E_eta odd/t^3 {e3:+.2e}", flush=True)
        lam5 = 0.05 * scan['E_stat_base'] / abs(scan['base_integrals'][cname])
        c3 = rec[1e-3]['c3_lin']; c4 = rec[1e-3]['c4_eta']
        t_star = -3 * lam5 * c3 / (4 * c4)
        E_star = c4 * t_star ** 4 + lam5 * c3 * t_star ** 3
        rec['lambda5'] = lam5; rec['t_star_quartic_model'] = t_star; rec['E_star_quartic_model'] = E_star
        print(f"   -> lambda(5%) {lam5:.3e}: cubic-quartic model minimum t* {t_star:+.2e}, E* {E_star:+.2e}", flush=True)
        out[f'{cname}_seed{seed}'] = {str(k): v for k, v in rec.items()}
json.dump(out, open(f'results/twist_generic_{ROUTE}.json', 'w'), indent=1)
print('generic twist scan complete', flush=True)
