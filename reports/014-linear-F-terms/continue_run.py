"""Review round 1, resolution 2: continue a persisted endpoint with more L-BFGS
until the gradient gate (|dE/dM|_inf,free <= 0.1) passes or 4 cycles of 100.
Usage: python continue_run.py A P1P3_f0.2_s-1"""
import json, os, sys
import numpy as np, torch
ROUTE, TAG = sys.argv[1], sys.argv[2]; sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, field
res = json.load(open(L.RESULTS)); r = res[TAG]; a, b = r['class']; lam = r['lambda']
M_raw = torch.tensor(np.load(os.path.join(L.FIELDS, f'{ROUTE}_{TAG}.npz'))['M'], dtype=DT, device=DEV).requires_grad_(True)
hist = [{'E': L.E_of(M_raw.detach(), lam, a, b).item(), 'g_inf': L.grad_inf_free(M_raw, lam, a, b)}]
print('start', hist[0], flush=True)
for cyc in range(4):
    opt = torch.optim.LBFGS([M_raw], max_iter=100, history_size=25, tolerance_grad=1e-9, tolerance_change=0, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); E = L.E_of(M_raw, lam, a, b); E.backward(); return E
    opt.step(closure)
    hist.append({'E': L.E_of(M_raw.detach(), lam, a, b).item(), 'g_inf': L.grad_inf_free(M_raw, lam, a, b),
                 'tail_eta': L.tail_fit(field(M_raw.detach()), 'eta')['slope']})
    print('cycle', cyc + 1, hist[-1], flush=True)
    if hist[-1]['g_inf'] <= 0.1:
        break
np.savez_compressed(os.path.join(L.FIELDS, f'{ROUTE}_{TAG}_continued.npz'), M=M_raw.detach().cpu().numpy())
json.dump({'tag': TAG, 'history': hist, 'passes_gate': hist[-1]['g_inf'] <= 0.1}, open(f'results/continuation_{ROUTE}_{TAG}.json', 'w'), indent=1)
print('continuation complete', flush=True)
