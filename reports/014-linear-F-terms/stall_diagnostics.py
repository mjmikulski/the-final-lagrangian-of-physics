"""Review round 1, resolution 1: what stalls route B?

For every persisted route-B endpoint: all three spectral gaps on the free
sites (min over sites), the site of maximal |dE/dM| and the gaps THERE, and a
directional finite-difference check of the full derivative at the endpoint
along the gradient direction (eps = 1e-5, 1e-6). Also for the F1Q/FQQ/F1T
classes the min 1-2 gap, since the only singular denominators in metrics_B
are 1/(l2 - l1) and 1/(l3 - l1) (Q and T are smooth through the 2-3
crossing, Riesz). Usage: python stall_diagnostics.py B  (reads the committed float32 endpoint fields in results/fields/)
"""
import json, os, sys
import numpy as np, torch
ROUTE = sys.argv[1]; sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, FREE, N, H, ETA, field, e_static
res = json.load(open(L.RESULTS))
out = {}
for tag, r in res.items():
    if not (isinstance(r, dict) and 'status' in r):
        continue
    a, b = r['class']; lam = r['lambda']
    fn = os.path.join(L.FIELDS, f'{ROUTE}_{tag}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f'endpoint field missing: {fn} (committed float32 copies live in results/fields/)')
    Mr = torch.tensor(np.load(fn)['M'], dtype=DT, device=DEV)
    Mf = field(Mr)
    x = torch.einsum('ab,...bc->...ac', ETA, Mf).cpu()
    ls = torch.sort(torch.linalg.eigvals(x).real, dim=-1, descending=True).values   # t, 1, 2, 3
    g_t1 = (ls[..., 0] - ls[..., 1]); g_12 = (ls[..., 1] - ls[..., 2]); g_23 = (ls[..., 2] - ls[..., 3])
    Fm = FREE.cpu()
    Mv = Mr.clone().requires_grad_(True)
    g = torch.autograd.grad(L.E_of(Mv, lam, a, b), Mv)[0]
    g = 0.5 * (g + g.transpose(-1, -2))
    gabs = g.abs().amax(dim=(-1, -2)).cpu() * Fm.to(DT)
    site = np.unravel_index(int(gabs.argmax()), gabs.shape)
    V = g * FREE[..., None, None].to(DT); V = V / V.norm()
    auto = float((g * V).sum())
    fd = {}
    for eps in (1e-5, 1e-6):
        Ep = L.E_of(Mr + eps * V, lam, a, b).item(); Em = L.E_of(Mr - eps * V, lam, a, b).item()
        fd[str(eps)] = (Ep - Em) / (2 * eps)
    # same-chamber step (review round 4): h = min 1-2 gap / 100; ||V||_F = 1 so
    # each site moves by at most h and Weyl's bound keeps g12 >= 0.98 g12 along the secant
    h = float(g_12[Fm].min()) / 100.0
    Ep = L.E_of(Mr + h * V, lam, a, b).item(); Em = L.E_of(Mr - h * V, lam, a, b).item()
    fd['same_chamber'] = (Ep - Em) / (2 * h); fd['same_chamber_h'] = h
    entry = {'grad_inf_free': float(gabs.max()), 'argmax_site': [int(s) for s in site],
             'gaps_at_argmax': {'t-1': float(g_t1[site]), '1-2': float(g_12[site]), '2-3': float(g_23[site])},
             'gaps_min_free': {'t-1': float(g_t1[Fm].min()), '1-2': float(g_12[Fm].min()), '2-3': float(g_23[Fm].min())},
             'site_of_min_12_gap': [int(s) for s in np.unravel_index(int((g_12 + (~Fm) * 1e9).argmin()), g_12.shape)],
             'directional': {'autograd': auto, 'fd': fd, 'rel_err': {k: abs(auto - v) / max(abs(v), 1e-30) for k, v in fd.items() if k != 'same_chamber_h'}},
             'status_main': r['status'], 'grad_inf_recorded': r.get('grad_inf_free')}
    out[tag] = entry
    print(f"[{tag:16s}] |g|inf {entry['grad_inf_free']:.2e} at {site}: gaps t-1 {entry['gaps_at_argmax']['t-1']:.3f} "
          f"1-2 {entry['gaps_at_argmax']['1-2']:.2e} 2-3 {entry['gaps_at_argmax']['2-3']:.2e} | min free: 1-2 {entry['gaps_min_free']['1-2']:.2e} "
          f"2-3 {entry['gaps_min_free']['2-3']:.2e} | FD rel err {entry['directional']['rel_err']['1e-05']:.1e} / {entry['directional']['rel_err']['1e-06']:.1e} | same-chamber h {h:.1e}: {entry['directional']['rel_err']['same_chamber']:.1e}", flush=True)
json.dump(out, open(f'results/stall_diagnostics_{ROUTE}.json', 'w'), indent=1)
print('stall diagnostics complete', flush=True)
