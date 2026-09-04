"""Where does the linear integral live on the base electron profile? Shell
sums of the linear class densities versus radius, and the fraction from the
outer zone (r > 20, next to the pinned shell). Usage: python radial_profile.py A|B"""
import json, sys
import numpy as np, torch
ROUTE = sys.argv[1]; sys.argv = ['lattice_linear_v2.py', ROUTE]
import importlib
L = importlib.import_module('lattice_linear_v2')
from lattice_grid_defs import DEV, DT, N, H, ETA, field, load_or_make_base, d1, F4_of
import lattice_grid_defs as G
Mg = field(load_or_make_base())
x = (torch.arange(N, dtype=DT, device=DEV) - (N - 1) / 2) * H
X, Y, Z = torch.meshgrid(x, x, x, indexing='ij'); r = torch.sqrt(X**2 + Y**2 + Z**2)
bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 24), (24, 30)]
out = {}
dens = {c: L.dens_class(Mg, a, b) for c, (a, b) in L.CLASSES.items()}
dens['phi(null)'] = L.dens_class(Mg, 0, 0, [ETA.expand_as(Mg)] * 4)
# eta static density for scale
e_u = 0.0
for st in ('fwd', 'bwd'):
    A = [d1(Mg, ax, st) for ax in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            F = G.comm(A[i], A[j]); e_u = e_u + 0.5 * 4.0 * G._L['inner_X'](F, ETA)
dens['eta_static'] = e_u
print(f"{'class':12s}" + ''.join(f"{f'[{lo},{hi})':>12s}" for lo, hi in bins) + f"{'total':>12s}{'frac r>20':>11s}")
for c, dd in dens.items():
    row = []
    for lo, hi in bins:
        m = (r >= lo) & (r < hi); row.append(float(H ** 3 * dd[m].sum()))
    tot = float(H ** 3 * dd.sum()); outer = float(H ** 3 * dd[r >= 20].sum())
    out[c] = {'shells': row, 'total': tot, 'frac_outer': outer / tot if tot else None}
    print(f"{c:12s}" + ''.join(f"{v:12.3e}" for v in row) + f"{tot:12.3e}{(outer / tot if tot else float('nan')):11.3f}")
json.dump({'bins': bins, 'classes': out}, open(f'results/radial_profile_{ROUTE}.json', 'w'), indent=1)
