"""What the threshold kappa_c means on the electron: the weighted term on the relaxed frozen hedgehog.

On the committed static_base_n32 (E0 = 100 conventions, box 12, n = 32, quadrature points of the
trilinear interpolant) compute the static energy E_stat = int (F.F + V), the signed weighted integral
X1 = int sqrt(e3/e3vac) phi, its absolute version int sqrt(e3/e3vac) |phi|, the plain int phi (a boundary
term: should be small relative to int |phi|) and the shell profile of phi. Then kappa X1 / E_stat is the
relative weight of the term at coupling kappa. Writes results/k0_lattice_phi.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e0_route'))
from lagrangian import operator, terms, ETA
from soliton import Grid

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
S = torch.tensor([1.0, -1.0, -1.0, -1.0])

d = torch.load(os.path.join(HERE, '..', '016-time-axis-screening', 'results', 'fields', 'static_base_n32.pt'), map_location='cpu', weights_only=False)
E = tuple(float(x) for x in d['E'])
grid = Grid(d['n'], d['box'], E=E, device='cpu')
u = d['u'].double()
M, dM = grid.derivatives(u)                      # [4 quad, n+1, n+1, n+1, 4, 4] and [..., mu, 4, 4]
N = operator(M)
dN = torch.einsum('ab,...mbc->...mac', ETA, dM)[..., 1:, :, :]        # spatial derivatives of the mixed N
F = torch.einsum('...iab,...jbc->...ijac', dN, dN) - torch.einsum('...jab,...ibc->...ijac', dN, dN)
Fcov = torch.einsum('a,...ijab->...ijab', S, F)
phi = torch.einsum('...ijij->...', Fcov[..., :, :, 1:, 1:])
t1 = torch.diagonal(N, dim1=-2, dim2=-1).sum(-1)
t2 = torch.diagonal(N @ N, dim1=-2, dim2=-1).sum(-1)
t3 = torch.diagonal(N @ N @ N, dim1=-2, dim2=-1).sum(-1)
e3 = (t1 ** 3 - 3 * t1 * t2 + 2 * t3) / 6
e3vac = E[0] * E[1] * E[2] + E[0] * E[1] * E[3] + E[0] * E[2] * E[3] + E[1] * E[2] * E[3]
w = torch.sqrt(e3.clamp_min(0) / e3vac)
kin, pot, _ = terms(M, dM, E, frozen=True)
vol = grid.vol
E_stat = float((-kin + pot).sum() * vol)
r = grid.xq.norm(dim=-1)
shells = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]
out = {'E': E, 'n': d['n'], 'box': d['box'], 'E_stat': E_stat,
       'int_phi': float(phi.sum() * vol), 'int_abs_phi': float(phi.abs().sum() * vol),
       'X1_signed': float((w * phi).sum() * vol), 'int_w_abs_phi': float((w * phi.abs()).sum() * vol),
       'shell_phi': [float(phi[(r >= a) & (r < b)].mean()) for a, b in shells],
       'shell_w': [float(w[(r >= a) & (r < b)].mean()) for a, b in shells],
       'fraction_of_X1_from_r_lt_3': float((w * phi)[r < 3].sum() / (w * phi).sum())}
for kappa in (0.01, 0.02, 0.05):
    out[f'weight_at_kappa_{kappa}'] = kappa * out['X1_signed'] / E_stat
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'results', 'appendix_volume_lattice_phi.json'), 'w'), indent=1)
