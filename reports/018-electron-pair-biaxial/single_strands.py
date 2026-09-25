"""Where the strands of one biaxial charge run: the transverse splitting e_2 - e_3 on spheres and on a slice.

Reads the relaxed field of single_biaxial.py (beta = 0.3, n = 32, box 12). On spheres of radius 1.5, 2, 3, 4,
4.5 about the charge: the positions (theta, phi) of the local minima of e_2 - e_3 below a quarter of beta,
the smallest gap e_1 - e_2 and the alignment |e_1 . r_hat| there. On the plane x = 0: e_2 - e_3 on a 121 x 121
grid (for the figure). Output: results/single_strands.json.
"""
import json
import math
import os

import torch
from soliton import Grid, to_matrix

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def eig(g, u, pts):
    n = -to_matrix(g.sample(u, pts))[..., 1:, 1:]
    return torch.linalg.eigh(n)


if __name__ == '__main__':
    d = torch.load(os.path.join(HERE, 'results', 'fields', 'single_biaxial_b0.3_n32.pt'), map_location='cpu', weights_only=False)
    beta = d['E'][2]
    g = Grid(d['n'], d['box'], E=d['E'], split=True, device='cpu')
    u = d['u'].double()
    out = {'beta': beta, 'spheres': {}}
    nth, nph = 90, 180
    th = (torch.arange(nth) + 0.5) * math.pi / nth
    ph = torch.arange(nph) * 2 * math.pi / nph
    T, P = torch.meshgrid(th, ph, indexing='ij')
    rh = torch.stack([T.sin() * P.cos(), T.sin() * P.sin(), T.cos()], -1)
    for r in (1.5, 2.0, 3.0, 4.0, 4.5):
        e, V = eig(g, u, r * rh)
        split = e[..., 1] - e[..., 0]
        # local minima on the sphere (8-neighbourhood, periodic in phi) below beta / 4
        pad = torch.cat([split[:, -1:], split, split[:, :1]], 1)
        pad = torch.cat([torch.full_like(pad[:1], 1e9), pad, torch.full_like(pad[:1], 1e9)], 0)
        nb = torch.stack([pad[1 + a:nth + 1 + a, 1 + b:nph + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)])
        is_min = (split <= nb.min(0).values) & (split < beta / 4)
        mins = [dict(theta=float(T[i, j]), phi=float(P[i, j]), split=float(split[i, j]),
                     rho_from_axis=float(r * torch.sin(T[i, j])),
                     align=float((V[i, j, :, 2] * rh[i, j]).sum().abs()))
                for i, j in is_min.nonzero().tolist()]
        out['spheres'][str(r)] = dict(minima=mins, n_minima=len(mins), split_median=float(split.median()),
                                      gap_min=float((e[..., 2] - e[..., 1]).min()))
        print(r, len(mins), [(round(m['theta'], 2), round(m['phi'], 2), round(m['split'], 3)) for m in mins], flush=True)
    s = torch.linspace(-d['box'] / 2 + 0.2, d['box'] / 2 - 0.2, 121)
    Y, Z = torch.meshgrid(s, s, indexing='ij')
    pts = torch.stack([torch.full_like(Y, 1e-3), Y, Z], -1)
    e, _ = eig(g, u, pts)
    out['slice_x0'] = dict(y=s.tolist(), z=s.tolist(), split=(e[..., 1] - e[..., 0]).tolist())
    json.dump(out, open(os.path.join(HERE, 'results', 'single_strands.json'), 'w'))
