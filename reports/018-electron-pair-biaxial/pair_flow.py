"""A charge and an anticharge with their cores held in Dirichlet balls: small-step gradient flow of the rest.

Uniaxial vacuum (beta = 0): spectrum (E_0, 1, 0.01, 0.01), the setting of the development repository
(pair.py --full --flow, n = 48, box 16, hold radius 1.2, d = 4). Biaxial vacuum (beta > 0): spectrum
(E_0, 1, beta, 0) with a constant frame at the boundary (charge direction z, the eigenvalue beta along x).

Seed: the director of the two defects (pair.py: electrostatic analogy, a radial hedgehog at +d/2 z and a
hyperbolic one at -d/2 z, z far away), each with the melted core of the relaxed single hedgehog (profiles
from the committed n = 48 field), plus, for beta > 0, the transverse splitting along t = x - (x.n) n, which
is singular where n is parallel to x: there the splitting is switched off, and the flow decides where the
strands of the transverse pair go.

Flow: gradient descent with backtracking (the step never raises the energy), gradient masked to the free
cells outside the two balls. Recorded every `every` steps: E, E_4, E_V and the signed degree of the charge
direction on spheres of radius 1.6 and 2.0 around each held core (outside the ball). At the end: the
transverse splitting e_2 - e_3 along the z axis (where the strand between the cores runs if it forms).
Output: results/pair_flow_<tag>.json (and the final field in results/fields/).

python pair_flow.py --beta 0.3 --d 6 --steps 3000
"""
import argparse
import json
import math
import os
import time

import torch
from lagrangian import eigvalsh3
from soliton import Grid, to_matrix, to_vector
from pair import single_profiles, pair_full, pair_director, signed_degree

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def biaxial_vacuum_field(x, E):
    """Constant biaxial vacuum: n = z with E_1, x with E_2 = beta, y with E_3 = 0."""
    M = torch.zeros(*x.shape[:-1], 4, 4, dtype=x.dtype, device=x.device)
    M[..., 0, 0] = E[0]
    M[..., 1, 1], M[..., 2, 2], M[..., 3, 3] = -E[2], -E[3], -E[1]
    return to_vector(M)


def seed(grid, E, d, prof, hold_radius, beta):
    # the core profiles of the single hedgehog (uniaxial, transverse eigenvalue 0.01) mapped onto the mean
    # transverse eigenvalue of this vacuum: the middle eigenvalue shifted, the gap rescaled
    mid = 0.5 * (E[2] + E[3])
    rs, gaps, mids = prof
    prof = (rs, gaps * (E[1] - mid) / (1.0 - 0.01), mids - 0.01 + mid)
    u, hold = pair_full(grid, (E[0], E[1], mid, mid), d, prof, hold_radius)
    if beta > 0:
        n = pair_director(grid.x, d)
        xh = torch.zeros_like(n)
        xh[..., 0] = 1.0
        t = xh - (xh * n).sum(-1, keepdim=True) * n
        w = (t * t).sum(-1)                                   # |t|^2: 0 where n is parallel to x
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        s = torch.cross(n, t, dim=-1)
        a = torch.zeros(3, dtype=grid.x.dtype, device=grid.x.device)
        a[2] = d / 2
        f = (torch.tanh((grid.x - a).norm(dim=-1)) ** 2 * torch.tanh((grid.x + a).norm(dim=-1)) ** 2) * w ** 2
        split = 0.5 * (E[2] - E[3]) * f[..., None, None] * (t[..., :, None] * t[..., None, :] - s[..., :, None] * s[..., None, :])
        M = to_matrix(u)
        M[..., 1:, 1:] = M[..., 1:, 1:] - split
        u = to_vector(M)
    return u, hold


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--d', type=float, default=4.0)
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--box', type=float, default=16.0)
    ap.add_argument('--hold', type=float, default=1.2)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--single', default=os.path.join(HERE, 'results', 'fields', 'static_base_n48.pt'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    tag = a.tag or f'b{a.beta:g}_d{a.d:g}'
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    E = (100.0, 1.0, 0.01, 0.01) if a.beta == 0 else (100.0, 1.0, a.beta, 0.0)
    grid = Grid(a.n, a.box, E=E, degree=0, device=dev)
    if a.beta > 0:
        grid.boundary = biaxial_vacuum_field(grid.boundary.new_zeros(*grid.boundary.shape[:-1], 3), E)
    prof = single_profiles(a.single)
    u, hold = seed(grid, E, a.d, prof, a.hold, a.beta)
    aa = torch.zeros(3, dtype=u.dtype, device=u.device)
    aa[2] = a.d / 2
    trace, tau, t0 = [], 2e-3, time.time()
    for k in range(a.steps + 1):
        with torch.no_grad():
            Ecur = float(grid.energy(u))
        if k % a.every == 0 or k == a.steps:
            E4, EV, _ = (float(t) for t in grid.energy_terms(u))
            deg = [[signed_degree(grid, u, c, r) for r in (1.6, 2.0)] for c in (aa, -aa)]
            trace.append(dict(step=k, E=Ecur, E4=E4, EV=EV, degrees=deg, tau=tau))
            print(f'[{tag}] step {k:5d}: E {Ecur:.5f} (E4 {E4:.4f}, EV {EV:.4f}), |deg| '
                  f'{abs(deg[0][0]):.3f} {abs(deg[1][0]):.3f} [{time.time()-t0:.0f}s]', flush=True)
        if k == a.steps:
            break
        p_ = u.clone().requires_grad_(True)
        g = torch.autograd.grad(grid.energy(p_), p_)[0] * hold
        tau = min(tau * 1.5, 2e-2)
        while True:
            trial = (u - tau * g).detach()
            with torch.no_grad():
                if float(grid.energy(trial)) <= Ecur or tau < 1e-8:
                    break
            tau *= 0.5
        u = trial
    # the transverse splitting along the z axis
    zs = torch.linspace(-a.box / 2 + 0.5, a.box / 2 - 0.5, 121, device=u.device)
    pts = torch.stack([torch.full_like(zs, 1e-3), torch.full_like(zs, 1e-3), zs], -1)
    ev = eigvalsh3(-to_matrix(grid.sample(u, pts))[..., 1:, 1:])
    out = dict(beta=a.beta, d=a.d, n=a.n, box=a.box, hold=a.hold, E=E, steps=a.steps, trace=trace,
               axis_z=zs.tolist(), axis_eigs=ev.tolist())
    json.dump(out, open(os.path.join(HERE, 'results', f'pair_flow_{tag}.json'), 'w'), indent=1)
    torch.save({'u': u.cpu(), 'n': a.n, 'box': a.box, 'E': E, 'd': a.d, 'beta': a.beta},
               os.path.join(HERE, 'results', 'fields', f'pair_flow_{tag}.pt'))
