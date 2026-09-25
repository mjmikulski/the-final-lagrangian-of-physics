"""One charge in a biaxial vacuum (E_0, 1, beta, 0): where do the strands of the transverse pair go?

The boundary is the hedgehog with the spherical frame (soliton.hedgehog with split=True): the charge direction
radial, the eigenvalue beta on theta-hat and 0 on phi-hat. That frame has index +1 at each pole, so the
transverse pair must carry defect lines along the z axis (the Euler number 2 of the sphere). The field is
relaxed in the frozen sector (L-BFGS, rounds of iterations) from the melted-core hedgehog; the reference is
the uniaxial vacuum (1, beta/2, beta/2), relaxed the same way in the same box.

Recorded: E_4, E_V; the energy per unit length in slabs of thickness h along z, integrated over the whole
cross-section, for both vacua (their difference far from the core is the energy the strands add per unit
length, to be compared with the 2D tension of strand2d.py); the transverse splitting e_2 - e_3 along the z and
x axes. Output: results/single_biaxial_b<beta>_n<n>.json.

python single_biaxial.py --beta 0.3 --n 48 --box 18
"""
import argparse
import json
import os
import time

import torch
from lagrangian import terms, eigvalsh3
from soliton import Grid, to_matrix

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def slab_profile(g, u):
    kin, pot, x = terms(*g.derivatives(u), g.E, frozen=g.freeze, X=g.X)
    dens = (-kin + pot) * g.vol                        # [quad, n+1, n+1, n+1]
    z = g.xq[..., 2]
    edges = torch.linspace(-g.n * g.h / 2, g.n * g.h / 2, g.n + 1, device=z.device)
    idx = torch.bucketize(z.flatten(), edges).clamp(1, g.n) - 1
    per = torch.zeros(g.n, dtype=dens.dtype, device=dens.device).index_add_(0, idx, dens.flatten())
    zc = 0.5 * (edges[1:] + edges[:-1])
    return zc.tolist(), (per / g.h).tolist()


def axis_split(g, u, axis):
    s = torch.linspace(-g.n * g.h / 2 + 0.3, g.n * g.h / 2 - 0.3, 121, device=u.device)
    pts = torch.full((121, 3), 1e-3, device=u.device, dtype=u.dtype)
    pts[:, axis] = s
    ev = eigvalsh3(-to_matrix(g.sample(u, pts))[..., 1:, 1:])
    return s.tolist(), (ev[..., 1] - ev[..., 2]).tolist(), ev[..., 0].tolist()


def run(E, split, n, box, rounds, iters, dev):
    g = Grid(n, box, E=E, split=split, device=dev)
    u = g.initial(width=1.0)
    t0 = time.time()
    for r in range(rounds):
        u, gn = g.minimize(u, g.energy, iters=iters)
        E4, EV, _ = (float(t) for t in g.energy_terms(u))
        print(f'  E {E} split {split}: round {r}: E {E4 + EV:.5f} (E4 {E4:.4f}, EV {EV:.4f}), |g| {gn:.1e} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    zc, per = slab_profile(g, u)
    ax_z = axis_split(g, u, 2)
    ax_x = axis_split(g, u, 0)
    return g, u, dict(E=E, split=split, E4=E4, EV=EV, grad=gn, slab_z=zc, slab_energy_per_length=per,
                      axis_z=dict(s=ax_z[0], split=ax_z[1], e1=ax_z[2]), axis_x=dict(s=ax_x[0], split=ax_x[1], e1=ax_x[2]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--box', type=float, default=12.0)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--iters', type=int, default=1500)
    a = ap.parse_args()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    out = {'beta': a.beta, 'n': a.n, 'box': a.box}
    gb, ub, out['biaxial'] = run((100.0, 1.0, a.beta, 0.0), True, a.n, a.box, a.rounds, a.iters, dev)
    gu, uu, out['uniaxial'] = run((100.0, 1.0, a.beta / 2, a.beta / 2), False, a.n, a.box, a.rounds, a.iters, dev)
    json.dump(out, open(os.path.join(HERE, 'results', f'single_biaxial_b{a.beta:g}_n{a.n}.json'), 'w'), indent=1)
    torch.save({'u': ub.cpu(), 'n': a.n, 'box': a.box, 'E': (100.0, 1.0, a.beta, 0.0), 'split': True},
               os.path.join(HERE, 'results', 'fields', f'single_biaxial_b{a.beta:g}_n{a.n}.pt'))
