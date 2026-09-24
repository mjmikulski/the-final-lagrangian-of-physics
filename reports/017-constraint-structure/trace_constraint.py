"""The trace constraint tr N = E_0 + E_1 + E_2 + E_3 on the committed lattice fields, and the blindness of the
derivative sector to the trace.

(a) Blindness: add s(x) * 1 to N (covariant M += s(x) eta, s a smooth bump of amplitude 0.3 that is not
    constant anywhere in the core) to every committed field and compare the three pieces of the static energy:
    the quartic term and K_u must not change, the potential must.
(b) The constraint: at the quadrature points of every field, |tr N - sum E| against the variation of the
    eigenvalue e_1 across the same points (a field on which the eigenvalues barely move would satisfy the
    constraint trivially).
(c) The rigid-rotation inertia of the frozen hedgehog for the internal, the spatial and the combined generator,
    and the coefficient b of the term linear in the angular velocity (report 015's rotor L = a w^2 + b w + c).
Output: results/trace_constraint.json.
"""
import json
import torch
from lagrangian import terms, eigenvalues, operator
from soliton import Grid, G_Z

torch.set_default_dtype(torch.float64)
FIELDS = '../016-time-axis-screening/results/fields/'
LIST = {'frozen hedgehog': ('static_base_n32.pt', 0.0, True),
        'time sector free, E0 = 10': ('static_unfrozen_E0_10.pt', 0.0, False),
        'time sector free, E0 = 100': ('static_unfrozen_E0_100.pt', 0.0, False),
        'time sector free, E0 = 1000': ('static_unfrozen_E0_1000.pt', 0.0, False),
        'K_u, c = 0.03': ('static_tilt0.03.pt', 0.03, False),
        'K_u, c = 0.1': ('static_tilt0.1.pt', 0.1, False),
        'K_u, c = 0.3': ('static_tilt0.3.pt', 0.3, False),
        'K_u, c = 3': ('static_tilt3.0.pt', 3.0, False)}
ETA_VEC = torch.tensor([1.0, 0, 0, 0, -1.0, 0, 0, -1.0, 0, -1.0])     # covariant M = eta, i.e. N -> N + 1


def pieces(grid, u, c):
    M, dM = grid.derivatives(u)
    kin, pot, x = terms(M, dM, grid.E, frozen=False, X={'tilt': c} if c else None)
    return float(-kin.sum() * grid.vol), float(pot.sum() * grid.vol), float(-x.sum() * grid.vol)


def rotor(grid, u, which):
    M, dM = grid.derivatives(u)
    G = G_Z.to(M)
    x, y = grid.xq[..., 0, None, None], grid.xq[..., 1, None, None]
    gen = {'internal': G @ M + M @ G.mT, 'spatial': y * dM[..., 1, :, :] - x * dM[..., 2, :, :]}
    gen['combined'] = gen['internal'] + gen['spatial']
    L = []
    for w in (1.0, -1.0, 0.0):
        d2 = torch.cat([(w * gen[which])[..., None, :, :], dM[..., 1:, :, :]], -3)
        kin, pot, xx = terms(M, d2, grid.E, frozen=False)
        L.append(float((kin + xx - pot).sum() * grid.vol))
    return {'a': 0.5 * (L[0] + L[1]) - L[2], 'b': 0.5 * (L[0] - L[1]),
            'generator_norm': float(gen[which].norm())}


if __name__ == '__main__':
    out = {}
    for name, (fn, c, frozen) in LIST.items():
        d = torch.load(FIELDS + fn)
        grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
        u = d['u'].double()
        r = grid.x.norm(dim=-1)
        s = 0.3 * torch.exp(-r ** 2 / 4)[..., None]
        base, shifted = pieces(grid, u, c), pieces(grid, u + s * ETA_VEC, c)
        M, _ = grid.derivatives(u)
        N = operator(M)
        e = eigenvalues(N, c=(d['E'][0] * d['E'][1]) ** 0.5)
        res = torch.diagonal(N, dim1=-2, dim2=-1).sum(-1) - sum(d['E'])
        out[name] = {'field': fn, 'c': c,
                     'quartic_change_under_trace_shift': abs(shifted[0] - base[0]) / abs(base[0]),
                     'K_u_change_under_trace_shift': (abs(shifted[2] - base[2]) / abs(base[2])) if c else None,
                     'potential_change_under_trace_shift': abs(shifted[1] - base[1]) / abs(base[1]),
                     'trace_residual_max': float(res.abs().max()),
                     'trace_residual_rms': float(res.pow(2).mean().sqrt()),
                     'e1_min': float(e[..., 1].min()), 'e1_max': float(e[..., 1].max()),
                     'relaxation_gradient_norm': d.get('grad')}
        print(name, out[name])
    d = torch.load(FIELDS + LIST['frozen hedgehog'][0])
    grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
    out['rotor, frozen hedgehog'] = {w: rotor(grid, d['u'].double(), w) for w in ('internal', 'spatial', 'combined')}
    print(out['rotor, frozen hedgehog'])
    json.dump(out, open('results/trace_constraint.json', 'w'), indent=1)
