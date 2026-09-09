"""The frozen hedgehog as a stationary point of the full 4x4 energy, and its stability in the time sector.

(a) The static energy of a frozen configuration (M_00 = E_0, M_0i = 0) does not depend on E_0: the
    derivatives have no time row, F is block-diagonal and (e_0 - E_0)^2 vanishes.
(b) The frozen minimiser is a stationary point of the full energy: its time-sector gradient vanishes.
(c) Rayleigh quotients of the full Hessian along time-sector directions -- M_0i = f(r) x_i (radial tilt),
    f(r) (z x x)_i, f(r) e_z, random localised, and M_00 = f(r) -- for several E_0 and several couplings c
    of the frame term K_u. Runs on the CPU on the committed frozen field.

python time_sector_check.py --E0 100 --tilt 0 0.001 0.01 0.03 0.1            # the coupling scan (results/time_sector_check.json)
python time_sector_check.py --E0 10 3 --tilt 0 --out results/time_sector_check_E0.json   # the E_0 dependence
"""
import argparse
import json
import torch
from soliton import Grid, to_matrix, to_vector

torch.set_default_dtype(torch.float64)


def directions(grid):
    x = grid.x
    r = x.norm(dim=-1, keepdim=True)
    f = torch.exp(-r ** 2 / 8)
    z = torch.zeros_like(x); z[..., 2] = 1
    cands = {'radial M_0i ~ x_i': f * x, 'rotational M_0i ~ (z x x)_i': f * torch.cross(z.expand_as(x), x, dim=-1),
             'uniform M_0i ~ e_z': f * z, 'random M_0i': f * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))}
    out = {}
    for name, v in cands.items():
        M = torch.zeros(*x.shape[:-1], 4, 4)
        M[..., 0, 1:] = v; M[..., 1:, 0] = v
        out[name] = to_vector(M) * grid.mask
    M = torch.zeros(*x.shape[:-1], 4, 4); M[..., 0, 0] = f[..., 0]
    out['M_00 ~ f(r)'] = to_vector(M) * grid.mask
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--static', default='results/fields/static_base_n32.pt')
    ap.add_argument('--E0', type=float, nargs='+', default=[100.0, 10.0, 3.0])
    ap.add_argument('--tilt', type=float, nargs='+', default=[0.0, 0.001, 0.01, 0.03, 0.1])
    ap.add_argument('--out', default='results/time_sector_check.json')
    a = ap.parse_args()
    d = torch.load(a.static)
    E = d['E']
    M = to_matrix(d['u'].to(torch.float64)); M[..., 0, 1:] = 0; M[..., 1:, 0] = 0
    report = []
    for E0 in a.E0:
        Ev = (E0, E[1], E[2], E[3])
        M[..., 0, 0] = E0
        u = to_vector(M)
        with torch.no_grad():
            Ef = float(Grid(d['n'], d['box'], E=Ev, device='cpu').energy(u))
        for c in a.tilt:
            gu = Grid(d['n'], d['box'], E=Ev, freeze_time=False, device='cpu', X=({'tilt': c} if c else None))
            p = u.clone().requires_grad_(True)
            Eu = gu.energy(p)
            g = torch.autograd.grad(Eu, p, create_graph=True)[0]
            gM = to_matrix(g * gu.mask)
            row = dict(E0=E0, tilt=c, E_frozen=Ef, E_full=float(Eu), grad_time=float(gM[..., 0, :].norm()),
                       grad_space=float(gM[..., 1:, 1:].norm()), rayleigh={})
            for name, v in directions(gu).items():
                Hv = torch.autograd.grad((g * v).sum(), p, retain_graph=True)[0] * gu.mask
                row['rayleigh'][name] = float((v * Hv).sum() / (v * v).sum())
            report.append(row)
            print(f'E0 = {E0:g}, c = {c:g}: frozen {Ef:.6f}, full {float(Eu):.6f}, |grad| time sector {row["grad_time"]:.1e} '
                  f'(space {row["grad_space"]:.1e}); Rayleigh: ' + ', '.join(f'{k.split()[0]} {v:+.5f}' for k, v in row['rayleigh'].items()), flush=True)
    json.dump(report, open(a.out, 'w'), indent=1)
