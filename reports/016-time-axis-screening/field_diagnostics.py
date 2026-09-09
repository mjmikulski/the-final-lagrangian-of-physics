"""CPU diagnostics of the committed lattice endpoints (results/fields/*.pt): energy terms, the frame term
K_u, the gradient norm on the free components, shell profiles of the radial tilt M_0i and of the charge
eigenvalue e_1, and the far-field density against the Coulomb law of the frozen hedgehog.

python field_diagnostics.py            # writes results/field_diagnostics.json
"""
import glob
import json
import math
import os
import torch
from soliton import Grid, to_matrix
from lagrangian import terms, eigenvalues, operator

torch.set_default_dtype(torch.float64)


def profiles(grid, u, dens, nb=12):
    """Shell means of the radial component of M_0i, of e_1 (eta-pencil eigenvalue), and of the energy density."""
    M = to_matrix(u)
    E0 = grid.E[0]
    r = grid.x.norm(dim=-1)
    radial = (M[..., 0, 1:] * grid.x).sum(-1) / r.clamp_min(1e-9)
    e = eigenvalues(operator(M), math.sqrt(E0 * grid.E[1]))
    edges = torch.linspace(0, grid.n * grid.h / 2, nb + 1)
    rq = grid.xq.norm(dim=-1)
    rows = []
    for i in range(nb):
        m = (r >= edges[i]) & (r < edges[i + 1])
        mq = (rq >= edges[i]) & (rq < edges[i + 1])
        rc = float(0.5 * (edges[i] + edges[i + 1]))
        rows.append(dict(r=rc, m0i_radial=float(radial[m].mean()), e1=float(e[..., 1][m].mean()), e0=float(e[..., 0][m].mean()),
                         density=float(dens[mq].mean()), coulomb_ratio=float(dens[mq].mean() * rc ** 4 / (4 * (grid.E[1] - grid.E[2]) ** 4))))
    return rows


def diagnose(path):
    d = torch.load(path)
    X, frozen = d.get('X'), d['frozen']
    grid = Grid(d['n'], d['box'], E=d['E'], freeze_time=frozen, X=X, device='cpu')
    u = d['u'].double().clone().requires_grad_(True)
    E4, EV, EX = grid.energy_terms(u)
    E = E4 + EV + EX
    g = torch.autograd.grad(E, u)[0] * grid.mask
    u = u.detach()
    with torch.no_grad():
        M, dM = grid.derivatives(u)
        kin, pot, xx = terms(M, dM, grid.E, frozen=frozen, X=X)
        dens = -kin + pot - xx
        # the frame term evaluated on this field with unit coupling, whatever the run's own coupling
        Ku = float(-terms(M, dM, grid.E, frozen=False, X={'tilt': 1.0})[2].sum() * grid.vol)
        tilt_max = float(to_matrix(u)[..., 0, 1:].abs().max())
    return dict(file=os.path.basename(path), n=d['n'], box=d['box'], E0=d['E'][0], frozen=bool(frozen), tilt=float((X or {}).get('tilt', 0.0)),
                E4=float(E4), EV=float(EV), E_Ku=float(EX), E=float(E), grad=float(g.norm()), Ku_unit=Ku, tilt_max=tilt_max,
                profiles=profiles(grid, u, dens))


if __name__ == '__main__':
    out = []
    for f in sorted(glob.glob('results/fields/*.pt')):
        row = diagnose(f)
        out.append(row)
        tail = [round(p['coulomb_ratio'], 3) for p in row['profiles'][-3:]]
        print(f"{row['file']:40s} E0={row['E0']:<6g} c={row['tilt']:<5g} E4={row['E4']:.4f} EV={row['EV']:.4f} E_Ku={row['E_Ku']:.4f} "
              f"E={row['E']:.6f} |grad|={row['grad']:.1e} K_u(unit)={row['Ku_unit']:.2e} max|M_0i|={row['tilt_max']:.3f} tail={tail}", flush=True)
    json.dump(out, open('results/field_diagnostics.json', 'w'), indent=1)
