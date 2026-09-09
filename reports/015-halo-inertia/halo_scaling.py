"""The halo mechanism measured on the relaxed lattice hedgehog: cost and inertia of an internal tilt.

The exterior of the hedgehog is rotated internally about the x axis by an angle delta s(r), with s a
smoothstep rising from 0 to 1 over [R1, R1 + L] (outside the core) and falling back to 0 over a second zone
of the same thickness before the box boundary. Prediction (halo_check.py, exact on the uniaxial
exterior): the static cost is (32 pi / 3) Delta^4 int theta'^2 dr and the inertia gained for rotation about z is
(64 pi / 3) Delta^4 int 4 sin^2(theta/2) (1 + r^2 theta'^2 / 2) dr, theta = delta s(r), Delta = E_1 - E_2; the
small-angle limit of the latter is (64 pi / 3) Delta^4 int theta^2 dr. Both are measured here by evaluating the
static energy and the rigid-rotor functional of the tilted field (CPU, committed fields).

python halo_scaling.py [--static results/fields/static_base_n48.pt --R1 3 --L 1 2 3 --delta 0.05 0.1 0.2 --tag n48_box18]
"""
import argparse
import json
import numpy as np
import torch
from soliton import Grid, to_matrix, to_vector

torch.set_default_dtype(torch.float64)


def smoothstep(t):
    t = t.clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


def profile(r, R1, L1, R2, L2):
    return smoothstep((r - R1) / L1) * (1 - smoothstep((r - R2) / L2))


def tilt(grid, u, delta, R1, L1, R2, L2):
    """M -> R M R^T with R the rotation about x by delta s(r)."""
    th = delta * profile(grid.x.norm(dim=-1), R1, L1, R2, L2)
    c, sn = torch.cos(th), torch.sin(th)
    R = torch.zeros(*th.shape, 4, 4, dtype=u.dtype)
    R[..., 0, 0] = 1; R[..., 1, 1] = 1
    R[..., 2, 2] = c; R[..., 2, 3] = -sn; R[..., 3, 2] = sn; R[..., 3, 3] = c
    return to_vector(R @ to_matrix(u) @ R.transpose(-1, -2))


def integrals(R1, L1, R2, L2, alpha):
    """Radial integrals of the tilt profile theta = alpha s(r) (fine quadrature): the exact cost integral
    int theta'^2 dr, the small-angle inertia integral int theta^2 dr, and the exact inertia integral
    int 4 sin^2(theta/2) (1 + r^2 theta'^2 / 2) dr of halo_check.py."""
    r = torch.linspace(0, R2 + L2 + 1, 200001)
    th = alpha * profile(r, R1, L1, R2, L2)
    dth = torch.gradient(th, spacing=(r,))[0]
    return (float(torch.trapezoid(dth ** 2, r)), float(torch.trapezoid(th ** 2, r)),
            float(torch.trapezoid(4 * torch.sin(th / 2) ** 2 * (1 + r ** 2 * dth ** 2 / 2), r)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--static', default='results/fields/static_base_n48.pt')
    ap.add_argument('--R1', type=float, default=3.0)
    ap.add_argument('--L', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    ap.add_argument('--delta', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    ap.add_argument('--margin', type=float, default=0.4)
    ap.add_argument('--tag', default='n48_box18')
    a = ap.parse_args()
    d = torch.load(a.static)
    grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
    u0 = d['u'].to(torch.float64)
    Delta = d['E'][1] - d['E'][2]
    with torch.no_grad():
        E0 = float(grid.energy(u0))
        a0, b0, c0 = (float(t) for t in grid.rotor(u0))
    print(f'{a.static}: E = {E0:.4f}, rigid inertia 2a = {2 * a0:.4f}', flush=True)
    rows = []
    for L in a.L:
        R2 = d['box'] / 2 - L - a.margin
        for delta in a.delta:
            I1, I2, I3 = integrals(a.R1, L, R2, L, delta)
            u = tilt(grid, u0, delta, a.R1, L, R2, L)
            with torch.no_grad():
                E = float(grid.energy(u))
                aa, b, c = (float(t) for t in grid.rotor(u))
            row = dict(L=L, delta=delta, R2=R2, x_cost=I1, x_inertia=I2, x_inertia_exact=I3,
                       cost=E - E0, inertia_excess=2 * aa - 2 * a0)
            print({k: round(v, 5) for k, v in row.items()}, flush=True)
            rows.append(row)
    xc = np.array([r['x_cost'] for r in rows]); yc = np.array([r['cost'] for r in rows])
    xi = np.array([r['x_inertia'] for r in rows]); yi = np.array([r['inertia_excess'] for r in rows])
    xe = np.array([r['x_inertia_exact'] for r in rows])
    slope_cost = float((xc * yc).sum() / (xc * xc).sum())
    slope_inertia = float((xi * yi).sum() / (xi * xi).sum())
    slope_inertia_exact = float((xe * yi).sum() / (xe * xe).sum())
    pred_cost, pred_inertia = 32 * np.pi / 3 * Delta ** 4, 64 * np.pi / 3 * Delta ** 4
    small = [r for r in rows if r['delta'] == min(a.delta)]
    p_delta = np.polyfit(np.log([r['delta'] for r in rows if r['L'] == a.L[0]]), np.log([r['cost'] for r in rows if r['L'] == a.L[0]]), 1)[0]
    q_delta = np.polyfit(np.log([r['delta'] for r in rows if r['L'] == a.L[0]]), np.log([r['inertia_excess'] for r in rows if r['L'] == a.L[0]]), 1)[0]
    print(f'prefactors through the origin: cost {slope_cost:.2f} (exact {pred_cost:.2f}); inertia against the small-angle integral {slope_inertia:.2f}, '
          f'against the exact integral {slope_inertia_exact:.2f} (exact {pred_inertia:.2f}); exponents in delta at L = {a.L[0]}: cost {p_delta:.2f}, inertia {q_delta:.2f}', flush=True)
    json.dump(dict(static=a.static, n=d['n'], box=d['box'], E=E0, rigid_inertia=2 * a0, Delta=Delta, rows=rows,
                   slope_cost=slope_cost, slope_inertia=slope_inertia, slope_inertia_exact=slope_inertia_exact,
                   predicted_cost=pred_cost, predicted_inertia=pred_inertia,
                   ratio_cost=slope_cost / pred_cost, ratio_inertia=slope_inertia / pred_inertia, ratio_inertia_exact=slope_inertia_exact / pred_inertia,
                   exponent_delta_cost=float(p_delta), exponent_delta_inertia=float(q_delta)),
              open(f'results/halo_scaling_{a.tag}.json', 'w'), indent=1)
