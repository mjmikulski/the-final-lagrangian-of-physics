"""The electron as a one-dimensional problem: spherical ansatz with a radial boost (symbolic.radial_ansatz),
minimised on a radial grid without a box. Profiles e_0(r), e_1(r), e_2(r) = e_3(r), chi(r).

Boundary conditions: e_1(0) = e_2(0) (isotropic centre), chi(0) = 0; at r = R the eigenvalues take their
vacuum values and chi is free (it settles at the screening value). Frozen variant: chi = 0 everywhere.

python radial_screened.py [--R 300 --n 3000 --E0 100]
"""
import argparse
import json
import math
import numpy as np
import sympy as sp
import torch
from symbolic import radial_ansatz

torch.set_default_dtype(torch.float64)


def build(E, tilt=0.0):
    H4, V, s = radial_ansatz()
    args = [s[k] for k in ('r', 'e0', 'e1', 'e2', 'S', 'C', 'de0', 'de1', 'de2', 'dchi')]
    dens = (H4 + V + tilt * s['Ku']).subs({s['E0']: E[0], s['E1']: E[1], s['E2']: E[2]})
    f = sp.lambdify(args, dens, 'torch')
    return f


def energy(f, r, e0, e1, e2, chi):
    """4 pi int (H4 + V) r^2 dr with midpoint values and finite differences."""
    rm = 0.5 * (r[1:] + r[:-1]); dr = r[1:] - r[:-1]
    mid = lambda q: 0.5 * (q[1:] + q[:-1])
    der = lambda q: (q[1:] - q[:-1]) / dr
    S, C = torch.sinh(mid(chi)), torch.cosh(mid(chi))
    dens = f(rm, mid(e0), mid(e1), mid(e2), S, C, der(e0), der(e1), der(e2), der(chi))
    return 4 * math.pi * (dens * rm ** 2 * dr).sum()


def spline_basis(r, knots, k=3):
    """Cubic B-spline design matrices (values and first derivatives) on the points r for the given knots."""
    from scipy.interpolate import BSpline
    t = np.concatenate([[knots[0]] * k, knots, [knots[-1]] * k])
    nb = len(t) - k - 1
    Bv = np.zeros((len(r), nb)); Bd = np.zeros((len(r), nb))
    for j in range(nb):
        c = np.zeros(nb); c[j] = 1
        sp_ = BSpline(t, c, k)
        Bv[:, j] = sp_(r); Bd[:, j] = sp_.derivative()(r)
    return torch.as_tensor(Bv), torch.as_tensor(Bd)


def solve_spline(E, R=300.0, nk=120, nq=6000, frozen=False, iters=3000, tilt=0.0):
    """Same problem with the four profiles expanded in cubic B-splines on log-spaced knots (no grid-scale
    noise possible). Quadrature on a fine log-spaced grid; the energy density from symbolic.radial_ansatz."""
    f = build(E, tilt)
    knots = np.concatenate([[0.0], np.logspace(-2, math.log10(R), nk - 1)])
    rq = np.concatenate([np.linspace(1e-4, 0.01, 40), np.logspace(-2, math.log10(R), nq)])
    wq = np.gradient(rq)                                   # quadrature weights (trapezoid-like)
    Bv, Bd = spline_basis(rq, knots)
    rq_t, wq_t = torch.as_tensor(rq), torch.as_tensor(wq)
    nb = Bv.shape[1]
    chi_star = math.asinh(math.sqrt((E[1] - E[2]) ** 2 / ((E[0] - E[2]) ** 2 - (E[1] - E[2]) ** 2)))
    # initial coefficients from smooth profiles (least squares)
    r0 = rq
    core = np.exp(-r0 ** 2 / 2)
    init = np.stack([np.zeros_like(r0), E[1] - (E[1] - E[2]) * core, E[2] + 0.3 * core, (0 * r0 if frozen else chi_star * (1 - core)) * E[0]])
    coef = torch.as_tensor(np.linalg.lstsq(Bv.numpy(), init.T, rcond=None)[0].T).clone()   # (4, nb): a, e1, e2, c = chi E0
    # constraints: vacuum values at R (last coefficient), chi(0) = 0 (first coefficient of c), e2(0) = e1(0) (first coefficients equal)
    free = torch.ones_like(coef, dtype=torch.bool)
    free[:, -1] = False
    free[3, -1] = not frozen
    free[3, 0] = False
    free[2, 0] = False
    if frozen:
        free[3] = False
    q = coef[free].clone().requires_grad_(True)

    def assemble(q):
        full = coef.clone()
        full[free] = q
        full[2, 0] = full[1, 0]
        a, e1, e2, c = full
        vals = [Bv @ v for v in (a, e1, e2, c)]
        ders = [Bd @ v for v in (a, e1, e2, c)]
        return vals, ders

    def energy_q(q):
        (a, e1, e2, c), (da, de1, de2, dc) = assemble(q)
        chi = c / E[0]; dchi = dc / E[0]
        dens = f(rq_t, E[0] + a, e1, e2, torch.sinh(chi), torch.cosh(chi), da, de1, de2, dchi)
        return 4 * math.pi * (dens * rq_t ** 2 * wq_t).sum()

    def closure():
        opt.zero_grad()
        Ev = energy_q(q)
        Ev.backward()
        return Ev
    for rnd in range(6):
        opt = torch.optim.LBFGS([q], lr=1, max_iter=iters, tolerance_grad=1e-10, tolerance_change=1e-16, history_size=100, line_search_fn='strong_wolfe')
        opt.step(closure)
        print(f'    spline round {rnd}: E = {float(energy_q(q).detach()):.5f}, |grad| = {float(q.grad.norm()):.1e}', flush=True)
        if float(q.grad.norm()) < 1e-7:
            break
    with torch.no_grad():
        (a, e1, e2, c), (da, de1, de2, dc) = assemble(q)
        chi = c / E[0]
        dens = f(rq_t, E[0] + a, e1, e2, torch.sinh(chi), torch.cosh(chi), da, de1, de2, dc / E[0])
        Etot = float(4 * math.pi * (dens * rq_t ** 2 * wq_t).sum())
        inside = float(4 * math.pi * (dens * rq_t ** 2 * wq_t)[rq_t < 6].sum())
        prof = dict(r=rq.tolist(), e0=(E[0] + a).numpy().tolist(), e1=e1.numpy().tolist(), e2=e2.numpy().tolist(), chi=chi.numpy().tolist(),
                    density=(4 * math.pi * dens * rq_t ** 2).numpy().tolist())
    return dict(E=Etot, E_inside_6=inside, grad=float(q.grad.norm()), frozen=frozen, R=R, nk=nk, E0=E[0], tilt=tilt, method='spline'), prof


def solve(E, R=300.0, n=3000, frozen=False, iters=3000, init=None):
    f = build(E)
    r = torch.cat([torch.linspace(0, 10, n // 2 + 1)[1:], torch.logspace(1, math.log10(R), n // 2 + 1)[1:]])
    r = torch.cat([torch.tensor([0.0]), r])
    # initial profiles: hedgehog with a melted core of size 1, boost rising to the screening value
    if init is None:
        core = torch.exp(-r ** 2 / 2)
        e1 = E[1] - (E[1] - E[2]) * core
        e2 = torch.full_like(r, E[2]) + 0.3 * core
        e0 = torch.full_like(r, E[0])
        chi_star = math.asinh(math.sqrt((E[1] - E[2]) ** 2 / ((E[0] - E[2]) ** 2 - (E[1] - E[2]) ** 2)))
        chi = torch.zeros_like(r) if frozen else chi_star * (1 - core)
    else:
        e0, e1, e2, chi = (torch.as_tensor(np.interp(r.numpy(), init['r'], init[k])) for k in ('e0', 'e1', 'e2', 'chi'))
    # free variables: interior values; the boundary values are held (centre isotropic, vacuum at R).
    # Scaled variables: e0 = E0 + a, chi = c / E0, so that all unknowns are O(1) and the Hessian is balanced.
    p = torch.stack([e0 - E[0], e1, e2, chi * E[0]]).clone()
    free = torch.ones_like(p, dtype=torch.bool)
    free[:, -1] = False                      # vacuum at R (chi free at R)
    free[3, -1] = not frozen
    free[3, 0] = False                       # chi(0) = 0
    if frozen:
        free[3] = False
    q = p[free].clone().requires_grad_(True)

    def assemble(q):
        full = p.clone()
        full[free] = q
        a, e1, e2, c = full
        e2c = torch.cat([e1[:1], e2[1:]])    # the centre is isotropic: e_2(0) := e_1(0)
        return E[0] + a, e1, e2c, c / E[0]

    def closure():
        opt.zero_grad()
        Ev = energy(f, r, *assemble(q))
        Ev.backward()
        return Ev
    for rnd in range(4):
        opt = torch.optim.LBFGS([q], lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-16, history_size=100, line_search_fn='strong_wolfe')
        opt.step(closure)
        with torch.no_grad():
            print(f'    round {rnd}: E = {float(energy(f, r, *assemble(q))):.5f}, |grad| = {float(q.grad.norm()):.1e}', flush=True)
        if float(q.grad.norm()) < 1e-6:
            break
    with torch.no_grad():
        e0, e1, e2, chi = assemble(q)
        Etot = float(energy(f, r, e0, e1, e2, chi))
        # split: gradient vs potential, and the energy inside r < 6 for comparison with the box
        prof = dict(r=r.numpy().tolist(), e0=e0.numpy().tolist(), e1=e1.numpy().tolist(), e2=e2.numpy().tolist(), chi=chi.numpy().tolist())
        m = 0.5 * (r[1:] + r[:-1]) < 6
        rm = 0.5 * (r[1:] + r[:-1]); dr = r[1:] - r[:-1]
        mid = lambda x: 0.5 * (x[1:] + x[:-1]); der = lambda x: (x[1:] - x[:-1]) / dr
        dens = f(rm, mid(e0), mid(e1), mid(e2), torch.sinh(mid(chi)), torch.cosh(mid(chi)), der(e0), der(e1), der(e2), der(chi))
        inside = float(4 * math.pi * (dens * rm ** 2 * dr)[m].sum())
    return dict(E=Etot, E_inside_6=inside, grad=float(q.grad.norm()), frozen=frozen, R=R, n=n, E0=E[0]), prof


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=float, default=300.0)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--E0', type=float, nargs='+', default=[100.0])
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--spline', type=int, default=0, help='number of B-spline knots (0 = plain grid)')
    ap.add_argument('--tilt', type=float, default=0.0, help='coefficient of the time-axis tilt term K_u')
    ap.add_argument('--which', default='both', choices=['both', 'frozen', 'screened'], help='the frozen problem does not depend on the tilt coupling')
    a = ap.parse_args()
    for E0 in a.E0:
        E = (E0, 1.0, 0.01, 0.01)
        out = {}
        for frozen in {'both': (True, False), 'frozen': (True,), 'screened': (False,)}[a.which]:
            res, prof = (solve_spline(E, a.R, a.spline, frozen=frozen, iters=a.iters, tilt=a.tilt) if a.spline else solve(E, a.R, a.n, frozen, a.iters))
            tag = 'frozen' if frozen else 'screened'
            print(f'E0 = {E0:g} {tag}: E = {res["E"]:.4f} (inside r < 6: {res["E_inside_6"]:.4f}), |grad| = {res["grad"]:.1e}', flush=True)
            i6 = np.searchsorted(np.array(prof['r']), 6.0)
            print(f'   e1(0) = {prof["e1"][0]:.3f}, e1(1) = {np.interp(1.0, prof["r"], prof["e1"]):.3f}, chi(r=6) = {prof["chi"][i6]:.4f}, chi(R) = {prof["chi"][-1]:.4f}, chi* = {math.asinh(math.sqrt((E[1]-E[2])**2/((E[0]-E[2])**2-(E[1]-E[2])**2))):.4f}', flush=True)
            out[tag] = dict(res, profile=prof)
        json.dump(out, open(f'results/radial_E0_{E0:g}{f"_nk{a.spline}" if a.spline else "_grid"}{f"_tilt{a.tilt:g}" if a.tilt else ""}.json', 'w'))
