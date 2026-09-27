"""Appendix to report 018: is the Bogomolny strand the lowest state, and what a two-derivative term changes.

(a) Adversarial seeds for the straight strand (beta = 0.3; spectrum 1, beta, 0), relaxed with the 2D lattice of
    strand2d.py at h = rho_1/2 / 8 (all seeds) and / 16 (the unmelted seeds, k = 1/2); box R = 8 rho_1/2:
      smooth    the melted-core seed of the report (control);
      unmelted  theta = k phi with the full splitting beta up to the line (no melted core): away from the line the
                texture depends on phi only, so F_xy = 0 and V = 0 there, and in the continuum its energy is zero;
                on the lattice the line cell carries gradients of order beta / h;
      wedge     unmelted, with the whole winding concentrated in a wedge next to the +x axis,
                theta = 2 pi k (phi / 2 pi)^8: a wall-like sheet of fast rotation ending on the line;
      kicked    the smooth seed plus a random perturbation of every node (amplitude beta / 3).
    Question: does any seed relax below the Bogomolny value T_B, and does the unmelted state's energy fall or
    grow as h -> 0?
(b) The two-derivative term kappa sum_i Tr(d_i N d_i N) (lagrangian.py's `sigma` term; added to the energy with a
    positive sign). Far from the line the winding then costs energy per unit length that grows like ln(R): the
    prediction on the 09-25 thread is 16 pi kappa k^2 b0^2 ln(R / xi) with b0 the half-splitting, so that the
    partitions of a charge's index are ordered by sum k^2. Measured: T(kappa) - T(0) for kappa in {1e-3, 1e-2},
    k in {1/2, 1}, R in {8, 16, 32} rho_1/2 at h = rho_1/2 / 8.
Output: results/strand_walls.json.
"""
import json
import math
import sys
import time

import torch
from lagrangian import terms
from strand2d import from_S, to_S, texture, G, DEV
from strand_bogomolny import r_half as r_half_bogomolny, bound

torch.set_default_dtype(torch.float64)
BETA = 0.3


def energy(u, h, E, kappa=0.0):
    tot = 0.0
    u00, u10, u01, u11 = u[:-1, :-1], u[1:, :-1], u[:-1, 1:], u[1:, 1:]
    X = {'sigma': kappa} if kappa else None
    for gx in G:
        for gy in G:
            val = (1 - gx) * (1 - gy) * u00 + gx * (1 - gy) * u10 + (1 - gx) * gy * u01 + gx * gy * u11
            dx = ((1 - gy) * (u10 - u00) + gy * (u11 - u01)) / h
            dy = ((1 - gx) * (u01 - u00) + gx * (u11 - u10)) / h
            M = torch.zeros(*val.shape[:-1], 4, 4, dtype=u.dtype, device=u.device)
            M[..., 0, 0] = E[0]
            M[..., 1:, 1:] = -to_S(val)
            dM = torch.zeros(*val.shape[:-1], 4, 4, 4, dtype=u.dtype, device=u.device)
            dM[..., 1, 1:, 1:] = -to_S(dx)
            dM[..., 2, 1:, 1:] = -to_S(dy)
            kin, pot, x = terms(M, dM, E, frozen=True, X=X)
            tot = tot + (h * h / 4) * (-kin + pot - x).sum()
    return tot


def wedge_texture(x, y, beta, k, p=8):
    phi = torch.atan2(y, x) % (2 * math.pi)
    th = 2 * math.pi * k * (phi / (2 * math.pi)) ** p
    c, s = torch.cos(2 * th), torch.sin(2 * th)
    S = torch.zeros(*x.shape, 3, 3, dtype=x.dtype, device=x.device)
    S[..., 0, 0] = 0.5 * beta + 0.5 * beta * c
    S[..., 1, 1] = 0.5 * beta - 0.5 * beta * c
    S[..., 0, 1] = S[..., 1, 0] = 0.5 * beta * s
    S[..., 2, 2] = 1.0
    return from_S(S)


def relax(seed, k, h, R, kappa=0.0, iters=4000, rng_seed=0, save=None, block=None):
    beta = BETA
    n = int(round(2 * R / h)) + 1
    c = torch.linspace(-R, R, n, device=DEV)
    x, y = torch.meshgrid(c, c, indexing='ij')
    x = x + 1e-7
    E = (100.0, 1.0, beta, 0.0)
    rh = r_half_bogomolny(beta, 0.5)
    ubnd = texture(x, y, beta, k, core=rh)                     # the boundary is the same smooth texture for all seeds
    if seed == 'smooth':
        u0 = ubnd.clone()
    elif seed == 'unmelted':
        u0 = texture(x, y, beta, k, core=1e-9)
    elif seed == 'wedge':
        u0 = wedge_texture(x, y, beta, k)
    elif seed == 'kicked':
        # one random draw on the h = rho_1/2 / 8 grid, interpolated bilinearly to the grid in use, so that the
        # same perturbation is compared across spacings
        g = torch.Generator(device='cpu').manual_seed(rng_seed)
        n8 = int(round(2 * R / (rh / 8))) + 1
        noise = torch.randn(1, 6, n8, n8, generator=g, dtype=torch.float64)
        noise = torch.nn.functional.interpolate(noise, size=(n, n), mode='bilinear', align_corners=True)
        u0 = ubnd + (beta / 3) * noise[0].permute(1, 2, 0).to(DEV)
    elif seed.startswith('file:'):                     # continue from a saved field on the same grid
        import numpy as np
        u0 = torch.tensor(np.load(seed[5:])['u'], device=DEV)
    bnd = torch.zeros(n, n, dtype=torch.bool, device=DEV)
    bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
    free = (~bnd)[..., None].to(u0)
    u0 = free * u0 + (1 - free) * ubnd
    p = u0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1, max_iter=iters, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=50, line_search_fn='strong_wolfe')

    def full():
        return free * p + (1 - free) * ubnd

    def closure():
        opt.zero_grad()
        e = energy(full(), h, E, kappa)
        e.backward()
        return e
    t0 = time.time()
    T_start = float(energy(u0, h, E, kappa))
    trace = []
    if block:                                   # L-BFGS in blocks, the energy recorded after each
        opt = torch.optim.LBFGS([p], lr=1, max_iter=block, tolerance_grad=1e-14, tolerance_change=1e-16,
                                history_size=50, line_search_fn='strong_wolfe')
        for it in range(iters // block):
            opt.step(closure)
            trace.append(float(energy(full().detach(), h, E, kappa)))
            print(f'   [{seed} k={k} h={h:.4f} kappa={kappa}] iter {(it + 1) * block}: T/T_B = {trace[-1] / bound(beta, k):.4f}', flush=True)
    else:
        opt.step(closure)
    u = full().detach()
    T = float(energy(u, h, E, kappa))
    S = to_S(u)
    ev = torch.linalg.eigvalsh(S[..., :2, :2])
    split = ev[..., 1] - ev[..., 0]
    uu = u.clone().requires_grad_(True)
    gr = torch.autograd.grad(energy(uu, h, E, kappa), uu)[0] * free
    rec = dict(seed=seed, k=k, h=h, R=R, kappa=kappa, n=n, T=T, T_start=T_start, T_bound=bound(beta, k),
               T_over_bound=T / bound(beta, k), T_start_over_bound=T_start / bound(beta, k),
               split_min=float(split.min()), r_half=float(torch.sqrt((split < 0.5 * beta).sum() * h * h / math.pi)),
               grad_inf=float(gr.abs().max()), wall_s=time.time() - t0, trace=trace, block=block)
    if save:
        import numpy as np
        np.savez_compressed(save, u=u.cpu().numpy(), h=h, R=R, beta=beta, k=k)
        rec['field'] = save
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'seeds'
    rh = r_half_bogomolny(BETA, 0.5)
    out = f'results/strand_walls_{part}.json'
    try:
        rows = json.load(open(out))
    except (FileNotFoundError, json.JSONDecodeError):
        rows = []
    done = {(r['seed'], r['k'], round(r['h'], 12), round(r['R'], 12), r['kappa'], r.get('iters', 4000)) for r in rows}

    def run(*a, **kw):
        key = (a[0], a[1], round(a[2], 12), round(a[3], 12), kw.get('kappa', 0.0), kw.get('iters', 4000))
        if key in done:
            return
        rows.append(relax(*a, **kw))
        rows[-1]['iters'] = kw.get('iters', 4000)
        json.dump(rows, open(out, 'w'), indent=1)

    if part == 'seeds':
        for seed in ('smooth', 'unmelted', 'wedge', 'kicked'):          # all seeds at h = rho_1/2 / 8
            for k in (0.5, 1.0):
                run(seed, k, rh / 8, 8 * rh)
        for seed in ('unmelted', 'wedge'):                              # the two unmelted seeds at h / 2, k = 1/2
            run(seed, 0.5, rh / 16, 8 * rh)
    if part == 'deep':
        # the kicked seed went below the bound at h / 8: relax it longer, at three spacings, from the same random
        # draw (made on the h / 8 grid and interpolated), and keep the fields
        for div in (8, 16, 32):
            run('kicked', 0.5, rh / div, 8 * rh, iters=20000, save=f'results/fields/kicked_h{div}.npz')
    if part == 'kick':
        # the descent below the bound: its course (energy every 1000 iterations), its spacing dependence and
        # whether the two-derivative term stops it
        run('kicked', 0.5, rh / 8, 8 * rh, iters=8000, block=1000, save='results/fields/kicked_h8_trace.npz')
        run('kicked', 0.5, rh / 8, 8 * rh, kappa=1e-2, iters=8000, block=1000, save='results/fields/kicked_h8_kappa.npz')
        run('kicked', 0.5, rh / 16, 8 * rh, iters=8000, block=1000, save='results/fields/kicked_h16_trace.npz')
    if part == 'continue16':
        # the h / 16 kicked run, continued from its saved field (is the slower descent only slower?)
        run('file:results/fields/kicked_h16_trace.npz', 0.5, rh / 16, 8 * rh, iters=16000, block=2000,
            save='results/fields/kicked_h16_cont.npz')
    if part == 'kick16kappa':
        # the kicked seed with kappa at the finer spacing (the h / 8 run converged to 0.845 T_B)
        run('kicked', 0.5, rh / 16, 8 * rh, kappa=1e-2, iters=8000, block=1000, save='results/fields/kicked_h16_kappa.npz')
    if part == 'split':
        # the full strand with kappa: does it split into two half strands that repel? (field kept)
        run('smooth', 1.0, rh / 4, 16 * rh, kappa=1e-2, iters=4001, save='results/fields/k1_kappa_R16.npz')
    if part == 'kappa':                                                 # far-field winding cost, h = rho_1/2 / 4
        for k in (0.5, 1.0):
            for Rm in (4, 8, 16):
                for kappa in (0.0, 1e-3, 1e-2):
                    run('smooth', k, rh / 4, Rm * rh, kappa=kappa)
