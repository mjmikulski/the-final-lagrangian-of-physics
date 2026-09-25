"""The straight strand of a biaxial vacuum: tension per unit length of a disclination of the transverse pair.

Vacuum spectrum (E_0, E_1, E_2, E_3) = (100, 1, beta, 0): the charge direction e_1 along the line (z), the
transverse pair (e_2, e_3) in the plane. Around the line the transverse pair turns by the angle k phi
(k = 1/2: half-disclination, k = 1: full), which a symmetric matrix allows for half-integer k. The field is
independent of z, so only F_xy enters, and in the frozen sector (M_00 = E_0, M_0i = 0) the energy per unit
length is T = integral over the plane of [2 |F_xy|^2 + V].

Far from the line the texture is a pure rotation of the 2x2 block with constant eigenvalues, so d_x N and
d_y N are both proportional to the same commutator [G, N] and F_xy = 0 exactly: the far field costs nothing,
and T is finite and set by the core, where the transverse pair melts (e_2 = e_3).

Lattice: square grid with spacing h in the box [-R, R]^2, the bilinear interpolant sampled at the 2 x 2 Gauss
points of every cell (no chequerboard null mode), Dirichlet boundary at the ideal texture, L-BFGS on the
interior nodes. Output: results/strand2d.json.

python strand2d.py                      # the beta scan for k = 1/2 and 1, and the resolution / box checks
"""
import json
import math
import sys
import time

import torch
from lagrangian import terms
from strand_bogomolny import r_half as r_half_bogomolny, bound

torch.set_default_dtype(torch.float64)
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
G = [0.5 - 0.5 / 3 ** 0.5, 0.5 + 0.5 / 3 ** 0.5]
IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def to_S(u):
    S = torch.zeros(*u.shape[:-1], 3, 3, dtype=u.dtype, device=u.device)
    for c, (a, b) in enumerate(IDX):
        S[..., a, b] = u[..., c]
        S[..., b, a] = u[..., c]
    return S


def from_S(S):
    return torch.stack([S[..., a, b] for a, b in IDX], -1)


def texture(x, y, beta, k, core=1.0):
    """Spatial block n (the operator's spatial part is -M_ij = n): e_1 = z with eigenvalue 1, the transverse
    pair rotated by k phi; the splitting b(rho) = beta (1 - exp(-rho^2/core^2)) melts it at the line."""
    rho = torch.sqrt(x * x + y * y)
    phi = torch.atan2(y, x)
    th = k * phi
    b = beta * (1 - torch.exp(-(rho / core) ** 2))
    s0 = 0.5 * beta
    c, s = torch.cos(2 * th), torch.sin(2 * th)
    S = torch.zeros(*x.shape, 3, 3, dtype=x.dtype, device=x.device)
    S[..., 0, 0] = s0 + 0.5 * b * c
    S[..., 1, 1] = s0 - 0.5 * b * c
    S[..., 0, 1] = S[..., 1, 0] = 0.5 * b * s
    S[..., 2, 2] = 1.0
    return from_S(S)


def energy(u, h, E):
    """T = sum over cells and Gauss points of (h^2/4) [2|F_xy|^2 + V] in the frozen sector."""
    tot = 0.0
    u00, u10, u01, u11 = u[:-1, :-1], u[1:, :-1], u[:-1, 1:], u[1:, 1:]
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
            kin, pot, _ = terms(M, dM, E, frozen=True)
            tot = tot + (h * h / 4) * (-kin + pot).sum()
    return tot


def relax(beta, k, R=None, h=None, iters=3000):
    """Grid scaled with the Bogomolny core radius rho_1/2 (proportional to sqrt(beta)): h = rho_1/2 / 8 and
    R = 8 rho_1/2 by default (the core decays like a Gaussian), so every beta is resolved alike."""
    rh = r_half_bogomolny(beta, 0.5)
    h = h or rh / 8
    R = R or 8 * rh
    n = int(round(2 * R / h)) + 1
    c = torch.linspace(-R, R, n, device=DEV)
    x, y = torch.meshgrid(c, c, indexing='ij')
    x = x + 1e-7          # keep the line off the node at the origin (the texture is singular there)
    E = (100.0, 1.0, beta, 0.0)
    u0 = texture(x, y, beta, k, core=rh)
    bnd = torch.zeros(n, n, dtype=torch.bool, device=DEV)
    bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
    free = (~bnd)[..., None].to(u0)
    p = u0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1, max_iter=iters, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=50, line_search_fn='strong_wolfe')

    def full():
        return free * p + (1 - free) * u0

    def closure():
        opt.zero_grad()
        e = energy(full(), h, E)
        e.backward()
        return e
    t0 = time.time()
    e_start = float(energy(u0, h, E))
    opt.step(closure)
    u = full().detach()
    uu = u.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(uu, h, E), uu)[0] * free
    T = float(energy(u, h, E))
    # core: the transverse splitting along the x axis, and the radius where it reaches half its vacuum value
    S = to_S(u)
    ev = torch.linalg.eigvalsh(S[..., :2, :2])
    split = (ev[..., 1] - ev[..., 0])
    # half-splitting radius from the area where the splitting is below half its vacuum value: independent of
    # where the core sits (the line is free to shift off the central node)
    r_half = float(torch.sqrt((split < 0.5 * beta).sum() * h * h / math.pi))
    ic = int(torch.argmin(split))
    cx, cy = float(x.flatten()[ic]), float(y.flatten()[ic])
    # radial profile of the splitting about the core, in bins of h
    rc = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).flatten()
    nb = int(R / h * 0.8)
    bins = (rc / h).long().clamp(max=nb)
    cnt = torch.zeros(nb + 1).index_add_(0, bins.cpu(), torch.ones_like(rc).cpu())
    sm = torch.zeros(nb + 1).index_add_(0, bins.cpu(), split.flatten().cpu())
    keep = cnt[:nb] > 0
    prof_r = ((torch.arange(nb) + 0.5) * h)[keep].tolist()
    prof_s = (sm[:nb] / cnt[:nb].clamp_min(1))[keep].tolist()
    # escape: how far the top eigenvector (the charge direction) leaves z at the line
    w, V = torch.linalg.eigh(S[n // 2, n // 2])
    rec = dict(beta=beta, k=k, R=R, h=h, T=T, T_bound=bound(beta, k), T_over_bound=T / bound(beta, k),
               T_start=e_start, grad_inf=float(g.abs().max()), split_min=float(split.min()), core_position=[cx, cy],
               r_half_split=r_half, r_half_bogomolny=r_half_bogomolny(beta, k),
               e1_z_at_line=float(V[2, -1].abs()), wall_s=time.time() - t0,
               profile_r=prof_r, profile_split=prof_s)
    print(json.dumps({k_: v for k_, v in rec.items() if not k_.startswith('profile')}), flush=True)
    return rec


if __name__ == '__main__':
    rows = []
    for k in (0.5, 1.0):
        for beta in (0.3, 0.1, 0.03, 0.01):
            rows.append(relax(beta, k))
            json.dump(rows, open('results/strand2d.json', 'w'), indent=1)
    rh = r_half_bogomolny(0.1, 0.5)
    for R, h in ((12 * rh, rh / 8), (8 * rh, rh / 12)):   # box and resolution checks at beta = 0.1
        for k in (0.5, 1.0):
            rows.append(relax(0.1, k, R=R, h=h))
            json.dump(rows, open('results/strand2d.json', 'w'), indent=1)
