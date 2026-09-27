"""The Bogomolny strand is a saddle of the full frozen 2D problem (review round 1 of the walls appendix).

Field: the exact k = 1/2 Bogomolny strand S_0 = A (+) 1, A = b0 I + b(r)(cos phi sigma_z + sin phi sigma_x) (b0 = beta/2,
8 k b b'/r = sqrt2 (b0 - b)). Perturbation: only the symmetric xz and yz entries, which tilt the charge direction,
  H_xz = f(r) cos 2phi,  H_yz = -f(r) sin 2phi,  f = (r/w)^2 exp(-r^2 / 2w^2) (1 - r^2/L^2)^3 for r < L (w = 0.65, L = 3),
regular at the line and zero before the box boundary. Two routes to the second variation Q in T(S_0 + eps H) = T_B + eps^2 Q:
  (1) the lattice energy of strand_walls.py, (T(eps) - T(0)) / eps^2 at eps = 1e-3, at nominal h = rho_1/2 / 8 and / 16
      (the grid spans [-3.3, 3.3], so its coordinate pitch differs from the nominal h by < 0.3%; both are recorded);
  (2) the radial integral of the expanded density (m = -2),
      Q = 2 pi int r dr { 4/r^2 [(b f')^2 + m^2 (b' f)^2 + 6 m b b' f f'] + [2 b (b0 - b) / ((1 - b0)^2 - b^2)] f^2 }.
Q < 0: arbitrarily small regular perturbations lower the tension. The smooth and unmelted seeds of the appendix have
S_xz = S_yz = 0, a symmetric subspace on which the gradient in those components vanishes, so they cannot leave it.
Output: results/strand_saddle.json.
"""
import json
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp, quad
from strand2d import from_S, DEV
from strand_walls import energy
from strand_bogomolny import r_half as r_half_bogomolny, bound

torch.set_default_dtype(torch.float64)
BETA, K, W, L = 0.3, 0.5, 0.65, 3.0
B0 = BETA / 2


def profile():
    """b(r) of the Bogomolny strand: 8 k b b' / r = sqrt2 (b0 - b), i.e. d(b^2)/ds = sqrt2 (b0 - b) / (4k), s = r^2/2."""
    sol = solve_ivp(lambda s, q: [math.sqrt(2) * (B0 - math.sqrt(max(q[0], 0))) / (4 * K)], [0, 60], [0.0],
                    max_step=2e-4, dense_output=True, rtol=1e-10, atol=1e-13)
    ss = np.linspace(0, 60, 600001)
    qq = sol.sol(ss)[0]
    return lambda r: np.sqrt(np.maximum(np.interp(np.asarray(r, dtype=float) ** 2 / 2, ss, qq), 0))


def f_of(r):
    r = np.asarray(r, dtype=float)
    return np.where(r < L, (r / W) ** 2 * np.exp(-r ** 2 / (2 * W ** 2)) * (1 - r ** 2 / L ** 2) ** 3, 0.0)


def fields(h, R, eps, bfun):
    n = int(round(2 * R / h)) + 1
    c = np.linspace(-R, R, n)
    x, y = np.meshgrid(c, c, indexing='ij')
    x = x + 1e-7
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    b = bfun(r)
    S = np.zeros((n, n, 3, 3))
    S[..., 0, 0] = B0 + b * np.cos(phi)            # cos 2 theta with theta = phi / 2
    S[..., 1, 1] = B0 - b * np.cos(phi)
    S[..., 0, 1] = S[..., 1, 0] = b * np.sin(phi)
    S[..., 2, 2] = 1.0
    f = f_of(r)
    S[..., 0, 2] = S[..., 2, 0] = eps * f * np.cos(2 * phi)
    S[..., 1, 2] = S[..., 2, 1] = -eps * f * np.sin(2 * phi)
    return from_S(torch.tensor(S, device=DEV))


def radial_Q(bfun):
    m = -2.0
    def b(r): return float(bfun(r))
    def db(r, d=1e-6): return (b(r + d) - b(max(r - d, 0))) / (r + d - max(r - d, 0))
    def f(r): return float(f_of(r))
    def df(r, d=1e-6): return (f(r + d) - f(max(r - d, 0))) / (r + d - max(r - d, 0))
    def integrand(r):
        quart = 4 / r ** 2 * ((b(r) * df(r)) ** 2 + m ** 2 * (db(r) * f(r)) ** 2 + 6 * m * b(r) * db(r) * f(r) * df(r))
        pot = 2 * b(r) * (B0 - b(r)) / ((1 - B0) ** 2 - b(r) ** 2) * f(r) ** 2
        return 2 * math.pi * r * (quart + pot)
    val, err = quad(integrand, 1e-6, L, limit=400)
    return val


if __name__ == '__main__':
    bfun = profile()
    rh = r_half_bogomolny(BETA, 0.5)
    E = (100.0, 1.0, BETA, 0.0)
    out = {'Q_radial': radial_Q(bfun), 'lattice': []}
    eps = 1e-3
    for div in (8, 16):
        h, R = rh / div, 3.3                          # box half-size 3.3 > L: the perturbation vanishes before the boundary
        T0 = float(energy(fields(h, R, 0.0, bfun), h, E))
        Te = float(energy(fields(h, R, eps, bfun), h, E))
        n = int(round(2 * R / h)) + 1
        out['lattice'].append(dict(h_nominal=h, h_coordinate=2 * R / (n - 1), T0=T0, T0_over_bound=T0 / bound(BETA, K), Q=(Te - T0) / eps ** 2))
        print(json.dumps(out['lattice'][-1]), flush=True)
    print('Q radial', out['Q_radial'])
    json.dump(out, open('results/strand_saddle.json', 'w'), indent=1)
    assert out['Q_radial'] < 0 and all(r['Q'] < 0 for r in out['lattice'])
