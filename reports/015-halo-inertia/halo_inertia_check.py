"""Second route for the inertia of a tilted halo: the model's own kinetic density on the analytic exterior.

For n = R_x(theta(r)) x-hat with the eigenvalues at their vacuum values, the tensor field and its spatial
derivatives are analytic; the rigid rotation about z of the whole configuration has the tangent
dM/dphi = G M + M G^T + (y d_x - x d_y) M with G the rotation generator (Grid.rotation of soliton.py). The
kinetic density 2 sum_i |[A_0, A_i]|^2 from lagrangian.terms (A_0 = Omega dM/dphi) is averaged over the sphere at
fixed r by Gauss-Legendre quadrature and compared with the closed form of halo_check.py,
    I(r) = (64 pi/3) Delta^4 4 sin^2(theta/2) (1 + r^2 theta'^2 / 2),
and with its small-angle limit (64 pi/3) Delta^4 theta^2. Profiles: the smoothstep of halo_scaling.py and a
compact sin^2 bump placed at R = 3 and at R = 100 (the placement test of review round 1).
Writes results/halo_inertia_check.json.
"""
import json
import math
import numpy as np
import torch
from lagrangian import terms

torch.set_default_dtype(torch.float64)
E = (100.0, 1.0, 0.01, 0.01)
D = E[1] - E[2]
G = np.zeros((3, 3)); G[0, 1], G[1, 0] = -1.0, 1.0            # rotation generator about z on vectors


def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def dRx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]])


def field(x, theta, dtheta):
    """M, d_i M and the rotation tangent dM/dphi at the point x for n = R_x(theta(r)) x/r."""
    r = np.linalg.norm(x); xh = x / r
    R, dR = Rx(theta(r)), dRx(theta(r)) * dtheta(r)
    n = R @ xh
    dxh = (np.eye(3) - np.outer(xh, xh)) / r                    # dxh[i, j] = d_j xh_i
    dn = R @ dxh + np.outer(dR @ xh, xh)                        # dn[i, j] = d_j n_i  (theta depends on r only)
    def M_of(nv):
        M = np.zeros((4, 4)); M[0, 0] = E[0]; M[1:, 1:] = -(E[2] * np.eye(3) + D * np.outer(nv, nv)); return M
    M = M_of(n)
    dM = np.zeros((4, 4, 4))
    for j in range(3):
        dM[1 + j, 1:, 1:] = -D * (np.outer(dn[:, j], n) + np.outer(n, dn[:, j]))
    # rigid rotation about z: n_phi(x) = R_z(phi) n(R_z(-phi) x); d/dphi at 0 = G n - (G x) . grad n
    Gx = G @ x
    ndot = G @ n - dn @ Gx
    Mdot = np.zeros((4, 4)); Mdot[1:, 1:] = -D * (np.outer(ndot, n) + np.outer(n, ndot))
    return M, dM, Mdot


def sphere_average(theta, dtheta, r, nq=24):
    """Sphere average at radius r of the kinetic density per Omega^2 and of the static density."""
    xg, wg = np.polynomial.legendre.leggauss(nq)
    kin_tot, stat_tot, wsum = 0.0, 0.0, 0.0
    for ct, w in zip(xg, wg):
        st = math.sqrt(1 - ct * ct)
        for k in range(2 * nq):
            ph = 2 * math.pi * (k + 0.5) / (2 * nq)
            x = r * np.array([st * math.cos(ph), st * math.sin(ph), ct])
            M, dM, Mdot = field(x, theta, dtheta)
            dM4 = torch.tensor(dM); dM4[0] = torch.tensor(Mdot)          # unit Omega
            kin, pot, _ = terms(torch.tensor(M)[None], dM4[None], E, frozen=False)
            dM0 = torch.tensor(dM)
            kin0, _, _ = terms(torch.tensor(M)[None], dM0[None], E, frozen=False)
            kin_tot += w * float(kin - kin0)      # the Omega^2 part of the Lagrangian density: +2 sum |[A_0, A_i]|^2
            stat_tot += w * float(-kin0)
            wsum += w
    return kin_tot / wsum, stat_tot / wsum


def closed(theta, dtheta, r):
    t, dt = theta(r), dtheta(r)
    exact = 64 * math.pi / 3 * D ** 4 * 4 * math.sin(t / 2) ** 2 * (1 + r * r * dt * dt / 2)
    lead = 64 * math.pi / 3 * D ** 4 * t * t
    cost = 32 * math.pi / 3 * D ** 4 * dt * dt
    return exact, lead, cost


def smoothstep(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)


out = []
profiles = {
    'smoothstep alpha=0.2, rising over [3,4], falling over [7.6,8.6] (box 18 scan, L = 1)':
        (lambda r: 0.2 * smoothstep((r - 3) / 1) * (1 - smoothstep((r - 7.6) / 1)),
         lambda r: 0.2 * (6 * ((r - 3) / 1) * (1 - (r - 3) / 1) / 1 if 3 < r < 4 else 0.0) * (1 - smoothstep((r - 7.6) / 1))
                   - 0.2 * smoothstep((r - 3) / 1) * (6 * ((r - 7.6) / 1) * (1 - (r - 7.6) / 1) / 1 if 7.6 < r < 8.6 else 0.0)),
    'sin^2 bump 0.1 sin^2(pi (r-3)/3) on [3,6]':
        (lambda r: 0.1 * math.sin(math.pi * (r - 3) / 3) ** 2 if 3 < r < 6 else 0.0,
         lambda r: 0.1 * 2 * math.sin(math.pi * (r - 3) / 3) * math.cos(math.pi * (r - 3) / 3) * math.pi / 3 if 3 < r < 6 else 0.0),
    'sin^2 bump 0.1 sin^2(pi (r-100)/3) on [100,103]':
        (lambda r: 0.1 * math.sin(math.pi * (r - 100) / 3) ** 2 if 100 < r < 103 else 0.0,
         lambda r: 0.1 * 2 * math.sin(math.pi * (r - 100) / 3) * math.cos(math.pi * (r - 100) / 3) * math.pi / 3 if 100 < r < 103 else 0.0),
}
worst = 0.0
for name, (theta, dtheta) in profiles.items():
    lo = 3.0 if 'bump' not in name or '(r-3)' in name else 100.0
    hi = lo + (5.6 if 'smoothstep' in name else 3.0)
    rs = np.linspace(lo + 0.05, hi - 0.05, 9)
    rows = []
    for r in rs:
        kin_avg, stat_avg = sphere_average(theta, dtheta, r)
        exact, lead, cost = closed(theta, dtheta, r)
        # per unit radius: 4 pi r^2 x density; kinetic part is (1/2) I(r) Omega^2 -> I(r) = 2 x 4 pi r^2 kin_avg
        I_model = 2 * 4 * math.pi * r * r * kin_avg
        E_model = 4 * math.pi * r * r * stat_avg - 16 * math.pi * D ** 4 / r ** 2      # minus the Coulomb term of the untilted hedgehog
        rows.append(dict(r=float(r), I_model=I_model, I_exact=exact, I_leading=lead, cost_model=E_model, cost_formula=cost))
        worst = max(worst, abs(I_model - exact) / max(exact, 1e-12), abs(E_model - cost) / max(cost, 1e-12) if cost > 1e-9 else 0.0)
    ratio = sum(x['I_exact'] for x in rows) / max(sum(x['I_leading'] for x in rows), 1e-300)
    out.append(dict(profile=name, rows=rows, exact_over_leading=ratio))
    print(f'{name}\n   model vs closed form, worst relative deviation so far {worst:.1e}; exact/leading inertia over this profile: {ratio:.2f}')
json.dump(dict(profiles=out, worst_relative_deviation=worst), open('results/halo_inertia_check.json', 'w'), indent=1)
assert worst < 1e-6
print('HALO_INERTIA OK')
