"""Which covariant terms see the screening? Candidate invariants on the null-direction hedgehog.

The asymptotic hedgehog is M = M_vac - Delta s s^T with s = (0, n) space-like; the screened electron
replaces it by the Kerr-Schild form with the null l = (1, n). Derivatives are analytic
(d_i n_j = (delta_ij - n_i n_j)/r), so a zero here is exact, not a lattice cancellation.

For every candidate the table reports the value on the frozen hedgehog and on the screened one, both in
units of the model's own Coulomb density 4 Delta^4 / r^4 at the same point.

python null_probe.py [--E0 100] [--E2 0.01] [--r 4]
"""
import argparse
import json
import numpy as np
import torch
from lagrangian import terms, operator, eta as eta_of, delta_M, eigenvalues

torch.set_default_dtype(torch.float64)
ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def field(x, E, m):
    """M and its spatial derivatives for the asymptotic hedgehog with a tilt M_0i = m n_i (m = 0: frozen)."""
    r = np.linalg.norm(x)
    n = x / r
    dn = (np.eye(3) - np.outer(n, n)) / r                      # dn[i, j] = d_i n_j
    E0, E1, E2 = E[0], E[1], E[2]
    D = E1 - E2
    M = np.zeros((4, 4))
    M[0, 0] = E0
    M[1:, 1:] = -(E2 * np.eye(3) + D * np.outer(n, n))
    M[0, 1:] = m * n
    M[1:, 0] = m * n
    dM = np.zeros((3, 4, 4))
    for i in range(3):
        dnn = np.outer(dn[i], n) + np.outer(n, dn[i])
        dM[i, 1:, 1:] = -D * dnn
        dM[i, 0, 1:] = m * dn[i]
        dM[i, 1:, 0] = m * dn[i]
    return M, dM


def invariants(M, dM, E):
    """Candidate scalars, all built covariantly from N = eta M and its derivatives."""
    N = ETA @ M
    dN = np.einsum('ab,ibc->iac', ETA, dM)
    Mt = torch.tensor(M)
    dM4 = torch.zeros(4, 4, 4, dtype=torch.float64)
    dM4[1:] = torch.tensor(dM)                                  # static: no time derivative
    kin, pot, _ = terms(Mt[None], dM4[None], E, frozen=False)
    Nt = operator(Mt[None])
    e = eigenvalues(Nt, c=(E[0] * E[1]) ** 0.5)
    dm = delta_M(Nt, e)[0].numpy()                              # the field's own Euclidean metric
    dmi = ETA @ dm @ ETA
    out = {}
    out['quartic F.F (the model)'] = float(-kin)
    out['quadratic, eta norm  tr(d_iN d_iN)'] = sum(np.trace(dN[i] @ dN[i]) for i in range(3))
    out['quadratic, Euclidean tr(d_iN dM d_iN^T dM^-1)'] = sum(np.trace(dN[i] @ dm @ dN[i].T @ dmi) for i in range(3))
    out['tr(N d_iN d_iN)'] = sum(np.trace(N @ dN[i] @ dN[i]) for i in range(3))
    out['tr(N d_iN N d_iN)'] = sum(np.trace(N @ dN[i] @ N @ dN[i]) for i in range(3))
    out['tr(N N d_iN d_iN)'] = sum(np.trace(N @ N @ dN[i] @ dN[i]) for i in range(3))
    out['tr(d_iN d_jN) tr(d_iN d_jN)'] = sum(np.trace(dN[i] @ dN[j]) * np.trace(dN[i] @ dN[j]) for i in range(3) for j in range(3))
    out['(tr d_iN)^2 (trace of the jet)'] = sum(np.trace(dN[i]) ** 2 for i in range(3))
    # eigenvalue gradients (K_lambda of the thread): first-order perturbation theory, d e_a = v_a^T eta dN v_a
    w, V = np.linalg.eig(N)
    idx = np.argsort(-w.real)
    w, V = w.real[idx], V.real[:, idx]
    ke = 0.0
    for a in range(4):
        v = V[:, a]; nrm = v @ ETA @ v
        ke += sum((v @ ETA @ dN[i] @ v / nrm) ** 2 for i in range(3))
    out['eigenvalue gradients sum (d_i e_a)^2'] = ke
    # aether-type: gradients of the time-like eigenvector u (frame-sensitive, not a function of the spectrum)
    u = V[:, 0] / np.sqrt(abs(V[:, 0] @ ETA @ V[:, 0]))
    du = np.zeros((3, 4))
    for i in range(3):
        for b in range(1, 4):
            v = V[:, b]; nrm = v @ ETA @ v
            du[i] += v * (v @ ETA @ dN[i] @ u) / (nrm * (w[0] - w[b]))
    out['aether (d_i u)^2, Euclidean'] = float(sum(du[i] @ du[i] for i in range(3)))
    out['aether divergence (d_i u^i)^2'] = float(sum(du[i][i + 1] for i in range(3)) ** 2)
    # chiral twist pseudo-scalar of the thread, tau = eps_ijk M_il d_j M_kl
    eps = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, k] = 1; eps[i, k, j] = -1
    tau = sum(eps[i, j, k] * M[1 + i, 1 + l] * dM[j, 1 + k, 1 + l] for i in range(3) for j in range(3) for k in range(3) for l in range(3))
    out['chiral twist tau^2'] = tau ** 2
    # the candidate cure: A_mu = Q d_mu N P_0, the part of the jet that tilts the field's own time axis
    P0 = np.outer(V[:, 0], V[:, 0] @ ETA) / (V[:, 0] @ ETA @ V[:, 0])
    Qp = np.eye(4) - P0
    A = [Qp @ dN[i] @ P0 for i in range(3)]
    out['time-axis tilt K_u = |Q dN P_0|^2'] = float(sum(np.trace(A[i] @ dm @ A[i].T @ dmi) for i in range(3)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--E0', type=float, default=100.0)
    ap.add_argument('--E2', type=float, default=0.01)
    ap.add_argument('--r', type=float, default=4.0)
    a = ap.parse_args()
    E = (a.E0, 1.0, a.E2, a.E2)
    D = E[1] - E[2]
    x = np.array([1.0, 2.0, 3.0]); x = x / np.linalg.norm(x) * a.r          # a generic point, no lattice symmetry
    coulomb = 4 * D ** 4 / a.r ** 4
    rows = {}
    for name, m in (('frozen', 0.0), ('screened', D)):
        M, dM = field(x, E, m)
        rows[name] = invariants(M, dM, E)
    print(f'E = {E}, r = {a.r}; values in units of the Coulomb density 4 Delta^4 / r^4 = {coulomb:.3e}\n')
    print(f'{"invariant":52s} {"frozen":>12s} {"screened":>12s}   sees the screening?')
    for k in rows['frozen']:
        f, s = rows['frozen'][k] / coulomb, rows['screened'][k] / coulomb
        sees = 'yes' if abs(s) > 1e-8 * max(1.0, abs(f)) else 'NO (blind)'
        print(f'{k:52s} {f:12.4e} {s:12.4e}   {sees}')
    # where exactly does the quartic vanish?
    print('\nthe cure: the tilt costs K_u and saves the quartic; the ratio decides.')
    for rr in (1.0, 2.0, 4.0, 10.0, 30.0):
        xx = np.array([1.0, 2.0, 3.0]); xx = xx / np.linalg.norm(xx) * rr
        q0 = invariants(*field(xx, E, 0.0), E)['quartic F.F (the model)']
        ku = invariants(*field(xx, E, D), E)['time-axis tilt K_u = |Q dN P_0|^2']
        print(f'  r = {rr:5.1f}: Coulomb density saved {q0:.4e}, K_u of the screened state {ku:.4e}, '
              f'break-even coupling c* = {q0 / ku:.2e}')
    ms = np.linspace(0.9 * D, 1.1 * D, 9)
    vals = [invariants(*field(x, E, m), E)['quartic F.F (the model)'] / coulomb for m in ms]
    print('\nquartic against the tilt amplitude m/Delta:')
    print('  ' + '  '.join(f'{m / D:.3f}:{v:.2e}' for m, v in zip(ms, vals)))
    breakeven = []
    for rr in (1.0, 2.0, 4.0, 10.0, 30.0):
        xx = np.array([1.0, 2.0, 3.0]); xx = xx / np.linalg.norm(xx) * rr
        q0 = invariants(*field(xx, E, 0.0), E)['quartic F.F (the model)']
        ku = invariants(*field(xx, E, D), E)['time-axis tilt K_u = |Q dN P_0|^2']
        breakeven.append(dict(r=rr, coulomb=q0, Ku=ku, c_star=q0 / ku, prediction=2 * D ** 2 / rr ** 2))
    json.dump(dict(E=E, r=a.r, coulomb_density=coulomb,
                   table={k: dict(frozen=rows['frozen'][k] / coulomb, screened=rows['screened'][k] / coulomb) for k in rows['frozen']},
                   quartic_vs_m={f'{m / D:.3f}': v for m, v in zip(ms, vals)}, breakeven=breakeven),
              open('results/null_probe.json', 'w'), indent=1)
