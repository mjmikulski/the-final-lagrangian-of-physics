"""K1: the vacuum under the weighted terms.

A. Frame twists in a periodic box (static, spectral derivatives): N(x) = R(x) N_vac R(x)^-1 with
   R = exp(t W(x)), W a smooth random so(1,3) field with several boost planes. On such textures the
   spectrum is constant, so L1 = sqrt(e3/e3vac) phi = phi and its box integral must vanish (phi is a
   total derivative): measured against the grid size n. L2 = adj.Phi / e3vac carries the rotated
   projector, so its integral need not vanish: I2(t) is fitted to c2 t^2 + c3 t^3 + c4 t^4 on
   t in +-{5e-4, 1e-3, 2e-3, 4e-3} (the protocol of report 014, result 5) and the odd part reported.
   The static F.F (quartic in t) gives the scale c4_FF.
B. The threshold kappa_c for L1 (pointwise, leading order): the worst static jet at the vacuum
   minimises rho = F.F / phi^2 over the three symmetric gradient matrices (bound rho >= 1/6); then, at
   fixed jet and amplitude lambda, the eigenvalues are shifted at fixed frame, which leaves F.F and phi
   unchanged and moves only V = sum (e_a - E_a)^2 and the weight:
       E(lambda, de) = sum_a de_a^2 - kappa (sqrt(e3(E+de)/e3vac) - 1) lambda^2 phi0 + lambda^4 Q0.
   kappa_c = sup{kappa : min over (lambda, de) of E >= 0}, for E3 = 0 and for E3 = E2 (lattice
   convention). The linear-response estimate kappa_lin = 2 sqrt(rho_min) / |g|, g_a = d sqrt(e3/e3vac)/d e_a,
   is reported alongside.
Writes results/k1_vacuum.json.
"""
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))
S = torch.tensor([1.0, -1.0, -1.0, -1.0])
out = {}


def e3_adj(N):
    """Batched: N[..., 4, 4] -> e3[...], adj[..., 4, 4] (Cayley-Hamilton)."""
    N2 = N @ N
    N3 = N2 @ N
    t1 = torch.diagonal(N, dim1=-2, dim2=-1).sum(-1)
    t2 = torch.diagonal(N2, dim1=-2, dim2=-1).sum(-1)
    t3 = torch.diagonal(N3, dim1=-2, dim2=-1).sum(-1)
    e1, e2 = t1, (t1 ** 2 - t2) / 2
    e3 = (t1 ** 3 - 3 * t1 * t2 + 2 * t3) / 6
    I = torch.eye(4, dtype=N.dtype).expand_as(N)
    adj = e3[..., None, None] * I - e2[..., None, None] * N + e1[..., None, None] * N2 - N3
    return e3, adj


def static_terms(dN):
    """dN[..., 3, 4, 4] = spatial derivatives of the mixed N (derivative index i = 1..3 stored at i-1).
    Returns (F.F static, phi, Phi[..., 4, 4]) with F_ij = [d_i N, d_j N] mixed, Frobenius norm (delta_M = 1
    in the vacuum frame). phi = sum_{i,j} eta^{ii} eta^{jj} F_{ij ij} pairs the derivative index with the SAME
    spatial matrix slot; Phi_{nu b} = sum_i (F_{i nu})^i_b."""
    F = torch.einsum('...iab,...jbc->...ijac', dN, dN) - torch.einsum('...jab,...ibc->...ijac', dN, dN)
    FF = 0.5 * torch.einsum('...ijab,...ijab->...', F, F)          # 2 sum_{i<j} |F_ij|^2 = 1/2 sum_{i,j}
    Fcov = torch.einsum('a,...ijab->...ijab', S, F)                 # lower the first matrix index with eta
    phi = torch.einsum('...ijij->...', Fcov[..., :, :, 1:, 1:])      # slots (alpha, beta) = (i, j), eta^{ii}eta^{jj} = +1
    Phi = torch.einsum('...ijib->...jb', F[..., :, :, 1:, :])        # Phi_{j b} = sum_i (F_ij)^i_b, rows j = 1..3
    Phi4 = torch.zeros(*F.shape[:-4], 4, 4, dtype=F.dtype)
    Phi4[..., 1:, :] = Phi
    return FF, phi, Phi4


def so13_field(n, rng, nmodes=2):
    """Smooth random so(1,3)-valued field on the periodic grid, all six generators, low Fourier modes."""
    x = torch.arange(n) * (2 * np.pi / n)
    X = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), -1)
    gens = []
    for (i, j) in ((0, 1), (0, 2), (0, 3)):
        G = torch.zeros(4, 4); G[i, j] = G[j, i] = 1.0; gens.append(G)
    for (i, j) in ((1, 2), (2, 3), (3, 1)):
        G = torch.zeros(4, 4); G[i, j], G[j, i] = -1.0, 1.0; gens.append(G)
    W = torch.zeros(n, n, n, 4, 4)
    for g in range(6):
        f = torch.zeros(n, n, n)
        for kx in range(-nmodes, nmodes + 1):
            for ky in range(-nmodes, nmodes + 1):
                for kz in range(-nmodes, nmodes + 1):
                    if kx == ky == kz == 0:
                        continue
                    c, ph = rng.standard_normal(), rng.uniform(0, 2 * np.pi)
                    f = f + c * torch.cos(kx * X[..., 0] + ky * X[..., 1] + kz * X[..., 2] + ph)
        W = W + f[..., None, None] * gens[g] / (2 * nmodes + 1) ** 1.5
    return W


def spectral_grad(N):
    """d_i N by FFT on the periodic box [0, 2pi)^3; N[n, n, n, 4, 4]."""
    n = N.shape[0]
    k = torch.fft.fftfreq(n, d=1.0 / n)
    Nk = torch.fft.fftn(N, dim=(0, 1, 2))
    out = []
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]; shape[ax] = n
        kk = k.reshape(shape)
        out.append(torch.fft.ifftn(1j * kk * Nk, dim=(0, 1, 2)).real)
    return torch.stack(out, -3)


def twist_integrals(n, t, W, N_vac):
    R = torch.linalg.matrix_exp(t * W)
    Ri = ETA @ R.transpose(-1, -2) @ ETA
    N = R @ N_vac @ Ri
    dN = spectral_grad(N)
    FF, phi, Phi4 = static_terms(dN)
    e3, adj = e3_adj(N)
    e3vac = float(e3_adj(N_vac)[0])
    L1 = torch.sqrt(e3.clamp_min(1e-300) / e3vac) * phi
    L2 = torch.einsum('...nb,b,...nb->...', adj, S, Phi4) / e3vac
    vol = (2 * np.pi) ** 3
    return {'I_phi': float(phi.mean() * vol), 'I_L1': float(L1.mean() * vol), 'I_L2': float(L2.mean() * vol),
            'I_FF': float(FF.mean() * vol), 'e3_spread': float((e3 - e3vac).abs().max() / e3vac),
            'phi_abs': float(phi.abs().mean() * vol)}


import sys
ONLY_B = '--only-B' in sys.argv
print('=== A. frame twists in a periodic box ===')
E_VAC = torch.tensor([100.0, 1.0, 0.01, 0.0])
N_VAC = torch.diag(E_VAC)
rng = np.random.default_rng(11)
A = {'grid_convergence': [], 'cubic_fit': []}
Wbig = {}
for seed in (range(3) if not ONLY_B else ()):
    rng_s = np.random.default_rng(100 + seed)
    for n in (16, 24, 32, 48):
        W = so13_field(n, np.random.default_rng(100 + seed))
        r = twist_integrals(n, 1.0, W, N_VAC)
        r['W_rms'] = float((W ** 2).sum((-1, -2)).mean().sqrt())
        A['grid_convergence'].append({'seed': seed, 'n': n, 't': 1.0, **r})
        print(f'  seed {seed} n {n:2d} t 1.0: int phi {r["I_phi"]:+.3e} (|phi| {r["phi_abs"]:.3e}), int L1 {r["I_L1"]:+.3e}, '
              f'int L2 {r["I_L2"]:+.3e}, int F.F {r["I_FF"]:.3e}, e3 spread {r["e3_spread"]:.1e}')
    # cubic fit at n = 32 on the 014 amplitudes (and n = 48 as the convergence check)
    for n in (32, 48):
        W = so13_field(n, np.random.default_rng(100 + seed))
        ts = [s * a for a in (5e-4, 1e-3, 2e-3, 4e-3) for s in (+1, -1)]
        rows = [(t, twist_integrals(n, t, W, N_VAC)) for t in ts]
        T = np.array([t for t, _ in rows])
        I2 = np.array([r['I_L2'] for _, r in rows])
        I1 = np.array([r['I_L1'] for _, r in rows])
        FFv = np.array([r['I_FF'] for _, r in rows])
        X = np.stack([T ** 2, T ** 3, T ** 4], 1)
        c2, c3, c4 = np.linalg.lstsq(X, I2, rcond=None)[0]
        resid = float(np.abs(I2 - X @ np.array([c2, c3, c4])).max() / max(np.abs(I2).max(), 1e-300))
        odd = {float(t): float((r['I_L2'] - dict(rows)[-t]['I_L2']) / 2) for t, r in rows if t > 0}
        c4ff = float(np.linalg.lstsq(T[:, None] ** 4, FFv, rcond=None)[0][0])
        A['cubic_fit'].append({'seed': seed, 'n': n, 'c2': float(c2), 'c3': float(c3), 'c4': float(c4), 'fit_resid': resid,
                               'odd_part': odd, 'c4_FF': c4ff, 'max_abs_I_L1': float(np.abs(I1).max())})
        print(f'  seed {seed} n {n}: L2 fit c2 {c2:+.3e} c3 {c3:+.3e} c4 {c4:+.3e} (resid {resid:.1e}); '
              f'F.F c4 {c4ff:.3e}; max|int L1| {np.abs(I1).max():.1e}')
if ONLY_B:
    A = json.load(open(os.path.join(HERE, 'results', 'appendix_volume_vacuum.json')))['A_twists']
out['A_twists'] = A

print('\n=== B. the threshold kappa_c for L1 ===')


def jet_terms(x):
    A = x.reshape(3, 4, 4)
    A = 0.5 * (A + A.transpose(-1, -2))                     # symmetric covariant d_i M
    dN = torch.einsum('ab,ibc->iac', ETA, A)
    FF, phi, _ = static_terms(dN)
    return FF, phi


def ratio(x):
    FF, phi = jet_terms(x)
    return FF / (phi ** 2 + 1e-300)


best = None
for start in range(24):
    x = torch.randn(48, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=400, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        r = ratio(x)
        r.backward()
        return r
    opt.step(closure)
    r = float(ratio(x.detach()))
    if best is None or r < best[0]:
        best = (r, x.detach().clone())
rho_min, xbest = best
FF0, phi0 = (float(v) for v in jet_terms(xbest))
scale = abs(phi0)
FF0, phi0 = FF0 / scale ** 2, phi0 / scale            # unit |phi| jet: rescale A by 1/sqrt(|phi|)
print(f'  rho_min = min F.F/phi^2 = {rho_min:.5f} (bound 1/6 = {1 / 6:.5f}); worst jet at unit |phi|: F.F = {FF0:.5f}')
B = {'rho_min': rho_min, 'bound': 1 / 6, 'FF_at_unit_phi': FF0}


def e3_of(ev):
    ev = np.asarray(ev)
    return sum(ev[i] * ev[j] * ev[k] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4))


for label, E in (('E3=0', (100.0, 1.0, 0.01, 0.0)), ('E3=E2 (lattice)', (100.0, 1.0, 0.01, 0.01))):
    E = np.array(E)
    e3vac = e3_of(E)
    g = np.array([(e3_of(E + 1e-6 * np.eye(4)[a]) - e3_of(E - 1e-6 * np.eye(4)[a])) / 2e-6 for a in range(4)]) / (2 * e3vac)
    kappa_lin = 2 * np.sqrt(rho_min) / np.linalg.norm(g)

    def worst_energy(kappa, lam):
        # minimise over de (4 params) with e3 > 0; phi0 sign chosen so that kappa*phi0 > 0 (destabilising)
        p = kappa * lam ** 2 * abs(phi0)

        def f(de):
            e3 = e3_of(E + de)
            if e3 <= 0:
                return 1e6
            return float((de ** 2).sum() - p * (np.sqrt(e3 / e3vac) - 1))
        best_v = None
        for s0 in ([0, 0, 0, 0], [0, 0, 0.05, 0.05], [0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 2, 2]):
            r = minimize(f, np.array(s0, float) * (1 if kappa > 0 else 1), method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 20000})
            if best_v is None or r.fun < best_v:
                best_v = r.fun
        return best_v + lam ** 4 * FF0

    lams = np.geomspace(1e-3, 30, 60)

    def stable(kappa):
        return all(worst_energy(kappa, lam) >= -1e-12 for lam in lams)

    # bracket the threshold from kappa = 1 in both directions, then bisect
    hi = 1.0
    if stable(hi):
        lo = hi
        while stable(hi) and hi < 1e4:
            lo, hi = hi, hi * 2
    else:
        while not stable(hi) and hi > 1e-6:
            hi = hi / 2
        lo = hi
        hi = 2 * hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    kappa_c = lo
    lam_star = min(lams, key=lambda l: worst_energy(hi, l))
    B[label] = {'E': E.tolist(), 'e3vac': float(e3vac), 'g': g.tolist(), 'g_norm': float(np.linalg.norm(g)),
                'kappa_lin': float(kappa_lin), 'kappa_c': float(kappa_c), 'lambda_at_threshold': float(lam_star)}
    print(f'  {label}: |g| = {np.linalg.norm(g):.2f}, kappa_lin = {kappa_lin:.4f}, kappa_c (exact de-minimisation) = {kappa_c:.4f}, '
          f'unstable amplitude lambda ~ {lam_star:.3g}')
out['B_threshold'] = B
json.dump(out, open(os.path.join(HERE, 'results', 'appendix_volume_vacuum.json'), 'w'), indent=1)
print('written results/appendix_volume_vacuum.json')
