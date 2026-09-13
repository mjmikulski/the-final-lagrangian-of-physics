"""K0: Euler-Lagrange test of the weighted terms (are they null Lagrangians or dynamical?).

E0 = 100 conventions: N = eta M, signature (+,-,-,-), F_{mu nu} = [d_mu N, d_nu N] (mixed matrices),
phi = sum_{mu nu} eta^{mu mu} eta^{nu nu} F_{mu nu mu nu} (all-covariant slots), Phi_{nu b} = sum_mu (F_{mu nu})^mu_b.
Terms: L1 = sqrt(e3/e3vac) phi, L2 = (adj(N) eta)^{nu b} Phi_{nu b} / e3vac, controls phi (null, 005/014)
and I1 = -F.F with the Frobenius norm (dynamical). e3 and adj are polynomials in N (no eigendecomposition).
EL_q = dL/dq - sum_mu d_mu(dL/da_mu) with the second jet as in report 014's null_test_linear.py:
d_mu g = jvp(g; (dq, da_nu) = (a_mu, D_{mu nu})). Random points near the vacuum spectrum (100, 1, 0.01, 0),
and points with a CONSTANT spectrum along the jet (pure frame jets: A_mu = [G_mu, N], D_{mu nu} =
[G_mu,[G_nu,N]] symmetrised) where L1 must be null exactly.
Writes results/k0_el_test.json.
"""
import json
import os

import numpy as np
import torch
from torch.func import jacrev, jvp

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))
S = torch.tensor([1.0, -1.0, -1.0, -1.0])
IU = torch.triu_indices(4, 4)
E_VAC = (100.0, 1.0, 0.01, 0.0)
E3_VAC = 1.0   # E0*E1*E2 with E3 = 0


def sym(q):
    M = torch.zeros(4, 4, dtype=q.dtype)
    M[IU[0], IU[1]] = q
    return M + M.T - torch.diag(torch.diag(M))


def vec(M):
    return M[IU[0], IU[1]]


def traces(N):
    N2 = N @ N
    N3 = N2 @ N
    t1, t2, t3 = torch.trace(N), torch.trace(N2), torch.trace(N3)
    e1 = t1
    e2 = (t1 ** 2 - t2) / 2
    e3 = (t1 ** 3 - 3 * t1 * t2 + 2 * t3) / 6
    adj = e3 * torch.eye(4) - e2 * N + e1 * N2 - N3
    return e3, adj


def F_of(A):
    return torch.einsum('mab,nbc->mnac', A, A) - torch.einsum('nab,mbc->mnac', A, A)


def L_of(kind, M, A):
    """M covariant symmetric, A[mu] = d_mu M covariant symmetric."""
    N = ETA @ M
    AN = torch.einsum('ab,mbc->mac', ETA, A)
    F = F_of(AN)                                     # F[mu,nu] mixed: (alpha up, beta down)
    Fcov = torch.einsum('a,mnab->mnab', S, F)
    phi = torch.einsum('m,n,mnmn->', S, S, Fcov)
    if kind == 'phi':
        return phi
    if kind == 'I1':
        w = torch.einsum('m,n->mn', S, S)
        return -torch.einsum('mn,mnab,mnab->', w, F, F)
    e3, adj = traces(N)
    if kind == 'L1':
        return torch.sqrt(e3 / E3_VAC) * phi
    if kind == 'L2':
        Phi = torch.einsum('mnmb->nb', F)
        return torch.einsum('nb,b,nb->', adj, S, Phi) / E3_VAC   # (adj eta)^{nu beta} Phi_{nu beta}
    raise ValueError(kind)


def EL(kind, q, a, D):
    def L_q(q_, a_):
        return L_of(kind, sym(q_), torch.stack([sym(a_[m]) for m in range(4)]))
    dLdq = jacrev(L_q, argnums=0)(q, a)

    def g(q_, a_):
        return jacrev(L_q, argnums=1)(q_, a_)
    total = torch.zeros(10)
    scale = float(dLdq.abs().max())
    for mu in range(4):
        _, dg = jvp(g, (q, a), (a[mu], D[mu]))
        total = total + dg[mu]
        scale += float(dg[mu].abs().max())
    return dLdq - total, scale + 1e-300


def so13(rng):
    b, w = rng.standard_normal(3) * 0.4, rng.standard_normal(3) * 0.4
    K = np.zeros((4, 4))
    K[0, 1:] = b; K[1:, 0] = b
    K[1, 2], K[2, 1] = -w[2], w[2]
    K[2, 3], K[3, 2] = -w[0], w[0]
    K[3, 1], K[1, 3] = -w[1], w[1]
    return K


def random_point(rng, pure_frame=False):
    from scipy.linalg import expm
    eta = np.diag([1.0, -1, -1, -1])
    Lam = expm(so13(rng))
    Lami = eta @ Lam.T @ eta
    ev = np.array(E_VAC) + (0.0 if pure_frame else 1.0) * rng.uniform(-0.1, 0.1, 4) * np.array([1, 1, 0.05, 0.05])
    N = Lam @ np.diag(ev) @ Lami
    M = eta @ N
    if pure_frame:
        Gs = [so13(rng) for _ in range(4)]
        AN = [G @ N - N @ G for G in Gs]
        DN = np.zeros((4, 4, 4, 4))
        for m in range(4):
            for n in range(4):
                X = Gs[m] @ AN[n] - AN[n] @ Gs[m]
                Y = Gs[n] @ AN[m] - AN[m] @ Gs[n]
                DN[m, n] = 0.5 * (X + Y)          # d_mu d_nu N along a bi-parameter frame path, symmetric
        A = [eta @ x for x in AN]
        D = np.einsum('ab,mnbc->mnac', eta, DN)
    else:
        A = [0.5 * (x + x.T) for x in rng.standard_normal((4, 4, 4))]
        D = np.zeros((4, 4, 4, 4))
        for m in range(4):
            for n in range(m, 4):
                s = rng.standard_normal((4, 4)); s = 0.5 * (s + s.T)
                D[m, n] = D[n, m] = s
    M = 0.5 * (M + M.T)
    q = vec(torch.tensor(M)).clone()
    a = torch.stack([vec(torch.tensor(0.5 * (x + x.T))) for x in A])
    Dq = torch.stack([torch.stack([vec(torch.tensor(0.5 * (D[m, n] + D[m, n].T))) for n in range(4)]) for m in range(4)])
    return q, a, Dq


def tangents(q):
    """Orthonormal bases (in the 10-vector metric) of the frame directions dM = eta [G, N] (6) and the
    eigenvalue directions dM = eta Lam P_a Lam^-1 (4) at the point M = sym(q)."""
    M = sym(q)
    N = ETA @ M
    lam, V = torch.linalg.eig(N)
    lam, V = lam.real, V.real
    Vi = torch.linalg.inv(V)
    eig_dirs = [vec(ETA @ (V[:, a:a + 1] @ Vi[a:a + 1, :])) for a in range(4)]
    gens = []
    for (i, j) in ((0, 1), (0, 2), (0, 3)):
        G = torch.zeros(4, 4); G[i, j] = G[j, i] = 1.0; gens.append(G)
    for (i, j) in ((1, 2), (2, 3), (3, 1)):
        G = torch.zeros(4, 4); G[i, j], G[j, i] = -1.0, 1.0; gens.append(G)
    frame_dirs = [vec(ETA @ (G @ N - N @ G)) for G in gens]
    def orth(vs):
        Q, _ = torch.linalg.qr(torch.stack(vs, 1))
        return Q
    return orth(frame_dirs), orth(eig_dirs)


def main():
    rng = np.random.default_rng(7)
    out = {}
    for label, pure in (('generic', False), ('pure_frame_constant_spectrum', True)):
        out[label] = {}
        for kind in ('phi', 'I1', 'L1', 'L2'):
            ratios, fr, ei = [], [], []
            for _ in range(4):
                q, a, D = random_point(rng, pure_frame=pure)
                el, sc = EL(kind, q, a, D)
                Qf, Qe = tangents(q)
                ratios.append(float(el.abs().max()) / sc)
                fr.append(float((Qf.T @ el).abs().max()) / sc)
                ei.append(float((Qe.T @ el).abs().max()) / sc)
            out[label][kind] = {'el_ratio': max(ratios), 'frame_projection': max(fr), 'eigenvalue_projection': max(ei),
                                'null': bool(max(ratios) < 1e-8)}
            print(f'{label:30s} {kind:4s} |EL|/scale = {max(ratios):.2e}  frame part {max(fr):.2e}  '
                  f'eigenvalue part {max(ei):.2e} -> {"NULL" if max(ratios) < 1e-8 else "dynamical"}')
    g = out['generic']
    assert g['phi']['null'] and not g['I1']['null']
    assert not g['L1']['null'] and not g['L2']['null'], 'both weighted terms must be dynamical at generic points'
    p = out['pure_frame_constant_spectrum']
    assert p['L1']['frame_projection'] < 1e-8, 'L1 must be null within the constant-spectrum orbit'
    assert p['L1']['eigenvalue_projection'] > 1e-3, 'L1 must push the eigenvalues (the (d lambda).(d frame) coupling)'
    json.dump(out, open(os.path.join(HERE, 'results', 'appendix_volume_el_test.json'), 'w'), indent=1)
    print('written results/appendix_volume_el_test.json')


if __name__ == '__main__':
    main()
