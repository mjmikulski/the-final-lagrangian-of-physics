"""L0 key test: which linear classes are null Lagrangians (Euler-Lagrange
expression identically zero) when the projectors depend on M.

L(M, A) with A_mu = d_mu M; P_a(M) = Lagrange polynomials in eta M with the
eigenvalues from torch.linalg.eigvals (differentiable off degeneracy).
EL_q = dL/dq - sum_mu d_mu(dL/da_mu), d_mu g = jvp(g; (dq, da_nu) =
(a_mu, D_{mu nu})) with D the second jet. Controls: phi (constant
coefficients) must be null; I1 must be dynamical. Writes results/null_test.json.
"""
import json
import numpy as np
import torch
from torch.func import jacrev, jvp
torch.set_default_dtype(torch.float64)
from linear_defs import EPS, SIGNS, even_diagrams, odd_diagrams, diagram_label

ETA = torch.diag(torch.tensor([-1.0, 1, 1, 1]))
EPS_T = torch.tensor(EPS)
IU = torch.triu_indices(4, 4)

def sym(q):
    M = torch.zeros(4, 4, dtype=q.dtype)
    M[IU[0], IU[1]] = q
    return M + M.T - torch.diag(torch.diag(M))

def vec(M):
    return M[IU[0], IU[1]]

def projectors(M):
    x = ETA @ M
    lam = torch.linalg.eigvals(x).real
    I = torch.eye(4, dtype=M.dtype)
    Ps = []
    for a in range(4):
        P = I.clone()
        for b in range(4):
            if b != a:
                P = P @ (x - lam[b] * I) / (lam[a] - lam[b])
        Ps.append(P @ ETA)          # (2,0): P^{mu nu} = P^mu_rho eta^{rho nu}
    return Ps

def order_by_vacuum(Ps, M):
    # sort projectors by eigenvalue so that P0 = timelike branch etc. is
    # consistent across samples: use the eigenvalue itself
    x = ETA @ M
    lam = torch.linalg.eigvals(x).real
    idx = torch.argsort(lam, descending=True)   # (8, 1, 0.3, 0) ordering
    return [Ps[i] for i in idx.tolist()]

def F_of(M, A):
    F = torch.zeros(4, 4, 4, 4, dtype=M.dtype)
    for m in range(4):
        for n in range(m + 1, 4):
            f = A[m] @ ETA @ A[n] - A[n] @ ETA @ A[m]
            F[m, n] = f; F[n, m] = -f
    return F

def L_of(d, M, A):
    mets = [ETA] + order_by_vacuum(projectors(M), M)
    F = F_of(M, A)
    if d[0] == 'I1':
        return torch.einsum('mnab,mM,nN,aA,bB,MNAB->', F, ETA, ETA, ETA, ETA, F)
    if d[0] == 'even':
        X, Y = mets[d[2]], mets[d[3]]
        if d[1] == '02-13':
            return torch.einsum('mnab,ma,nb->', F, X, Y)
        return torch.einsum('mnab,mb,na->', F, X, Y)
    return torch.einsum('rskl,rm,sn,ka,lb,mnab->', EPS_T, mets[d[2]], mets[d[3]], mets[d[4]], mets[d[5]], F)

def EL(d, q, a, D):
    """q: 10 (M), a: (4,10) (A_mu), D: (4,4,10) second jet (sym in mu nu)."""
    def L_q(q_, a_):
        return L_of(d, sym(q_), [sym(a_[m]) for m in range(4)])
    dLdq = jacrev(L_q, argnums=0)(q, a)
    def g(q_, a_):                         # dL/da_mu, shape (4,10)
        return jacrev(L_q, argnums=1)(q_, a_)
    total = torch.zeros(10, dtype=q.dtype)
    scale = float(dLdq.abs().max())
    for mu in range(4):
        tang_q = a[mu]
        tang_a = D[mu]                     # (4,10): D_{mu nu}
        _, dg = jvp(g, (q, a), (tang_q, tang_a))
        total = total + dg[mu]
        scale += float(dg[mu].abs().max())   # per-mu magnitudes: no 0/0
    el = dLdq - total
    return el, scale + 1e-300

def random_point(rng):
    # M with a distinct real spectrum near the model vacuum, random frame
    from linear_defs import rand_frame_np
    E = rand_frame_np(rng)
    h = np.array([-8.0, 1.0, 0.3, 0.0]) + rng.uniform(-0.1, 0.1, 4)
    M = E @ np.diag(h) @ E.T
    M = 0.5 * (M + M.T)
    A = [0.5 * (x + x.T) for x in rng.standard_normal((4, 4, 4))]
    D = np.zeros((4, 4, 4, 4))
    for m in range(4):
        for n in range(m, 4):
            s = rng.standard_normal((4, 4)); s = 0.5 * (s + s.T)
            D[m, n] = D[n, m] = s
    q = vec(torch.tensor(M)).clone()
    a = torch.stack([vec(torch.tensor(x)) for x in A])
    Dq = torch.stack([torch.stack([vec(torch.tensor(D[m, n])) for n in range(4)]) for m in range(4)])
    return q, a, Dq

def main():
    rng = np.random.default_rng(3)
    fl = json.load(open('results/linear_float.json'))
    reps = {k: tuple([v['diagram'][0], v['diagram'][1]] + v['diagram'][2:]) for k, v in fl['reps'].items()}
    tests = {'CONTROL_I1': ('I1',), 'CONTROL_phi': ('even', '02-13', 0, 0)}
    tests.update(reps)
    out = {}
    for name, d in tests.items():
        ratios = []
        for _ in range(3):
            q, a, D = random_point(rng)
            el, sc = EL(d, q, a, D)
            ratios.append(float(el.abs().max()) / sc)
        r = max(ratios)
        out[name] = {'label': diagram_label(d) if d[0] != 'I1' else 'I1', 'el_ratio': r,
                     'null': bool(r < 1e-8)}
        print(f"{name:12s} {out[name]['label']:26s} |EL|/scale = {r:.2e}  -> {'NULL' if r < 1e-8 else 'dynamical'}")
    assert out['CONTROL_phi']['null'] and not out['CONTROL_I1']['null']
    assert out['L11']['null'], 'chi (constant eps) must be null (005)'
    json.dump(out, open('results/null_test.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
