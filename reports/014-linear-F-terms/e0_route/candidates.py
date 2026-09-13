"""Task 3: scalars X linear in F built from F, eta, epsilon, powers of M, delta_M and eigenprojectors.

Every candidate is F_{mu nu alpha beta} T^{mu nu alpha beta}. Because F is antisymmetric in both
pairs, T reduces to two families (symmetric objects on the same pair give zero):
  cross  X[A,B]      = F_{mu nu alpha beta} A^{mu alpha} B^{nu beta}
  eps    X[k1..k4]   = eps^{rho sigma tau ups} (M^k1)^mu_rho (M^k2)^nu_sigma (M^k3)^alpha_tau (M^k4)^beta_ups F_{mu nu alpha beta}
with A, B in {eta, M eta, M^2 eta, M^3 eta, delta_M, P_a eta}. Powers above 3 and any function of
the eigenvalues in front are allowed but add nothing structurally (Cayley-Hamilton).

Classification on random polynomial fields: zero / null Lagrangian (total derivative, no effect on
the equations of motion) / nontrivial.
"""
import itertools
import torch
from lagrangian import ETA, E_DEFAULT, operator, eigenvalues, field_strength

torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(3)
E = E_DEFAULT
C = (E[0] * E[1]) ** 0.5
M_VAC = ETA @ torch.diag(torch.tensor(E))

EPS = torch.zeros(4, 4, 4, 4)
for p in itertools.permutations(range(4)):
    inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
    EPS[p] = -1.0 if inv % 2 else 1.0


def projector(N, e, a):
    I = torch.eye(4)
    P = I
    for b in range(4):
        if b != a:
            P = P @ (N - e[..., b, None, None] * I) / (e[..., a] - e[..., b])[..., None, None]
    return P


def objects(M):
    """Contravariant symmetric objects and (1,1) powers at a point (or batch)."""
    N = operator(M)
    e = eigenvalues(N, C)
    I = torch.eye(4)
    powers = [I, N, N @ N, N @ N @ N]
    sym = {'eta': ETA, 'M': N @ ETA, 'M2': powers[2] @ ETA, 'M3': powers[3] @ ETA,
           'dM': (2 * projector(N, e, 0) - I) @ ETA}
    for a in range(4):
        sym[f'P{a}'] = projector(N, e, a) @ ETA
    return powers, sym


def lowered_F(dM):
    return ETA @ field_strength(operator(dM))


def cross(A, B):
    return lambda M, dM: torch.einsum('...mnab,...ma,...nb->...', lowered_F(dM), objects(M)[1][A], objects(M)[1][B])


def eps_type(k):
    def X(M, dM):
        P = objects(M)[0]
        return torch.einsum('rstu,...mr,...ns,...at,...bu,...mnab->...', EPS, P[k[0]], P[k[1]], P[k[2]], P[k[3]], lowered_F(dM))
    return X


def candidates():
    names = ['eta', 'M', 'M2', 'M3', 'dM', 'P0', 'P1', 'P2', 'P3']
    cands = {f'F[{A},{B}]': cross(A, B) for i, A in enumerate(names) for B in names[i:]}
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    for (k1, k2), (k3, k4) in itertools.product(pairs, pairs):
        cands[f'eps[M^{k1},M^{k2}|M^{k3},M^{k4}]'] = eps_type((k1, k2, k3, k4))
    return cands


class PolyField:
    """M(x) = M_vac + A_i x^i + B_ij x^i x^j with random symmetric matrix coefficients."""

    def __init__(self, amp):
        s = lambda T: T + T.transpose(-1, -2)
        self.A = s(torch.randn(4, 4, 4, generator=g)) * amp
        B = s(torch.randn(4, 4, 4, 4, generator=g)) * amp
        self.B = 0.5 * (B + B.transpose(0, 1))

    def M(self, x):
        return M_VAC + torch.einsum('iab,i->ab', self.A, x) + torch.einsum('ijab,i,j->ab', self.B, x, x)

    def dM(self, x):
        return self.A + 2 * torch.einsum('ijab,j->iab', self.B, x)


def euler_lagrange(X, field, x):
    """Symmetrised variational derivative dX/dM - d_mu dX/d(d_mu M) along the field at x."""
    M, dM = field.M(x).requires_grad_(True), field.dM(x).requires_grad_(True)
    dXdM = torch.autograd.grad(X(M, dM), M, allow_unused=True)[0]
    dXdM = torch.zeros(4, 4) if dXdM is None else dXdM

    def momentum(y):
        M_, dM_ = field.M(y), field.dM(y)
        M_.requires_grad_(True); dM_.requires_grad_(True)
        return torch.autograd.grad(X(M_, dM_), dM_, create_graph=True)[0]

    jac = torch.autograd.functional.jacobian(momentum, x)
    el = dXdM - torch.einsum('mabm->ab', jac)
    scale = float(dXdM.norm() + jac.norm()) + 1e-300
    return 0.5 * (el + el.mT), scale


def classify(n_fields=4, n_points=40, amp=0.05, n_el=8):
    cands = candidates()
    fields = [PolyField(amp) for _ in range(n_fields)]
    points = [0.5 * torch.randn(4, generator=g) for _ in range(n_points)]
    values, els, verdict = {}, {}, {}
    for name, X in cands.items():
        vals = torch.stack([X(f.M(x), f.dM(x)) for f in fields for x in points])
        values[name] = vals
        with torch.no_grad():
            Fn = torch.stack([lowered_F(f.dM(x)).norm() for f in fields for x in points])
        if float((vals.abs() / Fn).max()) < 1e-10:
            verdict[name] = 'zero'
            continue
        el_rel, el_vecs = [], []
        for i in range(n_el):
            el, scale = euler_lagrange(X, fields[i % n_fields], points[i])
            el_rel.append(float(el.norm()) / scale)
            el_vecs.append(el.flatten())
        els[name] = torch.cat(el_vecs)
        verdict[name] = 'null' if max(el_rel) < 1e-7 else 'nontrivial'
    return values, els, verdict


def rank(columns, tol=1e-9):
    if not columns:
        return 0
    A = torch.stack([c / c.abs().max() for c in columns], 1)
    s = torch.linalg.svdvals(A)
    return int((s > tol * s[0]).sum())


def frame_structures(M, dMs):
    """The 12 well-defined eigenframe components of F: F_{ab,ab} (a<b) and F_{ab,cd} with
    {a,b,c,d} = {0,1,2,3}; every candidate must be a combination of them with coefficients
    depending on the eigenvalues only."""
    N = operator(M)
    ev, V = torch.linalg.eig(N)
    ev, V = ev.real, V.real
    order = ev.argsort(descending=True)
    V = V[:, order]
    V = V / torch.einsum('am,ab,bm->m', V, ETA, V).abs().sqrt()
    C = torch.stack([torch.einsum('mnab,mi,nj,ak,bl->ijkl', lowered_F(dM), V, V, V, V) for dM in dMs])
    diag = [C[:, a, b, a, b] for a in range(4) for b in range(a + 1, 4)]
    comp = [C[:, a, b, c, d] for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)),
                                                   ((1, 2), (0, 3)), ((1, 3), (0, 2)), ((2, 3), (0, 1))]]
    return torch.stack(diag, 1), torch.stack(comp, 1)


def decomposition_check(cands, n_samples=150):
    f = PolyField(0.05)
    M = f.M(0.5 * torch.randn(4, generator=g)) + ETA @ torch.diag(torch.tensor([0.0, 0.0, 0.4, 0.0]))  # separate e_2, e_3
    dMs = [PolyField(0.05).dM(0.5 * torch.randn(4, generator=g)) for _ in range(n_samples)]
    diag, comp = frame_structures(M, dMs)
    worst = {'F[': 0.0, 'ep': 0.0}
    for name, X in cands.items():
        x = torch.stack([X(M, dM) for dM in dMs])
        if float(x.abs().max()) < 1e-8 * float(torch.stack([lowered_F(dM).norm() for dM in dMs]).max()):
            continue
        basis = diag if name.startswith('F[') else comp
        res = (basis @ torch.linalg.lstsq(basis, x[:, None]).solution - x[:, None]).norm() / x.norm()
        if float(res) > worst[name[:2]]:
            worst[name[:2]] = float(res)
            worst[name[:2] + '_worst'] = name
    return worst


def static_frozen_check(cands):
    """Static fields with the time-like sector frozen: d_0 M = 0 and d_i M has no 0-row/column."""
    f = PolyField(0.05)
    x = 0.5 * torch.randn(4, generator=g)
    M, dM = f.M(x), f.dM(x)
    M[0, 1:] = M[1:, 0] = 0
    dM[0] = 0
    dM[:, 0, :] = dM[:, :, 0] = 0
    return {name: float(X(M, dM)) for name, X in cands.items()}


if __name__ == '__main__':
    values, els, verdict = classify()
    cands = candidates()
    groups = {v: [n for n in verdict if verdict[n] == v] for v in ('nontrivial', 'null', 'zero')}
    lines = [f'{v}: {len(names)}  {names if len(names) < 8 else ""}' for v, names in groups.items()]
    lines.append(f'max residual of the decomposition on the 6 diagonal / 6 complementary frame structures: {decomposition_check(cands)}')
    st = static_frozen_check(cands)
    lines.append('static + frozen time sector: non-vanishing candidates = ' + str([n for n, v in st.items() if abs(v) > 1e-12]))
    f = PolyField(0.05)
    M = f.M(0.5 * torch.randn(4, generator=g)) + ETA @ torch.diag(torch.tensor([0.0, 0.0, 0.4, 0.0]))
    M[0, 1:] = M[1:, 0] = 0
    dMs = []
    for _ in range(60):
        dM = PolyField(0.05).dM(0.5 * torch.randn(4, generator=g))
        dM[0] = 0
        dM[:, 0, :] = dM[:, :, 0] = 0
        dMs.append(dM)
    cols = [torch.stack([cands[n](M, dM) for dM in dMs]) for n in st if abs(st[n]) > 1e-12]
    lines.append('at fixed static-frozen M these candidates span (rank as functionals of F): ' + str(rank(cols, 1e-7)))
    for ln in lines:
        print(ln)
    with open('results/candidates_check.txt', 'w') as fh:
        fh.write('\n'.join(lines) + '\n\nclassification of every candidate:\n')
        for n, v in verdict.items():
            fh.write(f'{n}\t{v}\n')
    import json
    dec = decomposition_check(cands)
    json.dump({'n_candidates': len(cands), 'n_nontrivial': len(groups['nontrivial']), 'n_null': len(groups['null']), 'n_zero': len(groups['zero']),
               'null': groups['null'], 'decomposition_residual_cross': dec['F['], 'decomposition_residual_eps': dec['ep'],
               'static_frozen_survivors': [n for n, v in st.items() if abs(v) > 1e-12], 'static_frozen_rank': rank(cols, 1e-7),
               'verdict': verdict}, open('results/candidates.json', 'w'), indent=1)
