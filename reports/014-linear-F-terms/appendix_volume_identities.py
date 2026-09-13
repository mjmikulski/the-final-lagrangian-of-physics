"""K0: the two volume weights as members of report 014's linear module.

1. sympy: e_3 = [(tr N)^3 - 3 tr N tr N^2 + 2 tr N^3]/6 (Newton) for a general eta M;
   adj(N) = e_3 - e_2 N + e_1 N^2 - N^3 (Cayley-Hamilton), tr adj = e_3, adj = diag(prod_{c!=a} e_c).
2. vacuum values (E0, 1, 1/E0, 0): e_3 = 1, det = 0, adj = P_3.
3. decomposition (numeric, eigenframe): sqrt(e_3) phi = sqrt(e_3) sum_{a<b} 2 F_ab and
   adj . Phi = sum_{a<b} (w_a + w_b) F_ab, w_a = prod_{c!=a} e_c.
4. the weights on the relaxed lattice electrons (E0 = 100 conventions): range of e_3 / e_3(vac),
   shell means, positivity; det for contrast.
Writes results/k0_identities.json.
"""
import json
import os
import sys

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e0_route'))
from lagrangian import operator, eigenvalues, ETA
from soliton import to_matrix

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---------- 1. identities ----------
e = sp.symbols('e0:4', real=True)
D = sp.diag(*e)
e1 = sum(e)
e2 = sum(e[i] * e[j] for i in range(4) for j in range(i + 1, 4))
e3 = sum(e[i] * e[j] * e[k] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4))
e4 = e[0] * e[1] * e[2] * e[3]
tr = lambda X: X.trace()
newton = (tr(D) ** 3 - 3 * tr(D) * tr(D ** 2) + 2 * tr(D ** 3)) / 6
adj_poly = e3 * sp.eye(4) - e2 * D + e1 * D ** 2 - D ** 3
m = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f'm{min(i, j)}{max(i, j)}', real=True))
Ng = sp.diag(1, -1, -1, -1) * m
newton_g = (tr(Ng) ** 3 - 3 * tr(Ng) * tr(Ng ** 2) + 2 * tr(Ng ** 3)) / 6
cp = Ng.charpoly(sp.Symbol('lam'))
adj_g = e3.subs({}) * 0  # placeholder to keep sympy quiet
ident = {
    'newton_diag': sp.simplify(newton - e3) == 0,
    'newton_general_etaM': sp.expand(newton_g + cp.coeffs()[3]) == 0,
    'cayley_hamilton_adj': sp.simplify(adj_poly - D.adjugate()) == sp.zeros(4),
    'tr_adj_is_e3': sp.simplify(tr(adj_poly) - e3) == 0,
    'adj_diag': [str(sp.factor(adj_poly[a, a])) for a in range(4)],
}
E0 = sp.Symbol('E0', positive=True)
vac = {e[0]: E0, e[1]: 1, e[2]: 1 / E0, e[3]: 0}
ident['vacuum'] = {'e3': str(sp.simplify(e3.subs(vac))), 'det': str(e4.subs(vac)),
                   'adj_diag': [str(sp.simplify(adj_poly[a, a].subs(vac))) for a in range(4)],
                   'de3_de_a': [str(sp.simplify(sp.diff(e3, e[a]).subs(vac))) for a in range(4)]}
assert all(ident[k] for k in ('newton_diag', 'newton_general_etaM', 'cayley_hamilton_adj', 'tr_adj_is_e3'))
out['identities'] = ident
print('identities:', {k: v for k, v in ident.items() if k != 'adj_diag'})

# ---------- 3. decomposition in the F_ab basis (eigenframe) ----------
s = torch.tensor([1.0, -1.0, -1.0, -1.0])


def frame_components(A):
    """F_{mu nu} = [A_mu, A_nu] (mixed); F_ab = s_a s_b F_{abab} (all-covariant); phi = 2 sum_{a<b} F_ab;
    Phi_{nu beta} = sum_mu (F_{mu nu})^mu_beta."""
    F = torch.einsum('mab,nbc->mnac', A, A) - torch.einsum('nab,mbc->mnac', A, A)
    Fcov = torch.einsum('a,mnab->mnab', s, F)
    Fab = {(a, b): float(s[a] * s[b] * Fcov[a, b, a, b]) for a in range(4) for b in range(a + 1, 4)}
    phi = float(torch.einsum('m,n,mnmn->', s, s, Fcov))
    Phi = torch.einsum('mnmb->nb', F)
    return Fab, phi, Phi


dec = []
for trial in range(5):
    ev = torch.tensor([100.0, 1.0, 0.01, 0.0]) + 0.3 * torch.randn(4)
    A = torch.stack([ETA @ (lambda S: S + S.T)(torch.randn(4, 4)) for _ in range(4)])
    Fab, phi, Phi = frame_components(A)
    w = torch.tensor([float(torch.prod(ev[[c for c in range(4) if c != a]])) for a in range(4)])
    L2 = float(torch.einsum('n,n,nn->', s, w, Phi))              # adj^{nu beta} Phi_{nu beta}, adj = diag(w)
    L2_basis = sum((w[a] + w[b]) * Fab[(a, b)] for (a, b) in Fab)
    phi_basis = sum(2 * Fab[(a, b)] for (a, b) in Fab)
    dec.append({'L2_minus_basis': L2 - float(L2_basis), 'L2': L2, 'phi_minus_basis': phi - float(phi_basis)})
worst = max(max(abs(d['L2_minus_basis']) / max(1.0, abs(d['L2'])), abs(d['phi_minus_basis'])) for d in dec)


def R_G(A_cov, G_cov):
    """The double mixed trace of discussion #186 (comment 18275474): R_G = sum G_cd (d_mu M^{nu c} d_nu M^{mu d}
    - d_mu M^{mu c} d_nu M^{nu d}), indices raised with eta; A_cov[mu] = d_mu M covariant, G_cov covariant."""
    Aup = torch.einsum('n,c,mnc->mnc', s, s, A_cov)                       # d_mu M^{nu c}
    t1 = torch.einsum('cd,mnc,nmd->', G_cov, Aup, Aup)
    div = torch.einsum('mmc->c', Aup)                                    # d_mu M^{mu c}
    t2 = torch.einsum('cd,c,d->', G_cov, div, div)
    return float(t1 - t2)


# R_G with G = adj(N) eta is NOT L2 (review round 1): fit it in the even F_ab basis by least squares.
# For G = eta the double mixed trace is -phi (in the module); for G != eta its eta pairs a derivative index with
# a matrix index, so it is not a contraction of F at all and falls outside the report's grammar.
ev = torch.tensor([100.0, 1.0, 0.01, 0.0])
w = torch.tensor([float(torch.prod(ev[[c for c in range(4) if c != a]])) for a in range(4)])
rg = {}
for label, G_cov in (('eta', ETA.clone()), ('adj_eta', torch.diag(w) @ ETA)):
    rows, rhs, rhs_L2 = [], [], []
    for trial in range(40):
        A = torch.stack([(lambda S: S + S.T)(torch.randn(4, 4)) for _ in range(4)])
        AN = torch.einsum('ab,mbc->mac', ETA, A)
        Fab, phi, Phi = frame_components(AN)
        rows.append([Fab[k] for k in sorted(Fab)])
        rhs.append(R_G(A, G_cov))
        rhs_L2.append(float(torch.einsum('n,n,nn->', s, w, Phi)))
    X, y = np.array(rows), np.array(rhs)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rg[label] = {'basis_order': [f'F_{a}{b}' for (a, b) in sorted(Fab)], 'fit_coefficients': coef.tolist(),
                 'fit_residual_relative': float(np.abs(X @ coef - y).max() / np.abs(y).max()),
                 'L2_vs_R_G_relative': float(np.abs(np.array(rhs_L2) - y).max() / np.abs(y).max())}
assert rg['eta']['fit_residual_relative'] < 1e-10 and np.allclose(rg['eta']['fit_coefficients'], -2.0), rg['eta']
assert rg['adj_eta']['fit_residual_relative'] > 0.1 and rg['adj_eta']['L2_vs_R_G_relative'] > 0.1, rg['adj_eta']
rg['statement'] = ('R_eta = -phi (in the module); R_G with G = adj(N) eta is neither L2 nor any combination of the '
                   'six even generators: it is not a contraction of F (its eta pairs a derivative slot with a matrix slot).')
out['R_G'] = rg
print(f"R_G: G = eta -> -phi exactly (residual {rg['eta']['fit_residual_relative']:.1e}); G = adj eta -> outside the F_ab span "
      f"(fit residual {rg['adj_eta']['fit_residual_relative']:.2f}, differs from L2 by {rg['adj_eta']['L2_vs_R_G_relative']:.2f} relative)")
assert worst < 1e-10, worst
out['decomposition'] = {'trials': dec, 'worst_relative': worst,
                        'statement': 'L2 = sum_{a<b} (w_a + w_b) F_ab, w_a = prod_{c!=a} e_c; L1 = sqrt(e3) sum_{a<b} 2 F_ab'}
print(f'decomposition: worst relative residual {worst:.1e}')

# ---------- 4. the weights on the lattice electrons ----------
files = {
    'frozen_static_base_n32': os.path.join(HERE, '..', '016-time-axis-screening', 'results', 'fields', 'static_base_n32.pt'),
    'unfrozen_E0_100_report016': os.path.join(HERE, '..', '016-time-axis-screening', 'results', 'fields', 'static_unfrozen_E0_100.pt'),
    'tilt_c0.3_report016': os.path.join(HERE, '..', '016-time-axis-screening', 'results', 'fields', 'static_tilt0.3.pt'),
}
lat = {}
for label, path in files.items():
    d = torch.load(path, map_location='cpu', weights_only=False)
    M = to_matrix(d['u'].double())
    Ef = tuple(float(x) for x in d['E'])
    n = d['n']; h = d['box'] / n
    c = (torch.arange(n) + 0.5) * h - d['box'] / 2
    r = torch.stack(torch.meshgrid(c, c, c, indexing='ij'), -1).norm(dim=-1)
    N = operator(M)
    t1 = torch.diagonal(N, dim1=-2, dim2=-1).sum(-1)
    t2 = torch.diagonal(N @ N, dim1=-2, dim2=-1).sum(-1)
    t3 = torch.diagonal(N @ N @ N, dim1=-2, dim2=-1).sum(-1)
    e3l = (t1 ** 3 - 3 * t1 * t2 + 2 * t3) / 6
    det = torch.linalg.det(N)
    ev = eigenvalues(N, c=(Ef[0] * Ef[1]) ** 0.5)
    e3_vac = Ef[0] * Ef[1] * Ef[2] + Ef[0] * Ef[1] * Ef[3] + Ef[0] * Ef[2] * Ef[3] + Ef[1] * Ef[2] * Ef[3]
    shells = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 6)]
    lat[label] = {
        'E': Ef, 'n': n, 'box': d['box'], 'e3_vac': e3_vac,
        'e3_over_vac_min': float(e3l.min() / e3_vac), 'e3_over_vac_max': float(e3l.max() / e3_vac),
        'shell_means_e3_over_vac': [float(e3l[(r >= a) & (r < b)].mean() / e3_vac) for a, b in shells],
        'fraction_e3_nonpositive': float((e3l <= 0).double().mean()),
        'det_min': float(det.min()), 'det_max': float(det.max()),
        'smallest_eigenvalue_min': float(ev[..., 3].min()),
        'e1_range': [float(ev[..., 1].min()), float(ev[..., 1].max())],
    }
    print(f'{label}: e3/e3vac in [{lat[label]["e3_over_vac_min"]:.3f}, {lat[label]["e3_over_vac_max"]:.2f}], '
          f'shells {np.round(lat[label]["shell_means_e3_over_vac"], 2).tolist()}, '
          f'e3<=0 fraction {lat[label]["fraction_e3_nonpositive"]:.4f}')
out['lattice'] = lat
assert lat['frozen_static_base_n32']['e3_over_vac_min'] > 0.99 and lat['frozen_static_base_n32']['e3_over_vac_max'] > 10
json.dump(out, open(os.path.join(HERE, 'results', 'appendix_volume_identities.json'), 'w'), indent=1)
print('written results/appendix_volume_identities.json')
