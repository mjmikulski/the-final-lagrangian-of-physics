"""Rank and kernel of the kinetic form K, and the characteristic speeds, on four classes of backgrounds.

1. the vacuum, with and without the frame term K_u;
2. random static backgrounds: generic (all ten components and their spatial gradients random) and frozen
   (M_00 = E_0, M_0i = 0 and d M_0mu = 0, spatial block random);
3. every quadrature point of the committed lattice fields of report 016: the frozen hedgehog, the screened
   endpoint with the time sector free (E_0 = 100) and the endpoints with K_u at c = 0.1 and 3;
4. for each class, the same forms with the indefinite matrix norm (eta in place of delta_M), for comparison.

Route 2 (closed form, numpy) is used on every point, route 1 (autograd on the model code) on a sample, and the
two are compared. Output: results/kinetic_rank.json.
"""
import json
import numpy as np
import torch
from forms import (ETA, BASIS, NAMES, forms_autograd, forms_formula, rank_kernel, speeds2,
                   trace_direction, p0_direction)
from soliton import Grid

E = (100.0, 1.0, 0.01, 0.01)
FIELDS = '../016-time-axis-screening/results/fields/'
LATTICE = {'frozen hedgehog': ('static_base_n32.pt', 0.0),
           'time sector free (E0 = 100)': ('static_unfrozen_E0_100.pt', 0.0),
           'K_u, c = 0.1': ('static_tilt0.1.pt', 0.1),
           'K_u, c = 3': ('static_tilt3.0.pt', 3.0)}
DIRS = {'x': (1, 0, 0), 'z': (0, 0, 1), '(1,1,1)': (1, 1, 1)}
RTOL = 1e-9
rng = np.random.default_rng(17)


def boost_rotation(rng, size=0.5):
    """A random Lorentz transformation exp(sum of generators) acting on the covariant M by L^T M L."""
    from scipy.linalg import expm
    Gen = np.zeros((4, 4))
    b, w = rng.normal(size=3) * size, rng.normal(size=3) * size
    Gen[0, 1:], Gen[1:, 0] = b, -b                                     # antisymmetric: eta Gen is in so(1,3)
    Gen[1, 2], Gen[2, 1], Gen[1, 3], Gen[3, 1], Gen[2, 3], Gen[3, 2] = w[2], -w[2], -w[1], w[1], w[0], -w[0]
    L = expm(ETA @ Gen)
    assert np.allclose(L.T @ ETA @ L, ETA)
    return L


def random_background(rng, frozen):
    """A point background near the vacuum spectrum. Generic: a Lorentz-transformed diagonal M plus random
    symmetric spatial gradients in all ten components. Frozen: M_00 = E_0, M_0i = 0, spatial block a rotated
    diag(-E_1, -E_2, -E_3) with random spatial-block gradients."""
    M = np.diag([E[0], -E[1], -E[2], -E[3]]) + np.diag([0, 0, -0.05, 0.03]) * rng.normal(size=4)
    dM = np.zeros((4, 4, 4))
    if frozen:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        M[1:, 1:] = Q @ M[1:, 1:] @ Q.T
        for i in range(3):
            S = rng.normal(size=(3, 3))
            dM[1 + i, 1:, 1:] = S + S.T
    else:
        L = boost_rotation(rng)
        M = L.T @ M @ L
        for i in range(3):
            S = rng.normal(size=(4, 4))
            dM[1 + i] = S + S.T
    return M, dM


def overlap(Z, vecs):
    """How much of span(vecs) lies in the kernel Z (1 = fully): smallest singular value of the projection."""
    V = np.linalg.qr(np.array(vecs).T)[0]
    return float(np.linalg.svd(Z.T @ V, compute_uv=False).min()) if Z.shape[1] >= V.shape[1] else 0.0


def analyse_point(M, dM, c=0.0, sample_route1=False):
    out = {}
    K, G = forms_formula(M, dM, c, DIRS['z'])
    rk, w, Z = rank_kernel(K, RTOL)
    out['rank'] = rk
    out['kernel_contains_trace'] = overlap(Z, [trace_direction()]) if Z.shape[1] else 0.0
    out['kernel_contains_trace_and_P0'] = overlap(Z, [trace_direction(), p0_direction(M)]) if Z.shape[1] >= 2 else 0.0
    out['next_eigenvalue_ratio'] = float(np.sort(w)[10 - rk] / abs(w).max()) if rk else 0.0
    s_all, gk = [], 0.0
    for k in DIRS.values():
        k = np.array(k, float) / np.linalg.norm(k)
        K, G = forms_formula(M, dM, c, k)
        if rk:
            s, g = speeds2(K, G, RTOL)
            s_all += list(s)
            gk = max(gk, g)
    out['speed2_min'] = float(min(s_all)) if s_all else None
    out['speed2_max'] = float(max(s_all)) if s_all else None
    out['symbol_on_kernel'] = gk
    Ke, _ = forms_formula(M, dM, c, DIRS['z'], metric='eta')
    we = np.linalg.eigvalsh(0.5 * (Ke + Ke.T))
    out['eta_norm_negative_directions'] = int((we < -RTOL * abs(we).max()).sum()) if abs(we).max() > 0 else 0
    if sample_route1:
        dev = 0.0
        for k in DIRS.values():
            k = np.array(k, float) / np.linalg.norm(k)
            K1, G1 = forms_autograd(M, dM, E, c, k)
            K2, G2 = forms_formula(M, dM, c, k)
            sc = max(abs(K1).max(), abs(G1).max(), 1e-300)
            dev = max(dev, float(max(abs(K1 - K2).max(), abs(G1 - G2).max()) / sc))
        out['route_deviation'] = dev
    return out


def summarise(rows):
    s = {'points': len(rows)}
    ranks = [r['rank'] for r in rows]
    s['rank_counts'] = {str(k): ranks.count(k) for k in sorted(set(ranks))}
    for key in ('kernel_contains_trace', 'kernel_contains_trace_and_P0'):
        v = [r[key] for r in rows]
        s[key + '_min'] = float(min(v))
    v = [r['next_eigenvalue_ratio'] for r in rows]
    s['next_eigenvalue_ratio_min'] = float(min(v))
    s['next_eigenvalue_ratio_median'] = float(np.median(v))
    sp = [r for r in rows if r['speed2_min'] is not None]
    if sp:
        s['speed2_min'] = float(min(r['speed2_min'] for r in sp))
        s['speed2_max'] = float(max(r['speed2_max'] for r in sp))
        s['symbol_on_kernel_max'] = float(max(r['symbol_on_kernel'] for r in sp))
    s['eta_norm_points_with_negative_K'] = int(sum(r['eta_norm_negative_directions'] > 0 for r in rows))
    rd = [r['route_deviation'] for r in rows if 'route_deviation' in r]
    if rd:
        s['route1_route2_max_relative_deviation'] = float(max(rd))
        s['route1_sampled_points'] = len(rd)
    return s


def vacuum():
    M = np.diag([E[0], -E[1], -E[2], -E[3]])
    dM = np.zeros((4, 4, 4))
    res = {}
    for c in (0.0, 1.0):
        K1, _ = forms_autograd(M, dM, E, c)
        K2, _ = forms_formula(M, dM, c)
        rk, w, Z = rank_kernel(K2 + 1e-300 * np.eye(10), RTOL) if abs(K2).max() == 0 else rank_kernel(K2, RTOL)
        rk = 0 if abs(K2).max() == 0 else rk
        live = [NAMES[i] for i in range(10) if abs(K2[i, i]) > 1e-12]
        res['c = %g' % c] = {'rank': rk, 'dynamical_components': live,
                             'route_deviation': float(abs(K1 - K2).max())}
        if c:
            # plane wave of the three tilt components: K = G(k) for |k| = 1 on those components -> unit speed
            s = []
            for k in DIRS.values():
                k = np.array(k, float) / np.linalg.norm(k)
                K, G = forms_formula(M, dM, c, k)
                s += list(speeds2(K, G, RTOL)[0])
            res['c = %g' % c]['speed2'] = sorted(set(np.round(s, 12)))
    return res


def special_strata():
    """Exact point backgrounds on lower strata (review round 1): only the gradient triple matters.
    (a) a boost twist N = L(z)^-1 N_v L(z): one gradient d_zN = E01 - E10 (operator form), F = 0, exact static;
    (b) a rotation twist in the 12 plane: d_zN = E12 + E21; the tilt X = E03 - E30 commutes with it;
    (c) the uniaxial frozen hedgehog on the z axis (a = 0.8, b = 0.11, a' = 0.2, b' = -0.1 at r = 1):
        X = diag(0, 2, -1, -1) has K[X, X] > 0 and G(e_x) X = 0, an exact zero-speed dynamical direction."""
    def cov(Nop):                                   # covariant M from the operator N = eta M
        return ETA @ Nop
    Nv = np.diag([E[0], E[1], E[2], E[3]]).astype(float)
    out = {}
    for name, D in (('boost twist', np.eye(4)[[0]].T @ np.eye(4)[[1]] - np.eye(4)[[1]].T @ np.eye(4)[[0]]),
                    ('rotation twist', np.eye(4)[[1]].T @ np.eye(4)[[2]] + np.eye(4)[[2]].T @ np.eye(4)[[1]])):
        M = cov(Nv)
        dM = np.zeros((4, 4, 4))
        dM[3] = cov(D)
        rows = {}
        for c in (0.0, 1.0):
            K2, _ = forms_formula(M, dM, c)
            K1, _ = forms_autograd(M, dM, E, c)
            rows['c = %g' % c] = {'rank': rank_kernel(K2, RTOL)[0], 'route_deviation': float(abs(K1 - K2).max() / abs(K2).max())}
        rows['M03_kinetic_c0'] = float(forms_formula(M, dM, 0.0)[0][3, 3])
        out[name] = rows
    a, b, ap, bp, r = 0.8, 0.11, 0.2, -0.1, 1.0
    s = (a - b) / r
    Nop = np.diag([E[0], b, b, a])
    D = np.zeros((3, 4, 4))
    D[0, 1, 3] = D[0, 3, 1] = s
    D[1, 2, 3] = D[1, 3, 2] = s
    D[2] = np.diag([0, bp, bp, ap])
    M = cov(Nop)
    dM = np.zeros((4, 4, 4))
    for i in range(3):
        dM[1 + i] = cov(D[i])
    X = cov(np.diag([0.0, 2.0, -1.0, -1.0]))
    x = np.array([X[p, q] for p, q in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]])
    rows = {}
    for c in (0.0, 1.0):
        K2, G2 = forms_formula(M, dM, c, (1.0, 0.0, 0.0))
        K1, G1 = forms_autograd(M, dM, E, c, (1.0, 0.0, 0.0))
        rows['c = %g' % c] = {'rank': rank_kernel(K2, RTOL)[0], 'K_XX': float(x @ K2 @ x),
                              'G_X_norm': float(np.linalg.norm(G2 @ x)), 'speed2_min': float(speeds2(K2, G2, RTOL)[0].min()),
                              'route_deviation': float(max(abs(K1 - K2).max(), abs(G1 - G2).max()) / abs(K2).max())}
    out['uniaxial hedgehog, z axis, k = x'] = rows
    return out


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    record = {'rtol': RTOL, 'vacuum': vacuum(), 'special': special_strata(), 'random': {}, 'lattice': {}}
    print('special', record['special'])
    for frozen in (False, True):
        rows = []
        for i in range(300):
            M, dM = random_background(rng, frozen)
            rows.append(analyse_point(M, dM, 0.0, sample_route1=(i < 30)))
            rows.append(analyse_point(M, dM, 1.0, sample_route1=(i < 10)))
        record['random']['frozen' if frozen else 'generic'] = summarise(rows)
        print('random', 'frozen' if frozen else 'generic', record['random']['frozen' if frozen else 'generic'])
    for name, (fn, c) in LATTICE.items():
        d = torch.load(FIELDS + fn)
        grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
        Mq, dMq = grid.derivatives(d['u'].double())
        Mq, dMq = Mq.reshape(-1, 4, 4).numpy(), dMq.reshape(-1, 4, 4, 4).numpy()
        pts = rng.choice(len(Mq), size=4000, replace=False)
        rows = [analyse_point(Mq[j], dMq[j], c, sample_route1=(n < 40)) for n, j in enumerate(pts)]
        record['lattice'][name] = dict(field=fn, c=c, **summarise(rows))
        print(name, record['lattice'][name])
    json.dump(record, open('results/kinetic_rank.json', 'w'), indent=1)
