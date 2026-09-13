"""Linear-in-F invariants with frame (spectral-projector) coefficients.

One field strength F_{mu nu al be} (eta-commutator, reports 001/005) contracted
with metrics X in {eta, P_0, P_1, P_2, P_3}, P_a = s_a e_a (x) e_a built from an
eta-orthonormal frame e_a (e_a.eta.e_b = s_a delta_ab, s = (-1,1,1,1)); the
frame is the eigenframe of eta M in the model, but at a point it is free data
independent of the first derivatives A_mu (same argument as for u in 010).

Even diagrams: two eta-type pairs among the 4 slots with metric insertions --
patterns (02)(13) and (03)(12) ((01)(23) vanishes by antisymmetry): 2 x 25.
Odd diagrams: eps^{rskl} X0_{r m} X1_{s n} X2_{k a} X3_{l b} F^{mnab}: 5^4.
"""

from fractions import Fraction
from itertools import permutations, product

import numpy as np

from u_family_defs import ETA_NP, F_tensor_np, rand_sym_np

SIGNS = [-1, 1, 1, 1]
EPS = np.zeros((4, 4, 4, 4))
for p in permutations(range(4)):
    sgn, q = 1, list(p)
    for i in range(4):
        for j in range(i + 1, 4):
            if q[i] > q[j]:
                sgn = -sgn
    EPS[p] = sgn
METRIC_NAMES = ['eta', 'P0', 'P1', 'P2', 'P3']


# ---------- frames ----------
def boost_matrix(v):
    v = np.asarray(v, dtype=float)
    b2 = v @ v
    g = 1.0 / np.sqrt(1 - b2)
    L = np.eye(4)
    L[0, 0] = g
    L[0, 1:] = L[1:, 0] = -g * v
    L[1:, 1:] += (g - 1) * np.outer(v, v) / b2
    return L


def rand_frame_np(rng):
    """Random eta-orthonormal frame as columns of a Lorentz matrix."""
    from scipy.linalg import expm
    KB = np.zeros((3, 4, 4))
    for k in range(3):
        KB[k, 0, k + 1] = KB[k, k + 1, 0] = 1.0
    JR = np.zeros((3, 4, 4))
    for k, (i, j) in enumerate(((2, 3), (3, 1), (1, 2))):
        JR[k, i, j], JR[k, j, i] = -1.0, 1.0
    w = rng.standard_normal(6) * 0.6
    L = expm(np.einsum('i,iab->ab', w[:3], KB) + np.einsum('i,iab->ab', w[3:], JR))
    assert np.allclose(L.T @ ETA_NP @ L, ETA_NP, atol=1e-12)
    return L  # e_a = L[:, a]


def projectors_np(E):
    """(2,0) projectors P_a^{mu nu} = s_a e_a^mu e_a^nu; sum = eta^{mu nu}."""
    P = [SIGNS[a] * np.outer(E[:, a], E[:, a]) for a in range(4)]
    assert np.allclose(sum(P), ETA_NP, atol=1e-12)
    return P


def metrics_np(E):
    return [ETA_NP] + projectors_np(E)


# ---------- diagrams ----------
def even_diagrams():
    out = []
    for pattern in ('02-13', '03-12'):
        for x, y in product(range(5), repeat=2):
            out.append(('even', pattern, x, y))
    return out


def odd_diagrams():
    return [('odd', None) + tuple(xs) for xs in product(range(5), repeat=4)]


def eval_diagram_np(d, F, mets):
    if d[0] == 'even':
        _, pattern, x, y = d
        X, Y = mets[x], mets[y]
        if pattern == '02-13':
            return np.einsum('mnab,ma,nb->', F, X, Y)
        return np.einsum('mnab,mb,na->', F, X, Y)
    _, _, x0, x1, x2, x3 = d
    return np.einsum('rskl,rm,sn,ka,lb,mnab->', EPS, mets[x0], mets[x1],
                     mets[x2], mets[x3], F)


def diagram_label(d):
    if d[0] == 'even':
        return f"{d[1]}[{METRIC_NAMES[d[2]]},{METRIC_NAMES[d[3]]}]"
    return "eps[" + ",".join(METRIC_NAMES[i] for i in d[2:]) + "]"


# ---------- exact route ----------
def rat_lorentz(rng):
    """Rational Lorentz matrix: boost along a rational axis x rational rotation."""
    v = Fraction(int(rng.integers(1, 4)), int(rng.integers(4, 9)))
    c, s = (1 + v * v) / (1 - v * v), 2 * v / (1 - v * v)
    axes = [(Fraction(3, 5), Fraction(4, 5), Fraction(0)),
            (Fraction(0), Fraction(3, 5), Fraction(4, 5)),
            (Fraction(2, 7), Fraction(3, 7), Fraction(6, 7)),
            (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3))]
    n = axes[int(rng.integers(0, 4))]
    B = [[Fraction(0)] * 4 for _ in range(4)]
    B[0][0] = c
    for i in range(3):
        B[0][i + 1] = B[i + 1][0] = -s * n[i]
    for i in range(3):
        for j in range(3):
            B[i + 1][j + 1] = (1 if i == j else 0) + (c - 1) * n[i] * n[j]
    # rational rotation in a coordinate plane from a Pythagorean triple
    trip = [(Fraction(3, 5), Fraction(4, 5)), (Fraction(5, 13), Fraction(12, 13)),
            (Fraction(8, 17), Fraction(15, 17))][int(rng.integers(0, 3))]
    (i, j) = [(1, 2), (2, 3), (1, 3)][int(rng.integers(0, 3))]
    R = [[Fraction(1 if a == b else 0) for b in range(4)] for a in range(4)]
    R[i][i], R[j][j], R[i][j], R[j][i] = trip[0], trip[0], -trip[1], trip[1]
    L = [[sum(B[a][k] * R[k][b] for k in range(4)) for b in range(4)] for a in range(4)]
    for a in range(4):
        for b in range(4):
            val = sum(L[k][a] * SIGNS[k] * L[k][b] for k in range(4))
            assert val == (SIGNS[a] if a == b else 0)
    return L


def metrics_exact(L):
    eta = [[Fraction(SIGNS[a]) if a == b else Fraction(0) for b in range(4)]
           for a in range(4)]
    Ps = [[[SIGNS[a] * L[m][a] * L[n][a] for n in range(4)] for m in range(4)]
          for a in range(4)]
    return [eta] + Ps


def eval_diagram_exact(d, F4, mets):
    # F4: nested lists F[m][n][a][b] of Fraction
    tot = Fraction(0)
    if d[0] == 'even':
        _, pattern, x, y = d
        X, Y = mets[x], mets[y]
        for m in range(4):
            for n in range(4):
                for a in range(4):
                    for b in range(4):
                        f = F4[m][n][a][b]
                        if f == 0:
                            continue
                        w = X[m][a] * Y[n][b] if pattern == '02-13' else X[m][b] * Y[n][a]
                        if w:
                            tot += f * w
        return tot
    _, _, x0, x1, x2, x3 = d
    Xs = [mets[x0], mets[x1], mets[x2], mets[x3]]
    for p in permutations(range(4)):
        sgn, q = 1, list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if q[i] > q[j]:
                    sgn = -sgn
        r, s, k, l = p
        for m in range(4):
            xm = Xs[0][r][m]
            if xm == 0:
                continue
            for n in range(4):
                xn = Xs[1][s][n]
                if xn == 0:
                    continue
                for a in range(4):
                    xa = Xs[2][k][a]
                    if xa == 0:
                        continue
                    for b in range(4):
                        xb = Xs[3][l][b]
                        f = F4[m][n][a][b]
                        if xb == 0 or f == 0:
                            continue
                        tot += sgn * xm * xn * xa * xb * f
    return tot
