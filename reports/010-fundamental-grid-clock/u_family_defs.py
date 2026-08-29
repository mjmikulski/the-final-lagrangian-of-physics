"""Shared definitions for the u-decorated quadratic family (reports 001/005 lineage).

Slots 0..7: factor1 (mu,nu,al,be) = 0..3, factor2 = 4..7; derivative pairs (0,1) and
(4,5), matrix pairs (2,3) and (6,7). A diagram is (caps, pairs): caps = slots
contracted with u^a, pairs = eta^{ab} pairings of the rest. Both evaluation routes
sum one index per pair (eta is diagonal) — max 4^5 assignments.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
import torch

ETA_DIAG = [-1, 1, 1, 1]
ETA_NP = np.diag(np.array(ETA_DIAG, dtype=np.float64))
DERIV_PAIRS = [(0, 1), (4, 5)]
MATRIX_PAIRS = [(2, 3), (6, 7)]
ANTISYM = DERIV_PAIRS + MATRIX_PAIRS


def matchings(slots):
    if not slots:
        yield []
        return
    a = slots[0]
    for i in range(1, len(slots)):
        b = slots[i]
        rest = slots[1:i] + slots[i + 1:]
        for m in matchings(rest):
            yield [(a, b)] + m


def all_diagrams():
    diags = []
    for ncaps in (0, 2, 4):
        for caps in combinations(range(8), ncaps):
            rest = [s for s in range(8) if s not in caps]
            for pairs in matchings(rest):
                diags.append((tuple(caps), tuple(pairs)))
    return diags


def trivially_zero(caps, pairs):
    cs = set(caps)
    return any(set(p) <= cs for p in ANTISYM) or any(tuple(sorted(p)) in ANTISYM for p in pairs)


def F_from_A(A, eta):
    # A: list of 4 symmetric 4x4; returns F[mu][nu] matrices (lists or arrays)
    B = [A[m] @ eta for m in range(4)] if isinstance(A[0], np.ndarray) else None
    F = {}
    for mu in range(4):
        for nu in range(4):
            F[(mu, nu)] = A[mu] @ eta @ A[nu] - A[nu] @ eta @ A[mu]
    return F


def F_tensor_np(A):
    F = np.zeros((4, 4, 4, 4))
    for mu in range(4):
        for nu in range(4):
            F[mu, nu] = A[mu] @ ETA_NP @ A[nu] - A[nu] @ ETA_NP @ A[mu]
    return F


def eval_np(F1, F2, u, caps, pairs):
    letters = 'abcdefgh'
    ops = [F1, F2]
    subs = [letters[:4], letters[4:]]
    for (i, j) in pairs:
        ops.append(ETA_NP)
        subs.append(letters[i] + letters[j])
    for c in caps:
        ops.append(u)
        subs.append(letters[c])
    return np.einsum(','.join(subs) + '->', *ops, optimize=True)


def eval_torch(F1, F2, u, caps, pairs, eta_t):
    letters = 'abcdefgh'
    ops = [F1, F2]
    subs = [letters[:4], letters[4:]]
    for (i, j) in pairs:
        ops.append(eta_t)
        subs.append(letters[i] + letters[j])
    for c in caps:
        ops.append(u)
        subs.append(letters[c])
    return torch.einsum(','.join(subs) + '->', *ops)


def eval_exact(Fmat, u, caps, pairs):
    # Fmat: dict (mu,nu) -> 4x4 nested lists of Fraction; u: list of 4 Fraction
    caps = list(caps)
    pairs = list(pairs)
    idx = [0] * 8
    nsum = len(caps) + len(pairs)
    total = Fraction(0)
    stack = [(0, Fraction(1))]

    def rec(pos, weight):
        nonlocal total
        if pos == nsum:
            f1 = Fmat[(idx[0], idx[1])][idx[2]][idx[3]]
            if f1 == 0:
                return
            f2 = Fmat[(idx[4], idx[5])][idx[6]][idx[7]]
            if f2 == 0:
                return
            total += weight * f1 * f2
            return
        if pos < len(caps):
            c = caps[pos]
            for a in range(4):
                if u[a] == 0:
                    continue
                idx[c] = a
                rec(pos + 1, weight * u[a])
        else:
            sa, sb = pairs[pos - len(caps)]
            for a in range(4):
                idx[sa] = a
                idx[sb] = a
                rec(pos + 1, weight * ETA_DIAG[a])
    rec(0, Fraction(1))
    return total


def F_exact_from_A(A):
    # A: list of 4 4x4 nested lists of Fraction
    def mm(X, Y):
        return [[sum(X[a][c] * Y[c][b] for c in range(4)) for b in range(4)] for a in range(4)]

    def meta(X):
        return [[X[a][b] * ETA_DIAG[b] for b in range(4)] for a in range(4)]

    F = {}
    AE = [meta(A[m]) for m in range(4)]
    for mu in range(4):
        for nu in range(4):
            P = mm(AE[mu], A[nu])
            Q = mm(AE[nu], A[mu])
            F[(mu, nu)] = [[P[a][b] - Q[a][b] for b in range(4)] for a in range(4)]
    return F


def rand_sym_np(rng, scale=1.0):
    m = rng.standard_normal((4, 4)) * scale
    return (m + m.T) / 2


def rand_u_np(rng, wscale=0.6):
    w = rng.standard_normal(3) * wscale
    return np.concatenate([[np.sqrt(1 + w @ w)], w])


def generic_F_np(rng):
    R = rng.standard_normal((4, 4, 4, 4))
    R = R - R.transpose(1, 0, 2, 3)
    R = R - R.transpose(0, 1, 3, 2)
    return R / 4


RATIONAL_AXES = [
    (Fraction(3, 5), Fraction(4, 5), Fraction(0)),
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(2, 7), Fraction(3, 7), Fraction(6, 7)),
    (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3)),
    (Fraction(0), Fraction(4, 5), Fraction(3, 5)),
]


def rand_sym_exact(rng):
    m = [[Fraction(0)] * 4 for _ in range(4)]
    for a in range(4):
        for b in range(a, 4):
            v = Fraction(int(rng.integers(-4, 5)), int(rng.integers(1, 4)))
            m[a][b] = v
            m[b][a] = v
    return m


def rand_u_exact(rng):
    v = Fraction(int(rng.integers(1, 4)), int(rng.integers(4, 8)))
    c = (1 + v * v) / (1 - v * v)
    s = 2 * v / (1 - v * v)
    n = RATIONAL_AXES[int(rng.integers(0, len(RATIONAL_AXES)))]
    u = [c, s * n[0], s * n[1], s * n[2]]
    assert -u[0] ** 2 + u[1] ** 2 + u[2] ** 2 + u[3] ** 2 == -1
    return u


# named diagrams (001 recipes + the B_k carrier)
NAMED = {
    'I1': ((), ((0, 4), (1, 5), (2, 6), (3, 7))),
    'I2': ((), ((0, 6), (1, 7), (2, 4), (3, 5))),
    'I3': ((), ((0, 4), (2, 5), (1, 6), (3, 7))),
    'I4': ((), ((0, 2), (4, 6), (1, 5), (3, 7))),
    'I5': ((), ((0, 2), (4, 6), (1, 7), (3, 5))),
    'I6': ((), ((0, 2), (1, 3), (4, 6), (5, 7))),
    'Bk_carrier': ((0, 4), ((1, 5), (2, 6), (3, 7))),
}
