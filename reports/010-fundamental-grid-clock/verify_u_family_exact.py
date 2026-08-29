"""E1 exact route: class structure, rank and identities over Q (Fraction arithmetic).

Confirms the float route independently: trivial zeros are exactly zero, the 21-class
partition has exactly constant cross-ratios, the class-representative span has rank 18
over Q, the 3 linear identities are found exactly, and the G-contracted I1 variants
decompose with the small-integer coefficients seen in float.
"""

import json
from fractions import Fraction

import numpy as np

from u_family_defs import (ETA_DIAG, NAMED, all_diagrams, eval_exact,
                           F_exact_from_A, rand_sym_exact, rand_u_exact,
                           trivially_zero)

rng = np.random.default_rng(20260830)


def exact_sample():
    A = [rand_sym_exact(rng) for _ in range(4)]
    return F_exact_from_A(A), rand_u_exact(rng)


def slot_contract(T, X, slot):
    # T: nested 4-level list of Fraction, X: 4x4 metric; contract slot with X
    out = [[[[Fraction(0)] * 4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    v = T[a][b][c][d]
                    if v == 0:
                        continue
                    idx = [a, b, c, d]
                    for e in range(4):
                        x = X[e][idx[slot]]
                        if x == 0:
                            continue
                        j = idx.copy()
                        j[slot] = e
                        out[j[0]][j[1]][j[2]][j[3]] += x * v
    return out


def F4_exact(Fmat):
    return [[[[Fmat[(mu, nu)][a][b] for b in range(4)] for a in range(4)]
             for nu in range(4)] for mu in range(4)]


def eval_I1_metrics(Fmat, Xd, Xm):
    F = F4_exact(Fmat)
    W = slot_contract(F, Xd, 0)
    W = slot_contract(W, Xd, 1)
    W = slot_contract(W, Xm, 2)
    W = slot_contract(W, Xm, 3)
    tot = Fraction(0)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    v = F[a][b][c][d]
                    if v:
                        tot += v * W[a][b][c][d]
    return tot


def exact_rank_and_nullspace(M):
    # M: list of rows of Fractions; returns rank and nullspace basis over Q
    import copy
    rows = len(M)
    cols = len(M[0])
    A = copy.deepcopy(M)
    piv_cols = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = 1 / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv_cols.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in piv_cols]
    null = []
    for fc in free:
        v = [Fraction(0)] * cols
        v[fc] = Fraction(1)
        for ri, pc in enumerate(piv_cols):
            v[pc] = -A[ri][fc]
        null.append(v)
    return r, null


def main():
    with open('results/u_family_float.json') as f:
        fl = json.load(f)
    diagrams = all_diagrams()

    def canon(caps, pairs):
        return (tuple(sorted(caps)), tuple(sorted(tuple(sorted(p)) for p in pairs)))

    rep_diags = [(tuple(v['caps']), tuple(tuple(p) for p in v['pairs']))
                 for v in fl['representatives'].values()]

    # rebuild the float class partition from scratch for the exact check
    from enumerate_u_family import find_classes, sample_realizable, value_matrix
    V = value_matrix(sample_realizable(40), diagrams)
    zero_cols, classes = find_classes(V)

    # 1) exact zeros + cross-ratio confirmation on 12 exact samples
    samples = [exact_sample() for _ in range(12)]
    vals = [[eval_exact(F, u, *diagrams[i]) for (F, u) in samples] for i in range(735)]
    for i, d in enumerate(diagrams):
        if trivially_zero(*d):
            assert all(v == 0 for v in vals[i]), f'trivial zero {i} nonzero exactly'
    for i in zero_cols:
        assert all(v == 0 for v in vals[i]), f'float-zero diagram {i} nonzero exactly'
    n_checked = 0
    for cl in classes:
        ref = cl[0]
        s0 = next(s for s in range(12) if vals[ref][s] != 0)
        for i in cl[1:]:
            for s in range(12):
                assert vals[i][s] * vals[ref][s0] == vals[ref][s] * vals[i][s0], \
                    f'cross-ratio breaks in class of {ref} at diagram {i}'
            n_checked += 1
    print(f'exact: {len(zero_cols)} zeros confirmed, cross-ratios exact for '
          f'{n_checked} member diagrams in {len(classes)} classes')

    # 2) rank over Q on 24 samples, representatives only
    samples24 = samples + [exact_sample() for _ in range(12)]
    R = [[eval_exact(F, u, *d) for d in rep_diags] for (F, u) in samples24]
    rank, null = exact_rank_and_nullspace(R)
    print(f'exact rank over Q: {rank} (float said {fl["value_rank_realizable"]}); '
          f'{len(null)} linear identities')
    assert rank == fl['value_rank_realizable']

    idents = []
    for v in null:
        den = np.lcm.reduce([x.denominator for x in v if x != 0])
        w = [int(x * den) for x in v]
        g = np.gcd.reduce([abs(x) for x in w if x != 0])
        w = [int(x // g) for x in w]
        idents.append(w)
        terms = ' '.join(f'{c:+d}*C{k}' for k, c in enumerate(w) if c != 0)
        print(f'  identity: {terms} = 0')

    # 3) exact decomposition of G-contracted I1 (verifies the float named_map)
    idx = {c: k for k, c in enumerate(fl['representatives'].keys())}
    for wh, expect in [('deriv', {'C3': 1, 'C10': 4}),
                       ('matrix', {'C3': 1, 'C16': 4}),
                       ('both', {'C3': 1, 'C10': 4, 'C16': 4, 'C19': 16})]:
        ok = True
        for (F, u), row in zip(samples24[:4], R[:4]):
            G = [[ETA_DIAG[a] * (1 if a == b else 0) + 2 * u[a] * u[b]
                  for b in range(4)] for a in range(4)]
            E = [[Fraction(ETA_DIAG[a]) if a == b else Fraction(0)
                  for b in range(4)] for a in range(4)]
            Xd = G if wh in ('deriv', 'both') else E
            Xm = G if wh in ('matrix', 'both') else E
            lhs = eval_I1_metrics(F, Xd, Xm)
            rhs = sum(Fraction(c) * row[idx[cn]] for cn, c in expect.items())
            ok = ok and (lhs == rhs)
        assert ok, f'I1_G_{wh} exact decomposition fails'
        print(f'  I1_G_{wh}: exact decomposition confirmed '
              f'({ " + ".join(f"{v}*{k}" for k, v in expect.items()) })')

    with open('results/u_family_exact.json', 'w') as f:
        json.dump({'rank_Q': rank, 'n_identities': len(null),
                   'identities_int': idents,
                   'n_classes_confirmed': len(classes),
                   'n_zero_confirmed': len(zero_cols)}, f, indent=1)
    print('E1 exact route: ALL PASS')


if __name__ == '__main__':
    main()
