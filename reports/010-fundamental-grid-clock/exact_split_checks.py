"""E2 exact route: Fraction-arithmetic confirmations of the split verdicts.

1. T1 at generic level: every class representative has s != 0 on exact static
   samples (A0 = 0, random rational A_i, random rational unit-timelike u) —
   no exact covariant pure-kinetic invariant exists in the family.
2. m == 0 candidates from the float route (C3 = I1, C16) hold exactly on
   generic exact samples: I(+A0) == I(-A0).
3. Degree <= 2 in A0 holds exactly for every representative.

Writes results/exact_split_checks.json.
"""

import json
from fractions import Fraction

import numpy as np

from u_family_defs import (F_exact_from_A, eval_exact, rand_sym_exact,
                           rand_u_exact)

rng = np.random.default_rng(20260901)


def load_reps():
    with open('results/u_family_float.json') as f:
        fl = json.load(f)
    return {name: (tuple(v['caps']), tuple(tuple(p) for p in v['pairs']))
            for name, v in fl['representatives'].items()}


def zero_mat():
    return [[Fraction(0)] * 4 for _ in range(4)]


def scale_mat(m, lam):
    return [[lam * m[a][b] for b in range(4)] for a in range(4)]


def main():
    reps = load_reps()

    # 1) T1 generic: s != 0 exactly for every class (20 static samples)
    s_seen_nonzero = {n: False for n in reps}
    for _ in range(20):
        A = [zero_mat()] + [rand_sym_exact(rng) for _ in range(3)]
        u = rand_u_exact(rng)
        F = F_exact_from_A(A)
        for n, (caps, pairs) in reps.items():
            if eval_exact(F, u, caps, pairs) != 0:
                s_seen_nonzero[n] = True
    assert all(s_seen_nonzero.values()), \
        f'exact static zero found: {[n for n, v in s_seen_nonzero.items() if not v]}'

    # 2) + 3) generic exact samples: degree guard and m == 0 for C3, C16
    m_zero_candidates = ['C3', 'C16']
    m_zero_holds = {n: True for n in m_zero_candidates}
    for _ in range(12):
        A0 = rand_sym_exact(rng)
        Ai = [rand_sym_exact(rng) for _ in range(3)]
        u = rand_u_exact(rng)
        vals = {}
        for lam in (0, 1, -1, 2):
            F = F_exact_from_A([scale_mat(A0, Fraction(lam))] + Ai)
            vals[lam] = {n: eval_exact(F, u, *reps[n]) for n in reps}
        for n in reps:
            s = vals[0][n]
            m = (vals[1][n] - vals[-1][n]) / 2
            k = (vals[1][n] + vals[-1][n]) / 2 - s
            assert vals[2][n] == s + 2 * m + 4 * k, f'{n}: exact degree > 2'
        for n in m_zero_candidates:
            if vals[1][n] != vals[-1][n]:
                m_zero_holds[n] = False
    assert all(m_zero_holds.values()), f'm==0 fails exactly: {m_zero_holds}'

    out = {'T1_generic_no_pure_kinetic': True,
           'm_zero_exact': m_zero_candidates,
           'degree_le_2_exact': True,
           'n_static_samples': 20, 'n_generic_samples': 12}
    with open('results/exact_split_checks.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('T1 (generic level): no class has s == 0 exactly -> CONFIRMED on 20 samples')
    print('m == 0 exact for C3 (I1) and C16; degree <= 2 exact for all 21 reps')
    print('E2 exact checks: ALL PASS')


if __name__ == '__main__':
    main()
