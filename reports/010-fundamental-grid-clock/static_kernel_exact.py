"""Review finding 1 (PR #12): the static-part kernel of the family SPAN.

The original T1 check looped over individual representatives; the reviewer
exhibits P = C0 - 4*C2 + C5 with s(P) = 0 exactly. This script settles the
span-level question over Q:

  1. verifies the reviewer's counterexample exactly (s = 0, k = 0, m != 0);
  2. computes the exact kernel of the static map s: span{C0..C20} -> functions
     (sampled on 30 exact static configurations) modulo the three family
     identities, and the kernel of the joint (s, k) map on generic configs;
  3. asserts the grid-relevant statement: every kernel direction of s is also
     a kernel direction of k — i.e. NO invariant in the span has s == 0 with
     k != 0 (no exact pure-kinetic brake exists), and the s-kernel modulo the
     identities is spanned by the reviewer's pure-linear P, which drops out
     of H identically (degree-1 terms cancel in the Legendre transform, E0).

Writes results/static_kernel_exact.json.
"""

import json
from fractions import Fraction

import numpy as np

from u_family_defs import (F_exact_from_A, eval_exact, rand_sym_exact,
                           rand_u_exact)
from verify_u_family_exact import exact_rank_and_nullspace

rng = np.random.default_rng(20260903)


def load_reps():
    with open('results/u_family_float.json') as f:
        fl = json.load(f)
    return {n: (tuple(v['caps']), tuple(tuple(p) for p in v['pairs']))
            for n, v in fl['representatives'].items()}


def zero_mat():
    return [[Fraction(0)] * 4 for _ in range(4)]


def scale_mat(m, lam):
    return [[lam * m[a][b] for b in range(4)] for a in range(4)]


def main():
    reps = load_reps()
    names = list(reps.keys())
    idx = {n: i for i, n in enumerate(names)}

    # 1) the reviewer's counterexample P = C0 - 4 C2 + C5, exact split
    coeff = {n: Fraction(0) for n in names}
    coeff['C0'], coeff['C2'], coeff['C5'] = (Fraction(1), Fraction(-4),
                                             Fraction(1))
    m_seen = False
    for _ in range(12):
        A0 = rand_sym_exact(rng)
        Ai = [rand_sym_exact(rng) for _ in range(3)]
        u = rand_u_exact(rng)
        vals = {}
        for lam in (0, 1, -1):
            F = F_exact_from_A([scale_mat(A0, Fraction(lam))] + Ai)
            vals[lam] = {n: eval_exact(F, u, *reps[n]) for n in names}
        sP = sum(coeff[n] * vals[0][n] for n in names)
        mP = sum(coeff[n] * (vals[1][n] - vals[-1][n]) for n in names) / 2
        kP = sum(coeff[n] * (vals[1][n] + vals[-1][n]) for n in names) / 2 - sP
        assert sP == 0 and kP == 0, 'counterexample split not (0, m, 0)'
        m_seen = m_seen or (mP != 0)
    assert m_seen, 'P has m == 0 on all samples?'
    print('1. reviewer counterexample CONFIRMED exactly: '
          'P = C0 - 4*C2 + C5 has s == 0, k == 0, m != 0')

    # 2) exact kernels over Q: static map (30 static samples) and the joint
    #    (s, k) map (30 generic samples via the lambda split)
    S_rows, SK_rows, G_rows = [], [], []
    for _ in range(30):
        A0 = rand_sym_exact(rng)
        Ai = [rand_sym_exact(rng) for _ in range(3)]
        u = rand_u_exact(rng)
        vals = {}
        for lam in (0, 1, -1):
            F = F_exact_from_A([scale_mat(A0, Fraction(lam))] + Ai)
            vals[lam] = [eval_exact(F, u, *reps[n]) for n in names]
        s_row = vals[0]
        k_row = [(vals[1][i] + vals[-1][i]) / 2 - vals[0][i]
                 for i in range(len(names))]
        S_rows.append(s_row)
        SK_rows.append(s_row)
        SK_rows.append(k_row)
        G_rows.append(vals[1])
    rank_G, _ = exact_rank_and_nullspace(G_rows)
    rank_S, null_S = exact_rank_and_nullspace(S_rows)
    rank_SK, null_SK = exact_rank_and_nullspace(SK_rows)
    print(f'2. exact ranks over Q: generic {rank_G}, static-map {rank_S}, '
          f'joint (s,k)-map {rank_SK}')
    print(f'   kernel dims: s: {len(null_S)}, (s,k): {len(null_SK)} '
          f'(family identities: {21 - rank_G})')

    # 3) the grid-relevant statement: ker(s) == ker(s,k) as subspaces
    assert rank_S == rank_SK, \
        'a pure-kinetic direction exists: ker(s) is strictly larger than ker(s,k)'
    # and the reviewer's P lies in ker(s) beyond the family identities
    assert rank_G - rank_S == 1, \
        f'expected a 1-dim s-kernel modulo identities, got {rank_G - rank_S}'
    print('3. ker(s) == ker(s, k): every static-free direction is also '
          'kinetic-free (pure linear). NO pure-kinetic invariant exists in '
          'the span; the s-kernel modulo the 3 identities is 1-dimensional '
          '(the reviewer\'s P), and degree-1 terms drop from H identically.')

    with open('results/static_kernel_exact.json', 'w') as f:
        json.dump({'counterexample_confirmed': True,
                   'rank_generic': rank_G, 'rank_static_map': rank_S,
                   'rank_joint_sk_map': rank_SK,
                   'kernel_dim_mod_identities': rank_G - rank_S,
                   'kernel_generator': 'C0 - 4*C2 + C5 (= I6 - 4*I5 + I2)',
                   'pure_kinetic_exists': False}, f, indent=1)
    print('static-kernel exact: ALL PASS')


if __name__ == '__main__':
    main()
