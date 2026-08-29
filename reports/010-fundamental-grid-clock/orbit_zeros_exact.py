"""Exact-rational verification of the rank-1-orbit zeros (E2 follow-up).

On the canonical orbit M = v v^T (v unit timelike, v = o e0), static tangents
are A_i = w_i v^T + v w_i^T with w_i = omega_i v, omega_i in so(1,3); the
timelike eigen-axis of eta M is exactly u = -eta v (rational when v is), and
the sign drops from every even-cap diagram. Evaluates all 21 class
representatives on 20 exact random orbit points and classifies exact zeros.

Expected from the float route: C15-C20 zero (structural <= 1e-19); C6-C8 and
C12-C14 zero within FD noise -- decided exactly here; C9-C11 nonzero (the
p = 2 leak carriers). Writes results/orbit_zeros_exact.json.
"""

import json
from fractions import Fraction

import numpy as np

from u_family_defs import ETA_DIAG, eval_exact, rand_u_exact

rng = np.random.default_rng(20260902)

# so(1,3) basis in the array convention of 006 (boosts symmetric, rotations
# antisymmetric); rational combinations
KB = []
for k in range(3):
    m = [[Fraction(0)] * 4 for _ in range(4)]
    m[0][k + 1] = m[k + 1][0] = Fraction(1)
    KB.append(m)
JR = []
for (i, j) in ((2, 3), (3, 1), (1, 2)):
    m = [[Fraction(0)] * 4 for _ in range(4)]
    m[i][j], m[j][i] = Fraction(-1), Fraction(1)
    JR.append(m)


def rand_so13(rng):
    cs = [Fraction(int(rng.integers(-3, 4)), int(rng.integers(1, 4)))
          for _ in range(6)]
    m = [[sum(cs[k] * KB[k][a][b] for k in range(3))
          + sum(cs[3 + k] * JR[k][a][b] for k in range(3))
          for b in range(4)] for a in range(4)]
    return m


def mv(m, x):
    return [sum(m[a][b] * x[b] for b in range(4)) for a in range(4)]


def orbit_point(rng):
    v = rand_u_exact(rng)
    M = [[v[a] * v[b] for b in range(4)] for a in range(4)]
    A = [[[Fraction(0)] * 4 for _ in range(4)]]
    for _ in range(3):
        om = rand_so13(rng)
        w = mv(om, v)
        A.append([[w[a] * v[b] + v[a] * w[b] for b in range(4)]
                  for a in range(4)])
    u = [-ETA_DIAG[a] * v[a] for a in range(4)]   # exact eigen-axis of eta M
    # sanity: (eta M) u = -u exactly, u.eta.u = -1
    etaM = [[ETA_DIAG[a] * M[a][b] for b in range(4)] for a in range(4)]
    assert mv(etaM, u) == [-x for x in u]
    assert sum(ETA_DIAG[a] * u[a] * u[a] for a in range(4)) == -1
    return M, A, u


def main():
    with open('results/u_family_float.json') as f:
        fl = json.load(f)
    reps = {n: (tuple(v['caps']), tuple(tuple(p) for p in v['pairs']))
            for n, v in fl['representatives'].items()}
    from u_family_defs import F_exact_from_A
    zero = {n: True for n in reps}
    for _ in range(20):
        M, A, u = orbit_point(rng)
        F = F_exact_from_A(A)
        for n, (caps, pairs) in reps.items():
            if eval_exact(F, u, caps, pairs) != 0:
                zero[n] = False
    zeros = sorted([n for n, z in zero.items() if z],
                   key=lambda s: int(s[1:]))
    nonz = sorted([n for n, z in zero.items() if not z],
                  key=lambda s: int(s[1:]))
    print('exact zeros on the rank-1 static orbit:', zeros)
    print('nonzero on the orbit:', nonz)
    exp_zero = {'C6', 'C7', 'C8', 'C12', 'C13', 'C14',
                'C15', 'C16', 'C17', 'C18', 'C19', 'C20'}
    assert set(zeros) >= exp_zero, f'expected zeros missing: {exp_zero - set(zeros)}'
    assert {'C9', 'C10', 'C11'} <= set(nonz), 'leak carriers unexpectedly zero'
    with open('results/orbit_zeros_exact.json', 'w') as f:
        json.dump({'exact_zeros': zeros, 'nonzero': nonz,
                   'n_samples': 20}, f, indent=1)
    print('orbit-zeros exact: PASS')


if __name__ == '__main__':
    main()
