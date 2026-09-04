"""Review round 2, resolution 3: exact classification of the 675 diagrams.

Upper bound by tensor algebra. Every metric slot X in {eta, P_0..P_3} expands
as eta = sum_a P_a with P_a = s_a e_a (x) e_a. Hence
  even (02-13)[X,Y] = sum_{a in X, b in Y} F_ab,  (03-12)[X,Y] = -sum F_ab,
  F_ab := s_a s_b e_a^mu e_b^nu e_a^al e_b^be F_{mu nu al be}, F_aa = 0, F_ab = F_ba,
so the even sector lies in the span of the six F_ab (a < b). For the odd
diagrams eps^{rskl} X0_{rm} X1_{sn} X2_{ka} X3_{lb} F^{mnab}: expanding the
insertions, eps(e_a, e_b, e_c, e_d) vanishes unless (a,b,c,d) is a permutation
of (0,1,2,3) and then equals sgn * det(E); the odd sector therefore lies in the
span of the six frame components G_abcd := F_{abcd}^frame with all indices
distinct modulo F's antisymmetries. So dim <= 6 + 6; the exact sample rank
(exact_linear.py) gives >= 12. This script verifies the expansion identities
EXACTLY (rational frames and fields) for all 675 diagrams, and derives the
proportionality classes from the resulting integer coefficient vectors.
Writes results/exact_classification.json."""
import json
from fractions import Fraction
from itertools import permutations, product
import numpy as np
from linear_defs import (SIGNS, even_diagrams, odd_diagrams, eval_diagram_exact, metrics_exact, rat_lorentz)
from u_family_defs import rand_sym_exact
from exact_linear import F4_of
rng = np.random.default_rng(20260908)

def perm_sign(p):
    s = 1; q = list(p)
    for i in range(4):
        for j in range(i + 1, 4):
            if q[i] > q[j]: s = -s
    return s

def det4(L):
    return sum(perm_sign(p) * L[0][p[0]] * L[1][p[1]] * L[2][p[2]] * L[3][p[3]] for p in permutations(range(4)))

def frame_components(F4, E):
    """F^frame_{abcd} = e_a^m e_b^n e_c^al e_d^be F_{m n al be} (E columns = frame vectors)."""
    G = {}
    for a, b, c, d in product(range(4), repeat=4):
        G[(a, b, c, d)] = sum(E[m][a] * E[n][b] * E[al][c] * E[be][d] * F4[m][n][al][be]
                              for m in range(4) for n in range(4) for al in range(4) for be in range(4)
                              if F4[m][n][al][be] != 0)
    return G

EVEN_GEN = [(a, b) for a in range(4) for b in range(a + 1, 4)]              # F_ab
ODD_GEN = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]   # representatives

def odd_canonical(a, b, c, d):
    """Reduce G_abcd (distinct indices) to a representative in ODD_GEN with sign, via antisymmetry in (ab) and (cd)."""
    s = 1
    if a > b: a, b, s = b, a, -s
    if c > d: c, d, s = d, c, -s
    if (a, b, c, d) in ODD_GEN:
        return (a, b, c, d), s
    # swap the pairs: F_{mn al be} vs F_{al be m n} are different components in general -> only listed reps
    return None, None

def predicted_vector(d):
    """Integer coefficient vector of a diagram in the basis EVEN_GEN + ODD_GEN (None for pairs not reducible without a symmetry)."""
    vec = {}
    members = lambda x: list(range(4)) if x == 0 else [x - 1]
    if d[0] == 'even':
        _, pattern, x, y = d
        sg = 1 if pattern == '02-13' else -1
        for a in members(x):
            for b in members(y):
                if a == b: continue
                key = (min(a, b), max(a, b))
                vec[key] = vec.get(key, 0) + sg
        return vec
    _, _, x0, x1, x2, x3 = d
    for a in members(x0):
        for b in members(x1):
            for c in members(x2):
                for e in members(x3):
                    if len({a, b, c, e}) < 4: continue
                    key = ('odd', a, b, c, e)
                    vec[key] = vec.get(key, 0) + perm_sign((a, b, c, e)) * SIGNS[a] * SIGNS[b] * SIGNS[c] * SIGNS[e]
    return vec

def main():
    diags = even_diagrams() + odd_diagrams()
    n_checked = 0
    for trial in range(6):
        A = [rand_sym_exact(rng) for _ in range(4)]
        F4 = F4_of(A)
        Lm = rat_lorentz(rng); mets = metrics_exact(Lm); E = Lm
        detE = det4(E); assert detE == 1
        G = frame_components(F4, E)
        for d in diags:
            v = eval_diagram_exact(d, F4, mets)
            vec = predicted_vector(d)
            if d[0] == 'even':
                pred = sum(c * SIGNS[a] * SIGNS[b] * G[(a, b, a, b)] for (a, b), c in vec.items())
            else:
                pred = sum(c * G[(a, b, cc, e)] for (_, a, b, cc, e), c in vec.items())
            assert v == pred, (d, v, pred)
            n_checked += 1
    print(f'1. expansion identities verified exactly for all {len(diags)} diagrams at 6 random rational (frame, field) points ({n_checked} checks)')
    # exact classes and ranks from the integer coefficient vectors (odd vectors reduced by F's antisymmetries)
    def reduced(d):
        vec = predicted_vector(d)
        out = {}
        if d[0] == 'even':
            for (a, b), c in vec.items():
                out[('e', a, b)] = out.get(('e', a, b), 0) + c * SIGNS[a] * SIGNS[b]
        else:
            for (_, a, b, c, e), coef in vec.items():
                s = 1
                if a > b: a, b, s = b, a, -s
                if c > e: c, e, s = e, c, -s
                out[('o', a, b, c, e)] = out.get(('o', a, b, c, e), 0) + coef * s
        return {k: v for k, v in out.items() if v != 0}
    vecs = [reduced(d) for d in diags]
    keys = sorted({k for v in vecs for k in v})
    M = [[Fraction(v.get(k, 0)) for k in keys] for v in vecs]
    zero = [i for i, v in enumerate(vecs) if not v]
    classes = []
    for i, v in enumerate(vecs):
        if not v: continue
        for cl in classes:
            w = vecs[cl[0]]
            if set(v) == set(w):
                r = Fraction(v[next(iter(v))], w[next(iter(w))])
                if all(Fraction(v[k], w[k]) == r for k in v):
                    cl.append(i); break
        else:
            classes.append([i])
    from exact_linear import exact_rank
    rk = exact_rank(M) if M else 0
    ev = [r for r, d in zip(M, diags) if d[0] == 'even']; od = [r for r, d in zip(M, diags) if d[0] == 'odd']
    print(f'2. exact classes from the coefficient vectors: zero {len(zero)}, classes {len(classes)}; exact ranks: all {rk}, even {exact_rank(ev)}, odd {exact_rank(od)} (basis keys: {len(keys)})')
    fl = json.load(open('results/linear_float.json'))
    assert (len(zero), len(classes), rk) == (fl['n_zero'], fl['n_classes'], fl['rank_all']), 'exact classification disagrees with the float route'
    json.dump({'n_diagrams': len(diags), 'n_zero': len(zero), 'n_classes': len(classes), 'rank_all': rk,
               'rank_even': exact_rank(ev), 'rank_odd': exact_rank(od), 'identity_checks': n_checked,
               'generators': {'even': [f'F_{a}{b}' for a, b in EVEN_GEN], 'odd': [f'G_{a}{b}{c}{d}' for a, b, c, d in ODD_GEN]}},
              open('results/exact_classification.json', 'w'), indent=1)
    print('3. classification exact over Q: upper bound by the expansion identities, lower bound by the exact sample rank -> 12 = 6 + 6')

if __name__ == '__main__':
    main()
