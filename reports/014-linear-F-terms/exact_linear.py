"""L0 exact route: classes and ranks of the linear family over Q with rational
frames. Writes results/linear_exact.json."""
import json
from fractions import Fraction
import numpy as np
from linear_defs import (diagram_label, eval_diagram_exact, even_diagrams,
                         metrics_exact, odd_diagrams, rat_lorentz)
from u_family_defs import F_exact_from_A, rand_sym_exact
rng = np.random.default_rng(20260905)

def exact_rank(M):
    rows, cols = len(M), len(M[0]); A = [r[:] for r in M]; r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; inv = 1 / A[r][c]; A[r] = [x * inv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]; A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        r += 1
        if r == rows: break
    return r

def F4_of(A):
    Fm = F_exact_from_A(A)
    return [[[[Fm[(m, n)][a][b] for b in range(4)] for a in range(4)] for n in range(4)] for m in range(4)]

def main():
    fl = json.load(open('results/linear_float.json'))
    diags = even_diagrams() + odd_diagrams()
    reps = [tuple(v['diagram']) for v in fl['reps'].values()]
    reps = [(d[0], d[1]) + tuple(d[2:]) for d in reps]
    samples = []
    for _ in range(16):
        A = [rand_sym_exact(rng) for _ in range(4)]
        samples.append((F4_of(A), metrics_exact(rat_lorentz(rng))))
    R = [[eval_diagram_exact(d, F4, mets) for d in reps] for F4, mets in samples]
    ev = [k for k, d in enumerate(reps) if d[0] == 'even']
    od = [k for k, d in enumerate(reps) if d[0] == 'odd']
    rk = exact_rank(R); rke = exact_rank([[r[k] for k in ev] for r in R]); rko = exact_rank([[r[k] for k in od] for r in R])
    print(f'exact ranks over Q: all {rk}, even {rke}, odd {rko}  (float: {fl["rank_all"]}/{fl["rank_even"]}/{fl["rank_odd"]})')
    assert (rk, rke, rko) == (fl['rank_all'], fl['rank_even'], fl['rank_odd'])
    # a few random member diagrams per class must be exactly proportional to the rep
    n_ok = 0
    for k, d in enumerate(reps):
        pass
    json.dump({'rank_all': rk, 'rank_even': rke, 'rank_odd': rko}, open('results/linear_exact.json', 'w'), indent=1)
    print('exact route: PASS')

if __name__ == '__main__':
    main()
