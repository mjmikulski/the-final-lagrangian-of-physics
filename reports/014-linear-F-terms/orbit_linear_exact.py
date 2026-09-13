"""L1c: which linear classes vanish identically on the rank-rich dressing orbit
M = o M0 o^T, M0 = diag(-8, 1, 3/10, 0) (exact rationals; eigenframe by exact
nullspaces of eta M - lambda). Writes results/orbit_linear.json."""
import json
from fractions import Fraction
import numpy as np
from linear_defs import diagram_label, eval_diagram_exact, rat_lorentz
from u_family_defs import ETA_DIAG, F_exact_from_A
from exact_linear import F4_of
rng = np.random.default_rng(20260906)
KB, JR = [], []
for k in range(3):
    m = [[Fraction(0)] * 4 for _ in range(4)]; m[0][k + 1] = m[k + 1][0] = Fraction(1); KB.append(m)
for (i, j) in ((2, 3), (3, 1), (1, 2)):
    m = [[Fraction(0)] * 4 for _ in range(4)]; m[i][j], m[j][i] = Fraction(-1), Fraction(1); JR.append(m)
H0 = [Fraction(-8), Fraction(1), Fraction(3, 10), Fraction(0)]

def mm(X, Y): return [[sum(X[a][c] * Y[c][b] for c in range(4)) for b in range(4)] for a in range(4)]
def tr(X): return [[X[b][a] for b in range(4)] for a in range(4)]

def nullvec(Mx):
    """exact right null vector of a 4x4 rational matrix of rank 3"""
    A = [r[:] for r in Mx]; piv = []; r = 0
    for c in range(4):
        p = next((i for i in range(r, 4) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; inv = 1 / A[r][c]; A[r] = [x * inv for x in A[r]]
        for i in range(4):
            if i != r and A[i][c] != 0:
                f = A[i][c]; A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
    free = [c for c in range(4) if c not in piv][0]
    v = [Fraction(0)] * 4; v[free] = Fraction(1)
    for ri, pc in enumerate(piv): v[pc] = -A[ri][free]
    return v

def frame_metrics(M):
    etaM = [[ETA_DIAG[a] * M[a][b] for b in range(4)] for a in range(4)]
    mets = [[[Fraction(ETA_DIAG[a]) if a == b else Fraction(0) for b in range(4)] for a in range(4)]]
    for lam in H0:                    # eigenvalues of eta M0 are (8,1,3/10,0)? sign: eta M0 = diag(8,1,3/10,0)
        pass
    lams = [Fraction(8), Fraction(1), Fraction(3, 10), Fraction(0)]
    for lam in lams:
        X = [[etaM[a][b] - (lam if a == b else 0) for b in range(4)] for a in range(4)]
        e = nullvec(X)
        nrm = sum(ETA_DIAG[a] * e[a] * e[a] for a in range(4))
        assert nrm != 0
        mets.append([[e[m] * e[n] / nrm for n in range(4)] for m in range(4)])
    # sanity: projectors sum to eta
    for a in range(4):
        for b in range(4):
            assert sum(mets[k][a][b] for k in range(1, 5)) == mets[0][a][b]
    return mets

def orbit_point():
    o = rat_lorentz(rng)
    M0 = [[H0[a] if a == b else Fraction(0) for b in range(4)] for a in range(4)]
    M = mm(mm(o, M0), tr(o))
    A = [[[Fraction(0)] * 4 for _ in range(4)]]
    for _ in range(3):
        cs = [Fraction(int(rng.integers(-3, 4)), int(rng.integers(1, 4))) for _ in range(6)]
        om = [[sum(cs[k] * KB[k][a][b] for k in range(3)) + sum(cs[3 + k] * JR[k][a][b] for k in range(3)) for b in range(4)] for a in range(4)]
        Ai = mm(om, M); Ai = [[Ai[a][b] + Ai[b][a] for b in range(4)] for a in range(4)]   # om M + M om^T
        A.append(Ai)
    return M, A

def main():
    fl = json.load(open('results/linear_float.json'))
    reps = {k: tuple([v['diagram'][0], v['diagram'][1]] + v['diagram'][2:]) for k, v in fl['reps'].items()}
    zero = {k: True for k in reps}
    for _ in range(12):
        M, A = orbit_point()
        mets = frame_metrics(M)
        F4 = F4_of(A)
        for k, d in reps.items():
            if eval_diagram_exact(d, F4, mets) != 0:
                zero[k] = False
    zs = [k for k, z in zero.items() if z]; nz = [k for k, z in zero.items() if not z]
    print('exactly zero on the rank-rich static orbit:', [fl['reps'][k]['label'] for k in zs])
    print('nonzero on the orbit:', len(nz), 'classes')
    json.dump({'orbit_zero': zs, 'orbit_nonzero': nz}, open('results/orbit_linear.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
