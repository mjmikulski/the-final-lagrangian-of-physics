"""L1c': the linear sector on the rank-1 canonical orbit (report 006's ansatz
family) is inert -- exact rational check.

On M = g v v^T the only defined projectors are P_t (timelike axis, eigenvector
of eta M is eta v) and the rest R = eta - P_t (three degenerate eigenvalues).
Linear classes with the frame {eta, P_t, R}: F_{tt} = 0 (antisymmetry),
F_{tR} = 0 (a P_t cap on a matrix slot: F_{ij} = g^2 (w_j w_i^T - w_i w_j^T)
with w_i^T eta v = 0 -- the matrix-cap theorem of report 010), and
F_{RR} = phi - 2 F_{tR} + F_{tt} = phi, a null Lagrangian. Same for eps.
Writes results/orbit1_linear.json.
"""
import json
from fractions import Fraction
import numpy as np
from linear_defs import eval_diagram_exact, even_diagrams, odd_diagrams, rat_lorentz
from u_family_defs import ETA_DIAG, F_exact_from_A
from exact_linear import F4_of
from orbit_linear_exact import KB, JR, mm, tr
rng = np.random.default_rng(20260907)

def orbit1_point(g=Fraction(13, 10)):
    o = rat_lorentz(rng)
    v = [o[a][0] for a in range(4)]                       # v = o e0
    M = [[g * v[a] * v[b] for b in range(4)] for a in range(4)]
    A = [[[Fraction(0)] * 4 for _ in range(4)]]
    for _ in range(3):
        cs = [Fraction(int(rng.integers(-3, 4)), int(rng.integers(1, 4))) for _ in range(6)]
        om = [[sum(cs[k] * KB[k][a][b] for k in range(3)) + sum(cs[3 + k] * JR[k][a][b] for k in range(3)) for b in range(4)] for a in range(4)]
        Ai = mm(om, M); A.append([[Ai[a][b] + Ai[b][a] for b in range(4)] for a in range(4)])
    e = [ETA_DIAG[a] * v[a] for a in range(4)]           # eigenvector of eta M: eta v
    nrm = sum(ETA_DIAG[a] * e[a] * e[a] for a in range(4))
    Pt = [[e[m] * e[n] / nrm for n in range(4)] for m in range(4)]
    eta = [[Fraction(ETA_DIAG[a]) if a == b else Fraction(0) for b in range(4)] for a in range(4)]
    R = [[eta[m][n] - Pt[m][n] for n in range(4)] for m in range(4)]
    # sanity: (eta M) e = lambda e
    etaM = [[ETA_DIAG[a] * M[a][b] for b in range(4)] for a in range(4)]
    Me = [sum(etaM[a][b] * e[b] for b in range(4)) for a in range(4)]
    lam = Me[0] / e[0]
    assert all(Me[a] == lam * e[a] for a in range(4))
    return A, [eta, Pt, R]

def main():
    # metrics index: 0 eta, 1 Pt, 2 R (only these exist on the rank-1 orbit)
    ev = [('even', pat, x, y) for pat in ('02-13', '03-12') for x in range(3) for y in range(3)]
    od = [('odd', None, a, b, c, d) for a in range(3) for b in range(3) for c in range(3) for d in range(3)]
    zero = {str(d): True for d in ev + od}
    vals_phi = []
    for _ in range(10):
        A, mets = orbit1_point()
        F4 = F4_of(A)
        phi = eval_diagram_exact(('even', '02-13', 0, 0), F4, mets)
        for d in ev + od:
            v = eval_diagram_exact(d, F4, mets)
            if d[0] == 'even' and (1 not in d[2:]):     # no P_t cap: must equal +-phi (null)
                assert v == phi or v == -phi, (d, v, phi)
            elif v != 0:
                zero[str(d)] = False
    nz = [k for k, z in zero.items() if not z]
    nz_even = [k for k in nz if 'even' in k]
    print('rank-1 orbit: classes with a P_t cap that are NOT exactly zero:', [k for k in nz if 'even' in k and ", 1" in k])
    print('nonzero odd diagrams:', len([k for k in nz if 'odd' in k]), 'of', len(od))
    out = {'even_with_Pt_cap_all_zero': all(("1" not in k) for k in nz_even), 'nonzero_odd': [k for k in nz if 'odd' in k]}
    json.dump(out, open('results/orbit1_linear.json', 'w'), indent=1)
    print(out)

if __name__ == '__main__':
    main()
