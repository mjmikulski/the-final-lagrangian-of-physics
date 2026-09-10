"""With the time sector free and no frame term, the energy of the charged configuration has infimum zero
(review round 1). The family N = C - a(r) P with C = diag(E_0 + Delta, B, B, B), B = E_2 = E_3,
P = l l^T eta, l = (1, x/r) null, a smooth profile a(r) rising from 0 (r <= R/2) to
a_* = (E_1 - B)(E_0 - B)/(E_0 + E_1 - 2B) (r >= R):
  (i)  d_i N = -a' x_i/r P - a d_i P and these commute for every a, a', so F = 0 everywhere;
  (ii) the spectrum of N is (B, B) together with the roots of lam^2 - (A + B) lam + AB + a(A - B), A = E_0 + Delta,
       which are exactly (E_0, E_1) at a = a_*: outside r = R the potential vanishes too;
  (iii) inside, V <= 2 Delta^2, so E(R) <= (8 pi / 3) Delta^2 R^3 -> 0, while the charge (degree of the charge
       direction on spheres r > R) is 1.
(i) and (ii) are checked with sympy at a generic point with a, a' symbolic; E(R) is integrated numerically
for the smoothstep profile with the model's own density (lagrangian.terms, analytic Cartesian derivatives).
Writes results/screened_infimum.json.
"""
import json
import math
import numpy as np
import sympy as sp
import torch
from lagrangian import terms

torch.set_default_dtype(torch.float64)
E = (100.0, 1.0, 0.01, 0.01)
B = E[2]; Delta = E[1] - B; A = E[0] + Delta
a_star = (E[1] - B) * (E[0] - B) / (E[0] + E[1] - 2 * B)

# (i), (ii) symbolic
x, y, z = sp.symbols('x y z', real=True)
a, ap = sp.symbols('a ap', real=True)
E0s, E1s, Bs = sp.symbols('E0 E1 B', positive=True)
lam = sp.Symbol('lam')
As = E0s + E1s - Bs
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
l = sp.Matrix([1, x / r, y / r, z / r])
eta = sp.diag(1, -1, -1, -1)
P = l * l.T * eta
C = sp.diag(As, Bs, Bs, Bs)
dN = [-(ap * v / r) * P - a * P.diff(v) for v in (x, y, z)]
point = {x: 1, y: 2, z: 2}          # r = 3, all entries rational
Fmax = 0
for i in range(3):
    for j in range(i + 1, 3):
        Fij = sp.simplify((dN[i] * dN[j] - dN[j] * dN[i]).subs(point))
        Fmax = max(Fmax, max(abs(e) for e in Fij))
Np = (C - a * P).subs(point)
cp = sp.factor(Np.charpoly(lam).as_expr())
target = sp.factor((lam - Bs) ** 2 * (lam ** 2 - (As + Bs) * lam + As * Bs + a * (As - Bs)))
astar_s = (E1s - Bs) * (E0s - Bs) / (E0s + E1s - 2 * Bs)
cp_star = sp.factor(cp.subs(a, astar_s))
target_star = sp.factor((lam - E0s) * (lam - E1s) * (lam - Bs) ** 2)
ok_F = Fmax == 0
ok_cp = sp.simplify(cp - target) == 0
ok_star = sp.simplify(cp_star - target_star) == 0
print(f'F = 0 for symbolic a, a\': {ok_F}; characteristic polynomial (lam-B)^2 (lam^2 - (A+B) lam + AB + a(A-B)): {ok_cp}; '
      f'at a = a_*: (lam-E0)(lam-E1)(lam-B)^2: {ok_star}')

# (iii) numeric energy of the family for several core sizes R
def smoothstep(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)

def field(xv, R):
    rr = np.linalg.norm(xv); n = xv / rr
    t = (rr - R / 2) / (R / 2)
    s = smoothstep(t); ds = (6 * t * (1 - t) / (R / 2)) if 0 < t < 1 else 0.0
    av, dav = a_star * s, a_star * ds
    ETA = np.diag([1.0, -1, -1, -1])
    lv = np.concatenate([[1.0], n])
    Pm = np.outer(lv, lv) @ ETA
    N = np.diag([A, B, B, B]) - av * Pm
    dn = (np.eye(3) - np.outer(n, n)) / rr
    dN = np.zeros((4, 4, 4))
    for i in range(3):
        dl = np.concatenate([[0.0], dn[i]])
        dP = (np.outer(dl, lv) + np.outer(lv, dl)) @ ETA
        dN[1 + i] = -dav * n[i] * Pm - av * dP
    M = ETA @ N
    dM = np.einsum('ab,ibc->iac', ETA, dN)
    return M, dM

def density(xv, R):
    M, dM = field(xv, R)
    kin, pot, _ = terms(torch.tensor(M)[None], torch.tensor(dM)[None], E, frozen=False)
    return float(-kin), float(pot)

dirs = [np.array(v) / np.linalg.norm(v) for v in ([1, 2, 3], [-2, 1, 0.5], [0.3, -0.7, 2])]
rows = []
for R in (1.0, 0.3, 0.1):
    xg, wg = np.polynomial.legendre.leggauss(60)
    rs = R / 2 + (R / 2) * (xg + 1) / 2; ws = wg * (R / 4)          # the transition zone r in [R/2, R]
    Ein, worst_kin, worst_pot_out = 0.0, 0.0, 0.0
    for rr, w in zip(rs, ws):
        k, p = np.mean([density(rr * d, R) for d in dirs], axis=0)
        Ein += 4 * math.pi * rr ** 2 * p * w
        worst_kin = max(worst_kin, abs(k))
    # inside r < R/2 the field is constant: V = 2 Delta^2 there (eigenvalues (A, B, B, B))
    Ecore = 4 * math.pi / 3 * (R / 2) ** 3 * 2 * Delta ** 2
    for rr in (R * 1.5, 5.0, 50.0):
        k, p = density(rr * dirs[0], R)
        worst_kin = max(worst_kin, abs(k)); worst_pot_out = max(worst_pot_out, abs(p))
    rows.append(dict(R=R, E=Ein + Ecore, bound=8 * math.pi / 3 * Delta ** 2 * R ** 3, quartic_max=worst_kin, potential_outside_max=worst_pot_out))
    print(f'R = {R:4g}: E = {Ein + Ecore:.5f} (bound (8 pi/3) Delta^2 R^3 = {rows[-1]["bound"]:.5f}); |quartic| <= {worst_kin:.1e}; V outside <= {worst_pot_out:.1e}')
Rs = np.array([q['R'] for q in rows]); Es = np.array([q['E'] for q in rows])
expo = float(np.polyfit(np.log(Rs), np.log(Es), 1)[0])
print(f'E(R) ~ R^{expo:.2f}')
out = dict(F_zero_symbolic=bool(ok_F), charpoly_ok=bool(ok_cp), exterior_spectrum_exact=bool(ok_star), a_star=a_star, rows=rows, exponent=expo)
json.dump(out, open('results/screened_infimum.json', 'w'), indent=1)
assert ok_F and ok_cp and ok_star and abs(expo - 3) < 0.1 and all(q['E'] <= q['bound'] * 1.001 and q['quartic_max'] < 1e-12 and q['potential_outside_max'] < 1e-12 for q in rows)
print('INFIMUM OK')
