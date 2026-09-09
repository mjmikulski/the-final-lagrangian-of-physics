"""Two exact statements behind the screening, checked with sympy.

1. Proposition (null-direction hedgehog). For the asymptotic hedgehog with the time axis tilted radially,
   M_00 = E_0, M_0i = m n_i, M_ij = -(E_2 delta_ij + Delta n_i n_j), n = x/r, the operator N = eta M at
   m = Delta is N = const - Delta l l^T eta with l = (1, n) null. Then [d_i N, d_j N] = 0 and
   tr(d_i N d_j N) = 0 identically: the field strength of the model and every Lorentz-invariant scalar
   polynomial in the first derivatives vanish. Checked symbolically at a generic point, with E_0, E_2,
   Delta symbolic.
2. Far-field zero in the eigenframe. With the eigenvalues held at their vacuum values and a constant
   boost of rapidity chi along the charge axis (symbolic.radial_ansatz), the static density
   4 Delta^4 / r^4 of the frozen hedgehog vanishes exactly at sinh^2 chi = Delta^2 / ((e_0 - e_2)^2 - Delta^2),
   where the frame term K_u equals 2 sinh^2 chi (e_0 - e_2)^2 / r^2; the break-even coupling is
   c* = 2 Delta^2 ((e_0 - e_2)^2 - Delta^2) / ((e_0 - e_2)^2 r^2), i.e. 2 Delta^2 / r^2 up to O(Delta^2 / E_0^2).
Writes results/prop4_check.json.
"""
import json
import sympy as sp
from symbolic import radial_ansatz

x, y, z = sp.symbols('x y z', real=True)
E0, E2, D, m = sp.symbols('E0 E2 Delta m', positive=True)
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
n = sp.Matrix([x, y, z]) / r
eta = sp.diag(1, -1, -1, -1)
M = sp.zeros(4, 4)
M[0, 0] = E0
for i in range(3):
    M[0, i + 1] = M[i + 1, 0] = m * n[i]
    for j in range(3):
        M[i + 1, j + 1] = -(E2 * (1 if i == j else 0) + D * n[i] * n[j])
N = eta * M
dN = [N.diff(v) for v in (x, y, z)]
point = {x: sp.Rational(1, 2), y: sp.Rational(3, 5), z: sp.Rational(-7, 4)}          # generic, exact
out = {}
worst_F, worst_tr, worst_F0 = 0, 0, 0
for i in range(3):
    for j in range(i + 1, 3):
        Fij = (dN[i] * dN[j] - dN[j] * dN[i]).subs(point)
        worst_F = max(worst_F, max(abs(v) for v in [sp.simplify(e.subs(m, D)) for e in Fij]) if Fij != sp.zeros(4, 4) else 0)
        worst_F0 = max(worst_F0, max(sp.Abs(v).subs({E0: 100, E2: sp.Rational(1, 100), D: sp.Rational(99, 100)}) for v in Fij.subs(m, 0)))
    for j in range(3):
        tr = sp.simplify((dN[i] * dN[j]).trace().subs(point).subs(m, D))
        worst_tr = max(worst_tr, abs(tr))
out['F_at_m_Delta_max_entry'] = str(worst_F)
out['trace_dNdN_at_m_Delta_max'] = str(worst_tr)
out['F_at_m_0_max_entry_numeric'] = float(worst_F0)
print(f'[N, N] commutators at m = Delta: max entry {worst_F} (frozen hedgehog, same point: {float(worst_F0):.3e}); tr(dN dN) at m = Delta: {worst_tr}')
# nilpotent structure: N - const = -Delta l l^T eta with l null
l = sp.Matrix([1, n[0], n[1], n[2]])
P = (l * l.T * eta).subs(point)
out['l_null'] = str(sp.simplify((l.T * eta * l)[0]))
out['P_squared_zero'] = bool(sp.simplify(P * P) == sp.zeros(4, 4))
out['P_traceless'] = str(sp.simplify(P.trace()))
Nres = sp.simplify((N.subs(m, D) + D * l * l.T * eta).subs(point))
out['N_plus_Delta_P_is_constant_diag'] = str(Nres)
print(f'l^T eta l = {out["l_null"]}, P^2 = 0: {out["P_squared_zero"]}, tr P = {out["P_traceless"]}; N + Delta P = {Nres.tolist()}')
# eigenvalues at the tilted point, numerically: they stay near (E_0, 1, E_2, E_2)
import numpy as np
num = {E0: 100, E2: sp.Rational(1, 100), D: sp.Rational(99, 100)}
Nn = np.array(N.subs(m, D).subs(point).subs(num).evalf(20).tolist(), dtype=float)
ev = sorted(np.linalg.eigvals(Nn).real.tolist(), reverse=True)
out['eigenvalues_tilted'] = ev
out['eigenvalue_shift_from_vacuum'] = [ev[0] - 100, ev[1] - 1, ev[2] - 0.01, ev[3] - 0.01]
print('eigenvalues of N on the screened hedgehog (E0 = 100, E2 = 0.01):', [round(v, 6) for v in ev], ' shifts from (E0, 1, E2, E2):', [f'{v:.1e}' for v in out['eigenvalue_shift_from_vacuum']])

# 2. far-field zero in the eigenframe
H4, V, s = radial_ansatz()
far = sp.factor(H4.subs({s['de0']: 0, s['de1']: 0, s['de2']: 0, s['dchi']: 0}))
e0, e1, e2, S, rr = s['e0'], s['e1'], s['e2'], s['S'], s['r']
Dl = e1 - e2
S2 = Dl ** 2 / ((e0 - e2) ** 2 - Dl ** 2)
zero = sp.simplify(far.subs(S, sp.sqrt(S2)))
Ku = sp.factor(s['Ku'].subs({s['de0']: 0, s['de1']: 0, s['de2']: 0, s['dchi']: 0}))
Ku_star = sp.simplify(Ku.subs(S, sp.sqrt(S2)))
c_star = sp.simplify(far.subs(S, 0) / Ku_star)
out['farfield_frozen'] = str(sp.factor(far.subs(S, 0)))
out['farfield_at_screening_rapidity'] = str(zero)
out['Ku_general'] = str(Ku)
out['Ku_at_screening_rapidity'] = str(Ku_star)
out['c_star'] = str(c_star)
out['c_star_leading'] = str(sp.simplify(sp.series(c_star.subs({e1: e2 + D, e0: E0}), E0, sp.oo, 2).removeO()))
print('far-field density, chi = 0:', out['farfield_frozen'])
print('far-field density at sinh^2 chi = Delta^2/((e0-e2)^2 - Delta^2):', zero)
print('K_u there:', Ku_star, ';  break-even c* =', c_star, ' -> leading order', out['c_star_leading'])
assert worst_F == 0 and worst_tr == 0 and zero == 0
json.dump(out, open('results/prop4_check.json', 'w'), indent=1)
print('PROP4 OK')
