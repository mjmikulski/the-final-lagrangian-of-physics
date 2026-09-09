"""Symbolic check of the halo formulas: cost and inertia of an internal tilt of the uniaxial exterior.

Exterior density 4 Delta^4 sum_{i<j} (n . (d_i n x d_j n))^2 (exterior_check.py) with n = R_x(theta(r)) x-hat:
the hedgehog direction rotated internally about the x axis by an angle theta(r).

Static cost: the r^-2 of the angular derivatives cancels against the volume element, and only theta' enters.
Inertia: rigid rotation of the whole configuration about z, n_phi(x) = R_z(phi) n(R_z(-phi) x), gives the
exact velocity dn/dphi = w x n with w = z-hat - R_x(theta) z-hat (the tilt breaks the hedgehog's rotation
invariance), and the kinetic density 4 Delta^4 sum_i (n . (dn/dt x d_i n))^2 has a radial component too
(review round 1). Both the exact integrand and its small-amplitude limit are derived; the leading term is
(64 pi / 3) Delta^4 theta^2, the exact one (64 pi / 3) Delta^4 4 sin^2(theta/2) (1 + r^2 theta'^2 / 2).
"""
import json
import sympy as sp

th, ph, r = sp.symbols('theta phi r', positive=True)
d, dp = sp.symbols('delta deltap', real=True)          # tilt angle and its radial derivative
D = sp.Symbol('Delta', positive=True)

n0 = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
Rx = sp.Matrix([[1, 0, 0], [0, sp.cos(d), -sp.sin(d)], [0, sp.sin(d), sp.cos(d)]])
n = Rx * n0

# derivatives in the orthonormal spherical frame: d_r carries the delta' piece, the angular ones do not
n_r = sp.diff(n, d) * dp
n_th = sp.diff(n, th) / r
n_ph = sp.diff(n, ph) / (r * sp.sin(th))
B = [(n.T * sp.Matrix(a).cross(sp.Matrix(b)))[0] for a, b in ((n_th, n_ph), (n_ph, n_r), (n_r, n_th))]
dens = 4 * D ** 4 * sum(b ** 2 for b in B)
dens = sp.simplify(sp.expand_trig(sp.simplify(dens)))
tot = sp.simplify(sp.integrate(sp.integrate(sp.simplify(dens * r ** 2 * sp.sin(th)), (ph, 0, 2 * sp.pi)), (th, 0, sp.pi)))
print('energy per unit radius (Coulomb term plus the cost of the tilt):')
print(sp.factor(sp.expand(tot)))
c_cost = sp.simplify(sp.expand(tot).coeff(dp, 2) / D ** 4)
print('  coefficient of deltap^2 :', c_cost, ' vs 32 pi/3 =', sp.nsimplify(sp.Rational(32, 3) * sp.pi))

# inertia: exact velocity of the rigid rotation about z, w = z - R_x z
om = sp.Symbol('Omega', positive=True)
w = sp.Matrix([0, 0, 1]) - Rx * sp.Matrix([0, 0, 1])
ndot = om * w.cross(n)
Bt = [(n.T * sp.Matrix(ndot).cross(sp.Matrix(a)))[0] for a in (n_th, n_ph, n_r)]
kin = 4 * D ** 4 * sum(b ** 2 for b in Bt)
tot_k = sp.integrate(sp.integrate(sp.simplify(kin * r ** 2 * sp.sin(th)), (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
tot_k = sp.simplify(sp.expand(tot_k))
inertia = sp.simplify(2 * tot_k / om ** 2)
print('\nexact inertia per unit radius (2 x kinetic / Omega^2):')
print(sp.factor(inertia))
closed = sp.Rational(64, 3) * sp.pi * D ** 4 * 4 * sp.sin(d / 2) ** 2 * (1 + r ** 2 * dp ** 2 / 2)
print('  equals (64 pi/3) Delta^4 4 sin^2(delta/2) (1 + r^2 deltap^2/2):', sp.simplify(inertia - closed) == 0)
lead = sp.simplify(sp.series(inertia.subs(dp, 0), d, 0, 3).removeO() / d ** 2 / D ** 4)
print('  leading term at small angle and deltap = 0 :', lead, ' vs 64 pi/3 =', sp.nsimplify(sp.Rational(64, 3) * sp.pi))
# the angular-only part (the first version of this check) is the deltap = 0 limit
inertia_angular = sp.simplify(inertia.subs(dp, 0))
json.dump(dict(energy_per_unit_radius=str(sp.factor(sp.expand(tot))), cost_coefficient_over_Delta4=str(c_cost),
               cost_coefficient_numeric=float(c_cost), cost_is_32pi_over_3=bool(sp.simplify(c_cost - sp.Rational(32, 3) * sp.pi) == 0),
               inertia_exact_per_unit_radius=str(sp.factor(inertia)),
               inertia_closed_form='(64 pi/3) Delta^4 4 sin^2(delta/2) (1 + r^2 deltap^2 / 2)',
               inertia_closed_form_verified=bool(sp.simplify(inertia - closed) == 0),
               inertia_leading_coefficient_over_Delta4=str(lead), inertia_leading_numeric=float(lead),
               inertia_leading_is_64pi_over_3=bool(sp.simplify(lead - sp.Rational(64, 3) * sp.pi) == 0),
               inertia_angular_only=str(sp.factor(inertia_angular))),
          open('results/halo_check.json', 'w'), indent=1)
ok = sp.simplify(c_cost - sp.Rational(32, 3) * sp.pi) == 0 and sp.simplify(lead - sp.Rational(64, 3) * sp.pi) == 0 and sp.simplify(inertia - closed) == 0
print('HALO_CHECK OK' if ok else 'HALO_CHECK MISMATCH')
