"""Symbolic check of the halo formulas: cost and inertia of an internal tilt of the uniaxial exterior."""
import json
import sympy as sp

th, ph, r = sp.symbols('theta phi r', positive=True)
d, dp = sp.symbols('delta deltap', real=True)          # tilt angle and its radial derivative
D = sp.Symbol('Delta', positive=True)

# hedgehog direction, then an internal rotation by the angle delta(r) about the x axis
n0 = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
Rx = sp.Matrix([[1, 0, 0], [0, sp.cos(d), -sp.sin(d)], [0, sp.sin(d), sp.cos(d)]])
n = Rx * n0

# derivatives in the spherical frame: d_r has the delta' piece, the angular ones do not
n_r = sp.diff(n, d) * dp
n_th = sp.diff(n, th) / r
n_ph = sp.diff(n, ph) / (r * sp.sin(th))
B = [ (n.T * sp.Matrix(a).cross(sp.Matrix(b)))[0] for a, b in ((n_th, n_ph), (n_ph, n_r), (n_r, n_th)) ]
dens = 4 * D ** 4 * sum(b ** 2 for b in B)
dens = sp.simplify(sp.expand_trig(sp.simplify(dens)))
# integrate over the sphere at fixed r, keeping the volume element r^2 sin(th)
tot = sp.integrate(sp.integrate(sp.simplify(dens * r ** 2 * sp.sin(th)), (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
tot = sp.simplify(tot)
print('energy per unit radius (cost of the tilt plus the Coulomb term):')
print(sp.factor(sp.expand(tot)))
print('  coefficient of deltap^2 :', sp.simplify(sp.expand(tot).coeff(dp, 2)), ' vs 32 pi/3 Delta^4 =', sp.nsimplify(sp.Rational(32, 3) * sp.pi))

# inertia: rigid rotation about z of the tilted configuration; velocity of n is Omega * (z x n) minus the
# co-rotating piece, which for the internal tilt leaves Omega * delta * (y_hat x n) at leading order
om = sp.Symbol('Omega', positive=True)
ndot = om * d * sp.Matrix([0, 1, 0]).cross(n)
Bt = [ (n.T * sp.Matrix(ndot).cross(sp.Matrix(a)))[0] for a in (n_th, n_ph) ]
kin = 4 * D ** 4 * sum(b ** 2 for b in Bt)
tot_k = sp.integrate(sp.integrate(sp.simplify(kin * r ** 2 * sp.sin(th)), (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
tot_k = sp.simplify(sp.expand(tot_k))
print('\nkinetic energy per unit radius at leading order in the tilt:')
print(sp.factor(tot_k))
print('  inertia coefficient (2 x kinetic / Omega^2):', sp.simplify(2 * tot_k / om ** 2 / d ** 2),
      ' vs 64 pi/3 Delta^4 =', sp.nsimplify(sp.Rational(64, 3) * sp.pi))
c_cost = sp.simplify(sp.expand(tot).coeff(dp, 2) / D ** 4)
c_inertia = sp.simplify(2 * tot_k / om ** 2 / d ** 2 / D ** 4)
coulomb = sp.simplify(sp.expand(tot).subs(dp, 0) / D ** 4)
json.dump(dict(energy_per_unit_radius=str(sp.factor(sp.expand(tot))), cost_coefficient_over_Delta4=str(c_cost),
               cost_coefficient_numeric=float(c_cost), inertia_coefficient_over_Delta4=str(c_inertia),
               inertia_coefficient_numeric=float(c_inertia), coulomb_term_over_Delta4=str(coulomb),
               cost_is_32pi_over_3=bool(sp.simplify(c_cost - sp.Rational(32, 3) * sp.pi) == 0),
               inertia_is_64pi_over_3=bool(sp.simplify(c_inertia - sp.Rational(64, 3) * sp.pi) == 0)),
          open('results/halo_check.json', 'w'), indent=1)
print('HALO_CHECK OK' if sp.simplify(c_cost - sp.Rational(32, 3) * sp.pi) == 0 and sp.simplify(c_inertia - sp.Rational(64, 3) * sp.pi) == 0 else 'HALO_CHECK MISMATCH')
