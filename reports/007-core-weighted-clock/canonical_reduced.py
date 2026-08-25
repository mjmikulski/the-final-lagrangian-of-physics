"""Reduced clock sector, exact (sympy): the canonical structure of a
quartic-in-velocity theory with a configuration-dependent kinetic
coefficient K = K1 + c*K4.

Derived and asserted:
  L(omega) = K/2 omega^2 + B/4 omega^4
  pi = K omega + B omega^3,  H = K/2 omega^2 + 3B/4 omega^4
  Legendre degeneracy (fold) at omega^2 = -K/(3B), fold momentum
  pi_c = 2 sqrt(3) K sqrt(-K/B) / 9
  interior clock: omega*^2 = -K/(3B), E(omega*) = -K^2/(12B),
  existing iff K < 0 and B > 0  (report 003's branched structure)
  clock-on condition in c: K1 + c*K4 < 0.
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
K, B, om = sp.symbols("K B omega", real=True)
L = K * om ** 2 / 2 + B * om ** 4 / 4
p_of_om = sp.diff(L, om)
H = sp.simplify(p_of_om * om - L)
assert sp.simplify(H - (K * om ** 2 / 2 + 3 * B * om ** 4 / 4)) == 0

dpi = sp.diff(p_of_om, om)
om_deg = sp.solve(sp.Eq(dpi, 0), om)
assert any(sp.simplify(o ** 2 + K / (3 * B)) == 0 for o in om_deg)

crit = sp.solve(sp.Eq(sp.diff(H, om), 0), om)
om_star2 = -K / (3 * B)
E_star = sp.simplify(H.subs(om, sp.sqrt(om_star2)))
assert sp.simplify(E_star + K ** 2 / (12 * B)) == 0
assert any(o == 0 for o in crit)

pi_c = sp.simplify(p_of_om.subs(om, sp.sqrt(om_deg[1] ** 2)))
assert sp.simplify(pi_c - 2 * sp.sqrt(3) * K * sp.sqrt(-K / B) / 9) == 0

# second derivative of E at omega*: positive (a genuine minimum) when
# K < 0, B > 0
d2E = sp.diff(H, om, 2).subs(om ** 2, om_star2)
d2E = sp.simplify(d2E)
assert sp.simplify(d2E + 2 * K) == 0        # d2E = -2K > 0 for K < 0

print("reduced clock sector, exact:")
print(f"  pi = {p_of_om}")
print(f"  H  = {H}")
print(f"  fold: omega^2 = -K/(3B), pi_c = {pi_c}")
print(f"  clock: omega*^2 = -K/(3B), E* = {E_star}, E''(omega*) = {d2E}")
print("  existence iff K < 0 and B > 0; with K = K1 + c*K4 the clock-on"
      " condition is c*K4 < -K1 (localized by any core-supported c)")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"H": str(H), "pi_c": str(pi_c), "E_star": str(E_star),
           "d2E_at_star": str(d2E)},
          open(os.path.join(HERE, "results", "canonical_reduced.json"),
               "w"), indent=1)
print("ALL EXACT CHECKS PASS; written results/canonical_reduced.json")
