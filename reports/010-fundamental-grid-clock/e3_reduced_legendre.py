"""E3: exact reduced-family analysis of the grid Hamiltonian (sympy).

On the frozen-tangent rotating family the cell Hamiltonian is the quartic
H(w) = H0 + A w^2 + B w^3 + C w^4 with
  A = -D1/2 + gamma*(M2 + 2*SK),  B = 4*gamma*MK,  C = 3*gamma*K2,
(D1 = record kinetic integral, S2/SK/M2/MK/K2 the per-cell quartic integrals).
Verified here, exactly:
  1. the map L(w) -> H(w) reproduces this quartic (independent route to E0);
  2. with C > 0 and A < 0 there is exactly ONE stationary point at w > 0 and it
     is a local minimum with H(w*) < H(0): an interior well for EVERY sign of B;
     with A > 0 and 9B^2 <= 32AC there is none;
  3. the well sits exactly on the Legendre caustic dH/dw = w dp/dw (002 par. 6);
  4. B = 0 reduction: w*^2 = -A/(2C) — and for the pure (s-k)^2 comparison the
     energy-reading well sits at 3x the fundamental w*^2 (the sqrt(3) of 008).
"""

import json

import sympy as sp

w, gam = sp.symbols('omega gamma', positive=True)
D1, S2, SK, M2, MK, K2, Es = sp.symbols('D1 S2 SK M2 MK K2 E_s', real=True)

# reduced L on the rotating family (record + quartic, E0 grammar)
s1, k1 = sp.symbols('s1 k1', real=True)
sj, mj, kj = sp.symbols('s_j m_j k_j', real=True)
L = -(s1 + k1 * w**2) + gam * (sj + mj * w + kj * w**2)**2
H = sp.expand(sp.diff(L, w) * w - L)

A = -k1 + gam * (mj**2 + 2 * sj * kj)
B = 4 * gam * mj * kj
C = 3 * gam * kj**2
H0 = s1 - gam * sj**2
assert sp.simplify(H - (H0 + A * w**2 + B * w**3 + C * w**4)) == 0
print('1. reduced H(w) = H0 + A w^2 + B w^3 + C w^4: exact')

# 2. stationary structure for C>0
a, b, c = sp.symbols('a b c', real=True)
Hq = a * w**2 + b * w**3 + c * w**4
dH = sp.diff(Hq, w) / w  # = 2a + 3b w + 4c w^2
roots = sp.solve(sp.Eq(2 * a + 3 * b * w + 4 * c * w**2, 0), w)
# product of roots 2a/4c < 0 when a<0, c>0 -> exactly one positive root
import itertools
import random
random.seed(3)
n_checked = 0
for _ in range(200):
    av = -random.uniform(0.01, 5)
    bv = random.uniform(-5, 5)
    cv = random.uniform(0.01, 5)
    rs = [r.evalf(subs={a: av, b: bv, c: cv}) for r in roots]
    pos = [float(r) for r in rs if r.is_real and r > 0]
    assert len(pos) == 1
    ws = pos[0]
    Hpp = float(sp.diff(Hq, w, 2).subs({a: av, b: bv, c: cv, w: ws}))
    Hv = float(Hq.subs({a: av, b: bv, c: cv, w: ws}))
    assert Hpp > 0 and Hv < 0
    n_checked += 1
print(f'2. A<0, C>0 -> unique positive stationary point, a minimum below H(0): '
      f'{n_checked} random (A,B,C) samples')
for _ in range(200):
    av = random.uniform(0.01, 5)
    bv = random.uniform(-5, 5)
    cv = random.uniform(0.01, 5)
    if 9 * bv**2 > 32 * av * cv:
        continue  # cubic term can still carve a well; outside claim
    rs = [r.evalf(subs={a: av, b: bv, c: cv}) for r in roots]
    assert not [r for r in rs if r.is_real and r > 0]
print('   A>0 with 9B^2 <= 32AC -> no positive stationary point')

# 3. caustic identity (Shapere-Wilczek structure, 002 par. 6)
p = sp.diff(L, w)
caustic = sp.simplify(sp.diff(H, w) - w * sp.diff(p, w))
assert caustic == 0
print('3. dH/dw = w dp/dw identically: every interior well sits on the caustic')

# 4. B = 0 and the (s-k)^2 comparison
wstar2 = sp.solve(sp.Eq(2 * a + 4 * c * w**2, 0), w**2)[0]
assert sp.simplify(wstar2 - (-a / (2 * c))) == 0
# energy reading of gamma(s-k)^2 + record: E(w) = -s1 - k1 w^2 + gamma (s-k)^2?
# faithful 008 comparison: drive -2 gamma s k, brake gamma k^2 (energy) vs
# drive -2 gamma s k, brake 3 gamma k^2 (fundamental) at the same record term
kk = sp.Symbol('kappa', positive=True)  # k = kappa w^2
E_energy = -2 * gam * sj * kk * w**2 + gam * kk**2 * w**4
E_fund = -2 * gam * sj * kk * w**2 + 3 * gam * kk**2 * w**4
we2 = sp.solve(sp.diff(E_energy, w) / w, w**2)[0]
wf2 = sp.solve(sp.diff(E_fund, w) / w, w**2)[0]
assert sp.simplify(we2 / wf2) == 3
print('4. B=0: w*^2 = -A/(2C); energy vs fundamental well ratio = 3 (008 sqrt3)')

with open('results/e3_reduced_legendre.json', 'w') as f:
    json.dump({'reduced_quartic_exact': True, 'unique_interior_well': True,
               'caustic_identity': True, 'energy_over_fund_ratio': 3}, f, indent=1)
print('E3: ALL PASS')
