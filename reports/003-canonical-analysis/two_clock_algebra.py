"""Independent check of the P240 two-clock fixed-J interaction algebra
(substrate-framework issue #146, comment of 2026-08-20 17:57 UTC).

For two coupled clocks with inertia matrix I = [[I0, C], [C, I0]] and
E_J = (1/4) J^T I^{-1} J:
  fixed J = (j, j):   DeltaE = -C j^2 / (2 I0 (I0 + C))   (attractive iff C > 0)
  fixed omega:        DeltaE = +2 C omega^2               (opposite leading sign)
The Legendre-dual sign flip is the standard fixed-current vs fixed-flux
behavior; the physical load rests entirely on C(r) = A/r with A > 0,
which is NOT established here (their open question 2).
"""
import sympy as sp

j, I0, C, w = sp.symbols("j I0 C omega", positive=True)
I = sp.Matrix([[I0, C], [C, I0]])

dE_J = sp.simplify(sp.Rational(1, 4) * (sp.Matrix([j, j]).T * I.inv()
                                        * sp.Matrix([j, j]))[0]
                   - j ** 2 / (2 * I0))
assert sp.simplify(dE_J + j ** 2 * C / (2 * I0 * (I0 + C))) == 0
print("fixed-J: DeltaE =", sp.factor(dE_J), " (P240 formula reproduced)")

dE_w = sp.simplify((sp.Matrix([w, w]).T * I * sp.Matrix([w, w]))[0]
                   - 2 * I0 * w ** 2)
assert sp.simplify(dE_w - 2 * C * w ** 2) == 0
print("fixed-omega: DeltaE = +2 C omega^2  (opposite sign, as they state)")
print("VERIFIED: both identities exact.")
