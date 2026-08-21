"""Route 2 (sympy, exact): the structural claims as polynomial/differential
identities.

1. For a GENERAL 3x3 coefficient matrix W (columns v_i, no symmetry):
   A_i = v_i e0^T + e0 v_i^T  =>  F_ij = A_i eta A_j - A_j eta A_i is the
   purely spatial wedge -(v_i v_j^T - v_j v_i^T); consequently
   N1 = N2 = N3 = 0 and all four one-eps pseudoscalars vanish identically
   (symbolic zero after expansion).
2. For SYMMETRIC V (the ansatz case: V is the Hessian of a potential):
   I2 = I1, I5 = I4, I3 = I1/2, I6 = 4 I4 - I1, and the closed forms
   e1 = 2[(tr G)^2 - tr G^2] (G = V^2), e4 = tr[(V^2 - trV V)^2].
3. Eigenvalue identities (a, b, c symbolic):
   e1 - e4 = (ab-bc)^2 + (bc-ca)^2 + (ca-ab)^2,
   4 e4 - e1 = 4 (ab+bc+ca)^2.
4. Virial identity for a single radial dressing (f(x) symbolic):
   with lam_r = f + x f', lam_t = f:
   x^2 (3 e1 - 4 e4) = -4 d/dx [x^3 f^4]
   => integral(3 e1 - 4 e4) = 0 whenever x^3 f^4 -> 0 at both ends
   => S1/S4 = 4/3 exactly on every admissible radial profile.
"""
import itertools

import sympy as sp

# --- 1. general W: wedge form, N's and pseudoscalars vanish ----------------
W = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"w{a}{i}"))
eta = sp.diag(-1, 1, 1, 1)
e0 = sp.Matrix([1, 0, 0, 0])


def a_mat(i):
    v4 = sp.Matrix([0, W[0, i], W[1, i], W[2, i]])
    return v4 * e0.T + e0 * v4.T


A = [a_mat(i) for i in range(3)]
F = {}
for i in range(3):
    for j in range(3):
        F[(i, j)] = A[i] * eta * A[j] - A[j] * eta * A[i]

wedge_ok = True
for i in range(3):
    for j in range(3):
        Wij = sp.zeros(4, 4)
        for a in range(3):
            for b in range(3):
                Wij[a + 1, b + 1] = -(W[a, i] * W[b, j] - W[a, j] * W[b, i])
        wedge_ok &= sp.simplify(F[(i, j)] - Wij) == sp.zeros(4, 4)
print(f"1. F_ij = -(v_i ^ v_j), purely spatial: {bool(wedge_ok)}")
assert wedge_ok


def F4(i, j, a, b):
    """F_{mu nu alpha beta} with all-lower indices; derivative indices
    spatial only (i, j in 1..3 mapped from 0..2), zero otherwise."""
    return F[(i, j)][a, b]


def contract_pairs(pairs, sym_V=None):
    """Complete eta-contraction of F x F given slot pairings (report 001
    slot convention: 0-3 first factor, 4-7 second)."""
    total = sp.Integer(0)
    rng = range(1, 4)  # derivative indices spatial (slots 0,1,4,5)
    for m in rng:
        for n in rng:
            for a in range(4):
                for b in range(4):
                    for p in rng:
                        for q in rng:
                            for c in range(4):
                                for d in range(4):
                                    idx = [m, n, a, b, p, q, c, d]
                                    w = sp.Integer(1)
                                    ok = True
                                    for (s, t) in pairs:
                                        if idx[s] != idx[t]:
                                            ok = False
                                            break
                                        w *= eta[idx[s], idx[s]]
                                    if not ok:
                                        continue
                                    total += (w * F4(m - 1, n - 1, a, b)
                                              * F4(p - 1, q - 1, c, d))
    return sp.expand(total)


I_REPS = {
    "I1": [(0, 4), (1, 5), (2, 6), (3, 7)],
    "I2": [(0, 6), (1, 7), (2, 4), (3, 5)],
    "I3": [(0, 4), (1, 6), (2, 5), (3, 7)],
    "I4": [(0, 2), (4, 6), (1, 5), (3, 7)],
    "I5": [(0, 2), (4, 6), (1, 7), (3, 5)],
    "I6": [(0, 2), (1, 3), (4, 6), (5, 7)],
}
I = {k: contract_pairs(p) for k, p in I_REPS.items()}

N1 = sp.expand(I["I3"] - (I["I1"] + I["I2"]) / 4)
N2 = sp.expand((I["I1"] - I["I2"]) / 4 - I["I4"] + I["I5"])
N3 = sp.expand(I["I1"] - 4 * I["I4"] + I["I6"])
print(f"   N1 = N2 = N3 = 0 identically: "
      f"{N1 == 0 and N2 == 0 and N3 == 0}")
assert N1 == 0 and N2 == 0 and N3 == 0

# one-eps pseudoscalars: every diagram needs a time index on F -> 0.
# Symbolic check on the four named representatives of report 005.
EPS = {}
for perm in itertools.permutations(range(4)):
    mat = sp.zeros(4, 4)
    s, pl = 1, list(perm)
    for i in range(4):
        j = pl.index(min(pl[i:]), i)
        if j != i:
            pl[i], pl[j] = pl[j], pl[i]
            s = -s
    EPS[perm] = s


def eps_contract(eps_slots, pairs):
    total = sp.Integer(0)
    rng_d = range(1, 4)
    for m in rng_d:
        for n in rng_d:
            for a in range(4):
                for b in range(4):
                    for p in rng_d:
                        for q in rng_d:
                            for c in range(4):
                                for d in range(4):
                                    idx = [m, n, a, b, p, q, c, d]
                                    key = tuple(idx[s] for s in eps_slots)
                                    if len(set(key)) < 4:
                                        continue
                                    w = sp.Integer(EPS[key])
                                    ok = True
                                    for (s, t) in pairs:
                                        if idx[s] != idx[t]:
                                            ok = False
                                            break
                                        w *= eta[idx[s], idx[s]]
                                    if not ok:
                                        continue
                                    total += (w * F4(m - 1, n - 1, a, b)
                                              * F4(p - 1, q - 1, c, d))
    return sp.expand(total)


P_named = {
    "P_dd": ((0, 1, 4, 5), [(2, 6), (3, 7)]),
    "P_mm": ((2, 3, 6, 7), [(0, 4), (1, 5)]),
    "P_dm": ((0, 1, 6, 7), [(2, 4), (3, 5)]),
    "P_cp": ((0, 1, 2, 3), [(4, 6), (5, 7)]),
}
pz = {k: eps_contract(*v) for k, v in P_named.items()}
print(f"   one-eps pseudoscalars vanish identically: "
      f"{all(v == 0 for v in pz.values())}")
assert all(v == 0 for v in pz.values())

# --- 2. symmetric V: collapse + closed forms -------------------------------
subs_sym = {W[a, i]: W[i, a] for a in range(3) for i in range(a + 1, 3)}
Is = {k: sp.expand(v.subs(subs_sym)) for k, v in I.items()}
Vs = W.subs(subs_sym)
G = Vs * Vs
e1_closed = sp.expand(2 * (G.trace() ** 2 - (G * G).trace()))
Phi = G - Vs.trace() * Vs
e4_closed = sp.expand((Phi * Phi).trace())
col = {
    "I2 - I1": Is["I2"] - Is["I1"],
    "I5 - I4": Is["I5"] - Is["I4"],
    "I3 - I1/2": sp.expand(Is["I3"] - Is["I1"] / 2),
    "I6 - (4I4 - I1)": sp.expand(Is["I6"] - 4 * Is["I4"] + Is["I1"]),
    "I1 - e1_closed": sp.expand(Is["I1"] - e1_closed),
    "I4 - e4_closed": sp.expand(Is["I4"] - e4_closed),
}
print("2. symmetric-V collapse identities: "
      f"{all(v == 0 for v in col.values())}")
assert all(v == 0 for v in col.values())

# --- 3. eigenvalue identities ----------------------------------------------
a, b, c = sp.symbols("a b c")
e1e = 4 * (a**2 * b**2 + b**2 * c**2 + c**2 * a**2)
e4e = (a * (b + c)) ** 2 + (b * (c + a)) ** 2 + (c * (a + b)) ** 2
lo = sp.expand(e1e - e4e - ((a * b - b * c) ** 2 + (b * c - c * a) ** 2
                            + (c * a - a * b) ** 2))
hi = sp.expand(4 * e4e - e1e - 4 * (a * b + b * c + c * a) ** 2)
print(f"3. e1-e4 and 4e4-e1 sum-of-squares identities: {lo == 0 and hi == 0}")
assert lo == 0 and hi == 0

# --- 4. virial identity -----------------------------------------------------
x = sp.Symbol("x", positive=True)
f = sp.Function("f")(x)
lam_r, lam_t = f + x * sp.diff(f, x), f
e1r = 4 * lam_t**2 * (2 * lam_r**2 + lam_t**2)
e4r = 4 * lam_r**2 * lam_t**2 + 2 * lam_t**2 * (lam_r + lam_t) ** 2
virial = sp.simplify(x**2 * (3 * e1r - 4 * e4r)
                     + 4 * sp.diff(x**3 * f**4, x))
print(f"4. x^2(3e1 - 4e4) + 4 d/dx[x^3 f^4] = 0: {virial == 0}")
assert virial == 0

print("\nALL SYMBOLIC CHECKS PASS")
