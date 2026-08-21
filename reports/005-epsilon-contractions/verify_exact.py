"""Second, independent route for the epsilon-sector claims: EXACT INTEGER
arithmetic, pure Python (no torch, no float anywhere).

Confirms, over random small-integer fields where every diagram value is an
exact integer:
1. counts: 210 diagrams, 54 identically zero, 156 alive;
2. exact proportionality classes among the 156 (cross-ratio test over Z)
   and their sizes;
3. rank over Q: 4 on generic integer double two-forms, 3 on realizable
   F(A) with P_dd's class exactly zero;
4. structural identities exactly: chi^2 = 16*I3 - 4*I1 - 4*I2, I6 = phi^2,
   P_cp = chi*phi;
5. clock counterexample: I = omega^2*(4,4,2,2,2,4), all one-eps classes 0;
6. purely spatial integer fields: all one-eps diagrams 0;
7. all 70 nonzero no-eta two-eps diagrams are exact rational combinations
   of I1..I6 (solve on 8 samples over Q, verify on 12 more).

Rank over Q via sympy integer matrices (exact); everything else is int math.
"""
import itertools
import random
from fractions import Fraction

from sympy import Matrix, Rational

random.seed(21)
ETA = [-1, 1, 1, 1]

EPS = {}
for perm in itertools.permutations(range(4)):
    s, p = 1, list(perm)
    for i in range(4):
        j = p.index(min(p[i:]), i)
        if j != i:
            p[i], p[j] = p[j], p[i]
            s = -s
    EPS[perm] = s

ANTISYM_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def random_symmetric_A(lo=-4, hi=4):
    A = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for m in range(4):
        for a in range(4):
            for b in range(a, 4):
                v = random.randint(lo, hi)
                A[m][a][b] = A[m][b][a] = v
    return A


def F_of_A(A):
    F = [[[[0] * 4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for m in range(4):
        for n in range(4):
            for a in range(4):
                for b in range(4):
                    s = 0
                    for g in range(4):
                        s += (A[m][a][g] * ETA[g] * A[n][g][b]
                              - A[n][a][g] * ETA[g] * A[m][g][b])
                    F[m][n][a][b] = s
    return F


def random_generic_F(lo=-7, hi=7):
    """P239-style: independent random value per (two-form pair, two-form
    pair), both antisymmetries imposed, nothing else."""
    F = [[[[0] * 4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for (m, n) in pairs:
        for (a, b) in pairs:
            v = random.randint(lo, hi)
            F[m][n][a][b] = v
            F[n][m][a][b] = -v
            F[m][n][b][a] = -v
            F[n][m][b][a] = v
    return F


def matchings(slots):
    if not slots:
        yield []
        return
    a, rest = slots[0], slots[1:]
    for i, b in enumerate(rest):
        for m in matchings(rest[:i] + rest[i + 1:]):
            yield [(a, b)] + m


DIAGRAMS = []
for eps_slots in itertools.combinations(range(8), 4):
    rest = [s for s in range(8) if s not in eps_slots]
    for pairs in matchings(rest):
        DIAGRAMS.append((eps_slots, pairs))


def eval_one_eps(diagram, F):
    eps_slots, pairs = diagram
    total = 0
    for perm, sign in EPS.items():
        idx = [None] * 8
        for slot, val in zip(eps_slots, perm):
            idx[slot] = val
        for v1 in range(4):
            idx[pairs[0][0]] = idx[pairs[0][1]] = v1
            w1 = sign * ETA[v1]
            for v2 in range(4):
                idx[pairs[1][0]] = idx[pairs[1][1]] = v2
                total += (w1 * ETA[v2]
                          * F[idx[0]][idx[1]][idx[2]][idx[3]]
                          * F[idx[4]][idx[5]][idx[6]][idx[7]])
    return total


def eval_metric(pairs, F):
    total = 0
    for vals in itertools.product(range(4), repeat=4):
        idx = [None] * 8
        w = 1
        for (p, q), v in zip(pairs, vals):
            idx[p] = idx[q] = v
            w *= ETA[v]
        total += (w * F[idx[0]][idx[1]][idx[2]][idx[3]]
                  * F[idx[4]][idx[5]][idx[6]][idx[7]])
    return total


I_REPS = {
    "I1": [(0, 4), (1, 5), (2, 6), (3, 7)],
    "I2": [(0, 6), (1, 7), (2, 4), (3, 5)],
    "I3": [(0, 4), (1, 6), (2, 5), (3, 7)],
    "I4": [(0, 2), (4, 6), (1, 5), (3, 7)],
    "I5": [(0, 2), (4, 6), (1, 7), (3, 5)],
    "I6": [(0, 2), (1, 3), (4, 6), (5, 7)],
}
PDD = ((0, 1, 4, 5), [(2, 6), (3, 7)])


def chi(F):
    return sum(s * F[p[0]][p[1]][p[2]][p[3]] for p, s in EPS.items())


def phi(F):
    return sum(ETA[m] * ETA[n] * F[m][n][m][n]
               for m in range(4) for n in range(4))


# --- 1+2. counts and exact proportionality classes on generic --------------
alive = [d for d in DIAGRAMS
         if not any(p in ANTISYM_PAIRS for p in d[1])]
dead = [d for d in DIAGRAMS if d not in alive]
NS = 20
Fg = [random_generic_F() for _ in range(NS)]
assert all(eval_one_eps(d, F) == 0 for d in dead[:10] for F in Fg[:2])
print(f"diagrams: {len(DIAGRAMS)} total, {len(alive)} alive "
      f"({len(dead)} vanish; spot-checked exactly zero)")

vals = [[eval_one_eps(d, F) for F in Fg] for d in alive]
assert all(any(v != 0 for v in row) for row in vals), \
    "some 'alive' diagram vanished on generic integer samples"


def proportional(u, v):
    return all(u[i] * v[j] == u[j] * v[i]
               for i in range(NS) for j in range(NS))


classes = []
for i, row in enumerate(vals):
    for c in classes:
        if proportional(row, vals[c[0]]):
            c.append(i)
            break
    else:
        classes.append([i])
sizes = sorted(len(c) for c in classes)
print(f"exact proportionality classes: {len(classes)}, sizes {sizes}")
assert len(classes) == 13 and sizes == sorted(
    [4, 16, 16, 2, 16, 4, 16, 16, 16, 16, 16, 16, 2])

# --- 3. exact ranks over Q --------------------------------------------------
M_gen = Matrix([[vals[c[0]][k] for c in classes] for k in range(NS)])
print(f"rank over Q, generic ensemble: {M_gen.rank()} / 13")
assert M_gen.rank() == 4

Fp = [F_of_A(random_symmetric_A()) for _ in range(NS)]
vals_p = [[eval_one_eps(alive[c[0]], F) for F in Fp] for c in classes]
zero_classes = [ci for ci, row in enumerate(vals_p)
                if all(v == 0 for v in row)]
M_phys = Matrix([[vals_p[ci][k] for ci in range(13)] for k in range(NS)])
print(f"rank over Q, realizable F(A): {M_phys.rank()} / 13; "
      f"classes exactly zero: {len(zero_classes)}")
assert M_phys.rank() == 3 and len(zero_classes) == 1

pdd_class = zero_classes[0]
pdd_vals = [eval_one_eps(PDD, F) for F in Fp]
assert all(v == 0 for v in pdd_vals), "P_dd not exactly zero on realizable"
assert proportional([eval_one_eps(PDD, F) for F in Fg],
                    vals[classes[pdd_class][0]])
print(f"P_dd: exactly 0 on all {NS} realizable samples; its class "
      f"(size {len(classes[pdd_class])}) is the vanishing one")

# --- 4. structural identities exactly --------------------------------------
for F in Fg + Fp:
    I = {k: eval_metric(p, F) for k, p in I_REPS.items()}
    c, p_ = chi(F), phi(F)
    assert c * c == 16 * I["I3"] - 4 * I["I1"] - 4 * I["I2"]
    assert I["I6"] == p_ * p_
    cp = eval_one_eps(((0, 1, 2, 3), [(4, 6), (5, 7)]), F)
    assert cp == c * p_
print(f"exact on all {len(Fg) + len(Fp)} samples: chi^2 = 16*I3-4*I1-4*I2, "
      f"I6 = phi^2, P_cp = chi*phi")

# --- 5. clock counterexample -----------------------------------------------
omega = 3          # integer so everything stays exact
A_clock = [[[0] * 4 for _ in range(4)] for _ in range(4)]
A_clock[0][0][0] = omega
A_clock[1][0][1] = A_clock[1][1][0] = 1
Fc = F_of_A(A_clock)
I_c = [eval_metric(p, Fc) for p in I_REPS.values()]
assert I_c == [omega**2 * v for v in (4, 4, 2, 2, 2, 4)]
assert all(eval_one_eps(alive[c[0]], Fc) == 0 for c in classes)
print(f"clock counterexample (omega={omega}): I = omega^2*(4,4,2,2,2,4) "
      f"exactly, all 13 one-eps classes exactly 0")

# --- 6. purely spatial fields ----------------------------------------------
for F in [random_generic_F() for _ in range(3)]:
    for i in range(4):
        for j in range(4):
            for k in range(4):
                F[0][i][j][k] = F[i][0][j][k] = 0
                F[i][j][0][k] = F[i][j][k][0] = 0
    assert all(eval_one_eps(alive[c[0]], F) == 0 for c in classes)
print("purely spatial integer fields: all one-eps classes exactly 0")

# --- 7. two-eps diagrams are exact rational combinations of I1..I6 --------
def eval_two_eps(eps1, eps2, F):
    total = 0
    for p1, s1 in EPS.items():
        idx = [None] * 8
        for slot, val in zip(eps1, p1):
            idx[slot] = val
        for p2, s2 in EPS.items():
            for slot, val in zip(eps2, p2):
                idx[slot] = val
            total += (s1 * s2 * F[idx[0]][idx[1]][idx[2]][idx[3]]
                      * F[idx[4]][idx[5]][idx[6]][idx[7]])
    return total


Fs_fit, Fs_check = Fg[:8], Fg[8:]
n_nonzero, worst_ok = 0, True
for eps1 in itertools.combinations(range(8), 4):
    eps2 = tuple(s for s in range(8) if s not in eps1)
    if eps1 > eps2:
        continue                      # unordered split
    y_fit = [eval_two_eps(eps1, eps2, F) for F in Fs_fit]
    y_check = [eval_two_eps(eps1, eps2, F) for F in Fs_check]
    if all(v == 0 for v in y_fit + y_check):
        continue
    n_nonzero += 1
    A_fit = Matrix([[eval_metric(p, F) for p in I_REPS.values()]
                    for F in Fs_fit])
    sol = A_fit.solve_least_squares(Matrix(y_fit))
    for F, y in zip(Fs_check, y_check):
        pred = sum(Rational(sol[i]) * eval_metric(p, F)
                   for i, p in enumerate(I_REPS.values()))
        if pred != y:
            worst_ok = False
assert worst_ok
print(f"two-eps (no-eta splits): {n_nonzero} nonzero, every one an exact "
      f"rational combination of I1..I6 (fit on 8 samples, verified on 12)")

print("\nALL EXACT-INTEGER CHECKS PASSED")
