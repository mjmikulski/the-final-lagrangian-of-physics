"""E0: the grid Legendre theorem, verified symbolically (sympy, exact rationals).

Claim (prereg_hamiltonian_grid.md par. 2): every u-decorated quadratic contraction
of F x F is a polynomial of degree <= 2 in Mdot, I = s + m + k, and the canonical
Hamiltonian of L = alpha*I_i + gamma*(I_j)^2 - V is exactly

    H = alpha*(-s_i + k_i) + gamma*(-s_j^2 + m_j^2 + 2 s_j k_j + 4 m_j k_j + 3 k_j^2) + V.

Route (a): 1-dof reduced family. Route (b): full 10-symbol symmetric Mdot on random
rational backgrounds, explicit p = dL/dq, H = sum p q - L, for one representative
diagram of every structural type (all-eta aligned, mixed pairing, u on derivative
slots, u on matrix slots, u on one factor only, u mixed across factors).
"""

import json
import random
import sympy as sp

ETA = sp.diag(-1, 1, 1, 1)
random.seed(20260829)


def part_a():
    w, al, ga, V = sp.symbols('w alpha gamma V')
    si, mi, ki, sj, mj, kj = sp.symbols('s_i m_i k_i s_j m_j k_j')
    L = al * (si + mi * w + ki * w**2) + ga * (sj + mj * w + kj * w**2)**2 - V
    H = sp.expand(sp.diff(L, w) * w - L)
    Hf = sp.expand(al * (-si + ki * w**2)
                   + ga * (-sj**2 + mj**2 * w**2 + 2 * sj * kj * w**2
                           + 4 * mj * kj * w**3 + 3 * kj**2 * w**4) + V)
    assert sp.simplify(H - Hf) == 0
    return True


def rand_sym():
    m = sp.zeros(4, 4)
    for a in range(4):
        for b in range(a, 4):
            v = sp.Rational(random.randint(-3, 3), random.choice([1, 2, 3]))
            m[a, b] = v
            m[b, a] = v
    return m


def rational_u():
    # boost of e0 along a rational unit axis, rational rapidity functions
    v = sp.Rational(random.choice([1, 2]), random.choice([2, 3, 5]))
    c, s = (1 + v**2) / (1 - v**2), 2 * v / (1 - v**2)
    axes = [(sp.Rational(3, 5), sp.Rational(4, 5), 0),
            (sp.Integer(1), sp.Integer(0), sp.Integer(0)),
            (sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7))]
    n = random.choice(axes)
    u = sp.Matrix([c, s * n[0], s * n[1], s * n[2]])
    assert sp.simplify(-u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2) == -1
    return u


MD = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f'md{min(a,b)}{max(a,b)}'))
MD_SYMS = sorted({MD[a, b] for a in range(4) for b in range(4)}, key=str)


def build_F(A):
    F = {}
    for mu in range(4):
        for nu in range(4):
            F[(mu, nu)] = A[mu] * ETA * A[nu] - A[nu] * ETA * A[mu]
    return F


# diagram = (caps, pairs) over slots 0..7; factor1 slots (0,1,2,3) = (mu,nu,al,be),
# factor2 slots (4,5,6,7); deriv pairs (0,1) and (4,5), matrix pairs (2,3), (6,7)
def eval_diagram(F, u, caps, pairs):
    idx = [None] * 8
    total = sp.Integer(0)
    nsum = len(caps) + len(pairs)

    def rec(pos, weight):
        nonlocal total
        if pos == nsum:
            f1 = F[(idx[0], idx[1])][idx[2], idx[3]]
            f2 = F[(idx[4], idx[5])][idx[6], idx[7]]
            total += weight * f1 * f2
            return
        if pos < len(caps):
            for a in range(4):
                idx[caps[pos]] = a
                rec(pos + 1, weight * u[a])
        else:
            sa, sb = pairs[pos - len(caps)]
            for a in range(4):
                idx[sa] = a
                idx[sb] = a
                rec(pos + 1, weight * ETA[a, a])
    rec(0, sp.Integer(1))
    return sp.expand(total)


def degree_split(expr):
    parts = {0: sp.Integer(0), 1: sp.Integer(0), 2: sp.Integer(0)}
    p = sp.Poly(expr, *MD_SYMS)
    for mono, coef in p.terms():
        d = sum(mono)
        assert d <= 2, f'degree {d} > 2 in Mdot'
        parts[d] += coef * sp.prod(s**e for s, e in zip(MD_SYMS, mono))
    return parts[0], parts[1], parts[2]


def euler_H(L):
    return sp.expand(sum(q * sp.diff(L, q) for q in MD_SYMS) - L)


DIAGRAMS = {
    'I1_all_eta': ([], [(0, 4), (1, 5), (2, 6), (3, 7)]),
    'I3_mixed': ([], [(0, 4), (2, 5), (1, 6), (3, 7)]),
    'u_deriv_both': ([0, 4], [(1, 5), (2, 6), (3, 7)]),
    'u_matrix_both': ([2, 6], [(0, 4), (1, 5), (3, 7)]),
    'u_one_factor': ([0, 2], [(1, 4), (3, 6), (5, 7)]),
    'u_mixed_slots': ([0, 6], [(1, 4), (2, 5), (3, 7)]),
}


def part_b():
    out = {}
    for name, (caps, pairs) in DIAGRAMS.items():
        A = {0: MD, 1: rand_sym(), 2: rand_sym(), 3: rand_sym()}
        u = rational_u()
        F = build_F(A)
        I = eval_diagram(F, u, caps, pairs)
        s, m, k = degree_split(I)

        assert euler_H(I) - sp.expand(-s + k) == 0, f'{name}: quadratic H law fails'
        Hsq = euler_H(sp.expand(I**2))
        Hsq_f = sp.expand(-s**2 + m**2 + 2 * s * k + 4 * m * k + 3 * k**2)
        assert sp.expand(Hsq - Hsq_f) == 0, f'{name}: quartic H law fails'

        out[name] = {'m_nonzero': m != 0, 's_nonzero': s != 0, 'k_nonzero': k != 0}
        print(f'{name:16s} s!=0: {s != 0!s:5s} m!=0: {m != 0!s:5s} k!=0: {k != 0!s:5s}  PASS')
    return out


def part_c():
    # full grid Lagrangian, two distinct diagrams at once
    al, ga, V = sp.symbols('alpha gamma V')
    A = {0: MD, 1: rand_sym(), 2: rand_sym(), 3: rand_sym()}
    u = rational_u()
    F = build_F(A)
    Ii = eval_diagram(F, u, *DIAGRAMS['I1_all_eta'])
    Ij = eval_diagram(F, u, *DIAGRAMS['u_deriv_both'])
    si, mi, ki = degree_split(Ii)
    sj, mj, kj = degree_split(Ij)
    L = al * Ii + ga * sp.expand(Ij**2) - V
    H = sp.expand(sum(q * sp.diff(L, q) for q in MD_SYMS) - L)
    Hf = sp.expand(al * (-si + ki)
                   + ga * (-sj**2 + mj**2 + 2 * sj * kj + 4 * mj * kj + 3 * kj**2) + V)
    assert sp.expand(H - Hf) == 0, 'full grid H law fails'
    return True


if __name__ == '__main__':
    a = part_a()
    print('part A (1-dof reduced family): PASS')
    b = part_b()
    c = part_c()
    print('part C (full grid L = alpha I_i + gamma I_j^2 - V): PASS')
    with open('results/legendre_theorem_check.json', 'w') as f:
        json.dump({'part_a': a, 'part_b': b, 'part_c': c}, f, indent=1)
    print('E0 theorem checks: ALL PASS')
