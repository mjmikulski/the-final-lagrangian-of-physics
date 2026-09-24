"""Symbolic checks (sympy) of the three algebraic facts the report rests on, with fully symbolic matrices.

The background derivatives d_i N = eta S_i (S_i generic symmetric), the perturbation X = eta S_X, and the matrix
inner product <A, B> = Tr(A g B^T g^-1) with g = diag(g0, g1, g2, g3) symbolic and positive (delta_M is
diagonal in the field's eigenframe; the identities below use only bilinearity, so the choice of frame is
immaterial). Derivative indices are contracted with eta = diag(1, -1, -1, -1).

1. Quadratic in the velocity: with d_0 N = lam * V, -F.F is a0 + a2 lam^2 exactly (no odd powers).
2. Blind to the trace: d_mu N -> d_mu N + s_mu 1 leaves every F_{mu nu} and (1 - P) d_mu N P unchanged for any
   projector P.
3. The principal symbol: the t^2 coefficient of -F.F at d_0 N = t w X, d_i N = D_i + t k_i X equals
   2 (w^2 - |k|^2) sum_i <W_i, W_i> + 2 <k.W, k.W>,  W_i = [X, D_i]  (identically in all symbols).
Output: results/symbolic.json.
"""
import json
import time
import sympy as sp

ETA = sp.diag(1, -1, -1, -1)
g = sp.symbols('g0:4', positive=True)
Gm, Gi = sp.diag(*g), sp.diag(*[1 / x for x in g])


def sym(name):
    s = sp.symbols(f'{name}0:10')
    it = iter(s)
    A = sp.zeros(4, 4)
    for a in range(4):
        for b in range(a, 4):
            A[a, b] = A[b, a] = next(it)
    return A


def ip(A, B):
    return (A * Gm * B.T * Gi).trace()


def comm(A, B):
    return A * B - B * A


def minus_FF(dN):
    """-F.F = -sum_{mu,nu} eta^mm eta^nn <F_mn, F_mn> for derivative matrices dN[0..3]."""
    s = [1, -1, -1, -1]
    tot = 0
    for m in range(4):
        for n in range(4):
            if m != n:
                F = comm(dN[m], dN[n])
                tot += -s[m] * s[n] * ip(F, F)
    return tot


if __name__ == '__main__':
    t0 = time.time()
    out = {}
    D = [ETA * sym(f's{i}_') for i in range(3)]
    X = ETA * sym('x')
    V = ETA * sym('v')
    lam, t, w = sp.symbols('lam t w')
    k = sp.symbols('k1:4')

    # 1. parity in the velocity
    L = sp.expand(minus_FF([lam * V] + D))
    poly = sp.Poly(L, lam)
    out['velocity_powers'] = sorted(int(m[0]) for m in poly.monoms())
    print('1. powers of the velocity in -F.F:', out['velocity_powers'], f'[{time.time() - t0:.0f} s]')

    # 2. trace blindness
    s_ = sp.symbols('s0:4')
    dN = [V] + D
    ok = all(sp.expand(comm(dN[m] + s_[m] * sp.eye(4), dN[n] + s_[n] * sp.eye(4)) - comm(dN[m], dN[n])) == sp.zeros(4, 4)
             for m in range(4) for n in range(m + 1, 4))
    P = sp.diag(1, 0, 0, 0)          # a projector in its own frame; conjugation does not change the argument
    ok_ku = sp.expand((sp.eye(4) - P) * (V + s_[0] * sp.eye(4)) * P - (sp.eye(4) - P) * V * P) == sp.zeros(4, 4)
    out['F_blind_to_trace'] = bool(ok)
    out['K_u_blind_to_trace'] = bool(ok_ku)
    print('2. F and K_u blind to the trace:', ok, ok_ku, f'[{time.time() - t0:.0f} s]')

    # 3. principal symbol
    L = minus_FF([t * w * X] + [D[i] + t * k[i] * X for i in range(3)])
    c2 = sp.expand(L).coeff(t, 2)
    W = [comm(X, D[i]) for i in range(3)]
    kW = sum((k[i] * W[i] for i in range(3)), sp.zeros(4, 4))
    k2 = sum(x ** 2 for x in k)
    target = 2 * (w ** 2 - k2) * sum(ip(W[i], W[i]) for i in range(3)) + 2 * ip(kW, kW)
    diff = sp.expand(c2 - target)
    out['symbol_identity'] = diff == 0
    print('3. principal symbol identity:', out['symbol_identity'], f'[{time.time() - t0:.0f} s]')
    json.dump(out, open('results/symbolic.json', 'w'), indent=1)
    assert out['velocity_powers'] == [0, 2] and out['F_blind_to_trace'] and out['K_u_blind_to_trace'] and out['symbol_identity']
