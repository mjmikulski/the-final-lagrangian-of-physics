"""The Lagrangian in the eigenframe, symbolically (sympy), checked against lagrangian.py.

Parametrisation: N = M eta = Lambda D Lambda^-1 with D = diag(e_0 .. e_3) and Lambda in SO(1,3), so that
d_mu N = Lambda A_mu Lambda^-1 with A_mu = [Gamma_mu, D] + d_mu D and Gamma_mu = Lambda^-1 d_mu Lambda in so(1,3).
In components (A_mu)_ab = Gamma_mu,ab (e_b - e_a) + delta_ab d_mu e_a, and F_mu nu = Lambda [A_mu, A_nu] Lambda^-1.
The Euclidean norm with delta_M is the plain Frobenius norm in the eigenframe, so
    L = 2 sum_i |[A_0, A_i]|^2 - 2 sum_{i<j} |[A_i, A_j]|^2 - sum_a (e_a - E_a)^2.
The so(1,3) element Gamma has 6 components: three boosts Gamma_{0i} (= Gamma_{i0}) and three rotations
Gamma_{ij} = -Gamma_{ji}; as a mixed tensor Gamma = eta * (antisymmetric matrix).

python symbolic.py      # prints the eigenframe Lagrangian and the numerical check
"""
import itertools
import numpy as np
import sympy as sp

ETA = sp.diag(1, -1, -1, -1)


def so13(b, w):
    """Element of so(1,3) as a mixed tensor: boosts b = (b1, b2, b3), rotations w = (w1, w2, w3)."""
    K = sp.zeros(4, 4)
    for i in range(3):
        K[0, i + 1] = b[i]
        K[i + 1, 0] = b[i]
    K[1, 2], K[2, 1] = -w[2], w[2]
    K[2, 3], K[3, 2] = -w[0], w[0]
    K[3, 1], K[1, 3] = -w[1], w[1]
    return K


def frame_lagrangian():
    """L as a polynomial in the connection components and the eigenvalues (and their derivatives)."""
    e = sp.symbols('e0:4', real=True)
    Es = sp.symbols('E0:4', real=True)
    de = [[sp.Symbol(f'de{a}_{mu}', real=True) for a in range(4)] for mu in range(4)]   # d_mu e_a
    b = [[sp.Symbol(f'b{i + 1}_{mu}', real=True) for i in range(3)] for mu in range(4)]
    w = [[sp.Symbol(f'w{i + 1}_{mu}', real=True) for i in range(3)] for mu in range(4)]
    D = sp.diag(*e)
    A = []
    for mu in range(4):
        G = so13(b[mu], w[mu])
        A.append(G * D - D * G + sp.diag(*de[mu]))
    def frob(X):
        return sum(X[i, j] ** 2 for i in range(4) for j in range(4))
    F = {(m, n): A[m] * A[n] - A[n] * A[m] for m in range(4) for n in range(m + 1, 4)}
    L = 2 * sum(frob(F[(0, i)]) for i in (1, 2, 3)) - 2 * sum(frob(F[(i, j)]) for i, j in ((1, 2), (1, 3), (2, 3))) \
        - sum((e[a] - Es[a]) ** 2 for a in range(4))
    return sp.expand(L), dict(e=e, E=Es, de=de, b=b, w=w, A=A, F=F)


def numeric_check(n=3, seed=0):
    """Compare the eigenframe Lagrangian with lagrangian.py at random points."""
    import torch
    from lagrangian import lagrangian as L_torch
    rng = np.random.default_rng(seed)
    L, s = frame_lagrangian()
    f = sp.lambdify([s['e'], s['E'], s['de'], s['b'], s['w']], L, 'numpy')
    eta = np.diag([1.0, -1, -1, -1])
    worst = 0.0
    for _ in range(n):
        ev = np.array([100.0, 1.0, 0.3, 0.1]) + rng.normal(size=4) * 0.05
        Ev = (100.0, 1.0, 0.01, 0.0)
        K = rng.normal(size=(4, 6)) * 0.3                    # Gamma_mu components (b, w) for mu = 0..3
        dev = rng.normal(size=(4, 4)) * 0.3                  # d_mu e_a
        # build Lambda = exp(theta) for a random so(1,3) theta, then M and d_mu M at a point
        th = rng.normal(size=6) * 0.3
        theta = np.array(so13(th[:3], th[3:]), dtype=float)
        from scipy.linalg import expm
        Lam = expm(theta)
        Lami = eta @ Lam.T @ eta                             # Lorentz inverse
        D = np.diag(ev)
        N = Lam @ D @ Lami
        dN = []
        for mu in range(4):
            G = np.array(so13(K[mu, :3], K[mu, 3:]), dtype=float)
            Amu = G @ D - D @ G + np.diag(dev[mu])
            dN.append(Lam @ Amu @ Lami)
        M = eta @ N                                          # covariant M = eta N
        dM = np.stack([eta @ d for d in dN])
        assert np.allclose(M, M.T) and all(np.allclose(d, d.T) for d in dM)
        val_t = float(L_torch(torch.tensor(M), torch.tensor(dM), Ev))
        val_s = float(f(ev, Ev, dev, K[:, :3], K[:, 3:]))
        worst = max(worst, abs(val_t - val_s) / max(1.0, abs(val_t)))
        print(f'  lagrangian.py {val_t:+.6e}   eigenframe {val_s:+.6e}')
    print(f'worst relative deviation {worst:.1e}')
    return worst


if __name__ == '__main__':
    L, s = frame_lagrangian()
    print('number of terms in L:', len(L.args))
    # structure: the coefficient of each monomial in the connection is a polynomial in eigenvalue differences
    b, w = s['b'], s['w']
    print('example coefficient, boost-boost term  b1_0^2 b2_1^2:', sp.factor(L.coeff(b[0][0], 2).coeff(b[1][1], 2)))
    print('example coefficient, rotation-rotation term  w3_1^2 w3_2^2:', sp.factor(L.coeff(w[1][2], 2).coeff(w[2][2], 2)))
    print('example, rotation-boost term  w3_1^2 b1_2^2:', sp.factor(L.coeff(w[1][2], 2).coeff(b[2][0], 2)))
    numeric_check()


def radial_ansatz():
    """Spherically symmetric configuration with a radial boost, in the eigenframe language.

    Frame Lambda(x) = R(n) B(chi(r)): R rotates e_z to n = x/r (charge direction v_1 = n), B boosts along n with
    rapidity chi(r) (mixing v_0 and v_1). Eigenvalues e_0(r), e_1(r), e_2 = e_3 = E_2. At the point x = r e_z:
    R = 1, R^-1 d_x R = J_y / r, R^-1 d_y R = -J_x / r, R^-1 d_z R = 0 (J_a the rotation generators), so
    Gamma_i = B^-1 (R^-1 d_i R) B + delta_iz chi' K_z with K_z the boost generator; d_i D = delta_iz D'.
    Returns the static energy density H = 2 sum_{i<j} |[A_i, A_j]|^2 with A_i = [Gamma_i, D] + d_i D, and V."""
    r, e0, e1, e2, chi, de0, de1, de2, dchi = sp.symbols('r e0 e1 e2 chi de0 de1 de2 dchi', real=True)
    E0, E1, E2 = sp.symbols('E0 E1 E2', real=True)
    Jx, Jy, Kz = so13([0, 0, 0], [1, 0, 0]), so13([0, 0, 0], [0, 1, 0]), so13([0, 0, 1], [0, 0, 0])
    S, C = sp.symbols('S C', real=True)          # sinh chi, cosh chi with C^2 = 1 + S^2
    B = sp.eye(4)
    B[0, 0] = B[3, 3] = C
    B[0, 3] = B[3, 0] = S
    Bi = ETA * B.T * ETA
    G = [Bi * (Jy / r) * B, Bi * (-Jx / r) * B, dchi * Kz]
    D = sp.diag(e0, e2, e2, e1)                  # the charge axis is e_z at the pole; the pair (x, y) stays degenerate
    dD = [sp.zeros(4, 4), sp.zeros(4, 4), sp.diag(de0, de2, de2, de1)]
    A = [G[i] * D - D * G[i] + dD[i] for i in range(3)]
    H4 = 0
    for i in range(3):
        for j in range(i + 1, 3):
            Fij = A[i] * A[j] - A[j] * A[i]
            H4 += 2 * sum(Fij[p, q] ** 2 for p in range(4) for q in range(4))
    P = sp.Poly(sp.expand(H4), C)                  # reduce powers of cosh with C^2 = 1 + S^2
    red = 0
    for (k,), coeff in P.terms():
        red += coeff * (1 + S ** 2) ** (k // 2) * C ** (k % 2)
    H4 = sp.expand(red)
    # K_u = |Q dN P_0|^2 in the eigenframe: the entries of A_i in the time column, rows a != 0
    Ku = sum(A[i][a, 0] ** 2 for i in range(3) for a in (1, 2, 3))
    P = sp.Poly(sp.expand(Ku), C)
    Ku = sp.expand(sum(coeff * (1 + S ** 2) ** (k // 2) * C ** (k % 2) for (k,), coeff in P.terms()))
    V = (e0 - E0) ** 2 + (e1 - E1) ** 2 + 2 * (e2 - E2) ** 2
    return H4, V, dict(r=r, e0=e0, e1=e1, e2=e2, chi=chi, S=S, C=C, de0=de0, de1=de1, de2=de2, dchi=dchi, E0=E0, E1=E1, E2=E2, Ku=Ku)


def check_rodrigues():
    """Finite-difference check of R^-1 d_i R at the pole for the Rodrigues rotation taking e_z to n."""
    def R(n):
        n = np.asarray(n, float); n = n / np.linalg.norm(n)
        k = np.cross([0, 0, 1], n); c = n[2]
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + K + K @ K / (1 + c)
    h = 1e-6; r0 = 2.0
    Rx = (R([h, 0, r0]) - R([-h, 0, r0])) / (2 * h)
    Ry = (R([0, h, r0]) - R([0, -h, r0])) / (2 * h)
    Jx = np.array(so13([0, 0, 0], [1, 0, 0]), float)[1:, 1:]; Jy = np.array(so13([0, 0, 0], [0, 1, 0]), float)[1:, 1:]
    print('Rodrigues derivative check: |d_x R - J_y / r| =', np.abs(Rx - Jy / r0).max(), ' |d_y R + J_x / r| =', np.abs(Ry + Jx / r0).max())


if __name__ == '__main__':
    import sys
    if '--radial' in sys.argv:
        check_rodrigues()
        H4, V, s = radial_ansatz()
        S, C = s['S'], s['C']
        print('gradient energy density of the radial ansatz (S = sinh chi, C = cosh chi), collected:')
        print(sp.collect(H4, [s['dchi'], s['de1'], s['de0']], sp.factor))
        frozen = sp.factor(H4.subs({S: 0, C: 1, s['dchi']: 0, s['de0']: 0}))
        print('chi = 0 (frozen hedgehog):', frozen)
        far = sp.factor(H4.subs({s['de0']: 0, s['de1']: 0, s['dchi']: 0}))
        print('constant eigenvalues, constant boost:', far)
        print('K_u (time-axis tilt), constant eigenvalues and boost:', sp.factor(s['Ku'].subs({s['de0']: 0, s['de1']: 0, s['dchi']: 0})))
        print('zeros in S (sinh chi):', sp.solve(sp.Eq(far, 0), S))
    else:
        L, s = frame_lagrangian()
        print('number of terms in L:', len(L.args))
        b, w = s['b'], s['w']
        print('example coefficient, boost-boost term  b1_0^2 b2_1^2:', sp.factor(L.coeff(b[0][0], 2).coeff(b[1][1], 2)))
        print('example coefficient, rotation-rotation term  w3_1^2 w3_2^2:', sp.factor(L.coeff(w[1][2], 2).coeff(w[2][2], 2)))
        print('example, rotation-boost term  w3_1^2 b1_2^2:', sp.factor(L.coeff(w[1][2], 2).coeff(b[2][0], 2)))
        numeric_check()
