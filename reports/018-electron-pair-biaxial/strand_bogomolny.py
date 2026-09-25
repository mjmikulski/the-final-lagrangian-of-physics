"""The Bogomolny bound for the straight strand in the planar sector, and its check with sympy.

Planar sector: charge direction e_1 = z with eigenvalue 1 fixed, the transverse 2x2 block
s0 1 + b(rho) (cos 2 theta sigma_z + sin 2 theta sigma_x) with theta = k phi and s0 = beta/2 fixed. Then
d_rho n and d_phi n are b' R and 2 k b R' with [R, R'] = 2 J (J the 2x2 rotation generator, |J|^2 = 2), so

    2 |F_xy|^2 = 64 k^2 (b b' / rho)^2,       V = 2 (b - b0)^2,   b0 = beta / 2,

and with q = b^2, s = rho^2 / 2:  T = 2 pi int ds [16 k^2 q_s^2 + 2 (sqrt q - b0)^2]
                                  >= 2 pi int 8 sqrt2 k |q_s| (b0 - sqrt q) ds = (16 sqrt2 pi / 3) k b0^3,
i.e. T_B = (2 sqrt2 pi / 3) k beta^3, attained by 4 k q_s = sqrt2 (b0 - sqrt q), whose half-splitting radius is
rho_1/2 = sqrt(2 * 4 k b0 / sqrt2 * (2 ln 2 - 1)). The cubic power of beta comes from the potential being
quadratic in the eigenvalue deviations; a potential quartic in b near b = 0 (traces of powers) gives beta^4.
Output: results/strand_bogomolny.json.
"""
import json
import math
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def symbolic_check():
    """Energy density of the planar ansatz, computed from the matrices (not from the formula above)."""
    rho, phi, k, s0 = sp.symbols('rho phi k s0', positive=True)
    b = sp.Function('b')(rho)
    th = k * phi
    S = sp.Matrix([[s0 + b * sp.cos(2 * th), b * sp.sin(2 * th), 0], [b * sp.sin(2 * th), s0 - b * sp.cos(2 * th), 0], [0, 0, 1]])
    Sr, Sp = S.diff(rho), S.diff(phi)
    F = (Sr * Sp - Sp * Sr) / rho                      # F_xy = F_(rho phi) / rho
    dens = sp.simplify(2 * sum(F[i, j] ** 2 for i in range(3) for j in range(3)))
    target = 64 * k ** 2 * (b * b.diff(rho) / rho) ** 2
    return sp.simplify(dens - target) == 0


def bound(beta, k):
    return 2 * math.sqrt(2) * math.pi / 3 * k * beta ** 3


def r_half(beta, k):
    b0 = beta / 2
    s_half = (4 * k * b0 / math.sqrt(2)) * (2 * math.log(2) - 1)
    return math.sqrt(2 * s_half)


if __name__ == '__main__':
    ok = symbolic_check()
    rows = [dict(beta=b, k=k, T_bound=bound(b, k), r_half=r_half(b, k)) for k in (0.5, 1.0) for b in (0.3, 0.1, 0.03, 0.01)]
    json.dump({'density_identity': bool(ok), 'rows': rows}, open(os.path.join(HERE, 'results', 'strand_bogomolny.json'), 'w'), indent=1)
    print('density identity:', ok)
    for r in rows:
        print(r)
    assert ok
