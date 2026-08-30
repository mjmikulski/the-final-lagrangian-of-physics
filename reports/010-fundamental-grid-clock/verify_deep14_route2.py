"""Route 2 for the deep-bracket record (closes the limitation named in
this report: "an independent route-2 energy verification on them
remains open"). From-scratch numpy re-implementation of the C10 cell
Hamiltonian of the docstring of lattice_grid_defs.py,

  E_cell(M; om) = e_static(M, eta) - 2 H^3 sum k1(om)
                  + gamma H^3 sum(-s^2 + m^2 + 2 s k + 4 m k + 3 k^2),

evaluated on the persisted deep-bracket fields deep14_om*.npz with the
persisted frozen tangent a0_e4_frozen.npz, and compared against the
committed per-rung energies of e5_deep_bracket.json (final level).
No torch, no import of the report's stack. The C10 density is
implemented directly from its definition: the U = u u^T cap on the
two leading derivative slots and eta pairs elsewhere,
  D_C10(F) = U^{mu rho} eta^{nu sigma} <F_{mu nu}, F_{rho sigma}>_ee,
with U = (G - eta)/2 built from the same spectral formula as the
statics repair; (s, m, k) extracted per stencil from the three
evaluations at rates (0, +om, -om) and stencil-summed with the 1/2
weight before squaring.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
need = [os.path.join(R, f) for f in
        ('a0_e4_frozen.npz', 'e5_deep_bracket.json', 'pre_e4.json')]
if not all(os.path.exists(p) for p in need):
    print('verify_deep14_route2: NOT REPRODUCED HERE -- persisted '
          'artifacts absent.')
    sys.exit(0)

N, L = 32, 48.0
Hh = L / N
SG, DELTA, W1 = 8.0, 0.3, 0.000724023879
C_P = [SG ** p + 1.0 + DELTA ** p for p in range(1, 5)]
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
SV = np.array([-1.0, 1.0, 1.0, 1.0])

with open(os.path.join(R, 'pre_e4.json')) as f:
    pe = json.load(f)
gam = pe['cells']['C10']['gamma'] * 14.0
deep = json.load(open(os.path.join(R, 'e5_deep_bracket.json')))
a0 = np.load(os.path.join(R, 'a0_e4_frozen.npz'))['a0']


def d1(f, ax, st):
    o = np.zeros_like(f)
    lo = [slice(None)] * f.ndim
    hi = [slice(None)] * f.ndim
    sl = [slice(None)] * f.ndim
    lo[ax], hi[ax] = slice(0, -1), slice(1, None)
    sl[ax] = slice(0, -1) if st == 'fwd' else slice(1, None)
    o[tuple(sl)] = (f[tuple(hi)] - f[tuple(lo)]) / Hh
    return o


def comm(A, B):
    return A @ ETA @ B - B @ ETA @ A


def G_of(M):
    x = np.einsum('ab,...bc->...ac', ETA, M)
    I4 = np.broadcast_to(np.eye(4), M.shape)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def F_blocks(A, V):
    """F_{mu nu} matrices for mu<nu with the frozen time leg V:
    returns dict {(mu,nu): (4,4) field}; F antisymmetric in (mu,nu)."""
    F = {}
    for i in range(3):
        F[(0, i + 1)] = comm(V, A[i])
        for j in range(i + 1, 3):
            F[(i + 1, j + 1)] = comm(A[i], A[j])
    return F


def dens_I1_eta(F):
    """Full eta contraction of F.F (class C3): sum over mu<nu with
    eta^mm eta^nn weights, matrix slots eta-eta; the 1/2*4 convention
    of the stack per (mu<nu) term."""
    tot = 0.0
    for (m, n), Fm in F.items():
        w = SV[m] * SV[n]
        tot = tot + w * 0.5 * 4.0 * np.einsum(
            '...ab,ac,bd,...cd->...', Fm, ETA, ETA, Fm)
    return tot


def dens_C10(F, U):
    """U-cap on the two leading derivative slots, eta on the rest:
    D = sum_{n,n'} eta^nn' <B_n, B_n'>_ee with B_n = sum_m U^{m m'}
    (as a per-site 4x4 matrix acting on the mu index pair) applied to
    F_{m n}. With U = u u^T this is |u^m F_{m n}|^2 summed with eta
    weights; implemented via the cap matrix directly."""
    # build F as full antisymmetric array in (mu, nu) for clarity
    tot = 0.0
    for n in range(4):
        for npr in range(4):
            # B_n = sum_m U[m, m'] F_{m', n}  contracted on the cap pair;
            # with the diagonal-eta trick the cap is a plain (4,4) field
            pass
    # direct: T_{n n'} = U^{m r} <F_{m n}, F_{r n'}>_ee, D = eta^{n n'} T
    def Fget(m, n):
        if m == n:
            return None
        if m < n:
            return F[(m, n)], 1.0
        return F[(n, m)], -1.0
    for n in range(4):
        for npr in range(4):
            if n == npr:
                w_out = SV[n]
            else:
                continue   # eta pair on (nu, sigma) is diagonal
            acc = 0.0
            done = False
            for m in range(4):
                for r in range(4):
                    Fa = Fget(m, n)
                    Fb = Fget(r, npr)
                    if Fa is None or Fb is None:
                        continue
                    Fm, sa = Fa
                    Fr, sb = Fb
                    # cap indices raised: U^{m r} = eta^mm eta^rr U_{m r}
                    pref = SV[m] * SV[r]
                    inner = np.einsum('...ab,ac,bd,...cd->...',
                                      Fm, ETA, ETA, Fr)
                    acc = acc + sa * sb * pref * Ucap[..., m, r] * inner
            tot = tot + w_out * 1.0 * acc
    return tot


rows = []
worst = 0.0
for om in (0.0, 0.1, 0.15, 0.2, 0.28):
    tag = f'deep14_om{str(om).replace(".", "")}.npz'
    M = np.load(os.path.join(R, tag))['M']
    G = G_of(M)
    Ucap = (G - ETA) / 2.0
    # statics: e_static(eta) = H^3 (sum s_I1_eta + W1 * v4)
    s_stat, k1 = 0.0, 0.0
    sC, mC, kC = 0.0, 0.0, 0.0
    for st in ('fwd', 'bwd'):
        A = [d1(M, ax, st) for ax in range(3)]
        F0 = F_blocks(A, np.zeros_like(M))
        Fp = F_blocks(A, om * a0)
        Fm_ = F_blocks(A, -om * a0)
        s_stat = s_stat + 0.5 * dens_I1_eta(
            {k: v for k, v in F0.items() if k[0] != 0})
        k1 = k1 + 0.5 * (dens_I1_eta(Fp) - dens_I1_eta(F0))
        D0 = 0.5 * dens_C10(F0, Ucap)
        Dp = 0.5 * dens_C10(Fp, Ucap)
        Dm = 0.5 * dens_C10(Fm_, Ucap)
        sC = sC + D0
        mC = mC + (Dp - Dm) / 2.0
        kC = kC + (Dp + Dm) / 2.0 - D0
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        v4 = v4 + (np.einsum('...kk->...', P) - C_P[p]) ** 2
    E_stat = Hh ** 3 * (2.0 * s_stat.sum() + W1 * v4.sum())
    quart = (-sC ** 2 + mC ** 2 + 2 * sC * kC + 4 * mC * kC
             + 3 * kC ** 2)
    base = E_stat - 2.0 * Hh ** 3 * k1.sum()
    myq = gam * Hh ** 3 * quart.sum()
    E = base + myq
    ref = deep['rungs'][str(om)]['levels'][-1]
    if os.environ.get('DIAG'):
        iq = ref - base
        myq_s = gam * Hh ** 3 * (-sC ** 2).sum()
        print(f'  DIAG om {om}: base {base:.6f} implied_q {iq:+.4e} '
              f'my_q {myq:+.4e} ratio '
              f'{iq / myq if myq else 0:.4f} | kin-only implied '
              f'{iq - myq_s:+.4e} mine {myq - myq_s:+.4e} r '
              f'{(iq - myq_s) / (myq - myq_s) if myq != myq_s else 0:.4f}')
    rel = abs(E - ref) / abs(ref)
    worst = max(worst, rel)
    rows.append({'omega': om, 'E_route2': E, 'E_committed': ref,
                 'rel': rel})
    print(f'om {om}: route2 {E:.9f} vs committed {ref:.9f} '
          f'(rel {rel:.2e})', flush=True)

print(f'worst relative difference: {worst:.2e}')
json.dump({'gamma': gam, 'rows': rows, 'worst_rel': worst},
          open(os.path.join(R, 'route2_deep14.json'), 'w'), indent=1)
assert worst < 1e-9, 'route-2 must reproduce the committed energies'
print('ROUTE-2 ENERGIES MATCH the committed deep-bracket record.')
