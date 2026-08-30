"""Lattice port of the u-family grid cells on the report-004 stack.

Reuses 004's lattice.py verbatim (runpy): sym stencils, pinned shell, embedded
electron seed, eta statics, generator catalog, envelope tangent. The u-caps are
realized differentiably through the working Lagrange Euclideanizer:
U = u (x) u = (G_of(M) - eta)/2 — exactly the object the 004/008 stack already
uses, so the lattice family is the working-metric realization of E1's exact
family (deviation of U from the exact eigen-projector is measured and recorded).

Cell Hamiltonian (E0 theorem, record normalization L = -u_eta-form - V):
  E_cell(M; om) = e_static(M,'eta') - 2 H^3 sum k1_acc(om)
                  + gamma H^3 sum [ -s^2 + (m)^2 + 2 s k + 4 m k + 3 k^2 ]
with (s, m, k) the per-site velocity split of the j-class density at Mdot=om*a0
(m and k carry their omega powers), stencil-averaged before squaring (008 conv).
"""

import os

import runpy

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PUB = os.path.join(_HERE, '..', '004-lattice-clock')
_L = runpy.run_path(os.path.join(_PUB, 'lattice.py'), run_name='not_main')
field, e_static, relax = _L['field'], _L['e_static'], _L['relax']
a0_of, gen_catalog = _L['a0_of'], _L['gen_catalog']
d1, comm, G_of, sym4 = _L['d1'], _L['comm'], _L['G_of'], _L['sym4']
seed_embedded, offblock = _L['seed_embedded'], _L['offblock']
H, DT, DEV, ETA = _L['H'], _L['DT'], _L['DEV'], _L['ETA']
N, FREE, M_VAC = _L['N'], _L['FREE'], _L['M_VAC']
W1, C_P = _L['W1'], _L['C_P']

RUNS = os.path.join(_HERE, 'results')
os.makedirs(RUNS, exist_ok=True)

REPS = {
    'C0': ((), ((0, 2), (1, 3), (4, 6), (5, 7))),
    'C1': ((), ((0, 2), (1, 4), (3, 6), (5, 7))),
    'C2': ((), ((0, 2), (1, 6), (3, 4), (5, 7))),
    'C3': ((), ((0, 4), (1, 5), (2, 6), (3, 7))),
    'C4': ((), ((0, 4), (1, 6), (2, 5), (3, 7))),
    'C5': ((), ((0, 6), (1, 7), (2, 4), (3, 5))),
    'C6': ((0, 2), ((1, 3), (4, 6), (5, 7))),
    'C7': ((0, 2), ((1, 4), (3, 6), (5, 7))),
    'C8': ((0, 2), ((1, 6), (3, 4), (5, 7))),
    'C9': ((0, 4), ((1, 2), (3, 6), (5, 7))),
    'C10': ((0, 4), ((1, 5), (2, 6), (3, 7))),
    'C11': ((0, 4), ((1, 6), (2, 5), (3, 7))),
    'C12': ((0, 6), ((1, 2), (3, 4), (5, 7))),
    'C13': ((0, 6), ((1, 4), (2, 5), (3, 7))),
    'C14': ((0, 6), ((1, 7), (2, 4), (3, 5))),
    'C15': ((2, 6), ((0, 3), (1, 4), (5, 7))),
    'C16': ((2, 6), ((0, 4), (1, 5), (3, 7))),
    'C17': ((2, 6), ((0, 4), (1, 7), (3, 5))),
    'C18': ((0, 2, 4, 6), ((1, 3), (5, 7))),
    'C19': ((0, 2, 4, 6), ((1, 5), (3, 7))),
    'C20': ((0, 2, 4, 6), ((1, 7), (3, 5))),
}
SLOT = 'mnabrscd'  # factor1 (mu nu | al be), factor2 (rho sig | ga de)


def U_of(M):
    return (G_of(M) - ETA) / 2.0


SV = torch.tensor([-1.0, 1, 1, 1], dtype=DT, device=DEV)  # eta diagonal


def diagram_einsum(caps, pairs):
    """Sign-trick einsum: eta is diagonal, so each eta pair is index alignment
    plus a 1-D sign operand; each U cap-pair is a per-site (4,4) operand."""
    letter = {}
    mids, kinds = [], []
    nxt = iter('mnabrscd')
    for (i, j) in pairs:
        c = next(nxt)
        letter[i] = letter[j] = c
        mids.append(c)
        kinds.append('sv')
    for i in range(0, len(caps), 2):
        c1, c2 = next(nxt), next(nxt)
        letter[caps[i]], letter[caps[i + 1]] = c1, c2
        mids.append(c1 + c2)
        kinds.append('U')
    f1 = ''.join(letter[s] for s in range(4))
    f2 = ''.join(letter[s] for s in range(4, 8))
    lhs = '...' + f1 + ',' + ','.join(
        ('...' if k == 'U' else '') + s for s, k in zip(mids, kinds)) \
        + ',...' + f2
    return lhs + '->...', kinds


def F4_of(A):
    """Per-site rank-4 F tensor from the 4 per-site A matrices (A[0] may be 0)."""
    S = A[1].shape[:3]
    F = torch.zeros(*S, 4, 4, 4, 4, dtype=DT, device=A[1].device)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            f = comm(A[mu], A[nu])
            F[..., mu, nu, :, :] = f
            F[..., nu, mu, :, :] = -f
    return F


def F4_stack(A, V):
    """[3, S, 4,4,4,4] stack of F4 at velocities (0, +V, -V), sharing the
    spatial blocks."""
    Fsp = F4_of([torch.zeros_like(A[0])] + A)
    K = torch.zeros_like(Fsp)
    for i in range(3):
        f = comm(V, A[i])
        K[..., 0, i + 1, :, :] = f
        K[..., i + 1, 0, :, :] = -f
    return torch.stack([Fsp, Fsp + K, Fsp - K])


def class_density(F4, U, caps, pairs):
    lhs, kinds = diagram_einsum(caps, pairs)
    ops = [F4]
    for k in kinds:
        ops.append(U if k == 'U' else SV)
    ops.append(F4)
    return torch.einsum(lhs, *ops)


def split_densities(M, a0, om, name):
    """Stencil-averaged (s, m, k) per site for class `name`; m, k carry their
    omega powers (Mdot = om * a0)."""
    caps, pairs = REPS[name]
    U = U_of(M) if caps else None
    d3 = 0.0
    for st in ('fwd', 'bwd'):
        A = [d1(M, ax, st) for ax in range(3)]
        d3 = d3 + 0.5 * class_density(F4_stack(A, om * a0), U, caps, pairs)
    s, dp, dm = d3[0], d3[1], d3[2]
    m = (dp - dm) / 2.0
    k = (dp + dm) / 2.0 - s
    return s, m, k


def quartic_H_density(s, m, k, sign_cross=1.0):
    """Legendre image of (I_j)^2 density; sign_cross=-1 is the drive-flip
    control (flips the omega-odd and s-k cross structure)."""
    return (-s ** 2 + m ** 2 + sign_cross * (2.0 * s * k + 4.0 * m * k)
            + 3.0 * k ** 2)


def record_drive(M, a0, om):
    """-2 H^3 sum k1 for the record term (I1 kinetic, eta), plus the m1 == 0
    check value."""
    s, m, k = split_densities(M, a0, om, 'C3')
    return -2.0 * H ** 3 * k.sum(), m.abs().max()


def e_cell(M, a0, om, gamma, jname, sign_cross=1.0):
    drive, _ = record_drive(M, a0, om)
    s, m, k = split_densities(M, a0, om, jname)
    return (e_static(M, 'eta') + drive
            + gamma * H ** 3 * quartic_H_density(s, m, k, sign_cross).sum())


# ---------------- validations ----------------
def validate(M, a0):
    out = {}
    # (a) statics identity: 2 H^3 sum s_C3 + V4 == e_static(...,'eta')
    s, m, k = split_densities(M, a0, 0.3, 'C3')
    Me = M @ ETA
    P, v4 = Me, 0.0
    for p in range(4):
        if p:
            P = P @ Me
        t = torch.einsum('...kk->...', P)
        v4 = v4 + (t - C_P[p]) ** 2
    lhs = (2.0 * H ** 3 * s.sum() + H ** 3 * W1 * v4.sum()).item()
    rhs = e_static(M, 'eta').item()
    out['statics_identity_rel'] = abs(lhs - rhs) / abs(rhs)
    out['m1_max'] = m.abs().max().item()
    # (b) U vs exact eigen projector on free sites (sampled)
    U = U_of(M)
    lam, vec = torch.linalg.eig(torch.einsum('ab,...bc->...ac', ETA, M).cpu())
    lam, vec = lam.real.to(DEV), vec.real.to(DEV)
    idx = lam.argmax(dim=-1)
    u = torch.gather(vec, -1, idx[..., None, None].expand(*M.shape[:3], 4, 1))[..., 0]
    n2 = -torch.einsum('...a,ab,...b->...', u, ETA, u)
    u = u / n2.clamp_min(1e-30).sqrt()[..., None]
    Ue = torch.einsum('...a,...b->...ab', u, u)
    dev = (U - Ue).abs().amax(dim=(-1, -2))[FREE]
    out['U_vs_eigen_max'] = dev.max().item()
    out['U_vs_eigen_mean'] = dev.mean().item()
    # (c) pointwise cross-check vs the E1/E2 numpy evaluator, one random cell
    import sys
    sys.path.insert(0, _HERE)
    from u_family_defs import eval_np, F_tensor_np
    rng = np.random.default_rng(7)
    Anp = [(lambda x: (x + x.T) / 2)(rng.standard_normal((4, 4))) for _ in range(4)]
    w3 = rng.standard_normal(3) * 0.5
    unp = np.concatenate([[np.sqrt(1 + w3 @ w3)], w3])
    Fnp = F_tensor_np(Anp)
    F4t = torch.zeros(1, 1, 1, 4, 4, 4, 4, dtype=DT, device=DEV)
    F4t[0, 0, 0] = torch.tensor(Fnp, dtype=DT, device=DEV)
    Ut = torch.tensor(np.outer(unp, unp), dtype=DT, device=DEV) \
        .expand(1, 1, 1, 4, 4)
    worst = 0.0
    for nm, (caps, pairs) in REPS.items():
        ref = eval_np(Fnp, Fnp, unp, caps, pairs)
        got = class_density(F4t, Ut, caps, pairs)[0, 0, 0].item()
        worst = max(worst, abs(got - ref) / max(abs(ref), 1e-12))
    out['pointwise_crosscheck_rel'] = worst
    return out


def load_or_make_base():
    """Eta-relaxed electron profile (004 gate). Cached; M5_FRESH=1 ignores
    the cache and regenerates from the committed seed (review round 2)."""
    path = os.path.join(RUNS, 'M_eta_base.npz')
    if os.path.exists(path) and not os.environ.get('M5_FRESH'):
        return torch.tensor(np.load(path)['M'], dtype=DT, device=DEV)
    M0 = seed_embedded()
    oracle = e_static(field(M0), 'eta').item()
    ref = 9.263660060
    assert abs(oracle - ref) / ref < 1e-6, f'gate: seed oracle {oracle} != {ref}'
    print(f'gate: seed E_eta = {oracle:.9f} (matches oracle); relaxing 3000...')
    Mr, traj = relax(M0, 'eta', 3000, tag='eta')
    Mf = field(Mr)
    print(f'gate: relaxed E_eta = {e_static(Mf, "eta").item():.6f}, '
          f'offblock = {offblock(Mf):.2e}')
    np.savez_compressed(path, M=Mr.cpu().numpy())
    return Mr
