"""E2: velocity split (s, m, k), statics filter, orbit leak orders, channel signs.

Per class representative of the u-family (plus the three 005 pseudoscalars):
  1. generic (s, m, k) nonvanishing flags with the degree<=2 guard;
  2. the 3x3-spatial statics filter (F1): vanishes / proportional-to-I1 / other;
  3. static-leak order on the canonical boost-hedgehog orbit (T1 measurement);
  4. clock-channel kinetic signs (F2/F3) on the 001 counterexample and on the
     dressed hedgehog with boost-x / rot-xy conjugation tangents;
  5. the assembled grid filter -> survivor lists.

Writes results/velocity_split.json.
"""

import json

import numpy as np
from scipy.linalg import expm

from u_family_defs import (ETA_NP, F_tensor_np, eval_np, rand_sym_np,
                           rand_u_np)

rng = np.random.default_rng(20260831)
E0 = np.array([1.0, 0, 0, 0])

KB = np.zeros((3, 4, 4))
for k in range(3):
    KB[k, 0, k + 1] = KB[k, k + 1, 0] = 1.0
JR_XY = np.zeros((4, 4))
JR_XY[1, 2], JR_XY[2, 1] = -1.0, 1.0

EPS = np.zeros((4, 4, 4, 4))
for perm, sign in [((0, 1, 2, 3), 1)]:
    pass
from itertools import permutations
for p in permutations(range(4)):
    sgn = 1
    q = list(p)
    for i in range(4):
        for j in range(i + 1, 4):
            if q[i] > q[j]:
                sgn = -sgn
    EPS[p] = sgn


def load_reps():
    with open('results/u_family_float.json') as f:
        fl = json.load(f)
    reps = {name: (tuple(v['caps']), tuple(tuple(p) for p in v['pairs']))
            for name, v in fl['representatives'].items()}
    return reps, fl


def eval_pseudo(name, F):
    Fu = np.einsum('mnab,mM,nN->MNab', F, ETA_NP, ETA_NP)
    if name == 'P_mm':
        return np.einsum('abgd,mnab,mngd->', EPS,
                         F, np.einsum('mnab,mM,nN->MNab', F, ETA_NP, ETA_NP))
    if name == 'P_dm':
        return np.einsum('mngd,mnab,abgd->', EPS, F,
                         np.einsum('mnab,aA,bB->ABmn', F, ETA_NP, ETA_NP)
                         .transpose(2, 3, 0, 1))
    if name == 'P_cp':
        chi = np.einsum('mnab,mnab->', EPS, F)
        phi = np.einsum('mnab,ma,nb->', F, ETA_NP, ETA_NP)
        return chi * phi
    raise KeyError(name)


PSEUDO = ['P_mm', 'P_dm', 'P_cp']


def evaluate(name, reps, A, u):
    if name in PSEUDO:
        return eval_pseudo(name, F_tensor_np(A))
    caps, pairs = reps[name]
    return eval_np(F_tensor_np(A), F_tensor_np(A), u, caps, pairs)


def smk(name, reps, A, u):
    def I(lam):
        Al = [lam * A[0], A[1], A[2], A[3]]
        return evaluate(name, reps, Al, u)
    i0, ip, im, i2 = I(0.0), I(1.0), I(-1.0), I(2.0)
    s, m, k = i0, (ip - im) / 2, (ip + im) / 2 - i0
    # tolerance floor: invariants are quartic in the A's, outputs may cancel to zero
    anorm = sum(np.linalg.norm(a) for a in A)
    scale = max(abs(ip), abs(im), abs(i2), 1e-300)
    assert abs(i2 - (s + 2 * m + 4 * k)) <= 1e-9 * scale + 1e-12 * anorm**4, \
        f'{name}: degree > 2'
    return s, m, k


# --- canonical boost-hedgehog orbit -----------------------------------------
def o_of(x, m_d, p=0.5):
    r = np.linalg.norm(x)
    return expm(m_d * r ** (-p) * np.einsum('i,iab->ab', x, KB))


def hedgehog_point(x, m_d, M0, h=1e-5):
    def Mfun(y):
        o = o_of(y, m_d)
        return o @ M0 @ o.T
    M = Mfun(x)
    A = [np.zeros((4, 4))]
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        A.append((Mfun(x + dp) - Mfun(x - dp)) / (2 * h))
    return M, A


def u_of_M(M):
    w, v = np.linalg.eig(ETA_NP @ M)
    for i in range(4):
        if abs(w[i].imag) > 1e-12:
            continue
        cand = v[:, i].real
        n2 = cand @ ETA_NP @ cand
        if n2 < 0:
            u = cand / np.sqrt(-n2)
            return u if u[0] > 0 else -u
    raise RuntimeError('no timelike eigenvector')


def main():
    reps, fl = load_reps()
    names = list(reps.keys()) + PSEUDO
    out = {}

    # 1) generic flags
    flags = {n: {'s': False, 'm': False, 'k': False} for n in names}
    for _ in range(60):
        A = [rand_sym_np(rng) for _ in range(4)]
        u = rand_u_np(rng)
        for n in names:
            s, m, k = smk(n, reps, A, u)
            sc = max(abs(s), abs(m), abs(k), 1e-300)
            for key, val in (('s', s), ('m', m), ('k', k)):
                if abs(val) > 1e-10 * sc:
                    flags[n][key] = True
    out['generic_flags'] = flags

    # 2) 3x3-spatial statics filter: A0 = 0, spatial-block A_i, u = e0
    sp_samples = []
    for _ in range(30):
        A = [np.zeros((4, 4))]
        for _i in range(3):
            b = rand_sym_np(rng)
            b[0, :] = 0
            b[:, 0] = 0
            A.append(b)
        sp_samples.append(A)
    spat = {n: np.array([evaluate(n, reps, A, E0) for A in sp_samples]) for n in names}
    i1v = spat['C3']
    scale = np.abs(i1v).max()
    statics = {}
    for n in names:
        v = spat[n]
        if np.abs(v).max() < 1e-12 * scale:
            statics[n] = {'kind': 'vanishes'}
        else:
            c = np.median(v / i1v)
            if np.max(np.abs(v - c * i1v)) < 1e-10 * scale:
                statics[n] = {'kind': 'prop_I1', 'ratio': float(c)}
            else:
                statics[n] = {'kind': 'other'}
    # validation: 001 spatial nullspace N1 = C4 - (C3+C5)/4 must vanish
    n1 = spat['C4'] - (spat['C3'] + spat['C5']) / 4
    assert np.abs(n1).max() < 1e-10 * scale, 'N1 spatial identity fails'
    out['statics_filter'] = statics

    # 3) orbit leak orders (canonical + rank-rich vacuum)
    m_ds = [0.0125, 0.025, 0.05, 0.1, 0.2]
    probes = [np.array([0.6, -0.3, 0.8]), np.array([-0.4, 0.9, 0.35])]
    leak = {}
    for vac_name, M0 in [('canonical', np.diag([1.0, 0, 0, 0])),
                         ('rank_rich', np.diag([-8.0, 1, 0.3, 0]))]:
        curves = {n: [] for n in names}
        for m_d in m_ds:
            acc = {n: 0.0 for n in names}
            ref = 0.0
            for x in probes:
                M, A = hedgehog_point(x, m_d, M0)
                u = u_of_M(M)
                for n in names:
                    acc[n] += abs(evaluate(n, reps, A, u))
                ref += abs(evaluate('C3', reps, A, u))
            for n in names:
                curves[n].append(acc[n] / ref)
        slopes = {}
        for n in names:
            c = np.array(curves[n])
            if c[:3].min() < 1e-13:
                slopes[n] = {'curve': c.tolist(), 'slope': None, 'note': 'below noise'}
            else:
                sl = np.polyfit(np.log(m_ds[:3]), np.log(c[:3]), 1)[0]
                slopes[n] = {'curve': c.tolist(), 'slope': float(sl)}
        leak[vac_name] = slopes
    out['orbit_leak'] = leak

    # 4) clock-channel kinetic parts
    channels = {}
    # (a) 001 counterexample
    A_cx = [np.diag([1.0, 0, 0, 0]), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    A_cx[1][0, 1] = A_cx[1][1, 0] = 1.0
    A_cx = [A_cx[0], A_cx[1], A_cx[2], A_cx[3]]
    channels['counterexample'] = {n: dict(zip(('s', 'm', 'k'),
                                  [float(x) for x in smk(n, reps, A_cx, E0)]))
                                  for n in names}
    # (b) hedgehog probe, boost-x and rot-xy conjugation tangents
    for vac_name, M0 in [('canonical', np.diag([1.0, 0, 0, 0])),
                         ('rank_rich', np.diag([-8.0, 1, 0.3, 0]))]:
        M, A = hedgehog_point(probes[0], 0.2, M0)
        u = u_of_M(M)
        for gen_name, gen in [('boost_x', KB[0]), ('rot_xy', JR_XY)]:
            A0 = gen @ M + M @ gen.T
            Ach = [A0, A[1], A[2], A[3]]
            channels[f'{vac_name}_{gen_name}'] = {
                n: dict(zip(('s', 'm', 'k'),
                            [float(x) for x in smk(n, reps, Ach, u)]))
                for n in names}
    out['channels'] = channels

    # 4b) drive-sign statistics over random static backgrounds (002-style)
    ksign = {n: 0 for n in names}
    nbg = 30
    for _ in range(nbg):
        Mbg = np.diag([-8.0, 1, 0.3, 0]) + 0.3 * rand_sym_np(rng)
        ubg = u_of_M(Mbg)
        A0 = KB[0] @ Mbg + Mbg @ KB[0].T
        Abg = [A0] + [rand_sym_np(rng) for _ in range(3)]
        for n in names:
            if smk(n, reps, Abg, ubg)[2] < 0:
                ksign[n] += 1
    out['k_negative_fraction_random_bg'] = {n: ksign[n] / nbg for n in names}

    # 5) grid filter (channel verdicts: exact counterexample primary for F2/F3;
    #    physical rank-rich orbit for the leak order)
    kscale_cx = max(abs(v['k']) for v in channels['counterexample'].values())
    kscale_rr = max(abs(v['k']) for v in channels['rank_rich_boost_x'].values())
    i_axis, j_axis = [], []
    for n in names:
        st = statics[n]
        k_cx = channels['counterexample'][n]['k']
        k_rr = channels['rank_rich_boost_x'][n]['k']
        if st['kind'] == 'prop_I1':
            alpha = -1 if st['ratio'] > 0 else 1
            if alpha * k_cx < -1e-12 * kscale_cx:
                i_axis.append({'name': n, 'alpha': alpha, 'k_cx': k_cx,
                               'statics_ratio': st['ratio']})
        lk = leak['rank_rich'][n]
        can_curve = np.array(leak['canonical'][n]['curve'])
        brake_cx = abs(k_cx) > 1e-12 * kscale_cx
        brake_rr = abs(k_rr) > 1e-12 * kscale_rr
        if (brake_cx or brake_rr) and lk['slope'] is not None and lk['slope'] > 1.5:
            j_axis.append({'name': n, 'k_cx': k_cx, 'k_rr_boost': k_rr,
                           'leak_slope_rr': lk['slope'],
                           'canonical_orbit_zero': bool(can_curve.max() < 1e-10),
                           'brake_on_counterexample': bool(brake_cx),
                           'm_flag': flags[n]['m'],
                           'k_neg_frac_bg': ksign[n] / nbg})
    out['i_axis'] = i_axis
    out['j_axis'] = j_axis

    with open('results/velocity_split.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)

    # summary print
    print('name       s m k | statics        | leak slope can/rr | k(cx)      k(boost)    k(rot)')
    for n in names:
        fl_ = flags[n]
        st = statics[n]
        stx = st['kind'] + (f"({st.get('ratio'):.3g})" if 'ratio' in st else '')
        ls = leak['canonical'][n]['slope']
        lr = leak['rank_rich'][n]['slope']
        lss = f"{ls:5.2f}" if ls is not None else '  -- '
        lrs = f"{lr:5.2f}" if lr is not None else '  -- '
        print(f"{n:10s} {int(fl_['s'])} {int(fl_['m'])} {int(fl_['k'])} | {stx:14s} | "
              f"{lss} / {lrs}     | {channels['counterexample'][n]['k']:+10.3e} "
              f"{channels['canonical_boost_x'][n]['k']:+10.3e} "
              f"{channels['canonical_rot_xy'][n]['k']:+10.3e}")
    print(f"\ni-axis ({len(i_axis)}): " + ', '.join(
        f"{d['name']}(a={d['alpha']}, k_cx={d['k_cx']:+.3g})" for d in i_axis))
    print(f"j-axis ({len(j_axis)}):")
    for d in j_axis:
        print(f"  {d['name']:6s} leak_rr p={d['leak_slope_rr']:.2f}  "
              f"canonical-orbit-zero={d['canonical_orbit_zero']!s:5s}  "
              f"brake_cx={d['brake_on_counterexample']!s:5s}  "
              f"m!=0={d['m_flag']!s:5s}  k_neg_frac_bg={d['k_neg_frac_bg']:.2f}")
    print(f"survivor cells: {len(i_axis) * len(j_axis)}")


if __name__ == '__main__':
    main()
