"""E1 float route: enumerate the u-decorated family, find classes and ranks.

Outputs results/u_family_float.json: diagram/zero/class counts, class sizes,
representatives, value ranks (generic + realizable), Jacobian rank, and the
named-object map (I1..I6, B_k carrier, G-contracted I1 variants).
"""

import json

import numpy as np
import torch

from u_family_defs import (ETA_NP, NAMED, all_diagrams, eval_np, eval_torch,
                           F_tensor_np, generic_F_np, rand_sym_np, rand_u_np,
                           trivially_zero)

rng = np.random.default_rng(20260829)
REL_TOL = 1e-10
SV_TOL = 1e-8


def sample_realizable(n):
    out = []
    for _ in range(n):
        A = [rand_sym_np(rng) for _ in range(4)]
        out.append((F_tensor_np(A), rand_u_np(rng)))
    return out


def sample_generic(n):
    return [(generic_F_np(rng), rand_u_np(rng)) for _ in range(n)]


def value_matrix(samples, diagrams):
    V = np.zeros((len(samples), len(diagrams)))
    for si, (F, u) in enumerate(samples):
        for di, (caps, pairs) in enumerate(diagrams):
            V[si, di] = eval_np(F, F, u, caps, pairs)
    return V


def find_classes(V):
    scale = np.abs(V).max()
    norms = np.linalg.norm(V, axis=0)
    zero_cols = [i for i in range(V.shape[1]) if norms[i] < 1e-10 * scale]
    live = [i for i in range(V.shape[1]) if i not in set(zero_cols)]
    classes = []
    for i in live:
        placed = False
        for cl in classes:
            j = cl[0]
            denom = V[:, j]
            mask = np.abs(denom) > 1e-8 * norms[j]
            r = np.median(V[mask, i] / denom[mask])
            if np.max(np.abs(V[:, i] - r * V[:, j])) <= REL_TOL * max(norms[i], 1e-300):
                cl.append(i)
                placed = True
                break
        if not placed:
            classes.append([i])
    return zero_cols, classes


def svd_rank(M):
    cn = np.linalg.norm(M, axis=0)
    M = M[:, cn > 1e-10 * cn.max()] / cn[cn > 1e-10 * cn.max()]
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > SV_TOL * s[0])), s


def jacobian_rank(diagrams, reps, npoints=3):
    eta_t = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    ranks = []
    for _ in range(npoints):
        a = torch.tensor(rng.standard_normal((4, 4, 4)), dtype=torch.float64, requires_grad=True)
        w = torch.tensor(rng.standard_normal(3) * 0.6, dtype=torch.float64, requires_grad=True)
        A = (a + a.transpose(1, 2)) / 2
        u = torch.cat([torch.sqrt(1 + w @ w).reshape(1), w])
        F = torch.zeros((4, 4, 4, 4), dtype=torch.float64)
        Fs = []
        for mu in range(4):
            row = []
            for nu in range(4):
                row.append(A[mu] @ eta_t @ A[nu] - A[nu] @ eta_t @ A[mu])
            Fs.append(row)
        F = torch.stack([torch.stack(r) for r in Fs])
        rows = []
        for ri in reps:
            caps, pairs = diagrams[ri]
            val = eval_torch(F, F, u, caps, pairs, eta_t)
            g_a, g_w = torch.autograd.grad(val, (a, w), retain_graph=True, allow_unused=True)
            g_a = torch.zeros_like(a) if g_a is None else g_a
            g_w = torch.zeros_like(w) if g_w is None else g_w
            rows.append(torch.cat([g_a.reshape(-1), g_w.reshape(-1)]).detach().numpy())
        J = np.array(rows)
        rn = np.linalg.norm(J, axis=1, keepdims=True)
        s = np.linalg.svd(J / np.maximum(rn, 1e-300), compute_uv=False)
        ranks.append(int(np.sum(s > SV_TOL * s[0])))
    return ranks


def pivot_basis(basis_vals, rank):
    # column-pivoted QR: first `rank` pivot columns form an independent sub-basis
    from scipy.linalg import qr
    _, _, piv = qr(basis_vals, pivoting=True)
    return sorted(piv[:rank].tolist())


def decompose(target_vals, basis_vals, names):
    coef, res, *_ = np.linalg.lstsq(basis_vals, target_vals, rcond=None)
    fit = basis_vals @ coef
    resid = np.max(np.abs(fit - target_vals)) / max(np.max(np.abs(target_vals)), 1e-300)
    return {n: c for n, c in zip(names, coef) if abs(c) > 1e-8}, float(resid)


def eval_I1_G(F, u, where):
    G = ETA_NP + 2 * np.outer(u, u)
    md = G if 'deriv' in where else ETA_NP
    mm = G if 'matrix' in where else ETA_NP
    return np.einsum('mnab,MNAB,mM,nN,aA,bB->', F, F, md, md, mm, mm, optimize=True)


def main():
    diagrams = all_diagrams()
    assert len(diagrams) == 735
    triv = sum(trivially_zero(*d) for d in diagrams)

    real_s = sample_realizable(40)
    gen_s = sample_generic(40)
    V_real = value_matrix(real_s, diagrams)
    V_gen = value_matrix(gen_s, diagrams)

    zero_r, classes_r = find_classes(V_real)
    zero_g, classes_g = find_classes(V_gen)
    for (caps, pairs) in [diagrams[i] for i in zero_r]:
        pass  # zero set recorded below

    # all trivially-zero diagrams must be measured zero on both ensembles
    tz = {i for i, d in enumerate(diagrams) if trivially_zero(*d)}
    assert tz <= set(zero_r) and tz <= set(zero_g), 'trivial zeros not measured zero'

    reps_r = [cl[0] for cl in classes_r]
    big_real = sample_realizable(100)
    big_gen = sample_generic(100)
    R_real = np.array([[eval_np(F, F, u, *diagrams[i]) for i in reps_r] for F, u in big_real])
    R_gen = np.array([[eval_np(F, F, u, *diagrams[i]) for i in reps_r] for F, u in big_gen])
    rank_real, sv_r = svd_rank(R_real)
    rank_gen, sv_g = svd_rank(R_gen)
    jranks = jacobian_rank(diagrams, reps_r)

    # named objects in the class-representative basis
    def dvals(caps, pairs):
        return np.array([eval_np(F, F, u, caps, pairs) for F, u in big_real])

    named_vals = {n: dvals(*d) for n, d in NAMED.items()}
    for wh in ('matrix', 'deriv', 'matrix+deriv'):
        named_vals[f'I1_G_{wh}'] = np.array([eval_I1_G(F, u, wh) for F, u in big_real])

    # direct class membership for literal named diagrams
    def canon(caps, pairs):
        return (tuple(sorted(caps)),
                tuple(sorted(tuple(sorted(p)) for p in pairs)))

    diag_index = {canon(*d): i for i, d in enumerate(diagrams)}
    named_class = {}
    for n, d in NAMED.items():
        i = diag_index[canon(*d)]
        for ci, cl in enumerate(classes_r):
            if i in cl:
                named_class[n] = f'C{ci}'
                break

    # unique decomposition in a pivoted independent sub-basis
    piv = pivot_basis(R_real, rank_real)
    rep_names = [f'C{k}' for k in piv]
    basis = R_real[:, piv]
    named_map = {}
    for n, tv in named_vals.items():
        co, resid = decompose(tv, basis, rep_names)
        named_map[n] = {'coef': co, 'resid': resid}
        assert resid < 1e-8, f'named object {n} not in family span (resid {resid})'

    diag_class = {}
    for ci, cl in enumerate(classes_r):
        for i in cl:
            diag_class[i] = ci

    out = {
        'n_diagrams': len(diagrams),
        'n_trivially_zero': triv,
        'n_zero_measured_realizable': len(zero_r),
        'n_zero_measured_generic': len(zero_g),
        'n_classes_realizable': len(classes_r),
        'n_classes_generic': len(classes_g),
        'class_sizes_realizable': sorted((len(c) for c in classes_r), reverse=True),
        'value_rank_realizable': rank_real,
        'value_rank_generic': rank_gen,
        'jacobian_ranks': jranks,
        'sv_tail_realizable': [float(x) for x in sv_r[-6:]],
        'representatives': {f'C{k}': {'caps': list(diagrams[i][0]),
                                      'pairs': [list(p) for p in diagrams[i][1]]}
                            for k, i in enumerate(reps_r)},
        'pivot_basis': [f'C{k}' for k in piv],
        'named_class': named_class,
        'named_map': named_map,
    }
    with open('results/u_family_float.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)

    print(f'diagrams 735, trivially zero {triv}, measured zero {len(zero_r)} (realizable) '
          f'/ {len(zero_g)} (generic)')
    print(f'classes: {len(classes_r)} realizable (sizes {out["class_sizes_realizable"]}), '
          f'{len(classes_g)} generic')
    print(f'value rank: {rank_real} realizable, {rank_gen} generic; '
          f'jacobian ranks {jranks}')
    print(f'pivot basis: {out["pivot_basis"]}')
    for n, c in named_class.items():
        print(f'  {n:16s} is a member of class {c}')
    for n, d in named_map.items():
        cs = ', '.join(f'{k}:{v:+.6g}' for k, v in sorted(d['coef'].items()))
        print(f'  {n:16s} -> {cs}   (resid {d["resid"]:.1e})')
    if rank_real < 6:
        print('BLOCKER: realizable rank < 6')


if __name__ == '__main__':
    main()
