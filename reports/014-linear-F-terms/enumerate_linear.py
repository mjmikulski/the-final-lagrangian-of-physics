"""L0 float route: classes, ranks, statics filter and velocity degree of the
frame-decorated linear-in-F family. Writes results/linear_float.json."""

import json

import numpy as np

from linear_defs import (diagram_label, eval_diagram_np, even_diagrams,
                         metrics_np, odd_diagrams, rand_frame_np)
from u_family_defs import ETA_NP, F_tensor_np, rand_sym_np

rng = np.random.default_rng(20260904)
REL = 1e-10


def sample(n, static=False, spatial_block=False, vacuum_frame=False):
    out = []
    for _ in range(n):
        A = [rand_sym_np(rng) for _ in range(4)]
        if static:
            A[0] = np.zeros((4, 4))
        if spatial_block:
            for i in range(1, 4):
                A[i][0, :] = 0
                A[i][:, 0] = 0
        E = np.eye(4) if vacuum_frame else rand_frame_np(rng)
        out.append((A, E))
    return out


def values(samples, diags):
    V = np.zeros((len(samples), len(diags)))
    for si, (A, E) in enumerate(samples):
        F = F_tensor_np(A)
        mets = metrics_np(E)
        for di, d in enumerate(diags):
            V[si, di] = eval_diagram_np(d, F, mets)
    return V


def classes_of(V):
    scale = np.abs(V).max()
    norms = np.linalg.norm(V, axis=0)
    zero = [i for i in range(V.shape[1]) if norms[i] < 1e-10 * scale]
    live = [i for i in range(V.shape[1]) if i not in set(zero)]
    cls = []
    for i in live:
        for cl in cls:
            j = cl[0]
            mask = np.abs(V[:, j]) > 1e-8 * norms[j]
            r = np.median(V[mask, i] / V[mask, j])
            if np.max(np.abs(V[:, i] - r * V[:, j])) <= REL * norms[i]:
                cl.append(i)
                break
        else:
            cls.append([i])
    return zero, cls


def svd_rank(M):
    cn = np.linalg.norm(M, axis=0)
    keep = cn > 1e-10 * cn.max()
    s = np.linalg.svd(M[:, keep] / cn[keep], compute_uv=False)
    return int(np.sum(s > 1e-8 * s[0]))


def main():
    ev, od = even_diagrams(), odd_diagrams()
    diags = ev + od
    S = sample(40)
    V = values(S, diags)
    zero, cls = classes_of(V)
    reps = [cl[0] for cl in cls]
    even_reps = [i for i in reps if diags[i][0] == 'even']
    odd_reps = [i for i in reps if diags[i][0] == 'odd']
    big = sample(120)
    R = values(big, [diags[i] for i in reps])
    rank_all = svd_rank(R)
    rank_even = svd_rank(R[:, [k for k, i in enumerate(reps) if diags[i][0] == 'even']])
    rank_odd = svd_rank(R[:, [k for k, i in enumerate(reps) if diags[i][0] == 'odd']])
    print(f'diagrams {len(diags)} (even {len(ev)}, odd {len(od)}); zero {len(zero)}; '
          f'classes {len(cls)} (even {len(even_reps)}, odd {len(odd_reps)})')
    print(f'ranks: all {rank_all}, even {rank_even}, odd {rank_odd}')

    # velocity degree: I(lambda A0) must be affine in lambda
    worst = 0.0
    for A, E in big[:20]:
        F1 = F_tensor_np(A)
        F2 = F_tensor_np([2 * A[0]] + A[1:])
        F0 = F_tensor_np([0 * A[0]] + A[1:])
        mets = metrics_np(E)
        for i in reps:
            d = diags[i]
            v0, v1, v2 = (eval_diagram_np(d, F0, mets), eval_diagram_np(d, F1, mets),
                          eval_diagram_np(d, F2, mets))
            sc = max(abs(v0), abs(v1), abs(v2), 1e-300)
            worst = max(worst, abs(v2 - (2 * v1 - v0)) / sc)
    print(f'velocity degree <= 1 guard: worst affine residual {worst:.1e}')
    assert worst < 1e-9

    # statics filter on the 3x3 spatial sector with the vacuum frame
    Ssp = sample(30, static=True, spatial_block=True, vacuum_frame=True)
    Vsp = values(Ssp, [diags[i] for i in reps])
    sc = np.abs(Vsp).max()
    spatial = {}
    for k, i in enumerate(reps):
        col = Vsp[:, k]
        spatial[diagram_label(diags[i])] = ('zero' if np.abs(col).max() < 1e-12 * sc
                                            else 'alive')
    rank_sp = svd_rank(Vsp)
    n_alive = sum(v == 'alive' for v in spatial.values())
    print(f'3x3-spatial sector (vacuum frame): {n_alive} classes alive, rank {rank_sp}')

    # static part on generic static fields (A0 = 0, random frame)
    Sst = sample(30, static=True)
    Vst = values(Sst, [diags[i] for i in reps])
    rank_st = svd_rank(Vst)
    print(f'generic static fields (A0 = 0, random frame): rank {rank_st}')

    out = {'n_diagrams': len(diags), 'n_zero': len(zero), 'n_classes': len(cls),
           'class_sizes': sorted((len(c) for c in cls), reverse=True),
           'rank_all': rank_all, 'rank_even': rank_even, 'rank_odd': rank_odd,
           'rank_spatial_3x3': rank_sp, 'rank_static_generic': rank_st,
           'reps': {f'L{k}': {'label': diagram_label(diags[i]),
                              'diagram': list(diags[i]),
                              'spatial_3x3': spatial[diagram_label(diags[i])]}
                    for k, i in enumerate(reps)}}
    json.dump(out, open('results/linear_float.json', 'w'), indent=1)
    for k, i in enumerate(reps):
        print(f"  L{k:<3d} {diagram_label(diags[i]):28s} class size {len(cls[k]):3d}  "
              f"3x3: {spatial[diagram_label(diags[i])]}")


if __name__ == '__main__':
    main()
