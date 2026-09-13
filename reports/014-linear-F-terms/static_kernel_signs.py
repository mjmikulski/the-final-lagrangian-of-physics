"""L1d: the static quadratic form of each 3x3-alive linear class on spatial-block
gradients (18 components) at the vacuum frame -- eigenvalue signs."""
import json
import numpy as np
from linear_defs import eval_diagram_np, metrics_np, diagram_label
from u_family_defs import F_tensor_np
def main():
    fl = json.load(open('results/linear_float.json'))
    E = np.eye(4); mets = metrics_np(E)
    iu = np.triu_indices(3)
    def A_of(g):   # g: 18 -> three spatial-block symmetric 4x4
        A = [np.zeros((4, 4))]
        for i in range(3):
            B = np.zeros((3, 3)); B[iu] = g[6 * i:6 * i + 6]; B = B + B.T - np.diag(np.diag(B))
            X = np.zeros((4, 4)); X[1:, 1:] = B; A.append(X)
        return A
    out = {}
    for k, v in fl['reps'].items():
        if v['spatial_3x3'] != 'alive': continue
        d = tuple([v['diagram'][0], v['diagram'][1]] + v['diagram'][2:])
        Q = np.zeros((18, 18))
        for i in range(18):
            for j in range(18):
                e = np.zeros(18); e[i] += 1; e[j] += 1
                f = np.zeros(18); f[i] += 1; f[j] -= 1
                Q[i, j] = (eval_diagram_np(d, F_tensor_np(A_of(e)), mets) - eval_diagram_np(d, F_tensor_np(A_of(f)), mets)) / 4
        w = np.linalg.eigvalsh(0.5 * (Q + Q.T))
        out[k] = {'label': v['label'], 'n_pos': int((w > 1e-12).sum()), 'n_neg': int((w < -1e-12).sum()), 'trace': float(np.trace(Q))}
        print(f"{k:4s} {v['label']:22s} eigen signs +{out[k]['n_pos']} / -{out[k]['n_neg']}  trace {out[k]['trace']:+.2e}")
    json.dump(out, open('results/static_kernel_signs.json', 'w'), indent=1)
if __name__ == '__main__':
    main()
