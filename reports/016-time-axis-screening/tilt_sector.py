"""Is the sector that K_u opens healthy? Kinetic form and characteristic speeds.

The quadratic form of the Lagrangian in the time derivatives, K_ab = d^2 L / d(d_0 M_a) d(d_0 M_b), and the
corresponding spatial form G_ab, are computed pointwise by automatic differentiation on a given background:
the vacuum, and the relaxed hedgehog at several radii. Positive eigenvalues of K mean no ghost; the
generalised eigenvalues of (G, K) are the squared characteristic speeds along the chosen direction.

python tilt_sector.py [--tilt 1.0]
"""
import argparse
import json
import numpy as np
import torch
from lagrangian import lagrangian
from soliton import to_matrix, to_vector, Grid

torch.set_default_dtype(torch.float64)
NAMES = ['M00', 'M01', 'M02', 'M03', 'M11', 'M12', 'M13', 'M22', 'M23', 'M33']


def forms(M, dM_bg, E, X, direction=2):
    """Kinetic and gradient quadratic forms of L at a point, on the background (M, dM_bg)."""
    def L_of(v0, vk):
        dM = dM_bg.clone()
        dM[0] = dM[0] + to_matrix(v0)
        dM[1 + direction] = dM[1 + direction] + to_matrix(vk)
        return lagrangian(M[None], dM[None], E, frozen=False, X=X)[0]

    z = torch.zeros(10, dtype=torch.float64)
    K = torch.autograd.functional.hessian(lambda v: L_of(v, z), z.clone())
    G = torch.autograd.functional.hessian(lambda v: L_of(z, v), z.clone())
    return K.numpy(), G.numpy()


RECORD = []


def report(tag, M, dM, E, X):
    K, G = forms(M, dM, E, X)
    kw = np.linalg.eigvalsh(0.5 * (K + K.T))
    gw = np.linalg.eigvalsh(0.5 * (G + G.T))
    live = [i for i in range(10) if abs(K[i, i]) > 1e-10]
    print(f'{tag}')
    print(f'   kinetic form eigenvalues : {np.array2string(kw, precision=3, suppress_small=True)}')
    print(f'   dynamical components     : {[NAMES[i] for i in live] or "none"}')
    # characteristic speeds on the subspace where the kinetic form is non-degenerate
    tol = 1e-8 * max(1.0, abs(kw).max())
    ev, V = np.linalg.eigh(0.5 * (K + K.T))
    keep = V[:, ev > tol]
    if keep.shape[1]:
        Kl = keep.T @ (0.5 * (K + K.T)) @ keep
        Gl = keep.T @ (-0.5 * (G + G.T)) @ keep        # -G: the spatial part enters with the opposite eta sign
        w = np.sort(np.linalg.eigvalsh(np.linalg.solve(Kl, Gl) if np.allclose(Kl, np.diag(np.diag(Kl))) else
                                       np.linalg.inv(np.linalg.cholesky(Kl)) @ Gl @ np.linalg.inv(np.linalg.cholesky(Kl)).T))
        print(f'   propagating modes {keep.shape[1]}, squared speeds along z: {np.array2string(w, precision=4)}')
        if w.min() < -1e-8:
            print('   WARNING: negative squared speed (gradient instability)')
    ghost = not (kw.min() > -1e-8 * max(1.0, abs(kw).max()))
    print(f'   ghost: {"KINETIC FORM HAS A NEGATIVE DIRECTION" if ghost else "none (kinetic form positive semi-definite)"}')
    RECORD.append(dict(tag=tag, kinetic_eigenvalues=kw.tolist(), dynamical=[NAMES[i] for i in live],
                       squared_speeds=(w.tolist() if keep.shape[1] else []), ghost=bool(ghost)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tilt', type=float, default=1.0)
    ap.add_argument('--static', default='results/fields/static_base_n32.pt')
    a = ap.parse_args()
    E = (100.0, 1.0, 0.01, 0.01)
    X = {'tilt': a.tilt} if a.tilt else None
    vac = torch.zeros(4, 4, dtype=torch.float64)
    vac[0, 0] = E[0]
    for i in range(3):
        vac[1 + i, 1 + i] = -E[2]
    vac[3, 3] = -E[1]
    for c, lab in ((None, 'base model'), (X, f'with K_u, c = {a.tilt:g}')):
        report(f'VACUUM, {lab}:', vac, torch.zeros(4, 4, 4, dtype=torch.float64), E, c)
    # the relaxed hedgehog: background gradients from the lattice, sampled on the z axis at several radii
    d = torch.load(a.static)
    grid = Grid(d['n'], d['box'], E=d['E'], device='cpu')
    u = d['u'].double()
    M_all, dM_all = grid.derivatives(u)
    r = grid.xq.norm(dim=-1)
    for rad in (0.5, 1.0, 2.0, 4.0):
        j = int((r - rad).abs().argmin())
        idx = np.unravel_index(j, r.shape)
        Mb, dMb = M_all[idx], dM_all[idx]
        for c, lab in ((None, 'base model'), (X, f'with K_u, c = {a.tilt:g}')):
            report(f'HEDGEHOG at r = {float(r[idx]):.2f}, {lab}:', Mb, dMb, E, c)
    json.dump(RECORD, open('results/tilt_sector.json', 'w'), indent=1)
