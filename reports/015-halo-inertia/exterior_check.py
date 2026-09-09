"""On a uniaxial exterior the model reduces to the Faddeev-Skyrme quartic term of the director.

Configuration: eigenvalues at their vacuum values (E_0, E_1, E_2, E_2) with the time axis frozen, the charge
direction n(x) an arbitrary smooth unit-vector field: N = diag(E_0, E_2, E_2, E_2) + Delta n n^T (spatial),
Delta = E_1 - E_2. Claim: the static energy density is 4 Delta^4 sum_{i<j} (n . (d_i n x d_j n))^2, the
Faddeev-Skyrme term without sigma term. Checked at random points of random smooth director fields with
analytic derivatives, against lagrangian.terms. Writes results/exterior_check.json.
"""
import json
import numpy as np
import torch
from lagrangian import terms

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(1)
E = (100.0, 1.0, 0.01, 0.01)
D = E[1] - E[2]
worst = 0.0
for trial in range(20):
    # n(x) = normalise(v(x)) with v a random quadratic vector polynomial; derivatives by the chain rule
    A = rng.normal(size=(3, 3)); B = rng.normal(size=(3, 3, 3)); B = B + B.transpose(0, 2, 1); v0 = rng.normal(size=3)
    x = rng.normal(size=3)
    v = v0 + A @ x + np.einsum('ijk,j,k->i', B, x, x)
    dv = A + 2 * np.einsum('ijk,k->ij', B, x)                     # dv[i, j] = d_j v_i
    nv = np.linalg.norm(v); n = v / nv
    dn = (dv - np.outer(n, n @ dv)) / nv                           # dn[i, j] = d_j n_i
    M = np.zeros((4, 4)); M[0, 0] = E[0]; M[1:, 1:] = -(E[2] * np.eye(3) + D * np.outer(n, n))
    dM = np.zeros((4, 4, 4))
    for j in range(3):
        dM[1 + j, 1:, 1:] = -D * (np.outer(dn[:, j], n) + np.outer(n, dn[:, j]))
    kin, pot, _ = terms(torch.tensor(M)[None], torch.tensor(dM)[None], E, frozen=False)
    model = float(-kin)
    fs = 4 * D ** 4 * sum(float(n @ np.cross(dn[:, i], dn[:, j])) ** 2 for i in range(3) for j in range(i + 1, 3))
    worst = max(worst, abs(model - fs) / max(abs(fs), 1e-300))
    if trial < 3:
        print(f'  model {model:.6e}   Faddeev-Skyrme 4 Delta^4 sum (n.(d_i n x d_j n))^2 {fs:.6e}')
print(f'worst relative deviation over 20 random points: {worst:.1e}')
json.dump(dict(worst_relative_deviation=worst, trials=20, E=E), open('results/exterior_check.json', 'w'), indent=1)
assert worst < 1e-10
print('EXTERIOR OK')
