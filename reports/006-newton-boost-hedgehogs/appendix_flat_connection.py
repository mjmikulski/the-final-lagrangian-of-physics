"""APPENDIX-flat-connection: the canonical ansatz is a flat so(1,3) connection.

Verifies, on report 006's boost-hedgehog ansatz M = o M_0 o^T:
  (1) omega_mu = (d_mu o) o^{-1} lies in so(1,3);
  (2) the transport identity (d_mu M) eta = [omega_mu, M eta]  (exact route:
      d M = omega M + M omega^T and omega^T eta = -eta omega);
  (3) flatness of the right-invariant Maurer-Cartan form,
      d_i omega_j - d_j omega_i - [omega_i, omega_j] = 0
      (NB the minus: the + convention belongs to the left form o^{-1} d o);
  (4) the same for a two-center dressing (006 allows clusters);
  (5) negative control: a generic so(1,3)-valued field is NOT flat.

Self-contained (numpy + scipy), seconds on CPU. Writes
results/appendix_flat_connection.json and asserts the structural content.
"""
import json
import os

import numpy as np
from scipy.linalg import expm

ETA = np.diag([-1.0, 1, 1, 1])
KB = np.zeros((3, 4, 4))
for k in range(3):
    KB[k, 0, k + 1] = KB[k, k + 1, 0] = 1.0
JR = np.zeros((3, 4, 4))
for k, (i, j) in enumerate(((2, 3), (3, 1), (1, 2))):
    JR[k, i, j], JR[k, j, i] = -1.0, 1.0

M0 = np.diag([1.0, 0, 0, 0])                # 006 vacuum, g = 1
X0 = np.array([0.6, -0.3, 0.8])
M_AMP, P = 0.2, 0.5                          # author's p = 1/2 profile

comm = lambda a, b: a @ b - b @ a
so13_res = lambda C: np.max(np.abs(C.T @ ETA + ETA @ C))


def o_of(x, m, p=P):
    r = np.linalg.norm(x)
    return expm(m * r ** (-p) * np.einsum("i,iab->ab", x, KB))


def grad3(f, x, h=1e-5):
    out = []
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        out.append((f(x + dp) - f(x - dp)) / (2 * h))
    return out


def flatness(o_field, x, h=1e-5):
    om_at = lambda y, i: grad3(o_field, y, h)[i] @ np.linalg.inv(o_field(y))
    om = [om_at(x, i) for i in range(3)]
    res = max(np.max(np.abs(grad3(lambda y: om_at(y, j), x, h)[i]
                            - grad3(lambda y: om_at(y, i), x, h)[j]
                            - comm(om[i], om[j])))
              for (i, j) in ((0, 1), (1, 2), (2, 0)))
    return om, res


out = {"ansatz": {"m": M_AMP, "p": P, "x0": X0.tolist(), "vacuum": "diag(1,0,0,0)"}}

# (1) + (2): algebra membership and the transport identity
o1 = lambda x: o_of(x, M_AMP)
M = o1(X0) @ M0 @ o1(X0).T
dM = grad3(lambda x: o_of(x, M_AMP) @ M0 @ o_of(x, M_AMP).T, X0)
om, res1 = flatness(o1, X0)
out["omega_so13_residual"] = max(so13_res(w) for w in om)
out["transport_identity_err"] = float(max(
    np.max(np.abs(dM[i] @ ETA - comm(om[i], M @ ETA))) for i in range(3)))

# (3): flatness, with FD refinement showing the residual is h^2-limited
_, res_h4 = flatness(o1, X0, h=1e-4)
out["flatness_residual"] = {"h=1e-5": float(res1), "h=1e-4": float(res_h4)}

# (4): two-center dressing
o2 = lambda x: o_of(x, M_AMP) @ o_of(x - np.array([1.5, 0.2, -0.4]), M_AMP)
_, res2 = flatness(o2, X0)
out["two_center_flatness_residual"] = float(res2)

# (5): negative control -- generic so(1,3)-valued field, not pure gauge
ng_at = lambda x, i: float(np.sin(x[0] + 2 * x[(i + 1) % 3])) * KB[i] + float(x[i]) * JR[i]
ng = [ng_at(X0, i) for i in range(3)]
out["negative_control_curvature"] = float(max(
    np.max(np.abs(grad3(lambda x: ng_at(x, j), X0)[i]
                  - grad3(lambda x: ng_at(x, i), X0)[j]
                  - comm(ng[i], ng[j])))
    for (i, j) in ((0, 1), (1, 2), (2, 0))))

assert out["omega_so13_residual"] < 1e-9
assert out["transport_identity_err"] < 1e-9
assert out["flatness_residual"]["h=1e-5"] < 1e-9
assert out["flatness_residual"]["h=1e-4"] > 5 * out["flatness_residual"]["h=1e-5"]
assert out["two_center_flatness_residual"] < 1e-9
assert out["negative_control_curvature"] > 1e-2

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "results", "appendix_flat_connection.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print("\nAll asserts passed: the ansatz is pure gauge (flat), the negative "
      "control is not.")
