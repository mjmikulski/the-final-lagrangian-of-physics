"""Route 2 for the load-bearing MEASURED inequalities (METHOD rule 3).

Independent of measure_energies.py at every layer -- no torch, no closed
forms, no midpoint grid:
- densities from the TENSOR route: dM built from the ansatz definition,
  A_i by central finite differences of dM (not the analytic Jacobian),
  F_ij = A_i eta A_j - A_j eta A_i, and I1/I4 by the report-001 eta
  contractions (einsum over all slots);
- single-hedgehog self-energies by 1D scipy.quad over the radius of the
  tensor-route density (spherical symmetry);
- two-body and cluster integrals by Sobol quasi-Monte Carlo (scipy.qmc)
  over the box, same core-ball cutoff geometry as route 1 where route 1
  uses one.

Verified against route 1 (results/energy_results.json) and, decisively,
the inequalities themselves:
1. virial S1/S4 = 4/3 on singles (gaussian and screened-power);
2. E1int, E4int > 0 and X = 3E1int - 4E4int > 0 with t1 > 4/3 at the
   assembly-binding points of the two clean screened-power profiles;
3. X > 0 for the cutoff-free gaussian pair at d = 0.5 and 1.0;
4. the chain-7 cluster witness ratio S1/S4 (> 1.5, matches route 1).

Runtime ~10-20 min CPU. Out: results/verify_results.json
"""
import json
import os

import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
ETA4 = np.diag([-1.0, 1.0, 1.0, 1.0])
E0 = np.array([1.0, 0, 0, 0])
FD_H = 1e-5
results = {}


def dM_batch(pts, centers, f_np):
    """delta M at points (n,3): sum_a f(r_a) (r4 e0^T + e0 r4^T)."""
    n = pts.shape[0]
    out = np.zeros((n, 4, 4))
    for c in centers:
        rv = pts - np.asarray(c)
        r = np.linalg.norm(rv, axis=1)
        r4 = np.zeros((n, 4))
        r4[:, 1:] = rv
        f = f_np(r)
        out += f[:, None, None] * (r4[:, :, None] * E0[None, None, :]
                                   + E0[None, :, None] * r4[:, None, :])
    return out


def densities_tensor(pts, centers, f_np):
    """e1, e4 via finite-difference A_i and full eta contractions."""
    n = pts.shape[0]
    A = np.zeros((3, n, 4, 4))
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = FD_H
        A[i] = (dM_batch(pts + dp, centers, f_np)
                - dM_batch(pts - dp, centers, f_np)) / (2 * FD_H)
    F = np.zeros((3, 3, n, 4, 4))
    for i in range(3):
        for j in range(3):
            F[i, j] = (A[i] @ ETA4 @ A[j] - A[j] @ ETA4 @ A[i])
    # I1 = F_{ij ab} eta^{aa'} eta^{bb'} F_{ij a'b'} (spatial derivative
    # metric = +1); I4 = Phi_{jb} eta^{bb'} Phi_{jb'} with
    # Phi_{jb} = sum_{i,a} eta^{ia}|spatial F_{ij ab} = sum_i F_{ij, i+1, b}
    e1 = np.einsum("ijnab,ac,bd,ijncd->n", F, ETA4, ETA4, F)
    Phi = np.zeros((n, 3, 4))
    for i in range(3):
        for j in range(3):
            Phi[:, j, :] += F[i, j, :, i + 1, :]
    e4 = np.einsum("njb,bc,njc->n", Phi, ETA4, Phi)
    return e1, e4


def self_radial(f_np, eps, Rmax):
    """Single hedgehog at origin: 1D radial quadrature of the tensor
    density (evaluated on the z axis; spherical symmetry)."""
    def dens(r, which):
        pt = np.array([[0.31 * r, 0.24 * r,
                        np.sqrt(1 - 0.31 ** 2 - 0.24 ** 2) * r]])
        e1, e4 = densities_tensor(pt, [[0.0, 0.0, 0.0]], f_np)
        return 4 * np.pi * r * r * (e1[0] if which == 1 else e4[0])
    S1 = quad(lambda r: dens(r, 1), eps, Rmax, limit=300)[0]
    S4 = quad(lambda r: dens(r, 4), eps, Rmax, limit=300)[0]
    return S1, S4


def qmc_pair(zcenters, f_np, R, eps, n_log2=22, chunk=200000, seed=5):
    """Axisymmetric two-center integral by Sobol QMC on the (rho,z) box
    with weight 2 pi rho and the same core-ball cutoff as route 1."""
    sob = qmc.Sobol(2, scramble=True, seed=seed)
    n = 2 ** n_log2
    u = sob.random(n)
    rho = u[:, 0] * R
    z = (u[:, 1] * 2 - 1) * R
    area = 2 * R * R
    tot = np.zeros(2)
    centers = [[0.0, 0.0, zc] for zc in zcenters]
    for i0 in range(0, n, chunk):
        rr, zz = rho[i0:i0 + chunk], z[i0:i0 + chunk]
        keep = np.ones(rr.shape[0], dtype=bool)
        for zc in zcenters:
            keep &= rr ** 2 + (zz - zc) ** 2 > eps ** 2
        pts = np.stack([rr[keep], np.zeros(keep.sum()), zz[keep]], 1)
        e1, e4 = densities_tensor(pts, centers, f_np)
        w = 2 * np.pi * rr[keep]
        tot += np.array([(e1 * w).sum(), (e4 * w).sum()])
    return tot * area / n


def qmc_cluster(centers, f_np, R=10.0, n_log2=23, chunk=200000, seed=7):
    sob = qmc.Sobol(3, scramble=True, seed=seed)
    n = 2 ** n_log2
    u = (sob.random(n) * 2 - 1) * R
    vol = (2 * R) ** 3
    tot = np.zeros(2)
    for i0 in range(0, n, chunk):
        e1, e4 = densities_tensor(u[i0:i0 + chunk], centers, f_np)
        tot += np.array([e1.sum(), e4.sum()])
    return tot * vol / n


def power_np(p, mu):
    return lambda r: np.exp(-mu * r) * np.maximum(r, 1e-300) ** (-p)


gauss_np = lambda r: np.exp(-r ** 2)

route1 = json.load(open(os.path.join(HERE, "results",
                                     "energy_results.json")))

# --- 1. virial on singles ---------------------------------------------------
print("1. virial via tensor-route radial quadrature:")
vir = {}
for tag, f_np, eps in [("gauss", gauss_np, 1e-8),
                       ("p0.5_mu0.1", power_np(0.5, 0.1), 1e-4)]:
    S1, S4 = self_radial(f_np, eps, 60.0)
    vir[tag] = S1 / S4
    print(f"   {tag}: S1/S4 = {S1 / S4:.7f}")
    assert abs(S1 / S4 - 4 / 3) < 1e-3
results["virial"] = vir

# --- 2. every clean row (all separations of all clean profiles) ------------
# Covers the assembly extrema by construction: the row set is read from the
# route-1 results, so the ceiling-setting row and the minimal-X rows are
# all re-derived here. The route-2 ceiling is persisted and asserted below
# against the route-2 chain witness.
print("2. screened-power pairs, ALL clean rows (tensor route, "
      "radial-S + QMC-E):")
pairs = {}
t1_route2 = []
clean_keys = [k for k, v in route1["tails"].items() if not v["uv_flagged"]]
for key in clean_keys:
    p = float(key.split("_")[0][1:])
    mu = float(key.split("mu")[1])
    R = min(45.0, 14.0 / mu)
    f_np = power_np(p, mu)
    S1, S4 = self_radial(f_np, 0.05, R)
    for r1 in route1["tails"][key]["rows"]:
        d = r1["d"]
        E = qmc_pair([-d, d], f_np, R, 0.05)
        i1, i4 = E[0] - 2 * S1, E[1] - 2 * S4
        X = 3 * i1 - 4 * i4
        rel1 = abs(i1 - r1["E1int"]) / abs(r1["E1int"])
        rel4 = abs(i4 - r1["E4int"]) / abs(r1["E4int"])
        pairs[f"{key}_d{d}"] = {"E1int": i1, "E4int": i4, "t1": i1 / i4,
                                "X": X, "rel_vs_route1": [rel1, rel4]}
        t1_route2.append(i1 / i4)
        print(f"   p={p} mu={mu} d={d}: E1int={i1:+.2f} E4int={i4:+.2f} "
              f"t1={i1 / i4:.4f} X={X:+.2f}  "
              f"(vs route1: {rel1:.1%}/{rel4:.1%})")
        assert i1 > 0 and i4 > 0 and X > 0 and i1 / i4 > 4 / 3
        assert rel1 < 0.05 and rel4 < 0.05
results["pairs"] = pairs
results["t1_ceiling_route2"] = max(t1_route2)
print(f"   route-2 clean ceiling: {max(t1_route2):.4f} "
      f"(route 1: {route1['t1_ceiling_clean']:.4f})")
assert abs(max(t1_route2) - route1["t1_ceiling_clean"]) < 0.02

# --- 3. cutoff-free gaussian pair ------------------------------------------
print("3. gaussian pair, no cutoff:")
Sg1, Sg4 = self_radial(gauss_np, 1e-8, 15.0)
gp = {}
for d in (0.5, 1.0):
    E = qmc_pair([-d, d], gauss_np, 10.0, 0.0, n_log2=21)
    i1, i4 = E[0] - 2 * Sg1, E[1] - 2 * Sg4
    X = 3 * i1 - 4 * i4
    ref = route1["gauss_pocket"][str(d)]
    gp[str(d)] = {"E1int": i1, "E4int": i4, "X": X}
    print(f"   d={d}: E1int={i1:+.4f} (route1 {ref['E1int']:+.4f})   "
          f"X={X:+.3f}")
    assert X > 0
    assert abs(i1 - ref["E1int"]) < 0.15
results["gauss_pair"] = gp

# --- 4. chain-7 cluster witness --------------------------------------------
print("4. chain-7 witness (tensor route, 3D QMC):")
chain7 = [[i * 1.1 - 3.3, 0, 0] for i in range(7)]
S = qmc_cluster(chain7, gauss_np)
ratio = S[0] / S[1]
ref = route1["clusters"]["chain7"]
print(f"   S1/S4 = {ratio:.4f}   (route1 {ref:.4f})")
assert ratio > 1.5 and abs(ratio - ref) / ref < 0.01
results["chain7_ratio"] = ratio
# the alpha<0 no-go branch, entirely within route 2:
assert results["t1_ceiling_route2"] < ratio
print(f"   route-2 no-go margin: ceiling {results['t1_ceiling_route2']:.4f}"
      f" < witness {ratio:.4f}")

with open(os.path.join(HERE, "results", "verify_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nALL ROUTE-2 MEASUREMENT CHECKS PASS; written "
      "results/verify_results.json")
