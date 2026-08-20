"""Stage C candidate tests: B1 (Berry linear term) vs B3 (boost condensate).

B1 killer (analytic, confirmed numerically): a term linear in dM/dt drops
out of the Hamiltonian under the Legendre transform,
  L = kin q'^2 + kappa B q' - V  =>  p = 2 kin q' + kappa B,
  H = p q' - L = kin q'^2 + V,
so it cannot tilt E(omega); it only changes dynamics (precession).

B3 (Mexican hat in the boost channel of the G-split):
  L_C = rot2 - a*boost2 + b*boost2^2,
with rot2/boost2 the per-point channel densities of report-001 machinery.
Pointwise: bounded below by -a^2/4b globally, vanishes on spatial fields
(boost2 = 0 there -> exact 3x3), and on clock textures
  E(omega) = E_stat - a k2 omega^2 + b k2^2 omega^4,   k2 = boost2 at omega=1,
giving omega*^2 = a/(2 b k2): a finite clock from free minimization.
"""
import json
import os

import torch

torch.manual_seed(11)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
G_T, DELTA, SG = 8.0, 0.3, 8.0
M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=torch.float64))


def G_lagrange(M):
    x = ETA @ M
    I4 = torch.eye(4, dtype=M.dtype)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def channels(F, M):
    """(rot2, boost2) matrix-pair channel densities in the Hamiltonian
    reading: the derivative pair is contracted with the Euclideanizer G
    too (energy in the M-selected frame), so both channels are
    nonnegative squares and L(F,G,G) = rot2 + boost2 >= 0."""
    G = G_lagrange(M)
    lGG = torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F, G, G, G, G, F)
    lGe = torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F, G, G, ETA, ETA, F)
    return (lGG + lGe) / 2, (lGG - lGe) / 2


def F_of(A):
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)


results = {}

# --- B1: numeric confirmation of the Hamiltonian drop-out ---------------
kin, kap, B_, V_ = 0.7, 1.3, 2.1, 0.4
qd = torch.linspace(-3, 3, 7, dtype=torch.float64)
p = 2 * kin * qd + kap * B_
H = p * qd - (kin * qd ** 2 + kap * B_ * qd - V_)
drop = (H - (kin * qd ** 2 + V_)).abs().max().item()
print(f"B1 Legendre drop-out residual: {drop:.1e} (0 = kappa term gone)")
results["B1_hamiltonian_dropout"] = drop

# --- B3 on the clock counterexample -------------------------------------
# background A_1 texture + clock velocity A_0 = omega * diag(1,0,0,0)
def E_of_omega(om, a, b, A_texture):
    A = A_texture.clone()
    A[0] = om * torch.diag(torch.tensor([1.0, 0, 0, 0],
                                        dtype=torch.float64))
    r2, b2 = channels(F_of(A), M_VAC)
    return (r2 - a * b2 + b * b2 ** 2).item()

At = torch.zeros(4, 4, 4, dtype=torch.float64)
At[1, 0, 1] = At[1, 1, 0] = 1.0                  # the P239 clock texture
a_, b_ = 1.0, 0.05
om = torch.linspace(0, 6, 121, dtype=torch.float64)
E = torch.tensor([E_of_omega(o.item(), a_, b_, At) for o in om])
k = int(E.argmin())
om_star = om[k].item()
# analytic prediction: boost2(omega) = k2 om^2 -> om*^2 = a/(2 b k2)
A1 = At.clone()
A1[0] = torch.diag(torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))
k2 = channels(F_of(A1), M_VAC)[1]
om_pred = (a_ / (2 * b_ * k2)) ** 0.5
print(f"B3 counterexample: min E at omega* = {om_star:.3f} "
      f"(predicted {om_pred:.3f}), E(omega*) = {E[k]:.4f} < E(0) = {E[0]:.4f}")
# the three-functional family on the same texture, for the report figure:
# current (eta reading) = rot2 - boost2, bounded (all-G) = rot2 + boost2,
# condensate = rot2 - a*boost2 + b*boost2^2
curves = {"eta": [], "G": [], "condensate": []}
for o in om:
    A = At.clone()
    A[0] = o.item() * torch.diag(torch.tensor([1.0, 0, 0, 0],
                                              dtype=torch.float64))
    r2, b2 = channels(F_of(A), M_VAC)
    curves["eta"].append((r2 - b2).item())
    curves["G"].append((r2 + b2).item())
    curves["condensate"].append((r2 - a_ * b2 + b_ * b2 ** 2).item())
results["family_curves"] = {"omega": om.tolist(), **curves,
                            "a": a_, "b": b_}
results["B3_counterexample"] = {"omega_star": om_star,
                                "omega_pred": om_pred.item(),
                                "E_min": E[k].item(), "E_0": E[0].item(),
                                "curve_omega": om.tolist(),
                                "curve_E": E.tolist()}

# --- B3 global boundedness: random-direction dive scan ------------------
# lesson of the eigenvalue-lift collapse: scan random A directions at
# growing amplitude; L_C must stay >= -a^2/(4b) everywhere.
floor = -a_ ** 2 / (4 * b_)
worst = 0.0
for _ in range(200):
    A = torch.randn(4, 4, 4, dtype=torch.float64)
    A = A + A.transpose(-1, -2)
    for amp in (0.5, 1.0, 2.0, 5.0, 10.0):
        r2, b2 = channels(F_of(amp * A), M_VAC)
        val = (r2 - a_ * b2 + b_ * b2 ** 2).item()
        worst = min(worst, val)
print(f"B3 dive scan (1000 points): min density {worst:.3f} "
      f"vs analytic floor {floor:.3f}")
results["B3_dive_scan"] = {"min_density": worst, "floor": floor}

# --- B3 3x3 guard: boost2 vanishes on spatial fields --------------------
worst_b2 = 0.0
for _ in range(10):
    A = torch.zeros(4, 4, 4, dtype=torch.float64)
    r = torch.randn(3, 3, 3, dtype=torch.float64)
    A[1:, 1:, 1:] = r + r.transpose(-1, -2)
    M = M_VAC.clone()
    Sp = torch.randn(3, 3, dtype=torch.float64)
    Q = torch.linalg.qr(Sp)[0]
    M[1:, 1:] = Q @ torch.diag(torch.tensor([1.0, DELTA, 0.0],
                                            dtype=torch.float64)) @ Q.T
    r2, b2 = channels(F_of(A), M)
    worst_b2 = max(worst_b2, (b2.abs() / r2.abs()).item())
print(f"B3 3x3 guard: |boost2|/rot2 on on-potential spatial fields "
      f"<= {worst_b2:.1e}")
results["B3_spatial_guard"] = worst_b2

# --- Legendre check (review point 6): both readings of the condensate ---
# If L_C is fundamental: p = dL/dw = -2ak2 w + 4bk2^2 w^3, H = pw - L
#   => H(w) = -a k2 w^2 + 3b k2^2 w^4, finite minimum at w*^2 = a/(6bk2):
#   the finite clock SURVIVES the Legendre transform, shifted by sqrt(3).
# If the energy functional is fundamental (the relaxation-stack reading),
# the section above is already the Hamiltonian and L follows with b/3.
k2v = k2.item()
omH = (a_ / (6 * b_ * k2v)) ** 0.5
EH = [(-a_ * k2v * o ** 2 + 3 * b_ * k2v ** 2 * o ** 4) for o in om.tolist()]
kH = min(range(len(EH)), key=lambda i: EH[i])
print(f"Legendre (L-fundamental): H-minimum at omega = {om[kH].item():.3f} "
      f"(analytic {omH:.3f}), depth {min(EH):.4f} (analytic {-a_**2/(12*b_):.4f})")
# universal identity dH/dw = w * dp/dw: the nonzero-velocity energy minimum
# sits exactly at the Legendre caustic (Shapere-Wilczek branched-Hamiltonian
# structure) -- verified on the toy:
import numpy as np
w = np.linspace(0.01, 3, 2000)
p = -2 * a_ * k2v * w + 4 * b_ * k2v ** 2 * w ** 3
H = -a_ * k2v * w ** 2 + 3 * b_ * k2v ** 2 * w ** 4
dH = np.gradient(H, w)
dp = np.gradient(p, w)
caustic = w[np.argmin(np.abs(dp))]
hmin = w[np.argmin(H)]
print(f"caustic dp/dw=0 at w = {caustic:.3f}, H-minimum at w = {hmin:.3f} "
      f"(coincide: {abs(caustic-hmin) < 0.01})")
results["legendre"] = {"omega_star_H": omH, "depth_H": -a_**2/(12*b_),
                       "caustic_at_minimum": bool(abs(caustic-hmin) < 0.01)}

# --- full 6x6 kinetic matrix (review point 17): mixed terms included ----
# for the all-G quadratic sector K_ij = sum_i <C_i, C_j>_G is a Gram
# matrix in a positive product => positive semidefinite analytically;
# numerical eigenvalues confirm (and quantify the gap).
GEN = []
for (i, j), boost in ((( 1, 2), False), ((1, 3), False), ((2, 3), False),
                      ((0, 1), True), ((0, 2), True), ((0, 3), True)):
    W = torch.zeros(4, 4, dtype=torch.float64)
    if boost:
        W[i, j] = W[j, i] = 1.0
    else:
        W[i, j], W[j, i] = -1.0, 1.0
    GEN.append(W)
Ms = M_VAC.clone()
Sp = torch.randn(3, 3, dtype=torch.float64)
Ms[1:, 1:] = M_VAC[1:, 1:] + 0.5 * (Sp + Sp.T) / 2
As = torch.zeros(4, 4, 4, dtype=torch.float64)
r = torch.randn(3, 3, 3, dtype=torch.float64)
As[1:, 1:, 1:] = r + r.transpose(-1, -2)
Gm = G_lagrange(Ms)
tangents = []
for W in GEN:
    a0 = W @ Ms + Ms @ W.T
    tangents.append(a0 / a0.norm())
K = torch.zeros(6, 6, dtype=torch.float64)
for i in range(6):
    for j in range(6):
        acc = 0.0
        for k in (1, 2, 3):
            Ci = tangents[i] @ ETA @ As[k] - As[k] @ ETA @ tangents[i]
            Cj = tangents[j] @ ETA @ As[k] - As[k] @ ETA @ tangents[j]
            acc = acc + torch.einsum("ab,ac,bd,cd->", Ci, Gm, Gm, Cj)
        K[i, j] = acc
ev = torch.linalg.eigvalsh(K)
print(f"kinetic matrix (all-G, 6x6 incl. mixed): eigenvalues "
      f"{[f'{e:.2f}' for e in ev.tolist()]} -> min {ev.min().item():.3f}")
results["kinetic_matrix_allG"] = {"eigenvalues": ev.tolist(),
                                  "min_eig": ev.min().item()}

with open(os.path.join(HERE, "results", "clock_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("written: clock_results.json")
