"""Stage A/B prototype: covariant rot/boost split of F and the
sign-flipped kinetic term L_B = rot^2 - g^4 boost^2 (Jarek 2026-08-20).

Everything is pointwise algebra (values of M and A_mu at one point) --
enough to gate covariance, 3x3 reduction, and the clock-channel sign
table before any lattice work.

The split needs a covariant choice of the time axis; the only available
source is M itself. Two routes, deliberately both:
  exact:  u = timelike eigenvector of eta.M  (exact 3x3 reduction,
          but eig is neither smooth nor complex-step safe)
  soft:   K_n = eta (M eta)^{n-1} M eta / tr((eta M)^n)  -> -u u^T
          (all-polynomial: smooth + complex-step safe; 3x3 reduction
          only up to (lambda_spatial/sg)^n -- measured below)
Either gives the covariant 'Euclideanizer' G = eta - 2 K (positive at
the vacuum), and per-(matrix-)pair channel norms
  rot^2   = (<F,F>_G + <F,F>_eta)/2,
  boost^2 = (<F,F>_G - <F,F>_eta)/2,
so Jarek's target is X_B = (G+eta)/2 - g^4 (G-eta)/2 as the matrix-pair
contraction metric.
"""
import json
import os

import torch

torch.manual_seed(5)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
G_T, DELTA, S = 8.0, 0.3, 1.0
SG = S * G_T
M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=torch.float64))


def K_soft(M, n):
    P = torch.linalg.matrix_power(M @ ETA, n - 1) @ M
    return (ETA @ P @ ETA) / torch.trace(
        torch.linalg.matrix_power(ETA @ M, n))


def G_soft(M, n=4):
    return ETA - 2.0 * K_soft(M, n)


def G_exact(M):
    lam, V = torch.linalg.eig(ETA @ M)
    k = torch.argmax(lam.real)
    u = V[:, k].real
    u = u / torch.sqrt(-(u @ ETA @ u))          # u.eta.u = -1 (timelike)
    if u[0] < 0:
        u = -u
    return ETA + 2.0 * torch.outer(u, u)


def F_of(A):
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)             # (4,4,4,4) F_{mn ab}


def L_I1(F, D, X):
    """I1-type full contraction; D on the derivative pair, X on the
    matrix pair."""
    return torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F, D, D, X, X, F)


def kinH(A_bg, a0, X):
    """Hamiltonian-reading omega^2 coefficient: sum_i <[a0,A_i],[a0,A_i]>_X
    (matrix-pair metric X; the derivative sum enters with +)."""
    out = 0.0
    for i in (1, 2, 3):
        C = a0 @ ETA @ A_bg[i] - A_bg[i] @ ETA @ a0
        out = out + torch.einsum("ab,ac,bd,cd->", C, X, X, C)
    return out


def gen_catalog():
    gens = {}
    for name, (i, j), boost in (("rot_xy", (1, 2), False),
                                ("rot_xz", (1, 3), False),
                                ("rot_yz", (2, 3), False),
                                ("boost_x", (0, 1), True),
                                ("boost_y", (0, 2), True),
                                ("boost_z", (0, 3), True)):
        W = torch.zeros(4, 4, dtype=torch.float64)
        if boost:
            W[i, j] = W[j, i] = 1.0
        else:
            W[i, j], W[j, i] = -1.0, 1.0
        gens[name] = W
    return gens


def conj_tangent(W, M):
    return W @ M + M @ W.T


def rand_lambda(scale=0.3):
    W = torch.zeros(4, 4, dtype=torch.float64)
    W[1, 2], W[2, 1] = -scale, scale             # rotation piece
    W[0, 3] = W[3, 0] = 0.7 * scale              # boost piece
    return torch.matrix_exp(W)


def spatial_field(M_spatial_scale=0.5):
    """Arbitrary admissible 3x3-sector point: exact vacuum time row,
    RANDOM spatial block (the Candidate-K lesson: not just uniaxial)."""
    M = M_VAC.clone()
    Sp = torch.randn(3, 3, dtype=torch.float64)
    M[1:, 1:] = M_VAC[1:, 1:] + M_spatial_scale * (Sp + Sp.T) / 2
    A = torch.zeros(4, 4, 4, dtype=torch.float64)
    r = torch.randn(3, 3, 3, dtype=torch.float64)
    A[1:, 1:, 1:] = r + r.transpose(-1, -2)
    return M, A


results = {}

# --- gate 1: covariance (both routes), with a negative control ----------
M = M_VAC + 0.15 * (lambda R: (R + R.T) / 2)(
    torch.randn(4, 4, dtype=torch.float64))
A = (lambda R: R + R.transpose(-1, -2))(
    torch.randn(4, 4, 4, dtype=torch.float64))
L = rand_lambda()
Linv = ETA @ L.T @ ETA
Mp = L @ M @ L.T
# A'_m = (L^-1)^n_m  L A_n L^T   (derivative index + both matrix indices)
Ap = torch.stack([sum(Linv[n, m] * (L @ A[n] @ L.T) for n in range(4))
                  for m in range(4)])

def density(Mx, Ax, Gfun):
    Fx = F_of(Ax)
    Gx = Gfun(Mx)
    XB = (Gx + ETA) / 2 - G_T ** 4 * (Gx - ETA) / 2
    return L_I1(Fx, ETA, XB)

for name, Gfun in (("exact", G_exact), ("soft_n4", lambda m: G_soft(m, 4))):
    v, vp = density(M, A, Gfun), density(Mp, Ap, Gfun)
    rel = ((v - vp).abs() / v.abs()).item()
    Lbad = L + 0.05 * torch.randn(4, 4, dtype=torch.float64)
    vbad = density(Lbad @ M @ Lbad.T, torch.stack(
        [sum((ETA @ Lbad.T @ ETA)[n, m] * (Lbad @ A[n] @ Lbad.T)
             for n in range(4)) for m in range(4)]), Gfun)
    neg = ((v - vbad).abs() / v.abs()).item()
    results[f"covariance_{name}"] = {"rel": rel, "neg_control": neg}
    print(f"gate1 covariance [{name}]: rel {rel:.2e} (neg control {neg:.2e})")

# --- gate 2: 3x3 reduction on arbitrary spatial fields ------------------
print("\ngate2: 3x3 reduction error |L_B - L_current|/|L_current| "
      "on random spatial fields")
red = {}
for trial in range(5):
    Ms, As = spatial_field()
    Fs = F_of(As)
    base = L_I1(Fs, ETA, ETA)
    row = {}
    row["exact"] = ((L_I1(Fs, ETA, (G_exact(Ms) + ETA) / 2
                          - G_T ** 4 * (G_exact(Ms) - ETA) / 2) - base)
                    .abs() / base.abs()).item()
    for n in (2, 4, 6, 8):
        Gn = G_soft(Ms, n)
        XB = (Gn + ETA) / 2 - G_T ** 4 * (Gn - ETA) / 2
        row[f"soft_n{n}"] = ((L_I1(Fs, ETA, XB) - base).abs()
                             / base.abs()).item()
    red = {k: max(red.get(k, 0.0), v) for k, v in row.items()}
print("  worst over 5 trials:", {k: f"{v:.1e}" for k, v in red.items()})
results["reduction_3x3"] = red

# --- gate 3: clock-channel sign table -----------------------------------
print("\ngate3: kin per generator (background: spatial field + vacuum M)")
Ms, As = spatial_field()
Gx = G_exact(Ms)
XB = (Gx + ETA) / 2 - G_T ** 4 * (Gx - ETA) / 2
table = {}
for name, W in gen_catalog().items():
    a0 = conj_tangent(W, Ms)
    a0 = a0 / a0.norm()
    row = {"eta": kinH(As, a0, ETA).item(),
           "G": kinH(As, a0, Gx).item(),
           "B": kinH(As, a0, XB).item()}
    table[name] = row
    print(f"  {name:8s} eta {row['eta']:+9.4f}   G {row['G']:+9.4f}   "
          f"B {row['B']:+11.2f}")
results["kin_table"] = table

# --- gate 4: the P239 clock counterexample under L_B --------------------
w = 1.0
Ac = torch.zeros(4, 4, 4, dtype=torch.float64)
Ac[0] = w * torch.diag(torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))
Ac[1, 0, 1] = Ac[1, 1, 0] = 1.0
Fc = F_of(Ac)
Gc = G_exact(M_VAC)
XBc = (Gc + ETA) / 2 - G_T ** 4 * (Gc - ETA) / 2
vals = {"eta,eta": L_I1(Fc, ETA, ETA).item(),
        "eta,G": L_I1(Fc, ETA, Gc).item(),
        "eta,B": L_I1(Fc, ETA, XBc).item(),
        "G,B": L_I1(Fc, Gc, XBc).item()}
print("\ngate4 counterexample densities (D-metric,X-metric):",
      {k: f"{v:+.3f}" for k, v in vals.items()})
results["counterexample"] = vals

# --- gate 5: complex-step safety of the soft route ----------------------
Mc = (M_VAC + 0.1 * torch.eye(4, dtype=torch.float64)).to(torch.complex128)
V = (lambda R: (R + R.T) / 2)(torch.randn(4, 4, dtype=torch.float64))
t = 1e-30

def dens_soft_c(Mx):
    ETAc = ETA.to(torch.complex128)
    P = torch.linalg.matrix_power(Mx @ ETAc, 3) @ Mx
    K = (ETAc @ P @ ETAc) / torch.trace(
        torch.linalg.matrix_power(ETAc @ Mx, 4))
    Gn = ETAc - 2 * K
    XBn = (Gn + ETAc) / 2 - G_T ** 4 * (Gn - ETAc) / 2
    Fx = F_of(A).to(torch.complex128)
    return torch.einsum("mnab,mp,nq,ac,bd,pqcd->", Fx, ETAc, ETAc,
                        XBn, XBn, Fx)

cs = dens_soft_c(Mc + 1j * t * V.to(torch.complex128)).imag / t
Mr = (M_VAC + 0.1 * torch.eye(4, dtype=torch.float64)).requires_grad_(True)
Pr = torch.linalg.matrix_power(Mr @ ETA, 3) @ Mr
Kr = (ETA @ Pr @ ETA) / torch.trace(torch.linalg.matrix_power(ETA @ Mr, 4))
Gr = ETA - 2 * Kr
XBr = (Gr + ETA) / 2 - G_T ** 4 * (Gr - ETA) / 2
dens = torch.einsum("mnab,mp,nq,ac,bd,pqcd->", F_of(A), ETA, ETA, XBr,
                    XBr, F_of(A))
(g_auto,) = torch.autograd.grad(dens, Mr)
rel_cs = ((cs - (g_auto * V).sum()).abs() / cs.abs()).item()
print(f"\ngate5 complex-step vs autograd (soft route): rel {rel_cs:.2e}")
results["complex_step_soft"] = rel_cs

with open(os.path.join(HERE, "results", "proto_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nwritten: proto_results.json")


# --- gate 6: A1 frozen-spectrum Lagrange projector ----------------------
# q = cubic through q(sg)=1, q(1)=q(delta)=q(0)=0: exact spectral
# projector whenever the eta.M spectrum sits at the potential's target,
# polynomial always (branch-free, complex-step safe).
def G_lagrange(M):
    x = ETA @ M
    I4 = torch.eye(4, dtype=M.dtype)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


print("\ngate6: A1 Lagrange-projector route")
v, vp = density(M, A, G_lagrange), density(Mp, Ap, G_lagrange)
print(f"  covariance rel {((v - vp).abs() / v.abs()).item():.2e}")
red_a1 = 0.0
red_a1_onpot = 0.0
for trial in range(5):
    Ms2, As2 = spatial_field()
    Fs2 = F_of(As2)
    base = L_I1(Fs2, ETA, ETA)
    GA = G_lagrange(Ms2)
    XBA = (GA + ETA) / 2 - G_T ** 4 * (GA - ETA) / 2
    red_a1 = max(red_a1, ((L_I1(Fs2, ETA, XBA) - base).abs()
                          / base.abs()).item())
    # on-potential spatial field: spectrum exactly (1, delta, 0)
    Q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    Ms2[1:, 1:] = Q @ torch.diag(torch.tensor(
        [1.0, DELTA, 0.0], dtype=torch.float64)) @ Q.T
    GA = G_lagrange(Ms2)
    XBA = (GA + ETA) / 2 - G_T ** 4 * (GA - ETA) / 2
    red_a1_onpot = max(red_a1_onpot, ((L_I1(Fs2, ETA, XBA) - base).abs()
                                      / base.abs()).item())
print(f"  3x3 reduction worst: random spatial M {red_a1:.1e}, "
      f"on-potential spatial M {red_a1_onpot:.1e}")
GA = G_lagrange(Ms)
kb = {n: (kinH(As, conj_tangent(W, Ms) / conj_tangent(W, Ms).norm(),
             GA)).item() for n, W in gen_catalog().items()}
print("  kin (all-G_A1):", {k: f"{v:+.2f}" for k, v in kb.items()})
results["A1_lagrange"] = {"reduction_random": red_a1,
                          "reduction_onpotential": red_a1_onpot,
                          "kin": kb}
with open(os.path.join(HERE, "results", "proto_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("updated: proto_results.json")
