"""Route 1 (float64 torch): structure of the canonical boost-hedgehog
ansatz (J. Duda's notebook, 2026-08-21; see README Provenance).

Claims checked here (random points/profiles/centers):
1. First-order dressing: M = o M0 o^T with o = exp(m f(r) r.K) truncated at
   O(m) gives dM/dm|_0 = g f(r) (r e0^T + e0 r^T) -- matrix-exp autograd vs
   closed form.
2. A_i = d_i dM is boost-type (v_i e0^T + e0 v_i^T) with V_{ai} = (v_i)_a =
   d_i u_a, u = g f r; V is SYMMETRIC (u is a gradient field).
3. The eta-commutator F_ij = A_i eta A_j - A_j eta A_i = -(v_i v_j^T -
   v_j v_i^T): PURELY SPATIAL double two-form (all time components 0).
4. Consequences on the ansatz (pointwise identities):
   I2 = I1, I5 = I4, I3 = I1/2, I6 = 4 I4 - I1  (so the six invariants
   collapse to the two independent densities I1, I4);
   N1 = N2 = N3 = 0 (the exact-3x3-preserving directions are invisible);
   all one-eps pseudoscalars = 0 (report 005 spatial vanishing).
5. Closed forms: I1 = 2[(tr G)^2 - tr G^2], G = V^2;
   I4 = tr[(V^2 - trV * V)^2].
6. The author's notebook spatial term Hs = -sum_{3 pairs}(rot components
   of the PLAIN commutator)^2 equals -I1/4 on the ansatz.
7. Eigenvalue identities behind the pointwise lemma, on random (a,b,c):
   e1 - e4 = (ab-bc)^2 + (bc-ca)^2 + (ca-ab)^2,
   4 e4 - e1 = 4 (ab+bc+ca)^2   =>   rho = e1/e4 in [1, 4];
   sampled rho range on realizable two-hedgehog fields approaches both ends.

Out: printed report + results/structure_results.json
"""
import json
import itertools
import os

import torch

torch.manual_seed(6)
HERE = os.path.dirname(os.path.abspath(__file__))
DT = torch.float64
ETA = torch.diag(torch.tensor([-1.0, 1, 1, 1], dtype=DT))
E0 = torch.zeros(4, dtype=DT)
E0[0] = 1.0

K = torch.zeros(3, 4, 4, dtype=DT)          # boost generators
for i in range(3):
    K[i, 0, i + 1] = K[i, i + 1, 0] = 1.0

EPS = torch.zeros(4, 4, 4, 4, dtype=DT)
for perm in itertools.permutations(range(4)):
    s, p = 1, list(perm)
    for i in range(4):
        j = p.index(min(p[i:]), i)
        if j != i:
            p[i], p[j] = p[j], p[i]
            s = -s
    EPS[perm] = s

I_RECIPES = {   # einsum on F_{mu nu a b} (all-lower, contract with eta)
    "I1": "mnab,mp,nq,ac,bd,pqcd->",
    "I2": "mnab,ap,bq,mc,nd,pqcd->",
    "I3": "mnab,mp,aq,nc,bd,pqcd->",
}


def eval_I(F):
    e = ETA
    out = {}
    for k, spec in I_RECIPES.items():
        out[k] = torch.einsum(spec, F, e, e, e, e, F)
    Phi = torch.einsum("ma,mnab->nb", e, F)
    Phiu = torch.einsum("nc,bd,cd->nb", e, e, Phi)
    out["I4"] = torch.einsum("nb,nb->", Phi, Phiu)
    out["I5"] = torch.einsum("nb,bn->", Phi, Phiu)
    phi = torch.einsum("nb,nb->", e, Phi)
    out["I6"] = phi * phi
    return out


def pseudo(F):
    Fu = torch.einsum("ac,bd,mncd->mnab", ETA, ETA, F)
    return torch.stack([
        torch.einsum("mnrs,mnab,rsab->", EPS, F, Fu),
        torch.einsum("abcd,mnab,mncd->", EPS, F, torch.einsum(
            "am,bn,mncd->abcd", ETA, ETA, F)),
        torch.einsum("mngd,mnab,abgd->", EPS, F, Fu),
        torch.einsum("mnab,mnab->", EPS, F) * torch.einsum(
            "ma,nb,mnab->", ETA, ETA, F),
    ])


def f_profile(r, p, mu):
    return torch.exp(-mu * r) * r ** (-p)


def df_profile(r, p, mu):
    return torch.exp(-mu * r) * (-p * r ** (-p - 1) - mu * r ** (-p))


# --- 1. closed-form dM vs matrix-exp autograd ------------------------------
g = 1.3
centers = torch.tensor([[0.0, 0, 1.1], [0.2, -0.4, -0.9]], dtype=DT)
p_, mu_ = 0.5, 0.0
x = torch.tensor([0.7, -0.3, 0.4], dtype=DT)


def M_of_m(m, x):
    M0 = torch.zeros(4, 4, dtype=DT)
    M0[0, 0] = g
    o = torch.eye(4, dtype=DT)
    for c in centers:
        rv = x - c
        r = rv.norm()
        arg = m * f_profile(r, p_, mu_) * torch.einsum("i,iab->ab", rv, K)
        o = o @ torch.matrix_exp(arg)
    return o @ M0 @ o.T


m = torch.zeros((), dtype=DT).requires_grad_(True)
dM_auto = torch.autograd.grad(M_of_m(m, x).flatten() @ torch.randn(16, dtype=DT), m)
# cleaner: full Jacobian dM/dm at m=0
dM_auto = torch.autograd.functional.jacobian(
    lambda mm: M_of_m(mm, x), torch.zeros((), dtype=DT))


def dM_closed(x):
    out = torch.zeros(4, 4, dtype=DT)
    for c in centers:
        rv4 = torch.zeros(4, dtype=DT)
        rv4[1:] = x - c
        r = (x - c).norm()
        out += g * f_profile(r, p_, mu_) * (torch.outer(rv4, E0)
                                            + torch.outer(E0, rv4))
    return out


err1 = (dM_auto - dM_closed(x)).abs().max()
print(f"1. dM/dm|0: autograd(matrix_exp) vs closed form: {err1:.2e}")
assert err1 < 1e-12


# --- 2+3. A_i structure, V symmetric, F spatial ----------------------------
def V_of(x):
    V = torch.zeros(3, 3, dtype=DT)
    for c in centers:
        rv = x - c
        r = rv.norm()
        V += g * (df_profile(r, p_, mu_) / r * torch.outer(rv, rv)
                  + f_profile(r, p_, mu_) * torch.eye(3, dtype=DT))
    return V


A_auto = torch.autograd.functional.jacobian(dM_closed, x)   # (4,4,3)
V = V_of(x)
A_closed = torch.zeros(4, 4, 3, dtype=DT)
for i in range(3):
    v4 = torch.zeros(4, dtype=DT)
    v4[1:] = V[:, i]
    A_closed[:, :, i] = torch.outer(v4, E0) + torch.outer(E0, v4)
err2 = (A_auto - A_closed).abs().max()
errV = (V - V.T).abs().max()
print(f"2. A_i autograd vs boost-type closed form: {err2:.2e}; "
      f"V symmetric to {errV:.2e}")
assert err2 < 1e-12 and errV < 1e-12

A = A_auto.permute(2, 0, 1)                                 # (3,4,4)
F = torch.zeros(4, 4, 4, 4, dtype=DT)
for i in range(3):
    for j in range(3):
        F[i + 1, j + 1] = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
Fw = torch.zeros_like(F)
for i in range(3):
    for j in range(3):
        Fw[i + 1, j + 1, 1:, 1:] = -(torch.outer(V[:, i], V[:, j])
                                     - torch.outer(V[:, j], V[:, i]))
err3 = (F - Fw).abs().max()
time_norm = F[0].abs().max() + F[:, 0].abs().max() \
    + F[:, :, 0].abs().max() + F[..., 0].abs().max()
print(f"3. F = -v_i^v_j (purely spatial): {err3:.2e}; "
      f"time components: {time_norm:.2e}")
assert err3 < 1e-12 and time_norm == 0.0

# --- 4. invariant collapse + N-vanishing + pseudoscalars -------------------
I = eval_I(F)
G = V @ V
I1c = 2 * ((G.trace()) ** 2 - (G @ G).trace())
Phi_c = V @ V - V.trace() * V
I4c = (Phi_c @ Phi_c).trace()
checks = {
    "I2 - I1": I["I2"] - I["I1"],
    "I5 - I4": I["I5"] - I["I4"],
    "I3 - I1/2": I["I3"] - I["I1"] / 2,
    "I6 - (4 I4 - I1)": I["I6"] - (4 * I["I4"] - I["I1"]),
    "N1": I["I3"] - (I["I1"] + I["I2"]) / 4,
    "N2": (I["I1"] - I["I2"]) / 4 - I["I4"] + I["I5"],
    "N3": I["I1"] - 4 * I["I4"] + I["I6"],
    "I1 closed form": I["I1"] - I1c,
    "I4 closed form": I["I4"] - I4c,
}
scale = abs(I["I1"].item())
print(f"4. pointwise identities (scale I1 = {scale:.3e}):")
for k, v in checks.items():
    print(f"   {k:18s} {v.item():+.2e}")
    assert abs(v.item()) < 1e-10 * max(scale, 1.0)
P = pseudo(F)
print(f"   pseudoscalars max |P| = {P.abs().max():.2e}")
assert P.abs().max() < 1e-12

# --- 6. Jarek's Hs = -I1/4 -------------------------------------------------
coms = {}
for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    coms[(i, j)] = A[i] @ A[j] - A[j] @ A[i]        # PLAIN commutator, no eta
Hs = torch.zeros((), dtype=DT)
for c in coms.values():
    Hs -= c[1, 2] ** 2 + c[2, 3] ** 2 + c[3, 1] ** 2
err6 = Hs + I["I1"] / 4
print(f"6. Jarek's Hs + I1/4 = {err6.item():+.2e}")
assert abs(err6.item()) < 1e-10 * max(scale, 1.0)

# --- randomized sweep over profiles/centers/points -------------------------
results = {
    "dM_closed_form": err1.item(), "A_boost_type": err2.item(),
    "V_symmetric": errV.item(), "F_spatial_wedge": err3.item(),
    "F_time_components": time_norm.item(),
    "identities": {k: v.item() for k, v in checks.items()},
    "pseudoscalar_max": P.abs().max().item(),
    "jarek_Hs_plus_I1_over_4": err6.item(),
}

# --- 7. eigenvalue identities and the pointwise ratio range ----------------
abc = torch.randn(4000, 3, dtype=DT)
a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
e1e = 4 * (a * a * b * b + b * b * c * c + c * c * a * a)
e4e = (a * (b + c)) ** 2 + (b * (c + a)) ** 2 + (c * (a + b)) ** 2
id_lo = e1e - e4e - ((a * b - b * c) ** 2 + (b * c - c * a) ** 2
                     + (c * a - a * b) ** 2)
id_hi = 4 * e4e - e1e - 4 * (a * b + b * c + c * a) ** 2
print(f"7. eigenvalue identities: max|lo| = {id_lo.abs().max():.1e}, "
      f"max|hi| = {id_hi.abs().max():.1e}")
assert id_lo.abs().max() < 1e-10 and id_hi.abs().max() < 1e-10
results["eig_identity_lo"] = id_lo.abs().max().item()
results["eig_identity_hi"] = id_hi.abs().max().item()

# rho on realizable two-hedgehog fields (sampled)
g, p_, mu_ = 1.0, 0.5, 0.2
rmin, rmax = float("inf"), -float("inf")
for d in (0.7, 1.5, 2.5):
    centers = torch.tensor([[0.0, 0, -d], [0.0, 0, d]], dtype=DT)
    for _ in range(6):
        pts = torch.randn(4000, 3, dtype=DT) * 3
        Vb = torch.zeros(4000, 3, 3, dtype=DT)
        eye = torch.eye(3, dtype=DT)
        for ci in range(2):
            rv = pts - centers[ci]
            r = rv.norm(dim=1).clamp_min(0.05)
            f = torch.exp(-mu_ * r) * r ** (-p_)
            df = torch.exp(-mu_ * r) * (-p_ * r ** (-p_ - 1)
                                        - mu_ * r ** (-p_))
            Vb += (df / r)[:, None, None] * rv[:, :, None] * rv[:, None, :] \
                + f[:, None, None] * eye
        Gb = Vb @ Vb
        trG = Gb.diagonal(dim1=1, dim2=2).sum(1)
        e1b = 2 * (trG ** 2 - (Gb * Gb.transpose(1, 2)).sum((1, 2)))
        trV = Vb.diagonal(dim1=1, dim2=2).sum(1)
        Pb = Gb - trV[:, None, None] * Vb
        e4b = (Pb * Pb.transpose(1, 2)).sum((1, 2))
        ok = e4b > 1e-25 * e4b.max()
        q = e1b[ok] / e4b[ok]
        rmin, rmax = min(rmin, q.min().item()), max(rmax, q.max().item())
print(f"   sampled rho on two-hedgehog fields: [{rmin:.4f}, {rmax:.4f}]"
      f"   (bounds [1, 4])")
assert rmin > 1.0 - 1e-9 and rmax < 4.0 + 1e-9
results["rho_sampled"] = [rmin, rmax]

worst = 0.0
for trial in range(30):
    g = float(torch.rand(()) * 2 + 0.2)
    ncen = int(torch.randint(1, 4, ()))
    centers = torch.randn(ncen, 3, dtype=DT) * 2
    p_ = float(torch.rand(()) * 1.8 + 0.1)
    mu_ = float(torch.rand(()) * 0.8)
    x = torch.randn(3, dtype=DT) * 2
    if min((x - c).norm() for c in centers) < 0.15:
        continue
    A_ = torch.autograd.functional.jacobian(dM_closed, x).permute(2, 0, 1)
    F_ = torch.zeros(4, 4, 4, 4, dtype=DT)
    for i in range(3):
        for j in range(3):
            F_[i + 1, j + 1] = A_[i] @ ETA @ A_[j] - A_[j] @ ETA @ A_[i]
    I_ = eval_I(F_)
    sc = max(abs(I_["I1"].item()), 1e-30)
    resid = max(abs((I_["I2"] - I_["I1"]).item()),
                abs((I_["I5"] - I_["I4"]).item()),
                abs((I_["I3"] - I_["I1"] / 2).item()),
                abs((I_["I6"] - 4 * I_["I4"] + I_["I1"]).item()),
                float(pseudo(F_).abs().max())) / sc
    worst = max(worst, resid)
print(f"randomized sweep (30 configs): worst relative identity residual "
      f"= {worst:.2e}")
assert worst < 1e-9
results["random_sweep_worst"] = worst

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "structure_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nALL STRUCTURE CHECKS PASS; written results/structure_results.json")
