"""APPENDIX-no-connection-from-M: covariant derivatives D_mu = d_mu + C(M).

Four measured blocks (each with a negative control):
  1. THEOREM route check: every matrix function of A = M eta is
     eta-symmetric, so it cannot lie in so(1,3) unless it is zero -- no
     connection value can be built pointwise from M alone. Negative
     control: allowing d_mu M immediately yields [M eta, (d_mu M) eta],
     which IS in so(1,3) and nonzero.
  2. INDEX-MIXING escape C_{mu a b} = lam (eta_{mu a} u_b - eta_{mu b} u_a)
     with u(M) the unit timelike eigen-axis (covariantly normalized,
     u^T eta u = -1, future-oriented): opens F_0i on report 006's static
     ansatz, but (a) breaks the vacuum, |F_ij| = lam^2 exactly, and
     (b) the opened sector is COHERENTLY POWERED BY the same
     vacuum-breaking term: F_0i splits exactly (bilinearity) into a
     vacuum-powered part ~ lam^2 m and a remainder ~ lam m^2 that
     interfere destructively near m = lam.
  2b. CLASSIFICATION (review round 1): on the rank-1 orbit the equivariant
     connection space is exactly two-dimensional -- the tested eta-form
     and the reviewer's eps-form, and the eps component is INERT
     (C_mu u = 0 by antisymmetry, M = g u u^T => D = d identically).
  3. A scalar switch s(M) cannot rescue it: report 007 section-1 blindness,
     replicated here with an orthogonal-dressing negative control.
  4. OUTLOOK LEMMAS: the derivative-built so(1,3) element
     C_mu = lam [M eta, (d_mu M) eta] is vacuum-safe by construction, yet
     (a) on any static field C_0 = 0 => F(D)_{0i} = 0 exactly -- the time
     sector stays shut for derivative-built connections too; and (b) on
     the canonical ansatz the substitution is a pure rescaling,
     X~ = (1 - lam g^2) X (rank-1 vacuum projector + transport identity),
     so every constant-coefficient quadratic energy is (1 - lam g^2)^4
     times the baseline: report 006's sign map is invariant there.

Self-contained (numpy + scipy), seconds on CPU. Writes
results/appendix_no_connection.json and asserts the structural content.
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

M0 = np.diag([1.0, 0, 0, 0])                # report 006 vacuum, g = 1
X0 = np.array([0.6, -0.3, 0.8])
M_AMP, P = 0.2, 0.5

comm = lambda a, b: a @ b - b @ a
so13_res = lambda C: np.max(np.abs(C.T @ ETA + ETA @ C))
etasym_res = lambda A: np.max(np.abs(A.T @ ETA - ETA @ A))


def o_of(x, m):
    r = np.linalg.norm(x)
    return expm(m * r ** (-P) * np.einsum("i,iab->ab", x, KB))


def M_of(x, m):
    o = o_of(x, m)
    return o @ M0 @ o.T


def grad3(f, x, h=1e-5):
    out = []
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        out.append((f(x + dp) - f(x - dp)) / (2 * h))
    return out


out = {"ansatz": {"m": M_AMP, "p": P, "x0": X0.tolist()}}

# --- 1. no connection from M alone ---------------------------------------
rng = np.random.default_rng(3)
Mr = rng.normal(size=(4, 4))
Mr = Mr + Mr.T
A = Mr @ ETA
P_A = 0.7 * np.eye(4) - 1.3 * A + 0.4 * A @ A + 2.1 * np.linalg.matrix_power(A, 3)
M = M_of(X0, M_AMP)
dM = grad3(lambda x: M_of(x, M_AMP), X0)
Cd = [comm(M @ ETA, dM[i] @ ETA) for i in range(3)]
out["theorem"] = {
    "etasym_residual_powers": float(max(
        etasym_res(np.linalg.matrix_power(A, n)) for n in (1, 2, 3, 4))),
    "etasym_residual_polynomial": float(etasym_res(P_A)),
    "so13_residual_polynomial": float(so13_res(P_A)),   # O(1): NOT in so(1,3)
    "neg_control_so13_residual": float(max(so13_res(c) for c in Cd)),
    "neg_control_norm": float(np.linalg.norm(Cd[0])),
}

# --- 2. index-mixing connection ------------------------------------------
def u_of(Mx):
    # covariant normalization (PR review round 1): unit timelike,
    # u^T eta u = -1, future-oriented -- NOT the frame-dependent v/|v^0|
    w, V = np.linalg.eig(Mx @ ETA)
    v = np.real(V[:, np.argmax(np.abs(w))])
    n2 = -(v @ ETA @ v)
    assert n2 > 0, "eigen-axis not timelike"
    v = v / np.sqrt(n2)
    return v if v[0] > 0 else -v


def C_of(Mx, lam):
    uu = u_of(Mx)
    ud = ETA @ uu
    C = np.zeros((4, 4, 4))
    for mu in range(4):
        for a in range(4):
            for b in range(4):
                C[mu, a, b] = lam * ((a == mu) * ud[b] - ETA[mu, b] * uu[a])
    return C


def DM(Mx, dM4, lam):
    C = C_of(Mx, lam)                      # C M + M C^T keeps M symmetric
    return np.array([dM4[mu] + C[mu] @ Mx + Mx @ C[mu].T for mu in range(4)])


def F_of(D):
    return np.array([[D[a] @ ETA @ D[b] - D[b] @ ETA @ D[a] for b in range(4)]
                     for a in range(4)])


uM = u_of(M)
out["index_mixing"] = {
    "C_so13_residual": float(max(so13_res(c) for c in C_of(M, 0.3))),
    # equivariance of u, two routes: on the ansatz the eigen-axis IS the
    # boosted frame vector o e_0 (already unit timelike), so the covariant
    # u_of must return it exactly
    "u_eta_u": float(uM @ ETA @ uM),
    "u_vs_o_e0_residual": float(np.max(np.abs(uM - o_of(X0, M_AMP)[:, 0]))),
}

lams = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
vac_F = [float(np.max(np.abs(F_of(DM(M0, np.zeros((4, 4, 4)), la))[1:, 1:])))
         for la in lams]
out["index_mixing"]["vacuum"] = {"lams": lams, "F_ij": vac_F}

dM4 = np.zeros((4, 4, 4))
dM4[1:] = dM                               # d_0 M = 0: the field is static
stat = {"lams": lams,
        "F_0i": [float(np.max(np.abs(F_of(DM(M, dM4, la))[0, 1:]))) for la in lams],
        "F_ij": [float(np.max(np.abs(F_of(DM(M, dM4, la))[1:, 1:]))) for la in lams]}
out["index_mixing"]["static_ansatz"] = stat

# decomposition in the dressing amplitude m at lam = 0.1. F_0i is LINEAR
# in the spatial leg D_i, so the split D_i = D_i^vac + (D_i - D_i^vac)
# decomposes F_0i EXACTLY into a vacuum-powered part and a remainder:
#   F_0i = F(D_0, D_i^vac) + F(D_0, D_i - D_i^vac)
# The parts scale cleanly (lam^2 m and lam m^2); the FULL signal is their
# coherent (signed) sum and shows destructive interference near m = lam.
LAM = 0.1


def f0i_parts(mm, lam=LAM):
    MM = M_of(X0, mm)
    dd = np.zeros((4, 4, 4))
    dd[1:] = grad3(lambda x: M_of(x, mm), X0)
    D = DM(MM, dd, lam)
    Dvac = DM(M0, np.zeros((4, 4, 4)), lam)
    Dmix = D.copy()
    Dmix[1:] = Dvac[1:]                    # spatial legs frozen to vacuum value
    Drest = D.copy()
    Drest[1:] = D[1:] - Dvac[1:]
    Ff, Fv, Fr = (F_of(X)[0, 1:] for X in (D, Dmix, Drest))
    lin = np.max(np.abs(Ff - (Fv + Fr)))   # exact by bilinearity of F
    return (float(np.max(np.abs(Ff))), float(np.max(np.abs(Fv))),
            float(np.max(np.abs(Fr))), float(lin))


ms = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]
parts = [f0i_parts(m) for m in ms]
full = [p[0] for p in parts]
vacp = [p[1] for p in parts]
rest = [p[2] for p in parts]
lin_res = max(p[3] for p in parts)
sl_vac_m = float(np.polyfit(np.log(ms[:3]), np.log(vacp[:3]), 1)[0])
sl_rest_m = float(np.polyfit(np.log(ms[:3]), np.log(rest[:3]), 1)[0])
lam_scan = [0.1, 0.2, 0.4]
sl_vac_lam = float(np.polyfit(np.log(lam_scan),
                              np.log([f0i_parts(0.0125, la)[1] for la in lam_scan]),
                              1)[0])
out["index_mixing"]["decomposition"] = {
    "lam": LAM, "ms": ms, "F_0i_full": full,
    "vacuum_powered_part": vacp, "remainder": rest,
    "split_linearity_residual": lin_res,
    "rest_over_vacpart": [r / v for r, v in zip(rest, vacp)],
    "slope_vacpart_in_m": sl_vac_m,        # -> 1: linear in m
    "slope_remainder_in_m": sl_rest_m,     # -> 2
    "slope_vacpart_in_lam_at_m=0.0125": sl_vac_lam,   # -> 2: the lam^2 term
}

# --- 2b. classification on the rank-1 orbit (PR review round 1) -----------
# On the canonical orbit M = g u u^T, the only pointwise covariant data is
# (g, u). An equivariant C_{mu ab} (antisym in ab) is fixed by its value at
# M0, which must be invariant under the stabilizer SO(3) of u = e_0.
# Route 1 (algebra): the invariant subspace of the 24-dim space
# {T_{mu ab}} is spanned by exactly two tensors,
#   C1_{mu ab} = eta_{mu a} u_b - eta_{mu b} u_a      (the tested one)
#   C2_{mu ab} = eps_{mu ab nu} u^nu                  (reviewer's example)
# Route 2 (numeric): dimension of the joint fixed subspace of the three
# rotation actions, by SVD.
JR = np.zeros((3, 4, 4))
for k, (i, j) in enumerate(((2, 3), (3, 1), (1, 2))):
    JR[k, i, j], JR[k, j, i] = -1.0, 1.0

pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
basis = [(mu, p) for mu in range(4) for p in pairs]
vec = lambda T: np.array([T[mu, a, b] for (mu, (a, b)) in basis])
blocks = []
for k in range(3):
    R = expm(0.7 * JR[k])          # lowered spatial indices rotate with R
    act = np.zeros((24, 24))
    for col, (mu, (a, b)) in enumerate(basis):
        T = np.zeros((4, 4, 4))
        T[mu, a, b], T[mu, b, a] = 1.0, -1.0
        act[:, col] = vec(np.einsum("mn,ac,bd,ncd->mab", R, R, R, T))
    blocks.append(act - np.eye(24))
stack = np.vstack(blocks)
sv = np.linalg.svd(stack, compute_uv=False)
fixed_dim = int(np.sum(sv < 1e-10))

EPS = np.zeros((4, 4, 4, 4))       # eps_{0123} = +1
for p in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 0, 3, 2),
          (1, 2, 0, 3), (1, 3, 2, 0), (2, 0, 1, 3), (2, 1, 3, 0),
          (2, 3, 0, 1), (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0)):
    EPS[p] = 1.0
for p in ((0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 2, 1), (1, 0, 2, 3),
          (1, 2, 3, 0), (1, 3, 0, 2), (2, 0, 3, 1), (2, 1, 0, 3),
          (2, 3, 1, 0), (3, 0, 1, 2), (3, 1, 2, 0), (3, 2, 0, 1)):
    EPS[p] = -1.0
e0d = ETA @ np.array([1.0, 0, 0, 0])
C1 = np.zeros((4, 4, 4))
C2 = np.zeros((4, 4, 4))
for mu in range(4):
    for a in range(4):
        for b in range(4):
            C1[mu, a, b] = ETA[mu, a] * e0d[b] - ETA[mu, b] * e0d[a]
            C2[mu, a, b] = EPS[mu, a, b, 0]    # eps_{mu ab nu} u^nu, u = e_0
inv_res = float(max(np.max(np.abs(stack @ vec(C1))), np.max(np.abs(stack @ vec(C2)))))

# The eps component is INERT on the whole orbit: (C2)_{mu ab} u^b =
# eps_{mu ab nu} u^b u^nu = 0 by antisymmetry, and M = g u u^T, so
# C2 M + M C2^T = 0 identically => D = d, F(D) = F(d), no vacuum cost.
def DM_eps(Mx, dM4_, lam):
    uu = u_of(Mx)                  # C_mu^a_b = eta^{ac} eps_{mu c b nu} u^nu
    Ce = [lam * np.einsum("ac,cbn,n->ab", ETA, EPS[mu], uu) for mu in range(4)]
    return np.array([dM4_[mu] + Ce[mu] @ Mx + Mx @ Ce[mu].T for mu in range(4)])


eps_vac = float(np.max(np.abs(F_of(DM_eps(M0, np.zeros((4, 4, 4)), 0.3)))))
eps_inert = float(np.max(np.abs(DM_eps(M, dM4, 0.3) - dM4)))
out["classification"] = {
    "fixed_subspace_dim": fixed_dim,
    "C1_C2_invariance_residual": inv_res,
    "eps_component_vacuum_F": eps_vac,
    "eps_component_DM_minus_dM_on_ansatz": eps_inert,
}

# --- 3. scalar-switch blindness (report 007 section 1, replicated) --------
inv = lambda Mx: np.array([np.trace(np.linalg.matrix_power(Mx @ ETA, n))
                           for n in (1, 2, 3, 4)] + [np.linalg.det(Mx)])
K6 = np.concatenate([KB, JR])
Mv = np.diag([-8.0, 1.0, 0.3, 0.0])        # report 004 vacuum
I0 = inv(Mv)
rel = lambda Mx: np.max(np.abs(inv(Mx) - I0) / np.maximum(np.abs(I0), 1.0))
lor = max(rel((lambda oo: oo @ Mv @ oo.T)(
    expm(np.einsum("a,aij->ij", rng.normal(size=6) * 0.5, K6)))) for _ in range(500))
neg = max(rel((lambda q: q @ Mv @ q.T)(
    np.linalg.qr(rng.normal(size=(4, 4)))[0])) for _ in range(500))
out["blindness"] = {"lorentz_drift": float(lor), "neg_control_drift": float(neg)}

# --- 4. outlook lemma: derivative-built variant ---------------------------
def DM_b1(Mx, dM4_, lam):
    Am = Mx @ ETA
    Cb = [lam * comm(Am, dM4_[mu] @ ETA) for mu in range(4)]
    return np.array([dM4_[mu] + Cb[mu] @ Mx + Mx @ Cb[mu].T for mu in range(4)])


# rescaling lemma: A0 = M0 eta = -gP with P rank-1 => ad_{A0}^2 = id on
# commutators [kappa, A0]; by the transport identity every ansatz X_i is
# such a commutator, so X~ = (1 - lam g^2) X, F(D) = (1 - lam g^2)^2 F.
ad2 = lambda a, x: comm(a, comm(a, x))
kap = np.einsum("a,aij->ij", np.array([0.3, -1.2, 0.7]), KB)
kap[1:, 1:] += np.array([[0, 0.5, -0.2], [-0.5, 0, 0.9], [0.2, -0.9, 0]])
Z = comm(kap, M0 @ ETA)
lam_r = 0.37
resc = max(np.max(np.abs((dM[i] @ ETA - lam_r * ad2(M @ ETA, dM[i] @ ETA))
                         - (1 - lam_r) * (dM[i] @ ETA))) for i in range(3))
out["outlook_lemma"] = {
    "C_so13_residual": float(max(so13_res(c) for c in Cd)),
    "vacuum_F": float(np.max(np.abs(F_of(DM_b1(M0, np.zeros((4, 4, 4)), 0.3))))),
    "static_F_0i": float(np.max(np.abs(F_of(DM_b1(M, dM4, 0.3))[0, 1:]))),
    "ad2_identity_residual": float(np.max(np.abs(ad2(M0 @ ETA, Z) - Z))),
    "rescaling_residual_lam=0.37": float(resc),
}

# --- asserts --------------------------------------------------------------
t = out["theorem"]
assert t["etasym_residual_powers"] < 1e-9 and t["etasym_residual_polynomial"] < 1e-9
assert t["so13_residual_polynomial"] > 1.0          # p(A) is NOT in the algebra
assert t["neg_control_so13_residual"] < 1e-9 and t["neg_control_norm"] > 0.1
im = out["index_mixing"]
assert im["C_so13_residual"] < 1e-9
assert abs(im["u_eta_u"] + 1) < 1e-12          # covariant unit normalization
assert im["u_vs_o_e0_residual"] < 1e-9         # route 2: u = o e_0 exactly
cl = out["classification"]
assert cl["fixed_subspace_dim"] == 2           # exactly two equivariant tensors
assert cl["C1_C2_invariance_residual"] < 1e-10
assert cl["eps_component_vacuum_F"] < 1e-12    # reviewer's eps term: no cost...
assert cl["eps_component_DM_minus_dM_on_ansatz"] < 1e-12   # ...because inert
assert abs(im["vacuum"]["F_ij"][4] - 0.09) < 1e-12  # lam = 0.3 -> lam^2 exactly
assert im["vacuum"]["F_ij"][0] == 0.0               # negative control lam = 0
assert stat["F_0i"][0] == 0.0                       # lam = 0 reproduces 006
# generic lam opens the sector; lam = 0.2 ~ m f(r) is the isolated
# interference zero where the connection locally cancels the dressing
# gradient in the spatial leg (both sectors are linear in D_i)
assert all(f > 1e-3 for i, f in enumerate(stat["F_0i"][1:], 1) if stat["lams"][i] != 0.2)
assert stat["F_0i"][3] < 0.1 * min(stat["F_0i"][2], stat["F_0i"][4])   # the dip
assert stat["F_ij"][3] < 0.1 * min(stat["F_ij"][2], stat["F_ij"][4])
cx = im["decomposition"]
assert cx["split_linearity_residual"] < 1e-15       # exact bilinear split
assert abs(cx["slope_vacpart_in_m"] - 1) < 0.25     # vac part: lam^2 m
assert abs(cx["slope_remainder_in_m"] - 2) < 0.3    # remainder: lam m^2
assert abs(cx["slope_vacpart_in_lam_at_m=0.0125"] - 2) < 0.2
assert cx["rest_over_vacpart"][0] < 0.35            # small m: vacuum-powered
assert out["blindness"]["lorentz_drift"] < 1e-9
assert out["blindness"]["neg_control_drift"] > 0.1
ol = out["outlook_lemma"]
assert ol["vacuum_F"] == 0.0 and ol["static_F_0i"] == 0.0
assert ol["ad2_identity_residual"] == 0.0
assert ol["rescaling_residual_lam=0.37"] < 1e-10

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "results", "appendix_no_connection.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print("\nAll asserts passed.")
