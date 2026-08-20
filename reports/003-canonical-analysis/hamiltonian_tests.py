"""Canonical Hamiltonian, kinetic Hessian and reduced EOM for the
condensate candidates (report 002 par.9, steps 1-3).

Channel objects, all from four base contractions l_DX (D-metric on the
derivative pair, X-metric on the matrix pair, D,X in {G, eta}):

  B_k = (lGG - leG - lGe + lee)/4   kinetic-boost  (electric x boost)
  B_s = (lGG + leG - lGe - lee)/4   static-boost
  B_L = B_k - B_s                   the eta-natural Lorentz combination

Candidates for the quartic completion:
  C1 (energy-functional): H := quad + (-a B_H + b B_H^2), B_H = B_k + B_s
  C2 (naive Lorentzian):  L := L2 - a B_L + b B_L^2
  C3 (u-selected):        L := L2 - a B_k + b B_k^2   (B_k covariant via G(M))

Reduced 1-DOF mechanics (B_k = k qd^2, B_s = s fixed) done in sympy;
full 10-dim velocity space done in torch (autograd Pi, Hessians).
"""
import json
import os

import sympy as sp
import torch

torch.manual_seed(17)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
DELTA, SG = 0.3, 8.0
M_VAC = torch.diag(torch.tensor([-SG, 1.0, DELTA, 0.0], dtype=torch.float64))
A_, B_ = 1.0, 0.05                     # condensate couplings a, b

results = {}

# ================= 1. symbolic Legendre (sympy, exact) ==================
qd, k, s, a, b = sp.symbols("qdot k s a b", positive=True)

def legendre(L):
    p = sp.diff(L, qd)
    return sp.expand(p * qd - L)

# quadratic sector: L2 = T - V with T = k qd^2, V = v
v = sp.Symbol("v", positive=True)
H2 = legendre(k * qd ** 2 - v)
assert sp.simplify(H2 - (k * qd ** 2 + v)) == 0
# C2: L = -a(Bk - s) + b(Bk - s)^2, Bk = k qd^2
Bk = k * qd ** 2
H_C2 = sp.expand(legendre(-a * (Bk - s) + b * (Bk - s) ** 2))
H_C2_check = 3 * b * Bk ** 2 - (a + 2 * b * s) * Bk - a * s - b * s ** 2
assert sp.simplify(H_C2 - H_C2_check) == 0
# C3: L = -a Bk + b Bk^2
H_C3 = sp.expand(legendre(-a * Bk + b * Bk ** 2))
assert sp.simplify(H_C3 - (-a * Bk + 3 * b * Bk ** 2)) == 0
print("symbolic Legendre:")
print("  quadratic: H = T + V (exact; no Legendre subtlety)")
print(f"  C2: H = {H_C2_check}  -> static sector H(Bk=0) = -a*s - b*s**2:"
      "  UNBOUNDED below in s")
print("  C3: H = -a*Bk + 3*b*Bk**2  -> no static term; floor -a^2/(12b)")
results["symbolic"] = {
    "H_C2": str(H_C2_check),
    "H_C3": "-a*Bk + 3*b*Bk**2",
    "C2_static": "-a*s - b*s**2 (unbounded below)",
    "omega_star_C3_sq": "a/(6*b*k)"}

# ================= 2. torch channel machinery ===========================
def G_lagrange(M):
    x = ETA @ M
    I4 = torch.eye(4, dtype=M.dtype)
    q = (x @ (x - I4) @ (x - DELTA * I4)) / (SG * (SG - 1) * (SG - DELTA))
    return ETA - 2.0 * q @ ETA


def F_of(A):
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)


def contractions(F, M):
    """The matrix pair contracts with G (inverse-transpose variance,
    matching M -> Lambda M Lambda^T); the derivative pair needs the
    opposite variance, i.e. G_d = eta G eta (same numbers at the
    vacuum, correct transformation off it)."""
    G = G_lagrange(M)
    Gd = ETA @ G @ ETA
    l = {}
    for dn, D in (("G", Gd), ("e", ETA)):
        for xn, X in (("G", G), ("e", ETA)):
            l[dn + xn] = torch.einsum("mnab,mp,nq,ac,bd,pqcd->",
                                      F, D, D, X, X, F)
    return l


def channels(F, M):
    l = contractions(F, M)
    B_k = (l["GG"] - l["eG"] - l["Ge"] + l["ee"]) / 4
    B_s = (l["GG"] + l["eG"] - l["Ge"] - l["ee"]) / 4
    quad = l["GG"]                      # rot2 + boost2, H-reading, >= 0
    return quad, B_k, B_s


# symmetric-matrix parametrization of the velocity (10 components)
IDX = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
       (2, 2), (2, 3), (3, 3)]


def mdot_of(c):
    Md = torch.zeros(4, 4, dtype=c.dtype)
    for k_, (i, j) in enumerate(IDX):
        Md[i, j] = Md[i, j] + c[k_]
        if i != j:
            Md[j, i] = Md[j, i] + c[k_]
    return Md


def L_of(c, A_sp, M, kind):
    A = A_sp.clone()
    A[0] = mdot_of(c)
    quad, B_k, B_s = channels(F_of(A), M)
    A0 = A_sp.clone()
    A0[0] = torch.zeros(4, 4, dtype=c.dtype)
    _, _, _ = channels(F_of(A0), M)     # (kept for clarity; V inside quad)
    T = quad - channels(F_of(A0), M)[0]  # kinetic part of the quadratic
    V = channels(F_of(A0), M)[0]
    # the condensate REPLACES the quadratic boost-kinetic channel
    # (T contains +B_k via the all-G norm; without the subtraction the
    # -a term would just cancel against it at a = 1)
    if kind == "C2":
        BL = B_k - B_s
        return (T - B_k) - V - A_ * BL + B_ * BL ** 2
    if kind == "C3":
        return (T - B_k) - V - A_ * B_k + B_ * B_k ** 2
    raise ValueError(kind)


def H_of(c, A_sp, M, kind):
    c = c.clone().requires_grad_(True)
    L = L_of(c, A_sp, M, kind)
    (Pi,) = torch.autograd.grad(L, c, create_graph=True)
    return (Pi * c).sum() - L


# ---- backgrounds -------------------------------------------------------
def bg_spatial_boosted(scale):
    """static background with NONZERO static-boost content B_s."""
    A = torch.zeros(4, 4, 4, dtype=torch.float64)
    A[1, 0, 1] = A[1, 1, 0] = scale
    A[2, 0, 2] = A[2, 2, 0] = scale
    r = torch.randn(3, 3, 3, dtype=torch.float64) * 0.3
    A[1:, 1:, 1:] = A[1:, 1:, 1:] + r + r.transpose(-1, -2)
    return A


At_clock = torch.zeros(4, 4, 4, dtype=torch.float64)
At_clock[1, 0, 1] = At_clock[1, 1, 0] = 1.0
c_clock = torch.zeros(10, dtype=torch.float64)
c_clock[0] = 1.0                        # Mdot = omega * diag(1,0,0,0)

# ---- 2a. C2 static-sector unboundedness (numeric) ----------------------
print("\nC2 vs C3 static Hamiltonian, one background scaled up")
print("  (identity check: H_C2 - H_C3 = -a*B_s - b*B_s^2 exactly):")
A_base = bg_spatial_boosted(1.0)
rows = []
for sc in (1.0, 2.0, 4.0, 8.0, 16.0):
    A_sp = sc * A_base
    z = torch.zeros(10, dtype=torch.float64)
    hC2 = H_of(z, A_sp, M_VAC, "C2").item()
    hC3 = H_of(z, A_sp, M_VAC, "C3").item()
    _, _, Bs_val = channels(F_of(A_sp), M_VAC)
    Bs = Bs_val.item()
    ident = abs((hC2 - hC3) - (-A_ * Bs - B_ * Bs ** 2)) / max(abs(hC2), 1)
    rows.append({"scale": sc, "B_s": Bs, "H_C2": hC2, "H_C3": hC3,
                 "identity_rel": ident})
    print(f"  scale {sc:5.1f}: B_s = {Bs:12.1f}   H_C2 = {hC2:16.1f}   "
          f"H_C3 = {hC3:14.1f}   id {ident:.1e}")
results["static_sector"] = rows
assert all(r["identity_rel"] < 1e-9 for r in rows)
assert rows[-1]["H_C2"] < -1e5, "C2 static sector unbounded below"
assert all(r["H_C3"] >= 0 for r in rows), "C3 statics must stay >= 0"

# ---- 2b. C3 clock: H(omega) both from autograd and formula -------------
_, k2t, _ = channels(F_of((lambda A: (A.__setitem__(0, torch.diag(
    torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))), A)[1])(
        At_clock.clone())), M_VAC)
k2v = k2t.item()
om_star = (A_ / (6 * B_ * k2v)) ** 0.5
scan = []
for om in (0.0, 0.5 * om_star, om_star, 1.5 * om_star):
    h = H_of(om * c_clock, At_clock, M_VAC, "C3").item()
    # T_rot = 0 on this texture (pure boost commutator), so
    # H = -a Bk + 3b Bk^2 exactly
    pred = -A_ * k2v * om ** 2 + 3 * B_ * (k2v * om ** 2) ** 2
    scan.append({"omega": om, "H": h, "H_formula": pred})
    print(f"  C3 clock: omega {om:6.3f}  H(autograd) {h:9.4f}  "
          f"formula {pred:9.4f}")
results["C3_clock_scan"] = scan
results["C3_omega_star"] = om_star
assert all(abs(r["H"] - r["H_formula"]) < 1e-9 for r in scan)

# ---- 3. kinetic Hessians d2L/dMdot2 (10x10) ----------------------------
def hessL(c0, A_sp, M, kind):
    return torch.autograd.functional.hessian(
        lambda c: L_of(c, A_sp, M, kind), c0)


print("\nkinetic Hessian spectra (C3):")
hess_rows = {}
for name, c0, A_sp in (
        ("vacuum", torch.zeros(10, dtype=torch.float64),
         torch.zeros(4, 4, 4, dtype=torch.float64)),
        ("spatial_bg", torch.zeros(10, dtype=torch.float64),
         bg_spatial_boosted(1.0)),
        ("clock_0.5w*", 0.5 * om_star * c_clock, At_clock),
        ("clock_w*", om_star * c_clock, At_clock),
        ("clock_1.5w*", 1.5 * om_star * c_clock, At_clock)):
    ev = torch.linalg.eigvalsh(hessL(c0, A_sp, M_VAC, "C3"))
    hess_rows[name] = ev.tolist()
    n_neg = int((ev < -1e-9).sum())
    n_zero = int(((ev >= -1e-9) & (ev < 1e-9)).sum())
    print(f"  {name:12s} min {ev.min().item():+9.3f}  neg {n_neg}  "
          f"zero {n_zero}  max {ev.max().item():+9.3f}")
results["hessian_C3"] = hess_rows

# quadratic all-G sector alone: Gram => >= 0 (numeric confirmation)
def T_quad(c, A_sp):
    A = A_sp.clone()
    A[0] = mdot_of(c)
    return channels(F_of(A), M_VAC)[0]

ev2 = torch.linalg.eigvalsh(torch.autograd.functional.hessian(
    lambda c: T_quad(c, bg_spatial_boosted(1.0)),
    torch.zeros(10, dtype=torch.float64)))
print(f"  quadratic sector alone: min eig {ev2.min().item():+9.3f} (>=0)")
results["hessian_quadratic_min"] = ev2.min().item()
assert ev2.min().item() > -1e-9

# ---- 4. reduced EOM / stability ----------------------------------------
# on the reduced family L has no explicit phase dependence -> every
# constant omega solves the Euler-Lagrange equation; selection of omega*
# is by energy minimization + stability (branched Hamiltonian).
d2H = torch.autograd.functional.hessian(
    lambda c: H_of(c, At_clock, M_VAC, "C3"), om_star * c_clock)
evH = torch.linalg.eigvalsh(d2H)
print(f"\nHessian of H at the omega* state: min {evH.min().item():+.3f}, "
      f"eigs {[f'{e:+.2f}' for e in evH.tolist()]}")
results["hessian_H_at_omega_star"] = evH.tolist()
# along-omega second derivative of H(omega): 4 a k2 > 0 (symbolic)
results["d2H_domega2_at_star"] = 4 * A_ * k2v

# ---- 5. gates ----------------------------------------------------------
# covariance of C3 density (random Lambda, u via q(etaM) recomputed)
W = torch.zeros(4, 4, dtype=torch.float64)
W[1, 2], W[2, 1] = -0.3, 0.3
W[0, 3] = W[3, 0] = 0.2
Lam = torch.matrix_exp(W)
Linv = ETA @ Lam.T @ ETA
Mx = M_VAC + 0.1 * (lambda R: (R + R.T) / 2)(
    torch.randn(4, 4, dtype=torch.float64))
Ax = (lambda R: R + R.transpose(-1, -2))(
    torch.randn(4, 4, 4, dtype=torch.float64))


def density_C3(M, A):
    quad, B_k, _ = channels(F_of(A), M)
    return quad - A_ * B_k + B_ * B_k ** 2


Axp = torch.stack([sum(Linv[n, m] * (Lam @ Ax[n] @ Lam.T)
                       for n in range(4)) for m in range(4)])
v0 = density_C3(Mx, Ax)
v1 = density_C3(Lam @ Mx @ Lam.T, Axp)
cov = ((v0 - v1).abs() / v0.abs()).item()
print(f"\ngates: C3 covariance rel {cov:.2e}", end="")
results["C3_covariance"] = cov
# 3x3 reduction: on spatial fields (A0 = 0, spatial blocks) B_k = D = 0
A3 = torch.zeros(4, 4, 4, dtype=torch.float64)
r = torch.randn(3, 3, 3, dtype=torch.float64)
A3[1:, 1:, 1:] = r + r.transpose(-1, -2)
_, bk3, _ = channels(F_of(A3), M_VAC)
print(f";  B_k on spatial fields {bk3.item():.1e}")
results["Bk_spatial"] = bk3.item()
assert cov < 1e-9 and abs(bk3.item()) < 1e-12

with open(os.path.join(HERE, "results", "hamiltonian_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("written: hamiltonian_results.json")
