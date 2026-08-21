"""Route 1 (float64 torch): enumerate and analyze all one-epsilon quadratic
contractions of the M5 field strength; verify every numerical claim of the
report.

Sections:
1. enumeration: 210 diagrams, 54 identically zero, 156 alive, 13
   proportionality classes on generic double two-forms;
2. linear rank of the class span: 4 on generic, 3 on realizable F(A)
   (identically-vanishing classes are excluded before column normalization
   -- normalizing a roundoff-level column would fake a rank);
3. P_dd == 0 on realizable fields + the cyclic-trace identity behind it;
4. named basis {P_dd, P_mm, P_dm, P_cp} and rational expansion of all 13
   classes;
5. transformation behaviour: proper-Lorentz invariance, exact parity flip;
6. vanishing on purely spatial fields, on the P239 clock counterexample and
   on its proper-Lorentz orbit;
7. two-epsilon diagrams reduce to span{I1..I6};
8. structural identities: chi^2 = 16*I3 - 4*I1 - 4*I2 (= 16*N1), I6 = phi^2,
   P_cp = chi*phi;
9. functional (Jacobian) ranks of {I1..I6, P_dd, P_mm, P_dm, P_cp}:
   9 generic / 8 realizable, matching the two identified relations;
10. null-Lagrangian tests (exact polynomial Hessian contracted with a
    symmetric second jet): phi and chi are total derivatives; P_mm, P_dm,
    P_cp and the control I1 are dynamical.

Out: printed report + results/numerical_results.json (quoted in README.md).
"""
import itertools
import json
import os

import torch

torch.manual_seed(21)
HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=DTYPE))

EPS = torch.zeros(4, 4, 4, 4, dtype=DTYPE)
for perm in itertools.permutations(range(4)):
    s, p = 1, list(perm)
    for i in range(4):          # selection-sort parity
        j = p.index(min(p[i:]), i)
        if j != i:
            p[i], p[j] = p[j], p[i]
            s = -s
    EPS[perm] = s

LET = "abcdefgh"
ANTISYM_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
results = {}


def F_of_A(A):
    AeA = torch.einsum("...mac,cd,...ndb->...mnab", A, ETA, A)
    return AeA - AeA.transpose(-4, -3)


def physical_F(n):
    A = torch.randn(n, 4, 4, 4, dtype=DTYPE)
    return F_of_A(A + A.transpose(-1, -2))


def generic_F(n):
    T = torch.randn(n, 4, 4, 4, 4, dtype=DTYPE)
    T = T - T.transpose(1, 2)
    return T - T.transpose(3, 4)


def matchings(slots):
    if not slots:
        yield []
        return
    a, rest = slots[0], slots[1:]
    for i, b in enumerate(rest):
        for m in matchings(rest[:i] + rest[i + 1:]):
            yield [(a, b)] + m


def evaluate(diagram, F1, F2):
    eps_slots, pairs = diagram
    operands = [F1, F2, EPS]
    spec = ["..." + LET[:4], "..." + LET[4:],
            "".join(LET[s] for s in eps_slots)]
    for p, q in pairs:
        operands.append(ETA)
        spec.append(LET[p] + LET[q])
    return torch.einsum(",".join(spec) + "->...", *operands)


def evaluate_metric(pairs, F1, F2):
    operands, spec = [F1, F2], ["..." + LET[:4], "..." + LET[4:]]
    for p, q in pairs:
        operands.append(ETA)
        spec.append(LET[p] + LET[q])
    return torch.einsum(",".join(spec) + "->...", *operands)


# report 001 representatives, same slot convention
I_REPS = {
    "I1": [(0, 4), (1, 5), (2, 6), (3, 7)],
    "I2": [(0, 6), (1, 7), (2, 4), (3, 5)],
    "I3": [(0, 4), (1, 6), (2, 5), (3, 7)],
    "I4": [(0, 2), (4, 6), (1, 5), (3, 7)],
    "I5": [(0, 2), (4, 6), (1, 7), (3, 5)],
    "I6": [(0, 2), (1, 3), (4, 6), (5, 7)],
}
NAMED = {
    "P_dd": ((0, 1, 4, 5), [(2, 6), (3, 7)]),
    "P_mm": ((2, 3, 6, 7), [(0, 4), (1, 5)]),
    "P_dm": ((0, 1, 6, 7), [(2, 4), (3, 5)]),
    "P_cp": ((0, 1, 2, 3), [(4, 6), (5, 7)]),
}


def metric_invariants(F):
    return torch.stack([evaluate_metric(p, F, F) for p in I_REPS.values()], -1)


def named_values(F):
    return torch.stack([evaluate(d, F, F) for d in NAMED.values()], -1)


def chi(F):
    return torch.einsum("mnab,...mnab->...", EPS, F)


def phi(F):
    return torch.einsum("ma,nb,...mnab->...", ETA, ETA, F)


# --- 1. enumeration and classes --------------------------------------------
diagrams = []
for eps_slots in itertools.combinations(range(8), 4):
    rest = [s for s in range(8) if s not in eps_slots]
    for pairs in matchings(rest):
        diagrams.append((eps_slots, pairs))
alive = [d for d in diagrams
         if not any(p in ANTISYM_PAIRS for p in d[1])]
print(f"diagrams: {len(diagrams)} total, {len(alive)} survive the "
      f"antisym-trace kill ({len(diagrams) - len(alive)} vanish identically)")

NS = 40
Fg = generic_F(NS)
V = torch.stack([evaluate(d, Fg, Fg) for d in alive], 1)
cols = V / V.norm(dim=0, keepdim=True)
classes = []                     # (representative index into alive, members)
for i in range(len(alive)):
    for c in classes:
        if (cols[:, i] @ cols[:, c[0]]).abs() > 1 - 1e-10:
            c[1].append(i)
            break
    else:
        classes.append((i, [i]))
print(f"proportionality classes on generic F: {len(classes)} "
      f"with sizes {[len(m) for _, m in classes]}")
results["counts"] = {
    "total": len(diagrams), "identically_zero": len(diagrams) - len(alive),
    "n_classes": len(classes), "class_sizes": [len(m) for _, m in classes]}


# --- 2. linear rank on both ensembles --------------------------------------
def rank_of(Fs, label):
    W = torch.stack([evaluate(alive[c[0]], Fs, Fs) for c in classes], 1)
    norms = W.norm(dim=0)
    dead = norms < 1e-10 * norms.max()     # identically-zero classes: do NOT
    W = W[:, ~dead]                        # normalize roundoff into fake rank
    W = W / W.norm(dim=0, keepdim=True)
    S = torch.linalg.svdvals(W)
    r = int((S > S[0] * 1e-10).sum())
    print(f"rank of pseudoscalar span, {label}: {r} / {len(classes)}  "
          f"({int(dead.sum())} classes vanish; smallest kept sval {S[r-1]:.1e})")
    return r, [int(i) for i in torch.nonzero(dead).flatten()]


r_gen, dead_gen = rank_of(generic_F(NS), "generic F")
r_phys, dead_phys = rank_of(physical_F(NS), "physical F(A)")
results["rank_generic"], results["rank_physical"] = r_gen, r_phys
results["dead_classes_generic"] = dead_gen
results["dead_classes_physical"] = dead_phys

# --- 3. P_dd on realizable fields ------------------------------------------
# P_dd = eps^{mnrs} F_{mn ab} F_{rs}^{ab} = -4 eps^{mnrs} tr(B_m B_n B_r B_s)
# with B_m = A_m eta; the cyclic trace shift is an odd permutation under eps.
Ap = torch.randn(400, 4, 4, 4, dtype=DTYPE)
Ap = Ap + Ap.transpose(-1, -2)
Fp = F_of_A(Ap)
pdd = evaluate(NAMED["P_dd"], Fp, Fp)
Bp = torch.einsum("...mab,bc->...mac", Ap, ETA)
cyc = torch.einsum("mnrs,...mij,...njk,...rkl,...sli->...",
                   EPS, Bp, Bp, Bp, Bp)
scaleF = (Fp.flatten(1).norm(dim=1) ** 2).max()
pdd_gen = evaluate(NAMED["P_dd"], Fg, Fg)
print(f"P_dd: max {pdd.abs().max():.1e} on 400 realizable samples "
      f"(scale |F|^2 ~ {scaleF:.0e}); cyclic identity max {cyc.abs().max():.1e}; "
      f"generic mean |P_dd| {pdd_gen.abs().mean():.1e} (nonzero)")
results["pdd_physical_max"] = pdd.abs().max().item()
results["pdd_cyclic_identity_max"] = cyc.abs().max().item()
results["pdd_generic_mean"] = pdd_gen.abs().mean().item()

# --- 4. named basis and rational expansions --------------------------------
Fg2 = generic_F(60)
B = torch.stack([evaluate(d, Fg2, Fg2) for d in NAMED.values()], 1)
rB = int(torch.linalg.matrix_rank(B / B.norm(dim=0, keepdim=True),
                                  tol=1e-10))
print(f"named candidates {list(NAMED)}: rank {rB}")
results["named_rank"] = rB
expansions = {}
for ci, (rep, members) in enumerate(classes):
    y = evaluate(alive[rep], Fg2, Fg2)
    coef = torch.linalg.lstsq(B, y.unsqueeze(1)).solution.squeeze(1)
    res = (B @ coef - y).norm() / y.norm()
    expansions[f"class{ci:02d}"] = {
        "eps_slots": list(alive[rep][0]),
        "eta_pairs": [list(p) for p in alive[rep][1]],
        "size": len(members),
        "coef": [round(c, 6) for c in coef.tolist()],
        "resid": res.item()}
worst_exp = max(e["resid"] for e in expansions.values())
print(f"all 13 classes expand in the named basis, worst residual "
      f"{worst_exp:.1e}")
results["worst_expansion_residual"] = worst_exp
results["expansions"] = expansions


# --- 5. transformation behaviour -------------------------------------------
def lorentz_transform(F, L):
    return torch.einsum("am,bn,cp,dq,...abcd->...mnpq", L, L, L, L, F)


def random_proper_lorentz():
    G = torch.randn(4, 4, dtype=DTYPE) * 0.3
    K = G - ETA @ G.T @ ETA               # so(1,3): K^T eta + eta K = 0
    return torch.matrix_exp(K)


F1 = generic_F(1)[0]
L = random_proper_lorentz()
FL = lorentz_transform(F1, L)
dI = (metric_invariants(FL) - metric_invariants(F1)).abs().max()
dP = (named_values(FL) - named_values(F1)).abs().max()
print(f"proper Lorentz (det L = {torch.det(L):.6f}): "
      f"|dI| {dI:.1e}, |dP| {dP:.1e} "
      f"(values O({named_values(F1).abs().max():.0e}))")
PAR = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=DTYPE))
FP_ = lorentz_transform(F1, PAR)
dI_p = (metric_invariants(FP_) - metric_invariants(F1)).abs().max()
dP_p = (named_values(FP_) + named_values(F1)).abs().max()   # expect flip
print(f"parity: I_k unchanged to {dI_p:.1e}, P_i flip residual {dP_p:.1e}")
results["proper_invariance"] = {"dI": dI.item(), "dP": dP.item()}
results["parity"] = {"dI": dI_p.item(), "dP_flip_resid": dP_p.item()}

# --- 6. spatial fields, clock counterexample, clock orbit ------------------
Fsp = generic_F(10)
Fsp[:, 0], Fsp[:, :, 0], Fsp[:, :, :, 0], Fsp[..., 0] = 0, 0, 0, 0
sp_max = torch.stack([evaluate(alive[c[0]], Fsp, Fsp)
                      for c in classes]).abs().max()
print(f"purely spatial generic F: max |one-eps class| = {sp_max:.1e}")
results["spatial_max"] = sp_max.item()

omega = 1.0
A_clock = torch.zeros(4, 4, 4, dtype=DTYPE)
A_clock[0, 0, 0] = omega
A_clock[1, 0, 1] = A_clock[1, 1, 0] = 1.0
Fc = F_of_A(A_clock)
I_clock = metric_invariants(Fc)
P_clock = torch.stack([evaluate(alive[c[0]], Fc, Fc) for c in classes])
print(f"clock counterexample: I = {I_clock.tolist()} "
      f"(expected omega^2*[4,4,2,2,2,4]); max |one-eps class| "
      f"= {P_clock.abs().max():.1e}")
results["clock_I"] = I_clock.tolist()
results["clock_pseudo_max"] = P_clock.abs().max().item()
worst_orbit = 0.0
for _ in range(10):
    Fo = lorentz_transform(Fc, random_proper_lorentz())
    worst_orbit = max(worst_orbit, torch.stack(
        [evaluate(alive[c[0]], Fo, Fo) for c in classes]).abs().max().item())
print(f"clock orbit (proper-Lorentz images): max = {worst_orbit:.1e}")
results["clock_orbit_pseudo_max"] = worst_orbit

# parity maps realizable to realizable with I even / P odd (on A directly)
A1 = torch.randn(4, 4, 4, dtype=DTYPE)
A1 = A1 + A1.transpose(-1, -2)
AP1 = torch.einsum("mn,ac,ncd,db->mab", PAR, PAR, A1, PAR)
tA, tAP = (torch.cat([metric_invariants(F_of_A(x)),
                      named_values(F_of_A(x))]) for x in (A1, AP1))
par_even = (tAP[:6] - tA[:6]).abs().max()
par_odd = (tAP[6:] + tA[6:]).abs().max()
print(f"parity image of realizable A: I even to {par_even:.1e}, "
      f"P odd to {par_odd:.1e}")
results["parity_on_A"] = {"even": par_even.item(), "odd": par_odd.item()}

# --- 7. two-eps diagrams reduce to span{I1..I6} ----------------------------
Fg3 = generic_F(40)
Im = metric_invariants(Fg3)
Imn = Im / Im.norm(dim=0, keepdim=True)
worst2, n2 = 0.0, 0
for eps1_slots in itertools.combinations(range(8), 4):
    eps2_slots = tuple(s for s in range(8) if s not in eps1_slots)
    spec = ["..." + LET[:4], "..." + LET[4:],
            "".join(LET[s] for s in eps1_slots),
            "".join(LET[s] for s in eps2_slots)]
    y = torch.einsum(",".join(spec) + "->...", Fg3, Fg3, EPS, EPS)
    if y.norm() < 1e-12:
        continue
    n2 += 1
    yn = (y / y.norm()).unsqueeze(1)
    coef = torch.linalg.lstsq(Imn, yn).solution
    worst2 = max(worst2, (Imn @ coef - yn).norm().item())
print(f"two-eps diagrams: {n2} nonzero slot splits, worst residual off "
      f"span(I1..I6) = {worst2:.1e}")
results["two_eps_nonzero"] = n2
results["two_eps_worst_residual"] = worst2

# --- 8. structural identities ----------------------------------------------
worst_id = {}
for label, Fs in [("generic", generic_F(60)), ("physical", physical_F(60))]:
    I = metric_invariants(Fs)
    c, p = chi(Fs), phi(Fs)
    cp = evaluate(NAMED["P_cp"], Fs, Fs)
    scale = I.abs().max()
    worst_id[label] = {
        "chi2_16I3_4I1_4I2":
            ((c * c - (16 * I[:, 2] - 4 * I[:, 0] - 4 * I[:, 1]))
             .abs().max() / scale).item(),
        "I6_phi2": ((I[:, 5] - p * p).abs().max() / scale).item(),
        "Pcp_chiphi": ((cp - c * p).abs().max() / scale).item()}
    print(f"identities, {label} (relative): "
          + ", ".join(f"{k} {v:.1e}" for k, v in worst_id[label].items()))
results["identities"] = worst_id


# --- 9. Jacobian ranks of the ten invariants -------------------------------
def F_of_T(T):
    T = T - T.transpose(0, 1)
    return T - T.transpose(2, 3)


def F_of_B(B):
    return F_of_A(B + B.transpose(-1, -2))


def jac_rank(param, to_F, label):
    param = param.clone().requires_grad_(True)
    F = to_F(param)
    vals = torch.cat([metric_invariants(F), named_values(F)])
    rows = [torch.autograd.grad(v, param, retain_graph=True)[0].flatten()
            for v in vals]
    S = torch.linalg.svdvals(torch.stack(rows))
    r = int((S > S[0] * 1e-10).sum())
    print(f"{label}: Jacobian rank {r}/10")
    return r


print("functional independence of {I1..I6, P_dd, P_mm, P_dm, P_cp}:")
results["jac_rank_generic"] = jac_rank(
    torch.randn(4, 4, 4, 4, dtype=DTYPE), F_of_T, "  generic F")
results["jac_rank_physical"] = jac_rank(
    torch.randn(4, 4, 4, dtype=DTYPE), F_of_B, "  physical A")


# --- 10. null-Lagrangian tests ---------------------------------------------
# L depends on A = dM only, and is polynomial, so the Hessian route is exact:
# EL_{ab} = -d_mu[dL/dA_{mu ab}] = -H_{(mu ab),(nu cd)} d_mu d_nu M_{cd}.
def el_residual(lagr, ntry=6):
    worst, scale = 0.0, 0.0
    for _ in range(ntry):
        A = torch.randn(4, 4, 4, dtype=DTYPE)
        A = A + A.transpose(-1, -2)
        ddM = torch.randn(4, 4, 4, 4, dtype=DTYPE)
        ddM = ddM + ddM.transpose(0, 1)
        ddM = ddM + ddM.transpose(2, 3)
        Af = A.flatten().clone().requires_grad_(True)

        def lf(x):
            return lagr(x.view(4, 4, 4))
        H = torch.autograd.functional.hessian(lf, Af).view(4, 16, 4, 16)
        el = -torch.einsum("aibj,abj->i", H, ddM.reshape(4, 4, 16)).view(4, 4)
        el = (el + el.T) / 2                 # physical EL: M symmetric
        worst = max(worst, el.abs().max().item())
        scale = max(scale, H.abs().max().item() * ddM.abs().max().item())
    return worst, scale


print("null-Lagrangian tests (max |EL|; 0 => total derivative):")
null_results = {}
tests = {
    "phi": lambda A: phi(F_of_A(A)),
    "chi": lambda A: chi(F_of_A(A)),
    "P_mm": lambda A: evaluate(NAMED["P_mm"], F_of_A(A), F_of_A(A)),
    "P_dm": lambda A: evaluate(NAMED["P_dm"], F_of_A(A), F_of_A(A)),
    "P_cp": lambda A: evaluate(NAMED["P_cp"], F_of_A(A), F_of_A(A)),
    "I1_control": lambda A: evaluate_metric(I_REPS["I1"], F_of_A(A),
                                            F_of_A(A)),
}
for name, lagr in tests.items():
    w, s = el_residual(lagr)
    null_results[name] = {"el_max": w, "scale": s}
    print(f"  {name:10s} max|EL| = {w:.3e}   (scale {s:.1e})")
results["null_lagrangian"] = null_results

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "numerical_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nwritten: results/numerical_results.json")
