"""Phase 2 (numerical): verify the sympy classes and test independence.

Three questions, answered on float64 random tensors:
1. which slot symmetries the PHYSICAL F (built from A_m = dM/dx^m, M
   symmetric) actually has -- the assumption behind enumerate_sympy.py;
2. do all pairings inside one sympy class give the same number (cross-check
   of the canonicalization by an independent einsum evaluator);
3. independence of the 6 invariants: (a) LINEAR (rank of a sample-value
   matrix -- exactly what a constant-coefficient Lagrangian family needs),
   (b) FUNCTIONAL (Jacobian rank at a generic point) -- each both for a
   generic pair-antisymmetric F and for realizable algebraic F(A) tensors
   (degeneracies there would be invisible to pure combinatorics);
4. the channel-basis identities behind the alternative (S/A) basis.

Out: printed report + results/numerical_results.json (quoted in README.md).
"""
import json
import os
import string

import torch

torch.manual_seed(21)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))


def physical_F(n):
    # A: (n, 4 derivative idx, 4, 4), symmetric in the matrix slots
    A = torch.randn(n, 4, 4, 4, dtype=torch.float64)
    A = A + A.transpose(-1, -2)
    AeA = torch.einsum("smac,cd,sndb->smnab", A, ETA, A)
    return AeA - AeA.transpose(1, 2)          # (n, 4, 4, 4, 4) = F_{mn ab}


def generic_F(n):
    T = torch.randn(n, 4, 4, 4, 4, dtype=torch.float64)
    T = T - T.transpose(1, 2)
    return T - T.transpose(3, 4)


def rel(x, ref):
    return (x.norm() / ref.norm()).item()


def contract(F1, F2, pairs):
    """Evaluate one pairing: 8 slot letters + one eta per metric pair.

    pairs uses enumerate_sympy slot numbering: 0-3 first F, 4-7 second F."""
    letters = string.ascii_lowercase[:8]
    operands, spec = [F1, F2], [letters[:4], letters[4:]]
    for p, q in pairs:
        operands.append(ETA)
        spec.append(letters[p] + letters[q])
    return torch.einsum(",".join(spec) + "->", *operands)


classes = json.load(open(os.path.join(HERE, "results", "contraction_classes.json")))["classes"]
# sign-corrected representative pairing per class, in class order
REP = [(c["natural_sign"] * c["members"][0]["sign"], c["members"][0]["pairs"])
       for c in classes]
results = {}


def invariants(f):
    return torch.stack([s * contract(f, f, p) for s, p in REP])   # (6,)

# --- 1. slot symmetries of the physical F ---------------------------------
F = physical_F(1)[0]
sym = {
    "antisym_mn": rel(F + F.transpose(0, 1), F),
    "antisym_ab": rel(F + F.transpose(2, 3), F),
    "pair_exchange": rel(F - F.permute(2, 3, 0, 1), F),
    # first-Bianchi analog: antisymmetrize slots (n,a,b); zero for Riemann
    "bianchi": rel((F + F.permute(0, 2, 3, 1) + F.permute(0, 3, 1, 2)) / 3, F),
}
Phi = torch.einsum("ma,mnab->nb", ETA, F)
sym["Phi_sym_part"] = rel(Phi + Phi.T, Phi)
sym["Phi_antisym_part"] = rel(Phi - Phi.T, Phi)
results["symmetries_physical_F"] = sym
print("slot symmetries of physical F (relative residuals; 0 = exact):")
for k, v in sym.items():
    print(f"  {k:16s} {v:.3e}")

# --- 2. class consistency: every pairing in a class gives sign * value ----
print("\nclass consistency (max spread across member pairings):")
worst = 0.0
for c in classes:
    vals = [c["natural_sign"] * m["sign"] * contract(F, F, m["pairs"])
            for m in c["members"]]
    vals = torch.stack(vals)
    spread = ((vals - vals[0]).abs().max() / vals[0].abs()).item()
    worst = max(worst, spread)
    print(f"  {c['name']:45s} value {vals[0].item():+.6e}  spread {spread:.1e}")
results["class_consistency_worst"] = worst

# --- 3. linear independence: rank of the sample-value matrix --------------
def rank_report(build, n_samples, label):
    Fs = build(n_samples)
    V = torch.stack([invariants(f) for f in Fs])
    V = V / V.norm(dim=0, keepdim=True)       # (n_samples, 6), column-normalized
    S = torch.linalg.svdvals(V)
    rank = int((S > S[0] * 1e-10).sum())
    print(f"\n{label}: singular values {[f'{s:.2e}' for s in S.tolist()]}"
          f"\n  -> rank {rank} / {len(classes)}")
    out = {"singular_values": S.tolist(), "rank": rank}
    if rank < len(classes):
        _, _, Vh = torch.linalg.svd(V)
        null = Vh[-1]
        combo = " ".join(f"{c:+.4f}*{cl['name'].split(' ')[0]}"
                         for c, cl in zip(null.tolist(), classes))
        print(f"  nullspace relation: {combo} = 0")
        out["nullspace"] = null.tolist()
    return out

results["rank_generic"] = rank_report(generic_F, 60, "GENERIC pair-antisym F")
results["rank_physical"] = rank_report(physical_F, 60, "PHYSICAL F(A)")


# --- 4. functional independence: Jacobian rank at a generic point ---------
def F_of_T(T):
    T = T - T.transpose(0, 1)
    return T - T.transpose(2, 3)


def F_of_B(B):
    A = B + B.transpose(-1, -2)                       # (4,4,4), symmetric
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)


def jac_rank(param, to_F, label):
    param = param.clone().requires_grad_(True)
    vals = invariants(to_F(param))
    rows = [torch.autograd.grad(v, param, retain_graph=True)[0].flatten()
            for v in vals]                            # (6, n_params)
    S = torch.linalg.svdvals(torch.stack(rows))
    rank = int((S > S[0] * 1e-10).sum())
    print(f"{label}: Jacobian rank {rank}/6, "
          f"sv min/max {(S[-1] / S[0]).item():.2e}")
    return {"rank": rank, "sv_min_over_max": (S[-1] / S[0]).item()}


print("\nJacobian ranks (functional independence at a generic point):")
results["jacobian_generic"] = jac_rank(
    torch.randn(4, 4, 4, 4, dtype=torch.float64), F_of_T, "  dI/dT, generic")
results["jacobian_physical"] = jac_rank(
    torch.randn(4, 4, 4, dtype=torch.float64), F_of_B, "  dI/dA, physical")


# --- 5. symmetric/antisymmetric channel identities ------------------------
# Phi = S + As and pair exchange F = Fs + Fa diagonalize the trace and
# no-trace sectors: I4/I5 = <S,S> +/- <As,As>, I1/I2 = <Fs,Fs> +/- <Fa,Fa>.
def ip2(X, Y):
    return torch.einsum("nb,nm,bc,mc->", X, ETA, ETA, Y)


def ip4(X, Y):
    return torch.einsum("mnab,mp,nq,ac,bd,pqcd->", X, ETA, ETA, ETA, ETA, Y)


I = invariants(F)
S_, As_ = (Phi + Phi.T) / 2, (Phi - Phi.T) / 2
Fs_, Fa_ = (F + F.permute(2, 3, 0, 1)) / 2, (F - F.permute(2, 3, 0, 1)) / 2
name_of = [c["name"].split(" ")[0] for c in classes]
ii = {n: k for k, n in enumerate(name_of)}
chan = {
    "I4 = <S,S>+<A,A>": I[ii["I4"]] - (ip2(S_, S_) + ip2(As_, As_)),
    "I5 = <S,S>-<A,A>": I[ii["I5"]] - (ip2(S_, S_) - ip2(As_, As_)),
    "I1 = <Fs,Fs>+<Fa,Fa>": I[ii["I1"]] - (ip4(Fs_, Fs_) + ip4(Fa_, Fa_)),
    "I2 = <Fs,Fs>-<Fa,Fa>": I[ii["I2"]] - (ip4(Fs_, Fs_) - ip4(Fa_, Fa_)),
}
print("\nchannel-basis identities (absolute residuals, values O(100)):")
for k, v in chan.items():
    print(f"  {k:22s} {v.abs().item():.1e}")
results["channel_identities"] = {k: v.abs().item() for k, v in chan.items()}

with open(os.path.join(HERE, "results", "numerical_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("\nwritten: results/numerical_results.json")
