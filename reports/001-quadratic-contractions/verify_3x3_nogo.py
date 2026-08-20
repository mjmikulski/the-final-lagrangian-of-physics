"""Independent verification of the substrate-framework P239 3x3 no-go.

Source of the claims (not ours): vantasnerdan/substrate-framework, campaign
P239, attempt 0001 (PR #148, head 1c63909), evidence/quadratic-basis-note.md.
They state: (a) on purely spatial fields the six invariants have rank 3 with
nullspace N1 = I3 - (I1+I2)/4, N2 = (I1-I2)/4 - I4 + I5, N3 = I1 - 4 I4 + I6;
(b) the clock direction A_0 = w*diag(1,0,0,0), A_1 = E01 + E10 gives
(I1..I6) = w^2 (4,4,2,2,2,4), on which N1 = N2 = N3 = 0 while -I1 = -4 w^2,
so no constant-coefficient quadratic action preserving the exact 3x3 sector
can repair the negative clock channel.

This script re-checks both with OUR evaluator (built in check_torch.py),
i.e. a fully independent implementation of the invariants.
"""
import json
import os
import string

import torch

torch.manual_seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
ETA = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

classes = json.load(open(os.path.join(HERE, "results",
                                      "contraction_classes.json")))["classes"]
REP = {c["name"].split(" ")[0]: (c["natural_sign"] * c["members"][0]["sign"],
                                 c["members"][0]["pairs"]) for c in classes}


def contract(F, pairs):
    L = string.ascii_lowercase[:8]
    ops, spec = [F, F], [L[:4], L[4:]]
    for p, q in pairs:
        ops.append(ETA)
        spec.append(L[p] + L[q])
    return torch.einsum(",".join(spec) + "->", *ops)


def invariants(F):
    return {k: (s * contract(F, p)).item() for k, (s, p) in REP.items()}


def F_of_A(A):
    AeA = torch.einsum("mac,cd,ndb->mnab", A, ETA, A)
    return AeA - AeA.transpose(0, 1)


def nullspace(I):
    return (I["I3"] - (I["I1"] + I["I2"]) / 4,
            (I["I1"] - I["I2"]) / 4 - I["I4"] + I["I5"],
            I["I1"] - 4 * I["I4"] + I["I6"])


# (a) N1..N3 vanish identically on purely spatial fields
worst = 0.0
for _ in range(20):
    A = torch.zeros(4, 4, 4, dtype=torch.float64)
    r = torch.randn(3, 3, 3, dtype=torch.float64)
    A[1:, 1:, 1:] = r + r.transpose(-1, -2)
    I = invariants(F_of_A(A))
    scale = max(abs(v) for v in I.values())
    worst = max(worst, max(abs(n) for n in nullspace(I)) / scale)
print(f"(a) spatial nullspace: worst relative residual {worst:.1e}")
assert worst < 1e-12

# (b) the clock-direction counterexample
w = 1.7
A = torch.zeros(4, 4, 4, dtype=torch.float64)
A[0] = w * torch.diag(torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))
A[1, 0, 1] = A[1, 1, 0] = 1.0
I = invariants(F_of_A(A))
got = {k: I[k] / w ** 2 for k in ("I1", "I2", "I3", "I4", "I5", "I6")}
expect = {"I1": 4, "I2": 4, "I3": 2, "I4": 2, "I5": 2, "I6": 4}
print(f"(b) counterexample I_k / w^2: { {k: round(v, 10) for k, v in got.items()} }")
assert all(abs(got[k] - expect[k]) < 1e-12 for k in expect)
N = nullspace(I)
print(f"    N1, N2, N3 on it: {[round(n, 12) for n in N]}")
assert all(abs(n) < 1e-12 for n in N)

print("VERIFIED: both P239 claims confirmed by our independent evaluator.")
