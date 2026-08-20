"""Phase 1 (symbolic): enumerate quadratic scalar contractions of F_abcd.

F is the M5 field strength F_{mn|ab} = (d_m M eta d_n M - d_n M eta d_m M)_ab
with M symmetric: antisymmetric in (m,n) AND in (a,b), no further slot
symmetries (no pair exchange, no Bianchi -- established numerically in
check_torch.py). All 8 indices of F x F are paired with the metric; the
metric is symmetric, so every pairing is equivalent to one up/down dummy
pair. Canonicalization (Butler-Portugal via canon_bp) quotients by dummy
relabeling, both antisymmetries, and exchange of the two identical factors.

Out: results/contraction_classes.json (one entry per nonzero class: representative
pairing, all member pairings with signs, multiplicity).
"""
import itertools
import json
import os

from sympy.tensor.tensor import (TensorHead, TensorIndexType, TensorSymmetry,
                                 tensor_indices)

HERE = os.path.dirname(os.path.abspath(__file__))

L = TensorIndexType("L", dim=4)
F = TensorHead("F", [L] * 4, TensorSymmetry.direct_product(-2, -2))
IDX = tensor_indices("i0:8", L)


def matchings(slots):
    """All perfect matchings of slot list; 105 for 8 slots."""
    if not slots:
        yield []
        return
    a, rest = slots[0], slots[1:]
    for k in range(len(rest)):
        for m in matchings(rest[:k] + rest[k + 1:]):
            yield [(a, rest[k])] + m


def expr_of(matching):
    ind = [None] * 8
    for d, (p, q) in enumerate(matching):
        ind[p] = IDX[d]
        ind[q] = -IDX[d]
    return F(*ind[:4]) * F(*ind[4:])


def sign_split(canon):
    """(key modulo overall sign, sign); canon_bp folds -1 into the coeff."""
    s_pos, s_neg = str(canon), str(-canon)
    return (s_pos, +1) if s_pos <= s_neg else (s_neg, -1)


# named candidates to label the classes found by brute force
Phi_note = "Phi_nb := eta^{ma} F_{mn a b}  (single trace; no symmetry in nb)"
i0, i1, i2, i3, i4, i5 = IDX[:6]
CANDIDATES = {
    "I1 = F_abcd F^abcd (current kinetic term)":
        F(i0, i1, i2, i3) * F(-i0, -i1, -i2, -i3),
    "I2 = F_abcd F^cdab (pair exchange)":
        F(i0, i1, i2, i3) * F(-i2, -i3, -i0, -i1),
    "I3 = F_abcd F^acbd (mixed pairing)":
        F(i0, i1, i2, i3) * F(-i0, -i2, -i1, -i3),
    "I4 = Phi_ab Phi^ab":
        F(i0, i1, -i0, i3) * F(i4, -i1, -i4, -i3),
    "I5 = Phi_ab Phi^ba":
        F(i0, i1, -i0, i3) * F(i4, -i3, -i4, -i1),
    "I6 = Phi^2 (full double trace, squared)":
        F(i0, i1, -i0, -i1) * F(i4, i5, -i4, -i5),
}

classes = {}
n_zero = 0
for m in matchings(list(range(8))):
    canon = expr_of(m).canon_bp()
    if canon == 0:
        n_zero += 1
        continue
    key, sgn = sign_split(canon)
    classes.setdefault(key, []).append({"pairs": m, "sign": sgn})

names = {}
for name, cand in CANDIDATES.items():
    key, s_c = sign_split(cand.canon_bp())
    assert key in classes, f"candidate not found by enumeration: {name}"
    names[key] = (name, s_c)   # s_c: candidate = s_c * canonical-key expression

print(f"pairings: 105 total, {n_zero} vanish identically, "
      f"{len(classes)} distinct nonzero classes\n")
out = []
for key, members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
    name, s_c = names.get(key, ("UNNAMED", 1))
    print(f"[{len(members):3d} pairings]  {name}\n     canonical: {key}")
    out.append({"name": name, "canonical": key, "natural_sign": s_c,
                "members": members})
assert len(names) == len(classes), "some class has no named candidate"

with open(os.path.join(HERE, "results", "contraction_classes.json"), "w") as f:
    json.dump({"n_zero": n_zero, "classes": out, "phi_note": Phi_note}, f,
              indent=1)
print("\nwritten: results/contraction_classes.json")
