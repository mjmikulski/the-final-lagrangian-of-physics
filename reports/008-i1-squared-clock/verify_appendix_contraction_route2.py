"""Route 2 for APPENDIX-contraction-release: the report's from-scratch numpy energy route (verify_L_ladder_energies.py,
no torch) evaluated on the persisted fields of this appendix (float32, so energies agree with the float64 records
up to a common offset of 1.7e-5 from the float32 rounding of the stored fields; the offset is the same for every
field to 1e-7, so every energy difference is reproduced to that level, well below the depths of 3-7e-5).

Frobenius ladder: the same energy with the identity in place of G in both densities of the (I1)^2 term (statics
in G). Frozen tangent: rebuilt from the polished field. Released tangent: rebuilt from the evaluated field itself.
Asserts the qualitative content in this route: the 0.35 rung lies below omega = 0 by more than 5e-5 for the
Frobenius ladder, for the frozen and for the released tangent, and the 0.5 and 0.8 Frobenius rungs lie above.
Writes results/contraction_release/route2.json.
"""
import json
import os

import numpy as np

import verify_L_ladder_energies as v

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "contraction_release")
I4 = np.eye(4)


def e_frob(M, a0, om):
    G = v.G_of(M)
    i1G, i1F, kF = 0.0, 0.0, 0.0
    for st in ("fwd", "bwd"):
        A = [v.d1(M, ax, st) for ax in range(3)]
        for i in range(3):
            kF = kF + 0.5 * 4.0 * v.inner_pc(v.comm(om * a0, A[i]), np.broadcast_to(I4, M.shape))
            for j in range(i + 1, 3):
                F = v.comm(A[i], A[j])
                i1G = i1G + 0.5 * 4.0 * v.inner_pc(F, G)
                i1F = i1F + 0.5 * 4.0 * v.inner_pc(F, np.broadcast_to(I4, M.shape))
    Es = v.Hh ** 3 * (i1G.sum() + v.W1 * v.v4_density(M).sum())
    return Es + v.GAMMA * v.Hh ** 3 * ((i1F - kF) ** 2).sum()


def load(name):
    return np.load(os.path.join(RES, name))["M"].astype(np.float64)


if __name__ == "__main__":
    seed = np.load(os.path.join(HERE, "results", "L_ladder", "seeds", "m5_21_2b_end_A_T2_sym_e0_n32_d0.3_pinned.npz"))["M"]
    a0 = v.tangent(v.pinned_field(np.load(os.path.join(HERE, "results", "L_ladder", "M_G_polished_N32.npz"))["M"].astype(np.float64),
                                  32, seed.astype(np.float64)))
    out = {"frob": {}, "frozen8": {}, "released8": {}}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "frob.json")))["rungs"]}
    for om in (0.0, 0.1, 0.2, 0.35, 0.5, 0.8):
        E = float(e_frob(load(f"frob_om{str(om).replace('.', '')}.npz"), a0, om))
        out["frob"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "frozen8.json")))["rungs"]}
    for om in (0.0, 0.35):
        E = float(v.energy(load(f"frozen8_om{str(om).replace('.', '')}.npz"), a0, om))
        out["frozen8"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    rec = {r["omega"]: r["E_total"] for r in json.load(open(os.path.join(RES, "released8.json")))["rungs"]}
    for om in (0.2, 0.35):
        M = load(f"released8_om{str(om).replace('.', '')}.npz")
        E = float(v.energy(M, v.tangent(M), om))
        out["released8"][str(om)] = {"E": E, "record": rec[om], "dev": abs(E - rec[om])}
    devs = [x["E"] - x["record"] for grp in out.values() for x in grp.values()]
    out["common_offset"] = float(np.mean(devs))           # float32 storage of the fields: the same for every field
    out["offset_spread"] = float(max(devs) - min(devs))   # what matters for energy differences
    f, fz, rl = out["frob"], out["frozen8"], out["released8"]
    out["depths"] = {"frob 0.35": f["0.0"]["E"] - f["0.35"]["E"], "frozen 0.35": fz["0.0"]["E"] - fz["0.35"]["E"],
                     "released 0.35": fz["0.0"]["E"] - rl["0.35"]["E"], "released 0.2": fz["0.0"]["E"] - rl["0.2"]["E"]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "route2.json"), "w"), indent=1)
    assert out["offset_spread"] < 2e-7, "route 2 reproduces every energy difference of the records (float32 fields)"
    assert all(out["depths"][k] > 5e-5 for k in ("frob 0.35", "frozen 0.35", "released 0.35"))
    assert out["depths"]["released 0.2"] > 2e-5
    assert f["0.5"]["E"] > f["0.0"]["E"] and f["0.8"]["E"] > f["0.0"]["E"]
    print("route 2: all assertions pass")
