"""Producer 3 (CPU, from committed relax_all.json): the scaling
analysis. Fits the L-exponents of (a) the combined inertia of the
relaxed CB fields, (b) the TOTAL static energy of EQ and CB, and
(c) the axial line-tension excess lambda_z - lambda_x -- the
statics-extensivity check requested at plan review (a finite inertia
must not hide an axial-defect energy growing faster than the
hedgehog's own linear-in-L radial-texture energy). Also reports the
h-test pair (N=16, h=1.5) vs (N=32, h=0.75) at L=24 and the
delta = 0.3 vs 1/8 axial costs. Out: results/analysis.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
ra = json.load(open(os.path.join(R, "relax_all.json")))
C = ra["cases"]

out = {}
Ls = [C[f"EQ_N{n}"]["Lbox"] for n in (16, 24, 32)]


def expo(vals):
    return float(np.polyfit(np.log(Ls), np.log(vals), 1)[0])


for kind in ("EQ", "CB"):
    rows = [C[f"{kind}_N{n}"] for n in (16, 24, 32)]
    out[f"I_comb_{kind}"] = [r["post"]["I_comb"] for r in rows]
    out[f"exponent_I_{kind}"] = expo(out[f"I_comb_{kind}"])
    out[f"E_{kind}"] = [r["E_final"] for r in rows]
    out[f"exponent_E_{kind}"] = expo(out[f"E_{kind}"])
    out[f"lam_excess_{kind}"] = [r["post"]["lambda_axis_excess"]
                                 for r in rows]
    out[f"lam_z_{kind}"] = [r["post"]["lambda_z"] for r in rows]
    out[f"lam_x_{kind}"] = [r["post"]["lambda_x"] for r in rows]
    out[f"PR_{kind}"] = [r["post"]["PR"] for r in rows]

# h-test at L = 24: physical quantities should be h-stable; the pure
# discretization residual of the equivariant background should shrink
h1 = C["EQ_N16"]["post"]
h2 = C["EQ_h075"]["post"]
c1 = C["CB_N16"]["post"]
c2 = C["CB_h075"]["post"]
out["h_test"] = {
    "EQ_I_h15": h1["I_comb"], "EQ_I_h075": h2["I_comb"],
    "CB_I_h15": c1["I_comb"], "CB_I_h075": c2["I_comb"],
    "CB_minus_EQ_h15": c1["I_comb"] - h1["I_comb"],
    "CB_minus_EQ_h075": c2["I_comb"] - h2["I_comb"]}

# delta comparison of the axial cost (N = 32)
out["axial_cost_delta"] = {
    "d0300": C["EQ_N32"]["post"]["lambda_axis_excess"],
    "d0125": C["EQ_d125"]["post"]["lambda_axis_excess"]}

for k, v in out.items():
    print(k, v)
json.dump(out, open(os.path.join(R, "analysis.json"), "w"), indent=1)
print("written: results/analysis.json")
