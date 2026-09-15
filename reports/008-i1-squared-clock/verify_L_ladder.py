"""Structural assertions on the committed L-ladder records (results/L_ladder/*.json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "L_ladder")
L008 = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))
rec = {r["omega"]: r["E_total"] for r in L008["JG_E"]["rungs"]}

# the reimplementation reproduces the committed 008 rung energies exactly
v = json.load(open(os.path.join(RES, "validate.json")))
assert v["pass"] and v["worst_rel_dev"] < 1e-9, v
# the N = 32 chain from the committed 2b seed reproduces the committed polished statics (bitwise on the recorded stack)
c32 = json.load(open(os.path.join(RES, "chain_N32.json")))
assert abs(c32["E_G_polished"] - L008["E_stat0"]) < 1e-9 * L008["E_stat0"], c32["E_G_polished"]
assert c32.get("seed_vs_committed_seed_maxdev", 1.0) == 0.0
# the analytic-seed shortcut does NOT reproduce it (recorded as the negative that forced the 2b regeneration)
ca = json.load(open(os.path.join(RES, "chain_N32_analytic_seed.json")))
assert ca["vs_committed_polished"]["max_abs_dM"] > 0.1 and abs(ca["profile"]["omega_pred_E"] - 0.326) > 0.05
# the N = 32 ladder reproduces 008's well: interior minimum at 0.35 at every protocol level, depth plateau
l32 = json.load(open(os.path.join(RES, "ladder_N32.json")))
assert l32["interior"] and l32["min_omega"] == 0.35 and set(l32["min_omega_per_level"]) == {0.35}
r32 = {r["omega"]: r["E_total"] for r in l32["rungs"]}
for om in rec:
    assert r32[om] == rec[om], (om, r32[om], rec[om])   # on the corrected stack (chain's shell) the ladder is bitwise
assert l32["depth_per_level"] == L008["JG_E"]["depth_per_level"]
# the bracket rerun of every box: N = 32 bitwise against 008; the others recorded with persisted fields
for N in (32, 48, 64):
    f = os.path.join(RES, f"rungs_N{N}.json")
    if os.path.exists(f):
        rr = {r["omega"]: r["E_total"] for r in json.load(open(f))["rungs"]}
        assert set(rr) == {0.0, 0.1, 0.2, 0.35}, rr.keys()
        if N == 32:
            assert all(rr[om] == rec[om] for om in rr), "N = 32 bracket must be bitwise against 008"
# the independent numpy route on the persisted brackets (verify_L_ladder_energies.py --strict) is run by reproduce_appendix_L.sh
# the box ladder: statics and the frozen-profile prediction move monotonically with L
chains = {N: json.load(open(os.path.join(RES, f"chain_N{N}.json"))) for N in (32, 48, 64)
          if os.path.exists(os.path.join(RES, f"chain_N{N}.json"))}
Es = [chains[N]["E_G_polished"] for N in sorted(chains)]
oms = [chains[N]["profile"]["omega_pred_E"] for N in sorted(chains)]
assert all(a > b for a, b in zip(Es, Es[1:])), Es
assert all(a > b for a, b in zip(oms, oms[1:])), oms
# N = 48 and 64, first seven-rung run (analytic shell): recorded as the exploration
l48 = json.load(open(os.path.join(RES, "ladder_N48.json")))
assert l48["interior"] and l48["min_omega"] == 0.2
l64 = json.load(open(os.path.join(RES, "ladder_N64.json")))
assert l64["interior"] and l64["min_omega"] == 0.2
# the certified brackets (chain shell, persisted fields): a rung below omega = 0 in every box; the L = 72 pair
# within 2e-5 of each other; the L = 96 minimum at 0.2 at every level with both neighbours above omega = 0
b48 = json.load(open(os.path.join(RES, "rungs_N48.json")))
e48 = {r["omega"]: r["E_total"] for r in b48["rungs"]}
assert e48[0.2] - e48[0.0] < -5e-5 and e48[0.35] - e48[0.0] < -5e-5 and abs(e48[0.2] - e48[0.35]) < 2e-5, e48
assert set(b48["min_omega_per_level"]) <= {0.2, 0.35}
b64 = json.load(open(os.path.join(RES, "rungs_N64.json")))
e64 = {r["omega"]: r["E_total"] for r in b64["rungs"]}
assert set(b64["min_omega_per_level"]) == {0.2} and e64[0.2] - e64[0.0] < -1e-4, e64
assert e64[0.1] > e64[0.0] and e64[0.35] > e64[0.0]
# the independent route (run by reproduce_appendix_L.sh) must have matched every persisted energy
ir = json.load(open(os.path.join(RES, "independent_route.json")))
assert ir["worst_rel_dev"] < 1e-9, ir["worst_rel_dev"]
print("brackets: L = 48 min 0.35 (bitwise vs 008); L = 72 min", b48["min_omega"], "with the 0.2/0.35 pair within 2e-5;",
      "L = 96 min 0.2 at every level; independent route worst relative deviation", ir["worst_rel_dev"])
assert os.path.getsize(os.path.join(RES, "fig_L_ladder.png")) > 10000
print("L-ladder records consistent")
