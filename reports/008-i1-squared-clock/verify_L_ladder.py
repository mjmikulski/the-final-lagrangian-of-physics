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
for om in (0.2, 0.35, 0.5):
    assert abs(r32[om] - rec[om]) < 1e-5, (om, r32[om], rec[om])   # rung relaxations agree to ~1e-6 (not bitwise), the well and bracket exactly
# the box ladder: statics and the frozen-profile prediction move monotonically with L
chains = {N: json.load(open(os.path.join(RES, f"chain_N{N}.json"))) for N in (32, 48, 64)
          if os.path.exists(os.path.join(RES, f"chain_N{N}.json"))}
Es = [chains[N]["E_G_polished"] for N in sorted(chains)]
oms = [chains[N]["profile"]["omega_pred_E"] for N in sorted(chains)]
assert all(a > b for a, b in zip(Es, Es[1:])), Es
assert all(a > b for a, b in zip(oms, oms[1:])), oms
# N = 48: interior well, bracket stable at every level, minimum one rung below N = 32's
l48 = json.load(open(os.path.join(RES, "ladder_N48.json")))
assert l48["interior"] and l48["min_omega"] == 0.2 and set(l48["min_omega_per_level"]) == {0.2}
assert 5e-5 < l48["well_depth_vs_omega0"] < 8e-5
# N = 64: interior well, the sampled minimum at 0.2 at every level (as at N = 48), the depth NOT converged
# (it grows across the four L-BFGS cycles) -- recorded as such
l64 = json.load(open(os.path.join(RES, "ladder_N64.json")))
assert l64["interior"] and l64["min_omega"] == 0.2 and set(l64["min_omega_per_level"]) == {0.2}
assert all(c > 0 for c in l64["depth_changes"]), l64["depth_changes"]
assert l64["well_depth_vs_omega0"] > 2 * l48["well_depth_vs_omega0"]
print("N = 64: interior, min at", l64["min_omega"], "at all levels; depth", l64["well_depth_vs_omega0"], "(not plateaued)")
assert os.path.getsize(os.path.join(RES, "fig_L_ladder.png")) > 10000
print("L-ladder records consistent")
