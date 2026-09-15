"""Structural assertions on the committed L-ladder records (results/L_ladder/*.json).

Two kinds of record are told apart by their "shell" key: the preserved exploration (the first seven-rung runs
at N = 48 and 64, analytic ansatz on the pinned shell) keeps its recorded verdicts; everything with the chain's
seed on the shell (the N = 32 ladder, the bracket records, and whatever `all --fresh` regenerates) is checked
against the statements the appendix makes about such runs, with the observed run-to-run sensitivity in mind.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "L_ladder")
L008 = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))
rec = {r["omega"]: r["E_total"] for r in L008["JG_E"]["rungs"]}
FIRST_RUN = "analytic"


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


# the reimplementation reproduces the committed 008 rung energies exactly
v = load("validate.json")
assert v["pass"] and v["worst_rel_dev"] < 1e-9, v
# the N = 32 chain from the committed 2b seed reproduces the committed polished statics (bitwise on the recorded stack)
c32 = load("chain_N32.json")
assert abs(c32["E_G_polished"] - L008["E_stat0"]) < 1e-9 * L008["E_stat0"], c32["E_G_polished"]
assert c32.get("seed_vs_committed_seed_maxdev", 1.0) == 0.0
print("chain N = 32: E_stat", "bitwise" if c32["E_G_polished"] == L008["E_stat0"] else "within 1e-9", "against 008")
# the analytic-seed shortcut does NOT reproduce it (recorded as the negative that forced the 2b regeneration)
ca = load("chain_N32_analytic_seed.json")
assert ca["vs_committed_polished"]["max_abs_dM"] > 0.1 and abs(ca["profile"]["omega_pred_E"] - 0.326) > 0.05
# the N = 32 ladder (chain seed on the shell) reproduces 008's ladder: minimum 0.35 at every level, rung energies
# to 1e-9 relative (bitwise on the recorded stack), the depth plateau
l32 = load("ladder_N32.json")
assert FIRST_RUN not in l32.get("shell", ""), "ladder_N32.json must be the chain-seed run"
assert l32["interior"] and l32["min_omega"] == 0.35 and set(l32["min_omega_per_level"]) == {0.35}
r32 = {r["omega"]: r["E_total"] for r in l32["rungs"]}
for om in rec:
    assert abs(r32[om] - rec[om]) < 1e-9 * abs(rec[om]), (om, r32[om], rec[om])
print("ladder N = 32:", "bitwise" if all(r32[om] == rec[om] for om in rec) else "within 1e-9", "against 008's seven rungs")
# the box ladder: statics and the frozen-profile prediction move monotonically with L
chains = {N: load(f"chain_N{N}.json") for N in (32, 48, 64)}
Es = [chains[N]["E_G_polished"] for N in sorted(chains)]
oms = [chains[N]["profile"]["omega_pred_E"] for N in sorted(chains)]
assert all(a > b for a, b in zip(Es, Es[1:])), Es
assert all(a > b for a, b in zip(oms, oms[1:])), oms
# the seven-rung records at N = 48 and 64
for N in (48, 64):
    lad = load(f"ladder_N{N}.json")
    assert lad["interior"], (N, lad["min_omega"])
    if FIRST_RUN in lad.get("shell", ""):
        # the preserved exploration: verdicts as recorded (0.2 at every level in both boxes)
        assert lad["min_omega"] == 0.2 and set(lad["min_omega_per_level"]) == {0.2}, (N, lad["min_omega_per_level"])
        print(f"ladder N = {N}: first run (analytic shell), minimum 0.2 as recorded")
    else:
        # a regenerated run with the chain's seed: the appendix's statement for such runs
        assert lad["min_omega"] in (0.2, 0.35), (N, lad["min_omega"])
        print(f"ladder N = {N}: chain-seed run, minimum {lad['min_omega']}")
# the certified brackets (chain seed, persisted fields): the bracket rungs exactly; a rung below omega = 0 in every
# box; at L = 72 the minimum in the 0.2/0.35 pair with depth above 4e-5; at L = 96 the minimum at 0.2 with depth
# above 1e-4 (both hold in the two runs on record, whose E(omega) - E(0) differ by up to 4e-5 and 1.5e-4)
for N in (32, 48, 64):
    b = load(f"rungs_N{N}.json")
    assert FIRST_RUN not in b.get("shell", ""), (N, "bracket record must come from the chain-seed stack")
    e = {r["omega"]: r["E_total"] for r in b["rungs"]}
    assert set(e) == {0.0, 0.1, 0.2, 0.35}, e.keys()
    kmin = min(e, key=e.get)
    depth = e[0.0] - e[kmin]
    assert kmin != 0.0 and depth > 0, (N, e)
    if N == 32:
        assert all(abs(e[om] - rec[om]) < 1e-9 * abs(rec[om]) for om in e), "N = 32 bracket must reproduce 008"
    if N == 48:
        assert kmin in (0.2, 0.35) and depth > 4e-5, (kmin, depth)
    if N == 64:
        assert kmin == 0.2 and depth > 1e-4, (kmin, depth)
    print(f"bracket N = {N}: minimum {kmin}, depth {depth:.2e}")
# the independent route (run by reproduce_appendix_L.sh) must have matched every persisted energy
ir = load("independent_route.json")
assert ir["worst_rel_dev"] < 1e-9, ir["worst_rel_dev"]
assert os.path.getsize(os.path.join(RES, "fig_L_ladder.png")) > 10000
print("L-ladder records consistent; independent route worst relative deviation", ir["worst_rel_dev"])
