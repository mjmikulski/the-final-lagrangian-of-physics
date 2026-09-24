"""Assertions on the committed records of APPENDIX-contraction-release (structure, not floating-point tails)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "contraction_release")
s = json.load(open(os.path.join(RES, "structure.json")))["fields"]
for name, f in s.items():
    assert f["offblock_max"] == 0.0, "M_0i vanish identically on the report's fields"
    assert f["clock_channel_space_space_part"] == 0.0 and f["clock_channel_time_space_part"] > 0, \
        "the clock channel [a0, d_i M] has time-space entries only"
    assert f["int_k_frob"] == f["int_k_eta"] and f["int_i1s_frob"] == f["int_i1s_eta"], \
        "Frobenius and eta densities coincide in magnitude on this sector"
    assert f["G_minus_identity_max_free"] < 1e-3, "the working metric G is the identity up to 4e-4"

fr = json.load(open(os.path.join(RES, "frob.json")))
assert fr["min_omega_per_level"] == [0.35] * 5, "Frobenius: the minimum sits at 0.35 at every protocol level"
assert all(5e-5 < d < 8e-5 for d in fr["depth_per_level"]), "Frobenius depth within the report's range"
rows = {r["omega"]: r for r in fr["rungs"]}
assert rows[0.5]["E_total"] > rows[0.0]["E_total"] and rows[0.8]["E_total"] > rows[0.0]["E_total"]

rel = json.load(open(os.path.join(RES, "release.json")))
d = rel["depth_per_level"]
for key in ("frozen 0.35", "released 0.35", "released 0.2"):
    assert all(x > 2e-5 for x in d[key]), f"{key}: below omega = 0 at every level"
    assert max(d[key]) < 2 * min(d[key]), f"{key}: the depth does not grow with the protocol (no runaway)"
creep = rel["omega0_creep_per_cycle"]
assert all(-1e-5 < c < 0 for c in creep), "omega = 0 keeps creeping by 1e-6 to 1e-5 per cycle, the protocol floor"
print("verify_appendix_contraction: all assertions pass")
