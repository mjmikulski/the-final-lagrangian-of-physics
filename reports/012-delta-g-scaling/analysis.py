"""Producer 3 (CPU, committed artifacts): per-observable scaling
verdicts. For each observable X measured on the 3x3 grid of
(delta, g) -- delta in {1/8, 1/64, 1/512}, g in {8, 64, 512} -- this
computes (a) the maximal relative spread along delta at fixed g
("delta-flatness"; a flat observable is safe to extrapolate in delta),
and (b) the relative change along g at fixed delta, with a log-log
slope where the trend is monotone. Out: results/scaling_verdicts.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "results", "grid.json")))
pts = {(round(p["delta"], 6), p["g"]): p for p in G["points"]}
DS = [0.125, 0.015625, 0.001953]
GS = [8, 64, 512]
OBS = ["time_part_G", "time_part_eta", "om_pred", "C1", "C2",
       "I_pure", "I_comb", "mix34_curv", "depth", "E_stat"]

out = {"observables": {}}
for ob in OBS:
    rec = {}
    # delta-flatness per g
    flat = {}
    for g in GS:
        vals = [pts[(d, g)][ob] for d in DS]
        ref = max(abs(v) for v in vals)
        flat[str(g)] = (max(vals) - min(vals)) / ref if ref else 0.0
    rec["delta_spread_per_g"] = flat
    rec["delta_flat"] = max(flat.values()) < 0.02
    # g-trend at delta = 1/8
    vals = [pts[(0.125, g)][ob] for g in GS]
    rec["values_vs_g"] = vals
    rec["g_change_rel"] = (max(vals) - min(vals)) / max(
        abs(v) for v in vals)
    if all(v > 0 for v in vals) or all(v < 0 for v in vals):
        av = [abs(v) for v in vals]
        mono = (av[0] < av[1] < av[2]) or (av[0] > av[1] > av[2])
        rec["g_loglog_slope"] = float(np.polyfit(
            np.log(GS), np.log(av), 1)[0]) if mono else None
    else:
        rec["g_loglog_slope"] = None
    out["observables"][ob] = rec
    print(f"{ob}: delta-flat {rec['delta_flat']} "
          f"(max spread {max(flat.values()):.1%}); g-change "
          f"{rec['g_change_rel']:.1%}, slope {rec['g_loglog_slope']}")

# sign stability of the two time parts across the whole grid
signs_G = {np.sign(p["time_part_G"]) for p in G["points"]}
signs_e = {np.sign(p["time_part_eta"]) for p in G["points"]}
out["sign_stable_G"] = (signs_G == {-1.0})
out["sign_stable_eta"] = (signs_e == {1.0})
print("sign stability: G", out["sign_stable_G"],
      "| eta", out["sign_stable_eta"])

# precision across the grid
out["max_float32_rel"] = max(p["float32_rel"] for p in G["points"])
print(f"max float32 degradation: {out['max_float32_rel']:.1e}")

E = json.load(open(os.path.join(HERE, "results",
                                "extended_ladders_all.json")))
out["ladders"] = {str(c["g"]): {
    "interior": c["interior"], "min_omega": c["min_omega"],
    "om_pred": c["om_pred"],
    "min_over_pred": c["min_omega"] / c["om_pred"],
    "PR_at_min": next(r["PR"] for r in c["rows"]
                      if r["omega"] == c["min_omega"])}
    for c in E["cases"]}
for g, v in out["ladders"].items():
    print(f"ladder g={g}: interior {v['interior']}, min/pred "
          f"{v['min_over_pred']:.2f}, PR {v['PR_at_min']:.0f}")
json.dump(out, open(os.path.join(HERE, "results",
                                 "scaling_verdicts.json"), "w"),
          indent=1)
print("written: results/scaling_verdicts.json")
