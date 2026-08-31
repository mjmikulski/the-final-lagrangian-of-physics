"""Producer 2 (CPU): digest of the deep-continuation record -- final
bracket differences, drift rates over the last five cycles, and the
crossing cycles (where each difference changes sign), per arm.
Out: results/verdicts.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(HERE, "results", "window_deep.json")))

out = {"arms": {}}
for tag, arm in W["arms"].items():
    h = arm["history"]
    E = {om: h[om]["E"] for om in ("0.1", "0.15", "0.2", "0.28")}
    n = len(E["0.15"])
    d = {om: [E[om][i] - E["0.15"][i] for i in range(n)]
         for om in ("0.1", "0.2", "0.28")}
    rec = {"cycles": n - 1, "final_diffs": {k: v[-1] for k, v in d.items()},
           "drift_last5": {k: (v[-1] - v[-6]) / 5 for k, v in d.items()},
           "crossing_cycle": {}}
    for k, v in d.items():
        cc = None
        for i in range(1, n):
            if v[i - 1] > 0 >= v[i]:
                cc = i
                break
        rec["crossing_cycle"][k] = cc
    rec["verdict"] = arm["verdict"]
    rec["ginf_final"] = {om: h[om]["ginf"][-1] for om in h}
    out["arms"][tag] = rec
    print(tag, json.dumps(rec["final_diffs"]),
          "crossings", rec["crossing_cycle"],
          "drift5", {k: f"{v:+.1e}" for k, v in rec["drift_last5"].items()})
json.dump(out, open(os.path.join(HERE, "results", "verdicts.json"),
                    "w"), indent=1)
print("written: results/verdicts.json")
