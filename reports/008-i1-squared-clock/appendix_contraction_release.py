"""Two tests of the (I1^G)^2 well proposed by the model author on substrate-framework discussion #186 (2026-09-19):

(i)  "rerun one rung on the positive (H-adjoint / Frobenius) contraction -- if the well is the negative
     [rotation-boost] block, it disappears";
(ii) "at the well's omega, release the profile and see whether E keeps falling".

Stages (resumable; JSON per stage in results/contraction_release/):
  structure  on the polished field and the persisted bracket rungs of the report's box (N = 32): the off-block
             entries M_0i, the deviation of the working metric G from the identity, and the split of the clock
             channel |F_0i|^2 into time-space and space-space matrix entries (CPU or GPU, seconds)
  frob       the JG_E ladder with the Frobenius (identity) matrix contraction in both densities of the (I1)^2 term,
             statics in G, same gamma and fixed-depth protocol as the report (the report's J_ETA ladder changes the
             same term to eta in the same way)
  release    the tangent recomputed from the current field inside the energy (a0 = a0(M), the same envelope and
             normalisation) instead of the frozen a0 of the polished field; rungs 0.2 and 0.35 with a longer
             protocol (eight L-BFGS cycles), against the frozen-tangent rungs 0 and 0.35 run with the same
             protocol (their first five levels must reproduce the committed bracket record bitwise)
  all        the three stages in order
"""
import json
import os
import sys
import time

import numpy as np
import torch

from lattice_L import Stack, DT

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.join(HERE, "results", "L_ladder")
RES = os.path.join(HERE, "results", "contraction_release")
os.makedirs(RES, exist_ok=True)
GAMMA = json.load(open(os.path.join(HERE, "results", "i1sq_ladders.json")))["gamma"]
N = 32
SEED = os.path.join(LL, "seeds", f"m5_21_2b_end_A_T2_sym_e0_n{N}_d0.3_pinned.npz")
FROB_OMEGAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.8)


def jdump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def stack():
    return Stack(N, seed3=np.load(SEED)["M"].astype(np.float64))


def densities(st, M, a0, om, metric):
    """(static I1 density, clock density k >= 0) with the matrix slots contracted by metric in
    {'G', 'frob', 'eta'}; for 'eta' k is the magnitude that enters I1 with a plus (report section 1)."""
    if metric != "frob":
        return st.densities(M, a0, om, metric)
    I4 = torch.eye(4, dtype=DT, device=st.dev)
    V = om * a0
    i1s = torch.zeros(M.shape[:3], dtype=DT, device=st.dev)
    k = torch.zeros(M.shape[:3], dtype=DT, device=st.dev)
    for s in ("fwd", "bwd"):
        A = [st.d1(M, ax, s) for ax in range(3)]
        for i in range(3):
            k = k + 0.5 * 4.0 * st.inner_X(st.comm(V, A[i]), I4)
            for j in range(i + 1, 3):
                i1s = i1s + 0.5 * 4.0 * st.inner_X(st.comm(A[i], A[j]), I4)
    return i1s, k


def polished(st):
    return torch.tensor(np.load(os.path.join(LL, f"M_G_polished_N{N}.npz"))["M"], dtype=DT, device=st.dev)


def structure():
    st = stack()
    Mr0 = polished(st)
    Mg = st.field(Mr0)
    a0 = st.a0_of(st.gen_boost_x(), Mg)
    out = {"fields": {}}
    I4 = torch.eye(4, dtype=DT, device=st.dev)
    fields = {"polished": Mg}
    for tag in ("00", "01", "02", "035"):
        fields[f"rung {tag}"] = torch.tensor(np.load(os.path.join(LL, f"jge_N{N}_om{tag}.npz"))["M"], dtype=DT, device=st.dev)
    for name, M in fields.items():
        G = st.G_of(M)
        free = st.FREE
        V = a0
        T = S = 0.0
        for s in ("fwd", "bwd"):
            A = [st.d1(M, ax, s) for ax in range(3)]
            for i in range(3):
                F0 = st.comm(V, A[i])
                T += float((F0[..., 0, 1:] ** 2).sum() + (F0[..., 1:, 0] ** 2).sum() + (F0[..., 0, 0] ** 2).sum())
                S += float((F0[..., 1:, 1:] ** 2).sum())
        i1_G, k_G = densities(st, M, a0, 1.0, "G")
        i1_F, k_F = densities(st, M, a0, 1.0, "frob")
        i1_E, k_E = densities(st, M, a0, 1.0, "eta")
        out["fields"][name] = {
            "offblock_max": float(M[..., 0, 1:4].abs().max()),
            "G_minus_identity_max_free": float((G - I4)[free].abs().max()),
            "M00_plus_g_max": float((M[..., 0, 0] + 8.0).abs().max()),
            "clock_channel_time_space_part": T, "clock_channel_space_space_part": S,
            "int_i1s_G": float(i1_G.sum() * st.H ** 3), "int_i1s_frob": float(i1_F.sum() * st.H ** 3),
            "int_i1s_eta": float(i1_E.sum() * st.H ** 3),
            "int_k_G": float(k_G.sum() * st.H ** 3), "int_k_frob": float(k_F.sum() * st.H ** 3),
            "int_k_eta": float(k_E.sum() * st.H ** 3)}
        print(name, json.dumps(out["fields"][name]), flush=True)
    jdump(out, "structure.json")
    return out


def run(tag, omegas, e_total_of, cycles, Mr0, st, extra_record=None):
    rows = []
    t0 = time.time()
    for om in omegas:
        M_raw, E_levels, ginf = st.relax_rung(lambda Mr, om=om: e_total_of(Mr, om), Mr0, cycles=cycles)
        Mf = st.field(M_raw)
        row = {"omega": om, "E_total": float(e_total_of(M_raw, om).detach()), "E_levels": E_levels, "grad_inf": ginf,
               "offblock_max": float(Mf[..., 0, 1:4].abs().max()),
               "max_dM_from_start": float((Mf - st.field(Mr0)).abs().max())}
        if extra_record:
            row.update(extra_record(Mf, om))
        rows.append(row)
        print(f"  [{tag}] omega {om}: E {row['E_total']:.9f}, |g|inf {ginf:.1e}, levels "
              f"{['%.9f' % e for e in E_levels]} [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(os.path.join(RES, f"{tag}_om{str(om).replace('.', '')}.npz"), M=Mf.float().cpu().numpy())
        jdump({"tag": tag, "gamma": GAMMA, "cycles": cycles, "rungs": rows}, f"{tag}.json")
    return rows


def frob():
    st = stack()
    Mr0 = polished(st)
    Mg = st.field(Mr0)
    a0 = st.a0_of(st.gen_boost_x(), Mg)
    i1s0, k1 = densities(st, Mg, a0, 1.0, "frob")
    C1 = float((i1s0 * k1).sum() * st.H ** 3)
    C2 = float((k1 ** 2).sum() * st.H ** 3)

    def e_total(Mr, om):
        Mf = st.field(Mr)
        i1s, k = densities(st, Mf, a0, om, "frob")
        return st.e_static(Mf, "G") + GAMMA * st.H ** 3 * ((i1s - k) ** 2).sum()    # statics in G, as every ladder of the report

    rows = run("frob", FROB_OMEGAS, e_total, 4, Mr0, st)
    rec = {"tag": "frob", "gamma": GAMMA, "cycles": 4, "omega_pred_E": (C1 / C2) ** 0.5, "C1": C1, "C2": C2, "rungs": rows}
    nl = len(rows[0]["E_levels"])
    rec["min_omega_per_level"] = [rows[min(range(len(rows)), key=lambda i: rows[i]["E_levels"][lv])]["omega"] for lv in range(nl)]
    rec["depth_per_level"] = [rows[0]["E_levels"][lv] - min(r["E_levels"][lv] for r in rows) for lv in range(nl)]
    jdump(rec, "frob.json")
    print("frob verdict:", rec["min_omega_per_level"], rec["depth_per_level"], flush=True)
    return rec


def release():
    st = stack()
    Mr0 = polished(st)
    Mg = st.field(Mr0)
    W = st.gen_boost_x()
    a0_frozen = st.a0_of(W, Mg)

    def e_frozen(Mr, om):
        Mf = st.field(Mr)
        i1s, k = st.densities(Mf, a0_frozen, om, "G")
        return st.e_static(Mf, "G") + GAMMA * st.H ** 3 * ((i1s - k) ** 2).sum()

    def e_released(Mr, om):
        Mf = st.field(Mr)
        i1s, k = st.densities(Mf, st.a0_of(W, Mf), om, "G")
        return st.e_static(Mf, "G") + GAMMA * st.H ** 3 * ((i1s - k) ** 2).sum()

    def diag(Mf, om):
        a = st.a0_of(W, Mf)
        _, k = st.densities(Mf, a, max(om, 1e-9), "G")
        pr = float((k.sum() ** 2) / (k ** 2).sum().clamp_min(1e-30))
        return {"tangent_change_vs_frozen": float((a - a0_frozen).norm()), "PR_k_sites": pr}

    fr = run("frozen8", (0.0, 0.35), e_frozen, 8, Mr0, st, diag)
    rl = run("released8", (0.2, 0.35), e_released, 8, Mr0, st, diag)
    committed = {r["omega"]: r for r in json.load(open(os.path.join(LL, f"rungs_N{N}.json")))["rungs"]}
    out = {"frozen8": fr, "released8": rl,
           "frozen_first_levels_vs_committed": {str(r["omega"]): max(abs(a - b) for a, b in zip(r["E_levels"][:5], committed[r["omega"]]["E_levels"]))
                                                for r in fr}}
    e0 = fr[0]["E_levels"]
    out["depth_per_level"] = {r["tag"] if "tag" in r else f"{name} {r['omega']}": [e0[lv] - r["E_levels"][lv] for lv in range(len(e0))]
                              for name, rows in (("frozen", fr[1:]), ("released", rl)) for r in rows}
    out["omega0_creep_per_cycle"] = [e0[lv + 1] - e0[lv] for lv in range(len(e0) - 1)]
    jdump(out, "release.json")
    print("release:", json.dumps({k: out[k] for k in ("frozen_first_levels_vs_committed", "depth_per_level")}), flush=True)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
    if what in ("structure", "all"):
        structure()
    if what in ("frob", "all"):
        frob()
    if what in ("release", "all"):
        release()
