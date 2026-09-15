"""The L-ladder at fixed spacing for the (I1^G)^2 energy-functional clock of report 008.

OpenWave's M5.32 ledger (2026-09-02) rates report 008's well as "the only convergence-certified localized
clock on the original 4x4 field" and asks for the one thing 008 could not see in a single 32^3 box:
"an L-ladder at fixed h is cheap and decisive". This script runs the 004 -> 008 chain at N = 32, 48, 64
with h = 1.5 (L = 48, 72, 96): analytic 2b seed -> eta relax -> G relax -> polish -> JG_E ladder with the
fixed-depth protocol of 008 (500 Adam + four L-BFGS cycles per rung, every rung including omega = 0), at
the committed coupling gamma of 008 (a fixed Lagrangian across the ladder). Recorded per N: the
frozen-profile prediction omega_E = sqrt(C1/C2), the rung energies at every protocol level, the bracket
(min_omega_per_level), the depth of the well relative to omega = 0 per level, the participation ratio of
the ticking density, the far-field slope of the static density and the residuals.

Stages (resumable, JSON per stage in results/):
  validate   the reimplementation against the committed 004/008 artifacts at N = 32 (energies of the
             committed polished field and of the persisted rung fields; gamma, C1, C2, omega_E)
  chain N    the statics chain from the analytic seed at N; at N = 32 also compared with the committed
             polished field (the seed-shortcut check)
  ladder N   the JG_E ladder on the chain's polished field at N
  all        validate, then chain + ladder for N in NS
"""
import json
import os
import sys
import time

import numpy as np
import torch

from lattice_L import Stack, DT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "L_ladder")
os.makedirs(os.path.join(RES, "seeds"), exist_ok=True)
R004 = os.path.join(HERE, "..", "004-lattice-clock")
R008 = HERE
# the polished 004 field is not committed (004 results); if available, point M5_FIELDS_DIR at it for the direct check
POLISHED32 = os.path.join(os.environ.get("M5_FIELDS_DIR", os.path.join(R004, "results")), "M_G_polished.npz")
NS = (32, 48, 64)
OMEGAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)
SAVE_RUNGS = (0.0, 0.1, 0.2, 0.35)          # the persisted rung fields: the bracket of the well in every box
L008 = json.load(open(os.path.join(R008, "results", "i1sq_ladders.json")))
GAMMA = L008["gamma"]


def jdump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def e_extra_fn(st, a0, gamma):
    def e_extra(Mf, om):
        i1s, k = st.densities(Mf, a0, om, "G")
        return gamma * st.H ** 3 * ((i1s - k) ** 2).sum()
    return e_extra


def profile_numbers(st, Mg, gamma):
    a0 = st.a0_of(st.gen_boost_x(), Mg)
    i1s0, k1 = st.densities(Mg, a0, 1.0, "G")
    Es0 = st.e_static(Mg, "G").item()
    C1 = (st.H ** 3 * (i1s0 * k1).sum()).item()
    C2 = (st.H ** 3 * (k1 ** 2).sum()).item()
    gamma5 = 0.05 * Es0 / (st.H ** 3 * (i1s0 ** 2).sum()).item()
    lam = torch.linalg.eigvals(torch.einsum("ab,...bc->...ac", st.ETA, Mg)).real
    top2 = lam.sort(dim=-1, descending=True).values[..., :2]
    gap = (top2[..., 0] - top2[..., 1])[st.FREE].min().item()
    env = st.envelope()
    return a0, {"E_stat0": Es0, "C1": C1, "C2": C2, "omega_pred_E": (C1 / C2) ** 0.5,
                "gamma_5pct_of_this_N": gamma5, "gamma_used": gamma,
                "statics_deformation_at_gamma_used": gamma * (st.H ** 3 * (i1s0 ** 2).sum()).item() / Es0,
                "spectral_gap_min": gap, "tail_G": st.tail_fit(Mg, "G"),
                "envelope_mass_in_box": float((env ** 2).sum() * st.H ** 3),
                "offblock_max": Mg[..., 0, 1:4].abs().max().item()}


def validate():
    st = Stack(32)
    out = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    # the committed frozen tangent of 008 is the tangent of the committed polished field; with that field available
    # (M5_FIELDS_DIR) the energies and coefficients are compared directly, otherwise only the persisted rung fields
    a0_rec = torch.tensor(np.load(os.path.join(R008, "results", "a0_frozen.npz"))["a0"], dtype=DT, device=st.dev)
    seed_shell_dev = 0.0
    if os.path.exists(POLISHED32):
        Mp = torch.tensor(np.load(POLISHED32)["M"], dtype=DT, device=st.dev)
        seed_shell_dev = ((Mp - st.SEED) * st.mask).abs().max().item()
        Mg = st.field(Mp)
        a0, prof = profile_numbers(st, Mg, GAMMA)
        out["committed_field"] = {
            "shell_vs_analytic_seed_maxdev": seed_shell_dev,
            "E_stat_G": prof["E_stat0"], "E_stat_G_recorded": L008["E_stat0"],
            "a0_maxdev": (a0 - a0_rec).abs().max().item(),
            "gamma_5pct": prof["gamma_5pct_of_this_N"], "gamma_recorded": GAMMA,
            "C1": prof["C1"], "C1_recorded": L008["C1"], "C2": prof["C2"], "C2_recorded": L008["C2"],
            "omega_pred": prof["omega_pred_E"], "omega_pred_recorded": L008["omega_pred_E"]}
    else:
        out["committed_field"] = "M_G_polished.npz not available (set M5_FIELDS_DIR); rung fields checked with the persisted tangent"
        prof = {"E_stat0": L008["E_stat0"]}
    a0 = a0_rec
    e_extra = e_extra_fn(st, a0, GAMMA)
    rows = {r["omega"]: r for r in L008["JG_E"]["rungs"]}
    out["rung_fields"] = {}
    for om, tagom in ((0.2, "02"), (0.35, "035"), (0.5, "05")):
        Mf = torch.tensor(np.load(os.path.join(R008, "results", f"jge_rung_om{tagom}.npz"))["M"], dtype=DT, device=st.dev)
        E = (st.e_static(Mf, "G") + e_extra(Mf, om)).item()
        out["rung_fields"][str(om)] = {"E_total": E, "E_total_recorded": rows[om]["E_total"],
                                       "rel_dev": abs(E - rows[om]["E_total"]) / abs(rows[om]["E_total"])}
    worst = max(v["rel_dev"] for v in out["rung_fields"].values())
    out["worst_rel_dev"] = worst
    out["pass"] = bool(worst < 1e-8 and abs(prof["E_stat0"] - L008["E_stat0"]) < 1e-6 * L008["E_stat0"]
                       and seed_shell_dev < 1e-6)   # the committed shell carries float32 seed values (3e-8)
    print(json.dumps(out, indent=1), flush=True)
    jdump(out, "validate.json")
    return out


def seed_path(N):
    """The 3x3 electron seed of the M5.21.2b instrument at N (float32), regenerated by
    instrument/m5_21_2b_a_instrument.py relax seed=A term=T2 stencil=sym eps=0 n=N L=1.5N delta=0.3 bc=pinned
    maxit=8000 w2=0.0027581 (at N = 32 it reproduces reports/004-lattice-clock/results/seed_3x3_electron.npz bitwise)."""
    return os.path.join(RES, "seeds", f"m5_21_2b_end_A_T2_sym_e0_n{N}_d0.3_pinned.npz")


def chain(N, seed=None):
    """seed: path to the 2b endpoint npz (M float32 [N, N, N, 3, 3]); default results/seeds/<tag>.npz for N;
    'analytic' embeds the analytic ansatz (the seed-shortcut check)."""
    seed = seed or seed_path(N)
    seed3 = None if seed == "analytic" else np.load(seed)["M"].astype(np.float64)
    st = Stack(N, seed3=seed3)
    t0 = time.time()
    print(f"==== chain N = {N} (L = {st.L}, h = {st.H}) on {torch.cuda.get_device_name(0)}, seed {seed} ====", flush=True)
    out = {"N": N, "L": st.L, "H": st.H, "seed": seed}
    if N == 32 and seed3 is not None:
        s004 = np.load(os.path.join(R004, "results", "seed_3x3_electron.npz"))["M"].astype(np.float64)
        out["seed_vs_committed_seed_maxdev"] = float(np.abs(seed3 - s004).max())
    M0 = st.SEED.clone()
    out["E_seed_eta"] = st.e_static(st.field(M0), "eta").item()
    Mr = st.relax_adam(M0, "eta", 3000, lr=5e-4, tag=f"N{N} eta")
    out["E_eta_relaxed"] = st.e_static(st.field(Mr), "eta").item()
    Mr = st.relax_adam(Mr, "G", 3000, lr=5e-4, tag=f"N{N} G  ")
    out["E_G_relaxed"] = st.e_static(st.field(Mr), "G").item()
    Mr, gi = st.polish(Mr, tag=f"N{N}")
    Mg = st.field(Mr)
    out["E_G_polished"] = st.e_static(Mg, "G").item()
    out["grad_inf_polished"] = gi
    _, prof = profile_numbers(st, Mg, GAMMA)
    out["profile"] = prof
    if N == 32:
        # the committed statics of 008 are always compared; the field itself only if available (M5_FIELDS_DIR)
        out["vs_committed_E_stat0"] = {"E_stat0_recorded": L008["E_stat0"],
                                       "rel_dev": abs(out["E_G_polished"] - L008["E_stat0"]) / L008["E_stat0"]}
        if os.path.exists(POLISHED32):
            Mp = torch.tensor(np.load(POLISHED32)["M"], dtype=DT, device=st.dev)
            out["vs_committed_polished"] = {
                "E_stat_committed": st.e_static(st.field(Mp), "G").item(),
                "max_abs_dM": (st.field(Mp) - Mg).abs().max().item(),
                "rms_dM": (st.field(Mp) - Mg).pow(2).mean().sqrt().item()}
    out["wall_s"] = time.time() - t0
    np.savez_compressed(os.path.join(RES, f"M_G_polished_N{N}.npz"), M=Mr.cpu().numpy())
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1), flush=True)
    print("profile:", json.dumps(prof, indent=1), flush=True)
    jdump(out, f"chain_N{N}.json")
    return out


def box_stack(N):
    """The stack of the box with the shell frozen at the box's 2b seed, the same object the chain used (a first
    run built the rung stacks with the analytic ansatz on the shell instead, 3e-8 away in float32; recorded)."""
    return Stack(N, seed3=np.load(seed_path(N))["M"].astype(np.float64))


def ladder(N, omegas=OMEGAS, tag="ladder"):
    """The JG_E ladder on the polished field of the box; `omegas` = the rungs to run (all by default; the
    `rungs` mode runs a chosen subset with the same protocol and persists their fields)."""
    st = box_stack(N)
    t0 = time.time()
    print(f"==== {tag} N = {N}: rungs {list(omegas)} ====", flush=True)
    Mr0 = torch.tensor(np.load(os.path.join(RES, f"M_G_polished_N{N}.npz"))["M"], dtype=DT, device=st.dev)
    Mg = st.field(Mr0)
    a0, prof = profile_numbers(st, Mg, GAMMA)
    e_extra = e_extra_fn(st, a0, GAMMA)
    rungs = []
    for om in omegas:
        M_raw, E_levels, ginf = st.relax_rung(
            lambda Mr, om=om: st.e_static(st.field(Mr), "G") + e_extra(st.field(Mr), om), Mr0, cycles=4)
        Mf = st.field(M_raw)
        Es = st.e_static(Mf, "G").item()
        Ex = e_extra(Mf, om).item()
        _, kd = st.densities(Mf, a0, max(om, 1e-9), "G")
        pr = ((kd.sum() ** 2) / (kd ** 2).sum().clamp_min(1e-30)).item()
        # where does the ticking density sit: radius containing half of it
        X, Y, Z = st.coords()
        r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        order = torch.argsort(r.flatten())
        cum = torch.cumsum(kd.flatten()[order], 0)
        r_half = r.flatten()[order][int(torch.searchsorted(cum, 0.5 * cum[-1]).clamp(max=cum.numel() - 1))].item()
        rungs.append({"omega": om, "E_total": Es + Ex, "E_stat": Es, "E_extra": Ex, "PR_k_sites": pr,
                      "r_half_k": r_half, "grad_inf": ginf, "E_levels": E_levels})
        print(f"  [N{N}] omega {om}: E {Es+Ex:.6f} (extra {Ex:+.4f}), PR {pr:.0f}, r_half {r_half:.1f}, "
              f"|g|inf {ginf:.1e}, levels {['%.6f' % e for e in E_levels]} [{time.time()-t0:.0f}s]", flush=True)
        if om in SAVE_RUNGS:
            np.savez_compressed(os.path.join(RES, f"jge_N{N}_om{str(om).replace('.', '')}.npz"), M=Mf.cpu().numpy())
    out = {"N": N, "L": st.L, "H": st.H, "gamma": GAMMA, "profile": prof, "rungs": rungs, "shell": "chain seed",
           "max_grad_inf": max(r["grad_inf"] for r in rungs), "wall_s": time.time() - t0}
    r0 = {r["omega"]: r for r in rungs}
    if 0.0 in r0 and len(rungs) > 2:
        k = min(range(len(rungs)), key=lambda i: rungs[i]["E_total"])
        nlev = len(rungs[0]["E_levels"])
        min_per_level = [rungs[min(range(len(rungs)), key=lambda i: rungs[i]["E_levels"][lv])]["omega"] for lv in range(nlev)]
        depth_per_level = [r0[0.0]["E_levels"][lv] - min(r["E_levels"][lv] for r in rungs) for lv in range(nlev)]
        out.update({"min_omega": rungs[k]["omega"], "interior": bool(0 < k < len(rungs) - 1),
                    "min_omega_per_level": min_per_level, "depth_per_level": depth_per_level,
                    "depth_changes": [depth_per_level[i + 1] - depth_per_level[i] for i in range(nlev - 1)],
                    "well_depth_vs_omega0": r0[0.0]["E_total"] - rungs[k]["E_total"]})
        print(f"  [N{N}] verdict: min at omega {out['min_omega']} (interior {out['interior']}), per level "
              f"{min_per_level}, depth {out['well_depth_vs_omega0']:.3e}, depth per level {['%.2e' % d for d in depth_per_level]}",
              flush=True)
    else:
        print(f"  [N{N}] {tag} recorded for rungs {[r['omega'] for r in rungs]}", flush=True)
    jdump(out, f"{tag}_N{N}.json")
    if tag == "ladder" and set(SAVE_RUNGS) <= set(omegas):
        # the bracket subset of this run, in the record the independent route reads; it is written from the same
        # relaxations whose fields were persisted above, so the record and the fields never diverge
        # the subset carries the same verdict fields as a `rungs` run (the figure and the verifier read
        # depth_per_level); they are recomputed on the bracket alone
        br = [r for r in rungs if r["omega"] in SAVE_RUNGS]
        b0 = {r["omega"]: r for r in br}
        kb = min(range(len(br)), key=lambda i: br[i]["E_total"])
        nl = len(br[0]["E_levels"])
        dpl = [b0[0.0]["E_levels"][lv] - min(r["E_levels"][lv] for r in br) for lv in range(nl)]
        sub = dict(out, rungs=br, note="bracket subset of the full ladder run",
                   min_omega=br[kb]["omega"], interior=bool(0 < kb < len(br) - 1),
                   min_omega_per_level=[br[min(range(len(br)), key=lambda i: br[i]["E_levels"][lv])]["omega"] for lv in range(nl)],
                   depth_per_level=dpl, depth_changes=[dpl[i + 1] - dpl[i] for i in range(nl - 1)],
                   well_depth_vs_omega0=b0[0.0]["E_total"] - br[kb]["E_total"])
        jdump(sub, f"rungs_N{N}.json")
    return out


def record(N, omegas):
    """Evaluate persisted rung fields (jge_N{N}_om*.npz) with the stack and write rungs_N{N}.json: the record of
    a rerun whose fields were saved but whose JSON was not (E_levels are not available in that case)."""
    st = box_stack(N)
    Mr0 = torch.tensor(np.load(os.path.join(RES, f"M_G_polished_N{N}.npz"))["M"], dtype=DT, device=st.dev)
    a0, prof = profile_numbers(st, st.field(Mr0), GAMMA)
    e_extra = e_extra_fn(st, a0, GAMMA)
    rungs = []
    for om in omegas:
        Mf = torch.tensor(np.load(os.path.join(RES, f"jge_N{N}_om{str(om).replace('.', '')}.npz"))["M"], dtype=DT, device=st.dev)
        Es, Ex = st.e_static(Mf, "G").item(), e_extra(Mf, om).item()
        _, kd = st.densities(Mf, a0, max(om, 1e-9), "G")
        pr = ((kd.sum() ** 2) / (kd ** 2).sum().clamp_min(1e-30)).item()
        X, Y, Z = st.coords()
        r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        order = torch.argsort(r.flatten())
        cum = torch.cumsum(kd.flatten()[order], 0)
        r_half = r.flatten()[order][int(torch.searchsorted(cum, 0.5 * cum[-1]).clamp(max=cum.numel() - 1))].item()
        rungs.append({"omega": om, "E_total": Es + Ex, "E_stat": Es, "E_extra": Ex, "PR_k_sites": pr, "r_half_k": r_half,
                      "from_persisted_field": True})
        print(f"  [N{N}] record omega {om}: E {Es+Ex:.9f} (extra {Ex:+.4f}), PR {pr:.0f}, r_half {r_half:.1f}", flush=True)
    out = {"N": N, "L": st.L, "H": st.H, "gamma": GAMMA, "profile": prof, "rungs": rungs, "shell": "chain seed",
           "note": "rerun of the bracket rungs with the ladder protocol; fields persisted, record evaluated from them"}
    jdump(out, f"rungs_N{N}.json")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fresh = "--fresh" in sys.argv          # regenerate every stage even if its record exists (the default resumes)
    dry = "--dry-run" in sys.argv          # print the stages `all` would run, compute nothing
    mode = args[0] if args else "all"
    if mode == "validate":
        validate()
    elif mode == "chain":
        chain(int(args[1]), args[2] if len(args) > 2 else None)
    elif mode == "ladder":
        ladder(int(args[1]))
    elif mode == "rungs":
        # e.g. `rungs 48 0.1 0.2`: the chosen rungs, same protocol, fields persisted, record rungs_N48.json
        ladder(int(args[1]), tuple(float(x) for x in args[2:]), tag="rungs")
    elif mode == "record":
        record(int(args[1]), tuple(float(x) for x in args[2:]))
    else:
        if dry:
            for N in NS:
                for stage in ("chain", "ladder"):
                    rec = os.path.join(RES, f"{stage}_N{N}.json")
                    print(f"{stage} N = {N}: {'RUN' if fresh or not os.path.exists(rec) else 'skip (record exists)'}")
            sys.exit(0)
        v = validate()
        assert v["pass"], "reimplementation does not reproduce the committed 004/008 artifacts"
        for N in NS:
            if fresh or not os.path.exists(os.path.join(RES, f"chain_N{N}.json")):
                while not os.path.exists(seed_path(N)):
                    print(f"waiting for the 2b seed at N = {N}", flush=True)
                    time.sleep(300)
                chain(N)
            else:
                print(f"chain N = {N}: record exists, skipped (use --fresh to regenerate)", flush=True)
            if fresh or not os.path.exists(os.path.join(RES, f"ladder_N{N}.json")):
                ladder(N)
            else:
                print(f"ladder N = {N}: record exists, skipped (use --fresh to regenerate)", flush=True)
        print("ALL DONE", flush=True)
