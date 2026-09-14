# Appendix — the box ladder at fixed spacing: does the (I₁^G)² well survive a larger box?

*Added 2026-09-14 under METHOD §7 (new material extending the report;
no conclusion of report 008 changes). Requested by
OpenWave's M5.32 ledger (substrate-framework discussion #186,
2026-09-02), which rates the JG_E well of this report as "the only
convergence-certified localized clock on the original 4×4 field" and
asks for the one thing a single 32³ box cannot show: "an L-ladder at
fixed h is cheap and decisive". All numbers regenerate from
`reproduce_appendix_L.sh` (CPU checks in seconds; the full regeneration
is GPU-days and CPU-hours, sentinel-flagged); committed records in
`results/L_ladder/`.*

## What is measured

The report's ladder was run in one box, L = 48 (N = 32, h = 1.5). Here
the same object is built in boxes of L = 48, 72 and 96 (N = 32, 48, 64)
at the same spacing, with the same Lagrangian: γ = 70.61, the coupling
fixed by the 5% statics-deformation budget of the L = 48 box, is kept at
that value on every rung of the ladder (the per-box "5%" value is
recorded but not used). Each box goes through the identical chain that
produced the report's field:

1. the 3×3 electron seed of the M5.21.2b instrument (seed A, term T2,
   symmetrized stencils, pinned shell, 8000 FIRE steps, w₂ = 0.0027581)
   regenerated on the box by the instrument itself (`instrument/`, a
   copy of OpenWave's script with only its output directory changed);
2. the report-004 chain: embedding with M₀₀ = −g, 3000 Adam steps in the
   η statics, 3000 in the G statics, the gradient-gated polish (Adam
   annealed, then L-BFGS to ‖g‖∞ < 10⁻⁴);
3. the report's JG_E ladder: rungs ω ∈ {0, 0.1, 0.2, 0.35, 0.5, 0.8,
   1.2}, every rung relaxed by the fixed-depth protocol (500 Adam steps
   and four L-BFGS cycles), energies recorded after every level, the
   frozen tangent a₀ = boost-x conjugation tangent with the envelope
   exp(−(r/10)⁴) — a physical radius, so the same object in every box.

`lattice_L.py` is the report-004/008 stack with N as a parameter; two
checks pin it to the record. On the committed rung fields of this
report the energies are reproduced to 10⁻⁹ (`validate.json`); from the
committed 004 seed the chain reproduces the committed polished field
**bitwise** (E_stat 4.834717814 to every printed digit, max |ΔM| = 0),
and the instrument reproduces the committed seed bitwise at N = 32;
the rerun N = 32 ladder agrees with the committed rung energies to
10⁻⁶ on the bracket (the rung relaxation itself is not bitwise
reproducible; the well, its location and its plateau are). A
shortcut tried first — the analytic hedgehog ansatz relaxed directly,
skipping the instrument — lands in a different static minimum (E_stat
0.1% higher, max |ΔM| = 0.75, C₁ thirteen times larger, ω_E = 0.20
instead of 0.33; `chain_N32_analytic_seed.json`), which is why the seed
is regenerated rather than approximated.

## Results

| L (N) | E_stat | ω_E = √(C₁/C₂) | sampled minimum, per protocol level | depth vs ω = 0 | depth per level (10⁻⁵) | ticking density: sites, r½ | ‖g‖∞ on rungs |
|---|---|---|---|---|---|---|---|
| 48 (32) | 4.835 | 0.326 | 0.35 at all five levels | 6.6·10⁻⁵ | 6.6, 6.5, 6.4, 6.5, 6.6 | 101, 6.8 | 3·10⁻³ |
| 72 (48) | 4.548 | 0.280 | 0.2 at all five levels | 6.7·10⁻⁵ | 8.3, 7.5, 7.6, 7.6, 6.7 | 314, 7.8 | 1·10⁻² |
| 96 (64) | 4.377 | 0.249 | 0.2 at all five levels | 2.5·10⁻⁴ | 9.8, 13, 20, 23, 25 | 277, 6.1 | 3·10⁻² |

![L ladder](results/L_ladder/fig_L_ladder.png)

*Left: the ladder in each box at the final protocol level, with the
frozen-profile prediction ω_E marked on the axis. Right: the depth of
the well at each protocol level, the plateau criterion of §6.*

1. **The static electron changes with the box.** Moving the pinned
   shell from 24 to 36 and 48 lowers the polished static energy by 6%
   and then 4%, the tail-fit slope on the shells r ∈ [8, 16] changes
   sign (+0.5 → −0.7 → −0.8: at L = 48 those shells still feel the
   boundary), and the clock-channel coefficients grow (C₁: 0.9, 2.0,
   2.6·10⁻⁵; C₂: 0.9, 2.6, 4.3·10⁻⁴), so the frozen-profile prediction
   of §2 falls, 0.326 → 0.280 → 0.249.
2. **The well survives, and its location moves once.** In both larger
   boxes the sampled minimum sits at ω = 0.2 at every protocol level
   (0.35 in the report's box), one rung below, with the frozen-profile
   prediction (0.28, 0.25) between the 0.2 and 0.35 rungs; from L = 72
   to L = 96 the location does not move at the rung resolution. The
   ticking density stays inside the envelope (r½ 6–8) but spreads over
   about three times as many sites as in the report's box.
3. **The depth is converged only in the report's box.** At L = 48 the
   depth plateaus (§6's criterion); at L = 72 it drifts down by 20%
   across the four L-BFGS cycles (8.3 → 6.7·10⁻⁵, last change −0.9·10⁻⁵
   against the criterion of a final change below 10% of the depth); at
   L = 96 it grows monotonically, 1.0 → 2.5·10⁻⁴, and the rung residuals
   are 10⁻², ten times the report's. The fixed-depth protocol resolves
   the bracket in the larger boxes but not the 10⁻⁴-level energy
   differences; the L = 96 depth is a lower bound under this protocol,
   not a value.

## What this appendix does not show

- No continuum extrapolation (h fixed) and no protocol deepening: the
  L = 72 and L = 96 depths are not converged at four L-BFGS cycles, and
  whether more cycles restore a plateau or move the minimum is not
  measured; the rung grid is the report's, so "0.2 in both boxes" means
  the same rung, not the same continuous ω*.
- The frozen tangent and the boost-x generator are those of the report;
  no equivariant tangent.
- γ is fixed at the L = 48 value; the per-box 5% coupling (87.9 at
  L = 72, 93.5 at L = 96) would rescale the depth by that ratio and is
  not run.
- Whether the energy-functional reading is the H of a well-posed L
  remains the author-gated question of the report.

## Artifacts and reproduction

`lattice_L.py` (the stack with N as a parameter), `appendix_L_ladder.py`
(`validate` / `chain N` / `ladder N` / `all`, resumable, one JSON per
stage in `results/L_ladder/`), `instrument/m5_21_2b_a_instrument.py`
(the OpenWave seed instrument, output directory changed), the seeds
(`results/L_ladder/seeds/`, float32, with the instrument's row JSONs),
the polished fields for N = 32 and 48 (`M_G_polished_N*.npz`; the
64³ fields, 13–14 MB each, are not committed), `make_appendix_L_figure.py`,
`verify_L_ladder.py`.

```bash
pip install torch numpy matplotlib
bash reproduce_appendix_L.sh                  # CPU: checks the records and redraws the figure
M5_RUN_LADDER=1 bash reproduce_appendix_L.sh  # seeds (CPU hours) + chains and ladders (GPU, about a day)
```

Provenance: development in `duda-particle-model/i1sq_L_ladder/`
(2026-09-13/14; the run took 1, 5 and 8 GPU-hours for the three ladders
and 4 h for the N = 64 chain); the request in discussion #186, comment
18254268; the
seed recipe from OpenWave `research/scripts/m5_21_2b_a_instrument.py`
(w₂ identified by the gradient residual of the committed seed, 8·10⁻⁵).
