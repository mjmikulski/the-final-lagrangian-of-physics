# the final lagrangian of physics

Public research reports from work on the M5 liquid-crystal particle model
(Jarek Duda's program, simulated in [OpenWave](https://github.com/openwave-labs/openwave)).
The repo name quotes an ambition; the content is
more modest: one directory per problem, each with a markdown report, the code
behind every number and a reproduction script.

## Reports

| # | report | appendices | first published | last updated | headline result |
|---|--------|:---:|---|---|-----------------|
| 001 | [Quadratic contractions of the M5 field strength](reports/001-quadratic-contractions/) | [1](reports/001-quadratic-contractions/APPENDIX-parity-odd.md) | 2026-08-20 | 2026-08-21 | Exactly 6 independent quadratic invariants; the constant-coefficient extension cannot fix the clock sign while preserving the 3×3 sector (no-go verified) |
| 002 | [Covariant rot/boost split and a finite-frequency clock candidate](reports/002-covariant-split-and-clock/) |  | 2026-08-20 | 2026-08-20 | Field-selected covariant split (3 gated constructions); one-line covariant sign fix with K⪰0; linear clock terms cancel in H; a quartic boost term keeps a finite ω\* in both Legendre readings, sitting on the Shapere–Wilczek caustic |
| 003 | [Canonical analysis of the boost-condensate clock](reports/003-canonical-analysis/) |  | 2026-08-20 | 2026-08-20 | The naive Lorentzian completion is killed by its own Legendre transform (−bB_s² in statics); the u-selected completion is healthy: H = −aB_k + 3bB_k², finite ω\*, energy PSD at the clock, branched dynamics required |
| 004 | [Lattice hedgehog under the covariant G action](reports/004-lattice-clock/) |  | 2026-08-21 | 2026-08-21 | Statics survive on the lattice (gap 6.98, kin all-positive); Q1 of P240: no negative-curvature witness at the gradient-gated point (λ_min ≤ +1.1e-3, evidence not certificate); honest negative: the local-quartic clock delocalizes (PR ×22) instead of ticking |
| 005 | [Parity-odd (Levi-Civita) quadratic contractions](reports/005-epsilon-contractions/) |  | 2026-08-21 | 2026-08-21 | The ε sector closes: 3 (not 4) independent pseudoscalars on model fields — P239's J4 ≡ 0 by a cyclic-trace identity; the 3×3 no-go survives the entire ε sector; χ² = 16N₁ exactly; φ and χ are null Lagrangians |
| 006 | [Newton sign on boost hedgehogs](reports/006-newton-boost-hedgehogs/) | [1](reports/006-newton-boost-hedgehogs/APPENDIX-flat-connection.md) | 2026-08-22 | 2026-08-28 | On the canonical ansatz F is purely spatial: the 3×3-preserving repair space acts as identical zero, all quadratics collapse to two channels with 3S₁ = 4S₄ (virial) and ρ = e₁/e₄ ∈ [1,4]; measured repulsive tails with X > 0 close every sign branch — no constant-coefficient quadratic attracts and stays stable |
| 007 | [Core-weighted coefficients and the localized clock](reports/007-core-weighted-clock/) | [1](reports/007-core-weighted-clock/APPENDIX-no-connection-from-M.md) | 2026-08-25 | 2026-08-28 | c(M) and smooth topological coefficients are provably blind to boost dressings (clock-only mechanism); kinetic clock window opens at c ≈ −0.6 on the core; the local quartic delocalizes even under frozen masks, but the intensive quartic × core weight yields an interior ω\* = 0.8 with the boost density on ~100 core sites |
| 008 | [The simplest quartic (I₁)² and the repaired-metric clock](reports/008-i1-squared-clock/) | [1](reports/008-i1-squared-clock/APPENDIX-L-ladder.md) | 2026-08-27 | 2026-09-15 | (I₁)² ticks only in the repaired metric: the raw η contraction is inert (positive time part — measured no-go), while the G-form energy ansatz yields an interior well at ω = 0.35 (prediction 0.326) localized on ~100 sites; the completed square is convex — no dilution channel; the naive fundamental-Lagrangian reading is measured to run away (−γs² unbounded); γ-scaling ×4 confirmed, bounded γ budget (16γ breaks) |
| 009 | [The rotational sector: protocol-limited ladder, extensive rigid rotations, and a constrained fixed-J surrogate](reports/009-rotational-clock/) |  | 2026-08-28 | 2026-08-29 | The env-frozen rotational ladder is not protocol-convergent (deep relaxation reorders the bracket — the well claim of the first version is withdrawn); rigid rotations split into a measured trichotomy: extensive (non-equivariant texture, I ~ L^2.93), trivial (equivariant texture = stabilizer, ζM ≡ 0), and finite (core-breaking texture on an equivariant background, I ~ L^0.14, core-localized) — with a bounded constrained fixed-J surrogate (not a derived Routh reduction) and the texture test showing equivariant frames are symmetric at any δ |
| 010 | [The fundamental-reading clock grid](reports/010-fundamental-grid-clock/) | [1](reports/010-fundamental-grid-clock/APPENDIX-route2-deep14.md) | 2026-08-30 | 2026-08-31 | The u-decorated family closes at rank 18 over ℚ; the Legendre filter, a 1-dim purely-gyroscopic static kernel, and exact matrix-cap orbit zeros factorize the grid L = −I₁ + γ(I_j)² − V to {−I₁} × 16 diagram-ray cells (plus their gyroscopic λ-families, characterized but not scanned); at frozen-tuned γ no cell ticks (brake evasion — the naked concavity), but the γ-window is two-sided and measured: interior fundamental-reading wells at ×10/×14/×20 at fixed relaxation depth (ω\* 0.1–0.2, drive-flip control kills them) — the deep-bracket run shows the minimum migrates under deeper relaxation, so converged-level existence stays open |
| 011 | [The equivariant hedgehog and the rotational sector](reports/011-third-class-rotor/) |  | 2026-08-29 | 2026-08-30 | The spherical-frame (axially equivariant) hedgehog relaxes stably with a constant axial line tension ≈ 8·10⁻⁴ for L ≥ 36 (observable-level plateau; the cost falls with δ and is negative at δ = 1/8); neither a spectral nor a frame-twist core deformation survives statics; prescribed-J minimization grows large inertia spontaneously from the equivariant seed but the excess is peripheral (orbital lever, centroid r ≈ 15) — no core-localized spin mode is stabilized by anything tried |
| 012 | [Scaling toward the proposed vacuum hierarchy](reports/012-delta-g-scaling/) |  | 2026-08-30 | 2026-08-30 | Pre-registered 3×3 grid over (δ, g) toward the author's g ~ 10¹⁰, δ ~ 10⁻¹⁰: δ is a flat direction measured AT the target (10⁻¹⁰, 10⁻¹¹ match 1/8 to 0.1%); g is live and potential-variant-sensitive (drive grows in both variants, ω_pred rises in the original theory vs falls in the relative variant; absolute-potential breakdown at g = 512 documented at 13 ulp); both time-part signs stable at all points |
| 013 | [C10's candidate wells do not certify within a 24-cycle continuation](reports/013-window-convergence/) |  | 2026-08-31 | 2026-08-31 | A budget-bounded negative on the deepest-probed corner of report 010's open question: the C10 candidate wells at couplings ×10 and ×14 do not certify under a 24-cycle observable-level continuation and the drift shows no sign of saturating (×14 inverts the bracket structure; ×10 keeps close brackets with the top rung below the minimum); the boost bracket of report 008 holds its order under the same budget; the grammar-wide existence question stays open |
| 014 | [Terms linear in the field strength: twelve generators, blind to the clock and to Newton, and an unstable vacuum](reports/014-linear-F-terms/) | [1](reports/014-linear-F-terms/APPENDIX-volume-weights.md) | 2026-09-10 | 2026-09-13 | The one unscanned order: terms linear in F (quadratic in ∂M). Exactly twelve generators (6 even + 6 odd, exact over ℚ; the same twelve found independently in the E₀ = 100 conventions), all dynamical once the coefficients depend on the spectral frame while the constant φ, χ stay null; velocity degree ≤ 1, so no drive or brake in the fixed-velocity Legendre energy of 010; even sector inert on the canonical orbit of 006; on the electron a sign-weighted reweighting of the static density (both routes); and for each scanned class the uniform vacuum is a saddle at cubic order along generic frame twists, null combinations exempt |
| 015 | [The halo mechanism: a purely quartic gradient energy gives the hedgehog an unbounded moment of inertia](reports/015-halo-inertia/) |  | 2026-09-09 | 2026-09-09 | Outside the core the model is the Faddeev–Skyrme quartic without a sigma term; an internal tilt θ(r) of the halo costs exactly (32π/3)Δ⁴∫θ′² and carries the inertia (64π/3)Δ⁴∫4 sin²(θ/2)(1 + r²θ′²/2) (lattice: inertia to 2%, cost to 10–30%, two boxes), so the fixed-J infimum is the static energy for every J: no fixed-J minimiser is a rotor with finite frequency and the rotor clock condition E/Ω = 2J cannot be met by one (stationary non-minimal branches and quantization not addressed); fixed-J minimization in a box gives localized deformations with E/(2JΩ) ≈ 70–500 (fitted inertia prefactors within 2% of the exterior formula) |
| 016 | [The time sector of the electron: screening of the Coulomb energy by a tilt of the time axis, and the frame term that stops it](reports/016-time-axis-screening/) |  | 2026-09-10 | 2026-09-10 | With the time sector free, the frozen hedgehog is a saddle: the radial tilt of the time axis toward the light cone (a null-direction configuration on which F and every Lorentz scalar in ∂N vanish exactly) screens the Coulomb energy, and without a box the charged configuration has zero energy infimum (a family with F ≡ 0 and E ≤ (8π/3)Δ²R³, charge intact); box 12: 27.3 → 22.0, weakly dependent on E₀ (20.4–22.1 from 10 to 1000); every term built from the spectrum or the charge direction is blind to it; the frame term K_u (zero on the whole frozen record) restores the mass with a local threshold c ≈ 0.013 and a global one between 0.1 and 0.3, and opens the model's only linear dynamics (three light-speed tilt modes, no ghost) |
| 017 | [Constraint structure of the field M: the trace is not a field, the kinetic form has rank 9 on generic backgrounds, and every static background has characteristic speeds in [0, 1]](reports/017-constraint-structure/) |  | 2026-09-24 | 2026-09-24 | The Lagrangian is exactly quadratic in Ṁ with a Gram kinetic form K (no mixed or gyroscopic terms); the derivative sector is blind to N → N + s(x)1, so tr N = ΣE is an algebraic equation (second-class pair, 10 → 9); rank K = 9 on generic backgrounds, 8 on generic frozen ones (kernel trace and P₀), 5 on one-direction twists (c = 0), 0 on the vacuum (3 light-speed tilts with K_u); the principal symbol is a Lagrange identity, so squared characteristic speeds lie in [0, 1] on every static background (closed form on 17 200 points, checked against autograd on a subset), with 0 attained (not strongly hyperbolic); with the η matrix norm K is indefinite at every checked non-vacuum point; the combined rotation of the uniaxial hedgehog is a symmetry (b = 0, inertia 1% of the internal one) |

*Appendices* are links to the `APPENDIX-*.md` companion files of a merged report (METHOD rule 7),
numbered in file-name order;
*first published* is the date of the commit that added the report's README, *last updated* the
date of the last commit touching anything in the report's directory. All three come from git:
`python report_dates.py --markdown` regenerates the cells.

## How to reproduce

```bash
pip install -r requirements.txt          # Python >= 3.12
./reports/001-quadratic-contractions/reproduce.sh
```

Each report's `reproduce.sh` regenerates all results and asserts the
structural claims (counts, ranks, identities). Exact floating-point values may
differ in the last digits across machines/BLAS; the asserted structure may not.
Everyone is encouraged to reproduce and create an issue if sth does not agree.

## Method
See [METHOD.md](METHOD.md).

## Remarks
Reports are written by AI (Anthropic Claude) and reviewed by AI (OpenAI Codex),
with only general directions given by MJ Mikulski. 

The original Lagrangian was proposed by Jarek Duda.

Many ideas here come from conversations between Jarek and MJ.

Claims are script-backed but not verified by experts. 
This is a working record of an ongoing research experiment, not peer-reviewed publication.

Will AI (with human in the loop) find **the final Lagrangian of the Physics?** - We shall see.
