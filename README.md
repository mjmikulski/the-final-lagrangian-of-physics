# the final lagrangian of physics

Public research reports from work on the M5 liquid-crystal particle model
(Jarek Duda's program, simulated in [OpenWave](https://github.com/openwave-labs/openwave)).
The repo name quotes an ambition from the project correspondence; the content is
more modest: one directory per problem, each with a markdown report, the code
behind every number, and a reproduction script.

## Reports

| # | report | headline result |
|---|--------|-----------------|
| 001 | [Quadratic contractions of the M5 field strength](reports/001-quadratic-contractions/) | Exactly 6 independent quadratic invariants; the constant-coefficient extension cannot fix the clock sign while preserving the 3×3 sector (no-go verified) |
| 002 | [Covariant rot/boost split and a finite-frequency clock candidate](reports/002-covariant-split-and-clock/) | Field-selected covariant split (3 gated constructions); one-line covariant sign fix with K⪰0; linear clock terms cancel in H; a quartic boost term keeps a finite ω\* in both Legendre readings, sitting on the Shapere–Wilczek caustic |
| 003 | [Canonical analysis of the boost-condensate clock](reports/003-canonical-analysis/) | The naive Lorentzian completion is killed by its own Legendre transform (−bB_s² in statics); the u-selected completion is healthy: H = −aB_k + 3bB_k², finite ω\*, energy PSD at the clock, branched dynamics required |
| 004 | [Lattice hedgehog under the covariant G action](reports/004-lattice-clock/) | Statics survive on the lattice (gap 6.98, kin all-positive); Q1 of P240: no negative-curvature witness at the gradient-gated point (λ_min ≤ +1.1e-3, evidence not certificate); honest negative: the local-quartic clock delocalizes (PR ×22) instead of ticking |
| 005 | [Parity-odd (Levi-Civita) quadratic contractions](reports/005-epsilon-contractions/) | The ε sector closes: 3 (not 4) independent pseudoscalars on model fields — P239's J4 ≡ 0 by a cyclic-trace identity; the 3×3 no-go survives the entire ε sector; χ² = 16N₁ exactly; φ and χ are null Lagrangians |
| 006 | [Newton sign on boost hedgehogs](reports/006-newton-boost-hedgehogs/) | On the canonical ansatz F is purely spatial: the 3×3-preserving repair space acts as identical zero, all quadratics collapse to two channels with 3S₁ = 4S₄ (virial) and ρ = e₁/e₄ ∈ [1,4]; measured repulsive tails with X > 0 close every sign branch — no constant-coefficient quadratic attracts and stays stable |
| 007 | [Core-weighted coefficients and the localized clock](reports/007-core-weighted-clock/) | c(M) and smooth topological coefficients are provably blind to boost dressings (clock-only mechanism); kinetic clock window opens at c ≈ −0.6 on the core; the local quartic delocalizes even under frozen masks, but the intensive quartic × core weight yields an interior ω\* = 0.8 with the boost density on ~100 core sites |
| 008 | [The simplest quartic (I₁)² and the repaired-metric clock](reports/008-i1-squared-clock/) | (I₁)² ticks only in the repaired metric: the raw η contraction is inert (positive time part — measured no-go), while the G-form energy ansatz yields an interior well at ω = 0.35 (prediction 0.326) localized on ~100 sites; the completed square is convex — no dilution channel; the naive fundamental-Lagrangian reading is measured to run away (−γs² unbounded); γ-scaling ×4 confirmed, bounded γ budget (16γ breaks) |
| 009 | [The rotational sector: protocol-limited ladder, extensive rigid rotations, and a constrained fixed-J surrogate](reports/009-rotational-clock/) | The env-frozen rotational ladder is not protocol-convergent (deep relaxation reorders the bracket — the well claim of the first version is withdrawn); rigid rotations split into a measured trichotomy: extensive (non-equivariant texture, I ~ L^2.93), trivial (equivariant texture = stabilizer, ζM ≡ 0), and finite (core-breaking texture on an equivariant background, I ~ L^0.14, core-localized) — with a bounded constrained fixed-J surrogate (not a derived Routh reduction) and the texture test showing equivariant frames are symmetric at any δ |
| 010 | [The fundamental-reading clock grid](reports/010-fundamental-grid-clock/) | The u-decorated family closes at rank 18 over ℚ; the Legendre filter, a 1-dim purely-gyroscopic static kernel, and exact matrix-cap orbit zeros factorize the grid L = −I₁ + γ(I_j)² − V to {−I₁} × 16 diagram-ray cells (plus their gyroscopic λ-families, characterized but not scanned); at frozen-tuned γ no cell ticks (brake evasion — the naked concavity), but the γ-window is two-sided and measured: interior fundamental-reading wells at ×10/×14/×20 at fixed relaxation depth (ω\* 0.1–0.2, drive-flip control kills them) — the deep-bracket run shows the minimum migrates under deeper relaxation, so converged-level existence stays open |
| 011 | [The equivariant hedgehog and the rotational sector](reports/011-third-class-rotor/) | The spherical-frame (axially equivariant) hedgehog relaxes stably with a constant axial line tension ≈ 8·10⁻⁴ for L ≥ 36 (observable-level plateau; the cost falls with δ and is negative at δ = 1/8); neither a spectral nor a frame-twist core deformation survives statics; prescribed-J minimization grows large inertia spontaneously from the equivariant seed but the excess is peripheral (orbital lever, centroid r ≈ 15) — no core-localized spin mode is stabilized by anything tried |
| 012 | [Scaling toward the proposed vacuum hierarchy](reports/012-delta-g-scaling/) | Pre-registered 3×3 grid over (δ, g) toward the author's g ~ 10¹⁰, δ ~ 10⁻¹⁰: δ is a flat direction measured AT the target (10⁻¹⁰, 10⁻¹¹ match 1/8 to 0.1%); g is live and potential-variant-sensitive (drive grows in both variants, ω_pred rises in the original theory vs falls in the relative variant; absolute-potential breakdown at g = 512 documented at 13 ulp); both time-part signs stable at all points |
| 013 | [C10's candidate wells do not certify within a 24-cycle continuation](reports/013-window-convergence/) | A budget-bounded negative on the deepest-probed corner of report 010's open question: the C10 candidate wells at couplings ×10 and ×14 do not certify under a 24-cycle observable-level continuation and the drift shows no sign of saturating (×14 inverts the bracket structure; ×10 keeps close brackets with the top rung below the minimum); the boost bracket of report 008 holds its order under the same budget; the grammar-wide existence question stays open |
| 014 | [Linear-in-F terms with spectral coefficients](reports/014-linear-F-terms/) | The one unscanned order: terms linear in F (quadratic in ∂M). Twelve pointwise generators (6 even + 6 odd; exact upper and lower bounds), all dynamical once the coefficients depend on the spectral frame (the constant φ, χ stay null); no ω² part, so nothing for the clock; even sector inert on the canonical ansatz and the odd sector vanishing on its parity-symmetric configurations, so 006's Newton no-go stands there; the uniform vacuum is a saddle only at cubic order (any resulting state below the resolution of a relaxation); on the electron a sign-weighted reweighting of the static density with the core structure kept — and with exact spectral coefficients the relaxation runs to the eigenvalue-1/small-pair collisions where those coefficients are non-smooth |

## How to reproduce

```bash
pip install -r requirements.txt          # Python >= 3.12
./reports/001-quadratic-contractions/reproduce.sh
```

Each report's `reproduce.sh` regenerates all results and asserts the
structural claims (counts, ranks, identities). Exact floating-point values may
differ in the last digits across machines/BLAS; the asserted structure may not.

## Method

A few rules, borrowed from the best of
[OpenWave](https://github.com/openwave-labs/openwave) and
[substrate-framework](https://github.com/vantasnerdan/substrate-framework)
and nothing else — see [METHOD.md](METHOD.md).

Reports are written by Maciej J. Mikulski with AI assistance (Claude); every
claim is human-owned and script-backed. This is a working record of an ongoing
collaboration, not peer-reviewed publication.
