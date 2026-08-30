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
| 011 | [The equivariant hedgehog and the rotational sector](reports/011-third-class-rotor/) | The spherical-frame (axially equivariant) hedgehog relaxes stably with a constant axial line tension ≈ 8·10⁻⁴ for L ≥ 36 (observable-level plateau; the cost falls with δ and is negative at δ = 1/8); neither a spectral nor a frame-twist core deformation survives statics; prescribed-J minimization grows large inertia spontaneously from the equivariant seed but the excess is peripheral (orbital lever, centroid r ≈ 15) — no core-localized spin mode is stabilized by anything tried |

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
