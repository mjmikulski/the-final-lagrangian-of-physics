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
