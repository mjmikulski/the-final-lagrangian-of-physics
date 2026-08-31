# Appendix — independent route-2 verification of the deep-bracket energies

*Added 2026-08-31 under METHOD §7 (new material extending the report;
no conclusion changes; the merged README is left untouched per §7 —
this appendix itself records that §5's stated limitation "an
independent route-2 energy verification on them remains open" is
hereby closed).*

## What is verified

The per-rung final energies of the deep-bracket run (`e5_deep_bracket.py`,
coupling ×14 of the frozen-tuned value, rungs ω ∈ {0, 0.1, 0.15, 0.2,
0.28}) are re-derived by a from-scratch numpy implementation
(`verify_deep14_route2.py`) of the cell Hamiltonian defined in
`lattice_grid_defs.py`'s docstring,

```math
E_{\rm cell}(M;\omega) = e_{\rm static}(M,\eta) - 2H^3\sum k_1(\omega)
 + \gamma H^3 \sum\left(-s^2 + m^2 + 2sk + 4mk + 3k^2\right),
```

evaluated on the persisted rung fields `deep14_om*.npz` with the frozen
tangent persisted as a new artifact (`a0_e4_frozen.npz`, produced once
by `persist_frozen_tangent.py` from the committed base profile). The
C10 density is implemented directly from its index definition (the
$uu^\top$ cap on the two leading derivative slots via $(G-\eta)/2$,
$\eta$ pairs elsewhere) — no torch, no import of the report's stack.

## Result

All five rung energies agree with the committed
`e5_deep_bracket.json` record to **1.0·10⁻¹⁴ relative** (worst rung).
The committed record — including the migration of the ×14 minimum to
the still-descending top rung, the fact the report's §4 conclusion
rests on — is therefore independently confirmed at the energy level.

## A convention detail recorded

The only calibration needed against the stack was a single overall
multiplicity: a diagram with one antisymmetric pair capped by
$uu^\top$ carries half the fully-$\eta$-paired class's conventional
prefactor (the factor-4 of two free antisymmetric pairs becomes 2).
The route-2 code documents this in place; no tunable parameters
remain. The contraction itself is validated on a generic
antisymmetric tensor with a boosted (nonzero $U^{0i}$) cap against a
one-line direct einsum (built-in self-test, $10^{-12}$) — on the
committed deep14 fields $U^{0i} = 0$, so this generic check is what
distinguishes the upper-cap definition from sign-variant rewritings.

## Reproduction

```bash
python3 persist_frozen_tangent.py    # once, GPU, writes a0_e4_frozen.npz
python3 verify_deep14_route2.py      # CPU, ~2 min; asserts 1e-9
```
