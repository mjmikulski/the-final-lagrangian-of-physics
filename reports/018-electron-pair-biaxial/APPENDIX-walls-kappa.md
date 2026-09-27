# Appendix — is the Bogomolny strand the lowest state, and what a two-derivative term changes

*Added 2026-09-27 under METHOD §7 (new material; no conclusion of the report
is withdrawn, one is qualified: result 3's tension is that of the planar
branch, which is a saddle of the full frozen 2D problem). Prompted by substrate-framework discussion #186
(2026-09-25): OpenWave's R26 found that under its action the smooth-core
strand is a saddle and relaxes into π-jump walls of the pair anisotropy
("free under the commutator curvature along the wall"), and J. Duda
proposed a two-derivative term κ Tr(∂M ∂M) that would penalise the walls
and order the partitions of a charge's index by Σk². Both questions are
asked here for the potential of this report, V = Σ_a (e_a − E_a)², with
the 2D lattice of `strand2d.py` (β = 0.3; ρ½ is the Bogomolny core radius,
0.40 at k = ½). `strand_walls.py`; records `results/strand_walls_*.json`.*

## (a) Seeds that are not smooth

| seed (k = ½ unless stated) | h = ρ½/8 | h = ρ½/16 |
|---|---|---|
| smooth (the report's seed) | T_B (1.0001) | — |
| unmelted core: full splitting up to the line | starts at 260 T_B, relaxes to T_B | starts at 1030 T_B, relaxes to T_B |
| wedge: the winding in a thin sheet along a ray | 1.04 T_B, still falling (4000 iterations) | 1.21 T_B, still falling |
| kicked: smooth seed + random perturbation β/3 per node | **0.30 T_B** after 8000 iterations (0.20 after 20 000), still falling | **0.91 T_B** after 24 000, still falling |

The unmelted core costs ∝ 1/h² on the lattice (its line cell carries
gradients of order β/h) and melts back to the Bogomolny profile. The
smooth and unmelted seeds keep S_xz = S_yz = 0, a symmetric subspace that
gradient descent cannot leave, so their convergence does not test
stability outside it.

**The Bogomolny strand is a saddle (review round 1).** A regular
perturbation of the xz and yz entries alone (they tilt the charge
direction), f(r)(cos 2φ, −sin 2φ) with a profile vanishing at the line
and before the boundary, lowers the energy at second order:
T = T_B + ε²Q with Q = −0.32. The radial integral of the expanded
density gives this value, and so do energy differences on the lattice at
both spacings (`strand_saddle.py`). The quartic term carries the negative
part, and the potential adds +0.01. So T_B = (2√2π/3)kβ³ is the tension of
the planar branch (charge direction fixed along the line), which is a
saddle of the full frozen problem.

**The kicked seeds leave it.** The same random perturbation (drawn once
and interpolated) goes below T_B at both spacings, later at the finer
one. The lowest energies reached are 0.20 T_B (h = ρ½/8) and 0.91 T_B
(ρ½/16), and both runs are still falling. These are upper bounds on the
infimum of the lattice tension; they do not show that the infimum is
zero, which is left open. The low-energy states contain turns of the
transverse frame by 1.2–1.5 rad across a single cell, with the charge
direction tilted by up to |e₁·ẑ| = 0.88 and the eigenvalues near their
vacuum values. At h = ρ½/8 the line also leaves the centre of the box.
Whether a smooth continuum family reaches below T_B by more than the
second-order descent, and how far, is not settled here. This is the
same qualitative finding as R26 (the smooth-core strand is a saddle),
now for the eigenvalue potential.

## (b) The two-derivative term κ Σᵢ Tr(∂ᵢN ∂ᵢN)

1. **The winding cost.** On the smooth k = ½ strand the term adds
   16πκk²b₀² ln(R/ξ) per unit length (b₀ = β/2). Between R = 4, 8 and
   16 ρ½ the added energy grows by (2.76–2.83)·10⁻⁴ per unit of ln R at
   κ = 10⁻³, against the predicted 2.83·10⁻⁴. The growth is linear in κ.
2. **The full strand splits.** At k = 1 the logarithmic growth between
   R = 8 and 16 ρ½ is only 1.9–2.4 times that of k = ½, not 4. The relaxed k = 1 field at κ = 10⁻²
   (R = 16 ρ½) shows why: it has split into two half strands of winding π
   each, pushed apart to y = ±4.45 in a box of half-size 6.5, with no
   winding left at the centre. With κ, like half strands repel
   logarithmically, and two halves (Σk² = ½) cost less than one full
   strand (Σk² = 1). This is the ordering by Σk² proposed on #186, seen in
   the field.
3. **With κ the descent levels off.** With κ = 10⁻² the kicked seed stops
   falling: at
   h = ρ½/8 it converges (gradient 8·10⁻⁷) to 0.845 T_B, with the line at
   the centre and no sheets. At h = ρ½/16 it reaches 0.843 T_B (still
   decreasing slightly in its last block): the energies attained at the two
   tested spacings agree to 0.2%, unlike the descent without κ. That state lies below
   the planar strand with the same κ (about 1.15–1.19 T_B at these boxes);
   it is the lowest strand found with κ, not a demonstrated minimum, and its
   stability and its dependence on β are not measured.

![walls and kappa](results/fig_appendix_walls.png)

*Left: T/T_B against L-BFGS iterations for the kicked seed, at two
spacings, without κ and with κ = 10⁻². Right: the energy the κ term adds to the
smooth strand, divided by κ, against ln R, for k = ½ and 1, with the
slope 16πk²b₀² for each.*

## What this does not show

- No continuum limit is taken; whether the infimum is zero remains open.
  Two spacings show descent below T_B, slower at the finer one; neither
  run without κ has stopped, so the lowest lattice energy at either
  spacing is not known.
- One β (0.3) and one random perturbation. The κ-stabilised tension is
  measured at one κ (at two spacings), not as a function of β,
  so the β³ law of the report is not transferred to it.
- κ is the ungauged term. The gauge-covariant κ Tr(DM DM) of the 09-25
  proposal, with its connection, is not studied.
- 2D straight strands only; the 3D charge of the report is not rerun with
  κ.

## Reproduction

`python strand_saddle.py` (the second variation, CPU, minutes) and
`python strand_walls.py seeds|kick|continue16|kick16kappa|kappa|split`
(GPU, hours each; `kick` and `continue16` about 2 and 4 hours on an RTX
4070) regenerates the records; `make_appendix_walls_figure.py` draws the
figure from them and `verify_appendix_walls.py` asserts their structure;
`reproduce_appendix_walls.sh` recomputes the saddle witness (CPU, minutes;
skip it with `M5_SKIP_SADDLE=1`), redraws the figure and runs the assertions,
or regenerates everything with `M5_RUN=1`. The relaxed 2D fields (`results/fields/*.npz`) are assets of
the `report-018-fields` release.
