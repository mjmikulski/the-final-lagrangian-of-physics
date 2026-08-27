# Report 008 — The simplest quartic works: a core-localized clock from $(F_{abcd}F^{abcd})^2$ alone

*2026-08-27 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
answers the model author's direct question (correspondence 2026-08-26):
"does the simplest fourth-order term, $(F_{abcd}F^{abcd})^2$, work?" —
his candidate for the article-1 Lagrangian*

## Context and result

Report 007 built a working localized clock, but its mechanism needed
structure added by hand: an intensive (nonlocal) quartic and a
core-supported weight $c(M)$. The model author's preference for the
article is the *simplest* fourth-order term: the square of the first
invariant itself, $\gamma\,(I_1)^2$ with
$I_1 = F_{abcd}F^{abcd}$ (report 001 conventions).

**Result: it works — locally, with no added structure.** On a clock
configuration the $I_1$ density splits into a static part and a kinetic
part, and in the covariant $\eta$ convention the kinetic part enters
with a minus sign ($\eta^{00}=-1$):

$$I_1(x) = i_1^{\rm stat}(x) - i_1^{\rm kin}(x,\omega),$$

$$\gamma I_1^2 = \underbrace{\gamma\, (i_1^{\rm stat})^2}_{\text{statics
deformation}}\; \underbrace{-\,2\gamma\, i_1^{\rm stat}\, i_1^{\rm
kin}}_{\text{clock drive}}\; +\;
\underbrace{\gamma\,(i_1^{\rm kin})^2}_{\text{local quartic brake}}.$$

The cross term supplies the negative kinetic coefficient (the clock
drive) with no hand-added structure. Why this local quartic does *not*
delocalize, when every plain local quartic tried in 004/007 did, is a
**convexity** fact: the Mexican-hat form $-a\,b_k + 3b\,b_k^2$ is
concave in its linear drive, so spreading the ticking density at fixed
integral lowers the quartic cost (dilution pays); the completed square
$(i_1^{\rm stat}-i_1^{\rm kin})^2$ is convex in $i_1^{\rm kin}$ with
its pointwise minimum at the static template
$i_1^{\rm kin} = i_1^{\rm stat}$ — any spreading away from the
template only costs. (Honest measurement: on this hedgehog
$i_1^{\rm stat}$ is nearly flat across shells — max/min contrast
$\sim$1.8 — so the localization does NOT come from a concentrated
static weight; it comes from the convex template structure plus the
kinematics of the clock channel.)

**Measurements** (report-004 lattice stack, fresh-start ladders, every
rung relaxed):

1. **The well is where the formula says** (§2): the reduced frozen-profile
   energy is $E(\omega)\simeq -2\gamma C_1\omega^2 + \gamma C_2\omega^4$
   with $C_1 = \int i_1^{\rm stat} b_{k,1}$, $C_2 = \int b_{k,1}^2$,
   predicting $\omega_* = \sqrt{C_1/C_2} = 0.326$ **independent of
   $\gamma$**. The relaxed ladder J1 has its interior minimum at the
   sampled rung $\omega = 0.35$ (grid: 0.2/0.35/0.5), with the ticking
   density localized at PR $\approx 102$ sites.
2. **Sign control** (§3): flipping the cross-term sign (J0) removes the
   well — minimum at $\omega = 0$, monotonic rise. (A first run showed
   a fake interior minimum in *both* J1 and J0 because the $\omega=0$
   endpoint was not relaxed under the added statics deformation; the
   control caught it, and the protocol now relaxes every rung.)
3. **$\gamma$-scaling test** (§4): at $4\gamma$ the minimum stays at
   $\omega = 0.35$ and the well depth scales $\times 4.06$ (predicted
   $\times 4$) — position independent of $\gamma$, depth linear in
   $\gamma$, exactly the reduced formula.
4. **Adversarial localization check** (§4b): if the observed
   localization were merely the weakness of a shallow term, a much
   deeper well would delocalize the ticking the way 004/007's
   Mexican-hat quartics did. At $16\gamma$ (80% statics deformation,
   deliberately aggressive) the well deepens to $1.03\cdot10^{-3}$
   (ratio 15.7 — depth still linear in $\gamma$) and the ticking stays
   at PR $= 106$ sites at the minimum — the convexity mechanism, not
   weakness, holds the localization.
5. **The intensive variant is the wrong realization here** (§5):
   $(\int I_1)^2$ minimizes by trivially zeroing the integral (at
   $\omega\approx 5$ the kinetic integral cancels the static one and
   the term vanishes) — a degenerate mechanism, not a clock. The
   *local* density squared is the physical form — no nonlocality is
   needed, which answers report 007's author-gated locality question in
   the best possible way.

**Caveat, stated plainly:** the well is shallow. Its depth is
$\gamma C_1^2/C_2$ and $\gamma$ is bounded by how much deformation of
the static (3×3, lepton) sector one accepts: 5% deformation gives
depth $7\cdot10^{-5}$, 20% gives $2.7\cdot10^{-4}$ (lattice units,
against $E\approx 5$). The depth-vs-deformation trade-off is a physics
decision (author-gated), tied to the scale-anchoring choices.

## 1. Setup

Everything runs on report 004's stack: the $32^3$ lattice, the polished
hedgehog `M_G_polished.npz`, the working $G$ metric (report 002), the
frozen clock tangent $a_0$ (boost-x generator; the same fixed-tangent
protocol as 004/007 — see Limitations). The static $I_1$ density
$i_1^{\rm stat}$ is the spatial-pair commutator energy density of
`lattice.e_static`; the kinetic density at ladder frequency $\omega$ is
the boost-channel density $b_k(x,\omega) = \omega^2 b_{k,1}(x)$.
$\gamma$ is fixed by a 5% statics-deformation budget:
$\gamma \int (i_1^{\rm stat})^2 = 0.05\, E_{\rm stat}$, giving
$\gamma = 70.61$.

## 2. Faithful ladder J1

Fresh-start from the polished field, 500 Adam steps per rung, **every
rung relaxed including $\omega = 0$** (the statics deformation
$\gamma (i_1^{\rm stat})^2$ must relax at the endpoint too):

| $\omega$ | 0.0 | 0.1 | 0.2 | **0.35** | 0.5 | 0.8 | 1.2 |
|---|---|---|---|---|---|---|---|
| $E_{\rm total}$ | 5.0708 | 5.0708 | 5.0707 | **5.0707 (min)** | 5.0708 | 5.0718 | 5.0739 |
| PR $[b_k]$ (sites) | — | 101 | 101 | 102 | 104 | 119 | 168 |

Interior minimum at the sampled rung 0.35, bracketing
$E(0.2)>E(0.35)<E(0.5)$ at the $10^{-4}$ level, consistent with the
frozen-profile prediction $\omega_* = 0.326$; localization stays at
$\sim$100 core sites near the minimum (the delocalized floor of report
004 was 1962 sites).

![i1sq ladders](results/fig_i1sq_ladders.png)

**Route 2** (`verify_energies.py`): a from-scratch numpy
re-implementation (differences, Euclideanizer, potential, static $I_1$
density, boost channel, the $(I_1)^2$ term) evaluated on the persisted
J1 rung fields reproduces the recorded totals and confirms the sampled
well shape independently.

## 3. Sign control J0

Same ladder with the cross term flipped to $+$: minimum at
$\omega = 0$, monotonic rise ($5.0708 \to 5.0762$ at $\omega = 1.2$),
no interior well. The J1 signal is the mechanism, not protocol noise;
the run-1 endpoint artifact this control caught is documented in
`ladder_i1sq.py`'s docstring.

## 4. $\gamma$-scaling confirmation

At $4\gamma$ (20% statics deformation): minimum again at the 0.35 rung,
depth $2.7\cdot10^{-4}$ vs $6.6\cdot10^{-5}$ at $\gamma$ — ratio 4.06
against the predicted 4.0, with $\omega_*$ unmoved. Both the
$\gamma$-independence of the position and the linearity of the depth
follow the reduced formula; this is the report's second, independent
confirmation that the well is the cross-term physics.

## 4b. Deep-well localization (adversarial)

`gamma16_localization.py`: the J1 rung at $16\gamma$. Depth
$1.03\cdot10^{-3}$ (vs $6.6\cdot10^{-5}$ at $\gamma$; ratio 15.7,
linearity intact), PR at the $\omega=0.35$ minimum: **106 sites** —
unchanged from the shallow regime. Under the dilution mechanism of
004/007 a term this strong delocalized to the box floor; the completed
square does not, as its convexity predicts.

## 5. Why local, not intensive

The intensive realization $(\int I_1)^2/V$ has a qualitatively
different minimum: the integral $\int(i_1^{\rm stat} - i_1^{\rm kin})$
crosses zero near $\omega \approx 5$ and the term simply vanishes there
(E_extra $\to 0$ within $10^{-5}$), riding on an otherwise flat statics
— a degenerate global cancellation, not a localized clock. The figure
below shows the template and ticking radial profiles (with the honest
point that the template is nearly flat — see Context) and the intensive
variant's zeroing.

![mechanism](results/fig_mechanism.png)

## 6. Relation to report 007

Report 007's conclusion stands: *plain* local quartics (in the ticking
density alone) delocalize, and a core weight fixes it. The new fact is
that $(I_1)^2$ is not a plain quartic — expanding the square generates
the core weight automatically, with the static density as the weight.
The two reports solve the same delocalization problem by different
routes: 007 keeps the Mexican-hat shape and repairs its concavity with
an intensive form and a core weight; 008 shows the simplest covariant
term avoids the concavity altogether — the completed square is convex
in the ticking density, so the dilution channel never opens. For the
article the $(I_1)^2$ form is preferred (the author's candidate, local,
no auxiliary functional).

## Limitations

- **Frozen clock tangent**: as in 004/007, $a_0$ is fixed from the
  centered polished field (origin-centered envelope included); the
  reduced functional is not translation-covariant, and an equivariant
  tangent remains open (007 review round 2).
- **Shallow well**: depth bounded by the statics-deformation budget
  (see Caveat above); stability of the clock against perturbations at
  these depths is untested.
- **Sampled minimum**: $\omega_* $ is bracketed by rungs
  (0.2/0.35/0.5), not continuously resolved; the frozen-profile value
  0.326 sits inside the bracket.
- **One generator**: boost-x only (by the isotropy of the hedgehog and
  004's channel measurements the other boosts are equivalent; rotations
  — the angular-momentum question — are a separate line).
- $32^3$ box, one lattice spacing; no continuum extrapolation.

## Author-gated physics choices

- The statics-deformation budget for $\gamma$ (depth vs 3×3-sector
  purity) — ties into scale anchoring ($\omega_* = mc^2/\hbar$).
- Whether $(I_1)^2$ replaces or accompanies the 007 intensive term in
  the article Lagrangian.

## Equation-to-artifact map

| object | artifact |
|---|---|
| ladders J1/J0/J2, prediction $C_1, C_2$ | `ladder_i1sq.py` → `results/i1sq_ladders.json` |
| $\gamma$-scaling confirmation | `confirm_gamma_scaling.py` → `results/gamma_scaling.json` |
| deep-well localization check | `gamma16_localization.py` → `results/gamma16_localization.json` |
| independent energy route | `verify_energies.py` |
| persisted J1 rung fields, frozen tangent | `results/j1_rung_om*.npz`, `results/a0_frozen.npz` |
| figures (from committed artifacts) | `make_figures.py` |

## Reproduction

`bash reproduce.sh` — with report 004's fields available (or
`M5_FIELDS_DIR`) it reruns both lattice producers (sentinel-flagged),
the independent route and the figures, and asserts the well, the
control, the scaling ratio and the route-2 match; without fields it
verifies the committed artifacts' internal consistency and reports
NOT-REPRODUCED for the lattice legs.
