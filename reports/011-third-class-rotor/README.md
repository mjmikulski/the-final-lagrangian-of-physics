# Report 011 — The equivariant hedgehog: a constant axial line tension, and prescribed rotation buys peripheral (not core) inertia

*2026-08-29 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
follow-up of report 009's trichotomy: can the finite-inertia third
class (core-breaking texture on an equivariant background) be made a
*relaxed* configuration? Includes the statics-extensivity check added
at plan review (the axial-defect line tension vs box size).*

## Context and result

Report 009 ended with a measured trichotomy for rigid rotations of
the hedgehog: extensive (globally non-equivariant texture), trivial
(exactly equivariant texture: stabilizer), and finite (compact
core-breaking on an equivariant background — the third class,
measured there on a hand-built ansatz). This report asks the two
follow-up questions: **(A)** is the equivariant background itself a
healthy static configuration — in particular, does its unavoidable
axial frame defect cost an energy that scales badly with box size
("trading extensive inertia for extensive statics", the plan-review
concern)? **(B)** does the third-class core deformation *survive
relaxation*, i.e.\ is there a relaxed (not hand-built) representative?

**Answers, measured:**

1. **The axial defect carries a constant, small line tension —
   now established under an observable-level stopping rule** (§2;
   the first version's "shrinking" claim was a relaxation artifact,
   round 1; round 2 correctly noted the first continuation gated on
   the energy, not the observable). `lambda_plateau2.py` continues
   $L = 36$ and $L = 48$ until the tube excess itself drifts
   $< 1\%$ over four consecutive cycles: $L = 36$ settles at
   $8.271\cdot10^{-4}$ (four identical records), $L = 48$ at
   $7.735\cdot10^{-4}$ — **plateaus agreeing to $6.5\%$**. The
   axial line tension is constant in $L$ at
   $\lambda_{\rm axis} \approx (7.7$–$8.3)\cdot10^{-4}$ for
   $L \ge 36$ (the $L = 24$ box, $5.0\cdot10^{-3}$, is too small):
   first-order extensive — linear in $L$, like the radial-texture
   energy — small, but not vanishing. Known systematic: at $L = 24$
   halving $h$ shifts the excess by $+43\%$; an $h$-study at
   $L \ge 36$ is open.
2. **The $\delta = 1/8$ sign claim survives at trajectory level**
   (§2): the excess stays negative through all 24 continuation
   cycles ($-2.47 \to -2.27\cdot10^{-3}$, drifting slowly toward
   zero with $\lVert g\rVert_\infty = 0.19$ still high) — the
   qualitative statement (the axis is cheaper than the background at
   small $\delta$) holds on the measured trajectory; its asymptote
   is not yet converged.
3. **The spectral core deformation does not survive relaxation**
   (§3): the round-4 representative ($\varepsilon\chi(r)D$,
   $D = \hat x\hat x^\top - \hat y\hat y^\top$) changes eigenvalues,
   the potential punishes it, and after the deep protocol the
   inertia excess over the equivariant background collapses from
   $\sim 420$ (seed) to $\sim 5$ (relaxed) — within the noise of the
   diffuse lattice-anisotropy background (PR $\sim 10^{3-4}$).
4. **The spectrally neutral (pure frame-twist) deformation does not
   survive either** (§3): a core twist of the eigenframe about the
   $x$ axis ($\beta_0 = 0.6$, $V_4$-identical to the background)
   is likewise ironed out — raw inertia difference $-5.2$ after
   relaxation, excess density diffuse (PR $\approx 970$). **Nothing
   in the static energy stabilizes a symmetry-breaking core.**
5. **Prescribed rotation grows peripheral inertia without a seeded
   deformation — a qualitative observation, scoped in round 2**
   (§4). The EQ-start runs show that at $J = 2, 4, 6$ the minimizer
   grows large inertia ($I \approx 277/596/954$) from the purely
   equivariant seed, and the order parameter shows where: the
   kinetic-density excess is concentrated at $r \in (9, 18)$,
   centroid $\approx 14.7$ against the boundary at $18$ — an
   orbital-lever **peripheral** mode (density $\propto \rho^2$),
   presumably box-limited, not a stabilized core deformation.
   Round 2's accuracy audit is accepted in full: none of the eight
   branch runs met the stated tolerance (all hit the 16-cycle cap;
   residuals $0.017$–$0.183$; the $J = 4$ branch gap $0.014$ is
   smaller than the unresolved last-cycle decreases $0.017$), so
   **no branch-selection or matched-accuracy claim is made**, and
   the $J = 4 \to 0$ continuation shows only that the inertia
   *scalar* returns ($591 \to 108$) — configuration-space
   reversibility against a converged reference is not established
   (the continuation in fact found $E_{\rm stat} = 4.470$, below
   the unconverged $J = 0$ endpoint $4.485$). The earlier
   "centrifugally stabilized core rotor" headline stays withdrawn;
   the core mode remains unstabilized by everything tried here, and
   the spin question stays open (§5).

**Honest scope:** the relaxation budget (Adam 1000 + 6 L-BFGS cycles
from analytic seeds) does not converge the $32^3$ boxes to the
$10^{-3}$ residual class of the 004 stack ($\lVert g\rVert_\infty$
grows with $N$ up to $1.4\cdot10^{-1}$); total energies $E(L)$ are
therefore *not* interpreted (their apparent decrease with $L$ is a
budget artifact), and all conclusions rest on local, well-relaxed
quantities (tube tensions far from the center) and on
*differences* between identically-protocoled runs (EQ vs CB pairs).
The lattice's own breaking of axial symmetry dominates the relaxed
inertia measurements (PR grows to $\sim 10^4$); the deformation
signals are read against that background as matched-pair excesses.

## 1. Setup

Stack of report 004 with patched $(N, L, \delta)$ (`defs_011.py`):
$H = 1.5$ boxes $N = 16, 24, 32$; one $h$-test pair at $L = 24$
($h = 1.5$ vs $0.75$); $\delta \in \{0.3, 1/8\}$. Seeds: EQ
(spherical transverse frame, axis regularization
$g(\rho) = 1 - e^{-(\rho/\rho_0)^2}$, $\rho_0 = 3$), CB (EQ $+$
$\varepsilon\,\chi(r)\,D$ spectral core deformation,
$\varepsilon = 0.3$, $r_0 = 6$), CB2 (EQ with the eigenframe twisted
by $\beta(r) = \beta_0 e^{-(r/r_0)^2}$ about $\hat x$ — spectrally
neutral). Frozen shell at the (equivariant) seed formula in every
case. Deep protocol from the start (the 009 lesson): Adam 1000 +
6 L-BFGS cycles with recorded `E_levels`.

## 2. The statics-extensivity check, done to plateau

The tube pair ($\rho < 3$, $|{\rm axis}| > 6$) compares the axial
defect against the hedgehog's own radial-texture background at the
same distance. Round 1 correctly objected that the first version's
endpoints had incomparable residuals; round 2 that the first
continuation (`lambda_plateau.py`) gated on the total energy, so its
$L = 36$ record was not actually flat (it dipped to $4.9\cdot10^{-4}$
and was still rising $1.8\%$/cycle when the energy rule fired).
`lambda_plateau2.py` therefore continues $L = 36$ and $L = 48$ under
an **observable-level** rule — stop only when the tube excess drifts
$< 1\%$ over four consecutive cycles:

| $L$ | 24 | 36 | 48 |
|---|---|---|---|
| $\lambda_z - \lambda_x$, observable-level plateau | $5.0\cdot10^{-3}$ | $8.271\cdot10^{-4}$ | $7.735\cdot10^{-4}$ |
| how it stopped | converged (1 cycle) | +13 cycles, then four identical records | +4 cycles, drift $< 1\%$ |

The plateaus agree to $6.5\%$: the axial defect has a **constant
line tension** $\approx (7.7$–$8.3)\cdot10^{-4}$ (total axial cost
linear in $L$, the same order of extensivity as the radial-texture
energy — no worse class, but not vanishing; the first version's
"shrinking to 2\%" was the unconverged $L = 48$ point, and the first
continuation's "1.4\% agreement" was an energy-gated coincidence). $h$-sensitivity at $L = 24$: $+43\%$ under $h \to h/2$
(measured on well-converged endpoints, $\lVert g\rVert_\infty \le
5\cdot10^{-3}$); the $h$-study at plateau-relevant $L$ is open. At
$\delta = 1/8$ the excess is negative on the entire 24-cycle
trajectory ($-2.47 \to -2.27\cdot10^{-3}$; residual still high) —
the sign claim holds, the asymptote is not converged.

![statics](results/fig_statics.png)

## 3. Neither core deformation survives statics

Matched pairs at $N = 24$ (identical protocol, same boundary):

| deformation | seed $\Delta I$ | relaxed $\Delta I$ | excess density |
|---|---|---|---|
| CB (spectral, $\varepsilon\chi D$) | $+418$ | $+6.9$ | diffuse (PR $\approx 3700$) |
| CB2 (frame twist, $\beta_0 = 0.6$) | $+134$ | $-5.2$ (raw) | diffuse (PR $\approx 970$) |

The spectral deformation is removed by the potential (eigenvalue
penalties), the frame twist by the gradient terms (nothing protects
a local frame rotation). Combined with report 009: the third class
exists kinematically but **is not selected by the static energy** —
a relaxed static hedgehog is (up to lattice anisotropy) equivariant,
hence rotationally trivial.

![survival](results/fig_survival.png)

## 4. Prescribed rotation: spontaneous, peripheral, with a returning inertia scalar

Round 1 objected that the single CB-start run could not distinguish
selection from initialization; `centrifugal_branches.py` runs both
branches (EQ-start and CB-start) at $J = 0, 2, 4, 6$ with recorded
residuals, a hysteresis continuation, and a field-space order
parameter. Round 2's audit of this producer is accepted: **every run
hit the 16-cycle cap without meeting the $10^{-3}$ stopping rule**
(final per-cycle decreases $3.3\cdot10^{-3}$–$1.1\cdot10^{-2}$,
$\lVert g\rVert_\infty$ up to $0.183$), so the table below is a
**qualitative record**, not a matched-accuracy comparison:

| $J$ | 0 | 2 | 4 | 6 |
|---|---|---|---|---|
| $I$ (EQ-start) | 108 | 277 | 596 | 954 |
| $I$ (CB-start) | 110 | 297 | 641 | 1034 |
| $E_J$ (EQ / CB) | 4.485 / 4.493 | 4.489 / 4.509 | 4.519 / 4.534 | 4.564 / 4.532 |
| excess centroid $r$ (EQ) | — | 14.7 | 14.7 | 14.5 |

Three findings, at their honest strength. (i) **Qualitative
spontaneity**: the equivariant seed builds inertia of the same order
as the pre-broken one ($596$ vs $641$ at $J = 4$) — no seeded
deformation is needed. The branch $E_J$ orderings are **not
resolved** (gaps smaller than the unresolved decreases); no
selection claim. (ii) **The inertia scalar reverses**: continuing
the $J = 4$ endpoint at $J = 0$ returns $I: 591 \to 108$; because
the reference itself is unconverged, this is not yet
configuration-space reversibility (a field-distance comparison
against a converged reference is the open check).
(iii) **The order parameter overturns the core reading**: the excess
density lives in the outer shells, $r \in (9, 18)$ with centroid
$\approx 15$ against the boundary at $18$ — the minimizer buys
inertia where the orbital lever is longest (kinetic density
$\propto \rho^2$), i.e.\ at the periphery of the box. This is an
orbital-lever mode, expected to migrate outward with box size (not
tested here), **not** a stabilized core deformation. The
first version's core-rotor headline is withdrawn; what survives is
the mechanism (prescribed $J$ reshapes the field, with the
inertia scalar returning on the tested continuation) and
the negative: **nothing tried in this report stabilizes a
core-localized spin mode**.

The complementary small-$J$ scan and the noise-limited fixed-J scan
on the relaxed $32^3$ field are retained in the JSON record
(`rot_stabilization.json`, `fixedj_cb.json`) with their round-1
caveats.

## 5. Road forward

- The spin question is fully open after this report: rigid rotations
  (009) and prescribed-$J$ energy shaping (here) both fail to
  produce a core-localized angular-momentum carrier; the
  prescribed-$J$ route additionally needs a periphery-excluding
  formulation (a core-restricted collective ansatz, or a box-size
  sweep of the peripheral mode to expose its $L$-dependence) before
  any further claim.
- The axial line tension is constant and small; its $h$-dependence
  at plateau-relevant $L$ and the $\delta = 1/8$ asymptote are the
  remaining statics measurements.

![centrifugal](results/fig_centrifugal.png)

The complementary `fixedj_cb.py` scan on the *relaxed* $32^3$ field
is reported honestly as noise-limited: the relaxation creep
($\sim 10^{-2}$) exceeds the rotational energies ($\lesssim 3\cdot
10^{-3}$ at $J \le 0.8$), so only the qualitative $I(J)$ rise
($87 \to 108$) survives as a trace of the same mechanism at small
$J$.

## Limitations

- Relaxation budget as in Honest scope: no converged $E(L)$; all
  claims are matched-pair or local.
- The equivariant boundary makes the global combined rotation a
  symmetry of the configuration space up to the $1/r$ discretization
  residual (009 §6); the Noether construction at finite $h$ is
  approximate to that residual.
- One deformation magnitude per type ($\varepsilon = 0.3$,
  $\beta_0 = 0.6$); no claim that *no* static deformation of any
  shape survives — the two natural representatives were tested.
- $32^3$ and below; $h$-test at one pair.

## Equation-to-artifact map

| object | artifact |
|---|---|
| seeds, stack patching, protocol, measurement kit | `defs_011.py` |
| all relaxations (L-sweep, h-pair, $\delta = 1/8$) | `relax_all.py` → `results/relax_all.json`, `results/M_*.npz` |
| scaling analysis (inertia, energies, tube tensions) | `analysis.py` → `results/analysis.json` |
| frame-twist survival test | `frame_twist.py` → `results/frame_twist.json` |
| centrifugal stabilization test | `rotational_stabilization.py` → `results/rot_stabilization.json` |
| fixed-J scan on the relaxed field | `fixedj_cb.py` → `results/fixedj_cb.json` |
| per-cycle tube-observable trajectories (plateau) | `lambda_plateau.py` → `results/lambda_plateau.json` |
| observable-level plateau ($L = 36, 48$) | `lambda_plateau2.py` → `results/lambda_plateau2.json` |
| two-branch centrifugal test + hysteresis + order parameter | `centrifugal_branches.py` → `results/centrifugal_branches.json` |
| figures | `make_figures.py` |

## Reproduction

`bash reproduce.sh` — with report 004's stack (or `M5_FIELDS_DIR`)
reruns all producers (sentinel-flagged) and asserts the shrinking
axial excess, the negative-at-$1/8$ axial cost, the collapse of both
core deformations under statics, and the fixed-J records; without
the stack it checks the committed artifacts.
