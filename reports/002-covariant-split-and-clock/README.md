# Report 002 — A covariant rot/boost split, the one-line boundedness fix, and a finite clock from a boost condensate

*2026-08-20 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
stage result; pointwise algebra only — no lattice solitons yet (§6)*

## Context and results

Report 001 ended with a verified no-go: no constant-coefficient combination
of the six quadratic invariants can repair the negative clock channel while
preserving the 3×3 sector. Any repair must therefore be field-dependent,
higher order, or constrained. This report takes the field-dependent route and
establishes, at the level of exact pointwise algebra (float64, all claims
scripted):

1. **A covariant rot/boost split of $F$ exists.** The time axis is selected
   by the field $M$ itself; three concrete constructions all pass a numerical
   covariance gate, with a measured trade-off table (§2).
2. **A one-line covariant boundedness fix.** Contracting the matrix pair with
   the field-selected "Euclideanizer" $G$ flips every boost kin channel from
   negative to positive with the same magnitude, leaves every rotation
   channel bit-identical, and reduces exactly to the 3×3 sector (§3).
3. **Taking the proposed $R^2-(g^2\tilde R)^2$ literally as an energy density
   makes the runaway $\sim g^4$ worse** in the Hamiltonian reading; the
   intended minus must act at the force level, not the energy-density level
   (§3).
4. **A linear-in-$\partial_t M$ (Berry/Wess–Zumino) clock term cannot work
   for the energy question**: it drops out of the Hamiltonian exactly under
   the Legendre transform (honest negative, §4).
5. **A boost condensate delivers a finite clock.** With the split in hand,
   $\mathcal L_C=\mathrm{rot}^2-a\,\mathrm{boost}^2+b\,(\mathrm{boost}^2)^2$
   is globally bounded below, preserves the 3×3 sector exactly, and on clock
   textures its energy has a minimum at finite nonzero
   $\omega^*=\sqrt{a/(2b\,k_2)}$ — measured 1.600 vs analytic 1.581, with
   $E(\omega^*)<E(0)$ (§5, Fig. A).

## 1. Why field-dependent, and the object that does it

All constructions below are built from one covariant object: a projector onto
the timelike axis of $\eta M$. Near the vacuum
$M_{\mathrm{vac}}=\mathrm{diag}(-sg,1,\delta,0)$ the $\eta M$ spectrum is
$(sg,1,\delta,0)$ with an isolated large timelike branch, so $M$ itself
carries a preferred time direction — using it keeps everything Lorentz
covariant (the projector transforms with $M$), which is exactly what a
constant-coefficient construction cannot do (report 001, no-go).

The "Euclideanizer" is then

```math
G \;=\; \eta \;-\; 2\,P_t\,\eta ,
```

with $P_t$ the timelike projector: $G$ is positive at the vacuum
($\mathrm{diag}(1,1,1,1)$ in the vacuum frame) and defines the channel split

```math
\mathrm{rot}^2=\tfrac12\big(\langle F,F\rangle_G+\langle F,F\rangle_\eta\big),
\qquad
\mathrm{boost}^2=\tfrac12\big(\langle F,F\rangle_G-\langle F,F\rangle_\eta\big),
```

both nonnegative in the Hamiltonian reading (derivative pair contracted with
$G$ as well; energy in the $M$-selected frame).

## 2. Three constructions of $P_t$, gated

| route | $P_t$ | covariance gate | 3×3 reduction | smooth / complex-step |
|---|---|---|---|---|
| exact | timelike eigenvector of $\eta M$ | $7\cdot10^{-12}$ | **exact (0.0) on arbitrary spatial fields** | autograd OK; CS unsafe |
| soft powers | $(\eta M)^n/\mathrm{tr}(\eta M)^n$ | $6\cdot10^{-14}$ | $\sim(\lambda_{\rm sp}/sg)^n$: $5\cdot10^3\,(n{=}2)\to2.4\cdot10^{-2}\,(n{=}8)$ under the $g^4$ weight | fully safe ($9\cdot10^{-16}$) |
| Lagrange | $q(\eta M)$, $q$ = fixed cubic with $q(sg){=}1$, $q(1){=}q(\delta){=}q(0){=}0$ | $3.4\cdot10^{-13}$ | **exact ($9\cdot10^{-16}$) on on-potential spectra** (where the potential drives every field); $O(10)$ far off-shell | fully safe |

Negative controls (non-$\eta$-orthogonal $\Lambda$) fail the covariance gate
by $O(1)$, as they must. The exact route passes the strictest known form of
the 3×3-preservation test — arbitrary admissible spatial fields, not a
special subfamily. The Lagrange route is branch-free and polynomial: the
projector is exact precisely on the spectrum the potential enforces.

## 3. The clock-channel table (Fig. B)

kin$(M;a_0)$ = the $\omega^2$ coefficient of the energy for velocity
direction $a_0$ (conjugation tangents of the generator catalog), on a random
spatial background:

| channel | $\eta$ (current) | $G$ |
|---|---|---|
| rot xy / xz / yz | +44.4 / +36.9 / +33.4 | identical to $\eta$ |
| boost x / y / z | −51.8 / −13.9 / −41.7 | **+51.8 / +13.9 / +41.7** |

One contraction change closes the runaway channel covariantly — the
covariant version of the openwave M5.21.16 Euclidean fix, which was known to
break boost covariance; here covariance is gated. Taking
$R^2-(g^2\tilde R)^2$ literally as the contraction metric instead gives
boost channels at $-2.1\cdot10^5$ (the $g^4$ weight amplifies the wrong
sign in the Hamiltonian reading): the minus of the Newton program must be
realized at the interaction level, not as a negative energy density.

![Fig B](figs/figB_kin_channels.png)

## 4. Honest negative: linear clock terms drop out of the energy

For any term linear in the velocity,
$L=\mathrm{kin}\,\dot q^2+\kappa B(q)\dot q-V$:
$p=2\,\mathrm{kin}\,\dot q+\kappa B$ and
$H=p\dot q-L=\mathrm{kin}\,\dot q^2+V$ — the $\kappa$ term cancels exactly
(numerically $2.7\cdot10^{-15}$). Berry/Wess–Zumino terms change the
dynamics (precession), not the energy landscape, so they cannot by
themselves produce a minimum of $E$ at $\omega\neq0$.

## 5. The boost condensate: a finite clock from free minimization

```math
\mathcal L_C \;=\; \mathrm{rot}^2 \;-\; a\,\mathrm{boost}^2
\;+\; b\,\big(\mathrm{boost}^2\big)^2 , \qquad a,b>0 .
```

Measured properties (all in `results/clock_results.json`):

- **globally bounded below** by $-a^2/4b$ pointwise; a 1000-point
  random-direction dive scan (the lesson of the eigenvalue-lift collapse)
  finds minimum density 0.0 vs floor −5;
- **exact 3×3 preservation**: $\mathrm{boost}^2$ vanishes identically on
  spatial fields (measured 0.0) — Coulomb untouched, and the vacuum carries
  no clock;
- **finite clock**: on the clock texture, energy
  $E(\omega)=E_{\rm stat}-a k_2\omega^2+b k_2^2\omega^4$ has its minimum at
  $\omega^*=\sqrt{a/(2bk_2)}$: measured $\omega^*=1.600$, analytic 1.581
  (grid step 0.05), with $E(\omega^*)=-5.0<E(0)=0$;
- the three-functional comparison on one texture (Fig. A): current $\eta$
  diverges downward, all-$G$ is bounded but puts the minimum at
  $\omega^*=0$ (clock dies), the condensate has the Mexican-hat profile
  with a finite preferred frequency — the time-crystal structure this
  program has been looking for, at the algebra level.

![Fig A](figs/figA_clock_family.png)

## 5b. Relation to the $\varepsilon$-family (report 001's scope limit)

A natural question: is the rot channel the Levi-Civita contraction of $F$?
Half yes, in a precise sense — verified identities (`dual_identity.py`,
both to $3\cdot10^{-16}$):

```math
\mathrm{boost}^2 = 2\,\lVert E\rVert^2,\quad E_{\mu\nu|\beta}=u^\alpha F_{\mu\nu\alpha\beta};
\qquad
\mathrm{rot}^2 = 2\,\lVert B\rVert^2,\quad B^{\mu\nu|\gamma}=\tfrac12\,\varepsilon^{\gamma\alpha\beta\delta}u_\delta F_{\mu\nu\alpha\beta}.
```

The split is exactly the electric/magnetic decomposition of the matrix-pair
2-form w.r.t. the field-selected observer $u$, and the dual (magnetic) part
does carry an $\varepsilon$ — but squared, and two $\varepsilon$'s collapse
to metrics, so both channels are parity-even. None of the four genuine
one-$\varepsilon$ pseudoscalars is used ($E\!\cdot\!B$-type witness printed
nonzero and unused). Where this report sits relative to report 001's three
exclusions: it deliberately occupies two of them — field-dependent
contractions ($G=G(M)$) and quartic order in $F$ (the condensate term) —
as forced by the no-go; the parity-odd family remains untouched territory.

## 6. What this report does not show

- Everything is **pointwise algebra**: no lattice soliton has been relaxed
  under $\mathcal L_C$ yet — the electron hedgehog run (finite $\omega^*$
  with the full profile backreaction, Coulomb tail intact) is the next
  stage, on the existing lattice stack.
- **Newton's sign is not measured**: boundedness and attraction are
  different questions; the two-body read is future work (a bench also being
  built independently in substrate-framework P239).
- The rotation-direction generator for a concrete Berry-type dynamics, the
  $s=-1$ branch (odd powers of $\eta M$), and the physical anchoring of
  $a,b$ (Jarek: $\omega^*=mc^2/\hbar$) are open; the split definition
  choice among the three routes is the model author's call.
- Off-shell fields see $O(1)$ 3×3 contamination on the Lagrange route
  (exact only on-potential); the exact-eig route has no such caveat but is
  not complex-step differentiable.

## 7. Reproduction

```bash
pip install sympy torch
./reproduce.sh          # ~1 min CPU
```

Asserts: both covariance gates with working negative controls, exact-route
3×3 reduction = 0, the kin sign flips (boosts $\eta<0$, $G>0$; rotations
equal), the B1 Legendre drop-out, $|\omega^*-\omega^*_{\rm analytic}|$
within the scan step, the dive-scan floor, and the spatial guard. Regenerates
both figures.

## Equation-to-code map

| object | code |
|---|---|
| $G$ exact / soft / Lagrange | `proto.py::G_exact/G_soft`, `clock_tests.py::G_lagrange` |
| covariance gate + negative control | `proto.py` gate 1, gate 6 |
| 3×3 reduction measurements | `proto.py` gates 2, 6 |
| kin channel table | `proto.py` gate 3 (`kinH`) |
| literal $-g^4$ variant | `proto.py` gates 3–4 (`X_B`) |
| Legendre drop-out | `clock_tests.py` B1 block |
| channel densities (H-reading) | `clock_tests.py::channels` |
| $E(\omega)$, dive scan, spatial guard | `clock_tests.py` B3 blocks |
| E/B dual identities (§5b) | `dual_identity.py` |
| figures | `make_figures.py` |

## Provenance

- Conventions and the no-go baseline: [report 001](../001-quadratic-contractions/).
- The idea family (incl. the condensate and the frozen-spectrum projector)
  originates from an internal AI ideas round (2026-08-20); the positive
  field-dependent internal metric is independently the direction of
  substrate-framework P239 ("spectral-Cartan", PR #148) — the constructions,
  trade-off measurements, the $-g^4$ warning, the Legendre negative, and the
  condensate clock mechanism here are this report's own.
- Physics targets: J. Duda, e-mails and messages of 2026-08-13/20.
