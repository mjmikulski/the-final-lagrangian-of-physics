# Report 006 — The Newton sign on boost hedgehogs: no constant-coefficient quadratic Lagrangian attracts and stays stable

*2026-08-22 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
Newton-sign stage of the program; task and canonical ansatz set by the model's
author (J. Duda, notebook + correspondence of 2026-08-21)*

## Context and result

The M5 4×4 model must reverse the sign of the boost-sector (Newton)
interaction while keeping the spatial 3×3 (Coulomb) sector intact. The
author's proposal: add further Lorentz-covariant curvature² contractions
(the $I_2..I_6$ of report 001) to the $-I_1$ baseline; his notebook probes
the sign question on **boost hedgehogs** — vacuum
$M_0=g\,\mathrm{diag}(1,0,0,0)$ dressed by a radial first-order boost field
— and reports an attractive $1/d$ fit after flipping the sign of the
rotational-curvature square. This report treats that ansatz as canonical
and answers the general question: **can any constant-coefficient quadratic
Lagrangian make boost hedgehogs attract without destabilizing them?**

**Result: no — three exact structural facts and two measured signs close
the entire quadratic family at leading order.**

1. **Structure theorem (§2, exact).** On the canonical ansatz the field
   strength is a *purely spatial* double two-form,
   $F_{ij}=-v_i\wedge v_j$ with $V=(v_1v_2v_3)$ the (symmetric) Jacobian
   of the dressing. Consequently every quadratic invariant collapses to a
   two-parameter family $\alpha\,e_1+\beta\,e_4$; the exact
   3×3-preserving directions $N_1,N_2,N_3$ of P239/report 001 — *the
   author's proposed repair space* — and all parity-odd invariants of
   report 005 **vanish pointwise**: they cannot move the Newton sign at
   all while preserving the 3×3 sector.
2. **Virial identity (§3, exact).** On every single radial dressing,
   $x^2(3e_1-4e_4)=-4\,\frac{d}{dx}\!\left[x^3f^4\right]$, so
   $S_1/S_4=4/3$ exactly: single-hedgehog stability of
   $\alpha e_1+\beta e_4$ is the half-space $4\alpha+3\beta\ge0$.
3. **Pointwise lemma (§3, exact).** On eigenvalues $(a,b,c)$ of $V$:
   $e_1-e_4=(ab\!-\!bc)^2+(bc\!-\!ca)^2+(ca\!-\!ab)^2$ and
   $4e_4-e_1=4(ab\!+\!bc\!+\!ca)^2$, hence $\rho=e_1/e_4\in[1,4]$.
4. **Measured signs (§4).** Both channels have *repulsive* long-range
   tails on every tailed profile tested ($E^{\rm int}_1,E^{\rm int}_4>0$),
   and the marginal-direction excess $X=3E^{\rm int}_1-4E^{\rm int}_4$ —
   whose self part cancels exactly by the virial — is positive on every
   tested two-body configuration, cutoff-free configurations included.
   Multi-center witnesses reach $S_1/S_4=1.80$.
5. **No-go (§5).** (1)–(4) exclude attraction-with-stability for every
   $(\alpha,\beta)$, in all sign branches. In particular the M5 baseline
   itself gives a *repulsive* Newton tail on boost hedgehogs, and the
   author's sign-flip experiment buys attraction at the price of a
   bottomless self-energy — whose fitted numbers are additionally an
   artifact of an IR-divergent integral (§4, measured
   $S\simeq85.3\,R$).

The repair must therefore leave the constant-coefficient quadratic family:
higher order in $F$, field-dependent coefficients, non-commutator
derivative terms, or beyond-leading-order dressings (§7).

## 1. The canonical ansatz and conventions

Conventions of report 001 ($\eta$-commutator $F$, slot order, invariants
$I_1..I_6$; $e_1,e_4$ denote the densities of $I_1,I_4$). The ansatz
(author's notebook, first order in the dressing amplitude $m$): vacuum
$M_0=g\,\mathrm{diag}(1,0,0,0)$, static dressing by local boosts with
radial axis and profile $f$,

```math
o=\prod_a\exp\!\big(m\,f(r_a)\,\mathbf r_a\!\cdot\!\mathbf K\big),\qquad
M=o\,M_0\,o^\top,\qquad \mathbf r_a=\mathbf x-\mathbf c_a,
```

with $\mathbf K$ the boost generators. To $O(m)$,
$\delta M=g\,(\mathbf u\,e_0^\top+e_0\,\mathbf u^\top)$ with the dressing
field $\mathbf u=\sum_a f(r_a)\,\mathbf r_a$ — a **gradient** field
($\mathbf u=\nabla\phi$, $\phi=g\sum_a\!\int^{r_a}\!f(s)s\,ds$), so its
Jacobian $V_{ai}=\partial_i u_a$ is symmetric with closed form
$V=\sum_a g\big[\tfrac{f'(r_a)}{r_a}\mathbf r_a\mathbf r_a^\top+f(r_a)\mathbf 1\big]$.
The leading energy is $O(m^4)$ and static ($A_0=0$); $g=m=1$ throughout.
Energy densities follow the working-stack sign (report 004 statics:
positive $e_1$-type energy); the author's notebook density is $-e_1/4$
exactly (§2, verified).

## 2. Structure theorem

With $A_i=v_ie_0^\top+e_0v_i^\top$ ($v_i$ spatial) and
$\eta=\mathrm{diag}(-1,1,1,1)$:

```math
A_i\,\eta\,A_j=-v_iv_j^\top+(v_i\!\cdot\!v_j)\,e_0e_0^\top
\;\Rightarrow\;
F_{ij}=A_i\eta A_j-A_j\eta A_i=-(v_iv_j^\top-v_jv_i^\top)
=-\,v_i\wedge v_j ,
```

(the $e_0e_0^\top$ parts cancel by symmetry of the dot product): $F$ has
**spatial derivative indices** (statics) **and spatial matrix indices** —
a purely spatial double two-form, for arbitrary profiles and centers.
Verified two ways: symbolically on a general coefficient matrix
(`verify_symbolic.py` §1) and numerically against matrix-exponential
autograd (`check_structure.py` §§1–3, residuals $10^{-16}$, time
components exactly 0). Consequences (both routes, exact):

- **P239's spatial nullspace applies pointwise**: $N_1=N_2=N_3=0$, so the
  exact-3×3-preserving family $-I_1+aN_1+bN_2+cN_3$ has *identically* the
  baseline density — no choice of $(a,b,c)$ changes any energy of any
  configuration of this ansatz. The "add $N$-safe curvature² terms to fix
  Newton" route (and any TEGR-style tuned combination within it) is
  **empty at leading order**.
- All four one-$\varepsilon$ pseudoscalars of report 005 vanish (symbolic
  zero), consistently with 005's spatial-vanishing.
- The six invariants collapse: $I_2=I_1$, $I_5=I_4$, $I_3=I_1/2$,
  $I_6=4I_4-I_1$ (symmetry of $V$), with closed forms
  $e_1=2[(\mathrm{tr}\,G)^2-\mathrm{tr}\,G^2]$, $G=V^2$, and
  $e_4=\mathrm{tr}[(V^2-\mathrm{tr}V\cdot V)^2]$. The most general
  constant-coefficient quadratic energy density on the ansatz is
  $\alpha\,e_1+\beta\,e_4$.
- The author's notebook spatial term (rotational components of the plain
  commutators, squared, with a minus) equals $-e_1/4$ exactly: his
  experiment is the sign-flipped baseline, $(\alpha,\beta)=(-\tfrac14,0)$.

## 3. Two exact inequalities

**Virial identity.** For a single radial dressing, with
$\lambda_r=f+rf'$, $\lambda_t=f$ the eigenvalues of $V$:

```math
x^2\,(3e_1-4e_4)=-4\,\frac{d}{dx}\big[x^3f(x)^4\big]
\quad\Rightarrow\quad
3S_1=4S_4
```

for every profile with $r^3f^4\to0$ at both ends (checked symbolically as
a differential identity, and numerically to $10^{-8}$ by quadrature and
by a cutoff-free 2D grid). So the single-hedgehog energies of
$\alpha e_1+\beta e_4$ are $S_4(4\alpha+3\beta)/3$ with $S_4>0$:
**stability of singles $\iff 4\alpha+3\beta\ge0$.**

**Pointwise ratio lemma.** On eigenvalues $(a,b,c)$ of $V$:

```math
e_1-e_4=(ab-bc)^2+(bc-ca)^2+(ca-ab)^2\ \ge0,\qquad
4e_4-e_1=4\,(ab+bc+ca)^2\ \ge0,
```

so $\rho=e_1/e_4\in[1,4]$, both ends attained (isotropic $V$;
$ab+bc+ca=0$). Sampled realizable fields cover $[1.03,\,4.00]$. This
yields an independent, stronger no-go layer: *pointwise* non-negativity of
$\alpha e_1+\beta e_4$ forces $-\beta/\alpha\le1$ (for $\alpha>0$) or
$\beta/|\alpha|\ge4$ (for $\alpha<0$), both incompatible with attraction
(§5) by a wide margin.

## 4. Measurements

Instrument, route 1: closed-form $V$ on cylindrical (axisymmetric) or
full-3D grids, float64, GPU; interaction energies
$E^{\rm int}_k(d)=E_k(d)-2S_k$ with self-energies from the same scheme;
convergence scans in grid step, domain and core cutoff
($10^{-3}$–$10^{-2}$ relative; `results/energy_results.json`).
**Route 2 for every load-bearing measured inequality**
(`verify_measures.py`, numpy/scipy, no shared code): densities via the
tensor route (finite-difference $A_i$ from the ansatz $\delta M$,
$\eta$-commutator $F$, full report-001 contractions instead of the
closed forms), single-hedgehog energies by 1D radial quadrature, pair and
cluster integrals by Sobol quasi-Monte Carlo. Route 2 independently
confirms the virial ratio, $E^{\rm int}_k>0$, $t_1>4/3$ and $X>0$ at the
assembly-binding points (within 5% of route 1), the cutoff-free gaussian
$X>0$, and the chain-7 witness ratio (within 1%);
`results/verify_results.json`.

1. **Repulsive tails, both channels.** For screened power profiles
   $f=e^{-\mu r}r^{-p}$ in the Newton window $\mu d\le0.6$:
   $E^{\rm int}_1,E^{\rm int}_4>0$ throughout, and

   | profile | $t_1(d)=E^{\rm int}_1/E^{\rm int}_4$ |
   |---|---|
   | $p=0.3,\ \mu=0.2$ | $[1.335,\,1.357]$ |
   | $p=0.5,\ \mu=0.1$ | $[1.336,\,1.347]$ |
   | $p=0.5,\ \mu=0.2$ | $[1.340,\,1.380]$ |
   | $p=0.75,\ \mu=0.2$ (UV-cutoff-flagged) | $[1.370,\,1.427]$ |
   | $p=1.0,\ \mu=0.2$ (UV-cutoff-flagged) | $[1.51,\,1.55]$ |

   always **above 4/3**; equivalently the marginal-direction excess
   $X=3E^{\rm int}_1-4E^{\rm int}_4>0$ (stable under grid/cutoff variants,
   e.g. $+11.6\pm0.4$ at $p=0.5,\mu=0.1,d=2$; positive also on cutoff-free
   gaussian pairs at every separation, $+2.05$ at $d=0.5$). UV convergence
   of the self-energy is *strict* $p<3/4$ ($r^{2-4p}dr$; at $p=3/4$ it
   diverges logarithmically and the virial boundary term survives), so
   both $p\ge3/4$ rows are cutoff-flagged context, excluded from the
   assembly; the clean set is $p\in\{0.3,0.5\}$.
2. **Cluster witnesses.** Multi-center gaussian configurations reach
   $S_1/S_4=1.52$ (trio), $1.58$ (five), $1.61$ (ring), **$1.80$
   (chain of 7)** — all cutoff-free.
3. **The notebook protocol is IR-divergent.** For the unscreened
   $f=r^{-1/2}$ of the author's notebook, the self-energy grows linearly
   with the domain radius ($S_1\simeq85.3\,R$; $R$-scan 10–40), so the
   notebook's fitted constant and $1/d$ coefficient
   ($-863.7-167.7/d$, adaptive quadrature over an infinite domain) are not
   convergent quantities. The *qualitative* conclusion survives on
   convergent (screened) profiles: the flipped baseline
   $(\alpha,\beta)=(-\tfrac14,0)$ indeed attracts at range — and has
   negative-definite energy, i.e. the runaway.
4. **Compact-dressing pocket (curiosity).** Gaussian dressings (no power
   tail) show a genuine short-range attraction pocket in *both* channels
   near $d\approx$ core size ($E^{\rm int}_1(1.0)=-0.82$), turning
   repulsive again at $d=0.5$: molecular-type binding with no long-range
   tail — not a Newton mechanism, but possibly of interest elsewhere.

## 5. The no-go, assembled

Let $E=\alpha e_1+\beta e_4$ (the general quadratic on the ansatz, §2).
Require (i) non-negative energy of every single hedgehog and of the
measured multi-center configurations, (ii) an attractive long-range tail
on at least one tailed two-body profile of §4.

- $\boldsymbol{\alpha>0}$: (i) on singles $\iff4\alpha+3\beta\ge0$ (§3).
  Then for any tailed pair, using $X>0$:
  $\alpha E^{\rm int}_1+\beta E^{\rm int}_4
  >\tfrac{E^{\rm int}_4}{3}(4\alpha+3\beta)\ge0$ — **no attraction**.
- $\boldsymbol{\alpha=0}$: (i) $\Rightarrow\beta\ge0$; tails have
  $E^{\rm int}_4>0$ — no attraction.
- $\boldsymbol{\alpha<0}$: (i) on the chain-7 witness $\Rightarrow
  \beta\ge1.80\,|\alpha|$; attraction on any clean tailed profile needs
  $\beta<t_1|\alpha|\le1.38\,|\alpha|$ — contradiction (margin holds even
  against the flagged rows' $1.55$).

Hence **no constant-coefficient quadratic Lagrangian produces an
attractive long-range tail between boost hedgehogs while keeping the
ansatz-family energies non-negative** — at leading (frozen, $O(m^4)$)
order. Under the stronger pointwise-stability requirement the same follows
from the ratio lemma alone with margins $\ge30\%$. And inside the
exact-3×3-preserving family the question never even opens: the added terms
act as identical zero on this ansatz (§2).

## 6. The cost of leaving the 3×3-preserving family

Any quadratic modification that *does* move the Newton sector must carry a
nonzero $e_4$-channel component on spatial fields, i.e. it changes the
working 3×3 physics. Measured on the relaxed, gradient-polished 3×3
electron hedgehog of report 004: $\int I_4/\int I_1=0.763$ (free bulk,
stencil-averaged; `lattice_cost.py`). The supported reading: **there is no
invariant-channel suppression** — an $I_4$-type addition with coefficient
$\beta$ perturbs working 3×3 energies at relative order $0.76\,\beta$,
i.e. at the *same* order as its effect anywhere else, so any repair that
needs an order-one $I_4$ admixture reshapes 3×3 physics at order one and
forces a retuning the program's constraints (Coulomb, three leptons) so
far forbid. No lower bound on the couplings of the beyond-quadratic
alternatives of §7 is claimed. *Reproducibility status:* this figure is an
**external result** for this report — its input field is regenerated only
by report 004's `reproduce.sh`; `lattice_cost.py` re-certifies it when
that artifact is present and reports NOT-REPRODUCED-HERE otherwise
(recorded values: `results/lattice_cost_external.json`).

## 7. What this report does not show

- Everything is leading order in the dressing amplitude on the **frozen**
  ansatz: no backreaction (report 004's delocalization shows frozen
  conclusions can move), no second-order dressing terms (these
  reintroduce non-spatial $F$ components at $O(m^6)$ in the energy), no
  time dependence (clock sector off, $A_0=0$).
- Nothing beyond constant-coefficient quadratics: field-dependent
  coefficients $c_k(M)$, higher orders in $F$, non-commutator derivative
  terms and topological couplings remain open — they are now the *only*
  quadratic-adjacent doors to the Newton sign.
- "Stability" is non-negativity of the quadratic energy on the tested
  ansatz-family configurations (singles exactly, by the virial; measured
  clusters). The supremum of $S_1/S_4$ and of tail ratios over *all*
  configurations/profiles is measured, not characterized; the no-go needs
  only the measured witnesses.
- The tail measurements cover $p\in[0.3,1]$, $\mu\in[0.1,0.4]$ with the
  strictly UV-convergent subset ($p<3/4$, i.e. $p\in\{0.3,0.5\}$)
  carrying the assembly; the author's $p=1/2$ is inside. The observed
  pinch of all ratios near $4/3$ suggests a deeper asymptotic statement
  (tail ratios $\to4/3^+$) that we did not prove.
- The $I_4/I_1$ cost figure (§6) is external to this report's
  reproduction (input field owned by report 004's reproduce path).
- The two-body ansatz superposes dressings at $O(m)$; no relative boost
  phases/orientations beyond the radial-axis choice were scanned.
- No claim about the physical profile particles actually take; the no-go
  is uniform over the tested class.

## 8. Reproduction

```bash
pip install sympy torch numpy scipy   # Python >= 3.12; GPU optional
./reproduce.sh                        # ~15 min GPU + ~15 min CPU route 2
```

Asserts: the structure theorem and collapse identities (both routes), the
virial and eigenvalue identities (symbolic + numeric), repulsive tails
with $t_1>4/3$ and $X>0$ per profile, cluster witness above the clean
ceiling, the IR-divergence slope, the pocket signs, X-sign stability
under grid variants, the three no-go branch inequalities — and route 2
(`verify_measures.py`) independently re-derives every load-bearing
measured inequality with agreement bounds. The $I_4/I_1$ cost is
certified only when report 004's regenerated artifact is present;
otherwise the run states NOT-REPRODUCED-HERE explicitly and checks only
the external record's self-consistency with the README quote.

## Equation-to-code map

| object | code |
|---|---|
| ansatz $\delta M$, $A_i$, closed-form $V$ | `check_structure.py` §§1–2 |
| $F=-v\wedge v$, spatiality | `check_structure.py` §3, `verify_symbolic.py` §1 |
| $N_i=0$, pseudoscalars $=0$ | `check_structure.py` §4, `verify_symbolic.py` §1 |
| collapse $I_2..I_6$, closed forms $e_1,e_4$ | `check_structure.py` §4, `verify_symbolic.py` §2 |
| notebook $H_s=-e_1/4$ | `check_structure.py` §6 |
| eigenvalue identities, $\rho\in[1,4]$ | `check_structure.py` §7, `verify_symbolic.py` §3 |
| virial identity | `verify_symbolic.py` §4, `measure_energies.py` §1, `verify_measures.py` §1 |
| tails, $t_1$, $X$ | `measure_energies.py` §2, §6b; route 2: `verify_measures.py` §§2–3 |
| cluster witnesses | `measure_energies.py` §3; route 2: `verify_measures.py` §4 |
| IR divergence of the notebook protocol | `measure_energies.py` §4 |
| gaussian pocket | `measure_energies.py` §5 |
| $I_4/I_1$ on the 3×3 hedgehog | `lattice_cost.py` |

## Provenance

- Canonical ansatz and task: J. Duda, Mathematica notebook "newton for
  boost hedgehogs sign" and WhatsApp message of 2026-08-21 (working repo
  `duda-particle-model`: `papers/duda answer - newton for boost hedgehogs
  sign.pdf`, commit `48afcc8`); priority and ansatz-canonicity confirmed
  by MJ 2026-08-22.
- $F$, invariants, slot conventions: report 001 (merged); $N_1,N_2,N_3$
  spatial nullspace: P239 (substrate-framework PR #148, head `1c63909`),
  verified in 001 §5; pseudoscalar set and spatial vanishing: report 005
  (merged).
- 3×3 hedgehog field and lattice conventions: report 004 (merged); the
  polished field regenerated by its `reproduce.sh` (P1a); recorded values
  in `results/lattice_cost.json` measured on the 004-line field (working
  repo `covariant_split` @ `0bcea47`).
- Development history: working repo `duda-particle-model`,
  `teleparallel/` (2026-08-22), including the plan note
  `notes/plan_teleparallel.md`.
