# Report 005 — Parity-odd (Levi-Civita) quadratic contractions of the M5 field strength

*2026-08-21 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
stage result of the extended-kinetic-term program*

## Context and result

Report 001 classified the quadratic scalar contractions of the M5 field
strength built from metric pairings only, and explicitly excluded parity-odd
invariants (001 §6). The substrate-framework campaign P239 recorded, without
further analysis, "four independent one-epsilon pseudoscalars" from 156
nonzero diagrams. This report completes the parity-odd sector.

**Result.** The one-$\varepsilon$ sector has 210 diagrams, of which 54
vanish identically and 156 fall into 13 proportionality classes spanning a
**4-dimensional** space on generic double two-forms — confirming P239's
count on their ensemble. However, on **realizable** fields
$F(A)=[A_\mu,A_\nu]_\eta$ with $A_\mu=\partial_\mu M$ — the only fields the
M5 model has — one of the four,

```math
P_{dd}=\varepsilon^{\mu\nu\rho\sigma}F_{\mu\nu\alpha\beta}
F_{\rho\sigma}{}^{\alpha\beta}
\quad\text{(P239's } J4\text{)},
```

**vanishes identically**, by a three-line cyclic-trace argument (§3), so
the realizable pseudoscalar space is **3-dimensional**. Further:

1. the 3×3-preservation no-go of P239 / report 001 §5 **extends verbatim
   to the entire $\varepsilon$ sector** (§5): every one-$\varepsilon$
   invariant vanishes identically on purely spatial fields *and* on the
   clock counterexample, and a parity argument closes the remaining room;
2. all two-$\varepsilon$ diagrams reduce to the six metric invariants, so
   the complete constant-coefficient quadratic theory on model fields is
   **6 parity-even + 3 parity-odd** invariants (§6);
3. exact structural identities: $\chi^2=16I_3-4I_1-4I_2=16N_1$ — one of
   the P239 spatial-nullspace directions is globally a perfect square —
   $I_6=\varphi^2$ and $P_{cp}=\chi\varphi$ (§6);
4. the linear invariants $\varphi$ (even) and $\chi$ (odd) are **null
   Lagrangians** (total derivatives); the three realizable pseudoscalars
   are dynamical (§7).

Every claim runs on two independent routes: float64 torch einsum with
autograd Hessians (`check_torch.py`) and exact integer arithmetic in pure
Python (`verify_exact.py`) — ranks over $\mathbb{Q}$, and Euler–Lagrange
expressions via exact bilinear-coefficient extraction instead of autograd;
the headline claims additionally have short analytic proofs given below.

## 1. Objects and conventions

All conventions are inherited from report 001: $F_{\mu\nu\alpha\beta}$ is
the $\eta$-commutator field strength (001 Eq. 1), antisymmetric in
$(\mu\nu)$ and $(\alpha\beta)$ and nothing more; slots 0–3 are
$(\mu\nu\alpha\beta)$ of the first factor and 4–7 of the second;
$\eta=\mathrm{diag}(-1,1,1,1)$. New objects: the Levi-Civita tensor with
upper indices, $\varepsilon^{0123}=+1$, and the two *linear* invariants

```math
\varphi=\eta^{\mu\alpha}\eta^{\nu\beta}F_{\mu\nu\alpha\beta}
\quad\text{(parity even)},\qquad
\chi=\varepsilon^{\mu\nu\alpha\beta}F_{\mu\nu\alpha\beta}
\quad\text{(parity odd)}.
```

A one-$\varepsilon$ quadratic diagram assigns 4 of the 8 slots of
$F\otimes F$ to $\varepsilon$ and pairs the remaining 4 with $\eta$'s:
$\binom{8}{4}\cdot 3 = 210$ diagrams. The diagonal-Lorentz-action
assumption of 001 §1 is inherited (author-gated): under it every such
diagram is a scalar under proper Lorentz transformations and flips sign
under improper ones (verified numerically in §4). Under an internal-frame
reading of the matrix indices, the mixed classes would not be scalars.

## 2. Enumeration and classes

54 diagrams vanish identically ($\eta$ contracting an antisymmetric pair);
the surviving **156** — the same count as P239's verifier — fall into **13
proportionality classes** with sizes $\{2,2,4,4,16^{\times 9}\}$
(float clustering at $10^{-10}$; confirmed by *exact integer cross-ratios*
in `verify_exact.py`). Every class is a small-rational combination
(exact to $7\cdot10^{-16}$, `results/numerical_results.json:expansions`)
of four named representatives:

| name | definition | class size |
|---|---|---|
| $P_{dd}$ | $\varepsilon^{\mu\nu\rho\sigma}F_{\mu\nu\alpha\beta}F_{\rho\sigma}{}^{\alpha\beta}$ — $\varepsilon$ on the two derivative pairs | 2 |
| $P_{mm}$ | $\varepsilon^{\alpha\beta\gamma\delta}F_{\mu\nu\alpha\beta}F^{\mu\nu}{}_{\gamma\delta}$ — $\varepsilon$ on the two matrix pairs | 2 |
| $P_{dm}$ | $\varepsilon^{\mu\nu\gamma\delta}F_{\mu\nu\alpha\beta}F^{\alpha\beta}{}_{\gamma\delta}$ — derivative pair × matrix pair | 16 |
| $P_{cp}$ | $\chi\varphi$ — $\varepsilon$ entirely on one factor, double trace on the other | 4 |

These correspond to P239's $J4$, (a combination of) $J2/J3$, and $J1$
respectively.

## 3. Independence, and the realizable-field degeneracy

On generic double two-forms (a random value per pair of 2-form indices,
the same ensemble as P239's `_random_double_two_form`) the 13 classes span
a rank-**4** space — over $\mathbb{Q}$, exactly (`verify_exact.py`). P239's
"four independent one-epsilon pseudoscalars" is correct for that ensemble.

On realizable fields it is not the end of the story. With
$B_\mu\equiv A_\mu\eta$:

```math
P_{dd}
=-\varepsilon^{\mu\nu\rho\sigma}\,
\mathrm{tr}\!\left(F_{\mu\nu}\,\eta\,F_{\rho\sigma}\,\eta\right)
=-4\,\varepsilon^{\mu\nu\rho\sigma}\,
\mathrm{tr}\!\left(B_\mu B_\nu B_\rho B_\sigma\right)=0,
```

because the cyclic shift $(\mu\nu\rho\sigma)\to(\sigma\mu\nu\rho)$ leaves
the trace invariant while being an odd permutation of $\varepsilon$: the
sum equals minus itself. (First equality: $F_{\rho\sigma}{}^{\alpha\beta}
=(\eta F_{\rho\sigma}\eta)^{\alpha\beta}$ and $F_{\mu\nu}^\top=-F_{\mu\nu}$;
second: expand the commutators, $\varepsilon$ absorbs the four terms. The
argument needs only $F_{\mu\nu}=B_\mu B_\nu-B_\nu B_\mu$ — not symmetry of
$A$.) Numerics: $\max|P_{dd}|=2\cdot10^{-12}$ at scale $|F|^2\sim3\cdot10^4$
over 400 float samples, **exactly 0** on all 20 integer samples; on generic
tensors $P_{dd}\sim10^2\neq0$.

So on model fields the pseudoscalar space is
$\{P_{mm},P_{dm},P_{cp}\}$, rank **3** — over $\mathbb{Q}$ exactly, with
$P_{dd}$'s class (size 2) the unique vanishing one. The count "four" is a
property of the generic ensemble, not of the model: this is the same
realizable-vs-generic distinction that report 001 §3 checked for the
metric invariants ("degeneracies invisible to pure combinatorics"), applied
to the odd sector, where P239's verifier ran only the generic ensemble.

A practical footnote from the first pass of this analysis: an
identically-vanishing class evaluates to roundoff noise in float, and
*normalizing that column before an SVD promotes the noise to a fake rank
contribution*. The rank routine here excludes near-zero columns first;
the exact-integer route is immune by construction.

## 4. Transformation behaviour

For a random proper Lorentz transformation ($\det\Lambda=+1$, generated in
$\mathfrak{so}(1,3)$): all ten invariants move by $<10^{-12}$ at values
$O(10^2)$. Under parity $\Lambda=\mathrm{diag}(1,-1,-1,-1)$: the six $I_k$
are unchanged and all pseudoscalar classes flip sign, both *exactly* in
float64. Parity maps realizable fields to realizable fields
($A_\mu\mapsto P_\mu{}^\nu\,PA_\nu P$, still symmetric), with $I$ even and
$P$ odd exactly — used in §5.

## 5. The 3×3 no-go extends to the ε sector

P239 proved (and report 001 §5 verified) that no constant-coefficient
combination of $I_1..I_6$ repairs the negative clock channel while exactly
preserving the 3×3 sector. Allowing parity-odd terms does not reopen it:

1. **Automatic 3×3 preservation.** Every one-$\varepsilon$ invariant
   vanishes identically on purely spatial fields: $\varepsilon$ carries
   exactly one time index, which must land on a slot of $F$, and every
   component of a purely spatial $F$ with a time index is zero. (Numerics:
   exact 0.) The extended 3×3-preserving family is therefore simply
   ```math
   L=-I_1+aN_1+bN_2+cN_3+d_1P_{mm}+d_2P_{dm}+d_3P_{cp}.
   ```
2. **The counterexample survives.** On the clock direction
   $A_0=\omega\,\mathrm{diag}(1,0,0,0)$, $A_1=E_{01}+E_{10}$, all 13
   pseudoscalar classes vanish **exactly** (integer route: exactly;
   float route: 0.0, and $<2\cdot10^{-15}$ on random proper-Lorentz images
   of the configuration), while $(I_1..I_6)=\omega^2(4,4,2,2,2,4)$ as in
   001 §5. Hence $L=-4\omega^2<0$ for every $(a,b,c,d_i)$: **the no-go
   extends verbatim**.
3. **No other configuration helps either.** For any realizable $A$, its
   parity image is realizable with identical $I_k$ and negated $P_i$ (§4).
   Summing $L$ over the pair kills the odd part, and the even part is
   already fixed by the parity-even no-go; so on at least one member of
   every parity pair the sign is wrong. No constant-coefficient quadratic
   Lagrangian, parity-odd terms included, repairs the clock channel.

Repair candidates must therefore remain field-dependent, higher order,
constrained, or 3×3-relaxing — the list of report 001 §5 stands.

## 6. Closure at two ε's, and exact identities

All 70 nonzero two-$\varepsilon$ diagrams with disjoint slot assignments
lie in $\mathrm{span}\{I_1..I_6\}$ (float residual $8\cdot10^{-16}$;
integer route: exact rational combinations, fitted on 8 samples and
verified on 12 others). Diagrams where the two $\varepsilon$'s share
contracted dummies reduce by the $\varepsilon\varepsilon=\det(\eta$-block$)$
identity to metric strings. So nothing beyond one $\varepsilon$ is new,
and the complete constant-coefficient quadratic invariant theory is
**6 even + 3 odd** on model fields (6 + 4 on generic tensors).

Exact identities (integer route: exact; float: $\le 2\cdot10^{-15}$
relative):

```math
\chi^2=16I_3-4I_1-4I_2=16N_1,
\qquad I_6=\varphi^2,
\qquad P_{cp}=\chi\varphi,
```

with $N_1=I_3-\tfrac14(I_1+I_2)$ from P239's spatial nullspace. The first
identity makes $N_1$ globally pointwise non-negative — the square of the
linear pseudoscalar — vanishing exactly on the $\chi=0$ locus, which
contains all purely spatial fields *and* the clock counterexample. The only
functional relation among the ten quadratic invariants on generic tensors
is the trivial consequence $P_{cp}^2=(16I_3-4I_1-4I_2)\,I_6$; on realizable
fields additionally $P_{dd}=0$. Jacobian ranks match exactly: 9/10 generic,
8/10 realizable.

## 7. Null Lagrangians

Treating each invariant as a Lagrangian density in $A=\partial M$, the
Euler–Lagrange expression $-\partial_\mu[\partial L/\partial A_\mu]$ is
evaluated exactly (polynomial Hessian contracted with a random symmetric
second jet):

- $\varphi$ and $\chi$ are **null** (float route: EL $<2\cdot10^{-15}$ at
  scale $10^1$; exact route: EL exactly 0). The analytic reason covers
  both at once: for any *constant* coefficient tensor $c^{\mu\nu\alpha\beta}$,
  ```math
  c^{\mu\nu\alpha\beta}F_{\mu\nu\alpha\beta}
  =\partial_\mu\!\left[2\,c^{[\mu\nu]\alpha\beta}
  M_{\alpha\gamma}\eta^{\gamma\delta}\partial_\nu M_{\delta\beta}\right],
  ```
  because the $\partial_\mu\partial_\nu M$ remainder of the product rule
  dies on the antisymmetry of $F$ (hence of $c^{[\mu\nu]}$) in the
  derivative pair. **Every linear invariant of $F$ is a null Lagrangian**;
  $\varphi$ and $\chi$ are the cases $c=\eta\otimes\eta$ and
  $c=\varepsilon$. The two linear terms one could add to the action are
  therefore boundary terms.
- $P_{mm}$, $P_{dm}$, $P_{cp}$ are **dynamical** (float route: EL
  $\sim10^3$ at scale $10^3$, same order as the control $I_1$; exact
  route: EL a nonzero integer matrix on every sample).
- $P_{dd}$ is the Pontryagin-like null density on generic tensors; on
  model fields it is simply zero (§3).

## 8. What this report does not show

- Nothing about parity-odd terms with field-dependent coefficients
  (e.g. $c(M)P_i$), parity-odd terms of higher than quadratic order in
  $F$, or the odd sector after relaxing exact 3×3 preservation. In
  particular, the parity-even quartics $P_iP_j$ (automatically
  3×3-preserving, PSD for $i=j$) are noted but not analyzed.
- No physics selection: like P239, no chirality premise is asserted. This
  is bookkeeping plus no-go robustness, not advocacy for parity violation.
- The class decomposition is sampled (40 float + 20 exact-integer random
  fields), not symbolically canonicalized; a Butler–Portugal route with
  $\varepsilon$ (plus Schouten identities, which `canon_bp` does not
  handle) is left out. Exact proportionality over $\mathbb{Q}$ on 20
  random integer samples per pair leaves no realistic room here.
- The mixed-index classes rest on the diagonal-Lorentz-action assumption
  of 001 §1 (author-gated).
- Independence statements are generic-point; fine-tuned configurations can
  make particular invariants coincide.

## 9. Reproduction

```bash
pip install sympy torch          # Python >= 3.12, CPU wheels suffice
./reproduce.sh                   # ~2 min on a laptop
```

`reproduce.sh` regenerates `results/` and asserts every structural claim:
counts 210/54/156, the 13 class sizes, ranks 4 (generic) / 3 (realizable)
with exactly one vanishing class, $P_{dd}$ and the cyclic identity, exact
parity flips, spatial/clock/orbit vanishing, the two-$\varepsilon$
reduction, the three exact identities, Jacobian ranks 9/8, and the
null-Lagrangian split. Floating-point tails are machine-dependent and not
asserted; the integer route asserts equalities exactly.

## Equation-to-code map

| object | code |
|---|---|
| $\varepsilon$, diagrams, evaluator | `check_torch.py::EPS/evaluate`, `verify_exact.py::EPS/eval_one_eps` |
| classes (float / exact cross-ratio) | `check_torch.py` §1, `verify_exact.py` §1–2 |
| ranks (SVD with dead-column guard / over $\mathbb{Q}$) | `check_torch.py::rank_of`, `verify_exact.py` §3 |
| $P_{dd}$ and cyclic identity | `check_torch.py` §3, `verify_exact.py` §3 |
| named basis + rational expansions | `check_torch.py` §4 |
| Lorentz/parity tests | `check_torch.py` §5–6 |
| spatial, clock, orbit | `check_torch.py` §6, `verify_exact.py` §5–6 |
| two-$\varepsilon$ reduction | `check_torch.py` §7, `verify_exact.py` §7 |
| identities $\chi^2,\ \varphi^2,\ \chi\varphi$ | `check_torch.py` §8, `verify_exact.py` §4 |
| Jacobian ranks | `check_torch.py` §9 |
| EL / null-Lagrangian tests (autograd / exact bilinear coefficients) | `check_torch.py` §10, `verify_exact.py` §8 |

## Provenance

- $F$ definition and conventions: report 001 (this repo, merged
  2026-08-20), which pins OpenWave
  `openwave/xperiments/m5_liquid_crystal/research/m5_theory_canonical.md`
  (commit `70c8a1bc`).
- P239 odd-sector claim and ensemble: substrate-framework PR #148, head
  `1c63909`, `proposals/P239-m5-4x4-action/evidence/quadratic-basis-note.md`
  ("four independent one-epsilon pseudoscalars") and
  `attempts/0001/enumerate_quadratic_basis.py` (`ODD_BASIS` J1–J4,
  `_random_double_two_form`).
- Clock counterexample and $N_1,N_2,N_3$: ibid., verified in report 001 §5.
- Development history: working repo `duda-particle-model`,
  `epsilon_contractions/` (2026-08-21).
