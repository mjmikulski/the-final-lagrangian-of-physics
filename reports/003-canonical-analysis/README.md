# Report 003 — Canonical analysis of the boost-condensate clock

*2026-08-20 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
executes steps 1–3 of the consistency program adopted in
[report 002](../002-covariant-split-and-clock/) §9*

## Results

Report 002 left open which tensorial completion of the condensate is the
fundamental object and whether its canonical Hamiltonian is healthy. Both
questions are now answered, symbolically (sympy) and numerically (torch
autograd, exact agreement):

| candidate | quartic in $L$ | canonical $H$ | statics | verdict |
|---|---|---|---|---|
| C2 "naive Lorentzian", $B_L=B_k-B_s$ | $-aB_L+bB_L^2$ | $3bB_k^2-(a{+}2bB_s)B_k-aB_s-\boldsymbol{bB_s^2}$ | **unbounded below** | **dead** |
| **C3 "u-selected"** | $-aB_k+bB_k^2$ (replaces the quadratic boost-kinetic channel) | $-aB_k+3bB_k^2$ | untouched ($B_k\equiv0$ on spatial fields, measured 0.0) | **healthy** |
| C1 energy-functional | ($H$ posited directly) | $-aB_H+bB_H^2$ | bounded, but a *static* hat at $B_s=a/2b$ (flag) | reference |

Here $B_k$/$B_s$ are the kinetic/static parts of the covariant boost
channel ($B_k$ from $F_{0i}$, quadratic in $\dot M$; $B_s$ from the
boost-like matrix components of $F_{ij}$); the split of the derivative
pair uses the same field-selected axis $u(M)$ as report 002, so C3 is
covariant (gate $6.6\cdot10^{-15}$, after fixing the derivative-pair
metric variance $G_d=\eta G\eta$ — off-vacuum covariance was broken at
$1.2\cdot10^{-1}$ before this correction; vacuum numbers unchanged).

**Why C2 dies.** The Legendre transform flips the sign of the static
piece inside the square: $\Delta H=-aB_s-bB_s^2$ (identity verified to
0.0; $H\to-3.7\cdot10^{10}$ on a scale-16 sample). This is the same
disease as the eigenvalue-lift collapse of report 001's follow-up, now
proven at the canonical level rather than observed in relaxation.

**Why C3 is the candidate of record.** The quartic touches only the
kinetic boost norm selected by $u(M)$: statics and the 3×3/Coulomb sector
are exactly untouched, the canonical energy is
$H=-aB_k+3bB_k^2\ge-a^2/12b$, and the finite clock survives with
$\omega^{*2}=a/(6bk_2)$ (measured 0.913, autograd $H$ = formula to
$10^{-12}$ on all rungs).

## Hessians and dynamics

- **Vacuum kinetic Hessian: fully degenerate** (10 zero modes) — the
  quadratic form $\sum_i\langle[\dot M,A_i],[\dot M,A_i]\rangle_G$
  vanishes when $A_i=0$: constrained/Dirac structure, as expected.
- **Quadratic all-$G$ sector: positive semidefinite** (Gram argument;
  measured min eigenvalue $-0.0$).
- **Full C3 Lagrangian: indefinite velocity Hessian around $\omega^*$**
  (e.g. 2 negative eigenvalues at $\omega^*$, min $-5.3$; zones present
  at $0.5\,\omega^*$ and $1.5\,\omega^*$ and on spatial backgrounds):
  the Legendre map is non-invertible near the clock state — the
  branched-Hamiltonian treatment (Shapere–Wilczek) is **mandatory** for
  dynamics, exactly as the caustic identity of report 002 §6 predicted.
- **The energy itself is locally stable at the clock**: the Hessian of
  $H$ at the $\omega^*$ state is positive semidefinite
  (eigenvalues $0,\ldots,+8,+8,+32$; flat directions = background
  moduli) and $d^2H/d\omega^2=4ak_2=16>0$.
- **Reduced equations of motion**: the phase is cyclic on the rotating
  family, so every constant $\omega$ solves the Euler–Lagrange equations
  there; $\omega^*$ is selected by energy minimization plus stability,
  not by the EOM alone — the standard situation for isorotating states.

## What this report does not show

- No spatial-profile dynamics: the branched-Hamiltonian evolution across
  the indefinite zones is not constructed; stability is established for
  the energy at the state, not for full field dynamics.
- No lattice soliton yet (next per the program), no Newton-sign
  measurement, no anchoring of $a,b$ ($\omega^*=mc^2/\hbar$ remains the
  central open physics question).
- C1's static-condensate flag ($B_s=a/2b$ favored) is recorded, not
  explored.

## Reproduction

```bash
pip install sympy torch
./reproduce.sh     # < 1 min CPU
```

Asserts: the symbolic $H$ formulas for C2/C3 (including the fatal
$-bB_s^2$), $\omega^*$ vs analytic, $B_k=0$ on spatial fields, C3
covariance, $d^2H/d\omega^2>0$ and PSD Hessian of $H$ at $\omega^*$,
PSD quadratic sector, and the reality of the indefinite zone at
$\omega^*$.

## Provenance

Constructions and conventions: [report 002](../002-covariant-split-and-clock/)
(rev. 3). The C2/C3 question was posed by the external review of report
002 (its "single most important issue"); this report resolves it.
Branched Hamiltonians: Shapere & Wilczek, PRL 109, 160402 (2012).
