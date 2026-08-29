# Report 010 — The fundamental-reading clock grid: the u-decorated family, the Legendre filter, and a measured tick inside a two-sided γ-window

*2026-08-29 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
stage result of the extended-kinetic-term program; answers the program
question left open by reports 003/008: "the energy reading ticks, the
fundamental reading is unstable" — can a Lagrangian be written so the clock
is read directly from the canonical H?*

## Context and result

Report 008 established that the simplest quartic ticks as an *energy
functional* but its naive fundamental (Legendre) reading is unstable
(−γs² runaway); report 003's healthy C3 candidate was never lattice-tested.
This report systematically scans the Lagrangian grid

```math
L \;=\; \alpha\, I_i \;+\; \gamma\,(I_j)^2 \;-\; V(M),
\qquad \alpha\in\{-1,+1\},\ \gamma>0,
```

over a complete basis of quadratic invariants and measures which cells give
a canonical Hamiltonian with an interior clock well. Results, in order:

1. **The u-decorated family (the grid's alphabet).** All complete
   contractions of $F\otimes F$ with slots either $\eta$-paired or capped by
   the field-selected clock axis $u(M)$ (the object behind reports 002/004's
   metric $G=\eta+2uu$): 735 diagrams, 387 structural zeros, **21
   proportionality classes, rank 18 over $\mathbb{Q}$** (3 exact integer
   identities), identical on generic and realizable ensembles. The $\eta/G$
   grammar sits strictly inside: $I_1^{G\text{-deriv}}=I_1+4\,C10$,
   $I_1^{G\text{-matrix}}=I_1+4\,C16$ (exact); report 003's $B_k$ is the
   single diagram C10.
2. **The grid Legendre theorem.** Every family member splits by velocity
   degree, $I=s+m+k$, and exactly (proven symbolically on the full
   10-component $\dot M$):
   ```math
   H=\alpha(-s_i+k_i)+\gamma\left(-s_j^2+m_j^2+2s_jk_j+4m_jk_j+3k_j^2\right)+V .
   ```
   Degree-0 terms flip (the $-\gamma s_j^2$ disease of 008 §4), degree-1
   terms cancel (and are real: $I_3$ has $m\not\equiv0$), the brake arrives
   tripled. On the frozen-tangent family $H=H_0+A\omega^2+B\omega^3+
   C\omega^4$ with $A<0,C>0$ giving exactly one interior minimum, below
   $H(0)$, for **every** sign of the chirality term $B$ — always on the
   Legendre caustic (002 §6).
3. **No exact pure-kinetic invariant exists** (T1): every class has
   $s\not\equiv0$, exactly, on generic static fields — covariance obstructs
   pure kineticity (the Q0/Q1 tension of the quartic prereg, now
   family-wide). The verdict is therefore quantitative: static leaks on the
   physical hedgehog orbit are suppressed with measured orders
   $\ell\propto m_d^{2}$ (single caps) and $m_d^{4}$ (double caps).
4. **Matrix-cap orbit zeros (exact theorem).** On the rank-1 canonical orbit
   $M=vv^\top$: $F_{ij}=w_jw_i^\top-w_iw_j^\top$ carries no $v$-component,
   the exact eigen-axis is $u=-\eta v$, and $w_i^\top\eta v=0$ — so **every
   diagram with at least one matrix cap vanishes identically on the orbit**
   (12 classes; confirmed over $\mathbb{Q}$), while the derivative-capped
   C9–C11 are the only capped survivors. Same mechanism as the
   $\varepsilon$-connection inertness of the 007 appendix.
5. **The grid factorizes and then almost dies.** The statics filter forces
   $\alpha=-1$, $I_i=I_1$ — the record term, uniquely; 16 j-cells survive
   the analytic filters. On the lattice (004 stack, $\eta$-relaxed electron
   profile regenerated to the committed oracle bit-for-bit), with $\gamma$
   tuned on the frozen profile: **no cell has an interior well.** The
   relaxing field evades the brake: the drive $-k_1\omega^2$ is linear in
   density, the brake $3\gamma k_j^2$ quadratic, and dilution pays — the
   naked Mexican-hat concavity of 007/008, now in the fundamental reading.
   The leak suppression that protects the statics (filter F4) also removes
   the convex template ($2\gamma s_jk_j\approx0$) that localized 008's
   energy-reading clock. The $\gamma=0$ control reproduces the record
   disease (monotone descent, dive at $\omega=0.8$).
6. **The γ-window is two-sided, and the clock lives inside it.** Increasing
   $\gamma$ stops the evasion (ladders become monotone rising at
   $100\gamma$) before the $-\gamma s_j^2$ instability ignites at extreme
   $\gamma$ ($2.6\times10^9$ for C9: caught by the guard). In between —
   measured at ×10, ×14 and ×20 of the frozen-tuned value — **four cells
   tick in the fundamental reading**: C10 ($=B_k$, the covariantized 003
   mechanism), C19, C13, C16 — interior, protocol-level-stable wells at
   $\omega^*\approx0.1$–$0.2$ (decreasing with $\gamma$), depth
   $1.3$–$2.5\times10^{-3}$, ticking localized (PR ≈ 30–50 sites), statics
   untouched, and the ×10 depth settles under the 008 four-cycle plateau
   criterion. **The drive-flip control kills the
   well**; the cross-flip control is measured inert — consistent with the
   suppressed-leak structure: the working Hamiltonian is
   $H = E_{\rm stat}+V-k_1\omega^2+3\gamma k_j^2\omega^4$, i.e. the
   covariant completion of the quartic prereg's arm Q0, now with a lattice
   existence proof.

**Answer to the program question:** yes — the oscillation energy can be read
directly from the canonical H of a covariant Lagrangian of the family, but
only inside a bounded, two-sided γ-window, and the window is a *measured*
object, not a tuning convenience: below it the Mexican-hat concavity
delocalizes the clock, above it the drive dies, far above the F4 disease
returns.

## 1. Conventions and the family

Inherited from 001/005 (F, η, slots) and 002/004 ($u(M)$, $G$, spectral-gap
assumption). A diagram caps a subset of the 8 slots of $F\otimes F$ with
$u$ and $\eta$-pairs the rest; realizability of $(A_\mu,u)$ as independent
pointwise data is argued in the prereg (working repo). Two evaluation
routes everywhere: float64 einsum and exact Fraction arithmetic with
rational unit-timelike $u$; on the lattice the u-caps enter differentiably
through the working Lagrange Euclideanizer, $uu^\top=(G(M)-\eta)/2$
(deviation from the exact eigen-projector on the profile: max $2.0\times
10^{-4}$, mean $5.6\times10^{-5}$, recorded).

## 2. Enumeration and identities (E1)

`enumerate_u_family.py` + `verify_u_family_exact.py`: counts 735/387/21;
class sizes $4\times32,12\times16,2\times8,3\times4$; value rank 18 on both
ensembles (float SVD with dead-column guard; exact Gaussian elimination);
Jacobian rank 16 (two functional relations — the squares of the family's
two linear invariants $\varphi,\varphi_u$); the three integer identities and
the exact $\eta/G$ decompositions. The named map: $I_1..I_6=$ C3, C5, C4,
C1, C2, C0; $B_k=$ C10.

## 3. Velocity splits, statics filter, leaks, channels (E2)

`velocity_split.py` + `exact_split_checks.py` + `orbit_zeros_exact.py`:
degree ≤ 2 exact for all classes; $m\equiv0$ exactly for C3 and C16 (the
aligned-derivative-pairing proposition); statics filter: only C3 $=I_1$ is
pointwise proportional to the record static density on the 3×3 sector (001's
$N_1$ nullspace identity re-measured zero); channel validation: the 001
clock counterexample reproduces $\omega^2(4,4,2,2,2,4)$ bit-for-bit, the 005
pseudoscalars vanish there exactly, and 008's η-vs-G sign flip is
reproduced ($k(I_1)=+4$, $k(I_1^{G\text{-mat}})=-4$). $P_{dm}$ dies at the
lattice level: $K_2=0$ exactly on the electron boost channel.

## 4. The lattice campaign (E4)

004 stack verbatim (`lattice_grid_defs.py` runs `../004-lattice-clock/
lattice.py`); base profile regenerated from the committed seed (oracle
9.263660060 matched; relaxed $E_{\rm stat}=4.899587$ = 004's recorded
value); statics identity $2H^3\Sigma s_{C3}+V_4=e_{\rm static}^\eta$ exact
(0.0); record drive $D_1=0.18178$ on boost-x (0.079/0.086 on y/z),
matching 004's $K_1$ scale. Ladders: fresh-start rungs, Adam 300 + L-BFGS(80)
(two protocol levels recorded; auto-extension ×1.35 until the energy turns
up, runs away, or hits $\omega=3$), runaway guard.

Main sweep at frozen-tuned γ: 0/15 wells — creep-to-cap (C10, C13, C16,
C19: depth only −0.27 at $\omega=3$ — the brake almost holds) vs dive at
$\omega\approx1.2$–1.6 (depth −35, PR → 300+) vs early dive (C6–C8, the
largest-leak band, $\omega\approx0.7$).

γ-arm: ×100/×10⁴ monotone rising for C10/C19 (evasion stopped, drive dead);
C9 ×10⁴ ignites the $-\gamma s^2$ instability (guard-caught). Bisection:
C10 ×3 creeps, **×10 ticks, ×30 dead** — with the confirmation arm: fine
rungs sharpen the well to $\omega^*\approx0.2$
($E(0)>E(0.2)<E(0.28)$, level-stable, depth $-2.1\times10^{-3}$), and
C19/C13/C16 at their ×10 tick with nearly identical wells — consistent with
the mechanism: the well is carved by the shared record drive against a
normalized brake; the j-choice sets the brake scale and the window position.

Controls: γ=0 (record disease), drive-flip (well killed — the well is the
drive's physics), cross-flip (inert — the cross terms are measured
negligible at the suppressed leaks, as the theorem predicts).

![grid ladders](results/fig_grid_ladders.png)

![well anatomy](results/fig_well_anatomy.png)

Deep-protocol probe and the finer window map (`e5_arms.py`):

- **Depth plateau (008 criterion): settled.** Adam + four L-BFGS cycles at
  $\omega\in\{0,0.2,0.28\}$: the ×10 well depth per level is
  $(-2.38,-2.47,-2.43,-2.44,-2.45)\times10^{-3}$ — final change
  $9\times10^{-6}$, under 0.4% of the depth.
- **Window map:** ×5 still descends (min at the top rung), ×7 is the
  boundary (flat, level-unstable), **×10 ticks at $\omega^*\approx0.15$**
  (with the extra fine rungs: $E(0.1)>E(0.15)<E(0.2)$), **×14 at
  $\omega^*=0.15$** (depth $-1.6\times10^{-3}$, level-stable), **×20 at
  $\omega^*=0.1$** ($-1.3\times10^{-3}$, level-stable), ×30 monotone
  rising. $\omega^*(\gamma)$ decreases across the window, as the frozen
  formula $\omega^{*2}\propto\gamma^{-1}$ predicts qualitatively.
- **A position caveat, stated plainly:** at the deepest protocol level the
  ×10 upper bracket reverses ($E(0.28)<E(0.2)$) — near the lower window
  edge the well position drifts upward with relaxation depth, even though
  its depth plateaus. The ×14/×20 wells sit further inside the window and
  are level-stable at the standard protocol; their deep-protocol
  position-stability is untested (limitation below).

## 5. What this report does not show

- **No dynamics**: the wells sit on the Legendre caustic by theorem (E3);
  branched-Hamiltonian evolution and stability against perturbations are
  not constructed (002 §6 / 003 caveats carry over verbatim).
- **Frozen clock tangent** (004/007/008 protocol); one generator (boost-x);
  $32^3$ box, one spacing, no continuum extrapolation.
- The sweep protocol is lighter than 008's (Adam 300 + one L-BFGS cycle,
  two levels); the deep protocol ran for the ×10 well only, where the depth
  plateaus but the **position drifts up with relaxation depth** (the 0.28
  bracket reverses at the deepest level) — the deep-protocol
  position-stability of the ×14/×20 wells is untested. No persisted rung
  fields / independent route-2 energy verification yet (review-round
  candidate).
- The γ-window is mapped coarsely (×3/×5/×7/×10/×14/×20/×30/×100); no
  claim about its exact boundaries or their scaling with the leak order.
- $\omega^*$ is rung-resolved (0.1–0.2 across the window; no continuum
  minimum between the sampled rungs); depths are shallow
  ($1.3$–$2.5\times10^{-3}$ on $E_{\rm stat}\approx4.9$).
- The u-caps on the lattice are the working Lagrange realization of
  $uu^\top$ (exact on-spectrum; $2\times10^{-4}$ off); the exact-eigen
  variant is not run.
- Scope pins of the family: F stays the η-commutator; no ε-decorated
  diagrams (the three 005 pseudoscalars ride along and die); no linear-in-F
  terms.

## 6. Author-gated physics choices

- The physical reading (fundamental vs energy) stays author-gated; this
  report *adds* the measured fact that the fundamental reading has interior
  wells in a bounded γ-window.
- The window position ($\gamma$ anchoring) and the drive channel (boost-x)
  tie into scale anchoring ($\omega^*=mc^2/\hbar$), unresolved as before.
- Whether the article Lagrangian should adopt the C10 cell (the covariant
  $B_k$ quartic — 003's mechanism, now lattice-proven) is the author's
  call; C13/C16/C19 are measured equivalents at this protocol depth.

## Reproduction

```bash
pip install sympy torch numpy scipy matplotlib   # Python >= 3.12
bash reproduce.sh          # CPU: E0-E3 exact suites (~10 min)
                           # GPU (optional): lattice legs, hours
```

`reproduce.sh` asserts the structural claims (counts, ranks, identities,
theorem checks, orbit zeros, split guards) on CPU; with a CUDA device and
`M5_RUN_LATTICE=1` it regenerates the base profile against the 004 oracle
and reruns the ladder campaign (sentinel-flagged; hours). Committed JSONs
under `results/` carry every number quoted above; figures regenerate from
the committed JSONs via `make_figures.py`.

## Equation-to-artifact map

| object | artifact |
|---|---|
| grid Legendre theorem (full $\dot M$) | `legendre_theorem_check.py` → `results/legendre_theorem_check.json` |
| 735 diagrams, classes, ranks, named map | `enumerate_u_family.py`, `verify_u_family_exact.py` → `results/u_family_float.json`, `results/u_family_exact.json` |
| velocity splits, statics filter, leaks, channels | `velocity_split.py` → `results/velocity_split.json`; `exact_split_checks.py` |
| reduced quartic well + caustic (E3) | `e3_reduced_legendre.py` → `results/e3_reduced_legendre.json` |
| matrix-cap orbit zeros (exact) | `orbit_zeros_exact.py` → `results/orbit_zeros_exact.json` |
| lattice port + validations | `lattice_grid_defs.py` (gate: 004 oracle; statics identity; U-vs-eigen) |
| per-cell integrals, γ choice, ranking | `pre_e4.py` → `results/pre_e4.json` |
| ladders, controls, extensions | `e4_ladders.py` → `results/e4_cells.json` |
| γ-robustness + bisection | `e4_gamma_arm.py` → `results/e4_gamma_arm.json`, `results/e4_confirm.json` |
| deep-well plateau + window map | `e5_arms.py` → `results/e5_arms.json` |
| base profile (regenerated, committed) | `results/M_eta_base.npz` |
| figures | `make_figures.py` → `results/fig_*.png` |

## Provenance

- Conventions and prior results: reports 001–008 (this repo); equations of
  record and the Q0/Q1 quartic prereg: working repo `duda-particle-model`
  (`notes/prereg_quartic.md`, `notes/equations_of_record.md`).
- Plan and prereg for this campaign (with the deviations register):
  working repo `duda-particle-model`, `notes/plan_hamiltonian_grid.md`,
  `notes/prereg_hamiltonian_grid.md`, `hamiltonian_grid/NOTES.md`
  (commits `eddc4dd`..`c092de1`, 2026-08-29).
- The C2/C3 Legendre dichotomy: report 003; the −γs² runaway measurement:
  report 008 §4; the concavity/dilution mechanism: reports 007/008.
