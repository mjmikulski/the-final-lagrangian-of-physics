# Report 014 — Linear-in-F terms with spectral coefficients: twelve pointwise generators, all dynamical, statics-only in the fundamental reading, inert on the canonical orbit

*2026-09-04 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
closes the one order the program had not scanned: terms **linear** in the
field strength F (quadratic in ∂M — the order of an ordinary sigma-model
kinetic term), which the model of record does not contain.*

## Notation (self-contained)

M is the symmetric 4×4 field, A_μ = ∂_μM, η = diag(−1,1,1,1),
F_{μν} = A_μηA_ν − A_νηA_μ is the field strength (a 4×4 matrix in the
"matrix" indices αβ; F_{μναβ} is antisymmetric in μν and in αβ, reports
001/005). φ = η^{μα}η^{νβ}F_{μναβ} and χ = ε^{μναβ}F_{μναβ} are the two
linear invariants with constant coefficients; report 005 §7 proved both are
null Lagrangians (total derivatives). The eigenframe of ηM is
{e_a}, a = 0..3, with e_a·η·e_b = s_aδ_ab, s = (−1,1,1,1); the spectral
projectors as (2,0) tensors are P_a^{μν} = s_a e_a^μ e_a^ν, ΣP_a = η. In the
vacuum the spectrum of ηM is (8, 1, 0.3, 0); e_0 = e_t is the timelike axis
(the clock axis u of reports 002/010), e_1 the eigenvalue-1 axis, e_2, e_3
the two small axes (0.3 and 0). "Class C_k" of report 010 and "F1–F5"
filters are not used here. A "linear class" below is a complete contraction
of ONE F with metrics X ∈ {η, P_0, P_1, P_2, P_3} on its slot pairs, or with
ε and slot insertions.

## Context and result

Reports 001/005 enumerated the quadratic contractions F⊗F; report 010 the
u-decorated ones. Nothing linear in F was ever scanned beyond the constant
φ, χ — which are null. Linear terms c(M)·F are of order (∂M)², one order
below the record kinetic term −½I₁ ~ (∂M)⁴, so they are the minimal
extension of the "everything from F" grammar and the only one that can
change the far field. Results:

1. **Twelve pointwise generators, exactly.** With projector decorations
   the 675 diagrams (50 even, 625 with ε) reduce to 38 proportionality
   classes and rank **12 over ℚ: 6 even + 6 odd** — the rank of the family
   over constant scalars on the simple-spectrum stratum. Upper bound by
   tensor algebra (`exact_classification.py`): expanding every metric slot
   with η = Σ_a P_a, each even diagram is an integer combination of the six
   F_{ab} ≡ P_a^{μα}P_b^{νβ}F_{μναβ}, a < b (F_{aa} = 0 by antisymmetry,
   F_{ab} = F_{ba}), and each odd diagram, because ε of four frame vectors
   is ± det E unless two coincide, is an integer combination of the six
   frame components of F with all four indices distinct; these expansion
   identities are verified exactly for all 675 diagrams at six random
   rational (frame, field) points, and the classes and ranks are then read
   off the integer coefficient vectors. Lower bound: exact rank 12 of the
   evaluation matrix at sixteen rational points (`exact_linear.py`). Hence
   exactly 12. φ = Σ_{a<b} 2F_{ab}. **Scope (review round 1):** with
   general smooth spectral scalar functions of M as coefficients the family
   is a rank-12 *module* over those scalars, not a twelve-dimensional space;
   "complete" below means: every linear-in-F term with coefficients built
   from η, ε and the spectral data of M is a spectral-scalar combination of
   these twelve generators. Gap-weighted combinations are admitted and one
   is used in result 5.
2. **Every decorated class is dynamical.** An autograd Euler–Lagrange test
   with the projectors' full M-dependence (Lagrange polynomials in ηM with
   the eigenvalues as differentiable nodes) gives |EL|/scale = 0.15–0.85 for
   all 36 decorated classes, while the two constant-coefficient controls φ
   and χ come out null to 10⁻¹⁷ and I₁ dynamical — 005's theorem is
   reproduced by the instrument and does not extend to field-dependent
   coefficients.
3. **Structure theorems.** (a) Every linear class has velocity degree ≤ 1,
   so in the fundamental reading (the Legendre theorem of report 010) it
   contributes only its static part, with flipped sign, and a gyroscopic
   term: **linear terms cannot supply a clock drive or brake**; their arena
   is the static sector. (b) Because F is antisymmetric in (μν), every
   nonzero linear class pairs a derivative index with a matrix index — the
   whole sector exists only under the diagonal-Lorentz-action assumption of
   001 §1 (author-gated). (c) On the rank-1 canonical orbit of report 006
   (M = g vvᵀ) the even sector is inert: every class carrying a P_t cap
   vanishes identically (the matrix-cap theorem of 010 — the timelike axis
   is η-orthogonal to the matrix structure of F on the orbit) and the
   remainder equals ±φ, a null term; exact over ℚ. The odd sector is
   nonzero there but parity-odd, so its integral vanishes on
   parity-symmetric configurations. **The Newton-sign question of 006 is
   therefore not reopened by linear terms on the canonical ansatz.** On the
   rank-rich dressing orbit all 38 classes are nonzero.
4. **Statics on the 3×3 sector.** With the vacuum frame only the three
   spatial classes F_{12}, F_{13}, F_{23} survive (the classes with a P_t
   cap vanish, all odd classes vanish); they are dynamical inside the
   static sector, and their static quadratic forms on the 18 spatial-block
   gradient components are traceless and indefinite (signatures +4/−4) — a
   linear term can only be a bounded correction to the F² statics. On the
   η-relaxed electron profile (M_{0i} ≡ 0, so u = e_0 exactly) the
   u-capped classes vanish identically; only the spatial classes act.
5. **The coefficients are the obstruction.** The spatial-axis projectors
   are exactly the objects that are ill-defined where the electron's small
   eigenvalues exchange (gap 0.003–0.004 on the profile): the vacuum-pinned
   Lagrange route of report 004 (exact for P_t to 2·10⁻⁴) deviates from the
   true P_1, P_2, P_3 by up to 0.49, 1.0, 1.0 in the cores; local-node
   projectors with the two small axes merged, Q = P_2 + P_3 (smooth through
   the 2–3 crossing), are exact to 10⁻¹³. The unweighted splitting
   P_2 − P_3 is singular at the crossing, but its gap-weighted version is
   not: with X = ηM (mixed) and λ̄ = ½tr(XQ), the tensor T = (X − λ̄)Q is a
   smooth function of M and equals ½(λ_2−λ_3)(P_2−P_3) in the local
   eigenbasis (pointed out in review round 1). The static sector that is
   smooth through the 2–3 crossing is therefore three-dimensional:
   {F_{1Q} = F_{12}+F_{13}, F_{QQ} = 2F_{23}, F_{1T} = ½(λ_2−λ_3)(F_{12}−F_{13})},
   and all three are run on the lattice (route B). All three remain
   non-smooth where the eigenvalue-1 axis meets the small pair (a 1–2
   collision); the electron's smallest 1–2 gap is 3·10⁻³, and §4 shows
   the relaxation finding those points.
6. **Lattice λ-scan** (§4). The uniform vacuum is a saddle of the linear
   term at **cubic** order: along a generic compact frame twist of
   amplitude t the linear integral is c₃t³ + O(t⁴) with |c₃| = 0.12–1.3
   (three random twists, all classes; the O(t²) part is the null total
   derivative, and an inversion-symmetric twist used in the first version
   of this test hid the cubic term), while the η energy is c₄t⁴ with
   c₄ = 53–71. For a coupling λ the cubic–quartic model has its minimum at
   t* = −3λc₃/(4c₄) = 10⁻⁶–10⁻⁴ with energy gain −λ⁴·27c₃⁴/(256c₄³) =
   10⁻²³–10⁻¹⁵ — nonzero, sign-selected, and far below the resolution of a
   relaxation (a noisy vacuum relaxes back to E ≈ 2·10⁻⁶ for λ = 0 and ±λ).
   The linear term changes the vacuum in principle; at the couplings of the
   scan it does not change it measurably. For the projector classes F_{ab} it acts as a
   sign-weighted reweighting of the existing static density: energy shifts
   follow the frozen values λ·∫dens (route A: within 1–8% at 5% weight and
   5–26% at 20%; route B F_{1Q}, F_{QQ}: within 1% and 3%), the far-field
   exponent moves with the weight (0.007…0.99 across ±20%,
   class-dependent), and the core's eigenvalue-exchange structure survives
   (2–3 gaps 0.001–0.008 versus 0.0036). Route A: 12 of 13 branches pass
   all three gates at the end of the protocol (continuation ≤ 0.05% of
   the effect, |∇E|∞ ≤ 0.1, spatial-block restarts within 0.3%); the P1P3
   −20% branch ends at |∇E|∞ = 0.105 and passes after one further
   100-iteration cycle (0.042, energy −7·10⁻⁵, tail unchanged; §4). The gap-weighted class F_{1T} is **not** a small
   reweighting: its shifts deviate from the frozen values by 18–170% and
   change sign at +20%; at −20% weight it lowers the energy by 2.65 and
   doubles the small splitting. With the exact spectral coefficients
   (route B) nine of twelve λ-runs end with |∇E|∞ = 0.1–547 very near 1–2
   collisions (gaps 10⁻⁹–10⁻⁷ at the maximal-gradient sites, where P_1 and
   Q are non-smooth) and the optimizer makes no progress; only the three
   smallest-coupling runs relax. The gap-weighted class F_{1T} is the one
   term that acts on the core, along a trajectory that is not stationary
   (−20%: energy −2.65, the small splitting doubled, |∇E|∞ = 0.13, restart
   still descending).

## 1. Enumeration (`enumerate_linear.py`, `exact_linear.py`)

Realizable ensemble: random symmetric A_μ and an independent random
η-orthonormal frame (pointwise independence of the frame and the first
derivatives, as for u in 010). Float classes at 10⁻¹⁰ relative, ranks by SVD
with dead-column guard; exact route with rational Lorentz frames (rational
boosts × Pythagorean rotations) and exact Gaussian elimination: ranks
12/6/6 agree. Velocity-degree guard: I(λA_0) affine in λ to 10⁻¹⁴.

## 2. The Euler–Lagrange test (`null_test_linear.py`)

For L(M, A) with A_μ = ∂_μM, EL = ∂L/∂M − Σ_μ d_μ(∂L/∂A_μ), with
d_μ(∂L/∂A_μ) evaluated as a Jacobian-vector product along the tangent
(A_μ, ∂_μ∂_νM) on random second jets (torch.func). Normalization by the
sum of the per-μ magnitudes (a total derivative has cancelling per-μ
pieces, so a ratio of totals would be 0/0). Controls: a hand-written total
derivative (0.0), a hand-written dynamical term, φ, χ (null), I₁ (dynamical).

## 3. Structure (`orbit_linear_exact.py`, `orbit1_linear_exact.py`, `static_kernel_signs.py`)

Rank-rich orbit M = o·diag(−8,1,3/10,0)·oᵀ with exact rational o and exact
eigenframes (rational nullspaces): all 38 classes nonzero. Rank-1 orbit
M = g·vvᵀ with the only defined projectors {P_t, η − P_t}: the even sector
reduces to ±φ; the P_t-capped classes vanish exactly (10 random orbit
points). Static kernels: 18×18 quadratic forms of the 3×3-alive classes at
the vacuum frame, traceless, signatures (+8/−7 for the η-η class, +5/−5,
+4/−4 for the projector classes).

## 4. Lattice runs (`lattice_linear.py` route A, `lattice_linear_local.py` route B)

004 stack through report 010's `lattice_grid_defs.py`; base profile the
committed η-relaxed electron (010, `M_eta_base.npz`). E_λ = e_static(η) +
λ·H³Σ dens, dens the static density of a linear class with (A) vacuum-pinned
Lagrange projectors or (B) local-node projectors and the merged Q. In route B the
eigenvalue nodes are inside the autograd graph (torch.linalg.eigvals on the
per-site ηM), so the optimizer varies the stated functional; the full
derivative is checked against a directional finite difference along a
random (eigenvalue-changing) direction before any run. λ set so that
|λ·∫dens| is 5% and 20% of E_stat on the base profile, both signs, relaxed
with Adam 2000 + L-BFGS(100). Gates (review round 1): the final
true-objective gradient norm on free sites, a +100-iteration L-BFGS
continuation (energy, linear integral and tail changes recorded), and for
the 20% runs a restart from a perturbed start (σ = 10⁻² on free sites,
**spatial block only** — the η statics is unbounded below outside the
3×3 block, equations of record §2, so an all-component kick tests that
known instability, not the linear term; measured: every such kick ran
away to E ≈ −4800 with or without the linear term). A run is called
relaxed only if the continuation moves the energy by less than 1% of its
distance from the baseline, the final free-site |∇E|∞ is below 0.1 (the
baseline's is 0.016–0.03), and the spatial-block restart lands on the same
energy within the continuation tolerance. Diagnostics: total and η-static energy,
the linear integral, the far-field exponent of the η density on shells
r ∈ [8,16] (004's `tail_fit`), spectral gaps, the participation ratio of
the linear density, runaway guard.

Lattice validations (route B, committed): P_t vs the exact eigen-projector
2.6·10⁻¹⁸, P_1 and Q 1.8·10⁻¹³ (route A: P_1 0.49, P_2/P_3 1.0 in cores);
the discrete gradient of the lattice sum of φ is 7.6·10⁻¹⁷ of a dynamical
class's — the symmetric-stencil lattice respects the null Lagrangian
exactly; F_{tQ} integrates to 10⁻¹⁸ on the profile (u = e_0 exactly there).
Base integrals: F_{1Q} 134.9, F_{QQ} 266.1 (E_stat 4.90) — the linear
densities are two orders larger than the F² density, as their lower
gradient order implies.

**Vacuum stability (`twist_scan_generic.py`, `vacuum_condensation.py`,
`twist_scan.py`).** Along a frame-twist direction a of amplitude t the
pointwise density is λq(a)t² + p(a)t⁴ with q indefinite on the rotational
tangents of the vacuum manifold (signature (+2, −2, 0×5) for each spatial
class); but the O(t²) part of the density has constant (vacuum) spectral
coefficients and is therefore the null total derivative of report 005 —
it integrates to zero. The first version of this test used a twist linear
in x times a radial window, which is inversion-odd; for it the linear
integral scales as t⁴ (`twist_scan.py`, ratio 39 for t = 0.002 → 0.005),
and that was over-read as a general statement. Review round 2 pointed out
that the generic next term is cubic: δC = O(t) times F = O(t²). Generic
compact twists (three random smoothed rotation fields, zero on the shell,
both signs of t; `twist_scan_generic.py`) confirm it: the odd part of the
linear integral is c₃t³ with c₃ = +0.2449 (F_{1Q}, seed 456), −0.140,
−0.651 (other seeds) and |c₃| = 0.0017–1.3 across all classes, constant to
five digits over t = 5·10⁻⁴…4·10⁻³; the η energy is c₄t⁴ with
c₄ = 53.31–71.29 and no odd part. So for every λ ≠ 0 the uniform vacuum is
a saddle at cubic order: the cubic–quartic model gives a minimum at
t* = −3λc₃/(4c₄) (10⁻⁶–10⁻⁴ at the 5% couplings) with energy
−λ⁴·27c₃⁴/(256c₄³) = 10⁻²³–10⁻¹⁵. A noisy vacuum (σ = 10⁻² spatial-block
noise) relaxes back to E ≈ 2·10⁻⁶ for λ = 0 and ±λ
(`vacuum_condensation.py`): the condensate exists but is unresolvable at
these couplings. The linear term renormalizes the quartic twist stiffness
at relative order λ·(∫ℓ/E_stat) ~ 10⁻⁴ here.

**Where the linear integral lives (`radial_profile.py`).** On the base
profile about half of every linear integral — and half of the η static
energy itself — sits at r > 20, next to the pinned shell (a property of the
32³ box of report 004: the tail-fit shells r ∈ [8,16] carry a density that
grows outward, slope +0.44); the linear densities track the static density
within a factor 2 across shells. The energy shifts of the λ-runs are
therefore sign-weighted reweightings of the existing static density, and
the relaxed field responds by moving energy between shells.

**Route A (vacuum-pinned coefficients), 13 runs.** Baseline (λ = 0):
E = 4.841063 after the protocol (the base profile relaxes further from
4.8996), free-site |∇E|∞ = 0.016, tail exponent 0.441, small-pair gap
0.0036. Twelve of the thirteen runs pass the gates at the end of the protocol:
continuation changes 2–9·10⁻⁵ (≤ 0.05% of the effect), |∇E|∞ 0.007–0.05,
and the spatial-block restarts re-land within −0.2…−2.8·10⁻³ of the main
energies (≤ 0.3% of the effect) with tails reproduced to ±0.004. The
P1P3 −20% run ends at |∇E|∞ = 0.105, outside the gate; one further
100-iteration L-BFGS cycle (`continue_run.py`,
`results/continuation_A_P1P3_f0.2_s-1.json`) brings it to 0.042 with the
energy lower by 7·10⁻⁵ and the tail exponent unchanged (0.317), so its
quoted diagnostics stand.
The energy shifts follow the frozen value λ·∫dens: at 5% weight within
1–8% (P1P2 ±0.242…0.249 vs ±0.245; P2P3 −5%: −0.264), at 20% within 5–26%
with a sign-selected backreaction — for λ < 0 the relaxed field grows the linear integral
(P2P3: 125 → 189, ΔE = −1.23 vs −0.98 frozen), for λ > 0 it shrinks it
(125 → 91, ΔE = +0.85 vs +0.98). The far-field exponent moves with the
weight, class-dependently: P1P2 0.675 → 0.185 and P2P3 0.007 → 0.988 across
−20%…+20%, P1P3 non-monotonic (0.32…0.46); the eigenvalue-exchange gap
stays within 0.0012–0.0068 (baseline 0.0036) — the core structure survives.
The participation ratio of the linear density is 15–20·10³ sites: the term
is spread over the whole texture, not localized.

**Route B (differentiable spectral coefficients), 14 runs.** For F_{1Q}
and F_{QQ} the energy shifts track the frozen values (5%: within 1%; 20%:
within 3%); F_{1T} does not (18% and 36% off at ±5%, sign reversed at
+20%, 170% off at −20% — see below), and the control class F_{tQ} does
nothing (its integral stays 10⁻¹⁸, energy and tail unchanged). The difference from route A is the **gradient gate**: only
the smallest couplings relax — F_{1Q} +5%, F_{QQ} −5%, F_{1T} −5% end with
|∇E|∞ = 0.02–0.03 like the baseline — while the other nine runs stall with
|∇E|∞ = 0.11–547 and an L-BFGS step that makes no progress (continuation
change 0 to 10⁻⁷). What stalls them (`stall_diagnostics.py`, review rounds 1–2, computed
from the committed float32 copies of the endpoint states in
`results/fields/`): at the site of maximal |∂E/∂M| of every stalled
endpoint the **1–2 gap** (the eigenvalue-1 axis against the small-pair
cluster) is 6.6·10⁻⁹–2.9·10⁻⁷, while the 2–3 gap there is 0.7–1.1; in the
relaxed endpoints the smallest 1–2 gap on the free sites is
1.6·10⁻⁴–7·10⁻³ (baseline 3.3·10⁻³). The 2–3 crossing is smooth for all
three route-B classes (Q and T are Riesz projections of the cluster,
analytic while the cluster is separated from λ_t and λ_1) and is not the
cause; the only non-smooth points of P_1, Q and T are 1–2 collisions, and
the relaxation drives the field onto their neighbourhood. The gradient
there is ill-conditioned: recomputed from the float32 copies (a relative
perturbation of 10⁻⁷) the stalled endpoints' |∇E|∞ read 0.75–10⁴ against
0.11–547 at the end of the float64 runs, while the relaxed endpoints'
values are unchanged (0.019–0.033). A directional finite-difference check
along the gradient direction agrees with autograd to 10⁻⁹ at every relaxed
endpoint and disagrees by factors 2–2000 at every stalled one; with steps
of 10⁻⁶–10⁻⁵ and gaps of 10⁻⁹ the secant crosses the eigenvalue-ordering
surface, so this does not show that the derivative at the endpoint is wrong
(review round 2) — a within-chamber test would need steps below the gap,
beyond double precision at energies of order 5. What the data support is
the weaker statement: the stalled runs end very near 1–2 collisions, where
P_1 and Q are non-smooth, the optimizer makes no progress there, and
whether the within-stratum derivative diverges or only the evaluation of
1/(λ₂−λ₁) loses precision is not decided. Their energies and tails are
protocol endpoints near that surface, not stationary points. The electron
texture starts with a 1–2 gap of 3.3·10⁻³, i.e. that surface is close to
the base profile from the outset.
The gap-weighted class behaves differently from the
other two: at −20% weight the optimization trajectory *grows* the 2–3
splitting (smallest gap 0.0036 → 0.0081, linear integral 6.1 → 31.4) and
lowers the energy by 2.65 (frozen value −0.98), the tail exponent moving
to 0.61 — but this endpoint has |∇E|∞ = 0.128, fails the gate, and its
spatial-block restart descends by a further 0.032, so it is a
non-stationary trend along the trajectory, not a relaxed core; at +20% the
run ends near a collision with |∇E|∞ = 547, tail 1.74 and the integral
changed in sign. T rewards or penalizes biaxial splitting directly — the
one linear term that acts on the core rather than reweighting the far
field, and the one whose endpoints are least converged.
Spatial-block restarts for route B (20% runs): F_{1Q} and F_{QQ} re-land
within −3…−8·10⁻⁴ of the main energies (≤ 0.1% of the effect, tails within
0.004) even though their main runs are stalled — the walls are
reproducible endpoints; F_{1T} does not re-land (−1.3·10⁻² at +20%, i.e.
10% of its effect, and −3.2·10⁻² at −20%, 1.2%): its branches keep
descending, consistent with a core that is still reorganizing.

**Answer to the plan's question.** The linear sector is real but does not
open a new door: it cannot touch the clock (3a), it is inert on the
canonical ansatz so the Newton-sign no-go of 006 stands there (3c), it
destabilizes the vacuum only at cubic order with an unmeasurably small
condensate (6), and on the electron it reweights what F² already builds. Its only structural novelty is negative — the spectral coefficients
that make it dynamical are non-smooth where the eigenvalue-1 axis meets
the small pair, the electron texture already sits 3·10⁻³ away from such
points, and the energy minimization finds them. The one term with a
distinct action, the gap-weighted F_{1T}, acts on the biaxial splitting of
the core along a trajectory that does not reach a stationary point within
the protocol; whether a term that rewards eigenvalue splitting is wanted is
an author-gated choice.

## 5. What this report does not show

- No dynamics and nothing about the clock: by result 3(a) the sector is
  static-only in the fundamental reading; the gyroscopic part is recorded,
  not studied.
- The Newton-sign statement (3c) is for the canonical rank-1 ansatz; on the
  rank-rich electron the linear terms act through the lattice runs only,
  and no two-body lattice measurement is made here.
- Route B's stalled branches are not stationary points; their energies and
  tails are quoted as protocol endpoints near 1–2 collisions, not as
  relaxed minima; whether the derivative is singular there or only its
  evaluation is not decided.
- The vacuum condensate is inferred from the measured cubic and quartic
  coefficients through a cubic–quartic model; it is not resolved by a
  relaxation (its amplitude is 10⁻⁶–10⁻⁴). No regularization of the spectral coefficients (a gap
  floor, or a Riesz projection of the {1,2,3} cluster) is tried; whether
  the walls survive such a regularization is open.
- The far-field exponent is a shell fit on r ∈ [8,16] of a 32³ box whose
  static density grows outward there (baseline slope +0.44); it measures
  redistribution between shells, not an asymptotic power law.
- Coefficients are restricted to spectral data of M (η, ε, projectors and
  smooth spectral scalars); the λ-scan runs constant multiples of the
  generators plus the gap-weighted T; no scan over general spectral-scalar
  weights is made.
- Single lattice, single spacing; the far-field exponent is a shell fit on
  the 32³ box.
- Author-gated: the diagonal Lorentz action (without it the sector does not
  exist); whether terms outside the F-grammar (symmetric-in-μν quadratic
  terms such as tr(A_μηA_νη)η^{μν}) are admissible is not this report's
  question.

## Reproduction

```bash
pip install sympy torch numpy scipy matplotlib
bash reproduce.sh            # CPU suites (~15 min); M5_RUN_LATTICE=1 for the GPU legs
```

## Equation-to-artifact map

| object | artifact |
|---|---|
| 675 diagrams, classes, ranks, 3×3 filter, degree guard | `enumerate_linear.py` → `results/linear_float.json`; `exact_linear.py` → `results/linear_exact.json` |
| Euler–Lagrange (null) test with M-dependent projectors | `null_test_linear.py` → `results/null_test.json` |
| rank-rich orbit values; rank-1 orbit theorem | `orbit_linear_exact.py`, `orbit1_linear_exact.py` → `results/orbit_linear.json`, `results/orbit1_linear.json` |
| static kernel signatures | `static_kernel_signs.py` → `results/static_kernel_signs.json` |
| lattice λ-scan, routes A / B (differentiable spectral coefficients, gates) | `lattice_linear_v2.py A` / `B` → `results/lattice_linear_A.json`, `results/lattice_linear_B.json` |
| exact classification (upper bound) | `exact_classification.py` → `results/exact_classification.json` |
| spatial-block restarts; endpoint gaps and derivative checks; continuation; vacuum stability; twist scans; radial profile | `restart_check.py`, `stall_diagnostics.py`, `continue_run.py`, `vacuum_condensation.py`, `twist_scan_generic.py`, `twist_scan.py`, `radial_profile.py` → `results/restart_check_*.json`, `results/stall_diagnostics_B.json`, `results/continuation_*.json`, `results/vacuum_condensation_*.json`, `results/twist_generic_*.json`, `results/twist_scan_B.json`, `results/radial_profile_*.json` |
| endpoint states used by the diagnostics (float32 copies; the diagnostics JSON is computed from these copies) | `results/fields/*.npz` (route B endpoints, the P1P3 −20% endpoint and its continuation) |
| artifact-only assertions | `verify_artifacts.py` |
| figures | `make_figures.py` → `results/fig_*.png` |

## Provenance

Plan: working repo `duda-particle-model`, `notes/plan_linear_F_terms.md`;
development in `linear_F_terms/`. Conventions: reports 001, 005 (§7 null
theorem), 006 (ansatz, appendix), 010 (Legendre theorem, matrix-cap orbit
theorem, lattice port).
