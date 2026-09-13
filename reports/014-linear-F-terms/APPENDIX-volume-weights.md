# Appendix — volume weights on the linear sector: √e₃·φ and adj(M)·Φ

*Added 2026-09-13 under METHOD §7 (new material extending the report;
no conclusion of report 014 changes; revised after review round 1, which
corrected the normalisation of §5, the sign search of §5, the
identification of §2 and withdrew the pair-law estimate of §6). Motivation: the model author's
GR analogy √(−g)R, with a volume weight that does not vanish on the
rank-3 vacuum — w = e₃(ηM), the third elementary symmetric polynomial
of the spectrum, or the adjugate adj(ηM) as the analogue of √(−g)g^{μν}.
All numbers regenerate from `reproduce_appendix.sh` (CPU, about 25
minutes; the vacuum script dominates); committed records in
`results/appendix_volume_*.json`. Conventions of the E₀ = 100 route
(the report's Notation; the code snapshot in `e0_route/`, the fields of
report 016).*

## 0. Why an appendix to 014

A weight in front of the double trace φ (the Ricci-scalar analogue,
null with a constant coefficient by report 005) is exactly the
mechanism of result 2: a field-dependent coefficient makes a null
linear term dynamical, as the weight √(−g) does for the ΓΓ form of the
Einstein–Hilbert action. Both candidates below are members of the
rank-12 module of result 1 with **polynomial** coefficients — smooth
where the individual spectral projectors of the report are not (the
eigenvalue collisions of the electron, result 4) — and everything the
report established for the module applies to them. A related family was
proposed on substrate-framework discussion #186 (the model author,
2026-09-03: R_G, a double mixed trace with G ∈ {η, ηMη, M⁻¹, η + 2uuᵀ})
and closed as a Newton mediator there (OpenWave R14, 2026-09-05); §2
records that for G ≠ η that object is not a contraction of F at all,
so nothing from R14 transfers to the terms studied here.

## 1. The weights

With N = ηM and e_k(N) the elementary symmetric polynomials of its
spectrum (checked in sympy for a general symmetric M):

```math
e_3 = \tfrac16\big[(\mathrm{tr}N)^3 - 3\,\mathrm{tr}N\,\mathrm{tr}N^2 + 2\,\mathrm{tr}N^3\big],\qquad
\mathrm{adj}(N) = e_3\,\mathbb 1 - e_2 N + e_1 N^2 - N^3,\qquad \mathrm{tr}\,\mathrm{adj}(N) = e_3 ,
```

and in the eigenframe adj(N) = diag(w_a) with w_a = Π_{c≠a} e_c. On
the vacuum (E₀, 1, 1/E₀, 0): e₃ = E₀·E₁·E₂ = 1 exactly (the choice
E₂ = 1/E₀ makes the "volume" of the three live axes unity), det = 0,
adj = P₃ (the projector on the zero axis), and
∂e₃/∂e_a = (1/E₀, 1, E₀, E₀ + 1 + 1/E₀): the weight is sensitive to
the two small eigenvalues at relative order 1/E₂. In the lattice
convention E₃ = E₂ = 0.01, e₃ = 2.01 and w = (10⁻⁴, 0.01, 1, 1).

## 2. Decomposition in the basis of result 1

With F_ab = P_a^{μα}P_b^{νβ}F_{μναβ} and Φ_{νβ} = η^{μα}F_{μναβ}:

```math
L_1 := \sqrt{e_3}\;\varphi = \sqrt{e_3}\sum_{a<b} 2F_{ab},\qquad
L_2 := \mathrm{adj}(N)^{\nu\beta}\,\Phi_{\nu\beta} = \sum_{a<b}(w_a + w_b)\,F_{ab},
```

both to 10⁻¹⁴ on random points (`appendix_volume_identities.py`).
On the vacuum with E₃ = 0, L₂ reduces to F₀₃ + F₁₃ + F₂₃, a non-null
combination. L₂ is **not** the R_G of the discussion with G = adj
(review round 1): in R_G = Σ G_cd(∂_μM^{νc}∂_νM^{μd} − ∂_μM^{μc}∂_νM^{νd})
the weight sits on the internal pair and η pairs a derivative slot with
a matrix slot, whereas L₂ puts the weight on the free pair of Φ and
keeps F's internal η. For G = η the two coincide (R_η = −φ, fitted
exactly); for G = adj·η the least-squares fit of R_G onto the six even
generators leaves a residual of 0.8 and R_G differs from L₂ by order
one: R_G with G ≠ η is not a contraction of F and lies outside the
report's grammar (it is a two-derivative term of the non-commutator
kind that 006 §7 and the report's optional extension mention).

## 3. Euler–Lagrange structure

The test of `null_test_linear.py` (second jets, autograd) in the
E₀ = 100 conventions, with the Euler–Lagrange vector projected on the
six frame directions δM = η[G, N] and the four eigenvalue directions
(`appendix_volume_el_test.py`):

| point | φ | I₁ | L₁ | L₂ |
|---|---|---|---|---|
| generic (spectrum and jet random) | null (10⁻¹⁶) | dynamical | dynamical | dynamical |
| constant-spectrum jet, frame part | 0 | 0.4 | **10⁻¹¹** | 1.3 |
| constant-spectrum jet, eigenvalue part | 0 | 0.2 | **1.0** | 1.1 |

L₁ is a (∂λ)·(∂frame) coupling exactly: within the constant-spectrum
orbit it exerts no force, and where the frame varies it pushes the
eigenvalues. L₂ carries the rotated projector and acts on the frame as
well — the discussion's finding that R_G with an M-dependent G is not
a total derivative on two-plane boost textures.

## 4. The weights on the electron

On the committed electron of report 016 (E₀ = 100, box 12, n = 32):

- frozen sector: e₃/e₃_vac runs from 1.0 at the boundary to 17 in
  the core (shell means 16.5, 15, 11, 4.9, 1.8 for r < 0.5, 0.5–1,
  1–2, 2–4, 4–6), positive everywhere. The mechanism is the trace
  constraint: the sum of the eigenvalues is conserved, so where e₁
  drops from 1 to 0.44 the small pair rises to ≈ 0.3 and e₂ of the
  spatial spectrum grows ≈ 17×. The tail is long (1.8× at r = 6)
  because the eigenvalue deviations decay as 1/r².
- time sector free (the screened endpoint of 016): e₃ < 0 on
  5·10⁻⁴ of the sites (smallest eigenvalue down to −0.04), so √e₃
  needs either w = e₃² or the frozen sector.
- φ has a definite sign on the electron (∫|φ| = −∫φ = 186 in the box;
  ∫φ is the boundary flux of the hedgehog), and the weighted integral
  X₁ = ∫√(e₃/e₃_vac)φ = −356 against E_stat = 27.3, with 46% of X₁
  from r < 3 (`appendix_volume_lattice_phi.py`). A coupling κ
  therefore reweights the electron's static energy by
  κ·X₁/E_stat ≈ −13κ.

## 5. The vacuum

**Frame twists** (`appendix_volume_vacuum.py`, part A): on a periodic
box with spectral derivatives, N = R(x)N_vacR(x)⁻¹, R = exp(tW(x)),
W a smooth random so(1,3) field with all six generators (several
boost planes), grids n = 16–48, three twists:

- ∫φ = 0 to 10⁻¹⁶ relative to ∫|φ| on every grid (the total
  derivative), and ∫L₁ = 0 to 10⁻¹¹ at the amplitudes of result 5
  (at unit-rapidity amplitude the polynomial e₃ loses 6·10⁻⁵ to
  cancellation between terms of order 10⁶ and the residual is that
  round-off times φ). L₁ is exempt from the cubic saddle of result 5:
  the spectrum is constant on every frame twist, so L₁ = √e₃_vac·φ
  is a boundary term there to all orders.
- ∫L₂(t) fitted to c₂t² + c₃t³ + c₄t⁴ on t = ±{5, 10, 20, 40}·10⁻⁴:
  c₂ = 0 (the O(t²) part is the null term with vacuum coefficients),
  c₃ ≠ 0 with both signs across the three twists (−2.4, +2.6,
  +1.4·10⁵ in the box's units), converged between n = 32 and n = 48
  to four digits; the static F·F on the same twists (the model's
  normalisation, asserted against `e0_route/lagrangian.py`) is c₄ ≈
  1.5–2.1·10¹². The cubic term dominates below the crossover
  amplitude t* = |c₃|/c₄ ≈ 0.6–1.6·10⁻⁷ per unit coupling (an rms
  rapidity of 3·10⁻⁷): the vacuum saddle of result 5, for L₂ as for
  every individual class.

**The threshold for L₁** (part B). With F·F in the model's
normalisation (Σ over ordered pairs, i.e. 2Σ_{i<j}‖[A_i,A_j]‖²; the
first version of this appendix used half of it, review round 1) the
worst static jet at the vacuum attains the Cauchy–Schwarz bound:
ρ_min = min F·F/φ² = 1/3 exactly (from
|φ| ≤ 2Σ_{i<j}|([A_i,A_j])_{ij}| ≤ 2Σ‖[A_i,A_j]‖_F), and jets with
either sign of φ attain it. Shifting the eigenvalues at fixed jet
leaves F·F and φ unchanged and moves only V = Σ(e_a − E_a)² and the
weight. Two branches compete. Linear response,

```math
H \supset \sum_a \delta e_a^2 - \kappa\,(g\cdot\delta e)\,\varphi + F\!\cdot\!F
\;\ge\; F\!\cdot\!F - \tfrac{\kappa^2\varphi^2|g|^2}{4},\qquad
g_a = \partial_a\sqrt{e_3/e_{3,\rm vac}},
```

gives κ_lin = 2√ρ_min/|g| = 0.016 (E₃ = 0, |g| = 71) and 0.033
(E₃ = E₂, |g| = 35.5). The other branch is the **collapse of the
weight** (review round 1's counterexample): driving e₃ to zero along
δe₂ = δe₃ = −(E₂ + E₃)/2 costs only V_c = (E₂ + E₃)²/2 (5·10⁻⁵ or
2·10⁻⁴) and takes h = √(e₃/e₃_vac) − 1 to −1, so with the sign of κφ chosen
accordingly the gain is linear in κ|φ| and

```math
E \supset V_c + \lambda^4 Q_0 - \kappa|arphi_0|\lambda^2 \;\ge\; 0 \iff
\kappa \le \kappa_{
m col} = 2\sqrt{
ho_{\min} V_c}\,,
```

κ_col = 0.0082 (E₃ = 0) and 0.0163 (E₃ = E₂), below κ_lin in both
conventions. Exactly: the pointwise energy on the worst jet at
amplitude λ is f(p) + λ⁴Q₀ with p = κλ²|φ₀| of either sign and
f(p) = min_{δe}[Σδe² − p(√(e₃/e₃_vac) − 1)], so stability for all
amplitudes is κ² ≤ ρ_min·min_p p²/(−f(p)); one table of f(p) with the
shifts bounded by |δe_a| ≤ 1 gives the **local** threshold
κ_c = 0.0082 (E₃ = 0) and 0.0164 (E₃ = E₂), at the collapse branch
(h ≈ −1 at the minimiser). On the electron (§4) this is a reweighting
of the static energy by about 20% (0.016 × 13 in the lattice
convention).

**Without the bound there is no threshold at all.** Minimising over
the jet amplitude first leaves the reduced density

```math
G(\delta e) = V(\delta e) - rac{\kappa^2}{4
ho_{\min}}\Big(\sqrt{e_3/e_{3,
m vac}} - 1\Big)^2 ,
```

and e₃ grows as the cube of the eigenvalues while V is quadratic: for
every κ ≠ 0, G < 0 once the eigenvalues exceed t* ≈ 4ρ_min e₃_vac/κ²
(measured on the uniform ray δe_a = t: t*κ² constant to within the grid
for κ = 0.003, 0.01, 0.03; t* ≈ 10⁴ at κ = 0.01). This is the runaway
of an indefinite term against a quadratic potential — the disease of
003's C2 and of 008 §4 — in the eigenvalue channel: with the model's
V, a weighted null term κ·w·φ is bounded below pointwise only if its
weight grows no faster than the eigenvalues themselves (w² ≲ e²),
which √e₃ ~ e^{3/2} violates. The local threshold therefore describes
a metastable vacuum, separated from the runaway by eigenvalues three
to four orders of magnitude above the vacuum's largest; a bounded
weight (or a potential steeper than quadratic) would be needed for a
stable one, and neither is chosen here.

## 6. Two facts that need no run

1. **The two configurations on which gravity failed.** On the
   constant-spectrum orbit of report 006 L₁ = √e₃_vac·φ is a boundary
   term and L₂ reduces to the null φ or vanishes (result 3(b)); on
   the F ≡ 0 screening family of report 016 both vanish identically.
   Neither candidate can move the Newton sign where it was measured
   nor bound the time sector; and with velocity degree ≤ 1 (result
   3(a)) neither enters the kinetic form, so the count of dynamical
   components is unchanged and the trace stays an algebraic
   constraint (tr N = ΣE + (κ/2)φ ∂_c√(w/w_vac) for L₁).
2. **The two-body question is open.** A first version of this
   appendix claimed E_int ≲ C/d² from "L₁ is a total derivative outside
   the cores"; review round 1 showed the estimate wrong. The frame
   gradient of the hedgehog is O(1/r), not O(1/r²): for
   N = E₂1 + Δ n nᵀ, n = x/r, exactly Σᵢ‖∂ᵢN‖² = 4Δ²/r², and the weight
   tail w − w_vac = O(r⁻²) of §4 is not confined to a core, so a cross
   term such as (w₁ − w_vac)φ₂ has the overlap ∫d³x/(r₁²r₂²) ∝ 1/d.
   Whether such contributions cancel is not established here in either
   direction; the claim is withdrawn and no pair law is asserted for
   L₁ or L₂. What stands is fact 1: on the configurations of 006 and
   016 both terms vanish identically.

## 7. What this appendix does not show

- No lattice scan with the weight was run, and no two-body
  measurement: after the withdrawal in §6 the pair law of L₁ is
  simply not known; on the electron a single-soliton scan would
  measure a core reweighting of the kind result 4 already shows.
- The thresholds and the runaway of §5 are pointwise (spectrum
  shifted at fixed jet); a field configuration realising them over a
  region pays gradient costs not counted here, so κ_c is the scale of
  the local instability and t* the scale of the runaway, not certified
  bounds.
- The sign choice w = e₃ versus e₃² where e₃ < 0 (the screened
  electron) is not made.
- E₃ = 0 versus E₃ = E₂ changes the vacuum weights of L₂
  ((0, 0, 0, 1) versus (10⁻⁴, 0.01, 1, 1)); both are recorded, neither
  is selected.

## Equation-to-artifact map

| object | artifact |
|---|---|
| identities, vacuum values, decomposition (§1–2), e₃ on the electron (§4) | `appendix_volume_identities.py` → `results/appendix_volume_identities.json` |
| Euler–Lagrange projections (§3) | `appendix_volume_el_test.py` → `results/appendix_volume_el_test.json` |
| φ and the weighted integral on the electron (§4) | `appendix_volume_lattice_phi.py` → `results/appendix_volume_lattice_phi.json` |
| twists, cubic fit, ρ_min, κ_lin, κ_c (§5) | `appendix_volume_vacuum.py` → `results/appendix_volume_vacuum.json` |
| structural assertions | `reproduce_appendix.sh` |

## Provenance

Plan and development record: `notes/plan_volume_weighted_R.md` and
`volume_R/` in the development repository (2026-09-13). The R_G
family and its verdict: substrate-framework discussion #186, comments
18275474 (2026-09-03) and 18303935 (2026-09-05). The trace constraint
used in §4 and §6: development note of 2026-09-13 (report 017,
forthcoming).
