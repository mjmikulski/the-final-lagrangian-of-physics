# Report 009 — The rotational sector: a protocol-limited ladder, a vacuum-dominated channel, and the well-posed fixed-J route

*2026-08-28 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
the angular-momentum half of the article-1 program. Substantially
revised after review round 1, whose two findings both survived
deeper measurement and **reversed the original claims**: this report
now documents two measured obstructions and the formalism that the
rotational sector actually needs.*

## Context and result

Report 008 established the localized clock in the boost channel. The
electron's defining property, however, is angular momentum: the model
must rotate. This report's first version transplanted 008's ladder to
the frozen rotation tangent and claimed an interior well at the
predicted rung; review round 1 found (a) the bracket was not
converged, and (b) the reported "angular momentum" was not a charge.
Both findings are confirmed here by deeper measurement, and the
honest picture is:

1. **The env-frozen rotational ladder is not protocol-convergent —
   its well claim is withdrawn** (§2, §4). Continuing the relaxation
   of the persisted rungs (up to 24 L-BFGS cycles, tol $10^{-7}$)
   *reorders the bracket*: the energies after the budget order as
   $E(0.304) < E(0.152) < E(0.217) < E(0)$, with the non-stiff rungs
   still descending at $\sim 10^{-6}$/cycle without saturation while
   the $0.217$ rung alone is fully converged. The interior-minimum
   structure seen at fixed protocol depth does not survive; the
   ladder can only bound, not locate, any rotational well.
2. **The pure internal rotation channel is broad-and-stiff on this
   box — measured in a bounded constrained surrogate** (§5; scope
   corrected in review round 2). The functional
   $E_J = E_{\rm stat} + J^2/(2 I[M])$ with the interior conjugation
   tangent is bounded, minimizes cleanly, and matches the
   rigid-profile scaling ($[E(J)-E(0)]/[J^2/2I_0] \to 1.02$, $I$
   constant to $0.2\%$). Two honest scope limits, both from round 2:
   (i) this is **not a derived Routh reduction** for the
   pinned-boundary problem — the masked tangent's flow
   $M_\theta = e^{\theta m(x)W} M e^{-\theta m(x)W}$ carries an
   interface term $\propto (\partial_i m)[W, M]$ across the frozen
   shell, so the angle is not cyclic, $J$ is a *prescribed parameter*
   (not a Noether charge), and the interior mask is itself a sharp
   envelope; a genuine reduction needs the boundary rotated by the
   global generator (or the interface potential included). (ii) The
   single-box measurement ($I_0 = 3.75\cdot10^3$, kinetic-density
   PR $\approx 455$ on $32^3$) shows only that the channel is broad
   on this box, so the extensivity claim is now backed by the
   **measured box-size scaling** (§5a): at fixed spacing, boxes
   $L = 24, 36, 48$ give $I_{\rm pure} \sim L^{2.93}$ — the volume
   law, measured.
3. **The Skyrme-like combined generator was measured on this field
   and does not help — with the claim's scope corrected in round 2**
   (§6): for $\zeta = -(x\partial_y - y\partial_x) + [W, \cdot\,]$
   the residual $|\zeta M|(r)$ is *flat* ($\approx 1.13$ on every
   shell) and $I_{\rm comb} = 3.4\cdot10^3$, PR $= 312$
   (`combined_generator` producer in this report): **this hedgehog's
   radial texture is not axially equivariant**, so the combined
   rotation buys nothing *on this configuration*. Round 2's
   counterexample is accepted and recorded as constructive: unequal
   transverse eigenvalues do **not** by themselves forbid
   equivariance — the $\varphi$-wound texture
   $M = R_z(\varphi)\,B(\rho, z)\,R_z(\varphi)^\top$ has three
   distinct eigenvalues yet $\zeta M \equiv 0$ exactly. The
   measured obstruction is therefore a property of the *radial
   hedgehog texture*, not of biaxiality per se — confirmed
   analytically in §6: the uniaxial hedgehog AND a spherical-frame
   biaxial hedgehog at full $\delta = 0.3$ (the counterexample class)
   both show only a $\propto 1/r$ discretization residual, while the
   working texture's residual is $O(1)$ and flat. (A naive
   "shrink $\delta$ in the potential" route was also probed in the
   working repo and *backfires* — the breaking grows as
   $\delta \to 0$ at fixed texture, since the $(1, \delta)$ pair
   splitting grows and the seed texture does not become equivariant
   by itself.) The concrete spin candidate is the **axially
   equivariant defect ansatz** — $\zeta$ is its exact symmetry, so a
   finite, core-localized inertia and a genuine Noether $J$ are
   available; this is the next-report line.
4. The proxy $\tilde J = I_R\,\omega$ of the first version is retained
   in the JSON for the record but carries **no physical claim** (§3):
   it is convention-dependent (tangent normalization and envelope).

**What survives from the first version:** the raw ladder record and
its independent route-2 verification (the *energies* are correct; it
is their convergence status that changed), the sign-control result,
and the methodological correction about report 004's channel
construct (which cancels rotational densities, $G \approx \eta$ on
rotation tangents).

## 1. Setup

As in report 008 (`ladder_i1sq_defs.py` via runpy): working metric
$G$, $\gamma = 70.61$, fresh-start rungs, Adam + L-BFGS cycles with
recorded `E_levels`. Ladder tangent: frozen
$a_{\rm rot} = \mathrm{env}\cdot(WM - MW)/\lVert\cdot\rVert$. Fixed-J
scan tangent: pure interior generator, no envelope/normalization.

## 2. The ladder and its non-convergence

At fixed protocol depth the ladder showed an interior minimum at the
predicted rung $\omega_R = \sqrt{C_1^r/C_2^r} = 0.217$ (all recorded
levels), with the sign control clean at $\omega = 0$ — see
`results/rot_ladders.json` and the figure. Review round 1 tested the
persisted $0.152$ field with continued relaxation and found it
dropping below the claimed minimum; the deep run of §4 confirms and
extends this. **The well-location claim is withdrawn.** What remains
established: (i) the $0.217$ configuration is a genuinely converged
stationary point ($\lVert g\rVert_\infty = 2.7\cdot10^{-4}$, five
levels identical to $10^{-6}$); (ii) the other rungs are *not*
converged and keep descending — most plausibly toward the deeper
static family known from report 007 (the two-family structure), at a
rate growing with $\omega$.

![rot ladders](results/fig_rot_ladders.png)

**Route 2** (`verify_energies.py`): the persisted bracket energies
are reproduced from scratch to $10^{-9}$ relative — the *numbers*
are right; the *interpretation* changed.

## 3. The proxy, retained without claims

$\tilde J = I_R\,\omega$ with $I_R = 2\int k_1^r = 0.547$ (frozen-
tangent channel scale) is recorded in the JSON. It is not a canonical
momentum or Noether charge: rescaling the frozen tangent
$a \to \lambda a$ sends $\tilde J \to \lambda \tilde J$. No conclusion
rests on it.

## 4. Deep convergence: the bracket reorders

`deep_converge.py` continues the persisted rung fields (and a fresh
$\omega = 0$) for up to 24 L-BFGS cycles (tol $10^{-7}$):

| $\omega$ | 0.0 | 0.152 | **0.217** | 0.304 |
|---|---|---|---|---|
| $E$ start | 5.070782 | 5.070727 | 5.070726 | 5.070757 |
| $E$ after budget | 5.070735 | 5.070679 | **5.070726 (converged)** | 5.070655 |
| cycles / last change | 24 / $-1.1\cdot10^{-6}$ | 24 / $-1.2\cdot10^{-6}$ | 1 / $-3\cdot10^{-9}$ | 24 / $-1.4\cdot10^{-6}$ |

The post-budget ordering is $0.304 < 0.152 < 0.217 < 0.0$, still
descending — no statement about a rotational interior minimum
survives. Contrast with the boost channel: the identical deep run on
report 008's persisted bracket keeps its ordering
$E(0.35) < E(0.2) < E(0)$ under the same common creep (stable
differences there; unstable here).

## 5. Fixed-J: well-posed, and a measured no-go for the pure channel

The Routhian $E_J[M] = E_{\rm stat}[M] + J^2/(2 I[M])$ with
$I[M] = 2\int k_1[\zeta_{\rm int}(M)]$, $\zeta_{\rm int} =
(1-\text{shell})\,(WM - MW)$, minimized over $M$ at prescribed $J$
(`fixedj_scan.py`): bounded, no runaway, rigid-profile scaling
verified ($[E(J)-E(0)]/[J^2/2I_0] \to 1.02$ at $J = 0.8$; small-$J$
points sit at the relaxation noise floor), $\omega = J/I$ an output.
Measured: $I \approx 3.75\cdot10^3$, constant to $0.2\%$ over
$J \in [0, 0.8]$, kinetic-density PR $\approx 455$ — the channel
turns the whole (anisotropic-vacuum) interior, not the defect. This
is the quantitative obstruction on this box; the scaling below makes
it extensivity.

![fixed-J](results/fig_fixedj.png)

### 5a. Box-size scaling: the volume law, measured

`inertia_scaling.py` repeats the inertia measurement in three boxes at
fixed spacing $H = 1.5$ (central crops of the seed, same texture
pinned at a smaller radius; short static relax per box):

| $L$ | 24 | 36 | 48 | fit |
|---|---|---|---|---|
| $I_{\rm pure}$ | $3.59\cdot10^2$ | $1.33\cdot10^3$ | $2.71\cdot10^3$ | $\sim L^{2.93}$ |
| $I_{\rm comb}$ | $3.54\cdot10^2$ | $1.11\cdot10^3$ | $2.07\cdot10^3$ | $\sim L^{2.56}$ |

$I_{\rm pure}$ follows the volume law: the pure internal channel's
inertia is extensive, and $\omega = J/I \to 0$ at fixed $J$ as the
box grows — the isolated-rotor no-go for this channel, now with the
scaling evidence review round 2 asked for. ($I_{\rm comb}$ also grows
strongly on these sizes; its sub-volume exponent reflects the partial
cancellation and boundary effects, not finiteness.)

## 6. The texture test: what actually breaks the symmetry

`combined_texture.py` measures the off-axis shell profile of the
axial-symmetry residual $|\zeta M|(r)$ on three configurations:

| configuration | $|\zeta M|$ at $r = 4.5$ | at $r = 19.5$ | $I_{\rm comb}$ |
|---|---|---|---|
| working 004 field ($\delta = 0.3$) | 1.10 | 0.59 | $3.4\cdot10^3$ |
| uniaxial hedgehog (analytic) | 0.32 | 0.077 | $5.7\cdot10^2$ |
| spherical-frame biaxial, $\delta = 0.3$ (analytic) | 0.33 | 0.088 | $6.9\cdot10^2$ |

The two equivariant textures fall off as $\propto 1/r$ — the pure
discretization residual (exact symmetry in the continuum) — **at both
$\delta = 0$ and $\delta = 0.3$**, while the working texture stays
$O(1)$ out to the boundary. The symmetry breaking is a property of
the texture choice, not of the biaxial spectrum. (The spherical-frame
texture pays for its equivariance with a frame singularity on the $z$
axis — the well-known linear defect of biaxial hedgehogs — whose cost
scales with the transverse amplitude $\delta$; this is where a small
$\delta$ genuinely helps.)

![scaling and texture](results/fig_scaling.png)

## 7. Limitations and the road forward

- Rigid collective rotation of the **working radial texture** is
  exhausted (extensive inertia, measured scaling); the equivariant
  ansatz of §6 is the concrete spin candidate (next report): analytic
  spherical-frame seed with an equivariant boundary, statics
  competitiveness, finite $I(L)$, and a genuine fixed-J reduction
  (with an equivariant boundary the rotation is a symmetry of the
  configuration space, so the angle is cyclic and $J$ is a Noether
  charge — closing this report's §5 caveat (i)).
- The deep-convergence budget (24 cycles) bounds but does not close
  the creep; the boost-channel contrast (§4) is protocol-matched.
- $32^3$, one spacing; frozen-shell boundary as in 004–008.

## Equation-to-artifact map

| object | artifact |
|---|---|
| ladder record (fixed protocol depth) + proxy | `ladder_rot.py` → `results/rot_ladders.json` |
| deep-convergence run (the reordering) | `deep_converge.py` → `results/deep_converge.json` |
| fixed-J scan (pure internal channel) | `fixedj_scan.py` → `results/fixedj.json` |
| box-size scaling of the inertia | `inertia_scaling.py` → `results/inertia_scaling.json` |
| texture test (working vs equivariant analytic) | `combined_texture.py` → `results/combined_texture.json` |
| independent energy route | `verify_energies.py` |
| persisted bracket fields, frozen tangent | `results/rot_rung_om*.npz`, `results/a0r_frozen.npz` |
| figures (committed artifacts) | `make_figures.py` |

## Reproduction

`bash reproduce.sh` — with report 004's fields (or `M5_FIELDS_DIR`)
reruns all producers (sentinel-flagged) and asserts: the ladder
record's internal consistency, the deep-run reordering (the withdrawn
claim stays withdrawn), the fixed-J rigid-scaling ratio and the
extensive-inertia signature, and the route-2 match; without fields it
checks the committed artifacts.
