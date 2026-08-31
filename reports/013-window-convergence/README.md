# Report 013 — The window's candidate wells do not certify: the dilution drift outlasts a 24-cycle continuation

*2026-08-31 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
resolves, negatively on this budget, the sharp open question left by
report 010: do the fixed-depth interior minima of the canonical
Hamiltonian survive relaxation deepened to an observable-level
criterion?*

## Notation (self-contained)

Report 010 scans fundamental-reading Lagrangians
L = −(1/2) I₁ + γ (I_j)² − V(M) on the report-004 lattice stack
(32³ hedgehog, frozen boost clock tangent a₀, clock configurations
Ṁ = ω a₀) and finds, inside a two-sided window of the quartic
coupling γ, interior minima of the canonical Hamiltonian
E_cell(M; ω) at fixed relaxation depth for four invariant cells; its
deepest probe (six L-BFGS cycles at coupling ×14 of the frozen-tuned
value, cell C10 — the square of the time-leg of F along the field's
clock axis) saw the interior minimum at ω = 0.15 hold for four
levels and then migrate to the still-descending top rung. "Bracket
differences" below are d(ω) = E(ω) − E(0.15) on the rung set
{0, 0.1, 0.15, 0.2, 0.28}: an interior well at 0.15 requires
d(0.1) > 0 and d(0.2) > 0 (and d(0.28) > 0 for the wide bracket).
The observable-level stopping rule (the lesson of reports 011–012)
demands every d drift by < 5% of its magnitude over four consecutive
cycles before the run may stop.

## What was run

`continue_window.py`, functional imported verbatim from report 010's
committed stack:

- **×14, continued**: the five persisted sixth-cycle rung fields of
  report 010 (`deep14_om*.npz`, independently route-2-verified in the
  010 appendix) continued for 24 further L-BFGS(150) cycles per rung
  (30 total), energies and gradients recorded every cycle;
- **×10, fresh**: the same rung set from the base profile
  (Adam 500 + up to 24 cycles), same records;
- final fields persisted (`win_*_om*.npz`).

## Result: no observable-level plateau; both couplings lose the upper bracket

| arm | final d(0.1) | final d(0.2) | final d(0.28) | minimum | stop rule met |
|---|---|---|---|---|---|
| ×14 (+24 cycles) | −1.3·10⁻⁵ | −7.9·10⁻⁴ | −4.1·10⁻³ | top rung 0.28 | no |
| ×10 (24 cycles) | +2.2·10⁻³ | +9.7·10⁻⁴ | −2.9·10⁻⁴ | top rung 0.28 | no |

At ×14 the continuation dismantles the entire structure: d(0.2)
turns negative within three continued cycles, d(0.28) deepens
roughly linearly throughout (−1.6·10⁻³ at six cycles, −4.1·10⁻³ at
twenty-four, still deepening at −1.4·10⁻⁴ per cycle at the end), and
by cycle 14 even d(0.1) crosses zero: the sampled energies order
monotonically toward high ω. At ×10 the interior holds against the
near rungs for the whole run (d(0.1), d(0.2) stay positive and even
grow), but the wide bracket never certifies: d(0.28) hovers just
below zero for most of the run, makes one brief positive excursion
(cycles 20–22), and ends negative and deepening — the top rung sits
below the candidate minimum for essentially the entire budget, the
same reversal the shallower ×10 probe of report 010 showed. Neither arm comes near
the observable-level stopping rule.

![drift](results/fig_drift.png)

**Conclusion.** On this lattice and budget the answer to report
010's sharp open question is negative: inside the coupling window
the dilution drift is slowed but never saturates, and the candidate
interior minima of the canonical Hamiltonian do not certify at
depth — at ×14 the well's neighborhood inverts entirely; at ×10 the
failure begins later but proceeds identically. The contrast with the
energy-functional clock of report 008 is sharp and protocol-matched:
the boost-channel energy-reading bracket survived an identical
24-cycle continuation with its ordering intact (merged record,
report 009 §4), while the fundamental-reading wells of the grid do
not. Within everything measured so far, a converged
fundamental-reading clock on this stack requires either couplings
and cells outside report 010's single-invariant grammar (its own
closing hypothesis: a same-channel drive–brake pairing with a convex
template) or a different stabilization mechanism; the
energy-functional reading remains the only reading with a
convergence-certified localized clock.

## Limitations

- Two couplings (×10, ×14), one cell (C10), one rung set, 24-cycle
  budget: the negative is a budget statement, not a nonexistence
  theorem; a plateau beyond 24 cycles is not excluded (the ×14
  trends give no hint of one).
- The rung set is inherited from report 010; no rungs above 0.28
  were sampled, so where the drift would terminate is unknown
  (report 010's γ = 0 control dove at ω ≈ 0.8).
- Frozen tangent, single generator, 32³, one spacing — as in
  004–010.

## Equation-to-artifact map

| object | artifact |
|---|---|
| deep continuation, both arms, per-cycle records | `continue_window.py` → `results/window_deep.json` |
| digest: final differences, drifts, verdicts | `analysis.py` → `results/verdicts.json` |
| final rung fields | `results/win_*_om*.npz` |
| figure | `make_figures.py` |

## Reproduction

`bash reproduce.sh` — with report 010's stack available it reruns the
continuation (sentinel-flagged; hours on GPU) and asserts the
top-rung verdicts, the sign pattern of the final differences, the
×14 end-of-run deepening, and the absence of the stop-rule flag;
without the stack it checks the committed record.
