# Appendix — the canonical ansatz is a flat so(1,3) connection

*Added 2026-08-28 under METHOD §7 (new material extending the report).
This appendix changes no conclusion of report 006; it identifies the
geometric structure behind §2 and sharpens the escape map of §7. All
numbers regenerate from `appendix_flat_connection.py` (numpy + scipy,
seconds on CPU), committed record in
`results/appendix_flat_connection.json`.*

## Statement

The dressing of the canonical ansatz, $M = o\,M_0\,o^\top$ with
$o(x)\in SO^+(1,3)$ (any profiles, any centers), is **pure gauge**: the
right-invariant Maurer–Cartan form

```math
\omega_\mu = (\partial_\mu o)\,o^{-1} \in so(1,3)
```

carries the whole first-derivative content of the field,

```math
(\partial_\mu M)\,\eta = [\,\omega_\mu,\; M\eta\,],
\tag{A1}
```

and is **flat**:

```math
\partial_i\omega_j - \partial_j\omega_i - [\omega_i, \omega_j] = 0 .
\tag{A2}
```

(Sign convention: for the *right* form $(\partial o)o^{-1}$ the flatness
identity carries a minus; the plus convention belongs to the left form
$o^{-1}\partial o$, equivalently to pure gauge written as
$o\,\partial(o^{-1}) = -\omega$.)

Derivation of (A1), three lines: $\partial M = (\partial o)M_0o^\top +
oM_0(\partial o)^\top = \omega M + M\omega^\top$; membership
$\omega\in so(1,3)$ means $\omega^\top\eta = -\eta\,\omega$; hence
$(\partial M)\eta = \omega(M\eta) - (M\eta)\omega$. Derivation of (A2):
direct differentiation of $\omega_\nu=(\partial_\nu o)o^{-1}$ gives
$\partial_\mu\omega_\nu - \partial_\nu\omega_\mu = [\omega_\mu,\omega_\nu]$.
Both identities therefore have an exact route (the derivations above) and
an independent numeric route (below).

## What this adds to the report

- **§2 read geometrically.** The ansatz is a pure-gauge (zero-curvature)
  configuration of a Lorentz connection, and the field is static, so the
  time leg vanishes: $\omega_0 = 0$. The absence of time components of
  $F$ in the derivative slots is thus a property of the construction —
  exact at every order in the dressing amplitude $m$ — while the
  *matrix*-slot spatiality of §2 is the leading-order statement (as §7
  already records: second-order dressing terms reintroduce non-spatial
  matrix components at $O(m^6)$ in the energy).
- **The §7 escape map, sharpened.** Any covariant-derivative deformation
  $D_\mu = \partial_\mu + C_\mu$ of this report's setting must supply
  exactly what the ansatz lacks: **curvature, or a nonzero time leg
  $C_0$ on a static field.** Report 007's companion appendix
  (`APPENDIX-no-connection-from-M.md`) closes both requirements for
  every $C_\mu$ built pointwise from $M$ alone.
- **A bridge to standard gauge-theory language.** An external reader can
  now see at a glance that "boost hedgehog dressing" = pure gauge, which
  makes §2's collapse a one-line consequence rather than an ansatz-specific
  computation.

## Measured

On the report's canonical single-center dressing ($m=0.2$, $p=0.5$,
$x_0 = (0.6, -0.3, 0.8)$, vacuum $\mathrm{diag}(1,0,0,0)$), central
finite differences with step $h$:

| check | value |
|---|---|
| $\omega_i \in so(1,3)$, residual | $1.7\cdot10^{-11}$ |
| transport identity (A1), max error | $9.9\cdot10^{-12}$ |
| flatness (A2), $h=10^{-5}$ | $5.6\cdot10^{-12}$ |
| flatness (A2), $h=10^{-4}$ | $1.5\cdot10^{-10}$ |
| two-center dressing, flatness | $1.7\cdot10^{-11}$ |
| **negative control**: generic $so(1,3)$-valued field, curvature | $2.0$ ($O(1)$: NOT flat) |

The $h$-refinement shows the flatness residual is finite-difference
limited (grows $\sim h^2$), i.e. the zero is structural; the negative
control shows the test distinguishes flat from non-flat at $O(1)$.

## What this appendix does not claim

- No change to the scope of the no-go: the report's conclusions are
  neither widened nor narrowed. The virial identity, the collapse, and
  the sign assembly stand exactly as merged.
- Flatness is a property of the *tested ansatz family* (pure-gauge
  dressings of the vacuum), not of general field configurations of the
  model.

## Reproduction

```bash
python appendix_flat_connection.py   # asserts all rows; ~seconds, CPU
```

## Provenance

- Question and first derivation: working repo `duda-particle-model`,
  `notes/notatka_koneksja_kowariantna_2026-08-27.md` (revised 2026-08-28,
  commits `cdba6c2`, `74523e7`), probe `notes/connection_probe.py`.
- Conventions: report 006 §1 (this directory); $F$ and invariants:
  report 001 (merged).
