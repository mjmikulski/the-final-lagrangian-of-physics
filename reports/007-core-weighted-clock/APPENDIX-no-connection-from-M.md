# Appendix — no connection from M alone: the blindness theorem for endomorphisms

*Added 2026-08-28 under METHOD §7 (new material extending the report).
This appendix changes no conclusion of report 007; it extends §1's
closure of field-dependent **coefficients** to field-dependent
**connections** $D_\mu = \partial_\mu + C_\mu(M)$. All numbers
regenerate from `appendix_no_connection.py` (numpy + scipy, seconds on
CPU; committed record `results/appendix_no_connection.json`); the figure
regenerates from the committed JSON via `make_appendix_figure.py`.*

## 0. Why §1 does not settle this

A coefficient **multiplies**; a connection **adds**. For a multiplier,
"constant on the ansatz" (§1) is the harmless case. For an additive
$C_\mu$, a constant added to a small $\partial M$ *dominates* — §1's
conclusion does not transfer, and a separate argument is needed. The
argument below exists and is stronger: not "constant", but
**nonexistent**.

## 1. Theorem: no connection value can be built from M alone

Pointwise and Lorentz-covariantly, the only endomorphism-valued objects
available from $M$ and $\eta$ are matrix functions of $A = M\eta$ (any
product string of $M$'s and $\eta$'s with one free upper and one free
lower index reduces to a power of $M\eta$; spectral/matrix functions
reduce to polynomials by Cayley–Hamilton). Then, in four steps:

1. $M$ symmetric $\Rightarrow$ both index lowerings give the same
   $A = M\eta$.
2. $A^\top\eta = \eta M \eta = \eta A$: $A$ is **$\eta$-symmetric**
   (self-adjoint in the $\eta$ form).
3. Powers of $A$ commute, so every polynomial $p(A)$ is
   $\eta$-symmetric.
4. $so(1,3)$ requires $\eta$-**anti**symmetry, $C^\top\eta = -\eta C$.
   Both at once force $\eta\,p(A) = -\eta\,p(A)$, i.e. $p(A) = 0$. ∎

So there is no nonzero $so(1,3)$-valued $C_\mu(M)$ at all — the
connection analogue of §1, by an independent mechanism (signature of the
$\eta$-form, where §1 used invariance of the spectrum under similarity).

Measured (generic random symmetric $M$): $\eta$-symmetry residual of
$A, A^2, A^3, A^4$ and of a generic polynomial $\le 1.4\cdot10^{-14}$;
$so(1,3)$ residual of the polynomial $195$ ($O(1)$ — not in the
algebra). **Negative control** (the test is not an artifact): allowing
one derivative immediately unblocks the construction —
$[M\eta, (\partial_\mu M)\eta]$ is in $so(1,3)$ (residual $0.0$ exactly:
the commutator of two $\eta$-symmetric matrices is
$\eta$-antisymmetric) and nonzero on report 006's ansatz (norm $0.25$).
The blockade is specific to building from $M$ *alone*.

## 2. The pointwise escape — index mixing — and its vacuum cost

The theorem leaves exactly one pointwise loophole: mixing the derivative
slot $\mu$ with the matrix slots ($\delta^a_\mu$ is Lorentz-invariant).
Minimal covariant realization, built on the normalized timelike
eigen-axis $u(M)$ of $M\eta$ (the object report 002's $G$-metric is
built from):

```math
C_{\mu ab} = \lambda\,(\eta_{\mu a}\,u_b - \eta_{\mu b}\,u_a),
\qquad
D_\mu M = \partial_\mu M + C_\mu M + M C_\mu^\top ,
```

antisymmetric in $(ab)$, hence $C_\mu \in so(1,3)$ for any $u$
(measured residual $10^{-16}$-level; in the vacuum $C_i = -\lambda K_i$,
a boost). On report 006's **static** ansatz ($m = 0.2$) it does open the
time sector that the structure theorem of 006 §2 forbids at $\lambda=0$:

| $\lambda$ | $\max\lvert F(D)_{0i}\rvert$ | $\max\lvert F(D)_{ij}\rvert$ |
|---|---|---|
| 0 (control) | 0 exactly | $3.09\cdot10^{-2}$ |
| 0.05 | $1.94\cdot10^{-3}$ | $5.09\cdot10^{-2}$ |
| 0.1 | $4.65\cdot10^{-3}$ | $7.57\cdot10^{-2}$ |
| 0.4 | $3.71\cdot10^{-2}$ | $3.27\cdot10^{-1}$ |

But the construction pays twice, and the two payments are the same term:

**(a) The vacuum breaks.** In the vacuum $D_iM_0 =
-\lambda g\,(e_ie_0^\top + e_0e_i^\top)$, hence (three lines)

```math
F(D)_{ij}\big|_{\mathrm{vac}} = \lambda^2\,(e_je_i^\top - e_ie_j^\top)
\neq 0 :
```

a **uniform, nonzero field strength filling empty space** — energy
density $\propto$ volume for any quadratic-in-$F$ energy. Measured:
$\max|F| = \lambda^2$ **exactly** over the scan (figure, panel b);
negative control $\lambda = 0$ gives $0$ exactly.

**(b) The opened sector is powered by the vacuum-breaking term.** Since
$D_i = D_i^{\mathrm{vac}} + O(m)$ with $D_i^{\mathrm{vac}}$ the very
term of (a), and $D_0 = O(\lambda m)$,

```math
\lvert F(D)_{0i}\rvert \;\sim\; c_1\,\lambda^2 m \;+\; c_2\,\lambda m^2
```

— a crossover at $m\sim\lambda$, not a clean power. Measured at
$\lambda = 0.1$: the vacuum-powered part (spatial legs frozen to their
vacuum value) carries $0.89$ of the full $|F(D)_{0i}|$ at $m = 0.0125$,
decreasing to $0.35$ at $m = 0.4$; log–log slopes in $m$: $1.20$ on the
small side ($\to 1$), $1.77$ on the large side ($\to 2$); slope in
$\lambda$ at $m = 0.0125$: $1.94$ ($\to 2$) (figure, panel a). So there
is **no regime in which the time sector works and the vacuum does not
pay**: at small amplitude the "effect" *is* the pathology.

**(c) A scalar switch cannot rescue it.** Multiplying $C$ by $s(M)$
vanishing in the vacuum runs into §1's blindness: on the ansatz every
algebraic scalar of $M$ equals its vacuum value pointwise (replicated
here: max relative drift $3.5\cdot10^{-13}$ over 500 random local
Lorentz dressings of report 004's vacuum; negative control with
orthogonal dressings: drift $2.0$). If $s(\mathrm{vac}) = 0$, then
$s \equiv 0$ on the whole ansatz and $C \equiv 0$.

*Boundary of the claim:* (a)–(c) are shown for this minimal covariant
realization. The mechanism of (a) — any nonvanishing covariant vector
of $M$ is nonzero already in the vacuum — is general, but we do not
claim a classification of all index-mixed connections.

![appendix figure](results/fig_appendix_connections.png)

*Figure: (a) the opened time sector on the static ansatz vs dressing
amplitude $m$ at $\lambda = 0.1$ — the full signal against its
vacuum-powered part, with $\propto m$ and $\propto m^2$ guides and the
$m = \lambda$ crossover; (b) the price: uniform vacuum field strength,
measured points on the exact $\lambda^2$ law.*

## 3. Outlook lemmas: derivative-built connections are shut too

The negative control of §1 above, promoted to a definition —
$C_\mu = \lambda\,[M\eta, (\partial_\mu M)\eta]$ — is the natural
*derivative-built* connection: it lies in $so(1,3)$ exactly and is
vacuum-safe by construction ($\partial M = 0 \Rightarrow C = 0$;
measured: $F(D) = 0$ exactly in the vacuum).

**Lemma 1 (time sector).** On **any static field**
$\partial_0 M = 0 \Rightarrow C_0 = 0 \Rightarrow D_0M = 0$, hence

```math
F(D)_{0i} \equiv 0 \quad \text{exactly}
```

(measured: $0.0$ to the bit).

**Lemma 2 (sign map).** In $\eta$-lowered variables $X_\mu =
(\partial_\mu M)\eta$, $A = M\eta$, the substitution is
$\tilde X_\mu = X_\mu - \lambda\,[A,[A,X_\mu]]$ and
$F(D)_{\mu\nu}\eta = [\tilde X_\mu, \tilde X_\nu]$. On report 006's
canonical ansatz the vacuum is rank-1, $A_0 = M_0\eta = -g\,e_0e_0^\top$,
and $\mathrm{ad}_{P}^2 = \mathrm{id}$ on commutators $[\kappa, P]$
(measured residual $0.0$ exactly for generic $\kappa \in so(1,3)$);
by the transport identity of 006's companion appendix
(`APPENDIX-flat-connection.md`) every $X_i$ of the ansatz *is* such a
commutator. Hence, exactly on the whole canonical family,

```math
\tilde X = (1 - \lambda g^2)\,X
\quad\Rightarrow\quad
F(D) = (1 - \lambda g^2)^2\, F(\partial)
```

(measured: $\lvert\tilde X - (1-\lambda)X\rvert \le 4\cdot10^{-12}$ at
$g=1$, $\lambda = 0.37$): every constant-coefficient quadratic energy in
$F(D)$ is $(1-\lambda g^2)^4 \ge 0$ times its baseline, so **report
006's tails, virial identity and sign map are invariant on this family**
(leading frozen order, degenerate only at the isolated point
$\lambda g^2 = 1$). The variant's genuine content is confined to
non-pure-gauge configurations (defect cores, the rank-rich model
vacuum) — the higher-order program of 006 §7 (working-repo
`newton_ho` / `connection_dev` line), outside this appendix's scope.

## Consequence

Within pointwise constructions from $M$: **endomorphism-valued
connections do not exist (§1), and the index-mixed escape costs the
vacuum, with its apparent time sector powered by that very cost (§2).**
Every remaining door out of report 006's no-go requires $\partial M$ in
the coefficient or connection — i.e. leaves the field-dependent-
coefficient family entirely and enters the higher-order program, where
the natural derivative-built representative keeps the time sector shut
*and* leaves the 006 sign map invariant on the whole canonical family
(§3).

## What this appendix does not claim

- No change to report 007's results (§§1–5 stand as merged); the clock
  mechanism and ladder results are untouched.
- No classification of all index-mixed connections (boundary note in
  §2); no statement about connections built from $\partial M$ beyond the
  two lemmas of §3 — their behavior on non-pure-gauge configurations
  (defect cores, the rank-rich model vacuum) is open.
- The crossover constants $c_1, c_2$ are realization- and
  normalization-dependent; only the structure ($\lambda^2 m$ vs
  $\lambda m^2$, ratios and slopes above) is claimed.

## Reproduction

```bash
python appendix_no_connection.py     # asserts all numbers; ~seconds, CPU
python make_appendix_figure.py       # figure from the committed JSON
```

## Provenance

- Question (raised while discussing §1's blindness theorem) and first
  derivations: working repo `duda-particle-model`,
  `notes/notatka_koneksja_kowariantna_2026-08-27.md` (revised
  2026-08-28, commits `cdba6c2`, `74523e7`), probe
  `notes/connection_probe.py`; rescaling lemma and part-B gates:
  `connection_dev/` (same commit).
- Ansatz and conventions: report 006 §1; $u(M)$ / $G$-metric lineage:
  report 002; blindness theorem: this report §1.
