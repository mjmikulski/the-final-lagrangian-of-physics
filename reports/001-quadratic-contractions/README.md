# Report 001 — Independent quadratic contractions of the M5 field strength

*2026-08-20 · Maciej J. Mikulski (AI-assisted, see [METHOD](../../METHOD.md)) ·
stage result of the extended-kinetic-term program*

## Context and result

The current M5 Lagrangian uses a single quadratic curvature scalar, the full
contraction $F_{\mu\nu\alpha\beta}F^{\mu\nu\alpha\beta}$. The model's author
(J. Duda, 2026-08-20) proposed repairing the two known defects of the
$4\times4$ model — the Newton force sign and the divergence
$\omega\to\infty$ under free energy minimization — by adding further
Lorentz-covariant terms of the same curvature² order, e.g. the Ricci-like
square built from the trace of $F$. Prerequisite question: **given the actual
algebraic symmetries of $F$, how many inequivalent quadratic scalar
contractions exist?**

**Result.** Exactly **six** (table below). No linear identity relates them —
neither for a generic tensor with the symmetries of $F$ nor for realizable
algebraic $F(A)$ tensors — so a constant-coefficient Lagrangian family built
on them is genuinely six-dimensional; a Jacobian test additionally confirms
functional independence at generic points. The current Lagrangian uses one of
the six, leaving five new candidate terms.

**Follow-up (independent, verified here).** The same day, the
substrate-framework campaign P239 found the same six-element basis and proved
a **no-go**: on purely spatial fields the six invariants satisfy three linear
identities, and on an explicit clock-direction counterexample all three
deviation combinations vanish while the baseline term is strictly negative —
so **no constant-coefficient quadratic combination that exactly preserves the
working 3×3 sector can repair the negative clock channel**. We verified both
statements with our independent evaluator (§5).

Index convention: Greek indices run over spacetime $0..3$; $\mu,\nu$ is the
derivative pair, $\alpha,\beta$ the matrix pair, $\gamma,\delta$ dummies.

## 1. The field strength and its symmetries

The M5 field is a symmetric matrix field $M_{\alpha\beta}(x)$ ($4\times4$),
with gradients $A_\mu\equiv\partial_\mu M$ and $\eta=\mathrm{diag}(-1,1,1,1)$
used for all raising/lowering. The field strength is the $\eta$-commutator

$$
F_{\mu\nu}=A_\mu\,\eta\,A_\nu-A_\nu\,\eta\,A_\mu,
\qquad
F_{\mu\nu\alpha\beta}
=A_{\mu\alpha\gamma}\,\eta^{\gamma\delta}A_{\nu\delta\beta}
-A_{\nu\alpha\gamma}\,\eta^{\gamma\delta}A_{\mu\delta\beta}.
\tag{1}
$$

**Assumption (diagonal Lorentz action).** Derivative and matrix indices are
assumed to transform under the *same* Lorentz representation: under
$\Lambda\in SO(1,3)$, $M\to\Lambda M\Lambda^\top$ while $\partial_\mu$
transforms as a covector of the same $\Lambda$. Then $F$ is an honest rank-4
Lorentz tensor and any complete $\eta$-contraction of $F\otimes F$ is a
scalar — including contractions pairing a derivative index with a matrix
index. This is how the current M5 formulation treats $M$ (a spacetime
tensor; the numerical $SO(1,3)$ gates of the production stack act with one
common $\Lambda$ on all slots), but it is a structural choice: if
$\alpha,\beta$ were internal-frame indices (as in tetrad gravity), mixed
contractions would require a tetrad and $I_3$–$I_6$ below would not be
scalars. Author-gated.

**Symmetries** (derivations one-line; numerics in
`results/numerical_results.json`):

| candidate | status | evidence |
|---|---|---|
| $F_{\mu\nu\alpha\beta}=-F_{\nu\mu\alpha\beta}$ | holds | immediate from (1); residual exactly 0 |
| $F_{\mu\nu\alpha\beta}=-F_{\mu\nu\beta\alpha}$ | holds | $A_\mu$ symmetric $\Rightarrow(A_\mu\eta A_\nu)^\top=A_\nu\eta A_\mu$, so transposing matrix indices swaps the commutator; residual exactly 0 |
| pair exchange $F_{\mu\nu\alpha\beta}=F_{\alpha\beta\mu\nu}$ | fails | relative residual 1.07 |
| first Bianchi $F_{\mu[\nu\alpha\beta]}=0$ | fails | relative residual 0.31 |

So $F\in\Lambda^2\otimes\Lambda^2$ (36 components in $d=4$) with
distinguishable factors; nothing more may be assumed. Up to sign there is
exactly one nonzero single trace,
$\Phi_{\nu\beta}\equiv\eta^{\mu\alpha}F_{\mu\nu\alpha\beta}$ (no definite
symmetry; measured symmetric/antisymmetric parts 1.76 and 0.95 of
$\lVert\Phi\rVert$), and the double trace defines
$\varphi\equiv\eta^{\nu\beta}\Phi_{\nu\beta}$.

## 2. Enumeration

A quadratic scalar is a complete $\eta$-pairing of the eight slots of
$F\otimes F$: $(8-1)!!=105$ pairings. Any pairing contracting the two slots
of one antisymmetric pair vanishes; by inclusion–exclusion over the four
pairs:

$$
4\cdot15-\binom{4}{2}\cdot3+\binom{4}{3}\cdot1-\binom{4}{4}\cdot1
=60-18+4-1=45 \text{ vanish identically.}
$$

The surviving 60 organize by self-trace structure (no self-traces:
$4+4+16$ bijections; one self-trace each: $2\times16$ via $\Phi$; two each:
$4$ via $\varphi$) into exactly six classes. `enumerate_sympy.py`
canonicalizes all 105 pairings with the **Butler–Portugal algorithm**
(sympy `canon_bp`) — a group-theoretic canonicalization of tensor monomials
returning a unique representative under dummy relabeling and the
sign-carrying slot-symmetry group (both antisymmetries + exchange of the two
identical factors) — and confirms the hand count: 45 zeros, six classes with
multiplicities $4,4,16,16,16,4$.

| invariant | definition | multiplicity | einsum recipe (`Fup` = all indices raised) |
|---|---|---|---|
| $I_1$ | $F_{\mu\nu\alpha\beta}F^{\mu\nu\alpha\beta}$ (current term) | 4 | `('mnab,mnab->',F,Fup)` |
| $I_2$ | $F_{\mu\nu\alpha\beta}F^{\alpha\beta\mu\nu}$ (pair exchange) | 4 | `('mnab,abmn->',F,Fup)` |
| $I_3$ | $F_{\mu\nu\alpha\beta}F^{\mu\alpha\nu\beta}$ (split pairs) | 16 | `('mnab,manb->',F,Fup)` |
| $I_4$ | $\Phi_{\nu\beta}\Phi^{\nu\beta}$ | 16 | `('nb,nb->',Phi,Phiup)` |
| $I_5$ | $\Phi_{\nu\beta}\Phi^{\beta\nu}$ | 16 | `('nb,bn->',Phi,Phiup)` |
| $I_6$ | $\varphi^2$ | 4 | `phi**2` |

## 3. Numerical verification

`check_torch.py` re-evaluates every pairing independently of sympy
(mechanical einsum: eight slot letters + one $\eta$ per metric pair),
float64. Two ensembles: a *generic* tensor with the two antisymmetries, and
*realizable algebraic $F(A)$ tensors* — $F$ built via (1) from four random
symmetric $A_\mu$ (at a point, first derivatives of a smooth symmetric $M$
are unconstrained, so every such $A$ is realizable; no equations of motion,
vacuum spectrum, or boundary conditions are imposed).

- **Class consistency:** all member pairings of each class agree to machine
  precision (worst relative spread $3.3\cdot10^{-15}$ over all 60) — an
  independent cross-implementation check of the canonicalization.
- **Linear independence:** the $60\times6$ sample-value matrix has full rank
  6 in both ensembles (smallest singular value 0.28 after column
  normalization). No linear identity among the six exists — in particular no
  $d=4$ dimension-dependent identity, and no extra degeneracy from the
  commutator structure. $\varphi\not\equiv0$ on realizable fields.
- **Functional independence:** the value-matrix rank cannot exclude
  *nonlinear* relations (e.g. $I_6^2=I_1I_3$); these are excluded by the
  Jacobian criterion — $\partial I_k/\partial x_j$ has full row rank 6 at
  generic points, for $x$ = components of $F$ and of $A_\mu$ alike
  (autograd; $\sigma_{\min}/\sigma_{\max}=6.5\cdot10^{-2}$ and
  $8.1\cdot10^{-2}$).

## 4. Alternative basis: symmetric/antisymmetric channels

The same span, in channels with geometric meaning (identities verified to
$10^{-13}$; suggested in review):
$\Phi=S+A$ gives $\tfrac{I_4\pm I_5}{2}=\langle S,S\rangle,\langle A,A\rangle$;
viewing $F$ as a $6\times6$ matrix on 2-forms and splitting under pair
exchange $F=F_s+F_a$ gives
$\tfrac{I_1\pm I_2}{2}=\langle F_s,F_s\rangle,\langle F_a,F_a\rangle$.
The basis
$\{\lVert F_s\rVert^2,\lVert F_a\rVert^2,I_3,\lVert S\rVert^2,\lVert A\rVert^2,\varphi^2\}$
separates the Riemann-like sector ($F_s$) from the maximally non-Riemann one
($F_a$); norms are $\eta$-indefinite.

## 5. Independent replication and the 3×3 no-go

The substrate-framework campaign
[P239](https://github.com/vantasnerdan/substrate-framework/issues/147)
(attempt 0001, merged in
[PR #148](https://github.com/vantasnerdan/substrate-framework/pull/148),
head `1c63909`) independently derived the same symmetry facts and the same
six-element basis the same day — a genuine two-implementation replication
(our sympy + torch vs. their exact-integer verifier). They additionally
proved, and we independently re-verified with our evaluator
(`verify_3x3_nogo.py`):

1. **Spatial nullspace.** On purely spatial fields the six invariants have
   rank 3; the nullspace is
   $N_1=I_3-\tfrac14(I_1+I_2)$, $N_2=\tfrac14(I_1-I_2)-I_4+I_5$,
   $N_3=I_1-4I_4+I_6$ (our check: identically zero to $5\cdot10^{-16}$).
   The family preserving the working 3×3 action exactly is
   $-I_1+aN_1+bN_2+cN_3$.
2. **No-go.** On the realizable clock direction
   $A_0=\omega\,\mathrm{diag}(1,0,0,0)$, $A_1=E_{01}+E_{10}$ one gets
   $(I_1..I_6)=\omega^2(4,4,2,2,2,4)$, hence $N_1=N_2=N_3=0$ exactly while
   $-I_1=-4\omega^2<0$ (our check: exact). **No choice of $(a,b,c)$ can
   repair the negative clock channel while retaining the full 3×3 action.**

Consequence for the program: the constant-coefficient quadratic route is
closed under exact 3×3 preservation; candidate repairs must be
field-dependent (e.g. internal metrics built from $M$), higher order in $F$,
constrained (fixed-$J$), or must relax exact 3×3 preservation. Credit for
the no-go belongs to P239; this report contributes the independent
verification.

## 6. What this report does not show

- Nothing about parity-odd invariants: only metric pairings are enumerated
  here. (P239 additionally reports four independent one-$\varepsilon$
  pseudoscalars; not verified here.)
- Nothing beyond quadratic order in $F$, and nothing about field-dependent
  coefficients $c_k(M)$ or contractions involving $M$ itself.
- No statement that any of the five new terms improves the physics: the
  no-go (§5) in fact closes the simplest use of them.
- Independence statements are generic-point; fine-tuned configurations can
  make particular $I_k$ coincide.
- The mixed-index invariants $I_3$–$I_6$ rest on the diagonal-Lorentz-action
  assumption (§1), which is the model author's structural choice to confirm.

## 7. Reproduction

```bash
pip install sympy torch          # Python >= 3.12, CPU wheels suffice
./reproduce.sh                   # ~1 min on a laptop
```

`reproduce.sh` regenerates `results/` and asserts every structural claim of
this report (counts 105/45/6, multiplicities, exact antisymmetries, failed
extra symmetries, ranks 6/6 for values and Jacobians, channel identities,
and both no-go checks). Floating-point tails are machine-dependent and are
not asserted.

## Equation-to-code map

| object | code |
|---|---|
| $F(A)$, Eq. (1) | `check_torch.py::physical_F`, `verify_3x3_nogo.py::F_of_A` |
| generic $\Lambda^2\otimes\Lambda^2$ tensor | `check_torch.py::generic_F` |
| all 105 pairings + Butler–Portugal classes | `enumerate_sympy.py` (`matchings`, `canon_bp`) |
| pairing evaluator (einsum + $\eta$'s) | `check_torch.py::contract` |
| symmetry residuals, class spread, value ranks | `check_torch.py` sections 1–3 |
| Jacobian ranks | `check_torch.py::jac_rank` |
| channel identities (§4) | `check_torch.py` section 5 |
| nullspace + counterexample (§5) | `verify_3x3_nogo.py` |

## Provenance

- $F$ definition and conventions: OpenWave
  `openwave/xperiments/m5_liquid_crystal/research/m5_theory_canonical.md` and
  `m5_21_3_note.md` §1 (repo state 2026-08-08, commit `70c8a1bc`).
- Task statement: J. Duda, e-mail 2026-08-20 (models-of-particles), and the
  task note in the working repo.
- P239 claims verified in §5: substrate-framework PR #148, head `1c63909`,
  `proposals/P239-m5-4x4-action/evidence/quadratic-basis-note.md`.
