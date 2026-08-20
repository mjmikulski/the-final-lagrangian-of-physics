# the final lagrangian of physics

Public research reports from work on the M5 liquid-crystal particle model
(Jarek Duda's program, simulated in [OpenWave](https://github.com/openwave-labs/openwave)).
The repo name quotes an ambition from the project correspondence; the content is
more modest: one directory per problem, each with a markdown report, the code
behind every number, and a reproduction script.

## Reports

| # | report | headline result |
|---|--------|-----------------|
| 001 | [Quadratic contractions of the M5 field strength](reports/001-quadratic-contractions/) | Exactly 6 independent quadratic invariants; the constant-coefficient extension cannot fix the clock sign while preserving the 3×3 sector (no-go verified) |
| 002 | [Covariant rot/boost split and a finite-frequency clock candidate](reports/002-covariant-split-and-clock/) | Field-selected covariant split (3 gated constructions); one-line covariant sign fix with K⪰0; linear clock terms cancel in H; a quartic boost term keeps a finite ω\* in both Legendre readings, sitting on the Shapere–Wilczek caustic |

## How to reproduce

```bash
pip install -r requirements.txt          # Python >= 3.12
./reports/001-quadratic-contractions/reproduce.sh
```

Each report's `reproduce.sh` regenerates all results and asserts the
structural claims (counts, ranks, identities). Exact floating-point values may
differ in the last digits across machines/BLAS; the asserted structure may not.

## Method

Five rules, borrowed from the best of
[OpenWave](https://github.com/openwave-labs/openwave) and
[substrate-framework](https://github.com/vantasnerdan/substrate-framework)
and nothing else — see [METHOD.md](METHOD.md).

Reports are written by Maciej J. Mikulski with AI assistance (Claude); every
claim is human-owned and script-backed. This is a working record of an ongoing
collaboration, not peer-reviewed publication.
