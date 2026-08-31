"""Persist the frozen clock tangent a0 (boost-x conjugation direction
on the base profile) as a committed artifact, so the independent
route-2 evaluator takes it as an input instead of re-deriving it
through this stack. One-time producer for APPENDIX-route2.
"""
import json

import numpy as np

from lattice_grid_defs import a0_of, field, gen_catalog, load_or_make_base

with open('results/pre_e4.json') as f:
    pe = json.load(f)
Mr = load_or_make_base()
a0 = a0_of(gen_catalog()[pe['generator']], field(Mr))
np.savez_compressed('results/a0_e4_frozen.npz', a0=a0.cpu().numpy())
print('written: results/a0_e4_frozen.npz, generator', pe['generator'])
