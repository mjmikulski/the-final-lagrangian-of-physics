"""The rigidly rotating branch at fixed angular momentum, re-evaluated on the CPU from the committed fields.

For each point of the branch (results/fields/spinning_base_n32.pt, found on the GPU by run_spinning.py) the
rigid-rotor functional L(Omega) = a Omega^2 + b Omega + c is evaluated exactly (the Lagrangian is quadratic
in the time derivative), giving Omega = (J - b) / (2a), E(J) = (J - b)^2 / (4a) - c, the static energy -c of
the deformed shape, its split into the kinetic part (J - b)^2 / (4a) and the deformation energy -c - E_static(0), and the
inertia 2a of the shape (equal to J^2 / (2 E_kin)). Also the rigid inertia of the undeformed hedgehog and its rotation
invariance (the rotation tangent vanishes, so a = 0 up to lattice symmetry breaking).

python rotor_branch.py            # writes results/rotor_branch.json
"""
import json
import torch
from soliton import Grid, to_matrix

torch.set_default_dtype(torch.float64)
base = torch.load('results/fields/static_base_n32.pt')
grid = Grid(base['n'], base['box'], E=base['E'], device='cpu')
u0 = base['u'].double()
with torch.no_grad():
    E0 = float(grid.energy(u0))
    a0, b0, c0 = (float(t) for t in grid.rotor(u0))
print(f'static hedgehog: E = {E0:.6f}, rigid inertia 2a = {2 * a0:.4f}')
br = torch.load('results/fields/spinning_base_n32.pt')['branch']
rows = []
for pt in br:
    u = pt['u'].double()
    with torch.no_grad():
        a, b, c = (float(t) for t in grid.rotor(u))
        J = pt['J']
        Om = (J - b) / (2 * a)
        E = (J - b) ** 2 / (4 * a) - c
        Mv = to_matrix(u)
        row = dict(J=J, a=a, b=b, E_static_shape=-c, E=E, E_above_static=E - E0, E_kin=(J - b) ** 2 / (4 * a), E_def=-c - E0,
                   Omega=Om, inertia_shape=2 * a, E_over_Omega=E / Om, grad_at_end=pt['grad'])
        # centre of the potential energy density (displacement of the core off the axis)
        M, dM = grid.derivatives(u)
        from lagrangian import terms
        kin, pot, x = terms(M, dM, grid.E, frozen=True)
        row['centre_potential'] = ((pot[..., None] * grid.xq).sum((0, 1, 2, 3)) / pot.sum()).tolist()
    rows.append(row)
    print(f"J = {J:5.1f}: E - E_static = {row['E_above_static']:.5f} = kinetic {row['E_kin']:.5f} + deformation {row['E_def']:.5f}, Omega = {Om:.5f}, "
          f"inertia 2a = {2 * a:.1f} ({2 * a / (2 * a0):.0f} x rigid), E/Omega = {row['E_over_Omega']:.0f} (2J = {2 * J}), |grad| {pt['grad']:.1e}", flush=True)
# the halo-tilt alternative in the same box: a tilt that spans the box radius R costs c_E delta^2 / R and carries the
# inertia c_I delta^2 R (halo_check.py); at fixed J the optimal angle gives E_tilt = J sqrt(2 c_E / c_I) / R = J / R.
# The box branch beats it unless R > J / E_kin, which is the crossover radius quoted in the report.
import math
Delta = base['E'][1] - base['E'][2]
cE, cI = 32 * math.pi / 3 * Delta ** 4, 64 * math.pi / 3 * Delta ** 4
Rbox = base['box'] / 2
for row in rows:
    row['tilt_estimate_in_box'] = row['J'] * math.sqrt(2 * cE / cI) / Rbox
    row['crossover_radius'] = row['J'] * math.sqrt(2 * cE / cI) / row['E_kin']
    print(f"J = {row['J']:5.1f}: halo-tilt estimate in this box {row['tilt_estimate_in_box']:.2f} vs branch kinetic {row['E_kin']:.4f}; tilt wins beyond R = {row['crossover_radius']:.0f}")
json.dump(dict(E_static=E0, rigid_inertia=2 * a0, box=base['box'], n=base['n'], rows=rows), open('results/rotor_branch.json', 'w'), indent=1)
