import capytaine as cpt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 1. CREATE THE BODY
try:
    body = cpt.VerticalCylinder(radius=2.0, length=10.0, center=(0, 0, -2))
except Exception:
    mesh = cpt.meshes.procedural_meshes.create_vertical_cylinder(radius=2.0, length=10.0, center=(0, 0, -2))
    body = cpt.FloatingBody(mesh=mesh)

body.add_all_rigid_body_dofs()
body.keep_only_dofs(['Heave'])

# 2. SOLVE
solver = cpt.BEMSolver()
omega_range = np.linspace(0.1, 4.0, 30)
wave_direction = 0.0

problems = []
for w in omega_range:
    problems.append(cpt.RadiationProblem(body=body, omega=w, radiating_dof="Heave"))
    problems.append(cpt.DiffractionProblem(body=body, omega=w, wave_direction=wave_direction))

print("Solving BEM problems...")
results = solver.solve_all(problems)
dataset = cpt.assemble_dataset(results)

# 3. STRIP ALL COMPLEX TYPES
# This is the "Nuclear Option" to ensure to_netcdf never fails
for var in dataset.data_vars:
    if var == 'excitation_force':
        print("Splitting excitation_force into magnitude and phase...")
        dataset['excitation_mag'] = np.abs(dataset[var])
        dataset['excitation_phase'] = xr.apply_ufunc(np.angle, dataset[var])
    else:
        # For added_mass and radiation_damping, we only need the real part
        print(f"Converting {var} to real...")
        dataset[var] = dataset[var].real

# Drop the original complex excitation force
if 'excitation_force' in dataset.data_vars:
    dataset = dataset.drop_vars('excitation_force')

# Convert coordinates to strings
for coord in dataset.coords:
    if dataset[coord].dtype.name == 'category' or dataset[coord].dtype.kind in 'UO':
        dataset[coord] = dataset[coord].astype(str)

# 4. SAVE
dataset.to_netcdf("hydro_coefficients.nc")
print("Success! All variables converted to real/split and saved.")

# 5. PLOT
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(omega_range, dataset.radiation_damping.sel(radiating_dof="Heave", influenced_dof="Heave"))
plt.title("Radiation Damping (Real)")

plt.subplot(1, 2, 2)
plt.plot(omega_range, dataset.excitation_mag.sel(influenced_dof="Heave", wave_direction="0.0"))
plt.title("Excitation Magnitude")
plt.tight_layout()
plt.show()