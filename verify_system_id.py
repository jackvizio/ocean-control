import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from system_id import get_radiation_state_space

# 1. Extract Matrices (Now 4th Order)
print("Calculating 4th-Order State-Space...")
A, B, C, D = get_radiation_state_space("hydro_coefficients.nc")

# 2. Load BEM Data
ds = xr.open_dataset("hydro_coefficients.nc")
w_bem = ds.omega.values
B_bem = ds.radiation_damping.sel(radiating_dof="Heave", influenced_dof="Heave").values

# 3. Frequency Response Calculation
# We calculate the Real part of the transfer function: Re{ C * inv(jwI - A) * B + D }
w_fine = np.linspace(w_bem.min(), w_bem.max(), 300)
B_fit = []

# Identity matrix for the calculation
I = np.eye(A.shape[0])

for freq in w_fine:
    # H(jw) = C @ inv(jwI - A) @ B + D
    s = 1j * freq
    resp = C @ np.linalg.inv(s * I - A) @ B + D
    # We take the real part because radiation damping is the dissipative component
    B_fit.append(np.real(resp[0,0]))

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_bem, B_bem, 'ro', markersize=6, label='BEM Data (Capytaine)')
plt.plot(w_fine, B_fit, 'b-', linewidth=2, label='4th-Order State-Space Fit')

plt.title("Radiation Damping Verification (Multi-Resonant Fit)", fontsize=14)
plt.xlabel(r"Frequency $\omega$ [rad/s]", fontsize=12)
plt.ylabel(r"Damping $B(\omega)$ [N·s/m]", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# Show the fit quality in the terminal
print(f"Max BEM value: {np.max(B_bem):.2f}")
print(f"Max Fit value: {np.max(B_fit):.2f}")

print("Displaying plot...")
plt.show()