import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import xarray as xr
from excitation_system import get_causal_excitation_ss

# 1. PARAMETERS
T_shift = 15.0  # The causalization delay from your thesis
order = 10      # State-space model order

# 2. GET DATA
# Ae, Be, Ce, De: The state-space approximation
# chi_target: The complex frequency response (BEM data * e^-jwt)
Ae, Be, Ce, De, chi_target = get_causal_excitation_ss(T_shift=T_shift, order=order)

# 3. RECONSTRUCT SS FREQUENCY RESPONSE
# We evaluate the resulting system across a fine frequency range
w_fine = np.linspace(0.1, 4.0, 500)
# Use signal.freqresp to get the complex response of the (A,B,C,D) system
_, chi_approx = signal.freqresp((Ae, Be, Ce, De), w=w_fine)

# 4. PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# MAGNITUDE COMPARISON
# We need the original frequencies for the scatter plot
ds = xr.open_dataset("hydro_coefficients.nc")
w_bem = ds.omega.values

ax1.plot(w_bem, np.abs(chi_target), 'ro', label='Target (BEM + Shift)')
ax1.plot(w_fine, np.abs(chi_approx), 'b-', label=f'SS Fit (Order {order})')
ax1.set_ylabel('Magnitude [N/m]')
ax1.set_title('Excitation Force: State-Space Verification')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PHASE COMPARISON
ax2.plot(w_bem, np.angle(chi_target), 'ro', label='Target')
ax2.plot(w_fine, np.angle(chi_approx), 'b-', label='SS Fit')
ax2.set_ylabel('Phase [rad]')
ax2.set_xlabel('Frequency $\omega$ [rad/s]')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()