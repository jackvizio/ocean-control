import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from excitation_system import get_causal_excitation_ss

# 1. Load Data
# Try a simpler, more stable approximation
Ae_c, Be_c, Ce_c, De_c, _ = get_causal_excitation_ss(T_shift=5.0, order=4)
Ae = np.real(Ae_c)
Be = np.real(Be_c).flatten()
Ce = np.real(Ce_c).flatten()
De = float(np.real(De_c))

# --- THE STABILIZER ---
import scipy.linalg as la

# 1. Decompose the system
vals, vecs = la.eig(Ae)

# 2. Force any positive real parts to be negative (Mirroring)
# This keeps the frequency but flips the growth to decay
real_parts = np.real(vals)
imag_parts = np.imag(vals)
stable_real_parts = -np.abs(real_parts) # Force negative
stable_vals = stable_real_parts + 1j*imag_parts

# 3. Reconstruct Ae
Ae = np.real(vecs @ np.diag(stable_vals) @ la.inv(vecs))
# ----------------------

# 2. Test Input (2m Amplitude Wave)
dt = 0.01
t = np.arange(0, 60, dt)
eta = 2.0 * np.sin(0.8 * t)

# 3. Simulate Force
x_e = np.zeros(len(Ae))
fe_history = []

for i in range(len(t)):
    # Standard State-Space: fe = C*x + D*u
    fe = (Ce @ x_e) + De * eta[i]
    # Update state: dx = A*x + B*u
    dx_e = (Ae @ x_e) + Be * eta[i]
    x_e += dx_e * dt
    fe_history.append(fe)

# 4. Check results
print(f"Max Excitation Force: {max(fe_history):.2f} N")
plt.plot(t, fe_history)
plt.title("Excitation Force (f_e) Diagnostic")
plt.ylabel("Force [N]")
plt.show()