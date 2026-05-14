import matplotlib.pyplot as plt
from environment import SeaState
from excitation_system import get_causal_excitation_ss # Using the SS fit from earlier

# 1. Setup Sea State (WAFO Style)
sea = SeaState(Hs=2.5, Tp=8.0)
time = np.linspace(0, 200, 2000) # 200 seconds

# 2. Setup Frequency and Elevation
ds = xr.open_dataset("hydro_coefficients.nc")
w = ds.omega.values
eta_up = sea.generate_elevation(w, time)

# 3. Get the Causal System from your Thesis (Eq 2.10)
Ae, Be, Ce, De = get_causal_excitation_ss("hydro_coefficients.nc", T_shift=10.0)

# 4. Simulate the Filter: input(eta) -> output(force)
# This is much faster and more accurate than manual summing
tout, fe, x = signal.lsim((Ae, Be, Ce, De), U=eta_up, T=time)

# 5. Plot the result
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(time, eta_up, color='teal')
ax1.set_ylabel("Wave Elevation [m]")
ax1.set_title("Input: Up-wave Surface Elevation")

ax2.plot(time, fe, color='darkblue')
ax2.set_ylabel("Excitation Force [N]")
ax2.set_xlabel("Time [s]")
ax2.set_title("Output: Causalized Excitation Force (Thesis Eq 2.10)")

plt.tight_layout()
plt.show()