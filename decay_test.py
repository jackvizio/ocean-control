import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PHYSICAL CONSTANTS ---
RHO = 1025      # Density of seawater (kg/m^3)
G = 9.81        # Gravity (m/s^2)
RADIUS = 2.0    # Radius of a cylindrical buoy (m)
AREA = np.pi * RADIUS**2

# --- WEC PARAMETERS ---
mass = 5000     # Mass of the buoy (kg)
added_mass = 2000 # Simplified "Added Mass" from water (kg)
total_mass = mass + added_mass

# Hydrostatic Stiffness (k = rho * g * Area)
k_stiff = RHO * G * AREA

# Damping Coefficient (c) - this determines how fast the motion dies out
damping_c = 1500 # (N s/m)

def physics_engine(t, state):
    """
    state[0] = position z (m)
    state[1] = velocity v (m/s)
    """
    z, v = state

    # 1. Calculate Forces
    f_buoyancy = -k_stiff * z
    f_damping = -damping_c * v

    # 2. Acceleration (F = ma -> a = F/m)
    acceleration = (f_buoyancy + f_damping) / total_mass

    return [v, acceleration]

# --- RUN THE TEST ---
t_span = (0, 30)       # 30 seconds simulation
t_eval = np.linspace(0, 30, 1000)
initial_condition = [1.0, 0] # Lifted 1 meter out of water, released from rest

sol = solve_ivp(physics_engine, t_span, initial_condition, t_eval=t_eval)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Heave Position (z)', color='blue')
plt.axhline(0, color='black', linestyle='--') # Still water level
plt.title("WEC Free Decay Test")
plt.xlabel("Time (s)")
plt.ylabel("Displacement from Equilibrium (m)")
plt.grid(True)
plt.legend()
plt.show()