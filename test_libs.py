import numpy as np
import scipy.linalg as la
from system_id import get_radiation_state_space
print("--- Step 1: System ID Library Loaded ---")

try:
    Ar, Br, Cr, Dr = get_radiation_state_space()
    print("SUCCESS: Radiation State-Space generated.")
except Exception as e:
    print(f"FAILED at Radiation: {e}")

from excitation_system import get_causal_excitation_ss
print("--- Step 2: Excitation Library Loaded ---")

try:
    Ae, Be, Ce, De, _ = get_causal_excitation_ss(T_shift=5.0, order=4)
    print("SUCCESS: Excitation State-Space generated.")
except Exception as e:
    print(f"FAILED at Excitation: {e}")