import numpy as np
import xarray as xr
from scipy import signal


def get_causal_excitation_ss(filename="hydro_coefficients.nc", T_shift=15.0, order=8):
    ds = xr.open_dataset(filename)
    w = ds.omega.values

    # Reconstruct complex chi
    mag = ds.excitation_mag.sel(influenced_dof="Heave", wave_direction="0.0").values
    phase = ds.excitation_phase.sel(influenced_dof="Heave", wave_direction="0.0").values
    chi = mag * np.exp(1j * phase)

    # Causalize
    chi_causal = chi * np.exp(-1j * w * T_shift)

    # IMPROVED NUMERICAL FITTING (using signal.invres for a cleaner rational fit)
    # We use a frequency-weighted least squares approach
    s = 1j * w
    # Normalizing s to prevent numerical explosion
    s_norm = s / np.max(w)

    # Build a better conditioned matrix for the rational fit
    # System: (b0 + b1*s + ... + bn*s^n) / (1 + a1*s + ... + an*s^n) = chi
    A_mat = []
    for i in range(order + 1):
        A_mat.append(s_norm ** i)
    for i in range(1, order + 1):
        A_mat.append(-chi_causal * (s_norm ** i))

    A_mat = np.column_stack(A_mat)

    # Add weights: Wave energy is usually centered,
    # let's weight the fit toward the peak magnitude
    weights = np.abs(chi_causal) / np.max(np.abs(chi_causal))
    W = np.diag(weights + 0.1)  # 0.1 floor to keep tail info

    sol, _, _, _ = np.linalg.lstsq(W @ A_mat, W @ chi_causal, rcond=None)

    # Rescale the coefficients back from normalized s
    num_norm = sol[:order + 1]
    den_norm = np.concatenate([[1.0], sol[order + 1:]])

    # Convert normalized coeffs back to standard s-domain
    w_max = np.max(w)
    num = [num_norm[i] / (w_max ** i) for i in range(len(num_norm))][::-1]
    den = [den_norm[i] / (w_max ** i) for i in range(len(den_norm))][::-1]

    # Ensure denominator is monic
    num = np.array(num) / den[0]
    den = np.array(den) / den[0]

    # Realize State-Space
    Ae, Be, Ce, De = signal.tf2ss(num, den)

    return Ae, Be, Ce, De, chi_causal