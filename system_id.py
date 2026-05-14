import numpy as np
import xarray as xr
from scipy import signal
from scipy.optimize import curve_fit


def radiation_model(w, a, b, c):
    """Stable 2nd-order radiation damping model."""
    return (a * w ** 2) / ((w ** 2 - b) ** 2 + c * w ** 2)


def get_radiation_state_space(filename="hydro_coefficients.nc"):
    ds = xr.open_dataset(filename)
    w = ds.omega.values
    B_w = ds.radiation_damping.sel(radiating_dof="Heave", influenced_dof="Heave").values

    # Fit the physical curve
    p0 = [np.max(B_w), w[np.argmax(B_w)] ** 2, 1.0]
    popt, _ = curve_fit(radiation_model, w, B_w, p0=p0)

    # Convert to State-Space: H(s) = (a_scale * s) / (s^2 + sqrt(c)*s + b)
    # This is a standard 'Radiation Approximation'
    num = [popt[0] / np.sqrt(popt[2]), 0]
    den = [1, np.sqrt(popt[2]), popt[1]]

    A, B, C, D = signal.tf2ss(num, den)
    return A, B, C, D