import numpy as np

def jonswap_spectrum(w, Hs, Tp, gamma=3.3):
    """
    Calculates the JONSWAP spectral density.
    w: frequency array [rad/s]
    Hs: Significant wave height [m]
    Tp: Peak period [s]
    gamma: Peak enhancement factor (usually 3.3)
    """
    wp = 2 * np.pi / Tp
    sigma = np.where(w <= wp, 0.07, 0.09)

    # Base Pierson-Moskowitz Spectrum
    S_pm = (5 / 16) * (Hs ** 2) * (wp ** 4) * (w ** -5) * np.exp(-1.25 * (w / wp) ** -4)

    # Peak Enhancement
    r = np.exp(-(w - wp) ** 2 / (2 * sigma ** 2 * wp ** 2))
    S_j = S_pm * (gamma ** r)

    return S_j


def generate_irregular_waves(t_sim, Hs, Tp, gamma=3.3, seed=42):
    """
    Acts like WAFO: Converts spectrum to a time-series wave elevation.
    """
    np.random.seed(seed)  # For reproducibility in your paper

    # 1. Define frequency range (0.3 to 3.0 rad/s covers most ocean energy)
    N = 200
    w = np.linspace(0.3, 4.0, N)
    dw = w[1] - w[0]

    # 2. Get spectral density
    S = jonswap_spectrum(w, Hs, Tp, gamma)

    # 3. Random Phase Summation
    # Amplitude for each bin: Ai = sqrt(2 * S(w) * dw)
    eta = np.zeros_like(t_sim)
    phases = np.random.uniform(0, 2 * np.pi, N)

    for i in range(N):
        Ai = np.sqrt(2 * S[i] * dw)
        eta += Ai * np.cos(w[i] * t_sim + phases[i])

    return eta