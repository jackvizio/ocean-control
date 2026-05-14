import numpy as np
import xarray as xr
from scipy import signal


class SeaState:
    def __init__(self, Hs, Tp, gamma=3.3):
        self.Hs = Hs
        self.Tp = Tp
        self.wp = 2 * np.pi / Tp
        self.gamma = gamma
        self.g = 9.81

    def get_spectrum(self, w):
        """Calculates JONSWAP PSD based on Thesis Eq 2.11"""
        # alpha is chosen so the integral of the spectrum matches Hs
        alpha = 0.0081 * (self.g ** 2)  # Simplified base alpha

        sigma = np.where(w < self.wp, 0.07, 0.09)
        r = np.exp(-(w - self.wp) ** 2 / (2 * sigma ** 2 * self.wp ** 2))

        # Pierson-Moskowitz Part
        S = (alpha / w ** 5) * np.exp(-1.25 * (self.wp / w) ** 4)
        # JONSWAP enhancement
        S = S * (self.gamma ** r)

        # Correct alpha to match target Hs: Hs = 4 * sqrt(m0)
        m0_current = np.trapz(S, w)
        S = S * (self.Hs / (4 * np.sqrt(m0_current))) ** 2
        return S

    def generate_elevation(self, w, time):
        S = self.get_spectrum(w)
        dw = np.gradient(w)
        A = np.sqrt(2 * S * dw)
        phases = np.random.uniform(0, 2 * np.pi, len(w))

        eta = np.zeros_like(time)
        for i, t in enumerate(time):
            eta[i] = np.sum(A * np.cos(w * t + phases))
        return eta