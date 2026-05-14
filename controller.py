import numpy as np
import scipy.linalg as la
import pywt


class WEC_Controller:
    def __init__(self, Ar, Br, Cr):
        # Physical constants from Table I
        self.K_hat = 23800
        self.alpha_r = 1.2

        # Reduced levels to 2 to avoid boundary effect warnings with small buffers
        self.levels = 2

        # Augmented System Construction (State + Integrator)
        n = Ar.shape[0]
        self.A_aug = np.zeros((n + 1, n + 1))
        self.A_aug[:n, :n] = Ar
        self.A_aug[n, :n] = Cr
        self.B_aug = np.zeros((n + 1, 1))
        self.B_aug[:n, 0] = Br

        # 1. BASELINE TUNING (Standard Industrial LQT benchmark)
        Q_base = np.eye(n + 1) * 1.0
        Q_base[n, n] = 1e5
        # In main.py inside the controller initialization
        R_base = np.array([[1.0]])  # Increase Baseline cost slightly

        try:
            P_base = la.solve_continuous_are(self.A_aug, self.B_aug, Q_base, R_base)
            self.K_baseline = la.inv(R_base) @ (self.B_aug.T @ P_base)
        except la.LinAlgError:
            self.K_baseline = np.zeros((1, n + 1))

        # 2. MRA TUNING (The "Efficiency-First" Strategy)
        # Band 0: Approximation (Large Waves) -> Extremely Aggressive
        # Band 1 & 2: Details (High-Freq Jitter) -> High Penalty to save electricity
        self.band_gains = []

        # controller.py
        # Band 0: Wave energy | Band 1: Noise | Band 2: Jitter
        q_weights = [1e6, 1e2, 1e1]
        r_weights = [5e-3, 1e0, 1e1]  # Changed from 1e-6 to 5e-3

        for i in range(len(q_weights)):
            Qi = np.eye(n + 1) * 1.0
            Qi[n, n] = q_weights[i]
            Ri = np.array([[r_weights[i]]])

            try:
                Pi = la.solve_continuous_are(self.A_aug, self.B_aug, Qi, Ri)
                Ki = la.inv(Ri) @ (self.B_aug.T @ Pi)
                self.band_gains.append(Ki)
            except la.LinAlgError:
                self.band_gains.append(self.K_baseline)

    def get_reference(self, fe):
        """Calculates optimal velocity reference based on excitation force."""
        return (1 / (2 * self.alpha_r * self.K_hat)) * fe

    def compute_baseline(self, x_r, error_int):
        """Standard LQT control using the unified gain matrix."""
        x_aug = np.append(x_r, error_int)
        return -float(self.K_baseline @ x_aug)

    def compute_mra(self, x_history, w_history):
        """
        Multiresolution Analysis Control.
        Decomposes states into bands to apply frequency-specific optimization.
        """
        # Wavelet decomposition using db4
        # We use mode='periodization' to reduce edge artifacts further
        # controller.py - inside compute_mra
        coeffs_x = pywt.wavedec(x_history, 'db4', level=self.levels, axis=0, mode='reflect')
        coeffs_w = pywt.wavedec(w_history, 'db4', level=self.levels, axis=0, mode='reflect')

        total_force = 0
        for j in range(len(coeffs_x)):
            # Use the most recent coefficient from the current decomposition level
            x_b = coeffs_x[j][-1]
            w_b = coeffs_w[j][-1]
            x_aug_b = np.append(x_b, w_b)

            total_force += -float(self.band_gains[j] @ x_aug_b)

        return total_force