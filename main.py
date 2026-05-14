import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import xarray as xr

# --- EXTERNAL MODULES ---
from system_id import get_radiation_state_space
from excitation_system import get_causal_excitation_ss
from waves import generate_irregular_waves
from controller import WEC_Controller


def stabilize_matrix(A):
    """Ensures radiation states remain stable over long simulations."""
    T, Z = la.schur(A, output='real')
    for i in range(len(T)):
        if T[i, i] > 0: T[i, i] = -T[i, i]
    return Z @ T @ Z.T


def run_simulation():
    # --- 1. PHYSICAL PARAMETERS ---
    Ar_raw, Br_raw, Cr_raw, Dr_raw = get_radiation_state_space()
    Ar = stabilize_matrix(np.real(Ar_raw))
    Br, Cr = np.real(Br_raw).flatten(), np.real(Cr_raw).flatten()
    Dr = float(np.real(Dr_raw).item())

    Ae_raw, Be_raw, Ce_raw, De_raw, _ = get_causal_excitation_ss(T_shift=5.0, order=4)
    Ae = stabilize_matrix(np.real(Ae_raw))
    Be, Ce = np.real(Be_raw).flatten(), np.real(Ce_raw).flatten()
    De = float(np.real(De_raw).item())

    rho, g, radius, draft = 1025.0, 9.81, 0.8, 2.0
    D = 2 * radius
    ds = xr.open_dataset("hydro_coefficients.nc")
    mh = float(np.real(ds.added_mass.sel(radiating_dof="Heave", influenced_dof="Heave").values[-1]))
    M_total = ((np.pi * radius ** 2 * draft) * rho) + mh
    kb = 19700  # Hydrostatic stiffness

    # --- 2. PTO & SAFETY LIMITS ---
    # These prevent the "Crazy" displacement seen in image_9f2974
    Kt, Rs = 150.0, 0.53
    Max_Force = 6e4  # 60kN Saturation limit to protect the hardware

    # --- 3. ENVIRONMENT SETUP ---
    dt = 0.05
    t_sim = np.arange(0, 100, dt)
    Hs, Tp = 2.5, 8.0  # Sea State
    eta_series = generate_irregular_waves(t_sim, Hs=Hs, Tp=Tp)

    # Calculate Theoretical Wave Power for CWR
    p_wave_per_m = (rho * g ** 2 / (64 * np.pi)) * Tp * Hs ** 2
    total_available_kw = (p_wave_per_m * D) / 1000

    ctrl = WEC_Controller(Ar, Br, Cr)
    results = {}

    # --- 4. SIMULATION LOOP ---
    for mode in ['Baseline', 'MRA']:
        x_r, x_e = np.zeros(len(Ar)), np.zeros(len(Ae))
        z, v, error_int = 0.0, 0.0, 0.0

        # Buffer for MRA DB4 Wavelets
        buf_len = 64
        x_buf = [np.zeros(len(Ar))] * buf_len
        w_buf = [0.0] * buf_len

        z_hist, p_elec_hist = [], []
        phi = 0.96  # Leaky integrator to keep the buoy centered at z=0

        for i in range(len(t_sim)):
            fe = (Ce @ x_e) + De * eta_series[i]
            fr = (Cr @ x_r)
            v_ref = ctrl.get_reference(fe)
            error_int = (error_int * phi) + (v - v_ref) * dt

            # Select Control Logic
            if mode == 'Baseline':
                f_ctrl_raw = ctrl.compute_baseline(x_r, error_int)
            else:
                x_buf.append(x_r.copy());
                x_buf.pop(0)
                w_buf.append(error_int);
                w_buf.pop(0)
                f_ctrl_raw = ctrl.compute_mra(np.array(x_buf), np.array(w_buf))

            # APPLY PHYSICAL SATURATION
            f_ctrl = np.clip(f_ctrl_raw, -Max_Force, Max_Force)

            # NET POWER CALCULATION (P_mech - P_copper_loss)
            current_i = f_ctrl / Kt
            p_mech = -f_ctrl * v
            p_loss = Rs * (current_i ** 2)
            p_elec = p_mech - p_loss

            # NUMERICAL INTEGRATION (Forward Euler)
            dv = (fe - fr - (kb * z) - (Dr * v) + f_ctrl) / M_total
            x_e += ((Ae @ x_e) + Be * eta_series[i]) * dt
            x_r += ((Ar @ x_r) + Br * v) * dt
            v += dv * dt
            z += v * dt

            z_hist.append(z)
            p_elec_hist.append(p_elec)

        results[mode] = {'z': z_hist, 'p': p_elec_hist}

    # --- 5. METRICS ---
    start_idx = int(20 / dt)  # Remove transients from the first 20s
    avg_p_base = np.mean(results['Baseline']['p'][start_idx:]) / 1000
    avg_p_mra = np.mean(results['MRA']['p'][start_idx:]) / 1000
    cwr_base = (avg_p_base / total_available_kw) * 100
    cwr_mra = (avg_p_mra / total_available_kw) * 100

    print(f"--- SIMULATION COMPLETE ---")
    print(f"MRA CWR: {cwr_mra:.2f}% | Baseline CWR: {cwr_base:.2f}%")

    # --- 6. PLOTTING ---
    plt.figure(figsize=(12, 10))

    # Displacement Subplot
    plt.subplot(2, 1, 1)
    plt.plot(t_sim, eta_series, color='gray', alpha=0.3, label='Wave Elevation')
    plt.plot(t_sim, results['Baseline']['z'], '--', label='Baseline')
    plt.plot(t_sim, results['MRA']['z'], label='MRA', alpha=0.8)
    plt.title(f'Heave Response (MRA: {cwr_mra:.1f}% CWR | Base: {cwr_base:.1f}% CWR)')
    plt.ylabel('Displacement [m]')
    plt.ylim([-5, 5])  # Keep axis focused on the physics
    plt.legend();
    plt.grid(True)

    # Power Subplot
    plt.subplot(2, 1, 2)
    plt.plot(t_sim, np.array(results['Baseline']['p']) / 1000, '--', alpha=0.6, label='Baseline')
    plt.plot(t_sim, np.array(results['MRA']['p']) / 1000, label='MRA', color='darkorange')
    plt.axhline(y=avg_p_mra, color='darkorange', linestyle=':', label=f'MRA Mean: {avg_p_mra:.2f}kW')
    plt.axhline(y=avg_p_base, color='steelblue', linestyle=':', label=f'Base Mean: {avg_p_base:.2f}kW')
    plt.ylabel('Net Electrical Power [kW]')
    plt.xlabel('Time [s]')
    plt.ylim([-10, 80])  # Prevents negative spikes from ruining the scale
    plt.legend();
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()