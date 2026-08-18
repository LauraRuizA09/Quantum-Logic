# ====================== scripts/03_fig3b_rabi.py ==========================
"""Fig. 3B: Rabi flopping. ¿De dónde salen realmente los 118 µs?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.errors import contrast_budget
from qlsim.modes import axial_modes, radial_modes
from qlsim.observables import fit_damped_sine, rabi_flopping
from qlsim.params import schmidt2005
from qlsim.uncertainty import binomial_sigma, monte_carlo, shot_noise

ps = schmidt2005()
tau = 1.0 / (2*np.pi*ps.value("gamma_3p1_hz"))
omega = np.pi / ps.value("t_probe")
t = np.linspace(0, 300e-6, 121)

# eta radial del Al+ (modo espectador Doppler) — origen del Debye-Waller
nmr = radial_modes(ps.value("m_logic"), ps.value("m_spec"),
                   ps.value("nu_z_logic"), ps.value("nu_x_logic"))
from qlsim.modes import dk_single_beam
eta_r = nmr.lamb_dicke(1, 1, dk_single_beam(ps.value("lambda_probe"),
                                            np.deg2rad(45)))
print(f"eta radial (27Al+) = {eta_r:.4f}   tau(3P1) = {tau*1e6:.1f} µs")

c = contrast_budget(tau_decay=tau, t_sequence=ps.value("t_sequence"),
                    nbar_radial=ps.value("nbar_radial"), eta_radial=eta_r,
                    detection_fidelity=ps.value("detection_fidelity"),
                    pi_pulse_error=ps.value("pi_pulse_error"))
p_ideal = rabi_flopping(t, omega_rabi=omega, contrast=c["total"],
                        tau_decay=tau, nbar_radial=ps.value("nbar_radial"),
                        eta_radial=eta_r)
p_data = shot_noise(p_ideal, int(ps.value("n_rep")), seed=11)
sig = binomial_sigma(p_ideal, int(ps.value("n_rep")))

fit = fit_damped_sine(t, p_data, sigma=sig)
print("\nAJUSTE (parametrización del paper: senoide amortiguada)")
for n, v, e in zip(fit["names"], fit["popt"], fit["perr"]):
    print(f"  {n:12s} = {v:9.4f} ± {e:.4f}")
print(f"\n  T_coh ajustado = {fit['popt'][2]:.1f} µs")
print(f"  T_coh reportado = {ps.value('t_coh_reported')*1e6:.0f} ± "
      f"{ps['t_coh_reported'].std*1e6:.0f} µs")

# --- descomposición: apaga causas una a una ---
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax[0].errorbar(t*1e6, p_data, yerr=sig, fmt="ko", ms=3, lw=.8, capsize=1.5,
               label="sintético")
ax[0].plot(t*1e6, p_ideal, "C0-", lw=1.8, label="modelo completo")
ax[0].plot(t*1e6, fit["model"](t, *fit["popt"]), "C3--", lw=1.2,
           label="ajuste senoide amortiguada")
ax[0].plot(t*1e6, rabi_flopping(t, omega_rabi=omega, contrast=c["total"],
                                tau_decay=1e9, nbar_radial=ps.value("nbar_radial"),
                                eta_radial=eta_r), ":", c="C2", lw=1.2,
           label="sin decaimiento espontáneo")
ax[0].plot(t*1e6, rabi_flopping(t, omega_rabi=omega, contrast=c["total"],
                                tau_decay=tau, nbar_radial=1e-9,
                                eta_radial=eta_r), ":", c="C4", lw=1.2,
           label="sin Debye-Waller")
ax[0].set(xlabel="duración del pulso [µs]", ylabel=r"$P_{|{\uparrow}\rangle}$",
          title="Fig. 3B — descomposición del amortiguamiento")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

names = ["decay", "debye_waller", "detection", "pi_pulses",
         "ground_state_cooling"]
vals = [float(np.atleast_1d(c[k])[0]) for k in names]
ax[1].barh(names, [1-v for v in vals], color="C1")
ax[1].set(xlabel="pérdida de contraste  $1-C_i$",
          title=f"Presupuesto: C_total = {float(np.atleast_1d(c['total'])[0]):.3f} "
                f"(reportado {ps.value('contrast_reported'):.2f})")
ax[1].grid(alpha=.3, axis="x")
for i, v in enumerate(vals):
    ax[1].text(1-v, i, f" {1-v:.4f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig("figures/03_fig3b_rabi.png", dpi=160)
print("\n-> figures/03_fig3b_rabi.png")