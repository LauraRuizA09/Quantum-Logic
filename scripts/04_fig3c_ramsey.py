# ===================== scripts/04_fig3c_ramsey.py =========================
"""Fig. 3C: el cúbit MOTIONAL bate a nu_mode -> calibración de nu_m."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.modes import axial_modes
from qlsim.observables import fit_ramsey, ramsey_motional
from qlsim.params import schmidt2005
from qlsim.uncertainty import binomial_sigma, shot_noise

ps = schmidt2005()
nm = axial_modes(ps.value("m_logic"), ps.value("m_spec"), ps.value("nu_z_logic"))
nu_m = nm.nu[0]                        # modo in-phase = modo de transferencia
print(f"nu_mode (in-phase, DERIVADO) = {nu_m/1e6:.5f} MHz")

t = np.linspace(0, 2.2e-6, 200)        # ~5 periodos
p_ideal = ramsey_motional(t, nu_mode=nu_m, contrast=0.9, t2_motion=500e-6)
p_data = shot_noise(p_ideal, int(ps.value("n_rep")), seed=5)
sig = binomial_sigma(p_ideal, int(ps.value("n_rep")))

fit = fit_ramsey(t, p_data, nu_m, sigma=sig)
nu_fit, dnu = fit["popt"][0]*1e6, fit["perr"][0]*1e6
print(f"nu_mode recuperado del ajuste = {nu_fit/1e6:.6f} ± {dnu/1e6:.6f} MHz")
print(f"precisión relativa = {dnu/nu_fit:.2e}  <- por esto Fig.3C calibra nu_m")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax[0].errorbar(t*1e6, p_data, yerr=sig, fmt="ko", ms=3, lw=.7, capsize=1.2)
ax[0].plot(t*1e6, fit["model"](t, *fit["popt"]), "C0-", lw=1.6)
ax[0].set(xlabel=r"tiempo de espera $t_d$ [µs]", ylabel=r"$P_{|{\downarrow}\rangle,\,^{9}Be^+}$",
          title=f"Fig. 3C — franjas a $\\nu_m$ = {nu_fit/1e6:.4f} MHz")
ax[0].grid(alpha=.3)

# ¿qué pasa si nu_m deriva y no lo corriges? -> pulsos de banda lateral fuera de resonancia
drift = np.linspace(-20e3, 20e3, 200)
from qlsim.pulses import rabi_lineshape
eta = nm.lamb_dicke(1, 0, 2*np.pi/267e-9*np.cos(np.deg2rad(45)))
om_sb = eta * np.pi / ps.value("t_probe")
t_pi_sb = np.pi/om_sb
ax[1].plot(drift/1e3, rabi_lineshape(drift*2*np.pi, om_sb, t_pi_sb))
ax[1].set(xlabel=r"deriva de $\nu_m$ [kHz]",
          ylabel="fidelidad del pulso $\\pi$ de banda lateral",
          title="Coste de NO calibrar $\\nu_m$")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/04_fig3c_ramsey.png", dpi=160)
print("\n-> figures/04_fig3c_ramsey.png")