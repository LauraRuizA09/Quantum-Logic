# ==================== scripts/02_fig3a_spectrum.py ========================
"""Réplica de la Fig. 3A: espectroscopía Rabi + verificación del 63 kHz."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.errors import contrast_budget
from qlsim.observables import fit_spectrum, spectrum
from qlsim.params import schmidt2005
from qlsim.pulses import rabi_fwhm_hz, sinc2_lineshape
from qlsim.uncertainty import binomial_sigma, monte_carlo, shot_noise

ps = schmidt2005()
t_p = ps.value("t_probe")
omega = np.pi / t_p                      # pulso pi por construcción
n_rep = int(ps.value("n_rep"))
detun = np.linspace(-150e3, 150e3, 61)

# --- 1. Predicción central + ruido de disparo (=datos sintéticos) ---
p_ideal = spectrum(detun, omega_rabi=omega, t_probe=t_p,
                   contrast=ps.value("contrast_reported"))
p_data = shot_noise(p_ideal, n_rep, seed=42)
sig = binomial_sigma(p_ideal, n_rep)

# --- 2. Banda de incertidumbre por Monte Carlo de los parámetros ---
def f(**d):
    c = contrast_budget(**d)["total"]
    om = np.pi / d["t_probe"]
    return spectrum(detun, omega_rabi=om, t_probe=d["t_probe"], contrast=c)

mc = monte_carlo(f, ps, n=1500, seed=7)

# --- 3. Ajuste de los "datos" y comparación con lo reportado ---
fit = fit_spectrum(detun, p_data, sigma=sig)
print("AJUSTE DE LA FIG. 3A")
for n, v, e in zip(fit["names"], fit["popt"], fit["perr"]):
    print(f"  {n:14s} = {v:10.4f} ± {e:.4f}")
print(f"\n  FWHM ajustado    = {fit['fwhm']/1e3:.2f} kHz")
print(f"  FWHM analítico   = {rabi_fwhm_hz(t_p)/1e3:.2f} kHz  (= 0.7993/t_pi)")
print(f"  FWHM reportado   = {ps.value('fwhm_reported')/1e3:.1f} ± "
      f"{ps['fwhm_reported'].std/1e3:.1f} kHz")
z = (fit["fwhm"] - ps.value("fwhm_reported")) / ps["fwhm_reported"].std
print(f"  z-score vs paper = {z:+.2f}  "
      f"({'✅ compatible' if abs(z) < 2 else '❌ discrepante'})")

# --- 4. Figura ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].fill_between(detun/1e3, mc["lo95"], mc["hi95"], alpha=.18, color="C0",
                   label="modelo, IC 95 % (params)")
ax[0].fill_between(detun/1e3, mc["lo68"], mc["hi68"], alpha=.35, color="C0",
                   label="modelo, IC 68 %")
ax[0].plot(detun/1e3, mc["median"], "C0-", lw=1.6, label="modelo, mediana")
ax[0].errorbar(detun/1e3, p_data, yerr=sig, fmt="ko", ms=3.5, lw=.9,
               capsize=1.8, label=f"sintético, N={n_rep}/punto")
ax[0].axhline(p_ideal.max()/2, ls=":", c="gray", lw=1)
ax[0].set(xlabel="desintonización de la sonda [kHz]",
          ylabel=r"$P_{|{\uparrow}\rangle,\,^{27}\!Al^+}$",
          title=f"Fig. 3A — FWHM = {fit['fwhm']/1e3:.1f} kHz")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

# comparación de formas de línea (ejercicio E3)
ax[1].plot(detun/1e3, p_ideal/p_ideal.max(), label="Rabi exacta (pulso $\\pi$)")
ax[1].plot(detun/1e3, sinc2_lineshape(detun*2*np.pi, t_p), "--",
           label="$\\mathrm{sinc}^2$ (área pequeña)")
ax[1].set(xlabel="desintonización [kHz]", ylabel="normalizado",
          title="¿Por qué importa el modelo de forma de línea?")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_yscale("log")
ax[1].set_ylim(1e-4, 2)
fig.tight_layout(); fig.savefig("figures/02_fig3a_spectrum.png", dpi=160)
print("\n-> figures/02_fig3a_spectrum.png")