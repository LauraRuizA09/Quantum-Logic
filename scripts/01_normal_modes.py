# ====================== scripts/01_normal_modes.py ========================
"""Modos normales 9Be+/27Al+ y parámetros de Lamb-Dicke, con incertidumbre."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.modes import axial_modes, dk_single_beam, radial_modes
from qlsim.params import schmidt2005
from qlsim.uncertainty import monte_carlo

ps = schmidt2005()

# --- valor central ---
nm = axial_modes(ps.value("m_logic"), ps.value("m_spec"), ps.value("nu_z_logic"))
print("MODOS AXIALES (9Be+ / 27Al+)")
print(nm.describe())

dk_s = dk_single_beam(ps.value("lambda_probe"), ps.value("theta_probe"))
dk_l = ps.value("dk_raman_logic")
for p, lbl in enumerate(nm.labels):
    print(f"\n  {lbl}:")
    print(f"    eta(27Al+, sonda 267nm@45deg) = {nm.lamb_dicke(1, p, dk_s):.4f}")
    print(f"    eta(9Be+,  Raman 313nm)       = {nm.lamb_dicke(0, p, dk_l):.4f}  ⚠️ dk ASSUMED")

nmr = radial_modes(ps.value("m_logic"), ps.value("m_spec"),
                   ps.value("nu_z_logic"), ps.value("nu_x_logic"))
print("\nMODOS RADIALES (x)")
print(nmr.describe())

# --- incertidumbre por Monte Carlo ---
def f(**d):
    out = []
    for i in range(len(d["nu_z_logic"])):
        m = axial_modes(d["m_logic"][i], d["m_spec"][i], d["nu_z_logic"][i])
        out.append([m.nu[0], m.nu[1],
                    m.lamb_dicke(1, 0, dk_single_beam(d["lambda_probe"][i],
                                                      d["theta_probe"][i]))])
    return np.array(out)

mc = monte_carlo(f, ps, n=2000)
names = ["nu_in-phase [MHz]", "nu_out-of-phase [MHz]", "eta(Al+, in-phase)"]
scales = [1e-6, 1e-6, 1]
print("\n=== RESULTADOS CON INCERTIDUMBRE (mediana, IC 68%) ===")
for j, (n, s) in enumerate(zip(names, scales)):
    print(f"  {n:26s} = {mc['median'][j]*s:.5f} "
          f"[+{(mc['hi68'][j]-mc['median'][j])*s:.5f} "
          f"-{(mc['median'][j]-mc['lo68'][j])*s:.5f}]")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for j, (n, s) in enumerate(zip(names, scales)):
    ax[j].hist(mc["samples"][:, j] * s, bins=60, color="C0", alpha=0.8)
    ax[j].axvline(mc["median"][j] * s, color="k", lw=1.5)
    ax[j].set_xlabel(n); ax[j].set_ylabel("cuentas MC")
fig.suptitle("Propagación de incertidumbre: modos normales y Lamb-Dicke")
fig.tight_layout(); fig.savefig("figures/01_normal_modes.png", dpi=160)
print("\n-> figures/01_normal_modes.png")