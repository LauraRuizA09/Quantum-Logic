# =================== scripts/07_heating_budget.py =========================
"""⭐ El script MÁS importante para tu trampa: ¿cuántos pasos aguantas?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.errors import false_positive_rate, max_steps_budget
from qlsim.params import our_trap, schmidt2005

# ⬇️ CAMBIA ESTOS DOS NÚMEROS POR LOS DE TU TRAMPA
OUR_HEATING = (3.0, 0.5)       # cuantos/s, ± 1 sigma
OUR_STEP_TIME = 0.5e-3         # s por paso (pulso + medida + enfriado)

ps = our_trap(schmidt2005(), heating_rate=OUR_HEATING)

nd = np.logspace(-1, 3, 200)
for tau, c in [(0.5e-3, "C0"), (1.5e-3, "C1"), (5e-3, "C2")]:
    p = false_positive_rate(heating_rate=nd, t_sequence=tau,
                            nbar_transfer_after_cooling=0.02,
                            detection_fidelity=0.999)
    plt.loglog(nd, p, c, label=f"$\\tau$ = {tau*1e3:.1f} ms")
    plt.loglog(nd, false_positive_rate(heating_rate=nd, t_sequence=tau,
               nbar_transfer_after_cooling=0.02, detection_fidelity=0.999,
               n_projections=2), c+"--", lw=1,
               label=f"$\\tau$ = {tau*1e3:.1f} ms, 2 proyecciones")
plt.axhline(0.005, color="k", ls=":", label="0.5 % (Chou 2017)")
plt.axvline(10, color="gray", ls=":", label="10 /s (Chou, out-of-phase)")
plt.axvspan(OUR_HEATING[0]-OUR_HEATING[1], OUR_HEATING[0]+OUR_HEATING[1],
            alpha=.2, color="C3", label="NUESTRA trampa")
plt.xlabel(r"tasa de calentamiento $\dot{\bar n}$ [cuantos/s]")
plt.ylabel(r"$P_{\rm FP}$")
plt.title("Falsos positivos: el primer observable que muere")
plt.legend(fontsize=7); plt.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("figures/07_heating_budget.png", dpi=160)

n_lo = max_steps_budget(OUR_HEATING[0]+OUR_HEATING[1], OUR_STEP_TIME, 0.01)
n_hi = max_steps_budget(OUR_HEATING[0]-OUR_HEATING[1], OUR_STEP_TIME, 0.01)
n_mid = max_steps_budget(OUR_HEATING[0], OUR_STEP_TIME, 0.01)
print(f"\n⭐ PRESUPUESTO DE PASOS (P_FP <= 1 %):")
print(f"   N_max = {n_mid:.0f}  [{n_lo:.0f}, {n_hi:.0f}]  (IC 68 %)")
print(f"   -> techo duro para el número de pulsos de la política de RL.")
print(f"   Compara: Chou 2017 usa 13 pulsos por ciclo de barrido;")
print(f"            RL-QLS termina el 85 % de episodios en 83 pulsos (H3O+).")
print("\n-> figures/07_heating_budget.png")