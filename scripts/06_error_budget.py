"""06 — PRESUPUESTO DE ERROR Y ANÁLISIS INVERSO.

Éste es el script más formativo del repo. Hace tres cosas:

  (A) ANÁLISIS INVERSO. El paper reporta contraste = 0.93 y T_coh = 118 µs,
      pero NO reporta ni el tiempo de exposición al estado 3P1 ni el tiempo
      de defasaje. Aquí INVERTIMOS el modelo: ¿qué valores de `t_exposure` y
      `t2_dephasing` reproducen las observables publicadas?
      ⭐ Esto convierte dos parámetros ASSUMED en parámetros CONSTRAINED,
      que es lo máximo que se puede hacer sin datos crudos.

  (B) DESCOMPOSICIÓN DEL CONTRASTE en factores multiplicativos, cada uno con
      su incertidumbre por Monte Carlo. Nunca reportes "coincide con el 93 %":
      reporta el producto y DI CUÁL DOMINA.

  (C) DIAGRAMA DE TORNADO (sensibilidad de primer orden).
      ⭐ El ranking te dice QUÉ MEDIR PRIMERO en el laboratorio.

Referencia: Schmidt et al., Science 309, 749 (2005). DOI 10.1126/science.1114375
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import qlsim
from qlsim.errors import contrast_budget, contrast_total, error_ranking
from qlsim.observables import (debye_waller_factor, fit_damped_sine,
                              rabi_flopping, thermal_weights)
from qlsim.uncertainty import (binomial_sigma, monte_carlo, sensitivity,
                               shot_noise)

ps = qlsim.default_paramset()
V = ps.values()

TAU = ps.value("tau_decay")
ETA_R = ps.value("eta_radial")
NBAR_R = ps.value("nbar_radial")
OMEGA = np.pi / ps.value("t_probe")
C_TARGET = ps.value("contrast_reported")
T_COH_TARGET = ps.value("t_coh_reported")
N_REP = int(ps.value("n_rep"))


# =====================================================================
# (A) ANÁLISIS INVERSO
# =====================================================================
print(qlsim.banner("(A) ANÁLISIS INVERSO — deducir lo que el paper no reporta"))

print(f"Observables PUBLICADAS que usamos como restricción:")
print(f"   contraste = {C_TARGET:.3f} ± {ps.std('contrast_reported'):.3f}")
print(f"   T_coh     = {T_COH_TARGET*1e6:.0f} ± {ps.std('t_coh_reported')*1e6:.0f} µs")
print(f"\nParámetros FIJOS (reportados o derivados):")
print(f"   tau(3P1)     = {TAU*1e6:.1f} µs   [DERIVED de Gamma/2pi = 520 Hz]")
print(f"   eta_radial   = {ETA_R:.4f}        [DERIVED de modos normales]")
print(f"   nbar_radial  = {NBAR_R:.2f}       [⚠️ ASSUMED: límite Doppler]")


# ---- A1: ¿qué t_exposure reproduce el contraste de 0.93? ----
def contrast_of_texp(t_exp: float) -> float:
    kw = dict(V)
    kw["t_exposure"] = t_exp
    return contrast_total(**kw)


c_lo = contrast_of_texp(ps.value("t_probe"))        # cota superior de contraste
c_hi = contrast_of_texp(5 * TAU)                    # cota inferior
print(f"\n--- A1: inversión de `t_exposure` ---")
print(f"   rango alcanzable de contraste: [{c_hi:.4f}, {c_lo:.4f}]")

if c_hi < C_TARGET < c_lo:
    t_exp_fit = brentq(lambda x: contrast_of_texp(x) - C_TARGET,
                       ps.value("t_probe"), 5 * TAU, xtol=1e-10)
    # propaga la incertidumbre del contraste reportado a t_exposure
    s = ps.std("contrast_reported")
    t_lo = brentq(lambda x: contrast_of_texp(x) - (C_TARGET + s),
                  ps.value("t_probe"), 5 * TAU, xtol=1e-10)
    t_hi = brentq(lambda x: contrast_of_texp(x) - (C_TARGET - s),
                  ps.value("t_probe"), 5 * TAU, xtol=1e-10)
    print(f"   t_exposure inferido = {t_exp_fit*1e6:.2f} µs "
          f"[{min(t_lo,t_hi)*1e6:.2f}, {max(t_lo,t_hi)*1e6:.2f}] (IC 68 %)")
    print(f"   valor ASSUMED en params.py = {ps.value('t_exposure')*1e6:.1f} µs")
    print(f"   ✅ CONSISTENTE" if abs(t_exp_fit - ps.value("t_exposure"))
          < 3 * ps.std("t_exposure") else "   ⚠️ actualiza params.py")
    print(f"\n   INTERPRETACIÓN FÍSICA: t_exposure ≈ {t_exp_fit/ps.value('t_probe'):.1f} "
          f"× t_probe. Coherente con t_probe (12.6 µs) MÁS los dos pulsos pi "
          f"de banda lateral del mapeo, cuya duración el paper no reporta.")
else:
    t_exp_fit = ps.value("t_exposure")
    print(f"   ❌ el contraste objetivo cae FUERA del rango alcanzable.")
    print(f"      ==> algún otro factor domina: revisa nbar_radial o eta_radial.")
    print(f"      Esto NO es un bug: es información. Significa que el modelo "
          f"con los ASSUMED actuales no puede explicar el 0.93.")


# ---- A2: ¿qué t2_dephasing reproduce T_coh = 118 µs? ----
print(f"\n--- A2: inversión de `t2_dephasing` ---")
t_grid = np.linspace(0, 300e-6, 121)


def fitted_tcoh(t2: float) -> float:
    """T_coh que se OBTENDRÍA al ajustar una senoide amortiguada al modelo."""
    y = rabi_flopping(t_grid, omega_rabi=OMEGA, contrast=C_TARGET,
                      tau_decay=TAU, nbar_radial=NBAR_R, eta_radial=ETA_R,
                      t2_dephasing=t2)
    try:
        f = fit_damped_sine(t_grid, y, omega_guess_khz=OMEGA / (2 * np.pi * 1e3),
                            t_coh_guess_us=T_COH_TARGET * 1e6)
        return f["popt"][2] * 1e-6
    except RuntimeError:
        return np.nan


t2_scan = np.array([60e-6, 100e-6, 150e-6, 200e-6, 300e-6, 500e-6, 1e-3, np.inf])
tc_scan = np.array([fitted_tcoh(x) for x in t2_scan])
for a, b in zip(t2_scan, tc_scan):
    lab = "inf" if not np.isfinite(a) else f"{a*1e6:6.0f} µs"
    print(f"   t2 = {lab:>9s}  ->  T_coh ajustado = {b*1e6:6.1f} µs")

try:
    t2_fit = brentq(lambda x: fitted_tcoh(x) - T_COH_TARGET, 40e-6, 5e-3,
                    xtol=1e-9)
    print(f"\n   t2_dephasing inferido = {t2_fit*1e6:.1f} µs")
    print(f"   valor ASSUMED en params.py = {ps.value('t2_dephasing')*1e6:.0f} µs")
except ValueError:
    t2_fit = np.nan
    tc_inf = fitted_tcoh(np.inf)
    print(f"\n   ⚠️ Sin defasaje adicional (t2 = inf) el modelo ya da "
          f"T_coh = {tc_inf*1e6:.1f} µs.")
    if tc_inf < T_COH_TARGET:
        print(f"      Es MENOR que los {T_COH_TARGET*1e6:.0f} µs reportados: "
              f"el decaimiento espontáneo + Debye-Waller ya sobre-amortiguan.")
        print(f"      ==> nuestro nbar_radial ASSUMED ({NBAR_R:.2f}) es "
              f"probablemente DEMASIADO ALTO. Este es exactamente el tipo de "
              f"conclusión que justifica medirlo en el laboratorio.")
    else:
        print(f"      Es MAYOR: hace falta defasaje, pero fuera del rango "
              f"buscado. Amplía los límites de brentq.")

print(f"\n⭐ LECCIÓN: el análisis inverso convierte parámetros ASSUMED en "
      f"CONSTRAINED, y cuando NO converge te dice qué hipótesis está mal.\n"
      f"   Ambos resultados son publicables; el segundo es más valioso.")


# =====================================================================
# (B) DESCOMPOSICIÓN DEL CONTRASTE CON INCERTIDUMBRE
# =====================================================================
print(qlsim.banner("(B) PRESUPUESTO DE CONTRASTE (Monte Carlo)"))

FACTORS = ["decay", "debye_waller", "detection", "pi_pulses",
           "ground_state_cooling"]


def budget_vector(**d):
    b = contrast_budget(**d)
    return np.column_stack([np.atleast_1d(b[k]) for k in FACTORS]
                           + [np.atleast_1d(b["total"])])


mc = monte_carlo(budget_vector, ps, n=4000, seed=17)
names = FACTORS + ["TOTAL"]

print(f"{'factor':>22s} {'C_i (mediana)':>15s} {'IC 68 %':>22s} "
      f"{'pérdida 1-C_i':>15s}")
print("-" * 78)
for j, nm in enumerate(names):
    m, lo, hi = mc["median"][j], mc["lo68"][j], mc["hi68"][j]
    mark = "  ⬅️" if nm == "TOTAL" else ""
    print(f"{nm:>22s} {m:15.5f}   [{lo:.5f}, {hi:.5f}] {1-m:15.5f}{mark}")

c_med = mc["median"][-1]
z = (c_med - C_TARGET) / np.hypot(mc["std"][-1], ps.std("contrast_reported"))
print(f"\n   modelo   = {c_med:.4f}  [{mc['lo68'][-1]:.4f}, {mc['hi68'][-1]:.4f}] (IC 68 %)")
print(f"   reportado = {C_TARGET:.4f} ± {ps.std('contrast_reported'):.4f}")
print(f"   z-score   = {z:+.2f}  "
      f"{'✅ compatible' if abs(z) < 2 else '❌ discrepante (informativo)'}")

rank = error_ranking(contrast_budget(**V))
print(f"\n   DOMINANTE: '{rank[0][0]}' con pérdida {rank[0][1]:.4f} "
      f"({100*rank[0][1]/max(1e-12, 1-c_med):.0f} % del total)")


# =====================================================================
# (C) DIAGRAMA DE TORNADO — ¿qué medir primero?
# =====================================================================
print(qlsim.banner("(C) SENSIBILIDAD: qué medir primero en el laboratorio"))

knobs = ["t_exposure", "tau_decay", "nbar_radial", "eta_radial",
         "detection_fidelity", "pi_pulse_error", "nbar_transfer_after_cooling"]
sens = sensitivity(contrast_total, ps, knobs)

print(f"{'parámetro':>30s} {'dC/dp':>14s} {'sigma_p':>12s} "
      f"{'|dC/dp|*sigma':>15s} {'proc.':>9s}")
print("-" * 84)
for nm, deriv, contrib in sens:
    print(f"{nm:>30s} {deriv:14.4g} {ps.std(nm):12.4g} {contrib:15.5f} "
          f"{ps[nm].provenance.value:>9s}")

tot = np.hypot.reduce([c for _, _, c in sens])
print(f"\n   incertidumbre total (suma en cuadratura) = ±{tot:.4f}")
print(f"\n⭐ PRIORIDAD DE CALIBRACIÓN (los ASSUMED con mayor contribución):")
for i, (nm, _, c) in enumerate([s for s in sens
                                if ps[s[0]].provenance.value == "assumed"][:3], 1):
    print(f"   {i}. {nm:32s} contribuye ±{c:.4f} al contraste")


# =====================================================================
# FIGURA
# =====================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

# --- (1) cascada del contraste ---
ax = fig.add_subplot(gs[0, 0])
vals = [float(np.atleast_1d(contrast_budget(**V)[k]).ravel()[0]) for k in FACTORS]
cum, xs, ys = 1.0, [], []
for v in vals:
    xs.append(cum)
    cum *= v
    ys.append(cum)
ax.bar(range(len(FACTORS)), [x - y for x, y in zip(xs, ys)],
       bottom=ys, color="C3", alpha=0.85)
ax.axhline(cum, color="C0", lw=2, label=f"modelo = {cum:.3f}")
ax.axhline(C_TARGET, color="k", ls="--", lw=1.6,
           label=f"reportado = {C_TARGET:.2f}")
ax.axhspan(C_TARGET - ps.std("contrast_reported"),
           C_TARGET + ps.std("contrast_reported"), color="k", alpha=0.12)
ax.set_xticks(range(len(FACTORS)))
ax.set_xticklabels([f.replace("_", "\n") for f in FACTORS], fontsize=7)
ax.set(ylabel="contraste acumulado", title="(B) Cascada de pérdidas", ylim=(0, 1.05))
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

# --- (2) distribución MC del contraste total ---
ax = fig.add_subplot(gs[0, 1])
ax.hist(mc["samples"][:, -1], bins=70, color="C0", alpha=0.8, density=True)
for q, c, ls in [(mc["median"][-1], "k", "-"), (mc["lo68"][-1], "gray", ":"),
                 (mc["hi68"][-1], "gray", ":")]:
    ax.axvline(q, color=c, ls=ls, lw=1.4)
ax.axvline(C_TARGET, color="C3", lw=2, label="reportado")
ax.set(xlabel="contraste total", ylabel="densidad",
       title=f"(B) MC n=4000  ->  {c_med:.3f} $$\\pm$$ {mc['std'][-1]:.3f}")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# --- (3) tornado ---
ax = fig.add_subplot(gs[0, 2])
lbl = [s[0] for s in sens][::-1]
con = [s[2] for s in sens][::-1]
col = ["C3" if ps[n].provenance.value == "assumed" else "C0" for n in lbl]
ax.barh(range(len(lbl)), con, color=col, alpha=0.85)
ax.set_yticks(range(len(lbl)))
ax.set_yticklabels([n.replace("_", " ") for n in lbl], fontsize=7)
ax.set(xlabel="$$|\\partial C/\\partial p|\\cdot\\sigma_p$$",
       title="(C) Tornado — rojo = ASSUMED")
ax.grid(alpha=0.3, axis="x")

# --- (4) inversión de t_exposure ---
ax = fig.add_subplot(gs[1, 0])
te = np.linspace(ps.value("t_probe"), 4 * TAU, 250)
ax.plot(te * 1e6, [contrast_of_texp(x) for x in te], "C0-", lw=1.8)
ax.axhline(C_TARGET, color="k", ls="--", lw=1.5, label="0.93 reportado")
ax.axhspan(C_TARGET - ps.std("contrast_reported"),
           C_TARGET + ps.std("contrast_reported"), color="k", alpha=0.12)
ax.axvline(t_exp_fit * 1e6, color="C3", lw=1.8,
           label=f"inferido = {t_exp_fit*1e6:.1f} µs")
ax.axvline(TAU * 1e6, color="C2", ls=":", lw=1.4,
           label=f"$$\\tau(^3P_1)$$ = {TAU*1e6:.0f} µs")
ax.set(xlabel="$$t_{\\rm exposure}$$ [µs]", ylabel="contraste modelado",
       title="(A1) Inversión de $$t_{\\rm exposure}$$")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# --- (5) inversión de t2 ---
ax = fig.add_subplot(gs[1, 1])
fin = np.isfinite(t2_scan)
ax.semilogx(t2_scan[fin] * 1e6, tc_scan[fin] * 1e6, "C0o-", ms=5)
ax.axhline(T_COH_TARGET * 1e6, color="k", ls="--", lw=1.5,
           label="118 µs reportado")
ax.axhline(tc_scan[-1] * 1e6, color="C2", ls=":", lw=1.4,
           label=f"$$t_2=\\infty$$: {tc_scan[-1]*1e6:.0f} µs")
if np.isfinite(t2_fit):
    ax.axvline(t2_fit * 1e6, color="C3", lw=1.8,
               label=f"inferido = {t2_fit*1e6:.0f} µs")
ax.set(xlabel="$$t_2$$ (defasaje) [µs]", ylabel="$$T_{\\rm coh}$$ ajustado [µs]",
       title="(A2) Inversión de $$t_2$$")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# --- (6) Debye-Waller vs nbar: el mando más importante ---
ax = fig.add_subplot(gs[1, 2])
nb = np.logspace(-2, 1.3, 120)
for eta, ls in [(ETA_R, "-"), (ETA_R / 2, "--"), (2 * ETA_R, ":")]:
    n = np.arange(80)
    dw = debye_waller_factor(eta, n)
    c = [float((thermal_weights(x, 80) * np.sin(np.pi * dw / 2) ** 2).sum())
         for x in nb]
    ax.semilogx(nb, c, ls, label=f"$$\\eta_r$$ = {eta:.3f}")
ax.axvline(NBAR_R, color="C3", lw=1.8, label=f"$$\\bar n$$ ASSUMED = {NBAR_R:.1f}")
ax.axhline(C_TARGET, color="k", ls="--", lw=1.2)
ax.set(xlabel="$$\\bar n$$ radial", ylabel="contraste Debye-Waller",
       title="El mando que domina: $$\\bar n$$ radial")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

fig.suptitle("06 — Presupuesto