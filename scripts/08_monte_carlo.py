"""08 — MONTE CARLO GLOBAL: la tabla de incertidumbres del proyecto.

Propaga TODOS los parámetros del peldaño 1 simultáneamente hasta las seis
observables clave, y produce:

  (1) TABLA MAESTRA con mediana, IC 68 % e IC 95 % de cada observable
      -> es literalmente la tabla de resultados de tu poster/paper.
  (2) COMPARACIÓN con los valores publicados vía z-score.
  (3) MATRIZ DE CORRELACIÓN entre observables
      -> revela qué observables NO son independientes; si las usas todas
         como "evidencia de acuerdo", estarías contando lo mismo dos veces.
  (4) CONVERGENCIA del propio Monte Carlo (¿bastan 4000 muestras?)
      -> el error de MC escala como 1/sqrt(n); hay que demostrarlo, no asumirlo.
  (5) SEPARACIÓN estadístico vs sistemático: cuánto de la banda viene del
      ruido de disparo (N=700) y cuánto de la ignorancia de parámetros.

Referencia: Schmidt et al., Science 309, 749 (2005). DOI 10.1126/science.1114375
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

import qlsim
from qlsim.errors import contrast_budget, false_positive_rate, max_steps_budget
from qlsim.modes import axial_modes, dk_single_beam, radial_modes
from qlsim.pulses import rabi_fwhm_hz
from qlsim.uncertainty import binomial_sigma, shot_noise

N_MC = 6000
SEED = 2026
P_FP_TARGET = 0.01
T_STEP = 0.5e-3          # s por paso de secuencia (pulso + medida + enfriado)

ps = qlsim.default_paramset()


# =====================================================================
# FUNCIÓN MODELO: recalcula la CADENA COMPLETA (sin perder correlaciones)
# =====================================================================
def observables(**d) -> np.ndarray:
    """Devuelve (n, 6) con las observables clave.

    ⭐ DECISIÓN METODOLÓGICA: recalculamos los modos normales DENTRO de la
    función, en lugar de usar los Params derivados `nu_mode`/`eta_radial`.
    Así conservamos la correlación entre nu_mode, eta y las masas/frecuencias
    de las que ambos dependen. Usar los derivados como independientes
    sobreestimaría la incertidumbre (ejercicio E7).
    """
    n = len(d["nu_z_logic"])
    out = np.empty((n, 6))
    for i in range(n):
        nm = axial_modes(d["m_logic"][i], d["m_spec"][i], d["nu_z_logic"][i])
        nmr = radial_modes(d["m_logic"][i], d["m_spec"][i],
                           d["nu_z_logic"][i], d["nu_x_logic"][i])
        dk = dk_single_beam(d["lambda_probe"][i], d["theta_probe"][i])
        eta_ax = nm.lamb_dicke(1, 0, dk)
        eta_r = nmr.lamb_dicke(1, 1, dk)

        c = contrast_budget(
            tau_decay=d["tau_decay"][i], t_exposure=d["t_exposure"][i],
            nbar_radial=d["nbar_radial"][i], eta_radial=eta_r,
            detection_fidelity=d["detection_fidelity"][i],
            pi_pulse_error=d["pi_pulse_error"][i],
            nbar_transfer_after_cooling=d["nbar_transfer_after_cooling"][i],
        )["total"]

        pfp = false_positive_rate(
            heating_rate=d["heating_rate"][i], t_sequence=d["t_sequence"][i],
            nbar_transfer_after_cooling=d["nbar_transfer_after_cooling"][i],
            detection_fidelity=d["detection_fidelity"][i])

        nmax = max_steps_budget(
            d["heating_rate"][i], T_STEP, P_FP_TARGET,
            nbar_floor=d["nbar_transfer_after_cooling"][i],
            detection_fidelity=d["detection_fidelity"][i])

        out[i] = [nm.nu[0] / 1e6,                    # nu_mode [MHz]
                  eta_ax,                            # eta axial (Al+)
                  rabi_fwhm_hz(d["t_probe"][i]) / 1e3,  # FWHM [kHz]
                  float(np.atleast_1d(c).ravel()[0]),   # contraste
                  float(np.atleast_1d(pfp).ravel()[0]),  # P_FP
                  float(np.atleast_1d(nmax).ravel()[0])]  # N_max pasos
    return out


LABELS = [r"$\nu_{\rm mode}$ [MHz]", r"$\eta_{\rm axial}$",
          "FWHM [kHz]", "contraste", r"$P_{\rm FP}$", r"$N_{\rm max}$ pasos"]
KEYS = ["nu_mode_MHz", "eta_axial", "fwhm_kHz", "contrast", "p_fp", "n_max"]

# valores publicados (None si el paper no lo reporta)
PUBLISHED = {"nu_mode_MHz": (None, None), "eta_axial": (None, None),
             "fwhm_kHz": (63.0, 1.0), "contrast": (0.93, 0.02),
             "p_fp": (None, None), "n_max": (None, None)}

mc = qlsim.monte_carlo(observables, ps, n=N_MC, seed=SEED)
S = mc["samples"]


# =====================================================================
# (1) TABLA MAESTRA
# =====================================================================
print(qlsim.banner(f"(1) TABLA MAESTRA — Monte Carlo n = {N_MC}, seed = {SEED}"))
print(f"{'observable':>22s} {'mediana':>11s} {'IC 68 %':>24s} "
      f"{'IC 95 %':>24s} {'rel.':>8s}")
print("-" * 94)
for j, k in enumerate(KEYS):
    m = mc["median"][j]
    rel = mc["std"][j] / abs(m) if m else np.nan
    print(f"{k:>22s} {m:11.5g} "
          f"[{mc['lo68'][j]:10.5g},{mc['hi68'][j]:10.5g}] "
          f"[{mc['lo95'][j]:10.5g},{mc['hi95'][j]:10.5g}] {rel:7.1%}")

print("\n>>> Copia esta tabla al poster. Cada fila lleva IC, no solo el valor.")


# =====================================================================
# (2) COMPARACIÓN CON LO PUBLICADO
# =====================================================================
print(qlsim.banner("(2) BENCHMARKING contra los valores publicados"))
print(f"{'observable':>22s} {'modelo':>20s} {'publicado':>16s} {'z':>7s}  veredicto")
print("-" * 88)
for j, k in enumerate(KEYS):
    pv, pe = PUBLISHED[k]
    if pv is None:
        print(f"{k:>22s} {mc['median'][j]:11.5g} ± {mc['std'][j]:<7.3g} "
              f"{'— no reportado':>16s} {'—':>7s}  PREDICCIÓN del modelo")
        continue
    z = (mc["median"][j] - pv) / np.hypot(mc["std"][j], pe)
    verdict = "✅ compatible" if abs(z) < 2 else (
        "⚠️ tensión" if abs(z) < 3 else "❌ discrepante")
    print(f"{k:>22s} {mc['median'][j]:11.5g} ± {mc['std'][j]:<7.3g} "
          f"{pv:9.4g} ± {pe:<5.3g} {z:+7.2f}  {verdict}")

print("\n⚠️ Las filas 'PREDICCIÓN' NO son validaciones: son salidas del modelo\n"
      "   que el paper no publica. Etiquétalas como tales en el poster; es\n"
      "   exactamente lo que un revisor buscará.")


# =====================================================================
# (3) CORRELACIONES ENTRE OBSERVABLES
# =====================================================================
print(qlsim.banner("(3) MATRIZ DE CORRELACIÓN entre observables"))
C = np.corrcoef(S.T)
print("      " + "".join(f"{k[:9]:>11s}" for k in KEYS))
for i, k in enumerate(KEYS):
    print(f"{k[:9]:>9s} " + "".join(f"{C[i,j]:11.3f}" for j in range(len(KEYS))))

pairs = [(i, j, C[i, j]) for i in range(len(KEYS)) for j in range(i + 1, len(KEYS))]
strong = sorted(pairs, key=lambda t: -abs(t[2]))[:3]
print("\n   Correlaciones más fuertes:")
for i, j, c in strong:
    print(f"     {KEYS[i]:>18s} <-> {KEYS[j]:<18s} rho = {c:+.3f}")
print("\n⭐ Si dos observables están fuertemente correlacionadas, NO son dos\n"
      "   evidencias independientes de acuerdo: estarías contando lo mismo dos\n"
      "   veces al construir un chi-cuadrado global.")


# =====================================================================
# (4) CONVERGENCIA DEL MONTE CARLO
# =====================================================================
print(qlsim.banner("(4) CONVERGENCIA DEL MONTE CARLO"))
ns = np.unique(np.logspace(1.7, np.log10(N_MC), 14).astype(int))
conv = {j: [] for j in range(len(KEYS))}
for n in ns:
    for j in range(len(KEYS)):
        conv[j].append(np.median(S[:n, j]))

print(f"{'observable':>22s} {'mediana(n=N)':>14s} {'|Δ| último tramo':>18s} "
      f"{'error MC estimado':>19s}")
print("-" * 78)
for j, k in enumerate(KEYS):
    drift = abs(conv[j][-1] - conv[j][-4])
    err_mc = 1.253 * mc["std"][j] / np.sqrt(N_MC)     # error de la mediana
    ok = drift < 3 * err_mc
    print(f"{k:>22s} {conv[j][-1]:14.5g} {drift:18.4g} "
          f"{err_mc:19.4g}  {'✅' if ok else '⚠️ sube N_MC'}")
print(f"\n   Error de la mediana ≈ 1.253·σ/√n  (escala como 1/√n).")


# =====================================================================
# (5) ESTADÍSTICO vs SISTEMÁTICO
# =====================================================================
print(qlsim.banner("(5) ESTADÍSTICO (N=700) vs SISTEMÁTICO (parámetros)"))
n_rep = int(ps.value("n_rep"))
p_ref = 0.5
sig_stat = float(binomial_sigma(np.array([p_ref]), n_rep)[0])
sig_syst = mc["std"][KEYS.index("contrast")]
print(f"   ruido de disparo en p=0.5 con N={n_rep}:  σ_stat = {sig_stat:.4f}")
print(f"   ignorancia de parámetros (contraste):     σ_syst = {sig_syst:.4f}")
print(f"   razón σ_syst/σ_stat = {sig_syst/sig_stat:.2f}")
print(f"\n   ==> {'SISTEMÁTICO dominante: más repeticiones NO ayudan; hay que'
      ' medir mejor los parámetros ASSUMED.' if sig_syst > sig_stat else
      'ESTADÍSTICO dominante: promediar más repeticiones sí mejora.'}")


# =====================================================================
# FIGURA
# =====================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32)

# (a-f) histogramas de las seis observables
for j in range(6):
    ax = fig.add_subplot(gs[j // 3, j % 3])
    ax.hist(S[:, j], bins=70, color="C0", alpha=0.8, density=True)
    ax.axvline(mc["median"][j], color="k", lw=1.6, label="mediana")
    ax.axvspan(mc["lo68"][j], mc["hi68"][j], color="k", alpha=0.10, label="IC 68 %")
    ax.axvspan(mc["lo95"][j], mc["hi95"][j], color="k", alpha=0.05)
    pv, pe = PUBLISHED[KEYS[j]]
    if pv is not None:
        ax.axvline(pv, color="C3", lw=2, label="publicado")
        ax.axvspan(pv - pe, pv + pe, color="C3", alpha=0.15)
    else:
        ax.text(0.02, 0.95, "PREDICCIÓN\n(no publicado)", transform=ax.transAxes,
                fontsize=7, va="top", color="C2",
                bbox=dict(fc="w", ec="C2", alpha=0.8))
    ax.set(xlabel=LABELS[j], ylabel="densidad")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
    if KEYS[j] in ("p_fp", "n_max"):
        ax.set_xscale("log")

# (g) matriz de correlación
ax = fig.add_subplot(gs[2, 0])
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(KEYS, rotation=55, ha="right", fontsize=6)
ax.set_yticklabels(KEYS, fontsize=6)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                color="w" if abs(C[i, j]) > 0.55 else "k")
ax.set_title("(3) Correlaciones", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)

# (h) convergencia
ax = fig.add_subplot(gs[2, 1])
for j in (3, 4, 5):
    y = np.array(conv[j])
    ax.semilogx(ns, y / y[-1], "o-", ms=3, label=KEYS[j])
ax.axhspan(0.99, 1.01, color="k", alpha=0.10, label="±1 %")
ax.axhline(1, color="k", lw=0.8)
ax.set(xlabel="muestras MC", ylabel="mediana normalizada",
       title="(4) Convergencia del MC")
ax.legend(fontsize=6); ax.grid(alpha=0.3, which="both")

# (i) contraste vs N_max: el compromiso operativo
ax = fig.add_subplot(gs[2, 2])
finite = np.isfinite(S[:, 5]) & (S[:, 5] > 0)
sc = ax.scatter(S[finite, 5], S[finite, 3], c=S[finite, 4], s=4,
                cmap="viridis", norm=plt.matplotlib.colors.LogNorm(), alpha=0.55)
ax.axhline(0.93, color="C3", ls="--", lw=1.4, label="contraste publicado")
ax.axvline(13, color="C1", ls=":", lw=1.4, label="13 pulsos (Chou 2017)")
ax.axvline(83, color="C4", ls=":", lw=1.4, label="83 pulsos (RL-QLS, H$_3$O$^+$)")
ax.set_xscale("log")
ax.set(xlabel=r"$N_{\rm max}$ pasos ($P_{\rm FP}\leq 1\%$)", ylabel="contraste",
       title="Compromiso operativo")
ax.legend(fontsize=6); ax.grid(alpha=0.3, which="both")
plt.colorbar(sc, ax=ax, label=r"$P_{\rm FP}$", fraction=0.046)

fig.suptitle(f"08 — Monte Carlo global (n = {N_MC})  |  Schmidt et al., "
             f"Science 309, 749 (2005), DOI 10.1126/science.1114375", fontsize=11)
out = qlsim.figpath("08_monte_carlo.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"\n-> {out}")


# =====================================================================
# EXPORTACIÓN PARA EL PAPER
# =====================================================================
print(qlsim.banner("TABLA EN MARKDOWN (pégala en el poster)"))
print("| observable | modelo (mediana) | IC 68 % | publicado | z | tipo |")
print("|---|---|---|---|---|---|")
for j, k in enumerate(KEYS):
    pv, pe = PUBLISHED[k]
    if pv is None:
        print(f"| {LABELS[j]} | {mc['median'][j]:.5g} | "
              f"[{mc['lo68'][j]:.5g}, {mc['hi68'][j]:.5g}] | — | — | predicción |")
    else:
        z = (mc["median"][j] - pv) / np.hypot(mc["std"][j], pe)
        print(f"| {LABELS[j]} | {mc['median'][j]:.5g} | "
              f"[{mc['lo68'][j]:.5g}, {mc['hi68'][j]:.5g}] | "
              f"{pv:.4g} ± {pe:.3g} | {z:+.2f} | benchmark |")

print(qlsim.banner("SIGUIENTE PASO PARA NUESTRA TRAMPA"))
print("""
  Sustituye los parámetros medidos y vuelve a correr este script:

      ps = qlsim.add_derived_modes(qlsim.our_trap(
              nu_z_logic=(2.10e6, 5e3),
              heating_rate=(3.2, 0.4),
              nbar_radial=(0.8, 0.2),
              detection_fidelity=(0.9995, 0.0002)))

  Lo que cambia y por qué importa:
    * nu_mode y eta cambian  -> cambian todos los tiempos de pulso pi.
    * N_max cambia           -> techo DURO del nº de pulsos de la política RL.
    * el ranking del tornado (script 06) puede reordenarse por completo.

  Y en el peldaño 2 sustituye m_spec por 40CaH+ (qlsim.constants.M_CAH40):
  la razón de masas pasa de 3.0 a 1.03, luego los dos modos axiales se
  acercan y hay que revisar si siguen resueltos espectralmente.
""")