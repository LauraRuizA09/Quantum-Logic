# ========================= qlsim/observables.py ===========================
"""Generación y AJUSTE de las observables de las Figs. 3A, 3B, 3C.

Regla: cada generador acepta arrays de parámetros (para Monte Carlo) y
devuelve la probabilidad IDEAL. El ruido de disparo se añade aparte con
`uncertainty.shot_noise`, nunca dentro del modelo físico.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import eval_genlaguerre

from .pulses import rabi_lineshape


# ------------------------- Debye-Waller ----------------------------------
def debye_waller_factor(eta: float, n: int | np.ndarray) -> np.ndarray:
    """Factor de reducción de la Rabi de PORTADORA por un modo espectador:

        Omega_n / Omega_0 = exp(-eta^2/2) * L_n(eta^2)

    (L_n = polinomio de Laguerre). Para eta<<1: ~ 1 - eta^2 (n + 1/2).
    Ésta es la "reducción de Debye-Waller" que el paper cita como causa de
    la fluctuación tiro a tiro de la frecuencia de Rabi.
    """
    x = eta**2
    return np.exp(-x / 2.0) * eval_genlaguerre(n, 0, x)


def thermal_weights(nbar: float, n_max: int) -> np.ndarray:
    n = np.arange(n_max)
    p = (nbar / (1 + nbar)) ** n / (1 + nbar)
    return p / p.sum()


# ------------------------- Fig. 3A: espectro ------------------------------
def spectrum(detuning_hz, *, omega_rabi, t_probe, contrast, **_):
    """P_up vs desintonización, forma de línea Rabi de pulso cuadrado."""
    d = np.atleast_1d(detuning_hz) * 2 * np.pi
    om = np.atleast_1d(omega_rabi)[:, None]
    tp = np.atleast_1d(t_probe)[:, None]
    c = np.atleast_1d(contrast)[:, None]
    return np.squeeze(c * rabi_lineshape(d[None, :], om, tp))


def fit_spectrum(detuning_hz, p_obs, sigma=None):
    """Ajusta (centro, Rabi, contraste) y devuelve FWHM con incertidumbre."""
    def model(d, center, om_khz, contrast, t_probe_us):
        return contrast * rabi_lineshape((d - center) * 2 * np.pi,
                                         om_khz * 1e3 * 2 * np.pi,
                                         t_probe_us * 1e-6)
    p0 = [0.0, 40.0, 0.9, 12.6]
    popt, pcov = curve_fit(model, detuning_hz, p_obs, p0=p0, sigma=sigma,
                           absolute_sigma=sigma is not None, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    # FWHM numérico del modelo ajustado
    fine = np.linspace(-3e5, 3e5, 200001)
    y = model(fine, *popt)
    half = y.max() / 2
    idx = np.where(y >= half)[0]
    fwhm = fine[idx[-1]] - fine[idx[0]]
    return dict(popt=popt, perr=perr, fwhm=fwhm, model=model,
                names=["center_Hz", "Omega_kHz", "contrast", "t_probe_us"])


# ------------------------- Fig. 3B: Rabi flopping -------------------------
def rabi_flopping(t, *, omega_rabi, contrast, tau_decay, nbar_radial,
                  eta_radial, n_max_dw=25, **_):
    """Oscilaciones de Rabi con las TRES causas de amortiguamiento del paper.

    1) decaimiento espontáneo del 3P1  -> factor exp(-t/tau_decay)
    2) Debye-Waller de los modos radiales Doppler -> promedio incoherente
       sobre n con pesos térmicos, cada n con su propia Omega_n
    3) contraste global (preparación + detección + errores de pulso)

    NOTA IMPORTANTE: el resultado NO es exactamente una senoide amortiguada
    exponencialmente (el paper ajusta eso, pero es una parametrización, no
    la física). La suma térmica da un decaimiento con cola tipo ley de
    potencias. Comparar ambos es el ejercicio E4.
    """
    t = np.atleast_1d(t)[None, :]
    om = np.atleast_1d(omega_rabi)[:, None]
    c = np.atleast_1d(contrast)[:, None]
    tau = np.atleast_1d(tau_decay)[:, None]
    nb = np.atleast_1d(nbar_radial)
    et = np.atleast_1d(eta_radial)

    out = np.zeros((max(om.shape[0], nb.size), t.shape[1]))
    for i in range(out.shape[0]):
        j = min(i, nb.size - 1)
        w = thermal_weights(nb[j], n_max_dw)
        dw = debye_waller_factor(et[min(i, et.size - 1)], np.arange(n_max_dw))
        om_i = om[min(i, om.shape[0] - 1), 0] * dw            # (n_max_dw,)
        osc = (w[:, None] * np.sin(om_i[:, None] * t / 2.0) ** 2).sum(0)
        out[i] = c[min(i, c.shape[0] - 1), 0] * np.exp(-t[0] / tau[min(i, tau.shape[0] - 1), 0]) * osc
    return np.squeeze(out)


def fit_damped_sine(t, y, sigma=None):
    """Ajuste a la parametrización del paper: senoide amortiguada."""
    def model(t, contrast, om_khz, t_coh_us, offset):
        return offset + contrast / 2 * (
            1 - np.exp(-t / (t_coh_us * 1e-6))
            * np.cos(om_khz * 1e3 * 2 * np.pi * t))
    p0 = [0.9, 40.0, 118.0, 0.0]
    popt, pcov = curve_fit(model, t, y, p0=p0, sigma=sigma,
                           absolute_sigma=sigma is not None, maxfev=30000)
    return dict(popt=popt, perr=np.sqrt(np.diag(pcov)), model=model,
                names=["contrast", "Omega_kHz", "T_coh_us", "offset"])


# ------------------------- Fig. 3C: Ramsey motional ----------------------
def ramsey_motional(t_delay, *, nu_mode, contrast, t2_motion, phase0=0.0, **_):
    """Franjas de Ramsey del CÚBIT MOTIONAL: baten a nu_mode.

        P = 1/2 [1 + C exp(-t/T2) cos(2 pi nu_mode t + phi0)]

    Ver pregunta ❷: la separación energética entre |0> y |1> es hbar*w_m,
    luego la fase relativa acumulada es w_m * t_d.
    """
    t = np.atleast_1d(t_delay)[None, :]
    nu = np.atleast_1d(nu_mode)[:, None]
    c = np.atleast_1d(contrast)[:, None]
    t2 = np.atleast_1d(t2_motion)[:, None]
    ph = np.atleast_1d(phase0)[:, None]
    return np.squeeze(0.5 * (1 + c * np.exp(-t / t2)
                             * np.cos(2 * np.pi * nu * t + ph)))


def fit_ramsey(t, y, nu_guess, sigma=None):
    """Recupera nu_mode del ajuste: el uso #2 de la Fig. 3C (calibrar nu_m)."""
    def model(t, nu_mhz, contrast, t2_us, phase0):
        return 0.5 * (1 + contrast * np.exp(-t / (t2_us * 1e-6))
                      * np.cos(2 * np.pi * nu_mhz * 1e6 * t + phase0))
    p0 = [nu_guess / 1e6, 0.9, 500.0, 0.0]
    popt, pcov = curve_fit(model, t, y, p0=p0, sigma=sigma,
                           absolute_sigma=sigma is not None, maxfev=40000)
    return dict(popt=popt, perr=np.sqrt(np.diag(pcov)), model=model,
                names=["nu_mode_MHz", "contrast", "T2_us", "phase0"])