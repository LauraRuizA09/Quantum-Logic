# ============================ qlsim/errors.py =============================
"""Presupuesto de error: contraste (93 %) y falsos positivos.

REGLA: nunca reportes "coincide con el 93 % del paper". Reporta
93 % = producto de factores, cada uno con su incertidumbre, y di cuál domina.
"""
from __future__ import annotations

import numpy as np


def contrast_budget(*, tau_decay, t_sequence, nbar_radial, eta_radial,
                    detection_fidelity, pi_pulse_error, n_pi=3,
                    nbar_transfer_after_cooling=0.0, **_):
    """Descompone el contraste observable en factores multiplicativos.

    Returns dict con cada factor y el producto.
    """
    # 1) supervivencia al decaimiento espontáneo del 3P1
    c_decay = np.exp(-np.asarray(t_sequence) / np.asarray(tau_decay))

    # 2) Debye-Waller: dispersión de Omega -> pérdida de contraste ~ exp(-sigma^2 t^2/2)
    #    Aproximación de 2º orden: dOmega/Omega ~ eta_r^2 * sigma_n,
    #    con sigma_n = sqrt(nbar(nbar+1)) para un estado térmico.
    nb = np.asarray(nbar_radial)
    sigma_n = np.sqrt(nb * (nb + 1))
    rel_spread = np.asarray(eta_radial) ** 2 * sigma_n
    c_dw = 1.0 / (1.0 + 2.0 * rel_spread)     # cota empírica conservadora

    # 3) fidelidad de detección (lectura del ion lógico)
    c_det = 2.0 * np.asarray(detection_fidelity) - 1.0

    # 4) errores de área de los pulsos pi
    c_pulse = np.cos(np.pi / 2 * np.asarray(pi_pulse_error)) ** (2 * n_pi)

    # 5) población residual fuera de |0> tras el enfriado
    c_gsc = 1.0 / (1.0 + np.asarray(nbar_transfer_after_cooling))

    total = c_decay * c_dw * c_det * c_pulse * c_gsc
    return dict(decay=c_decay, debye_waller=c_dw, detection=c_det,
                pi_pulses=c_pulse, ground_state_cooling=c_gsc, total=total)


def false_positive_rate(*, heating_rate, t_sequence,
                        nbar_transfer_after_cooling=0.0,
                        detection_fidelity=1.0, n_projections=1, **_):
    """Tasa de falsos positivos de la lectura motional.

    P_FP = 1 - exp(-ndot * tau)               (calentamiento acumulado)
           + P(n>0 | enfriado imperfecto)     (independiente de tau)
           + (1 - F_det)                      (error de lectura)

    Con `n_projections` medidas consecutivas exigidas, P_FP -> P_FP^n
    (asumiendo independencia: OPTIMISTA, el calentamiento está correlacionado
    entre proyecciones consecutivas -> cota inferior. Documéntalo.)
    """
    nd = np.asarray(heating_rate)
    tau = np.asarray(t_sequence)
    p_heat = 1.0 - np.exp(-nd * tau)
    nb = np.asarray(nbar_transfer_after_cooling)
    p_gsc = nb / (1.0 + nb)
    p_det = 1.0 - np.asarray(detection_fidelity)
    p1 = np.clip(p_heat + p_gsc + p_det, 0.0, 1.0)
    return p1 ** n_projections


def max_steps_budget(heating_rate, t_step, p_fp_target=0.01):
    """¿Cuántos pasos de secuencia tolera NUESTRA trampa?

    Responde la pregunta ❹ y define el techo del número de pulsos de la
    política de RL del paper 3.
    """
    nd = np.asarray(heating_rate)
    tau_max = -np.log(1.0 - p_fp_target) / nd
    return tau_max / t_step