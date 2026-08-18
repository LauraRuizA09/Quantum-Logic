# ============================ qlsim/pulses.py =============================
"""Hamiltonianos de portadora / banda lateral y propagadores.

TEORÍA
------
En la imagen de interacción, a primer orden en eta (régimen de Lamb-Dicke),
con desintonización delta respecto a la resonancia correspondiente:

  Portadora:  H = (hbar*Omega/2)  (sp e^{i phi} + h.c.)  - hbar*delta*|up><up|
  RSB      :  H = (hbar*eta*Omega/2)(sp a e^{i phi} + h.c.) - hbar*delta*|up><up|
  BSB      :  H = (hbar*eta*Omega/2)(sp a^dag e^{i phi} + h.c.) - hbar*delta*|up><up|

⭐ LA PUERTA CONDICIONAL: en la RSB, H|down,0> = 0 exactamente, porque
   a|0> = 0  Y  sigma_-|down> = 0. NO es desintonización: es que |up,n=-1>
   no existe. `test_dark_state` lo verifica numéricamente.

⚠️ LÍMITE DE VALIDEZ: primer orden en eta requiere eta*sqrt(n+1) << 1.
   `check_lamb_dicke` te avisa cuando lo violas. Truncar n_max también
   introduce error: usa `convergence_scan`.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.linalg import expm

from .constants import HBAR
from .hilbert import Space


class PulseKind(Enum):
    CARRIER = "carrier"
    RSB = "rsb"      # red sideband: quita un fonón al subir
    BSB = "bsb"      # blue sideband: añade un fonón al subir


def hamiltonian(sp: Space, ion: int, kind: PulseKind, omega: float,
                eta: float = 0.0, detuning: float = 0.0,
                phase: float = 0.0) -> np.ndarray:
    """H/hbar del pulso (devuelve el Hamiltoniano DIVIDIDO por hbar, en rad/s).

    Parameters
    ----------
    omega : frecuencia de Rabi de portadora [rad/s].
    eta : parámetro de Lamb-Dicke (solo usado en RSB/BSB).
    detuning : delta [rad/s], respecto a la resonancia del tipo de pulso.
    phase : fase del campo [rad] (importa en Ramsey, no en Rabi/espectros).
    """
    sp_op = sp.sigma_plus(ion)
    if kind is PulseKind.CARRIER:
        coupling, rate = sp_op, omega / 2.0
    elif kind is PulseKind.RSB:
        coupling, rate = sp_op @ sp.a, eta * omega / 2.0
    elif kind is PulseKind.BSB:
        coupling, rate = sp_op @ sp.a.conj().T, eta * omega / 2.0
    else:
        raise ValueError(kind)

    drive = rate * (np.exp(1j * phase) * coupling
                    + np.exp(-1j * phase) * coupling.conj().T)
    det = -detuning * sp.projector_internal(ion, 1)
    return drive + det


def propagator(sp: Space, duration: float, **kw) -> np.ndarray:
    """U = exp(-i H t) con H en rad/s (ya dividido por hbar)."""
    return expm(-1j * hamiltonian(sp, **kw) * duration)


def pi_time(omega: float, eta: float = 1.0, kind: PulseKind = PulseKind.CARRIER,
            n: int = 0) -> float:
    """Duración de un pulso pi.

    Portadora: t = pi/Omega.
    Banda lateral n<->n+1: t = pi/(eta*Omega*sqrt(n+1)).

    ⚠️ La dependencia sqrt(n+1) es exactamente la razón por la que el
    calentamiento descalibra el pulso pi (pregunta ❹ de tu mini-test).
    """
    if kind is PulseKind.CARRIER:
        return np.pi / omega
    return np.pi / (eta * omega * np.sqrt(n + 1))


def check_lamb_dicke(eta: float, n_max: int, tol: float = 0.2) -> str | None:
    """Devuelve una advertencia si el desarrollo a primer orden es dudoso."""
    x = eta * np.sqrt(n_max)
    if x > tol:
        return (f"eta*sqrt(n_max) = {x:.3f} > {tol}: el primer orden en eta "
                f"puede fallar. Reduce n_max o incluye ordenes superiores.")
    return None


# --------------------------- soluciones analíticas -------------------------
def rabi_lineshape(detuning: np.ndarray, omega: float, t: float) -> np.ndarray:
    """Probabilidad de excitación de un pulso cuadrado (Rabi generalizada).

        P(delta) = Omega^2/(Omega^2+delta^2) * sin^2( sqrt(Omega^2+delta^2) t/2 )

    Para Omega*t = pi esto da la forma de línea de la Fig. 3A.
    """
    w_eff = np.sqrt(omega**2 + detuning**2)
    return (omega**2 / w_eff**2) * np.sin(w_eff * t / 2.0) ** 2


def rabi_fwhm_coefficient(tol: float = 1e-12) -> float:
    """Coeficiente x* tal que FWHM_Hz = x*/t_pi. Devuelve ~0.7993.

    Se obtiene resolviendo  P(x)=1/2  con  P(x)= sin^2(pi sqrt(1+x^2)/2)/(1+x^2)
    donde x = delta/Omega. NO está hardcodeado a propósito: quiero que veas
    de dónde sale el 63 kHz del paper.
    """
    from scipy.optimize import brentq

    def f(x):
        return np.sin(np.pi * np.sqrt(1 + x * x) / 2) ** 2 / (1 + x * x) - 0.5

    return brentq(f, 0.1, 1.5, xtol=tol)


def rabi_fwhm_hz(t_pi: float) -> float:
    """FWHM en Hz de una forma de línea Rabi de pulso pi de duración t_pi."""
    return rabi_fwhm_coefficient() / t_pi


def sinc2_lineshape(detuning: np.ndarray, t: float) -> np.ndarray:
    """sinc^2, la aproximación de área pequeña (la que ajusta Chou 2017).

    Compárala con `rabi_lineshape`: coinciden en las alas y difieren en el
    centro. Ejercicio E3.
    """
    x = detuning * t / 2.0
    return np.sinc(x / np.pi) ** 2