"""Modos normales de un cristal de dos iones de masas distintas.

TEORÍA (la deducción completa, porque es donde más se equivoca la gente)
-----------------------------------------------------------------------
AXIAL: en una trampa lineal el confinamiento axial es ELECTROSTÁTICO, luego
la curvatura del potencial es la misma para ambos iones (misma carga):

    U_trap = (1/2) k z_i^2   con   k = m_i * omega_{z,i}^2   IGUAL para i=1,2
    =>  omega_{z,2} = omega_{z,1} * sqrt(m_1/m_2)

Equilibrio (z1=-d/2, z2=+d/2), con C = e^2/(4 pi eps0):
    k*(d/2) = C/d^2   =>   C/d^3 = k/2          (¡identidad muy útil!)

Hessiano axial:
    V_11 = V_22 = k + 2C/d^3 = 2k ;  V_12 = -2C/d^3 = -k

Hessiano pesado por masas, A_ij = V_ij/sqrt(m_i m_j), con mu = m1/m2:
    A = omega_1^2 * [[2, -sqrt(mu)], [-sqrt(mu), 2*mu]]

Autovalores:  lambda_pm = omega_1^2 * [ (1+mu) ± sqrt((1-mu)^2 + mu) ]
  CHECK mu=1 ->  lambda = {1, 3} => omega = {omega_z, sqrt(3) omega_z}  ✓

RADIAL: el confinamiento viene del pseudopotencial RF, omega_r ∝ 1/m:
    omega_{r,2} = omega_{r,1} * (m_1/m_2)
La Coulomb transversal DES-confina:  V_11 = m1 wr1^2 - C/d^3 ; V_12 = +C/d^3

LAMB-DICKE: z_i = sum_p (b_{i,p}/sqrt(m_i)) sqrt(hbar/(2 omega_p)) (a_p + a_p^dag)
    =>  eta_{i,p} = dk_eff * (b_{i,p}/sqrt(m_i)) * sqrt(hbar/(2 omega_p))
  CHECK un ion:  eta = k sqrt(hbar/(2 m omega))  ✓
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import COULOMB_K, HBAR


@dataclass
class NormalModes:
    """Resultado del cálculo de modos normales de un par de iones."""
    freqs: np.ndarray        # (2,) frecuencias angulares, orden creciente
    vectors: np.ndarray      # (2,2) columnas = autovectores pesados por masa
    separation: float        # m, distancia de equilibrio
    masses: np.ndarray       # (2,)
    labels: tuple[str, str]  # etiquetas físicas

    @property
    def nu(self) -> np.ndarray:
        """Frecuencias en Hz."""
        return self.freqs / (2 * np.pi)

    def amplitude(self, ion: int, mode: int) -> float:
        """Amplitud FÍSICA (no pesada) del ion en el modo, b_i/sqrt(m_i)."""
        return self.vectors[ion, mode] / np.sqrt(self.masses[ion])

    def lamb_dicke(self, ion: int, mode: int, dk_eff: float) -> float:
        """Parámetro de Lamb-Dicke eta_{i,p} para un dk efectivo dado."""
        return abs(dk_eff * self.amplitude(ion, mode)
                   * np.sqrt(HBAR / (2 * self.freqs[mode])))

    def describe(self) -> str:
        lines = [f"separación de equilibrio d = {self.separation * 1e6:.3f} µm"]
        for p in range(2):
            a = [self.amplitude(i, p) for i in (0, 1)]
            phase = "in-phase" if a[0] * a[1] > 0 else "out-of-phase"
            lines.append(
                f"  modo {p} ({self.labels[p]:>13s}): "
                f"nu = {self.nu[p] / 1e6:8.4f} MHz  [{phase}]  "
                f"amplitudes relativas = ({a[0] / max(map(abs, a)):+.3f}, "
                f"{a[1] / max(map(abs, a)):+.3f})")
        return "\n".join(lines)


def equilibrium_separation(k_axial: float) -> float:
    """Distancia de equilibrio de dos iones: d = 2*(C/(16 pi eps0 k))^(1/3).

    Deducción: 4 k z0^3 = C con d = 2 z0.
    """
    z0 = (COULOMB_K / (4.0 * k_axial)) ** (1.0 / 3.0)
    return 2.0 * z0


def axial_modes(m1: float, m2: float, nu_z1: float) -> NormalModes:
    """Modos axiales. `nu_z1` = frecuencia secular axial de UN ion de masa m1.

    Parameters
    ----------
    m1, m2 : masas [kg]. Por convención m1 = ion lógico.
    nu_z1 : Hz.
    """
    w1 = 2 * np.pi * nu_z1
    k = m1 * w1**2                       # curvatura, común a ambos iones
    d = equilibrium_separation(k)
    c3 = COULOMB_K / d**3                # = k/2 (verificado en tests)

    V = np.array([[k + 2 * c3, -2 * c3],
                  [-2 * c3, k + 2 * c3]])
    m = np.array([m1, m2])
    A = V / np.sqrt(np.outer(m, m))      # Hessiano pesado por masas
    lam, vec = np.linalg.eigh(A)
    return NormalModes(np.sqrt(np.abs(lam)), vec, d, m,
                       ("axial in-phase", "axial out-of-phase"))


def radial_modes(m1: float, m2: float, nu_z1: float, nu_r1: float) -> NormalModes:
    """Modos radiales (una dirección transversal).

    `nu_r1` es la secular radial de UN ion de masa m1; para el otro escala
    como 1/m (pseudopotencial RF).
    """
    w1 = 2 * np.pi * nu_z1
    k = m1 * w1**2
    d = equilibrium_separation(k)
    c3 = COULOMB_K / d**3

    wr1 = 2 * np.pi * nu_r1
    wr2 = wr1 * (m1 / m2)                # escalado del pseudopotencial
    V = np.array([[m1 * wr1**2 - c3, +c3],
                  [+c3, m2 * wr2**2 - c3]])
    m = np.array([m1, m2])
    A = V / np.sqrt(np.outer(m, m))
    lam, vec = np.linalg.eigh(A)
    if np.any(lam <= 0):
        raise ValueError("Modo radial inestable: el cristal se rompe "
                         "(confinamiento radial demasiado débil). "
                         "Revisa nu_r1 vs nu_z1.")
    return NormalModes(np.sqrt(lam), vec, d, m,
                       ("radial rocking", "radial in-phase"))


def dk_single_beam(wavelength: float, theta: float) -> float:
    """|Delta k| efectivo de UN haz a ángulo `theta` del eje del modo."""
    return (2 * np.pi / wavelength) * np.cos(theta)