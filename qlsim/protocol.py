"""Protocolo QLS de 4 pasos (Schmidt 2005, Fig. 1) y medida proyectiva.

EL PROTOCOLO
------------
  Paso 0  |down>_S |down>_L |0>_m
  Paso 1  interrogación en S -> (alpha|down> + beta|up>)_S |down>_L |0>_m
  Paso 2  pulso pi RSB en S  -> |down>_S |down>_L (alpha|0> + beta|1>)_m
  Paso 3  pulso pi RSB en L  -> |down>_S (alpha|down> + beta|up>)_L |0>_m
  Paso 4  medida por fluorescencia en L

⭐ PUENTE AL PAPER 3 (RL): `transition_matrices` devuelve exactamente las
   A_k^(a) del MDP: A_k[j, i] = P(estado j | estado i, pulso a, resultado k).
   La suma sobre k de las sumas de columnas debe ser 1 (traza conservada):
   ese es el test CPTP más barato y potente que puedes escribir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hilbert import DOWN, UP, Space
from .pulses import PulseKind, pi_time, propagator


@dataclass
class Step:
    """Un paso de la secuencia: un pulso sobre un ion."""
    ion: int
    kind: PulseKind
    omega: float
    eta: float = 0.0
    duration: float | None = None   # None => pulso pi automático
    detuning: float = 0.0
    phase: float = 0.0
    label: str = ""

    def unitary(self, sp: Space) -> np.ndarray:
        t = self.duration
        if t is None:
            t = pi_time(self.omega, self.eta, self.kind)
        return propagator(sp, t, ion=self.ion, kind=self.kind,
                          omega=self.omega, eta=self.eta,
                          detuning=self.detuning, phase=self.phase)


def thermal_motion(n_max: int, nbar: float) -> np.ndarray:
    """Distribución térmica truncada y RENORMALIZADA del movimiento.

    ⚠️ La renormalización tras truncar introduce un sesgo. Con nbar=0.5 y
    n_max=6 el peso descartado es <1e-3, pero VERIFÍCALO (ejercicio E5).
    """
    if nbar <= 0:
        p = np.zeros(n_max)
        p[0] = 1.0
        return p
    n = np.arange(n_max)
    p = (nbar / (1 + nbar)) ** n / (1 + nbar)
    return p / p.sum()


def initial_state(sp: Space, nbar: float = 0.0) -> np.ndarray:
    """rho inicial: ambos iones en |down>, movimiento térmico con nbar."""
    pn = thermal_motion(sp.n_max, nbar)
    rho = np.zeros((sp.dim, sp.dim), dtype=complex)
    for n, w in enumerate(pn):
        v = sp.ket(DOWN, DOWN, n)
        rho += w * np.outer(v, v.conj())
    return rho


def apply_steps(rho: np.ndarray, sp: Space, steps: list[Step]) -> np.ndarray:
    """Evolución unitaria a través de una lista de pasos."""
    for s in steps:
        U = s.unitary(sp)
        rho = U @ rho @ U.conj().T
    return rho


def measure_logic(rho: np.ndarray, sp: Space, fidelity: float = 1.0
                  ) -> tuple[float, np.ndarray, np.ndarray]:
    """Medida proyectiva del ion lógico (fluorescencia).

    Returns
    -------
    p_bright : probabilidad de detectar |down>_L (brillante).
    rho_bright, rho_dark : estados colapsados y RENORMALIZADOS.

    `fidelity` modela error de detección simétrico: la probabilidad
    REPORTADA es  p_obs = F*p + (1-F)*(1-p). Los estados colapsados no
    cambian (el error es de lectura, no de proyección).
    """
    P_b = sp.projector_internal(1, DOWN)
    P_d = sp.projector_internal(1, UP)
    p_b = float(np.real(np.trace(P_b @ rho)))
    p_b_obs = fidelity * p_b + (1 - fidelity) * (1 - p_b)

    rho_b = P_b @ rho @ P_b
    rho_d = P_d @ rho @ P_d
    tr_b, tr_d = np.real(np.trace(rho_b)), np.real(np.trace(rho_d))
    rho_b = rho_b / tr_b if tr_b > 1e-15 else rho_b
    rho_d = rho_d / tr_d if tr_d > 1e-15 else rho_d
    return p_b_obs, rho_b, rho_d


def reset_motion(rho: np.ndarray, sp: Space, nbar: float = 0.0) -> np.ndarray:
    """⭐ Re-enfriamiento: TRAZA sobre el movimiento y lo reinicia.

    Este es EL paso irreversible (pregunta ❸ de tu mini-test): la emisión
    espontánea del enfriamiento destruye toda coherencia interno-movimiento,
    lo que justifica describir la dinámica con VECTORES DE POBLACIÓN en el
    paper de RL. Aquí lo implementamos honestamente como un mapa CPTP.

    Modelo: rho_new = (Tr_mot rho) ⊗ rho_thermal(nbar).
    ⚠️ Aproximación: asume enfriamiento perfecto e instantáneo y destrucción
    TOTAL de coherencias interno-movimiento. Cuantifica el error comparando
    con una simulación Lindblad (ejercicio E8).
    """
    r = rho.reshape(2, 2, sp.n_max, 2, 2, sp.n_max)
    rho_int = np.einsum("abncdn->abcd", r).reshape(4, 4)   # traza parcial
    pn = np.diag(thermal_motion(sp.n_max, nbar)).astype(complex)
    return np.kron(rho_int, pn)


# ---------------------------------------------------------------------------
@dataclass
class QLSResult:
    p_bright: float
    fidelity_mapping: float
    alpha2_true: float
    rho_final: np.ndarray = field(repr=False)


def qls_mapping(sp: Space, omega_s: float, eta_s: float,
                omega_l: float, eta_l: float,
                probe_detuning: float = 0.0,
                probe_duration: float | None = None,
                nbar_init: float = 0.0,
                detection_fidelity: float = 1.0,
                pi_error: float = 0.0) -> QLSResult:
    """Ejecuta el protocolo completo de la Fig. 1 y devuelve la señal.

    `pi_error` escala el área de TODOS los pulsos pi por (1+pi_error):
    modela miscalibración común (el caso pesimista, correlacionado).
    """
    rho = initial_state(sp, nbar_init)

    # Paso 1: interrogación (portadora en el ion de espectroscopía)
    t_probe = probe_duration or pi_time(omega_s)
    probe = Step(0, PulseKind.CARRIER, omega_s, duration=t_probe,
                 detuning=probe_detuning, label="interrogación")
    rho = apply_steps(rho, sp, [probe])

    # Verdad de referencia: población real de |up>_S antes del mapeo
    alpha2 = float(np.real(np.trace(sp.projector_internal(0, UP) @ rho)))

    # Pasos 2 y 3: mapeo S -> movimiento -> L
    scale = 1.0 + pi_error
    map_steps = [
        Step(0, PulseKind.RSB, omega_s, eta_s,
             duration=scale * pi_time(omega_s, eta_s, PulseKind.RSB),
             label="pi RSB en S"),
        Step(1, PulseKind.RSB, omega_l, eta_l,
             duration=scale * pi_time(omega_l, eta_l, PulseKind.RSB),
             label="pi RSB en L"),
    ]
    rho = apply_steps(rho, sp, map_steps)

    # Paso 4: medida. |up>_L (oscuro) corresponde a haber excitado S.
    p_bright, _, _ = measure_logic(rho, sp, detection_fidelity)
    p_up_L = 1.0 - p_bright

    fid = 1.0 - abs(p_up_L - alpha2)   # fidelidad del mapeo (1 = perfecto)
    return QLSResult(p_bright, fid, alpha2, rho)


# ---------------------------------------------------------------------------
def transition_matrices(sp: Space, step: Step, nbar_reset: float = 0.0,
                        k_max: int = 1) -> list[np.ndarray]:
    """⭐ Matrices A_k del MDP (puente al paper de RL, arXiv:2410.11839).

    Para cada resultado de medida motional k, devuelve A_k tal que
        p(k | S, a) = ||A_k S||_1      y      S' = A_k S / p(k|S,a)
    donde S es el vector de poblaciones internas (dimensión 4 = 2x2 aquí,
    y N_S estados moleculares en el paper 2/3).

    TEST OBLIGATORIO: sum_k colsum(A_k) == 1 para todo estado inicial.
    """
    n_int = 4
    A = [np.zeros((n_int, n_int)) for _ in range(k_max + 2)]
    U = step.unitary(sp)

    for i in range(n_int):
        s, l = divmod(i, 2)
        # estado inicial: interno |i>, movimiento reiniciado a nbar_reset
        rho = np.zeros((sp.dim, sp.dim), dtype=complex)
        for n, w in enumerate(thermal_motion(sp.n_max, nbar_reset)):
            v = sp.ket(s, l, n)
            rho += w * np.outer(v, v.conj())
        rho = U @ rho @ U.conj().T

        pops = sp.populations(rho)                     # (2, 2, n_max)
        for k in range(k_max + 1):
            block = pops[:, :, k] if k < k_max else pops[:, :, k_max:].sum(-1)
            A[k][:, i] = block.reshape(-1)
    return A[:k_max + 1]


def check_cptp(matrices: list[np.ndarray], tol: float = 1e-9) -> float:
    """Devuelve el máximo error |sum_k colsum(A_k) - 1|. Debe ser ~0."""
    total = sum(m.sum(axis=0) for m in matrices)
    return float(np.max(np.abs(total - 1.0)))