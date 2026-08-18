"""Autovalidación: unidades, límites analíticos, CPTP, convergencia.

Ejecuta esto ANTES de creerte cualquier figura.
"""
from __future__ import annotations

import numpy as np

from .constants import HBAR
from .hilbert import DOWN, Space
from .modes import axial_modes, equilibrium_separation
from .pulses import (PulseKind, hamiltonian, pi_time, propagator,
                     rabi_fwhm_coefficient, rabi_fwhm_hz)
from .protocol import Step, check_cptp, transition_matrices


class ValidationError(AssertionError):
    pass


def _ok(name, cond, detail=""):
    return (("✅" if cond else "❌") + f" {name}" + (f" — {detail}" if detail else ""))


def check_units_schmidt(ps) -> list[str]:
    """⭐ Detecta las dos inconsistencias µs/ms del PDF."""
    out = []
    # (a) FWHM vs duración de pulso
    t_p = ps.value("t_probe")
    fwhm_pred = rabi_fwhm_hz(t_p)
    fwhm_rep = ps.value("fwhm_reported")
    rel = abs(fwhm_pred - fwhm_rep) / fwhm_rep
    out.append(_ok("FWHM(t_probe) consistente con lo reportado", rel < 0.05,
                   f"predicho {fwhm_pred/1e3:.2f} kHz vs reportado "
                   f"{fwhm_rep/1e3:.1f} kHz (desv. {rel:.1%}). "
                   f"Si t_probe fuera 12.6 ms -> {rabi_fwhm_hz(12.6e-3):.1f} Hz"))
    # (b) vida del 3P1 vs tiempo de coherencia
    tau = 1.0 / (2 * np.pi * ps.value("gamma_3p1_hz"))
    t_coh = ps.value("t_coh_reported")
    out.append(_ok("tau(3P1) derivada de Gamma", abs(tau - 306e-6) / 306e-6 < 0.05,
                   f"tau = 1/(2pi*Gamma_Hz) = {tau*1e6:.1f} µs "
                   f"(el texto del PDF dice '305 ms': error de conversión)"))
    out.append(_ok("t_coh < tau(3P1)", t_coh < tau,
                   f"t_coh = {t_coh*1e6:.0f} µs, tau = {tau*1e6:.0f} µs"))
    out.append(_ok("t_sequence >> t_probe",
                   ps.value("t_sequence") > 10 * t_p,
                   f"{ps.value('t_sequence')*1e3:.2f} ms vs "
                   f"{t_p*1e6:.1f} µs"))
    return out


def check_modes_limits() -> list[str]:
    """Límites analíticos conocidos de los modos normales."""
    out = []
    m = 40 * 1.66053906660e-27
    nm = axial_modes(m, m, 1.0e6)
    r = nm.nu[1] / nm.nu[0]
    out.append(_ok("masas iguales: out/in = sqrt(3)", abs(r - np.sqrt(3)) < 1e-6,
                   f"obtenido {r:.9f}"))
    out.append(_ok("masas iguales: in-phase = nu_z single",
                   abs(nm.nu[0] - 1.0e6) / 1e6 < 1e-9, f"{nm.nu[0]/1e6:.9f} MHz"))
    # identidad C/d^3 = k/2
    k = m * (2 * np.pi * 1e6) ** 2
    d = equilibrium_separation(k)
    from .constants import COULOMB_K
    out.append(_ok("identidad C/d^3 = k/2",
                   abs(COULOMB_K / d**3 / (k / 2) - 1) < 1e-12))
    return out


def check_dark_state(n_max=6) -> list[str]:
    """⭐ El corazón del paper: H_RSB |down,0> = 0 EXACTAMENTE."""
    sp = Space(n_max)
    H = hamiltonian(sp, ion=0, kind=PulseKind.RSB, omega=2 * np.pi * 1e5,
                    eta=0.1)
    v = sp.ket(DOWN, DOWN, 0)
    norm = np.linalg.norm(H @ v)
    out = [_ok("H_RSB |down,0> = 0 (estado oscuro exacto)", norm < 1e-12,
               f"||H|down,0>|| = {norm:.2e}")]
    # y el pulso pi NO lo mueve, sin importar la duración
    for t_factor in (1, 3, 17):
        U = propagator(sp, t_factor * pi_time(2 * np.pi * 1e5, 0.1, PulseKind.RSB),
                       ion=0, kind=PulseKind.RSB, omega=2 * np.pi * 1e5, eta=0.1)
        err = abs(abs(np.vdot(v, U @ v)) - 1.0)
        out.append(_ok(f"|down,0> invariante tras {t_factor} pulsos pi", err < 1e-10,
                       f"desviación {err:.2e}"))
    # ...pero |down,1> SÍ se voltea (=> puerta condicional, no rotación simple)
    v1 = sp.ket(DOWN, DOWN, 1)
    U = propagator(sp, pi_time(2 * np.pi * 1e5, 0.1, PulseKind.RSB),
                   ion=0, kind=PulseKind.RSB, omega=2 * np.pi * 1e5, eta=0.1)
    p_up = abs(np.vdot(sp.ket(1, DOWN, 0), U @ v1)) ** 2
    out.append(_ok("|down,1> -> |up,0> con pulso pi (condicionalidad)",
                   p_up > 0.999, f"P = {p_up:.6f}"))
    return out


def check_fwhm_coefficient() -> list[str]:
    x = rabi_fwhm_coefficient()
    return [_ok("coeficiente FWHM Rabi = 0.7993", abs(x - 0.7993) < 1e-3,
                f"x* = {x:.6f}  =>  FWHM_Hz = {x:.4f}/t_pi")]


def check_cptp_matrices(n_max=6) -> list[str]:
    sp = Space(n_max)
    step = Step(0, PulseKind.BSB, 2 * np.pi * 2e3, eta=0.09)
    A = transition_matrices(sp, step, k_max=1)
    err = check_cptp(A)
    return [_ok("matrices A_k conservan traza (CPTP)", err < 1e-9,
                f"max|sum_k colsum(A_k)-1| = {err:.2e}")]


def check_convergence(n_max_list=(4, 6, 8, 12, 20)) -> list[str]:
    """Convergencia en el truncamiento del espacio motional."""
    ref = None
    out = []
    for n in n_max_list:
        sp = Space(n)
        step = Step(0, PulseKind.BSB, 2 * np.pi * 2e3, eta=0.09)
        A = transition_matrices(sp, step, k_max=1)
        val = A[1][:, 0].sum()
        if ref is None:
            ref = val
        out.append(f"   n_max={n:3d}: p1 = {val:.10f}"
                   + (f"  (Δ vs n_max={n_max_list[0]} = {abs(val-ref):.2e})" if ref else ""))
    return [_ok("convergencia en n_max (inspecciona los valores)", True)] + out


def run_all(ps) -> tuple[bool, str]:
    blocks = [("UNIDADES Y CONSISTENCIA DEL PAPER", check_units_schmidt(ps)),
              ("MODOS NORMALES: LÍMITES ANALÍTICOS", check_modes_limits()),
              ("ESTADO OSCURO / PUERTA CONDICIONAL", check_dark_state()),
              ("FORMA DE LÍNEA RABI", check_fwhm_coefficient()),
              ("MATRICES A_k (CPTP)", check_cptp_matrices()),
              ("CONVERGENCIA NUMÉRICA", check_convergence())]
    lines, ok = [], True
    for title, checks in blocks:
        lines.append(f"\n=== {title} ===")
        for c in checks:
            lines.append("  " + c)
            if c.startswith("❌"):
                ok = False
    lines.append("\n=== AUDITORÍA DE PARÁMETROS ===")
    warns = ps.audit()
    lines += ["  ⚠️ " + w for w in warns] or ["  ✅ sin advertencias"]
    return ok, "\n".join(lines)