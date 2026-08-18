"""Tests de regresión. `pytest tests/ -v` antes de cada commit."""
import numpy as np
import pytest

from qlsim.constants import M_AL27, M_BE9, U_MASS
from qlsim.hilbert import DOWN, Space
from qlsim.modes import axial_modes, equilibrium_separation
from qlsim.observables import debye_waller_factor
from qlsim.params import schmidt2005
from qlsim.protocol import (PulseKind, Step, check_cptp, initial_state,
                            reset_motion, thermal_motion, transition_matrices)
from qlsim.pulses import (hamiltonian, pi_time, propagator,
                          rabi_fwhm_coefficient, rabi_fwhm_hz, rabi_lineshape)


def test_equal_mass_limit():
    m = 40 * U_MASS
    nm = axial_modes(m, m, 1e6)
    assert nm.nu[0] == pytest.approx(1e6, rel=1e-9)
    assert nm.nu[1] / nm.nu[0] == pytest.approx(np.sqrt(3), rel=1e-9)


def test_coulomb_identity():
    """C/d^3 = k/2 debe cumplirse exactamente."""
    from qlsim.constants import COULOMB_K
    k = 40 * U_MASS * (2*np.pi*1e6)**2
    d = equilibrium_separation(k)
    assert COULOMB_K / d**3 == pytest.approx(k/2, rel=1e-12)


def test_dark_state_exact():
    """⭐ H_RSB |down,0> = 0: la puerta condicional."""
    sp = Space(8)
    H = hamiltonian(sp, ion=0, kind=PulseKind.RSB, omega=1e5, eta=0.3)
    assert np.linalg.norm(H @ sp.ket(DOWN, DOWN, 0)) < 1e-12


def test_rsb_is_not_factorizable():
    """Prueba de NO factorizabilidad: mismo estado interno, distinto destino."""
    sp, om, eta = Space(8), 2*np.pi*1e5, 0.1
    U = propagator(sp, pi_time(om, eta, PulseKind.RSB), ion=0,
                   kind=PulseKind.RSB, omega=om, eta=eta)
    p0 = abs(np.vdot(sp.ket(DOWN, DOWN, 0), U @ sp.ket(DOWN, DOWN, 0)))**2
    p1 = abs(np.vdot(sp.ket(1, DOWN, 0), U @ sp.ket(DOWN, DOWN, 1)))**2
    assert p0 > 0.9999 and p1 > 0.999   # sin cambio vs volteado


def test_sideband_sqrt_scaling():
    """Omega_n = eta*Omega*sqrt(n+1): origen de la descalibración por calor."""
    sp, om, eta = Space(12), 2*np.pi*1e5, 0.05
    for n in (0, 1, 3):
        t = pi_time(om, eta, PulseKind.BSB, n=n)
        U = propagator(sp, t, ion=0, kind=PulseKind.BSB, omega=om, eta=eta)
        p = abs(np.vdot(sp.ket(1, DOWN, n+1), U @ sp.ket(DOWN, DOWN, n)))**2
        assert p > 0.99, f"n={n}: P={p}"


def test_fwhm_coefficient():
    assert rabi_fwhm_coefficient() == pytest.approx(0.7993, abs=1e-3)


def test_fwhm_reproduces_63khz():
    """La verificación de unidades del paper."""
    assert rabi_fwhm_hz(12.6e-6) == pytest.approx(63.4e3, rel=0.02)
    assert rabi_fwhm_hz(12.6e-3) < 100     # lo que daría la lectura errónea


def test_lineshape_normalisation():
    assert rabi_lineshape(np.array([0.0]), np.pi/1e-5, 1e-5)[0] == pytest.approx(1.0)


def test_cptp_conservation():
    sp = Space(8)
    for kind in (PulseKind.BSB, PulseKind.RSB, PulseKind.CARRIER):
        A = transition_matrices(sp, Step(0, kind, 2*np.pi*2e3, eta=0.09), k_max=1)
        assert check_cptp(A) < 1e-9, kind


def test_reset_motion_is_trace_preserving():
    sp = Space(6)
    rho = initial_state(sp, nbar=0.7)
    U = propagator(sp, 1e-4, ion=0, kind=PulseKind.BSB, omega=2*np.pi*3e3, eta=.1)
    rho = U @ rho @ U.conj().T
    out = reset_motion(rho, sp, nbar=0.0)
    assert np.real(np.trace(out)) == pytest.approx(1.0, abs=1e-10)
    # y el movimiento queda en |0>
    assert sp.populations(out)[:, :, 0].sum() == pytest.approx(1.0, abs=1e-10)


def test_thermal_normalised():
    for nb in (0.0, 0.1, 2.0):
        assert thermal_motion(20, nb).sum() == pytest.approx(1.0)


def test_debye_waller_limits():
    assert debye_waller_factor(0.0, 0) == pytest.approx(1.0)
    # eta pequeño: 1 - eta^2 (n+1/2)
    eta, n = 0.02, 3
    assert debye_waller_factor(eta, n) == pytest.approx(1 - eta**2*(n+0.5), rel=1e-2)


def test_parameter_audit_flags_assumptions():
    warns = schmidt2005().audit()
    assert any("dk_raman_logic" in w for w in warns)
    assert any("nbar_radial" in w for w in warns)


def test_derived_mode_frequency_is_sane():
    nm = axial_modes(M_BE9, M_AL27, 3.80e6)
    assert 2.0e6 < nm.nu[0] < 3.0e6      # in-phase entre 2 y 3 MHz
    assert nm.nu[1] > nm.nu[0]