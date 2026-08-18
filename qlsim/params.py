"""Parámetros de Schmidt et al., Science 309, 749 (2005).
DOI: 10.1126/science.1114375

ADVERTENCIA CRÍTICA DE UNIDADES
-------------------------------
El texto extraído del PDF contiene DOS errores de conversión (pérdida del
símbolo 'µ'). Los corregimos aquí y `validate.py` verifica la corrección:

  (a) "63 kHz for t_p = 12.6 ms"  ->  t_p = 12.6 µs
      Prueba: FWHM_Hz = 0.7993/t_p  =>  0.7993/12.6e-6 = 63.4 kHz  ✓
              con 12.6 ms daría 63 Hz  ✗  (y la secuencia total dura ~1 ms)

  (b) "coherence time 118 ms, ~305-ms lifetime of 3P1"  ->  118 µs, ~306 µs
      Prueba: Gamma/2pi = 520 Hz  =>  tau = 1/(2pi*520) = 306 µs  ✓

LECCIÓN: toda cifra debe pasar un test de consistencia INDEPENDIENTE antes
de usarse como referencia de benchmarking.
"""
import numpy as np

from .constants import M_AL27, M_BE9
from .uncertainty import Param as P
from .uncertainty import ParamSet, Provenance as Pr

DOI = "10.1126/science.1114375"


def schmidt2005() -> ParamSet:
    """Devuelve el ParamSet completo del peldaño 1."""
    return ParamSet([
        # ---------------- Especies ----------------
        P("m_logic", M_BE9, 0.0, "kg", Pr.CODATA,
          "AME2020", "9Be+ = ion lógico"),
        P("m_spec", M_AL27, 0.0, "kg", Pr.CODATA,
          "AME2020", "27Al+ = ion de espectroscopía"),

        # ---------------- Trampa: frecuencias seculares de UN 9Be+ ----------
        # El paper: "wz ~ 2pi x 3.8 MHz, wx ~ 2pi x 13.8, wy ~ 2pi x 14.9".
        # El "~" justifica una incertidumbre del orden del 1%.
        P("nu_z_logic", 3.80e6, 0.04e6, "Hz", Pr.REPORTED, DOI,
          "eje débil (axial); '~' -> asignamos 1%"),
        P("nu_x_logic", 13.8e6, 0.15e6, "Hz", Pr.REPORTED, DOI, "radial"),
        P("nu_y_logic", 14.9e6, 0.15e6, "Hz", Pr.REPORTED, DOI, "radial"),

        # ---------------- Transición sondeada en 27Al+ ----------------------
        # |1S0,F=5/2,mF=5/2> -> |3P1,F'=7/2,mF'=7/2>, NO es la de reloj.
        P("lambda_probe", 267e-9, 1e-9, "m", Pr.REPORTED, DOI,
          "'~267 nm'; incertidumbre conservadora"),
        P("gamma_3p1_hz", 520.0, 10.0, "Hz", Pr.REPORTED,
          f"{DOI}; linewidth de Traebert et al. 1999",
          "Gamma/2pi = ~520 Hz  =>  tau = 306 us"),

        # ---------------- Geometría de haces ----------------
        P("theta_probe", np.deg2rad(45.0), np.deg2rad(2.0), "rad", Pr.REPORTED,
          DOI, "haces de prueba a 45 grados del eje débil"),
        # El Delta-k Raman del 9Be+ NO se reporta -> ASSUMED, y hay que decirlo.
        P("dk_raman_logic", np.sqrt(2) * 2 * np.pi / 313e-9,
          0.15 * np.sqrt(2) * 2 * np.pi / 313e-9, "1/m", Pr.ASSUMED, "",
          "Raman 9Be+ a 313 nm, geometría 90deg asumida (no reportada). "
          "15% de incertidumbre. SOLO afecta a eta_logic.", dist="normal"),

        # ---------------- Pulso de interrogación (Fig. 3A) ------------------
        P("t_probe", 12.6e-6, 0.2e-6, "s", Pr.REPORTED, f"{DOI}, Fig. 3A",
          "CORREGIDO de 'ms' a 'us' (ver docstring del módulo)"),
        P("fwhm_reported", 63.0e3, 1.0e3, "Hz", Pr.REPORTED, f"{DOI}, Fig. 3A",
          "ancho ajustado, limitado por transformada de Fourier"),

        # ---------------- Contraste y coherencia (Fig. 3B) ------------------
        P("contrast_reported", 0.93, 0.02, "", Pr.REPORTED, f"{DOI}, Fig. 3A",
          "normalizado al contraste de una banda lateral del 9Be+"),
        P("t_coh_reported", 118e-6, 5e-6, "s", Pr.REPORTED, f"{DOI}, Fig. 3B",
          "CORREGIDO de 'ms' a 'us'"),

        # ---------------- Estadística ----------------
        P("n_rep", 700, 0, "", Pr.REPORTED, DOI,
          "700 repeticiones por punto de datos"),
        P("t_sequence", 1.0e-3, 0.1e-3, "s", Pr.REPORTED, DOI,
          "'una secuencia experimental toma ~1 ms'"),

        # ---------------- Zeeman ----------------
        P("b_field", 3.0e-3, 0.05e-3, "T", Pr.REPORTED, f"{DOI}, Fig. 4B",
          "elegido para resolver espectralmente todas las transiciones"),

        # ---------------- Imperfecciones (NO reportadas: ASSUMED) -----------
        P("nbar_radial", 0.5, 0.4, "", Pr.ASSUMED, "",
          "n medio de los modos radiales enfriados solo por Doppler. "
          "Fuente del factor Debye-Waller. Estimación gruesa; dominará el "
          "presupuesto de contraste -> hay que MEDIRLO en nuestra trampa.",
          dist="lognormal"),
        P("heating_rate", 10.0, 8.0, "1/s", Pr.ASSUMED, "",
          "dn/dt del modo de transferencia. No reportado en 2005. "
          "Chou 2017 (10.1038/nature22338) da <1 cuanto/100 ms = <10 /s "
          "para el modo out-of-phase.", dist="lognormal"),
        P("detection_fidelity", 0.999, 0.001, "", Pr.ASSUMED, "",
          "no reportada en 2005; Chou 2017 alcanza >0.9999 en 40Ca+"),
        P("pi_pulse_error", 0.01, 0.01, "", Pr.ASSUMED, "",
          "error por pulso pi (miscalibración de area). 3 pulsos pi en la "
          "secuencia de mapeo.", dist="lognormal"),
        P("nbar_transfer_after_cooling", 0.02, 0.02, "", Pr.ASSUMED, "",
          "n residual del modo de transferencia tras enfriado de banda "
          "lateral resuelta.", dist="lognormal"),
    ])


def our_trap(base: ParamSet | None = None, **measured) -> ParamSet:
    """Plantilla para reemplazar parámetros del paper por los NUESTROS.

    Examples
    --------
    >>> ps = our_trap(nu_z_logic=(2.10e6, 5e3), heating_rate=(3.2, 0.4))
    """
    ps = base or schmidt2005()
    for name, spec in measured.items():
        val, std = spec if isinstance(spec, tuple) else (spec, 0.0)
        ps.override(name, val, std, note="calibración in-house")
    return ps