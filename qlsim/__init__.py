"""qlsim — modelo numérico validado de espectroscopía por lógica cuántica (QLS).

Peldaño 1: Schmidt et al., Science 309, 749-752 (2005).
           DOI 10.1126/science.1114375

Uso típico
----------
>>> import qlsim
>>> ps = qlsim.default_paramset()          # params del paper + modos derivados
>>> ok, report = qlsim.validate.run_all(ps)
>>> print(report)

Orden de importación
--------------------
constants -> uncertainty -> modes -> hilbert -> pulses -> protocol
          -> observables -> errors -> params -> validate
No hay dependencias circulares: `params` importa `modes` de forma perezosa
(dentro de `add_derived_modes`) precisamente para mantener este orden limpio.
"""
from __future__ import annotations

import pathlib as _pathlib

__version__ = "0.1.0"

# ---- capas del modelo, en orden de dependencia ----
from . import constants          # noqa: E402
from . import uncertainty        # noqa: E402
from . import modes              # noqa: E402
from . import hilbert            # noqa: E402
from . import pulses             # noqa: E402
from . import protocol           # noqa: E402
from . import observables        # noqa: E402
from . import errors             # noqa: E402
from . import params             # noqa: E402
from . import validate           # noqa: E402

# ---- reexportes de conveniencia (los que se usan en TODOS los scripts) ----
from .errors import (contrast_budget, contrast_total, error_ranking,  # noqa
                     false_positive_rate, max_steps_budget)
from .hilbert import DOWN, LOGIC, SPEC, UP, Space                     # noqa
from .modes import axial_modes, dk_single_beam, dk_two_beam, radial_modes  # noqa
from .params import add_derived_modes, our_trap, schmidt2005          # noqa
from .protocol import PulseKind, Step, check_cptp, qls_mapping, transition_matrices  # noqa
from .pulses import pi_time, rabi_fwhm_hz                             # noqa
from .uncertainty import (Param, ParamSet, Provenance, binomial_sigma,  # noqa
                          monte_carlo, sensitivity, shot_noise)

__all__ = [
    "__version__", "ROOT", "FIGDIR",
    # módulos
    "constants", "uncertainty", "modes", "hilbert", "pulses", "protocol",
    "observables", "errors", "params", "validate",
    # helpers
    "default_paramset", "figpath", "banner",
    # reexportes
    "Param", "ParamSet", "Provenance", "monte_carlo", "sensitivity",
    "shot_noise", "binomial_sigma", "schmidt2005", "add_derived_modes",
    "our_trap", "Space", "DOWN", "UP", "SPEC", "LOGIC", "PulseKind", "Step",
    "qls_mapping", "transition_matrices", "check_cptp", "pi_time",
    "rabi_fwhm_hz", "axial_modes", "radial_modes", "dk_single_beam",
    "dk_two_beam", "contrast_budget", "contrast_total", "error_ranking",
    "false_positive_rate", "max_steps_budget",
]

# ============================ rutas del repo ============================
ROOT = _pathlib.Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"


def figpath(name: str) -> str:
    """Ruta absoluta de una figura, creando `figures/` si no existe.

    Evita repetir el mismo boilerplate en los nueve scripts.

    >>> qlsim.figpath("06_error_budget.png")
    '/.../qls-replication/figures/06_error_budget.png'
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return str(FIGDIR / name)


# ============================ atajos de alto nivel ============================
def default_paramset(n_mc: int = 2000, seed: int = 3) -> ParamSet:
    """ParamSet del peldaño 1 CON las cantidades derivadas de modos normales.

    Equivale a `add_derived_modes(schmidt2005())`. Añade `nu_mode`,
    `nu_mode_oop`, `eta_axial_spec`, `eta_axial_logic`, `eta_radial`.

    ⚠️ LIMITACIÓN DECLARADA: al convertir las cantidades derivadas en Params
    independientes se pierde su correlación con los padres (masas, nu_z,
    lambda, theta). Para observables que dependan fuertemente de DOS de ellas
    a la vez, recalcula la cadena completa dentro de la función modelo
    (ver ejercicio E7).
    """
    return add_derived_modes(schmidt2005(), n_mc=n_mc, seed=seed)


def banner(title: str, width: int = 72, char: str = "=") -> str:
    """Encabezado uniforme para la salida de los scripts."""
    return f"\n{char * width}\n {title}\n{char * width}"