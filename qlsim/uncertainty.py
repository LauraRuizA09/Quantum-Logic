"""Parámetros con incertidumbre y procedencia + propagación Monte Carlo.

FILOSOFÍA
---------
Un número sin procedencia es una opinión. Un número sin incertidumbre es
una mentira. Este módulo hace imposible violar ninguna de las dos reglas.

Usamos Monte Carlo (no propagación lineal) porque:
  * las observables (contraste, P_FP, anchos de línea) son NO lineales;
  * queremos intervalos de confianza asimétricos cuando existan;
  * nos permite detectar correlaciones y modos bimodales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, Optional

import numpy as np


class Provenance(Enum):
    """De dónde viene el número. Determina cuánto puedes confiar en él."""
    CODATA = "codata"      # constante fundamental, incertidumbre despreciable
    REPORTED = "reported"  # publicado explícitamente en el paper
    DERIVED = "derived"    # calculado a partir de valores reportados
    MEASURED = "measured"  # calibración propia de nuestro laboratorio
    ASSUMED = "assumed"    # ⚠️ NO está en el paper: hipótesis nuestra


@dataclass(frozen=True)
class Param:
    """Cantidad escalar con incertidumbre 1-sigma, unidad y trazabilidad.

    Parameters
    ----------
    name : identificador legible.
    value : valor central (unidades SI).
    std : incertidumbre estándar (1 sigma), misma unidad. 0.0 => exacto.
    unit : string SI, solo documental (no hacemos álgebra de unidades).
    provenance : ver `Provenance`.
    source : DOI o referencia precisa (página/tabla/figura).
    note : cualquier advertencia (p.ej. "asumido: geometría no reportada").
    dist : 'normal' | 'uniform' | 'lognormal'. Para cotas usa 'uniform'.

    Examples
    --------
    >>> p = Param("tau_probe", 12.6e-6, 0.1e-6, "s", Provenance.REPORTED,
    ...           "10.1126/science.1114375, Fig. 3A")
    >>> float(p)
    1.26e-05
    """
    name: str
    value: float
    std: float = 0.0
    unit: str = ""
    provenance: Provenance = Provenance.ASSUMED
    source: str = ""
    note: str = ""
    dist: str = "normal"

    def __float__(self) -> float:
        return float(self.value)

    @property
    def rel(self) -> float:
        """Incertidumbre relativa (0 si el valor es 0)."""
        return self.std / abs(self.value) if self.value else 0.0

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Extrae `size` muestras de la distribución del parámetro."""
        if self.std == 0.0:
            return np.full(size, self.value)
        if self.dist == "normal":
            return rng.normal(self.value, self.std, size)
        if self.dist == "uniform":
            # std interpretada como semi-anchura de la caja
            return rng.uniform(self.value - self.std, self.value + self.std, size)
        if self.dist == "lognormal":
            # útil para tasas de calentamiento (positivas, cola larga)
            sigma = np.sqrt(np.log1p((self.std / self.value) ** 2))
            mu = np.log(self.value) - 0.5 * sigma**2
            return rng.lognormal(mu, sigma, size)
        raise ValueError(f"Distribución desconocida: {self.dist}")

    def __repr__(self) -> str:
        flag = " ⚠️" if self.provenance is Provenance.ASSUMED else ""
        return (f"{self.name} = {self.value:.6g} ± {self.std:.3g} {self.unit} "
                f"[{self.provenance.value}]{flag}")


class ParamSet:
    """Colección de `Param` con muestreo conjunto para Monte Carlo."""

    def __init__(self, params: Iterable[Param]):
        self._p: Dict[str, Param] = {p.name: p for p in params}

    def __getitem__(self, name: str) -> Param:
        return self._p[name]

    def __contains__(self, name: str) -> bool:
        return name in self._p

    def value(self, name: str) -> float:
        return self._p[name].value

    def add(self, p: Param) -> None:
        self._p[p.name] = p

    def override(self, name: str, value: float, std: float = 0.0,
                 note: str = "override local") -> None:
        """Sustituye un valor manteniendo el resto de metadatos.

        Úsalo para pasar de los parámetros del paper a los de NUESTRA trampa:
        >>> ps.override("nu_mode_inphase", 2.10e6, 5e3, "medido 2026-08-18")
        """
        old = self._p[name]
        self._p[name] = Param(name, value, std, old.unit, Provenance.MEASURED,
                              old.source, note, old.dist)

    def sample(self, size: int, seed: int = 0) -> Dict[str, np.ndarray]:
        """Muestreo INDEPENDIENTE de todos los parámetros.

        ⚠️ LIMITACIÓN CONOCIDA: asume no correlación. Si dos parámetros
        provienen de la misma calibración (p.ej. las intensidades sigma y pi
        calibradas con el mismo Stark shift), esto SUBESTIMA o SOBREESTIMA la
        incertidumbre. Documéntalo en el paper. Para correlaciones, pasa una
        matriz de covarianza a `sample_correlated` (ejercicio E7).
        """
        rng = np.random.default_rng(seed)
        return {k: p.sample(size, rng) for k, p in self._p.items()}

    def table(self, only_assumed: bool = False) -> str:
        """Tabla imprimible: es literalmente el apéndice de tu paper."""
        rows = ["| parámetro | valor | ± 1σ | unidad | procedencia | fuente |",
                "|---|---|---|---|---|---|"]
        for p in sorted(self._p.values(), key=lambda x: x.name):
            if only_assumed and p.provenance is not Provenance.ASSUMED:
                continue
            rows.append(f"| `{p.name}` | {p.value:.6g} | {p.std:.3g} | "
                        f"{p.unit} | {p.provenance.value} | {p.source} |")
        return "\n".join(rows)

    def audit(self) -> list[str]:
        """Devuelve advertencias: todo lo que sea ASSUMED o sin fuente."""
        w = []
        for p in self._p.values():
            if p.provenance is Provenance.ASSUMED:
                w.append(f"ASSUMED: {p.name} — {p.note or 'sin justificación'}")
            if p.provenance in (Provenance.REPORTED, Provenance.DERIVED) and not p.source:
                w.append(f"SIN FUENTE: {p.name}")
            if p.std == 0.0 and p.provenance not in (Provenance.CODATA,):
                w.append(f"SIN INCERTIDUMBRE: {p.name}")
        return w


def monte_carlo(fn: Callable[..., np.ndarray], ps: ParamSet, n: int = 4000,
                seed: int = 0, **fixed) -> Dict[str, np.ndarray]:
    """Propaga incertidumbres a través de `fn` por Monte Carlo vectorizado.

    `fn(**params, **fixed)` debe aceptar arrays de longitud n y devolver
    un array de forma (n,) o (n, m).

    Returns
    -------
    dict con 'samples', 'median', 'lo68', 'hi68', 'lo95', 'hi95', 'mean', 'std'.
    """
    draws = ps.sample(n, seed=seed)
    out = np.atleast_2d(np.asarray(fn(**draws, **fixed)))
    if out.shape[0] != n and out.shape[-1] == n:
        out = out.T                      # normaliza a (n, m)
    q = np.percentile(out, [2.5, 16, 50, 84, 97.5], axis=0)
    return dict(samples=out, lo95=q[0], lo68=q[1], median=q[2],
                hi68=q[3], hi95=q[4], mean=out.mean(0), std=out.std(0, ddof=1))


def shot_noise(p: np.ndarray, n_rep: int, seed: int = 0) -> np.ndarray:
    """Añade ruido binomial de N repeticiones a una probabilidad ideal.

    Esto es OBLIGATORIO para comparar con datos: el paper promedia 700
    repeticiones por punto, luego sigma_min = sqrt(p(1-p)/700) ~ 1.9% en p=0.5.
    Sin esto, tu "acuerdo con el experimento" no significa nada.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(p, 0.0, 1.0)
    return rng.binomial(n_rep, p) / n_rep


def binomial_sigma(p: np.ndarray, n_rep: int) -> np.ndarray:
    """Barra de error estadística esperada (aprox. normal)."""
    return np.sqrt(np.clip(p, 0, 1) * (1 - np.clip(p, 0, 1)) / n_rep)