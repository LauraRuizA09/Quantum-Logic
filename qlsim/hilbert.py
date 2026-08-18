# ============================ qlsim/hilbert.py ============================
"""Construcción del espacio de Hilbert  S (2) ⊗ L (2) ⊗ movimiento (n_max).

Convención de estados internos:  index 0 = |down> (=|S> brillante en Ca+),
                                 index 1 = |up>   (=|D> oscuro).
Ojo: en Schmidt 2005 |down> = estado base; en Chou 2017 |D> es el oscuro.
Mantén la convención UNA sola vez y documéntala.
"""
from __future__ import annotations

import numpy as np

DOWN, UP = 0, 1


class Space:
    """Gestiona dimensiones y operadores del espacio conjunto."""

    def __init__(self, n_max: int = 6):
        if n_max < 2:
            raise ValueError("n_max >= 2 (necesitas al menos |0> y |1>)")
        self.n_max = n_max
        self.dim = 2 * 2 * n_max
        self.shape = (2, 2, n_max)

    # ---------- operadores elementales ----------
    def _kron3(self, a, b, c):
        return np.kron(np.kron(a, b), c)

    @property
    def I2(self):
        return np.eye(2)

    @property
    def Imot(self):
        return np.eye(self.n_max)

    def sigma_plus(self, ion: int) -> np.ndarray:
        """|up><down| del ion 0 (spec) o 1 (logic)."""
        sp = np.array([[0.0, 0.0], [1.0, 0.0]])   # |1><0|
        ops = [self.I2, self.I2]
        ops[ion] = sp
        return self._kron3(ops[0], ops[1], self.Imot)

    def sigma_z(self, ion: int) -> np.ndarray:
        sz = np.diag([-1.0, 1.0])
        ops = [self.I2, self.I2]
        ops[ion] = sz
        return self._kron3(ops[0], ops[1], self.Imot)

    @property
    def a(self) -> np.ndarray:
        """Operador de aniquilación del modo de transferencia."""
        n = self.n_max
        a = np.zeros((n, n))
        for k in range(1, n):
            a[k - 1, k] = np.sqrt(k)
        return self._kron3(self.I2, self.I2, a)

    @property
    def n_op(self) -> np.ndarray:
        return self._kron3(self.I2, self.I2, np.diag(np.arange(self.n_max, dtype=float)))

    # ---------- estados y proyectores ----------
    def ket(self, s: int, l: int, n: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=complex)
        v[np.ravel_multi_index((s, l, n), self.shape)] = 1.0
        return v

    def projector_motion(self, n: int) -> np.ndarray:
        """POVM element O_n = |n><n| del movimiento (el del paper de RL)."""
        pn = np.zeros((self.n_max, self.n_max))
        pn[n, n] = 1.0
        return self._kron3(self.I2, self.I2, pn)

    def projector_internal(self, ion: int, state: int) -> np.ndarray:
        p = np.zeros((2, 2))
        p[state, state] = 1.0
        ops = [self.I2, self.I2]
        ops[ion] = p
        return self._kron3(ops[0], ops[1], self.Imot)

    def populations(self, rho: np.ndarray) -> np.ndarray:
        """Diagonal de rho reorganizada como (2, 2, n_max)."""
        return np.real(np.diag(rho)).reshape(self.shape)