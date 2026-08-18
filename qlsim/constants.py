"""Constantes físicas (CODATA 2018) y masas isotópicas.

Todas en unidades SI. Las masas atómicas vienen de AME2020.
NUNCA hardcodees constantes en otro módulo: impórtalas de aquí.
"""
import numpy as np

# --- CODATA 2018 (exactas por definición del SI) ---
H = 6.62607015e-34          # J s      (exacta)
HBAR = H / (2 * np.pi)      # J s
E_CHARGE = 1.602176634e-19  # C        (exacta)
K_B = 1.380649e-23          # J/K      (exacta)
C_LIGHT = 299792458.0       # m/s      (exacta)

# --- CODATA 2018 (con incertidumbre relativa ~1e-10, despreciable aquí) ---
EPS0 = 8.8541878128e-12     # F/m
U_MASS = 1.66053906660e-27  # kg  (unidad de masa atómica)
MU_N = 5.0507837461e-27     # J/T (magnetón nuclear, para el peldaño 2)

# --- Masas isotópicas [u] (AME2020) ---
M_BE9 = 9.0121831 * U_MASS    # 9Be+  (ion lógico, Schmidt 2005)
M_AL27 = 26.98153853 * U_MASS # 27Al+ (ion de espectroscopía)
M_CA40 = 39.9625909 * U_MASS  # 40Ca+ (peldaño 2)
M_CAH40 = M_CA40 + 1.00782503 * U_MASS  # 40CaH+ (menos e-, despreciable)

# Constante de Coulomb e^2/(4 pi eps0) — aparece en todos los modos normales
COULOMB_K = E_CHARGE**2 / (4 * np.pi * EPS0)  # J m