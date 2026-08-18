# Ejercicios guiados (modifica → grafica → concluye)

Formato de respuesta: **qué cambié → qué observé → qué aprendí → qué implica
para nuestra reproducción**.

## E1 — La puerta condicional se rompe con el calor  ⭐ (empieza aquí)
En `05_full_protocol.py`, barre `nbar_init` = 0, 0.05, 0.2, 0.5, 1.0 y grafica
el error máximo de mapeo.
- ¿A qué `nbar` el error supera el 1 %?
- Relaciónalo con el factor `sqrt(n+1)` de `pi_time`.
- **Conecta con tu pregunta ❹.**

## E2 — El techo de pasos de tu política de RL
En `07_heating_budget.py` cambia `OUR_HEATING` y `OUR_STEP_TIME`.
- ¿Qué combinación te permite los 83 pulsos que RL-QLS necesita para H3O+?
- ¿Y los ~8.3 pasos del caso CaH+ J∈{1,2}?
- Grafica `N_max` vs `ndot` en log-log. ¿Cuál es la pendiente y por qué?

## E3 — Rabi exacta vs sinc²
En `02_fig3a_spectrum.py`: ajusta los MISMOS datos con ambas formas de línea.
- ¿Cuánto sesga el centro de la resonancia elegir la forma equivocada?
- ¿En qué régimen de área de pulso son indistinguibles?
- Chou 2017 ajusta sinc²; Schmidt 2005 ajusta la Rabi exacta. ¿Quién tiene
  razón en cada caso?

## E4 — El amortiguamiento NO es exponencial
En `03_fig3b_rabi.py`, extiende `t` a 1 ms y grafica en escala log-y el
residuo entre `rabi_flopping` (suma térmica exacta) y el ajuste senoidal.
- ¿Cuándo divergen?
- ¿Qué sesgo introduce reportar `T_coh` de un ajuste exponencial?

## E5 — Convergencia de truncamientos
Verifica: (a) `n_max` en `Space`, (b) `n_max_dw` en `rabi_flopping`,
(c) truncamiento térmico en `thermal_motion`.
- Escribe una tabla `valor vs truncamiento` y define el criterio de
  convergencia (¿10⁻⁴ relativo?). **Esto va en tu paper.**

## E6 — ¿Quién domina el presupuesto de contraste?
En `03_fig3b_rabi.py`, calcula la derivada de `C_total` respecto de cada
parámetro (diferencias finitas) × su incertidumbre.
- Ordénalos: ese ranking te dice **qué medir primero en el laboratorio**.

## E7 — Correlaciones (avanzado)
`ParamSet.sample` asume independencia. Implementa `sample_correlated` con una
matriz de covarianza y comprueba el efecto de correlacionar
`nbar_radial` con `nu_x_logic` (ambos vienen del mismo enfriado Doppler).

## E8 — Validar la aproximación de "coherencia destruida"  ⭐⭐
`reset_motion` asume destrucción TOTAL de coherencias.
Implementa el enfriado como un canal Lindblad real con tasa finita y compara
la evolución de poblaciones tras 5 pasos.
- La discrepancia máxima es una **incertidumbre sistemática** de tu modelo.
- **Esto justifica formalmente el uso de vectores de población en el paper 3.**

## E9 — Razón de masas
Cambia `m_spec` a 40CaH+ (peldaño 2) y a un ion de 200 u.
- Grafica `eta` y la separación de frecuencias de modo vs razón de masas.
- Verifica la afirmación del paper: "se han hecho experimentos de lógica
  cuántica a razones de masa de hasta 3".

## E10 — Modo in-phase vs out-of-phase
Añade una función que module la sensibilidad al ruido de campo eléctrico
UNIFORME (∝ suma de amplitudes) y a un GRADIENTE (∝ diferencia).
- Verifica cuantitativamente por qué Chou 2017 elige el out-of-phase.