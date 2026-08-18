# Quantum-Logic

# QLS Replication Suite

Modelo numérico validado de espectroscopía por lógica cuántica (QLS),
construido como escalera de tres peldaños:

| Peldaño | Paper | DOI |
|---|---|---|
| 1 | Schmidt et al., *Science* **309**, 749 (2005) | `10.1126/science.1114375` |
| 2 | Chou et al., *Nature* **545**, 203 (2017) | `10.1038/nature22338` |
| 3 | Pipi, Tao et al., arXiv:2410.11839v4 (2026) | `10.48550/arXiv.2410.11839` |

## Estado
- [x] Peldaño 1: modos normales, mapeo QLS, Fig. 3A/3B/3C, presupuesto de error
- [ ] Peldaño 2: CaH+ hyperfine, bombeo óptico, Rabi/Ramsey moleculares
- [ ] Peldaño 3: matrices A_k → MDP → RL
- [ ] Reproducción con parámetros medidos de NUESTRA trampa

## Uso
```bash
pip install -r requirements.txt
python scripts/00_validate.py     # SIEMPRE primero
python scripts/02_fig3a_spectrum.py
pytest tests/ -v
```

## Reglas del repo
1. Ningún número entra al código sin `Param(..., provenance=..., source=DOI)`.
2. Ninguna observable se reporta sin banda de incertidumbre.
3. `scripts/00_validate.py` debe pasar antes de cada commit.
