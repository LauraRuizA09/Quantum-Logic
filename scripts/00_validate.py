# ======================== scripts/00_validate.py ==========================
"""SIEMPRE ejecuta esto primero. Si algo falla, no confíes en las figuras."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qlsim.params import schmidt2005
from qlsim.validate import run_all

ps = schmidt2005()
ok, report = run_all(ps)
print(report)
print("\n=== TABLA DE PARÁMETROS (pégala en el apéndice) ===")
print(ps.table())
print("\n=== SOLO LOS ASUMIDOS (lo que hay que medir en nuestra trampa) ===")
print(ps.table(only_assumed=True))
sys.exit(0 if ok else 1)