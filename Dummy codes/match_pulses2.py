import numpy as np
from cah_levels import manifold

states, E = [], []
for J in (1, 2):
    e, lab = manifold(J)
    states += lab
    E += list(e)
E = np.array(E); E -= E[0]
N = len(states)
mv = np.array([s[1] for s in states])
nm = lambda i: f"|{states[i][0]},{states[i][1]:+.1f},{states[i][2]}>"

tabS2 = {1: -1.72, 2: -1.44, 3: -1.03, 4: -0.23, 5: 4.40, 6: 26.13, 7: -6.12,
         8: -6.56, 9: -7.33, 10: 9.87, 11: -9.87, 12: 13.13, 13: -13.13}

trans = [(i, f, (E[f] - E[i]) / 1e3)
         for i in range(N) for f in range(N)
         if abs(abs(mv[f] - mv[i]) - 1.0) < 1e-9]

print("pulse  f_paper   identified transition(s)                        f_calc")
tot = 0
for p, fp in sorted(tabS2.items()):
    hits = [t for t in trans if abs(t[2] - fp) < 0.45]
    tot += len(hits)
    if not hits:
        print(f"{p:>4} {fp:+8.2f}   -- none within 0.45 kHz --")
    for k, (i, j, df) in enumerate(hits):
        tag = f"{p:>4} {fp:+8.2f}" if k == 0 else " " * 13
        print(f"{tag}   {nm(i)} -> {nm(j):<16}            {df:+7.2f}")
print(f"\n{tot} transitions assigned across 13 pulses")
