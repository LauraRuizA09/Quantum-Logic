import numpy as np

# ---- constants -------------------------------------------------------------
h    = 6.62607015e-34          # J s
muN  = 5.0507837393e-27        # J/T  (nuclear magneton)
R    = 144e9                   # Hz   rotational constant, 40CaH+  (Chou 2017)
gI   = 5.5856946893            # proton g-factor
B    = 0.36e-3                 # T

# per-J values from Chou 2017 Extended Data Table 2 (v=0)
CIJ  = {0:8.27e3, 1:8.26e3, 2:8.26e3, 3:8.26e3, 4:8.26e3, 5:8.25e3, 6:8.25e3}
GJ   = {0:-1.35, 1:-1.35,   2:-1.35,  3:-1.34,  4:-1.34,  5:-1.34,  6:-1.34}

I = 0.5

def ang_ops(j):
    """Jz, J+, J- in the |j,m> basis, m descending from +j."""
    m = np.arange(j, -j - 1, -1.0)
    n = len(m)
    Jz = np.diag(m)
    Jp = np.zeros((n, n))
    for k in range(1, n):                       # <m+1|J+|m>
        mm = m[k]
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - mm * (mm + 1))
    return Jz, Jp, Jp.T, m

def manifold(J):
    """Diagonalize H/h for one rotational manifold. Returns (energies_Hz, labels)."""
    Jz, Jp, Jm, mJ = ang_ops(J)
    Iz, Ip, Im, mI = ang_ops(I)
    nJ, nI = len(mJ), len(mI)
    kron = np.kron

    # H/h  =  R J(J+1)  - g muN B mJ/h  - gI muN B mI/h  - cIJ (I.J)
    IdJ, IdI = np.eye(nJ), np.eye(nI)
    IdotJ = kron(Jz, Iz) + 0.5 * (kron(Jp, Im) + kron(Jm, Ip))

    H = (R * J * (J + 1)) * np.eye(nJ * nI)
    H += -GJ[J] * muN * B / h * kron(Jz, IdI)
    H += -gI    * muN * B / h * kron(IdJ, Iz)
    H += -CIJ[J] * IdotJ

    # m = mJ + mI is a good quantum number -> block-diagonalize by m
    mtot = np.add.outer(mJ, mI).ravel()
    evals, labels = [], []
    for mval in sorted(set(np.round(mtot, 6)), reverse=True):
        idx = np.where(np.isclose(mtot, mval))[0]
        blk = H[np.ix_(idx, idx)]
        w = np.linalg.eigvalsh(blk)
        # xi = +/- : lower-energy root of a 2x2 block is xi=-, higher is xi=+
        # (extreme states m = +/-(J+1/2) are 1x1: pure product states)
        if len(w) == 1:
            evals.append(w[0]); labels.append((J, mval, '+' if mval > 0 else '-'))
        else:
            evals.append(w[0]); labels.append((J, mval, '+'))   # E = .. - X
            evals.append(w[1]); labels.append((J, mval, '-'))   # E = .. + X
    evals = np.array(evals)
    order = np.argsort(evals)
    return evals[order], [labels[i] for i in order]

for J in (1, 2):
    E, lab = manifold(J)
    E0 = E.min()
    print(f"--- J = {J} ---   ({len(E)} states, spread {(E.max()-E.min())/1e3:.2f} kHz)")
    for e, l in zip(E, lab):
        print(f"   |{l[0]},{l[1]:+.1f},{l[2]}>   {(e-E0)/1e3:9.3f} kHz")
    # target transition |J,-J+1/2,-> <-> |J,-J-1/2,->
    d = dict(zip([(a, round(b, 1), c) for a, b, c in lab], E))
    f = d[(J, -J + 0.5, '-')] - d[(J, -J - 0.5, '-')]
    print(f"   target |J,-J+1/2,-> - |J,-J-1/2,-> = {f/1e3:+.3f} kHz\n")
