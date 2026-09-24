"""
Minimal RL-QLS reproduction: CaH+ J in {1,2}, 16 states, 13-pulse library.
Physics (levels, pulse library) is real; the pulse action is idealised to a
perfect BSB pi-pulse instead of a full TDSE solve.
"""
import numpy as np
from cah_levels import manifold

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. states
states, E = [], []
for J in (1, 2):
    e, lab = manifold(J); states += lab; E += list(e)
E = np.array(E) - min(E)
NS = len(states)
mv = np.array([s[1] for s in states])

# Boltzmann at 300 K over these 16 levels: splittings ~40 kHz << kT/h ~ 6 THz
P0 = np.exp(-6.626e-34 * E / (1.381e-23 * 300)); P0 /= P0.sum()

# ---------------------------------------------------- 2. pulse library -> A
tabS2 = {1: -1.72, 2: -1.44, 3: -1.03, 4: -0.23, 5: 4.40, 6: 26.13, 7: -6.12,
         8: -6.56, 9: -7.33, 10: 9.87, 11: -9.87, 12: 13.13, 13: -13.13}
BW = 0.30                                   # kHz, spectral window of a pulse

library = []
for p, fp in sorted(tabS2.items()):
    tr = [(i, f) for i in range(NS) for f in range(NS)
          if abs(abs(mv[f] - mv[i]) - 1) < 1e-9
          and abs((E[f] - E[i]) / 1e3 - fp) < BW]
    library.append(tr)
NA = len(library)

def build_A(tr):
    """A1: population that WAS driven (ends in |f>,k=1). A0: population untouched."""
    A1 = np.zeros((NS, NS)); A0 = np.eye(NS)
    for i, f in tr:
        A1[f, i] = 1.0            # perfect pi-pulse: |v_if|^2 = 1
        A0[i, i] = 0.0            # ... so |u_ii|^2 = 0
    return A0, A1

A = [build_A(tr) for tr in library]

# ------------------------------------------------------------ 3. the MDP
ETA = 0.01                                   # purity threshold

def step(S, a):
    """Returns list of (k, prob, S_next). Eq 4a/4b, with the missing 1/p_k."""
    out = []
    for k in (0, 1):
        v = A[a][k] @ S
        p = v.sum()                          # = ||A_k S||_1, S >= 0
        if p > 1e-12:
            out.append((k, p, v / p))
    return out

def sample(S, a):
    br = step(S, a)
    ps = np.array([b[1] for b in br])
    return br[rng.choice(len(br), p=ps / ps.sum())]

done = lambda S: S.max() > 1 - ETA

# ---------------------------------------------------------- 4. baselines
def run_sweeping(cap=60):
    S, n = P0.copy(), 0
    while n < cap:
        _, _, S = sample(S, n % NA); n += 1
        if done(S): return n
    return cap

# ------------------------------------------------------- 5. numpy DQN
class MLP:
    def __init__(s, sizes):
        s.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        s.b = [np.zeros(b) for b in sizes[1:]]
        s.mW = [np.zeros_like(w) for w in s.W]; s.vW = [np.zeros_like(w) for w in s.W]
        s.mb = [np.zeros_like(x) for x in s.b]; s.vb = [np.zeros_like(x) for x in s.b]
        s.t = 0
    def __call__(s, x, cache=False):
        acts = [x]
        for i, (w, bb) in enumerate(zip(s.W, s.b)):
            x = x @ w + bb
            if i < len(s.W) - 1: x = np.maximum(x, 0)
            acts.append(x)
        return (x, acts) if cache else x
    def sgd(s, acts, dout, lr=5e-4):
        s.t += 1; gW, gb = [None]*len(s.W), [None]*len(s.b)
        for i in reversed(range(len(s.W))):
            gW[i] = acts[i].T @ dout; gb[i] = dout.sum(0)
            if i: dout = (dout @ s.W[i].T) * (acts[i] > 0)
        for i in range(len(s.W)):                      # Adam
            for g, m, v, prm in ((gW[i], s.mW, s.vW, s.W), (gb[i], s.mb, s.vb, s.b)):
                m[i] = 0.9*m[i] + 0.1*g; v[i] = 0.999*v[i] + 0.001*g*g
                mh = m[i]/(1-0.9**s.t); vh = v[i]/(1-0.999**s.t)
                prm[i] -= lr * mh/(np.sqrt(vh)+1e-8)

def train(episodes=1200, qmdp=True, cap=60, tau=1e-3):
    q, tgt = MLP([NS, 128, 128, NA]), MLP([NS, 128, 128, NA])
    tgt.W = [w.copy() for w in q.W]; tgt.b = [b.copy() for b in q.b]
    buf, lengths = [], []
    for ep in range(episodes):
        eps = 0.005 + (1.0 - 0.005) * np.exp(-ep / (0.3 * episodes))
        S, n = P0.copy(), 0
        while n < cap:
            a = rng.integers(NA) if rng.random() < eps else int(np.argmax(q(S)))
            br = step(S, a)
            _, _, S2 = sample(S, a)
            n += 1
            buf.append((S, a, -1.0, br if qmdp else [(0, 1.0, S2)], done(S2)))
            S = S2
            if done(S): break
        lengths.append(n)
        if len(buf) > 20000: buf = buf[-20000:]

        for _ in range(4):                                  # replay updates
            if len(buf) < 256: break
            batch = [buf[i] for i in rng.integers(len(buf), size=64)]
            Sb = np.array([b[0] for b in batch]); ab = [b[1] for b in batch]
            y = np.empty(64)
            for i, (_, _, r, br, dn) in enumerate(batch):
                boot = 0.0
                if not dn:                                  # Eq S18 (qMDP) / S17 (MDP)
                    for _, p, s2 in br:
                        a2 = int(np.argmax(q(s2)))          # double-DQN
                        boot += p * tgt(s2)[a2]
                y[i] = r + boot
            pred, acts = q(Sb, cache=True)
            d = np.zeros((64, NA))
            d[np.arange(64), ab] = np.clip(pred[np.arange(64), ab] - y, -1, 1) / 64
            q.sgd(acts, d)
            for i in range(len(q.W)):                       # soft target update
                tgt.W[i] += tau * (q.W[i] - tgt.W[i]); tgt.b[i] += tau * (q.b[i] - tgt.b[i])
    return q, np.array(lengths)

def evaluate(q, n=2000, cap=60):
    out = []
    for _ in range(n):
        S, k = P0.copy(), 0
        while k < cap:
            _, _, S = sample(S, int(np.argmax(q(S)))); k += 1
            if done(S): break
        out.append(k)
    return np.array(out)

if __name__ == "__main__":
    sw = np.array([run_sweeping() for _ in range(2000)])
    print(f"sweeping baseline : {sw.mean():.2f} steps  (median {np.median(sw):.0f})")
    for qmdp in (False, True):
        q, L = train(qmdp=qmdp)
        ev = evaluate(q)
        tag = "qMDP" if qmdp else " MDP"
        print(f"DQN [{tag}]        : {ev.mean():.2f} steps  (median {np.median(ev):.0f}) "
              f"| last-100 training avg {L[-100:].mean():.2f}")
