# ==================== scripts/05_full_protocol.py =========================
"""Simulación completa del protocolo de 4 pasos + matrices A_k del paper 3."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from qlsim.hilbert import Space
from qlsim.modes import axial_modes, dk_single_beam
from qlsim.params import schmidt2005
from qlsim.protocol import (PulseKind, Step, check_cptp, qls_mapping,
                            transition_matrices)
from qlsim.pulses import check_lamb_dicke

ps = schmidt2005()
nm = axial_modes(ps.value("m_logic"), ps.value("m_spec"), ps.value("nu_z_logic"))
eta_s = nm.lamb_dicke(1, 0, dk_single_beam(ps.value("lambda_probe"), np.deg2rad(45)))
eta_l = nm.lamb_dicke(0, 0, ps.value("dk_raman_logic"))
sp = Space(n_max=6)
print(check_lamb_dicke(max(eta_s, eta_l), sp.n_max) or "✅ régimen Lamb-Dicke OK")

om_s = np.pi / ps.value("t_probe")
om_l = 2*np.pi * 200e3

# --- barrido: ¿el protocolo transfiere fielmente alpha^2? ---
areas = np.linspace(0, 2*np.pi, 41)
true_, meas_ = [], []
for a in areas:
    r = qls_mapping(sp, om_s, eta_s, om_l, eta_l,
                    probe_duration=a/om_s,
                    nbar_init=ps.value("nbar_transfer_after_cooling"),
                    detection_fidelity=ps.value("detection_fidelity"),
                    pi_error=ps.value("pi_pulse_error"))
    true_.append(r.alpha2_true); meas_.append(1 - r.p_bright)
true_, meas_ = np.array(true_), np.array(meas_)
print(f"\nerror máximo de mapeo |medido - real| = {np.abs(meas_-true_).max():.4f}")

# --- matrices A_k: el puente al paper de RL ---
step = Step(0, PulseKind.BSB, 2*np.pi*2.087e3, eta=0.09, label="BSB pi")
A = transition_matrices(sp, step, nbar_reset=0.0, k_max=1)
print(f"\nA_0 =\n{np.round(A[0],4)}\nA_1 =\n{np.round(A[1],4)}")
print(f"test CPTP: max|sum_k colsum - 1| = {check_cptp(A):.2e}")
print("⭐ Éstas son exactamente las A_k^(a) del MDP de arXiv:2410.11839.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(areas/np.pi, true_, "C0-", label=r"$|\beta|^2$ real en $^{27}$Al$^+$")
ax[0].plot(areas/np.pi, meas_, "C3o--", ms=4, label="medido vía $^9$Be$^+$")
ax[0].set(xlabel="área del pulso de sonda [$\\pi$]", ylabel="población",
          title="Fidelidad del mapeo QLS"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(areas/np.pi, meas_-true_, "C2-")
ax[1].axhline(0, c="k", lw=.7)
ax[1].set(xlabel="área [$\\pi$]", ylabel="residuo", title="Error sistemático del mapeo")
ax[1].grid(alpha=.3)
im = ax[2].imshow(A[1], cmap="viridis"); plt.colorbar(im, ax=ax[2])
ax[2].set(title="$A_1$ (resultado k=1)", xlabel="estado inicial", ylabel="estado final")
fig.tight_layout(); fig.savefig("figures/05_full_protocol.png", dpi=160)
print("\n-> figures/05_full_protocol.png")