"""DFIG simulation engine — UI-agnostic.

Numba JIT kernels for the 7-state Park model (RK4 / RK45 Dormand-Prince),
control laws (DPC bang-bang, vector control with stator-flux orientation),
and a threaded SimEngine that runs the integration in the background while
the GUI consumes ring-buffer snapshots at vsync.

This module exposes everything the UI layer needs (constants, kernels,
SimEngine class). It has no dependency on any GUI toolkit.
"""

import math
import time
import threading
from types import SimpleNamespace

import numpy as np
from numba import njit

# === Machine parameters (defaults — runtime-editable via params[]) ===
# Live in a numpy array passed to all JIT functions, so the user can change
# them on the fly without recompiling the kernels.
#
# Defaults correspond to the "1 MVA generic" preset in MACHINE_PRESETS below
# (690 V / 50 Hz / 3 pp, symmetric leakage L_ls = L_lr = 0.10 pu, L_m = 3.0 pu),
# i.e. the commonly-cited bands from Pena 1996 / Ledesma 2005 / NREL ref.
Rs, Rr = 4.76e-3, 2.38e-3
Ls, Lr, M_ = 4.696e-3, 4.696e-3, 4.545e-3
NP, J, B = 3, 50.0, 3.3
D = Ls * Lr - M_ * M_

# params[] indices (fixed order — do not change)
P_RS, P_RR, P_LS, P_LR, P_M, P_NP, P_J, P_B = 0, 1, 2, 3, 4, 5, 6, 7
P_SAT_EN, P_PSI_SAT = 8, 9
NPARAMS = 10

# Default saturation: disabled (warning-only mode in the UI). The threshold
# default matches the 1 MVA / 690 V / 50 Hz preset: ψ_s,nom = V_n/(2π·f_s)
# = 2.196 Wb, knee at 1.2× → 2.635 Wb. Re-set per preset by the UI.
SAT_EN_DEFAULT = 0.0
PSI_S_SAT_DEFAULT = 1.2 * 690.0 / (2.0 * math.pi * 50.0)

PARAMS_DEFAULT = np.array(
    [Rs, Rr, Ls, Lr, M_, float(NP), J, B,
     SAT_EN_DEFAULT, PSI_S_SAT_DEFAULT],
    dtype=np.float64,
)

# UI-side definition of parameters: key, label, SI value, display scale, unit.
PARAMS_DEF = [
    ("Rs", "R_s",  Rs, 1e-3, "mΩ"),
    ("Rr", "R_r",  Rr, 1e-3, "mΩ"),
    ("Ls", "L_s",  Ls, 1e-3, "mH"),
    ("Lr", "L_r",  Lr, 1e-3, "mH"),
    ("M",  "M",    M_, 1e-3, "mH"),
    ("NP", "n_p",  float(NP), 1.0, ""),
    ("J",  "J",    J,  1.0,  "kg·m²"),
    ("B",  "b",    B,  1.0,  "N·m·s"),
]

# === History buffer layout ===
H_T, H_WM, H_CE, H_PS, H_QS, H_PR, H_SLIP = 0, 1, 2, 3, 4, 5, 6
H_ISD, H_ISQ, H_IRD, H_IRQ = 7, 8, 9, 10
H_PSD, H_PSQ, H_PRD, H_PRQ = 11, 12, 13, 14
H_QR, H_PRS, H_PRR = 15, 16, 17
# Power balance (kW). Stored in GENERATOR convention: positive = power
# delivered to the grid (P_s, P_r, Q_s, Q_r) or input from shaft (P_mech).
#   Pmech = -Cl·ωm     (mechanical input, >0 when shaft drives the machine)
#   Pem   = -Ce·ωm     (electrical output, >0 when machine generates)
#   Pfric =  b·ωm²     (friction losses, always ≥0)
#   Ploss =  P_Rs + P_Rr + Pfric (total losses, always ≥0)
#   dKEdt =  J·ωm·(dωm/dt) (kinetic energy derivative — convention-free)
H_PMECH, H_PEM, H_PFRIC, H_PLOSS, H_DKEDT = 18, 19, 20, 21, 22
NH = 23
HIST_FIELDS = ["t", "wm", "Ce", "Ps", "Qs", "Pr", "slip",
               "isd", "isq", "ird", "irq",
               "psd", "psq", "prd", "prq",
               "Qr", "PRs", "PRr",
               "Pmech", "Pem", "Pfric", "Ploss", "dKEdt"]

# Catalogue of plottable signals (hist key · display name · unit · RGB).
SIGNALS_LIST = [
    ("wm",   "ω_m",  "giri/s", 0.23, 0.51, 0.96),
    ("Ce",   "C_em", "kNm",   0.94, 0.27, 0.27),
    ("Ps",   "P_s",  "kW",    0.23, 0.51, 0.96),
    ("Qs",   "Q_s",  "kVAR",  0.55, 0.36, 0.96),
    ("Pr",   "P_r",  "kW",    0.94, 0.55, 0.27),
    ("Qr",   "Q_r",  "kVAR",  0.95, 0.45, 0.75),
    ("PRs",  "P_Rs", "kW",    0.45, 0.75, 0.95),
    ("PRr",  "P_Rr", "kW",    0.95, 0.65, 0.45),
    ("slip", "slip", "%",     0.92, 0.70, 0.03),
    ("isd",  "i_sd", "A",     0.40, 0.60, 1.00),
    ("isq",  "i_sq", "A",     0.20, 0.40, 0.85),
    ("ird",  "i_rd", "A",     1.00, 0.60, 0.40),
    ("irq",  "i_rq", "A",     0.85, 0.40, 0.20),
    ("psd",  "φ_sd", "mWb",   0.40, 0.85, 0.60),
    ("psq",  "φ_sq", "mWb",   0.20, 0.65, 0.40),
    ("prd",  "φ_rd", "mWb",   1.00, 0.50, 0.85),
    ("prq",  "φ_rq", "mWb",   0.85, 0.30, 0.65),
    # Energy balance
    ("Pmech", "P_mecc",  "kW", 0.30, 0.85, 0.50),
    ("Pem",   "P_em",    "kW", 0.95, 0.60, 0.30),
    ("Pfric", "P_fric",  "kW", 0.65, 0.55, 0.40),
    ("Ploss", "P_loss",  "kW", 0.55, 0.45, 0.35),
    ("dKEdt", "dKE/dt",  "kW", 0.75, 0.85, 0.30),
    # Magnitudes / derived (computed on the fly from existing fields, not
    # stored in the history buffer — see _gui_tick in the UI layer)
    ("ism",   "|i_s|",   "A",   0.23, 0.51, 0.96),
    ("irm",   "|i_r|",   "A",   0.94, 0.27, 0.27),
    ("phis",  "|φ_s|",   "mWb", 0.13, 0.77, 0.37),
    ("phir",  "|φ_r|",   "mWb", 0.93, 0.28, 0.60),
    ("fslip", "f_slip",  "Hz",  0.92, 0.70, 0.03),
]
SIG_BY_KEY = {k: dict(name=n, unit=u, r=r, g=g, b=b)
              for k, n, u, r, g, b in SIGNALS_LIST}

MAX_PTS = 8000                   # 60 s × 1 sample / 7.5 ms
LOG_INTERVAL = 5e-3              # history subsampling: 5 ms simulated
MIN_ITER_DT = 5e-3               # each iter advances ≥ 5 ms simulated
MAX_ITER_DT = 100e-3             # anti-stutter clamp
MAX_STEPS_PER_ITER = 200_000     # safety cap
MAX_SAMPLES_PER_TICK = 256
RESYNC_LAG = 0.200               # if sim falls behind > 200 ms, re-anchor

INT_EULER, INT_RK4, INT_RK45 = 0, 1, 2

# === Control (passed to JIT kernels as an array) ===
# CTRL[0]  = mode  (0=open-loop V_r, 1=DPC, 2=VC stator-flux + speed PI)
# DPC:
#   CTRL[1] = Ps_ref [W]
#   CTRL[2] = Qs_ref [VAR]
#   CTRL[3] = h_P    [W]
#   CTRL[4] = h_Q    [VAR]
#   CTRL[5] = Vdc    [V]   rotor voltage vector amplitude
# VC (vector control with stator-flux orientation,
#     PF=1 stator side + speed tracking):
#   CTRL[6]  = wm_ref [rad/s mech]   speed reference
#   CTRL[7]  = P_w                   speed-PI P gain (on fractional error)
#   CTRL[8]  = I_w                   speed-PI I gain
#   CTRL[9]  = P_id                  i_r,d PI P gain
#   CTRL[10] = I_id                  i_r,d PI I gain
#   CTRL[11] = P_iq                  i_r,q PI P gain
#   CTRL[12] = I_iq                  i_r,q PI I gain
#   CTRL[13] = tan(φ_s*)             stator-side Q_s/P_s target
#                                    0 = unity PF (sub-case), >0 inductive,
#                                    <0 capacitive
NCTRL = 14
C_MODE, C_PSREF, C_QSREF, C_HP, C_HQ, C_VDC = 0, 1, 2, 3, 4, 5
C_WMREF, C_PW, C_IW, C_PID, C_IID, C_PIQ, C_IIQ = 6, 7, 8, 9, 10, 11, 12
C_TANPHI = 13
MODE_OPEN, MODE_DPC, MODE_VC = 0.0, 1.0, 2.0
CTRL_DEFAULT = np.array([
    MODE_OPEN,
    0.0, 0.0, 500.0, 500.0, 100.0,           # DPC
    2*math.pi*50.0/3.0,                       # wm_ref ≈ synchronous speed
    1.0e4, 1.0e5,                             # P_w, I_w  (fractional error)
    0.1, 1.0,                                 # P_id, I_id
    1.0, 10.0,                                # P_iq, I_iq
    0.0,                                      # tan(φ_s*) = 0  → PF=1
], dtype=np.float64)

# === Controller state (3 PI integrators + last-sample timestamp) ===
# Persists across engine iterations; reset on simulation reset / mode change.
NCS = 4
CS_INT_ID, CS_INT_IQ, CS_INT_W, CS_T_LAST = 0, 1, 2, 3


# ============================================================
# Machine preset library
# ============================================================
# Realistic parameter sets for various rated powers, drawn from canonical
# academic / manufacturer sources (see survey notes in the project README
# or the chat history of 2026-04-30). All electrical parameters are stored
# in per-unit; the conversion to SI happens once via preset_to_si() when
# a preset is loaded into the engine.
#
# Typical pu bands across all DFIG sizes (used as sanity-check thresholds):
#   R_s, R_r:    0.003 – 0.015 pu
#   L_ls, L_lr:  0.05  – 0.25  pu
#   L_m:         1.5   – 6.0   pu

MACHINE_PRESETS = [
    {
        "name": "1 MVA generic · 690 V · 50 Hz · 3 pp  (default)",
        "Sn": 1.0e6, "Vn": 690.0, "fs": 50.0, "np": 3,
        "Rs_pu": 0.010, "Rr_pu": 0.005,
        "Lls_pu": 0.10, "Llr_pu": 0.10, "Lm_pu": 3.0,
        "J": 50.0, "b": 3.3,
        "source": "Synthetic — mid-range pu values, 1 MVA / 690 V class",
    },
    {
        "name": "7.5 kW · 415 V · 50 Hz · 2 pp  (Pena 1996 lab)",
        "Sn": 7.5e3, "Vn": 415.0, "fs": 50.0, "np": 2,
        "Rs_pu": 0.0461, "Rr_pu": 0.0348,
        "Lls_pu": 0.1253, "Llr_pu": 0.1253, "Lm_pu": 1.130,
        "J": 0.15, "b": 0.005,
        "source": "Pena, Clare, Asher — IEE Proc EPA 1996 (reference DFIG)",
    },
    {
        "name": "1.5 MW · 575 V · 60 Hz · 3 pp  (GE 1.5xle, MathWorks)",
        "Sn": 1.5e6, "Vn": 575.0, "fs": 60.0, "np": 3,
        "Rs_pu": 0.00706, "Rr_pu": 0.005,
        "Lls_pu": 0.171, "Llr_pu": 0.156, "Lm_pu": 2.90,
        "J": 53.5, "b": 0.0,
        "source": "MathWorks 'Wind Farm – DFIG Detailed Model' / NREL Miller 2003",
    },
    {
        "name": "2 MW · 690 V · 50 Hz · 2 pp  (Vestas V80 class)",
        "Sn": 2.0e6, "Vn": 690.0, "fs": 50.0, "np": 2,
        "Rs_pu": 0.00488, "Rr_pu": 0.00549,
        "Lls_pu": 0.0922, "Llr_pu": 0.0995, "Lm_pu": 3.95,
        "J": 80.0, "b": 0.0,
        "source": "Ledesma & Usaola — IEEE TEC 2005",
    },
    {
        "name": "5 MW · 950 V · 50 Hz · 3 pp  (NREL reference)",
        "Sn": 5.0e6, "Vn": 950.0, "fs": 50.0, "np": 3,
        "Rs_pu": 0.00540, "Rr_pu": 0.00607,
        "Lls_pu": 0.10, "Llr_pu": 0.11, "Lm_pu": 4.5,
        "J": 534.0, "b": 0.0,
        "source": "Jonkman et al. NREL/TP-500-38060 + Singh & Santoso 2011",
    },
]


def preset_to_si(p):
    """Resolve a preset dict to absolute SI parameter values, ready to load
    into engine.params[] and engine.ctrl. Returns a dict with keys:
        Rs, Rr, Ls, Lr, M, NP, J, B   (engine machine parameters)
        Vn, fs, Sn                    (rated nominals — for the controls UI)
        Tn                            (rated mechanical torque, for slider scaling)
    """
    Zb = p["Vn"] ** 2 / p["Sn"]
    Lb = Zb / (2.0 * math.pi * p["fs"])
    Lls = p["Lls_pu"] * Lb
    Llr = p["Llr_pu"] * Lb
    Lm  = p["Lm_pu"]  * Lb
    # Nominal mechanical (shaft) angular speed at synchronism:
    #   ω_m,n = 2π·fs / np   [rad/s mech]
    omega_m_nom = 2.0 * math.pi * p["fs"] / p["np"]
    Tn = p["Sn"] / omega_m_nom  # rated torque [N·m]
    return {
        "Rs": p["Rs_pu"] * Zb,
        "Rr": p["Rr_pu"] * Zb,
        "Ls": Lls + Lm,
        "Lr": Llr + Lm,
        "M":  Lm,
        "NP": float(p["np"]),
        "B":  p["b"],
        "J":  p["J"],
        "Vn": p["Vn"],
        "fs": p["fs"],
        "Sn": p["Sn"],
        "Tn": Tn,
    }


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True, fastmath=True, nogil=True)
def _lm_eff_factor(psi_sd, psi_sq, psi_s_sat, sat_enabled):
    """Magnetizing-inductance reduction factor due to magnetic saturation.
    Returns f ∈ (0, 1] such that L_m,eff = L_m0 · f.
    Smooth taper: σ = |ψ_s|/ψ_sat,  f = 1/(1 + max(0, σ−1)²).
    Disabled (sat_enabled < 0.5) or undefined threshold → 1.0 (linear).
    """
    if sat_enabled < 0.5 or psi_s_sat <= 0.0:
        return 1.0
    psi_s_mag = math.sqrt(psi_sd * psi_sd + psi_sq * psi_sq)
    sigma = psi_s_mag / psi_s_sat
    excess = sigma - 1.0
    if excess <= 0.0:
        return 1.0
    return 1.0 / (1.0 + excess * excess)


@njit(cache=True, fastmath=True, nogil=True)
def _compute_vc(s, Vs, ws, params, ctrl, ctrl_state, out_v):
    """Stator-flux-oriented vector control + speed PI.
    Park convention of this simulator: V_s,d = V_s, V_s,q = 0
        → at steady state φ_s,d ≈ 0,  φ_s,q ≈ -V_s/ω_s
        ⇒ relative to the thesis, d and q roles are SWAPPED:
            P_s ≈ -V_s·M/L_s · i_r,d   →  torque ∝ i_r,d
            Q_s ≈  V_s²/(L_s ω_s) + V_s·M/L_s · i_r,q
                                       →  Q_s = 0  ⇒  i_r,q* = -V_s/(M ω_s)

    Scheme:
        i_r,q*  = -V_s/(M·ω_s)              (constant, unity power factor)
        i_r,d*  = -PI_w(ω_m err)             (outer loop: Ce ∝ -i_r,d)
        v_r,d   = PI_d(i_r,d err) + ff_d
        v_r,q   = PI_q(i_r,q err) + ff_q
    Decoupling feed-forward derived by substituting ψ_r,d = α·i_r,d,
    ψ_r,q = α·i_r,q - M·V_s/(L_s·ω_s) into the rotor equations:
        ff_d = -slip·α·i_r,q + slip·M·V_s/(L_s·ω_s)
        ff_q = +slip·α·i_r,d
    where α = (L_s·L_r - M²)/L_s, slip = ω_s - n_p·ω_m.
    Hard saturation of |V_r| ≤ V_MAX (converter realism).
    """
    Ls0 = params[2]; Lr0 = params[3]; M0 = params[4]; NP = params[5]
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    # Magnetic saturation: taper L_m as a function of |ψ_s|. Leakages stay
    # constant; L_s and L_r recompute from L_ls + L_m,eff and L_lr + L_m,eff.
    f_sat = _lm_eff_factor(psd, psq, params[9], params[8])
    M_ = M0 * f_sat
    Lls = Ls0 - M0
    Llr = Lr0 - M0
    Ls = Lls + M_
    Lr = Llr + M_
    D = Ls * Lr - M_ * M_
    alpha = D / Ls
    wm = s[4]; t = s[6]
    ird = (Ls * prd - M_ * psd) / D
    irq = (Ls * prq - M_ * psq) / D

    # dt since previous control sample (safety clamp)
    dt = t - ctrl_state[3]
    if dt < 0.0: dt = 0.0
    if dt > 1e-2: dt = 1e-2

    ws_safe = ws if abs(ws) > 1e-3 else 1e-3

    # === Speed loop (outer) → i_r,d* reference ===
    # Ce ∝ -i_r,d ⇒ to accelerate we need i_r,d < 0
    wm_ref = ctrl[6]; P_w = ctrl[7]; I_w = ctrl[8]
    ref_safe = wm_ref if abs(wm_ref) > 1.0 else 1.0
    e_w = (wm_ref - wm) / ref_safe
    ctrl_state[2] += e_w * dt
    ird_ref = -(P_w * e_w + I_w * ctrl_state[2])
    IRD_MAX = 5000.0
    if ird_ref > IRD_MAX: ird_ref = IRD_MAX
    if ird_ref < -IRD_MAX: ird_ref = -IRD_MAX

    # === i_r,q* reference for arbitrary power factor ===
    # Q_s = V²/(Ls·ωs) + V·M/Ls·i_r,q  and  P_s = -V·M/Ls·i_r,d
    # Want Q_s = tan(φ*)·P_s  ⇒  i_r,q* = -V/(M·ωs) - tan(φ*)·i_r,d
    # tan(φ*)=0 → PF=1 (sub-case). >0 inductive (Q absorbed), <0 capacitive.
    tan_phi = ctrl[13]
    irq_ref = -Vs / (M_ * ws_safe) - tan_phi * ird

    # === Inner current loops ===
    P_id = ctrl[9]; I_id = ctrl[10]
    e_d = ird_ref - ird
    ctrl_state[0] += e_d * dt
    vrd_pi = P_id * e_d + I_id * ctrl_state[0]

    P_iq = ctrl[11]; I_iq = ctrl[12]
    e_q = irq_ref - irq
    ctrl_state[1] += e_q * dt
    vrq_pi = P_iq * e_q + I_iq * ctrl_state[1]

    # === Feed-forward decoupling ===
    slip = ws - NP * wm
    vrd_ff = -slip * alpha * irq + slip * M_ * Vs / (Ls * ws_safe)
    vrq_ff = +slip * alpha * ird

    vrd = vrd_pi + vrd_ff
    vrq = vrq_pi + vrq_ff

    # |V_r| saturation
    V_MAX = 500.0
    Vmag2 = vrd*vrd + vrq*vrq
    if Vmag2 > V_MAX*V_MAX:
        Vmag = math.sqrt(Vmag2)
        sc = V_MAX / Vmag
        vrd *= sc; vrq *= sc
        # Coarse anti-windup: scale integrators too
        ctrl_state[0] *= sc; ctrl_state[1] *= sc

    out_v[0] = vrd
    out_v[1] = vrq
    ctrl_state[3] = t


@njit(cache=True, fastmath=True, nogil=True)
def _compute_dpc(s, Vs, params, ctrl, out_v):
    """Compute (v_rd, v_rq) DPC bang-bang from current state.
    Convention: V_sd=Vs, V_sq=0 → P_s=Vs·i_sd, Q_s=-Vs·i_sq.
    Stator-flux orientation (steady-state):
        P_s ≈ -Vs·M/Ls · i_rd   →   to ↑P_s, ↓i_rd  →  v_rd negative
        Q_s ≈ Vs²/(ωs Ls) + Vs·M/Ls · i_rq  →  to ↑Q_s, ↑i_rq  →  v_rq positive
    """
    Ls0 = params[2]; Lr0 = params[3]; M0 = params[4]
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    f_sat = _lm_eff_factor(psd, psq, params[9], params[8])
    M_ = M0 * f_sat
    Lls = Ls0 - M0
    Llr = Lr0 - M0
    Ls = Lls + M_
    Lr = Llr + M_
    D = Ls * Lr - M_ * M_
    isd = (Lr * psd - M_ * prd) / D
    isq = (Lr * psq - M_ * prq) / D
    Ps = Vs * isd
    Qs = -Vs * isq
    eP = ctrl[1] - Ps
    eQ = ctrl[2] - Qs
    hP = ctrl[3]; hQ = ctrl[4]; Vdc = ctrl[5]
    if eP > hP:    sP = -1.0
    elif eP < -hP: sP = +1.0
    else:          sP = 0.0
    if eQ > hQ:    sQ = +1.0
    elif eQ < -hQ: sQ = -1.0
    else:          sQ = 0.0
    out_v[0] = Vdc * sP
    out_v[1] = Vdc * sQ


@njit(cache=True, fastmath=True, nogil=True)
def deriv(s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, out):
    """RHS of the system. In open-loop (use_held<0.5) the rotor voltage
    vector is recomputed continuously from (Vr, wr, t, thm) inside the
    integration step (smoothness preserved). In DPC (use_held>0.5) the
    values (vrd_h, vrq_h) sampled by the caller before the step are held
    constant for all intermediate evaluations of the step: this keeps the
    RHS Lipschitzian within the step, the estimated error stays small,
    and dt does not collapse."""
    Rs = params[0]; Rr = params[1]
    Ls0 = params[2]; Lr0 = params[3]; M0 = params[4]
    NP = params[5]; J = params[6]; B = params[7]
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    wm = s[4]; thm = s[5]; t = s[6]
    f_sat = _lm_eff_factor(psd, psq, params[9], params[8])
    M_ = M0 * f_sat
    Lls = Ls0 - M0
    Llr = Lr0 - M0
    Ls = Lls + M_
    Lr = Llr + M_
    D = Ls * Lr - M_ * M_
    if use_held > 0.5:
        vrd = vrd_h; vrq = vrq_h
    else:
        a = wr * t - ws * t + NP * thm
        vrd = Vr * math.cos(a); vrq = Vr * math.sin(a)
    isd = (Lr * psd - M_ * prd) / D
    isq = (Lr * psq - M_ * prq) / D
    ird = (Ls * prd - M_ * psd) / D
    irq = (Ls * prq - M_ * psq) / D
    wsl = ws - NP * wm
    Ce = NP * M_ * (isq * ird - isd * irq)
    out[0] = Vs - Rs * isd + ws * psq
    out[1] = -Rs * isq - ws * psd
    out[2] = vrd - Rr * ird + wsl * prq
    out[3] = vrq - Rr * irq - wsl * prd
    out[4] = (Ce - B * wm - Cl) / J
    out[5] = wm
    out[6] = 1.0


@njit(cache=True, fastmath=True, nogil=True)
def observe(s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, out):
    Rs = params[0]; Rr = params[1]
    Ls0 = params[2]; Lr0 = params[3]; M0 = params[4]
    NP = params[5]; J = params[6]; B = params[7]
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    wm = s[4]; thm = s[5]; t = s[6]
    f_sat = _lm_eff_factor(psd, psq, params[9], params[8])
    M_ = M0 * f_sat
    Lls = Ls0 - M0
    Llr = Lr0 - M0
    Ls = Lls + M_
    Lr = Llr + M_
    D = Ls * Lr - M_ * M_
    if use_held > 0.5:
        vrd = vrd_h; vrq = vrq_h
    else:
        a = wr * t - ws * t + NP * thm
        vrd = Vr * math.cos(a); vrq = Vr * math.sin(a)
    isd = (Lr * psd - M_ * prd) / D
    isq = (Lr * psq - M_ * prq) / D
    ird = (Ls * prd - M_ * psd) / D
    irq = (Ls * prq - M_ * psq) / D
    Ce = NP * M_ * (isq * ird - isd * irq)
    # Generator convention: P_s/Q_s/P_r/Q_r positive = delivered to grid.
    # The kernel's internal control (DPC, VC) works in motor convention;
    # the UI inverts P_s_ref / Q_s_ref before passing them in, so the user
    # sees a coherent gen-convention setpoint everywhere.
    Ps = -Vs * isd
    Qs = +Vs * isq
    Pr = -(vrd * ird + vrq * irq)
    Qr = -(vrq * ird - vrd * irq)
    PRs = Rs * (isd * isd + isq * isq)
    PRr = Rr * (ird * ird + irq * irq)
    slip = (ws - NP * wm) / ws * 100.0 if ws > 0 else 0.0
    # Power balance (W → kW for consistency with P_s, P_r, P_loss)
    Pmech = -Cl * wm                       # mechanical input (>0 if shaft drives)
    Pem   = -Ce * wm                       # electrical output (gen convention)
    Pfric = B * wm * wm                    # viscous friction losses
    Ploss = PRs + PRr + Pfric              # total losses
    # dKE/dt = J·ω·dω/dt = ω·(Ce - B·ω - Cl)  (steady state: ≈ 0)
    dKEdt = wm * (Ce - B * wm - Cl)
    out[0] = t
    out[1] = wm / (2.0 * math.pi)
    out[2] = Ce / 1e3
    out[3] = Ps / 1e3
    out[4] = Qs / 1e3
    out[5] = Pr / 1e3
    out[6] = slip
    out[7] = isd
    out[8] = isq
    out[9] = ird
    out[10] = irq
    out[11] = psd * 1e3
    out[12] = psq * 1e3
    out[13] = prd * 1e3
    out[14] = prq * 1e3
    out[15] = Qr / 1e3
    out[16] = PRs / 1e3
    out[17] = PRr / 1e3
    out[18] = Pmech / 1e3
    out[19] = Pem   / 1e3
    out[20] = Pfric / 1e3
    out[21] = Ploss / 1e3
    out[22] = dKEdt / 1e3


@njit(cache=True, fastmath=True, nogil=True)
def _maybe_sample(s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params,
                  last_log_t, samples, n_samples, max_samples, obs):
    if s[6] - last_log_t >= LOG_INTERVAL and n_samples < max_samples:
        observe(s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, obs)
        for j in range(NH):
            samples[n_samples, j] = obs[j]
        n_samples += 1
        last_log_t += LOG_INTERVAL
        # if very behind (e.g. resume from pause): snap to present
        if last_log_t < s[6] - 10.0 * LOG_INTERVAL:
            last_log_t = s[6]
    return n_samples, last_log_t


@njit(cache=True, fastmath=True, nogil=True)
def _sample_held(s, Vs, ws, params, ctrl, ctrl_state, v_buf):
    """Return (vrd_h, vrq_h) sampled from state per the active mode:
       1 = DPC bang-bang, 2 = VC vector control, anything else = (0,0)."""
    m = ctrl[0]
    if m > 1.5:
        _compute_vc(s, Vs, ws, params, ctrl, ctrl_state, v_buf)
    elif m > 0.5:
        _compute_dpc(s, Vs, params, ctrl, v_buf)
    else:
        v_buf[0] = 0.0; v_buf[1] = 0.0


@njit(cache=True, fastmath=True, nogil=True)
def advance_euler(s, t_target, Vs, ws, Vr, wr, Cl, dt, params, ctrl, ctrl_state,
                  last_log_t, samples, max_samples):
    k = np.empty(7)
    obs = np.empty(NH)
    v_buf = np.empty(2)
    use_held = 1.0 if ctrl[0] > 0.5 else 0.0
    n_steps = 0
    n_samples = 0
    vrd_h = 0.0; vrq_h = 0.0
    while s[6] < t_target and n_steps < MAX_STEPS_PER_ITER:
        if use_held > 0.5:
            _sample_held(s, Vs, ws, params, ctrl, ctrl_state, v_buf)
            vrd_h = v_buf[0]; vrq_h = v_buf[1]
        deriv(s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k)
        for i in range(7):
            s[i] += dt * k[i]
        n_steps += 1
        n_samples, last_log_t = _maybe_sample(
            s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params,
            last_log_t, samples, n_samples, max_samples, obs)
    return n_steps, n_samples, last_log_t


@njit(cache=True, fastmath=True, nogil=True)
def advance_rk4(s, t_target, Vs, ws, Vr, wr, Cl, dt, params, ctrl, ctrl_state,
                last_log_t, samples, max_samples):
    k1 = np.empty(7); k2 = np.empty(7); k3 = np.empty(7); k4 = np.empty(7)
    tmp = np.empty(7); obs = np.empty(NH); v_buf = np.empty(2)
    use_held = 1.0 if ctrl[0] > 0.5 else 0.0
    n_steps = 0
    n_samples = 0
    vrd_h = 0.0; vrq_h = 0.0
    while s[6] < t_target and n_steps < MAX_STEPS_PER_ITER:
        if use_held > 0.5:
            _sample_held(s, Vs, ws, params, ctrl, ctrl_state, v_buf)
            vrd_h = v_buf[0]; vrq_h = v_buf[1]
        deriv(s,   Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k1)
        for i in range(7): tmp[i] = s[i] + 0.5 * dt * k1[i]
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k2)
        for i in range(7): tmp[i] = s[i] + 0.5 * dt * k2[i]
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k3)
        for i in range(7): tmp[i] = s[i] + dt * k3[i]
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k4)
        h6 = dt / 6.0
        for i in range(7):
            s[i] += h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        n_steps += 1
        n_samples, last_log_t = _maybe_sample(
            s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params,
            last_log_t, samples, n_samples, max_samples, obs)
    return n_steps, n_samples, last_log_t


# Dormand-Prince 5(4) coefficients
@njit(cache=True, fastmath=True, nogil=True)
def advance_rk45(s, t_target, Vs, ws, Vr, wr, Cl,
                 dt_init, dt_min, dt_max, rtol, params, ctrl, ctrl_state,
                 last_log_t, samples, max_samples):
    k1 = np.empty(7); k2 = np.empty(7); k3 = np.empty(7); k4 = np.empty(7)
    k5 = np.empty(7); k6 = np.empty(7); k7 = np.empty(7)
    tmp = np.empty(7); s5 = np.empty(7); err_vec = np.empty(7); obs = np.empty(NH)

    n_steps = 0
    n_rej = 0
    n_samples = 0
    err_max = 0.0
    sum_dt = 0.0
    dt = dt_init
    if dt > dt_max: dt = dt_max
    if dt < dt_min: dt = dt_min

    v_buf = np.empty(2)
    use_held = 1.0 if ctrl[0] > 0.5 else 0.0
    vrd_h = 0.0; vrq_h = 0.0
    # In DPC: control is sampled at the start of each step (zero-order
    # hold); the same (vrd_h, vrq_h) pair is used for all intermediate
    # step evaluations, so the RHS is continuous within the step and dt
    # doesn't collapse due to hysteresis. Resample only after an ACCEPTED
    # step.
    sample_dpc = use_held > 0.5

    while s[6] < t_target and (n_steps + n_rej) < MAX_STEPS_PER_ITER:
        if s[6] + dt > t_target:
            dt = t_target - s[6]
        if dt <= 0.0:
            break

        if sample_dpc:
            _sample_held(s, Vs, ws, params, ctrl, ctrl_state, v_buf)
            vrd_h = v_buf[0]; vrq_h = v_buf[1]
            sample_dpc = False  # already sampled for this attempt

        deriv(s,   Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k1)
        for i in range(7): tmp[i] = s[i] + dt * (0.2 * k1[i])
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k2)
        for i in range(7): tmp[i] = s[i] + dt * (3.0/40.0*k1[i] + 9.0/40.0*k2[i])
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k3)
        for i in range(7): tmp[i] = s[i] + dt * (44.0/45.0*k1[i] - 56.0/15.0*k2[i] + 32.0/9.0*k3[i])
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k4)
        for i in range(7):
            tmp[i] = s[i] + dt * (19372.0/6561.0*k1[i] - 25360.0/2187.0*k2[i]
                                  + 64448.0/6561.0*k3[i] - 212.0/729.0*k4[i])
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k5)
        for i in range(7):
            tmp[i] = s[i] + dt * (9017.0/3168.0*k1[i] - 355.0/33.0*k2[i]
                                  + 46732.0/5247.0*k3[i] + 49.0/176.0*k4[i]
                                  - 5103.0/18656.0*k5[i])
        deriv(tmp, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k6)
        # 5th order solution
        for i in range(7):
            s5[i] = s[i] + dt * (35.0/384.0*k1[i] + 500.0/1113.0*k3[i]
                                 + 125.0/192.0*k4[i] - 2187.0/6784.0*k5[i]
                                 + 11.0/84.0*k6[i])
        deriv(s5, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params, k7)
        # error = (b5 - b4) · dt · k
        e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0
        e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
        for i in range(7):
            err_vec[i] = dt * (e1*k1[i] + e3*k3[i] + e4*k4[i]
                               + e5*k5[i] + e6*k6[i] + e7*k7[i])
        # scaled norm
        err_norm = 0.0
        for i in range(7):
            sc = max(abs(s[i]), abs(s5[i])) * rtol + 1e-10
            v = abs(err_vec[i]) / sc
            if v > err_norm: err_norm = v

        if err_norm <= 1.0:
            for i in range(7):
                s[i] = s5[i]
            n_steps += 1
            sum_dt += dt
            if err_norm > err_max: err_max = err_norm
            # PI-ish step adapt (safety 0.9, exp 1/5)
            if err_norm > 0.0:
                factor = 0.9 * err_norm ** (-0.2)
            else:
                factor = 5.0
            if factor > 5.0: factor = 5.0
            if factor < 0.2: factor = 0.2
            dt *= factor
            if dt > dt_max: dt = dt_max
            if dt < dt_min: dt = dt_min
            n_samples, last_log_t = _maybe_sample(
                s, Vs, ws, Vr, wr, vrd_h, vrq_h, use_held, Cl, params,
                last_log_t, samples, n_samples, max_samples, obs)
            # Accepted step → resample control for next step
            if use_held > 0.5:
                sample_dpc = True
        else:
            n_rej += 1
            factor = 0.9 * err_norm ** (-0.2)
            if factor < 0.1: factor = 0.1
            dt *= factor
            if dt < dt_min:
                # force acceptance: the system is degenerate at this dt
                dt = dt_min
            # Rejected step: keep the same (vrd_h, vrq_h) and retry with a
            # smaller dt — don't resample, avoids "noise" in the reject loop.

    avg_dt = sum_dt / n_steps if n_steps > 0 else 0.0
    return n_steps, n_samples, last_log_t, n_rej, err_max, avg_dt, dt


def warmup_jit():
    """Force AOT-cached compilation of the numba functions."""
    s = np.zeros(7)
    s[1] = -2.2; s[3] = -2.2; s[4] = 100.0
    samples = np.empty((4, NH))
    p = PARAMS_DEFAULT.copy()
    c_ol  = CTRL_DEFAULT.copy()
    c_dpc = CTRL_DEFAULT.copy(); c_dpc[C_MODE] = MODE_DPC
    c_vc  = CTRL_DEFAULT.copy(); c_vc[C_MODE]  = MODE_VC
    cs = np.zeros(NCS)
    advance_euler(s.copy(), 1e-3, 690.0, 314.159, 0.0, 0.0, 0.0, 1e-4, p, c_ol, cs.copy(), 0.0, samples, 4)
    advance_rk4(s.copy(),   1e-3, 690.0, 314.159, 0.0, 0.0, 0.0, 1e-4, p, c_ol, cs.copy(), 0.0, samples, 4)
    advance_rk45(s.copy(),  1e-3, 690.0, 314.159, 0.0, 0.0, 0.0,
                 1e-4, 1e-7, 1e-3, 1e-4, p, c_ol, cs.copy(), 0.0, samples, 4)
    advance_rk45(s.copy(),  1e-3, 690.0, 314.159, 0.0, 0.0, 0.0,
                 1e-4, 1e-7, 1e-3, 1e-4, p, c_dpc, cs.copy(), 0.0, samples, 4)
    advance_rk45(s.copy(),  1e-3, 690.0, 314.159, 0.0, 0.0, 0.0,
                 1e-4, 1e-7, 1e-3, 1e-4, p, c_vc, cs.copy(), 0.0, samples, 4)
    # Also exercise the saturation-on path so its specialization is cached.
    p_sat = p.copy(); p_sat[P_SAT_EN] = 1.0
    advance_rk45(s.copy(),  1e-3, 690.0, 314.159, 0.0, 0.0, 0.0,
                 1e-4, 1e-7, 1e-3, 1e-4, p_sat, c_ol, cs.copy(), 0.0, samples, 4)


# ============================================================
# Sim engine (background thread)
# ============================================================

class SimEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = np.zeros(7)
        # Runtime-editable machine parameters (passed to JIT kernels)
        self.params = PARAMS_DEFAULT.copy()
        # Numpy ring buffer (zero-garbage snapshot): one memcpy instead of 15× list copy
        self.hist_buf = np.zeros((MAX_PTS, NH), dtype=np.float64)
        self.hist_write = 0     # total samples written (monotonic)
        self.hist_count = 0     # min(hist_write, MAX_PTS)
        self.ctrl = {
            "Vs": 690.0, "fs": 50.0, "Vr": 0.0, "fr": 0.0, "Cl": 0.0,
            "logdt": -4.0, "rtol": 1e-9,
            "integrator": INT_RK45,
            "rt_lock": True,
            "speed_factor": 1.0,   # target = wall × speed
            # === Rotor-side control ===
            # mode: "open" | "dpc" | "vc"
            "mode": "open",
            # DPC
            "Ps_ref": 0.0,         # W
            "Qs_ref": 0.0,         # VAR
            "h_P": 500.0,          # W
            "h_Q": 500.0,          # VAR
            "Vdc_dpc": 100.0,      # V
            # VC (stator-flux orientation, PF=1 + speed)
            "wm_ref": 2.0 * math.pi * 50.0 / 3.0,  # rad/s mech, ≈ synchronous
            "P_w": 1.0e4, "I_w": 1.0e5,            # speed PI (fractional error)
            "P_id": 0.1, "I_id": 1.0,               # i_r,d PI
            "P_iq": 1.0, "I_iq": 10.0,              # i_r,q PI
            "tan_phi": 0.0,                          # Q_s/P_s target (0 = unity PF)
        }
        # Persistent controller state (3 PI integrators + last-sample timestamp)
        self.ctrl_state = np.zeros(NCS)
        self.stats = {
            "compiled": False,
            "running": False,
            "integrator": INT_RK4,
            "rt_lock": True,
            "steps_per_s": 0.0,
            "dt_eff_us": 0.0,
            "rt_factor": 0.0,    # true: sim_time / wall_time (incl. sleep)
            "headroom": 0.0,     # CPU margin: sim_dt / wall_busy
            "lag_ms": 0.0,       # how far behind we are vs wall
            "rejected_pct": 0.0,
            "err_norm_max": 0.0,
            "iter_wall_ms": 0.0,
            "resyncs": 0,
        }
        self.running = False
        self._stop = False
        self.thread = None
        # Real-time anchors: target_sim = t_sim_anchor + (wall - t_wall_anchor) * speed
        self._t_wall_anchor = 0.0
        self._t_sim_anchor = 0.0
        self._last_log_t = 0.0
        self._dt_rk45 = 1e-4  # carry-over dt for RK45
        self._reset_state_locked()

    def _reset_state_locked(self):
        # Initial flux from current V_s/f_s nominals (preset-aware).
        Vs0 = self.ctrl.get("Vs", 690.0)
        fs0 = self.ctrl.get("fs", 50.0)
        ws0 = 2.0 * math.pi * fs0
        p0 = Vs0 / ws0 if ws0 > 0 else 0.0
        Ls_p = self.params[P_LS]; M_p = self.params[P_M]; NP_p = self.params[P_NP]
        wm0 = ws0 / NP_p if NP_p > 0 else 0.0
        self.state[:] = [0.0, -p0, 0.0, -p0 * M_p / Ls_p,
                         wm0, 0.0, 0.0]
        self.hist_write = 0
        self.hist_count = 0
        self._last_log_t = 0.0
        self._t_wall_anchor = time.perf_counter()
        self._t_sim_anchor = 0.0
        self._dt_rk45 = 1e-4
        # Reset PI integrators and last-sample-time
        self.ctrl_state[:] = 0.0

    def reset(self):
        with self.lock:
            self._reset_state_locked()

    def _rebase_anchor_locked(self):
        """Re-align anchors to here-and-now (for speed change or resume)."""
        self._t_wall_anchor = time.perf_counter()
        self._t_sim_anchor = self.state[6]

    def toggle_run(self):
        with self.lock:
            self.running = not self.running
            if self.running:
                self._rebase_anchor_locked()
            self.stats["running"] = self.running
            return self.running

    def start(self):
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def shutdown(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=1.0)

    def snapshot_for_render(self):
        """Zero-garbage snapshot: one memcpy of the ring buffer."""
        with self.lock:
            n = self.hist_count
            write = self.hist_write
            if n == 0:
                arr = np.empty((0, NH), dtype=np.float64)
            elif write <= MAX_PTS:
                # buffer not yet full: contiguous range [0, n)
                arr = self.hist_buf[:n].copy()
            else:
                # buffer full: rotation
                head = write % MAX_PTS  # next write position = chronological start
                arr = np.empty((MAX_PTS, NH), dtype=np.float64)
                arr[:MAX_PTS - head] = self.hist_buf[head:]
                arr[MAX_PTS - head:] = self.hist_buf[:head]
            state = self.state.copy()
            stats = dict(self.stats)
            ctrl = dict(self.ctrl)
            params = self.params.copy()
        # Wrap as SimpleNamespace with a numpy view per field (no further copy)
        snap = SimpleNamespace()
        for i, f in enumerate(HIST_FIELDS):
            setattr(snap, f, arr[:, i])
        return snap, state, stats, ctrl, params

    def set_param(self, idx, value):
        """Update a machine parameter at runtime (in-place under lock).
        Negative values are rejected; zero is allowed (e.g. friction b = 0
        in wind-DFIG presets, sat_enabled = 0 to disable saturation)."""
        with self.lock:
            if 0 <= idx < NPARAMS and value >= 0:
                self.params[idx] = value

    def set_ctrl(self, **kw):
        with self.lock:
            speed_changed = ("speed_factor" in kw and
                             kw["speed_factor"] != self.ctrl.get("speed_factor"))
            mode_changed = ("mode" in kw and
                            kw["mode"] != self.ctrl.get("mode"))
            self.ctrl.update(kw)
            if speed_changed and self.running:
                # Re-anchor to here-and-now to avoid target jumps
                self._rebase_anchor_locked()
            if mode_changed:
                # Controller mode change: zero PI integrators to avoid
                # transients from stale state
                self.ctrl_state[:] = 0.0
                self.ctrl_state[CS_T_LAST] = self.state[6]

    # === main loop ===
    def _loop(self):
        try:
            warmup_jit()
        except Exception as e:
            print("warmup error:", e)
        with self.lock:
            self.stats["compiled"] = True

        sample_buf = np.empty((MAX_SAMPLES_PER_TICK, NH), dtype=np.float64)

        sm_steps_per_s = 0.0
        sm_dt_eff = 0.0
        sm_headroom = 0.0
        sm_rt_real = 0.0
        sm_iter = 0.0
        alpha = 0.15  # EWMA

        # For true RT factor: wall/sim time at the end of the previous iter
        t_wall_prev = None
        t_sim_prev = None
        resyncs = 0

        while not self._stop:
            with self.lock:
                running = self.running
                if not running:
                    self._rebase_anchor_locked()
                    self.stats["steps_per_s"] = 0.0
                    self.stats["dt_eff_us"] = 0.0
                    self.stats["rt_factor"] = 0.0
                    self.stats["headroom"] = 0.0
                    self.stats["lag_ms"] = 0.0
                    self.stats["iter_wall_ms"] = 0.0
                    t_wall_prev = None
            if not running:
                time.sleep(0.020)
                continue

            # Snapshot ctrl + state (under lock)
            with self.lock:
                Vs = self.ctrl["Vs"]; fs = self.ctrl["fs"]
                Vr = self.ctrl["Vr"]; fr = self.ctrl["fr"]
                Cl = self.ctrl["Cl"]
                dt_max = 10.0 ** self.ctrl["logdt"]
                integrator = self.ctrl["integrator"]
                rt_lock = self.ctrl["rt_lock"]
                rtol = self.ctrl["rtol"]
                speed = self.ctrl.get("speed_factor", 1.0)
                mode_str = self.ctrl.get("mode", "open")
                Ps_ref = self.ctrl.get("Ps_ref", 0.0)
                Qs_ref = self.ctrl.get("Qs_ref", 0.0)
                h_P = self.ctrl.get("h_P", 500.0)
                h_Q = self.ctrl.get("h_Q", 500.0)
                Vdc_dpc = self.ctrl.get("Vdc_dpc", 100.0)
                wm_ref = self.ctrl.get("wm_ref", 2.0*math.pi*50.0/3.0)
                P_w = self.ctrl.get("P_w", 1e4); I_w = self.ctrl.get("I_w", 1e5)
                P_id = self.ctrl.get("P_id", 0.1); I_id = self.ctrl.get("I_id", 1.0)
                P_iq = self.ctrl.get("P_iq", 1.0); I_iq = self.ctrl.get("I_iq", 10.0)
                tan_phi = self.ctrl.get("tan_phi", 0.0)
                state_local = self.state.copy()
                params_local = self.params.copy()
                ctrl_state_local = self.ctrl_state.copy()
                last_log_t = self._last_log_t
                t_wall_anchor = self._t_wall_anchor
                t_sim_anchor = self._t_sim_anchor
                dt_rk45 = self._dt_rk45

            ws = 2.0 * math.pi * fs
            wr = 2.0 * math.pi * fr
            mode_code = (MODE_VC if mode_str == "vc"
                         else MODE_DPC if mode_str == "dpc"
                         else MODE_OPEN)
            ctrl_arr = np.array([
                mode_code,
                Ps_ref, Qs_ref, h_P, h_Q, Vdc_dpc,
                wm_ref, P_w, I_w, P_id, I_id, P_iq, I_iq,
                tan_phi,
            ], dtype=np.float64)

            now = time.perf_counter()
            # target = t_sim_anchor + (now - t_wall_anchor) * speed
            target_sim = t_sim_anchor + (now - t_wall_anchor) * speed
            lag = target_sim - state_local[6]   # how far behind the target

            if rt_lock:
                # Auto-resync: if too far behind, re-align anchors
                if lag > RESYNC_LAG:
                    t_wall_anchor = now
                    t_sim_anchor = state_local[6]
                    target_sim = state_local[6]
                    lag = 0.0
                    resyncs += 1
                    with self.lock:
                        self._t_wall_anchor = t_wall_anchor
                        self._t_sim_anchor = t_sim_anchor

                t_target = min(target_sim, state_local[6] + MAX_ITER_DT)
                if t_target < state_local[6] + MIN_ITER_DT:
                    # still too early, wait
                    if speed > 0:
                        sleep_for = ((state_local[6] + MIN_ITER_DT - t_sim_anchor) / speed
                                     - (now - t_wall_anchor))
                    else:
                        sleep_for = 0.020
                    if sleep_for > 0:
                        time.sleep(min(sleep_for, 0.020))
                    continue
            else:
                t_target = state_local[6] + MAX_ITER_DT

            t_sim_start = state_local[6]
            t0 = time.perf_counter()

            n_rej = 0; err_max = 0.0; dt_avg = dt_max
            if integrator == INT_EULER:
                n_steps, n_samples, last_log_t = advance_euler(
                    state_local, t_target, Vs, ws, Vr, wr, Cl, dt_max,
                    params_local, ctrl_arr, ctrl_state_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)
            elif integrator == INT_RK4:
                n_steps, n_samples, last_log_t = advance_rk4(
                    state_local, t_target, Vs, ws, Vr, wr, Cl, dt_max,
                    params_local, ctrl_arr, ctrl_state_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)
            else:  # RK45
                dt_min = max(1e-9, dt_max * 1e-4)
                (n_steps, n_samples, last_log_t,
                 n_rej, err_max, dt_avg, dt_rk45) = advance_rk45(
                    state_local, t_target, Vs, ws, Vr, wr, Cl,
                    dt_rk45, dt_min, dt_max, rtol,
                    params_local, ctrl_arr, ctrl_state_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)

            t1 = time.perf_counter()
            wall_busy = t1 - t0
            sim_dt = state_local[6] - t_sim_start

            # Push samples + commit state — write into ring buffer (one memcpy)
            with self.lock:
                self.state[:] = state_local
                self.ctrl_state[:] = ctrl_state_local
                self._last_log_t = last_log_t
                self._dt_rk45 = dt_rk45
                if n_samples > 0:
                    write = self.hist_write
                    # block write handling wrap-around
                    start = write % MAX_PTS
                    end = start + n_samples
                    if end <= MAX_PTS:
                        self.hist_buf[start:end] = sample_buf[:n_samples]
                    else:
                        first = MAX_PTS - start
                        self.hist_buf[start:] = sample_buf[:first]
                        self.hist_buf[:n_samples - first] = sample_buf[first:n_samples]
                    self.hist_write = write + n_samples
                    self.hist_count = min(self.hist_count + n_samples, MAX_PTS)

            # Sleep if ahead of target (keeps the wall × speed lock)
            if rt_lock and speed > 0:
                # when state.t reaches the target → wall_at_target
                wall_at_target = t_wall_anchor + (state_local[6] - t_sim_anchor) / speed
                slack = wall_at_target - time.perf_counter()
                if slack > 0:
                    time.sleep(min(slack, 0.020))

            # Metrics after sleep — now wall_real is the TRUE elapsed time
            t_iter_end = time.perf_counter()
            if t_wall_prev is not None and wall_busy > 0:
                wall_real = t_iter_end - t_wall_prev
                if wall_real > 0:
                    sim_step = state_local[6] - t_sim_prev
                    # rt vs the TARGET: 1.0 = on-target at any speed
                    rt_target_inst = (sim_step / wall_real) / speed if speed > 0 else 0.0
                    headroom_inst = sim_dt / wall_busy   # how-much-faster-could-I-go
                    sps = n_steps / wall_busy
                    sm_steps_per_s = (1 - alpha) * sm_steps_per_s + alpha * sps
                    sm_dt_eff = (1 - alpha) * sm_dt_eff + alpha * dt_avg
                    sm_headroom = (1 - alpha) * sm_headroom + alpha * headroom_inst
                    sm_rt_real = (1 - alpha) * sm_rt_real + alpha * rt_target_inst
                    sm_iter = (1 - alpha) * sm_iter + alpha * (wall_busy * 1e3)
            t_wall_prev = t_iter_end
            t_sim_prev = state_local[6]

            with self.lock:
                self.stats["integrator"] = integrator
                self.stats["rt_lock"] = rt_lock
                self.stats["steps_per_s"] = sm_steps_per_s
                self.stats["dt_eff_us"] = sm_dt_eff * 1e6
                self.stats["rt_factor"] = sm_rt_real
                self.stats["headroom"] = sm_headroom
                self.stats["lag_ms"] = max(0.0, lag) * 1e3
                self.stats["resyncs"] = resyncs
                tot = n_steps + n_rej
                self.stats["rejected_pct"] = (n_rej / tot * 100.0) if tot > 0 else 0.0
                self.stats["err_norm_max"] = err_max
                self.stats["iter_wall_ms"] = sm_iter
