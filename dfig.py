#!/usr/bin/env python3
"""
DFIG — Simulatore Interattivo (GTK4 + Cairo)
Sim engine in Numba JIT su thread dedicato; GUI a 60 Hz su main loop.
Integratori: Eulero / RK4 / RK45 (Dormand-Prince adattativo).
"""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, Gdk
import cairo
import math
import time
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
from numba import njit

# === Parametri macchina (default — modificabili a runtime via params[]) ===
# I valori vivono in un array numpy passato a tutte le funzioni JIT,
# così l'utente li può cambiare al volo senza ricompilare i kernel.
Rs, Rr = 4.8e-3, 2.4e-3
Ls, Lr, M_ = 6.8e-3, 7.1e-3, 6.8e-3
NP, J, B = 3, 50.0, 3.3
D = Ls * Lr - M_ * M_

# Indici nell'array params (ordine fisso, non cambiare)
P_RS, P_RR, P_LS, P_LR, P_M, P_NP, P_J, P_B = 0, 1, 2, 3, 4, 5, 6, 7
NPARAMS = 8

PARAMS_DEFAULT = np.array([Rs, Rr, Ls, Lr, M_, float(NP), J, B], dtype=np.float64)

# Definizione UI dei parametri: chiave, label, valore SI, fattore di display, unità
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
NH = 18
HIST_FIELDS = ["t", "wm", "Ce", "Ps", "Qs", "Pr", "slip",
               "isd", "isq", "ird", "irq",
               "psd", "psq", "prd", "prq",
               "Qr", "PRs", "PRr"]

# Catalogo segnali plottabili (chiave hist · nome leggibile · unità · RGB)
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
]
SIG_BY_KEY = {k: dict(name=n, unit=u, r=r, g=g, b=b)
              for k, n, u, r, g, b in SIGNALS_LIST}

MAX_PTS = 8000                   # 60 s × 1 sample / 7.5 ms
LOG_INTERVAL = 5e-3              # subsampling history: 5 ms simulato
MIN_ITER_DT = 5e-3               # ogni iter copre ≥ 5 ms simulati
MAX_ITER_DT = 100e-3             # clamp anti-stutter
MAX_STEPS_PER_ITER = 200_000     # safety cap
MAX_SAMPLES_PER_TICK = 256
RESYNC_LAG = 0.200               # se sim resta indietro di > 200 ms, riaggancia anchor

INT_EULER, INT_RK4, INT_RK45 = 0, 1, 2


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True, fastmath=True, nogil=True)
def deriv(s, Vs, ws, Vr, wr, Cl, params, out):
    Rs = params[0]; Rr = params[1]
    Ls = params[2]; Lr = params[3]; M_ = params[4]
    NP = params[5]; J = params[6]; B = params[7]
    D = Ls * Lr - M_ * M_
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    wm = s[4]; thm = s[5]; t = s[6]
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
def observe(s, Vs, ws, Vr, wr, params, out):
    Rs = params[0]; Rr = params[1]
    Ls = params[2]; Lr = params[3]; M_ = params[4]
    NP = params[5]
    D = Ls * Lr - M_ * M_
    psd = s[0]; psq = s[1]; prd = s[2]; prq = s[3]
    wm = s[4]; thm = s[5]; t = s[6]
    a = wr * t - ws * t + NP * thm
    vrd = Vr * math.cos(a); vrq = Vr * math.sin(a)
    isd = (Lr * psd - M_ * prd) / D
    isq = (Lr * psq - M_ * prq) / D
    ird = (Ls * prd - M_ * psd) / D
    irq = (Ls * prq - M_ * psq) / D
    Ce = NP * M_ * (isq * ird - isd * irq)
    Ps = Vs * isd
    Qs = -Vs * isq
    Pr = vrd * ird + vrq * irq
    Qr = vrq * ird - vrd * irq
    PRs = Rs * (isd * isd + isq * isq)
    PRr = Rr * (ird * ird + irq * irq)
    slip = (ws - NP * wm) / ws * 100.0 if ws > 0 else 0.0
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


@njit(cache=True, fastmath=True, nogil=True)
def _maybe_sample(s, Vs, ws, Vr, wr, params, last_log_t, samples, n_samples, max_samples, obs):
    if s[6] - last_log_t >= LOG_INTERVAL and n_samples < max_samples:
        observe(s, Vs, ws, Vr, wr, params, obs)
        for j in range(NH):
            samples[n_samples, j] = obs[j]
        n_samples += 1
        last_log_t += LOG_INTERVAL
        # se molto indietro (es. uscita da pausa): aggancia al presente
        if last_log_t < s[6] - 10.0 * LOG_INTERVAL:
            last_log_t = s[6]
    return n_samples, last_log_t


@njit(cache=True, fastmath=True, nogil=True)
def advance_euler(s, t_target, Vs, ws, Vr, wr, Cl, dt, params,
                  last_log_t, samples, max_samples):
    k = np.empty(7)
    obs = np.empty(NH)
    n_steps = 0
    n_samples = 0
    while s[6] < t_target and n_steps < MAX_STEPS_PER_ITER:
        deriv(s, Vs, ws, Vr, wr, Cl, params, k)
        for i in range(7):
            s[i] += dt * k[i]
        n_steps += 1
        n_samples, last_log_t = _maybe_sample(s, Vs, ws, Vr, wr, params, last_log_t,
                                               samples, n_samples, max_samples, obs)
    return n_steps, n_samples, last_log_t


@njit(cache=True, fastmath=True, nogil=True)
def advance_rk4(s, t_target, Vs, ws, Vr, wr, Cl, dt, params,
                last_log_t, samples, max_samples):
    k1 = np.empty(7); k2 = np.empty(7); k3 = np.empty(7); k4 = np.empty(7)
    tmp = np.empty(7); obs = np.empty(NH)
    n_steps = 0
    n_samples = 0
    while s[6] < t_target and n_steps < MAX_STEPS_PER_ITER:
        deriv(s, Vs, ws, Vr, wr, Cl, params, k1)
        for i in range(7): tmp[i] = s[i] + 0.5 * dt * k1[i]
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k2)
        for i in range(7): tmp[i] = s[i] + 0.5 * dt * k2[i]
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k3)
        for i in range(7): tmp[i] = s[i] + dt * k3[i]
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k4)
        h6 = dt / 6.0
        for i in range(7):
            s[i] += h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        n_steps += 1
        n_samples, last_log_t = _maybe_sample(s, Vs, ws, Vr, wr, params, last_log_t,
                                               samples, n_samples, max_samples, obs)
    return n_steps, n_samples, last_log_t


# Dormand-Prince 5(4) coefficients
@njit(cache=True, fastmath=True, nogil=True)
def advance_rk45(s, t_target, Vs, ws, Vr, wr, Cl,
                 dt_init, dt_min, dt_max, rtol, params,
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

    while s[6] < t_target and (n_steps + n_rej) < MAX_STEPS_PER_ITER:
        if s[6] + dt > t_target:
            dt = t_target - s[6]
        if dt <= 0.0:
            break

        deriv(s, Vs, ws, Vr, wr, Cl, params, k1)
        for i in range(7): tmp[i] = s[i] + dt * (0.2 * k1[i])
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k2)
        for i in range(7): tmp[i] = s[i] + dt * (3.0/40.0*k1[i] + 9.0/40.0*k2[i])
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k3)
        for i in range(7): tmp[i] = s[i] + dt * (44.0/45.0*k1[i] - 56.0/15.0*k2[i] + 32.0/9.0*k3[i])
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k4)
        for i in range(7):
            tmp[i] = s[i] + dt * (19372.0/6561.0*k1[i] - 25360.0/2187.0*k2[i]
                                  + 64448.0/6561.0*k3[i] - 212.0/729.0*k4[i])
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k5)
        for i in range(7):
            tmp[i] = s[i] + dt * (9017.0/3168.0*k1[i] - 355.0/33.0*k2[i]
                                  + 46732.0/5247.0*k3[i] + 49.0/176.0*k4[i]
                                  - 5103.0/18656.0*k5[i])
        deriv(tmp, Vs, ws, Vr, wr, Cl, params, k6)
        # 5th order solution
        for i in range(7):
            s5[i] = s[i] + dt * (35.0/384.0*k1[i] + 500.0/1113.0*k3[i]
                                 + 125.0/192.0*k4[i] - 2187.0/6784.0*k5[i]
                                 + 11.0/84.0*k6[i])
        deriv(s5, Vs, ws, Vr, wr, Cl, params, k7)
        # error = (b5 - b4) · dt · k
        e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0
        e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
        for i in range(7):
            err_vec[i] = dt * (e1*k1[i] + e3*k3[i] + e4*k4[i]
                               + e5*k5[i] + e6*k6[i] + e7*k7[i])
        # norma scalata
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
            n_samples, last_log_t = _maybe_sample(s, Vs, ws, Vr, wr, params, last_log_t,
                                                   samples, n_samples, max_samples, obs)
        else:
            n_rej += 1
            factor = 0.9 * err_norm ** (-0.2)
            if factor < 0.1: factor = 0.1
            dt *= factor
            if dt < dt_min:
                # forziamo l'accettazione: il sistema è degenere a questo dt
                dt = dt_min

    avg_dt = sum_dt / n_steps if n_steps > 0 else 0.0
    return n_steps, n_samples, last_log_t, n_rej, err_max, avg_dt, dt


def warmup_jit():
    """Forza la compilazione AOT-cached delle funzioni numba."""
    s = np.zeros(7)
    s[1] = -2.2; s[3] = -2.2; s[4] = 100.0
    samples = np.empty((4, NH))
    p = PARAMS_DEFAULT.copy()
    advance_euler(s.copy(), 1e-3, 690.0, 314.159, 0.0, 0.0, 0.0, 1e-4, p, 0.0, samples, 4)
    advance_rk4(s.copy(),   1e-3, 690.0, 314.159, 0.0, 0.0, 0.0, 1e-4, p, 0.0, samples, 4)
    advance_rk45(s.copy(),  1e-3, 690.0, 314.159, 0.0, 0.0, 0.0,
                 1e-4, 1e-7, 1e-3, 1e-4, p, 0.0, samples, 4)


# ============================================================
# Sim engine (thread)
# ============================================================

class SimEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = np.zeros(7)
        # Parametri macchina editabili a runtime (passati ai kernel JIT)
        self.params = PARAMS_DEFAULT.copy()
        # Ring buffer numpy (zero-garbage snapshot): un memcpy invece di 15× list copy
        self.hist_buf = np.zeros((MAX_PTS, NH), dtype=np.float64)
        self.hist_write = 0     # numero totale di sample scritti (monotonico)
        self.hist_count = 0     # min(hist_write, MAX_PTS)
        self.ctrl = {
            "Vs": 690.0, "fs": 50.0, "Vr": 0.0, "fr": 0.0, "Cl": 0.0,
            "logdt": -4.0, "rtol": 1e-9,
            "integrator": INT_RK45,
            "rt_lock": True,
            "speed_factor": 1.0,   # target = wall × speed
        }
        self.stats = {
            "compiled": False,
            "running": False,
            "integrator": INT_RK4,
            "rt_lock": True,
            "steps_per_s": 0.0,
            "dt_eff_us": 0.0,
            "rt_factor": 0.0,    # vero: sim_time / wall_time (incl. sleep)
            "headroom": 0.0,     # margine CPU: sim_dt / wall_busy
            "lag_ms": 0.0,       # quanto siamo indietro rispetto al wall
            "rejected_pct": 0.0,
            "err_norm_max": 0.0,
            "iter_wall_ms": 0.0,
            "resyncs": 0,
        }
        self.running = False
        self._stop = False
        self.thread = None
        # Ancore real-time: target_sim = t_sim_anchor + (wall - t_wall_anchor) * speed
        self._t_wall_anchor = 0.0
        self._t_sim_anchor = 0.0
        self._last_log_t = 0.0
        self._dt_rk45 = 1e-4  # carry-over dt per RK45
        self._reset_state_locked()

    def _reset_state_locked(self):
        p0 = 690.0 / (2.0 * math.pi * 50.0)
        Ls_p = self.params[P_LS]; M_p = self.params[P_M]; NP_p = self.params[P_NP]
        self.state[:] = [0.0, -p0, 0.0, -p0 * M_p / Ls_p,
                         2.0 * math.pi * 50.0 / NP_p, 0.0, 0.0]
        self.hist_write = 0
        self.hist_count = 0
        self._last_log_t = 0.0
        self._t_wall_anchor = time.perf_counter()
        self._t_sim_anchor = 0.0
        self._dt_rk45 = 1e-4

    def reset(self):
        with self.lock:
            self._reset_state_locked()

    def _rebase_anchor_locked(self):
        """Riallinea ancore al qui-e-ora (per cambio speed o resume)."""
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
        """Snapshot zero-garbage: un memcpy del ring buffer."""
        with self.lock:
            n = self.hist_count
            write = self.hist_write
            if n == 0:
                arr = np.empty((0, NH), dtype=np.float64)
            elif write <= MAX_PTS:
                # buffer non ancora pieno: range contiguo [0, n)
                arr = self.hist_buf[:n].copy()
            else:
                # buffer pieno: rotazione
                head = write % MAX_PTS  # prossima posizione di scrittura = inizio cronologico
                arr = np.empty((MAX_PTS, NH), dtype=np.float64)
                arr[:MAX_PTS - head] = self.hist_buf[head:]
                arr[MAX_PTS - head:] = self.hist_buf[:head]
            state = self.state.copy()
            stats = dict(self.stats)
            ctrl = dict(self.ctrl)
            params = self.params.copy()
        # Wrap come SimpleNamespace con view numpy per ogni campo (zero copia ulteriore)
        snap = SimpleNamespace()
        for i, f in enumerate(HIST_FIELDS):
            setattr(snap, f, arr[:, i])
        return snap, state, stats, ctrl, params

    def set_param(self, idx, value):
        """Aggiorna un parametro macchina a runtime (modifica in-place sotto lock)."""
        with self.lock:
            if 0 <= idx < NPARAMS and value > 0:
                self.params[idx] = value

    def set_ctrl(self, **kw):
        with self.lock:
            speed_changed = ("speed_factor" in kw and
                             kw["speed_factor"] != self.ctrl.get("speed_factor"))
            self.ctrl.update(kw)
            if speed_changed and self.running:
                # Ricalibra ancore al qui-e-ora per evitare salti del target
                self._rebase_anchor_locked()

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

        # Per il vero RT factor: tempo wall/sim alla fine dell'iter precedente
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

            # Snapshot ctrl + stato (sotto lock)
            with self.lock:
                Vs = self.ctrl["Vs"]; fs = self.ctrl["fs"]
                Vr = self.ctrl["Vr"]; fr = self.ctrl["fr"]
                Cl = self.ctrl["Cl"]
                dt_max = 10.0 ** self.ctrl["logdt"]
                integrator = self.ctrl["integrator"]
                rt_lock = self.ctrl["rt_lock"]
                rtol = self.ctrl["rtol"]
                speed = self.ctrl.get("speed_factor", 1.0)
                state_local = self.state.copy()
                params_local = self.params.copy()
                last_log_t = self._last_log_t
                t_wall_anchor = self._t_wall_anchor
                t_sim_anchor = self._t_sim_anchor
                dt_rk45 = self._dt_rk45

            ws = 2.0 * math.pi * fs
            wr = 2.0 * math.pi * fr

            now = time.perf_counter()
            # target = t_sim_anchor + (now - t_wall_anchor) * speed
            target_sim = t_sim_anchor + (now - t_wall_anchor) * speed
            lag = target_sim - state_local[6]   # quanto siamo indietro rispetto al target

            if rt_lock:
                # Auto-resync: se siamo troppo indietro, riallinea le ancore
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
                    # ancora troppo presto, aspettiamo
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
                    state_local, t_target, Vs, ws, Vr, wr, Cl, dt_max, params_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)
            elif integrator == INT_RK4:
                n_steps, n_samples, last_log_t = advance_rk4(
                    state_local, t_target, Vs, ws, Vr, wr, Cl, dt_max, params_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)
            else:  # RK45
                dt_min = max(1e-9, dt_max * 1e-4)
                (n_steps, n_samples, last_log_t,
                 n_rej, err_max, dt_avg, dt_rk45) = advance_rk45(
                    state_local, t_target, Vs, ws, Vr, wr, Cl,
                    dt_rk45, dt_min, dt_max, rtol, params_local,
                    last_log_t, sample_buf, MAX_SAMPLES_PER_TICK)

            t1 = time.perf_counter()
            wall_busy = t1 - t0
            sim_dt = state_local[6] - t_sim_start

            # Push samples + commit stato — scrittura nel ring buffer (un memcpy)
            with self.lock:
                self.state[:] = state_local
                self._last_log_t = last_log_t
                self._dt_rk45 = dt_rk45
                if n_samples > 0:
                    write = self.hist_write
                    # scrittura in blocco gestendo wrap-around
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

            # Sleep se in vantaggio sul target (mantiene l'aggancio wall × speed)
            if rt_lock and speed > 0:
                # quando lo state.t avrà raggiunto il target → wall_at_target
                wall_at_target = t_wall_anchor + (state_local[6] - t_sim_anchor) / speed
                slack = wall_at_target - time.perf_counter()
                if slack > 0:
                    time.sleep(min(slack, 0.020))

            # Metriche dopo lo sleep — ora "wall_real" è il tempo VERO trascorso
            t_iter_end = time.perf_counter()
            if t_wall_prev is not None and wall_busy > 0:
                wall_real = t_iter_end - t_wall_prev
                if wall_real > 0:
                    sim_step = state_local[6] - t_sim_prev
                    # rt rispetto al TARGET: 1.0 = on-target a qualsiasi speed
                    rt_target_inst = (sim_step / wall_real) / speed if speed > 0 else 0.0
                    headroom_inst = sim_dt / wall_busy   # quanto-veloce-potrei-andare
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


# ============================================================
# Drawing (snapshot-based; identica logica di prima)
# ============================================================

def nice_ticks(lo, hi, n_target=6):
    """Tick 'tondi' (1·10ⁿ, 2·10ⁿ, 5·10ⁿ) tra lo e hi, ~n_target tick."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(n_target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm < 1.5: nice = 1.0
    elif norm < 3.0: nice = 2.0
    elif norm < 7.0: nice = 5.0
    else: nice = 10.0
    step = nice * mag
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + step * 1e-6:
        out.append(v)
        v += step
    return out


def fmt_axis(v, step):
    """Formatter compatto per label di asse, basato sullo step."""
    av = abs(v)
    if av < step * 0.001:
        return "0"
    if av >= 1000:
        if av >= 1e6: return f"{v/1e6:.1f}M"
        return f"{v/1000:.1f}k"
    if step >= 1:    return f"{v:.0f}"
    if step >= 0.1:  return f"{v:.1f}"
    if step >= 0.01: return f"{v:.2f}"
    return f"{v:.3g}"


def draw_dq(area, cr, w, h, snap, traces, t_win, title):
    cr.set_source_rgb(0.027, 0.047, 0.086); cr.rectangle(0, 0, w, h); cr.fill()
    S = min(w, h)
    if snap is None or len(snap.t) < 2:
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(10); cr.move_to(S / 2 - 30, S / 2); cr.show_text("premi RUN")
        return

    t_arr = snap.t
    tn = float(t_arr[-1]); ts = tn - t_win
    i0 = int(np.searchsorted(t_arr, ts, side="left"))
    n_visible = len(t_arr) - i0
    if n_visible < 2:
        return

    # Decimazione: cap a ~S punti per traiettoria
    stride = max(1, n_visible // max(int(S), 1))

    # Range vettorializzato
    am = 1.0
    for fd, fq, *_ in traces:
        dd = getattr(snap, fd)[i0::stride]
        qq = getattr(snap, fq)[i0::stride]
        if dd.size:
            v = max(float(np.abs(dd).max()), float(np.abs(qq).max()))
            if v > am: am = v
    am *= 1.2

    def mx(v): return S / 2 + v / am * (S / 2 - 14)
    def my(v): return S / 2 - v / am * (S / 2 - 14)

    cr.set_source_rgb(0.06, 0.09, 0.16); cr.set_line_width(0.5)
    for r in (0.25, 0.5, 0.75, 1.0):
        cr.arc(S / 2, S / 2, r * (S / 2 - 14), 0, 2 * math.pi); cr.stroke()
    cr.set_source_rgb(0.14, 0.18, 0.26); cr.set_line_width(1)
    cr.move_to(S / 2, 6); cr.line_to(S / 2, S - 6); cr.stroke()
    cr.move_to(6, S / 2); cr.line_to(S - 6, S / 2); cr.stroke()

    # Tick numerati sugli assi d e q
    cr.set_source_rgb(0.45, 0.52, 0.62); cr.set_font_size(10)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    tick_ticks = nice_ticks(-am, am, 5)
    for tv in tick_ticks:
        if abs(tv) < am * 0.05: continue   # salto lo "0" centrale
        if abs(tv) > am: continue
        # asse d (orizzontale)
        x = S / 2 + tv / am * (S / 2 - 14)
        cr.set_source_rgb(0.20, 0.26, 0.36); cr.set_line_width(1)
        cr.move_to(x, S / 2 - 3); cr.line_to(x, S / 2 + 3); cr.stroke()
        cr.set_source_rgb(0.45, 0.52, 0.62)
        label = fmt_axis(tv, tick_ticks[1] - tick_ticks[0] if len(tick_ticks) > 1 else 1)
        ext = cr.text_extents(label)
        cr.move_to(x - ext.width / 2, S / 2 + 14); cr.show_text(label)
        # asse q (verticale)
        y = S / 2 - tv / am * (S / 2 - 14)
        cr.set_source_rgb(0.20, 0.26, 0.36); cr.set_line_width(1)
        cr.move_to(S / 2 - 3, y); cr.line_to(S / 2 + 3, y); cr.stroke()
        cr.set_source_rgb(0.45, 0.52, 0.62)
        cr.move_to(S / 2 + 5, y + 3); cr.show_text(label)

    # Etichette assi d / q
    cr.set_source_rgb(0.55, 0.62, 0.72); cr.set_font_size(12)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.move_to(S - 14, S / 2 - 6); cr.show_text("d")
    cr.move_to(S / 2 + 6, 16); cr.show_text("q")

    for fd, fq, r, g, b, label in traces:
        dd_full = getattr(snap, fd); qq_full = getattr(snap, fq)
        dd = dd_full[i0::stride]; qq = qq_full[i0::stride]
        n_dec = len(dd)
        if n_dec < 2: continue
        # Tratte con alpha crescente (scia)
        for i in range(1, n_dec):
            alpha = 0.05 + 0.95 * i / n_dec
            cr.set_source_rgba(r, g, b, alpha); cr.set_line_width(1.3)
            cr.move_to(mx(float(dd[i-1])), my(float(qq[i-1])))
            cr.line_to(mx(float(dd[i])), my(float(qq[i]))); cr.stroke()
        ld, lq = float(dd_full[-1]), float(qq_full[-1])
        cr.set_source_rgb(r, g, b); cr.arc(mx(ld), my(lq), 5, 0, 2 * math.pi); cr.fill()
        cr.set_source_rgba(r, g, b, 0.5); cr.set_line_width(1.5)
        cr.move_to(S / 2, S / 2); cr.line_to(mx(ld), my(lq)); cr.stroke()
        ang = math.atan2(-lq, ld); ax, ay, al = mx(ld), my(lq), 10
        for da in (-0.3, 0.3):
            cr.move_to(ax, ay)
            cr.line_to(ax - al * math.cos(ang + da), ay + al * math.sin(ang + da))
            cr.stroke()

    cr.set_font_size(11)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ly = S - 10
    for *_, r, g, b, label in reversed(traces):
        cr.set_source_rgb(r, g, b); cr.move_to(8, ly); cr.show_text(f"● {label}"); ly -= 16

    cr.set_source_rgb(0.62, 0.70, 0.80); cr.set_font_size(13)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ext = cr.text_extents(title); cr.move_to((S - ext.width) / 2, 16); cr.show_text(title)


def draw_tplot(area, cr, w, h, snap, signal_keys, t_win, title,
               plot_idx, y_persist, cursor_state):
    """Plot tempo-serie con:
       - asse Y isteretico (si allarga, mai si stringe finché non si reset)
       - tick + grid sottile su entrambi gli assi
       - cursore verticale cross-plot + tooltip valori
       - tracce dinamiche dal catalogo SIG_BY_KEY
    """
    # Margini area di plot
    M_L, M_R, M_T, M_B = 60, 10, 28, 22
    plot_w = max(1, w - M_L - M_R)
    plot_h = max(1, h - M_T - M_B)

    # Sfondo
    cr.set_source_rgb(0.027, 0.047, 0.086); cr.rectangle(0, 0, w, h); cr.fill()

    # Titolo (alto-sx)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(13); cr.set_source_rgb(0.62, 0.70, 0.80)
    cr.move_to(8, 18); cr.show_text(title)

    # Empty state
    if not signal_keys:
        cr.set_source_rgb(0.30, 0.35, 0.42); cr.set_font_size(13)
        msg = "(plot vuoto — usa il composer TRACCE a destra)"
        ext = cr.text_extents(msg)
        cr.move_to((w - ext.width) / 2, h / 2); cr.show_text(msg)
        return
    if snap is None or len(snap.t) < 2:
        cr.set_source_rgb(0.30, 0.35, 0.42); cr.set_font_size(13)
        ext = cr.text_extents("premi RUN")
        cr.move_to((w - ext.width) / 2, h / 2); cr.show_text("premi RUN")
        return

    t_arr = snap.t
    tn = float(t_arr[-1]); ts = tn - t_win
    i0 = int(np.searchsorted(t_arr, ts, side="left"))
    n_visible = len(t_arr) - i0
    if n_visible < 2:
        return

    # Decimazione: max ~plot_w punti per stroke
    stride = max(1, n_visible // plot_w)

    # Range dati nella finestra visibile (vettorializzato)
    mn_d, mx_d = float("inf"), float("-inf")
    for key in signal_keys:
        if key not in SIG_BY_KEY: continue
        a = getattr(snap, key, None)
        if a is None: continue
        sub = a[i0::stride]
        if sub.size:
            v_mn = float(sub.min()); v_mx = float(sub.max())
            if v_mn < mn_d: mn_d = v_mn
            if v_mx > mx_d: mx_d = v_mx
    if mn_d == float("inf"):
        return
    if mn_d == mx_d:
        mn_d -= 1; mx_d += 1

    # ASSE Y ISTERETICO: si allarga, mai si stringe
    persist = y_persist.get(plot_idx)
    if persist is None:
        mn_p, mx_p = mn_d, mx_d
    else:
        mn_p, mx_p = persist
        if mn_d < mn_p: mn_p = mn_d
        if mx_d > mx_p: mx_p = mx_d
    y_persist[plot_idx] = (mn_p, mx_p)

    span = mx_p - mn_p
    pad = span * 0.06 if span > 0 else 1.0
    mn = mn_p - pad; mx = mx_p + pad

    def xof(t): return M_L + (t - ts) / t_win * plot_w
    def yof(v): return M_T + plot_h - (v - mn) / (mx - mn) * plot_h

    # === GRID + TICK ===
    y_ticks = nice_ticks(mn, mx, 5)
    x_ticks = nice_ticks(ts, tn, 6)
    y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
    x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1.0

    # Grid orizzontale (a livello dei tick Y)
    cr.set_source_rgb(0.10, 0.14, 0.20); cr.set_line_width(0.5)
    for v in y_ticks:
        y = yof(v)
        if M_T <= y <= M_T + plot_h:
            cr.move_to(M_L, y); cr.line_to(M_L + plot_w, y); cr.stroke()
    # Grid verticale (a livello dei tick X)
    for tt in x_ticks:
        x = xof(tt)
        if M_L <= x <= M_L + plot_w:
            cr.move_to(x, M_T); cr.line_to(x, M_T + plot_h); cr.stroke()

    # Linea zero (più marcata)
    if mn < 0 < mx:
        zy = yof(0.0)
        cr.set_source_rgb(0.20, 0.26, 0.36); cr.set_line_width(1.0)
        cr.set_dash([3, 3]); cr.move_to(M_L, zy); cr.line_to(M_L + plot_w, zy); cr.stroke()
        cr.set_dash([])

    # Cornice del plot
    cr.set_source_rgb(0.18, 0.22, 0.30); cr.set_line_width(1)
    cr.rectangle(M_L, M_T, plot_w, plot_h); cr.stroke()

    # Tracce — coordinate vettorializzate, decimate a stride
    t_sub = t_arr[i0::stride]
    xs_arr = M_L + (t_sub - ts) / t_win * plot_w
    sig_legend = []
    for key in signal_keys:
        if key not in SIG_BY_KEY: continue
        meta = SIG_BY_KEY[key]
        a = getattr(snap, key, None)
        if a is None: continue
        a_sub = a[i0::stride]
        if a_sub.size < 2: continue
        ys_arr = M_T + plot_h - (a_sub - mn) / (mx - mn) * plot_h
        cr.set_source_rgb(meta["r"], meta["g"], meta["b"]); cr.set_line_width(1.8)
        cr.move_to(float(xs_arr[0]), float(ys_arr[0]))
        for k in range(1, len(xs_arr)):
            cr.line_to(float(xs_arr[k]), float(ys_arr[k]))
        cr.stroke()
        sig_legend.append((key, meta))

    # === Tick label asse Y (sx) ===
    cr.set_source_rgb(0.55, 0.62, 0.72); cr.set_font_size(11)
    cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    for v in y_ticks:
        y = yof(v)
        if M_T - 4 <= y <= M_T + plot_h + 4:
            label = fmt_axis(v, y_step)
            ext = cr.text_extents(label)
            cr.move_to(M_L - ext.width - 4, y + 4)
            cr.show_text(label)
            # tick mark
            cr.set_line_width(1)
            cr.move_to(M_L - 3, y); cr.line_to(M_L, y); cr.stroke()

    # === Tick label asse X (sotto) ===
    for tt in x_ticks:
        x = xof(tt)
        if M_L <= x <= M_L + plot_w:
            label = f"{tt:.1f}s"
            ext = cr.text_extents(label)
            cr.move_to(x - ext.width / 2, h - 6)
            cr.show_text(label)
            # tick mark
            cr.move_to(x, M_T + plot_h); cr.line_to(x, M_T + plot_h + 3); cr.stroke()

    # Legenda (alto-dx)
    cr.set_font_size(12); lx = w - 8
    for key, meta in reversed(sig_legend):
        cr.set_source_rgb(meta["r"], meta["g"], meta["b"])
        label = f"{meta['name']} [{meta['unit']}]"
        ext = cr.text_extents(label)
        lx -= ext.width + 22
        cr.move_to(lx + 16, 18); cr.show_text(label)
        cr.set_line_width(2.0)
        cr.move_to(lx, 14); cr.line_to(lx + 14, 14); cr.stroke()

    # === Cursore + tooltip cross-plot ===
    # Il cursore è ancorato alla x dello schermo (sotto il puntatore del
    # mouse). Il tempo si ricava dalla x usando lo snap corrente: così,
    # mentre i dati scorrono, il cursore resta esattamente sotto al mouse
    # senza derivare né scattare.
    cx = cursor_state.get("x")
    if cx is not None and M_L <= cx <= M_L + plot_w:
        x_cur = cx
        ct = ts + (cx - M_L) / plot_w * t_win
        cr.set_source_rgba(0.92, 0.92, 0.95, 0.5)
        cr.set_line_width(1); cr.set_dash([3, 2])
        cr.move_to(x_cur, M_T); cr.line_to(x_cur, M_T + plot_h); cr.stroke()
        cr.set_dash([])

        # Binary search via numpy
        ts_arr = snap.t
        idx = int(np.searchsorted(ts_arr, ct, side="left"))
        if idx >= len(ts_arr): idx = len(ts_arr) - 1
        if idx > 0 and abs(ts_arr[idx - 1] - ct) < abs(ts_arr[idx] - ct):
            idx -= 1

        # Pallini sulle tracce alla x del cursore
        for key, meta in sig_legend:
            a = getattr(snap, key, None)
            if a is None or idx >= len(a): continue
            cr.set_source_rgb(meta["r"], meta["g"], meta["b"])
            cr.arc(x_cur, yof(a[idx]), 4, 0, 2 * math.pi); cr.fill()

        # Tooltip — numeri con larghezza fissa (niente più jitter per cifre variabili)
        lines = [(None, f"t = {ts_arr[idx]:>10.3f} s")]
        for key, meta in sig_legend:
            a = getattr(snap, key, None)
            if a is None or idx >= len(a): continue
            lines.append((meta, f"{meta['name']:<5s} = {a[idx]:>10.3f} {meta['unit']}"))

        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)
        max_w = 0
        for _, line in lines:
            ext = cr.text_extents(line)
            if ext.width > max_w: max_w = ext.width
        line_h = 16
        tip_w = max_w + 16
        tip_h = line_h * len(lines) + 10

        # Flip basato sulla posizione del cursore, non sulla larghezza del tooltip
        if x_cur > w * 0.6:
            tip_x = x_cur - tip_w - 8
        else:
            tip_x = x_cur + 8
        if tip_x < 2: tip_x = 2
        if tip_x + tip_w > w - 2: tip_x = w - tip_w - 2
        tip_y = M_T + 4
        if tip_y + tip_h > M_T + plot_h - 4:
            tip_y = M_T + plot_h - tip_h - 4

        cr.set_source_rgba(0.04, 0.06, 0.10, 0.93)
        cr.rectangle(tip_x, tip_y, tip_w, tip_h); cr.fill()
        cr.set_source_rgba(0.30, 0.35, 0.45, 1.0); cr.set_line_width(0.5)
        cr.rectangle(tip_x, tip_y, tip_w, tip_h); cr.stroke()

        for li, (meta, line) in enumerate(lines):
            if meta is None:
                cr.set_source_rgb(0.85, 0.88, 0.95)
            else:
                cr.set_source_rgb(meta["r"], meta["g"], meta["b"])
            cr.move_to(tip_x + 7, tip_y + line_h * (li + 1))
            cr.show_text(line)


# ============================================================
# UI helpers
# ============================================================

def make_slider(label, val, lo, hi, stp, cb):
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    nl = Gtk.Label(label=label); nl.set_halign(Gtk.Align.START)
    nl.set_markup(f'<span font_family="monospace" foreground="#94a3b8" size="small">{label}</span>')
    vl = Gtk.Label(); vl.set_halign(Gtk.Align.END)
    row.append(nl); row.append(vl); row.set_hexpand(True); nl.set_hexpand(True)
    adj = Gtk.Adjustment(value=val, lower=lo, upper=hi, step_increment=stp, page_increment=stp * 10)
    sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
    sc.set_draw_value(False); sc.set_hexpand(True)

    def on_ch(s):
        v = s.get_value()
        if abs(v) >= 1000: t = f"{v/1000:.1f}k"
        elif abs(v) >= 100: t = f"{v:.0f}"
        elif abs(v) >= 1: t = f"{v:.1f}"
        else: t = f"{v:.2f}"
        vl.set_markup(f'<span font_family="monospace" foreground="#e2e8f0" size="small">{t}</span>')
        cb(v)

    sc.connect("value-changed", on_ch); on_ch(sc)
    box.append(row); box.append(sc)
    return box


def make_param_input(label_text, initial_disp, unit, on_change):
    """Widget composito: label + entry numerica + frecce ▲▼ (±10%).
    `on_change(v_disp)` riceve il valore di display (in unità mostrate)."""
    def fmt(v):
        if abs(v) >= 1000: return f"{v:.1f}"
        if abs(v) >= 100:  return f"{v:.2f}"
        if abs(v) >= 1:    return f"{v:.3f}"
        return f"{v:.4f}"

    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
    lbl = Gtk.Label()
    lbl.set_markup(f'<span font_family="monospace" foreground="#94a3b8" size="small">{label_text}</span>')
    lbl.set_halign(Gtk.Align.START); lbl.set_size_request(34, -1)
    box.append(lbl)

    entry = Gtk.Entry()
    entry.set_text(fmt(initial_disp))
    entry.set_max_width_chars(8); entry.set_width_chars(8)
    entry.set_property("xalign", 1.0)
    box.append(entry)

    if unit:
        ul = Gtk.Label()
        ul.set_markup(f'<span font_family="monospace" foreground="#475569" size="x-small">{unit}</span>')
        ul.set_halign(Gtk.Align.START); ul.set_size_request(40, -1)
        box.append(ul)

    btn_dn = Gtk.Button(label="▼")
    btn_dn.set_size_request(24, 22)
    btn_up = Gtk.Button(label="▲")
    btn_up.set_size_request(24, 22)

    def parse_and_apply(new_v):
        if new_v <= 0: return  # protezione (R, L, J, B, NP devono essere > 0)
        entry.set_text(fmt(new_v))
        on_change(new_v)

    def on_entry_activate(_e):
        try:
            v = float(entry.get_text())
        except ValueError:
            entry.set_text(fmt(initial_disp))
            return
        parse_and_apply(v)

    def on_dn_clicked(_b):
        try: v = float(entry.get_text())
        except ValueError: return
        parse_and_apply(v * 0.9)

    def on_up_clicked(_b):
        try: v = float(entry.get_text())
        except ValueError: return
        parse_and_apply(v * 1.1)

    entry.connect("activate", on_entry_activate)
    btn_dn.connect("clicked", on_dn_clicked)
    btn_up.connect("clicked", on_up_clicked)
    box.append(btn_dn); box.append(btn_up)
    return box


# ============================================================
# Application
# ============================================================

def run():
    engine = SimEngine()
    engine.start()

    app = Gtk.Application(application_id="org.dfig.sim")
    timer_id = [None]
    twin = [10.0]

    def on_activate(a):
        win = Gtk.ApplicationWindow(application=a)
        win.set_title("DFIG — Simulatore Interattivo (real-time + adattativo)")
        win.set_default_size(1280, 960)

        css = Gtk.CssProvider()
        css.load_from_string("window { background-color: #060a12; }")
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        for m in ("set_margin_start", "set_margin_end", "set_margin_top", "set_margin_bottom"):
            getattr(vbox, m)(8)

        # ---- Header ----
        hdr = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        tbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        t1 = Gtk.Label(); t1.set_halign(Gtk.Align.START)
        t1.set_markup('<span font_family="monospace" font_weight="bold" foreground="#f1f5f9" size="large">DFIG — Simulatore Interattivo</span>')
        time_lbl = Gtk.Label(); time_lbl.set_halign(Gtk.Align.START)
        time_lbl.set_markup('<span font_family="monospace" foreground="#475569" size="small">compilazione JIT…</span>')
        tbox.append(t1); tbox.append(time_lbl); tbox.set_hexpand(True)

        bbx = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        run_btn = Gtk.Button(label="▶ RUN")
        rst_btn = Gtk.Button(label="RESET")

        def on_run(_):
            running = engine.toggle_run()
            run_btn.set_label("■ STOP" if running else "▶ RUN")

        def on_reset(_):
            engine.reset()
            for da in drawing_areas:
                da.queue_draw()

        run_btn.connect("clicked", on_run); rst_btn.connect("clicked", on_reset)
        bbx.append(run_btn); bbx.append(rst_btn)

        # Indicatore RT factor grande nell'header
        rt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        rt_box.set_halign(Gtk.Align.END); rt_box.set_valign(Gtk.Align.CENTER)
        rt_box.set_margin_end(12)
        rt_main = Gtk.Label(); rt_main.set_halign(Gtk.Align.END)
        rt_main.set_markup('<span font_family="monospace" font_weight="bold" foreground="#475569" size="xx-large">— ×</span>')
        rt_sub = Gtk.Label(); rt_sub.set_halign(Gtk.Align.END)
        rt_sub.set_markup('<span font_family="monospace" foreground="#475569" size="x-small">vs real-time</span>')
        rt_box.append(rt_main); rt_box.append(rt_sub)

        hdr.append(tbox); hdr.append(rt_box); hdr.append(bbx)
        vbox.append(hdr)

        # ---- Controls ----
        cbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        cbox.set_homogeneous(True)

        def panel(title, color, children):
            f = Gtk.Frame()
            b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            for m in ("set_margin_start", "set_margin_end", "set_margin_top", "set_margin_bottom"):
                getattr(b, m)(6)
            lbl = Gtk.Label(); lbl.set_halign(Gtk.Align.START)
            lbl.set_markup(f'<span font_family="monospace" font_weight="bold" foreground="{color}" size="small">{title}</span>')
            b.append(lbl)
            for c in children: b.append(c)
            f.set_child(b)
            return f

        cbox.append(panel("STATORE", "#3b82f6", [
            make_slider("|V_s|", 690, 0, 1200, 10, lambda v: engine.set_ctrl(Vs=v)),
            make_slider("f_s [Hz]", 50, 0, 200, 0.5, lambda v: engine.set_ctrl(fs=v)),
        ]))

        fslip_lbl = Gtk.Label(); fslip_lbl.set_halign(Gtk.Align.START)
        fslip_lbl.set_markup('<span font_family="monospace" foreground="#78716c" size="small">f_slip = 0.0 Hz</span>')
        cbox.append(panel("ROTORE", "#ef4444", [
            make_slider("|V_r|", 0, 0, 200, 0.5, lambda v: engine.set_ctrl(Vr=v)),
            make_slider("f_r [Hz]", 0, -50, 50, 0.5, lambda v: engine.set_ctrl(fr=v)),
            fslip_lbl,
        ]))

        cbox.append(panel("CARICO", "#22c55e", [
            make_slider("C_load [Nm]", 0, -15000, 15000, 100, lambda v: engine.set_ctrl(Cl=v)),
        ]))

        # --- Pannello SIMULAZIONE: configurazione + speed factor ---
        sim_info = Gtk.Label(); sim_info.set_halign(Gtk.Align.START)
        sim_info.set_markup(
            '<span font_family="monospace" size="small">'
            '<span foreground="#94a3b8">RK45 Dormand-Prince · rtol=1e-9 · real-time lock</span>\n'
            '<span foreground="#475569" size="x-small">CPU usata per accuratezza · target = wall × speed</span>'
            '</span>')
        sim_info.set_use_markup(True)

        # Slider log10(speed_factor) — range 0.01× a 100×, default 1×
        def set_speed(v):
            sf = 10.0 ** v
            engine.set_ctrl(speed_factor=sf)

        speed_slider = make_slider("speed (×wall)", 0.0, -2.0, 2.0, 0.05, set_speed)
        # override del label: mostra sf invece di v
        speed_slider_inner = speed_slider.get_first_child()  # row con le label
        speed_val_lbl = speed_slider_inner.get_last_child()
        def update_speed_lbl(scale_widget):
            v = scale_widget.get_value()
            sf = 10.0 ** v
            if sf >= 100: t = f"{sf:.0f}×"
            elif sf >= 10: t = f"{sf:.1f}×"
            elif sf >= 1: t = f"{sf:.2f}×"
            else: t = f"{sf:.3f}×"
            speed_val_lbl.set_markup(
                f'<span font_family="monospace" foreground="#22c55e" font_weight="bold" size="small">{t}</span>')
        scale_widget = speed_slider.get_last_child()  # Gtk.Scale
        scale_widget.connect("value-changed", update_speed_lbl)
        update_speed_lbl(scale_widget)

        cbox.append(panel("SIMULAZIONE", "#eab308", [sim_info, speed_slider]))
        vbox.append(cbox)

        # ---- Pannello PARAMETRI MACCHINA (editabili a runtime) ----
        param_frame = Gtk.Frame()
        param_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for m in ("set_margin_start", "set_margin_end", "set_margin_top", "set_margin_bottom"):
            getattr(param_outer, m)(6)
        ph_lbl = Gtk.Label(); ph_lbl.set_halign(Gtk.Align.START)
        ph_lbl.set_markup('<span font_family="monospace" font_weight="bold" foreground="#06b6d4" size="small">PARAMETRI MACCHINA  <span foreground="#475569" font_weight="normal">(editabili: scrivi un valore + Invio, oppure ▼/▲ per ±10%)</span></span>')
        param_outer.append(ph_lbl)

        param_grid = Gtk.Grid()
        param_grid.set_column_spacing(14); param_grid.set_row_spacing(2)
        # 4 colonne × 2 righe = 8 parametri
        for i, (key, label, default_si, scale, unit) in enumerate(PARAMS_DEF):
            display_v = default_si / scale
            col = i % 4
            row = i // 4

            def make_cb(idx, sc):
                def cb(v_disp):
                    engine.set_param(idx, v_disp * sc)
                return cb

            widget = make_param_input(label, display_v, unit, make_cb(i, scale))
            param_grid.attach(widget, col, row, 1, 1)

        param_outer.append(param_grid)
        param_frame.set_child(param_outer)
        vbox.append(param_frame)

        # ---- d-q planes + gauges + diagnostica ----
        drawing_areas = []

        dq_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        dq_row.set_halign(Gtk.Align.CENTER)

        da_curr = Gtk.DrawingArea()
        da_curr.set_size_request(280, 280)
        da_curr.set_content_width(280); da_curr.set_content_height(280)
        traces_curr = [
            ("isd", "isq", 0.23, 0.51, 0.96, "ī_s (statore)"),
            ("ird", "irq", 0.94, 0.27, 0.27, "ī_r (rotore)"),
        ]

        def make_dq_drawer(traces, title):
            return lambda a, cr, w, h, *_: draw_dq(
                a, cr, w, h, render_state["snap"], traces, twin[0], title)

        render_state = {"snap": None, "stats": {}, "ctrl": {}}

        da_curr.set_draw_func(make_dq_drawer(traces_curr, "CORRENTI (d,q) [A]"))
        drawing_areas.append(da_curr); dq_row.append(da_curr)

        da_flux = Gtk.DrawingArea()
        da_flux.set_size_request(280, 280)
        da_flux.set_content_width(280); da_flux.set_content_height(280)
        traces_flux = [
            ("psd", "psq", 0.23, 0.51, 0.96, "φ̄_s (statore)"),
            ("prd", "prq", 0.94, 0.27, 0.27, "φ̄_r (rotore)"),
        ]
        da_flux.set_draw_func(make_dq_drawer(traces_flux, "FLUSSI (d,q) [mWb]"))
        drawing_areas.append(da_flux); dq_row.append(da_flux)

        # --- Solo diagnostica numerica (i valori istantanei sono nel composer) ---
        gbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        gbox.set_size_request(220, -1)
        gbox.set_margin_start(8)
        gauge_labels = {}  # vuoto: i gauge ora vivono dentro il composer

        diag_title = Gtk.Label(); diag_title.set_halign(Gtk.Align.START)
        diag_title.set_markup('<span font_family="monospace" font_weight="bold" foreground="#64748b" size="small">DIAGNOSTICA NUMERICA</span>')
        gbox.append(diag_title)

        diag_defs = [
            ("integrator", "integratore", "",   "#a3a3a3"),
            ("rt_mode",    "modo",        "",   "#a3a3a3"),
            ("rt_factor",  "RT factor",   "×",  "#22c55e"),
            ("lag_ms",     "lag",         "ms", "#eab308"),
            ("headroom",   "headroom",    "×",  "#22c55e"),
            ("steps_per_s","step/s",      "",   "#3b82f6"),
            ("dt_eff_us",  "dt_eff",      "μs", "#3b82f6"),
            ("iter_wall_ms","iter wall",  "ms", "#a3a3a3"),
            ("rejected_pct","step rejet.","%",  "#eab308"),
            ("err_norm_max","‖err‖∞",     "",   "#ef4444"),
            ("resyncs",    "resync",      "",   "#ef4444"),
            ("ui_fps",     "UI fps",      "Hz", "#22c55e"),
        ]
        diag_labels = {}
        for key, name, unit, color in diag_defs:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            nl = Gtk.Label(); nl.set_halign(Gtk.Align.START); nl.set_hexpand(True)
            nl.set_markup(f'<span font_family="monospace" foreground="{color}" size="small">{name}</span>')
            vl = Gtk.Label(); vl.set_halign(Gtk.Align.END)
            vl.set_markup(f'<span font_family="monospace" foreground="#f1f5f9" font_weight="bold" size="small">— <span foreground="#475569" font_weight="normal">{unit}</span></span>')
            row.append(nl); row.append(vl)
            gbox.append(row)
            diag_labels[key] = (vl, unit)

        # gbox (DIAGNOSTICA NUMERICA) costruito ma NASCOSTO per ora
        gbox.set_visible(False)
        dq_row.append(gbox)

        # === Composer SEGNALI === (due colonne: scalari | dq) + moduli a fianco
        plot_count = 4
        plot_signals = [
            ["wm"],
            ["Ce"],
            ["Ps", "Qs", "Pr"],
            ["slip"],
        ]
        plot_titles = [f"Plot {j+1}" for j in range(plot_count)]
        y_persist = {}
        cursor_state = {"x": None}
        plot_drawing_areas = []

        # Suddivisione del catalogo in due gruppi
        SCALAR_KEYS = ["wm", "Ce", "Ps", "Qs", "Pr", "Qr", "PRs", "PRr", "slip"]
        DQ_KEYS     = ["isd", "isq", "ird", "irq", "psd", "psq", "prd", "prq"]

        comp_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        comp_box.set_margin_start(8)
        comp_title = Gtk.Label(); comp_title.set_halign(Gtk.Align.START)
        comp_title.set_markup('<span font_family="monospace" font_weight="bold" foreground="#64748b" size="small">SEGNALI · valore istantaneo · plot</span>')
        comp_box.append(comp_title)

        # HBox a 3 colonne: [grid scalari] [grid dq] [moduli derivati]
        comp_inner = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)

        comp_checks = {}
        value_labels = {}

        def update_plot_titles():
            for j in range(plot_count):
                names = [SIG_BY_KEY[k]["name"] for k in plot_signals[j] if k in SIG_BY_KEY]
                plot_titles[j] = (f"Plot {j+1} · " + ", ".join(names)) if names else f"Plot {j+1}"

        def on_check_toggled(btn, key, j):
            if btn.get_active():
                if key not in plot_signals[j]:
                    plot_signals[j].append(key)
            else:
                if key in plot_signals[j]:
                    plot_signals[j].remove(key)
            y_persist.pop(j, None)
            update_plot_titles()
            for d in plot_drawing_areas:
                d.queue_draw()

        def hdr(text, halign=Gtk.Align.START):
            l = Gtk.Label()
            l.set_markup(f'<span font_family="monospace" foreground="#475569" size="x-small">{text}</span>')
            l.set_halign(halign)
            return l

        def build_signal_grid(keys):
            g = Gtk.Grid()
            g.set_column_spacing(10); g.set_row_spacing(1)
            g.attach(hdr("segnale"), 0, 0, 1, 1)
            g.attach(hdr("valore", Gtk.Align.END), 1, 0, 1, 1)
            g.attach(hdr("u.m."), 2, 0, 1, 1)
            for j in range(plot_count):
                g.attach(hdr(f"P{j+1}", Gtk.Align.CENTER), j + 3, 0, 1, 1)
            for i, key in enumerate(keys):
                if key not in SIG_BY_KEY: continue
                meta = SIG_BY_KEY[key]
                color_hex = f"#{int(meta['r']*255):02x}{int(meta['g']*255):02x}{int(meta['b']*255):02x}"
                sl = Gtk.Label()
                sl.set_markup(f'<span font_family="monospace" foreground="{color_hex}" size="small">● {meta["name"]}</span>')
                sl.set_halign(Gtk.Align.START); sl.set_use_markup(True)
                g.attach(sl, 0, i + 1, 1, 1)

                vl = Gtk.Label(); vl.set_halign(Gtk.Align.END)
                vl.set_markup('<span font_family="monospace" foreground="#f1f5f9" font_weight="bold" size="small">—</span>')
                g.attach(vl, 1, i + 1, 1, 1)
                value_labels[key] = vl

                ul = Gtk.Label(); ul.set_halign(Gtk.Align.START)
                ul.set_markup(f'<span font_family="monospace" foreground="#475569" size="x-small">{meta["unit"]}</span>')
                g.attach(ul, 2, i + 1, 1, 1)

                for j in range(plot_count):
                    cb = Gtk.CheckButton()
                    cb.set_active(key in plot_signals[j])
                    cb.set_halign(Gtk.Align.CENTER)
                    cb.connect("toggled", on_check_toggled, key, j)
                    g.attach(cb, j + 3, i + 1, 1, 1)
                    comp_checks[(key, j)] = cb
            return g

        comp_inner.append(build_signal_grid(SCALAR_KEYS))
        comp_inner.append(build_signal_grid(DQ_KEYS))

        # Terza colonna: moduli e derivati (a fianco, non sotto)
        mod_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        mod_col.set_margin_start(4)
        mod_title = Gtk.Label(); mod_title.set_halign(Gtk.Align.START)
        mod_title.set_markup('<span font_family="monospace" font_weight="bold" foreground="#64748b" size="x-small">moduli e derivati</span>')
        mod_col.append(mod_title)

        mod_grid = Gtk.Grid()
        mod_grid.set_column_spacing(10); mod_grid.set_row_spacing(1)
        mod_defs = [
            ("ism",   "|i_s|",   "A",   "#3b82f6"),
            ("irm",   "|i_r|",   "A",   "#ef4444"),
            ("phis",  "|φ_s|",   "mWb", "#22c55e"),
            ("phir",  "|φ_r|",   "mWb", "#ec4899"),
            ("fslip", "f_slip",  "Hz",  "#eab308"),
        ]
        derived_labels = {}
        for i, (key, name, unit, color) in enumerate(mod_defs):
            nl = Gtk.Label(); nl.set_halign(Gtk.Align.START)
            nl.set_markup(f'<span font_family="monospace" foreground="{color}" size="small">{name}</span>')
            mod_grid.attach(nl, 0, i, 1, 1)
            vl = Gtk.Label(); vl.set_halign(Gtk.Align.END)
            vl.set_markup('<span font_family="monospace" foreground="#f1f5f9" font_weight="bold" size="small">—</span>')
            mod_grid.attach(vl, 1, i, 1, 1)
            ul = Gtk.Label(); ul.set_halign(Gtk.Align.START)
            ul.set_markup(f'<span font_family="monospace" foreground="#475569" size="x-small">{unit}</span>')
            mod_grid.attach(ul, 2, i, 1, 1)
            derived_labels[key] = vl
        mod_col.append(mod_grid)
        comp_inner.append(mod_col)

        update_plot_titles()
        comp_box.append(comp_inner)
        dq_row.append(comp_box)

        vbox.append(dq_row)

        # ---- Time window slider ----
        tw_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        tw_lbl = Gtk.Label()
        tw_lbl.set_markup('<span font_family="monospace" foreground="#94a3b8" size="small">Finestra:</span>')
        tw_val = Gtk.Label()
        tw_val.set_markup('<span font_family="monospace" foreground="#e2e8f0" size="small">10 s</span>')

        def set_twin(v):
            twin[0] = v
            tw_val.set_markup(f'<span font_family="monospace" foreground="#e2e8f0" size="small">{v:.0f} s</span>')

        tw_sl = make_slider("", 10, 1, 60, 1, set_twin); tw_sl.set_hexpand(True)
        tw_row.append(tw_lbl); tw_row.append(tw_sl); tw_row.append(tw_val)
        vbox.append(tw_row)

        # ---- Time-series plots (alti, asse Y isteretico, cursore) ----
        # plot_signals/y_persist/cursor_state/plot_drawing_areas già definiti
        # nel composer in dq_row.
        def make_drawer(j):
            return lambda a, cr, w, h, *_: draw_tplot(
                a, cr, w, h, render_state["snap"],
                plot_signals[j], twin[0], plot_titles[j],
                j, y_persist, cursor_state)

        def make_motion_handler(da):
            def on_motion(_ctrl, x, _y):
                cursor_state["x"] = float(x)
                for d in plot_drawing_areas:
                    d.queue_draw()
            return on_motion

        def on_leave(_ctrl):
            cursor_state["x"] = None
            for d in plot_drawing_areas:
                d.queue_draw()

        for j in range(plot_count):
            da = Gtk.DrawingArea()
            da.set_size_request(-1, 250)
            da.set_content_height(250); da.set_hexpand(True)
            da.set_draw_func(make_drawer(j))

            mc = Gtk.EventControllerMotion()
            mc.connect("motion", make_motion_handler(da))
            mc.connect("leave", on_leave)
            da.add_controller(mc)

            drawing_areas.append(da)
            plot_drawing_areas.append(da)
            vbox.append(da)

        # Reset Y persistente quando si fa RESET globale
        def on_reset_y(*_):
            y_persist.clear()
            cursor_state["x"] = None
            for d in plot_drawing_areas:
                d.queue_draw()
        rst_btn.connect("clicked", on_reset_y)

        # ---- Footer ----
        ft = Gtk.Label()
        ft.set_markup('<span font_family="monospace" foreground="#334155" size="x-small">'
                      'A_n=1MVA · V_n=690V · n_p=3 · R_s=4.8mΩ · R_r=2.4mΩ · '
                      'L_s=6.8mH · L_r=7.1mH · M=6.8mH · J=50kg·m² · b=3.3 · '
                      'JIT Numba · sim thread @ background</span>')
        vbox.append(ft)

        scroll = Gtk.ScrolledWindow()
        scroll.set_child(vbox)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        win.set_child(scroll)
        win.present()

        # ---- GUI tick @ 60 Hz ----
        INT_NAMES = {INT_EULER: "Eulero", INT_RK4: "RK4", INT_RK45: "RK45"}

        def fmt(v):
            if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
            if abs(v) >= 1e3: return f"{v/1e3:.2f}k"
            if abs(v) >= 1: return f"{v:.2f}"
            return f"{v:.3f}"

        # Counter per throttling delle label (le scritte non hanno bisogno di girare a 120 Hz)
        gui_counter = {"n": 0}

        def gui_tick():
            gui_counter["n"] += 1
            update_labels = (gui_counter["n"] % 6 == 0)  # ~20 Hz @ 120 Hz

            snap, st, stats, ctrl, prm = engine.snapshot_for_render()
            render_state["snap"] = snap
            render_state["stats"] = stats
            render_state["ctrl"] = ctrl

            # Plot redraw sempre (vsync)
            for da in drawing_areas:
                da.queue_draw()

            if not update_labels:
                return True

            # Estrai parametri runtime
            Rs_p = prm[0]; Rr_p = prm[1]
            Ls_p = prm[2]; Lr_p = prm[3]; M_p = prm[4]
            NP_p = prm[5]
            D_p = Ls_p * Lr_p - M_p * M_p

            # Header
            t_now = st[6]
            comp = "JIT pronto" if stats.get("compiled") else "compilazione JIT…"
            time_lbl.set_markup(
                f'<span font_family="monospace" foreground="#475569" size="small">'
                f'Park sincrono · t = {t_now:.2f} s · {comp}</span>')

            # f_slip
            ws_now = 2 * math.pi * ctrl["fs"]
            wm_now = st[4]
            fslip = (ws_now - NP_p * wm_now) / (2 * math.pi) if ws_now > 0 else 0.0
            fslip_lbl.set_markup(
                f'<span font_family="monospace" foreground="#78716c" size="small">f_slip = {fslip:.2f} Hz</span>')

            # Valori istantanei (calcolati al volo dallo state)
            psd, psq, prd, prq = st[0], st[1], st[2], st[3]
            isd = (Lr_p*psd - M_p*prd) / D_p
            isq = (Lr_p*psq - M_p*prq) / D_p
            ird = (Ls_p*prd - M_p*psd) / D_p
            irq = (Ls_p*prq - M_p*psq) / D_p
            Ce = NP_p * M_p * (isq*ird - isd*irq)
            Ps = ctrl["Vs"] * isd
            Qs = -ctrl["Vs"] * isq
            a = 2*math.pi*ctrl["fr"]*t_now - ws_now*t_now + NP_p*st[5]
            vrd = ctrl["Vr"]*math.cos(a); vrq = ctrl["Vr"]*math.sin(a)
            Pr = vrd*ird + vrq*irq
            Qr = vrq*ird - vrd*irq
            PRs = Rs_p*(isd*isd + isq*isq)
            PRr = Rr_p*(ird*ird + irq*irq)
            slip = (ws_now - NP_p*wm_now)/ws_now*100 if ws_now > 0 else 0.0

            # Valori per ogni segnale del catalogo (chiave hist → valore istantaneo)
            sig_values = {
                "wm":   wm_now / (2*math.pi),
                "Ce":   Ce / 1e3,
                "Ps":   Ps / 1e3,
                "Qs":   Qs / 1e3,
                "Pr":   Pr / 1e3,
                "Qr":   Qr / 1e3,
                "PRs":  PRs / 1e3,
                "PRr":  PRr / 1e3,
                "slip": slip,
                "isd":  isd, "isq":  isq,
                "ird":  ird, "irq":  irq,
                "psd":  psd * 1e3, "psq": psq * 1e3,
                "prd":  prd * 1e3, "prq": prq * 1e3,
            }
            for k, vl in value_labels.items():
                v = sig_values.get(k, 0.0)
                vl.set_markup(
                    f'<span font_family="monospace" foreground="#f1f5f9" font_weight="bold" size="small">'
                    f'{fmt(v)}</span>')

            # Moduli e derivati
            derived_values = {
                "ism":   math.hypot(isd, isq),
                "irm":   math.hypot(ird, irq),
                "phis":  math.hypot(psd, psq) * 1e3,
                "phir":  math.hypot(prd, prq) * 1e3,
                "fslip": fslip,
            }
            for k, vl in derived_labels.items():
                v = derived_values.get(k, 0.0)
                vl.set_markup(
                    f'<span font_family="monospace" foreground="#f1f5f9" font_weight="bold" size="small">'
                    f'{fmt(v)}</span>')

            # Diagnostica
            int_name = INT_NAMES.get(stats.get("integrator", INT_RK4), "?")
            rt_mode = "RT-lock" if stats.get("rt_lock", True) else "FREE"
            rt_factor = stats.get("rt_factor", 0.0)
            headroom = stats.get("headroom", 0.0)
            lag_ms = stats.get("lag_ms", 0.0)
            resyncs = stats.get("resyncs", 0)
            sps = stats.get("steps_per_s", 0.0)
            dt_eff = stats.get("dt_eff_us", 0.0)
            iter_ms = stats.get("iter_wall_ms", 0.0)
            rej = stats.get("rejected_pct", 0.0)
            err = stats.get("err_norm_max", 0.0)

            # Indicatore tracking del target (header)
            # rt_factor è ora il rapporto rispetto al target = sim_speed / (wall × speed_factor)
            # 1.0 = on-target a qualsiasi speed
            speed_now = ctrl.get("speed_factor", 1.0)
            if rt_factor <= 0.001 or not stats.get("running", False):
                rt_col_hdr = "#475569"; rt_text = "—"; sub_text = "in pausa"
            else:
                if rt_factor >= 0.97:
                    rt_col_hdr = "#22c55e"
                    sub_text = f"target {speed_now:.2f}× wall · {fmt(sps)} step/s · dt={dt_eff:.1f}μs"
                elif rt_factor >= 0.5:
                    rt_col_hdr = "#eab308"
                    sub_text = f"in ritardo {(1.0/rt_factor):.2f}× sul target {speed_now:.2f}× wall"
                else:
                    rt_col_hdr = "#ef4444"
                    sub_text = f"sim non regge target {speed_now:.2f}× wall (ritardo {(1.0/rt_factor):.1f}×)"
                rt_text = f"{rt_factor:.2f}×"
            rt_main.set_markup(
                f'<span font_family="monospace" font_weight="bold" foreground="{rt_col_hdr}" size="xx-large">'
                f'{rt_text}</span>')
            rt_sub.set_markup(
                f'<span font_family="monospace" foreground="{rt_col_hdr}" size="x-small">{sub_text}</span>')

            # colore RT factor (pannello diagnostica): verde se ~1, rosso se < 0.95
            rt_col = "#22c55e" if 0.95 <= rt_factor <= 1.05 else ("#eab308" if 0.8 <= rt_factor < 0.95 or 1.05 < rt_factor <= 1.5 else "#ef4444")

            lag_col = "#22c55e" if lag_ms < 5 else ("#eab308" if lag_ms < 50 else "#ef4444")
            head_col = "#22c55e" if headroom > 2 else ("#eab308" if headroom > 1 else "#ef4444")
            resync_col = "#475569" if resyncs == 0 else "#ef4444"

            dvals = {
                "integrator": (int_name, ""),
                "rt_mode":    (rt_mode, ""),
                "rt_factor":  (f"{rt_factor:.3f}", "×", rt_col),
                "lag_ms":     (f"{lag_ms:.1f}", "ms", lag_col),
                "headroom":   (f"{headroom:.1f}", "×", head_col),
                "steps_per_s": (fmt(sps), "/s"),
                "dt_eff_us":   (f"{dt_eff:.1f}", "μs"),
                "iter_wall_ms":(f"{iter_ms:.2f}", "ms"),
                "rejected_pct":(f"{rej:.1f}", "%"),
                "err_norm_max":(f"{err:.2e}", ""),
                "resyncs":     (f"{resyncs}", "", resync_col),
                "ui_fps":      (f"{stats.get('ui_fps', 0.0):.0f}", "Hz"),
            }
            for key, payload in dvals.items():
                vl, unit = diag_labels[key]
                if len(payload) == 3:
                    text, u, col = payload
                else:
                    text, u = payload; col = "#f1f5f9"
                vl.set_markup(
                    f'<span font_family="monospace" foreground="{col}" font_weight="bold" size="small">'
                    f'{text} <span foreground="#475569" font_weight="normal">{u}</span></span>')

            return True  # keep ticking

        # Aggancio al frame clock GTK (vsync del compositor/monitor) — 120 Hz
        # se il display è 120 Hz, 60 Hz se è 60 Hz, ecc. Niente cap arbitrario.
        ui_fps = {"last": 0.0, "ewma": 0.0}
        def on_frame(widget, frame_clock):
            now = time.perf_counter()
            if ui_fps["last"] > 0:
                ft = now - ui_fps["last"]
                if ft > 0:
                    fps_inst = 1.0 / ft
                    ui_fps["ewma"] = 0.9 * ui_fps["ewma"] + 0.1 * fps_inst
            ui_fps["last"] = now
            engine.stats["ui_fps"] = ui_fps["ewma"]
            gui_tick()
            return GLib.SOURCE_CONTINUE
        timer_id[0] = win.add_tick_callback(on_frame)

        def on_close(_w):
            engine.shutdown()
            return False

        win.connect("close-request", on_close)

    app.connect("activate", on_activate)
    app.run(None)


if __name__ == "__main__":
    run()
