#!/usr/bin/env python3
"""DFIG — Cross-platform UI on PySide6 + QOpenGLWidget + QPainter.

The simulation engine lives in `dfig_engine` (numba JIT, background thread,
ring-buffer snapshots). This module is the GUI layer only: every plot is a
QOpenGLWidget rendered via QPainter, with vsync from
QSurfaceFormat.setSwapInterval(1). The master plot drives the per-frame
tick (snapshot + label refresh + update() on the other plots), so the loop
auto-paces to the monitor refresh rate.
"""

import math
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

from dfig_engine import (
    PARAMS_DEF,
    HIST_FIELDS, SIG_BY_KEY,
    INT_EULER, INT_RK4, INT_RK45,
    SimEngine,
    MACHINE_PRESETS, preset_to_si,
)


# Typical pu bands for sanity check on machine parameters (warn outside).
PU_BANDS = {
    "Rs":  (0.003, 0.015),
    "Rr":  (0.003, 0.015),
    "Lls": (0.05,  0.25),
    "Llr": (0.05,  0.25),
    "Lm":  (1.5,   6.0),
}


# ------------------------------------------------------------
# Autopilot scenarios (VC mode only)
# ------------------------------------------------------------
# A scenario is a list of stages played sequentially. Each stage carries:
#   dur:           stage duration in seconds
#   desc:          short label shown in the UI during playback
#   wm_factor:     (optional) setpoint = factor × ω_sync (sync = f_s/n_p)
#   throttle:      (optional) load throttle ∈ [0, 1] applied to the C_load
#                  fondoscala in generator mode (negative shaft torque)
#   tan_phi:       (optional) hold value for tan(φ_s*)
#   tan_phi_ramp:  (optional) tuple (a, b) — linear ramp from a→b over `dur`
# The "ensure" dict applies once at scenario start (e.g. force generator mode).
AUTOPILOT_SCENARIOS = [
    {
        "name": "step velocità · ascending  (sub → super)",
        "stages": [
            {"dur": 5.0, "wm_factor": 0.85, "desc": "0.85 · sync (sub)"},
            {"dur": 5.0, "wm_factor": 0.95, "desc": "0.95 · sync (sub)"},
            {"dur": 5.0, "wm_factor": 1.00, "desc": "sync"},
            {"dur": 5.0, "wm_factor": 1.05, "desc": "1.05 · sync (super)"},
            {"dur": 5.0, "wm_factor": 1.10, "desc": "1.10 · sync (super)"},
            {"dur": 5.0, "wm_factor": 1.15, "desc": "1.15 · sync (super)"},
        ],
    },
    {
        "name": "step velocità · descending  (super → sub)",
        "stages": [
            {"dur": 5.0, "wm_factor": 1.15, "desc": "1.15 · sync (super)"},
            {"dur": 5.0, "wm_factor": 1.10, "desc": "1.10 · sync (super)"},
            {"dur": 5.0, "wm_factor": 1.05, "desc": "1.05 · sync (super)"},
            {"dur": 5.0, "wm_factor": 1.00, "desc": "sync"},
            {"dur": 5.0, "wm_factor": 0.95, "desc": "0.95 · sync (sub)"},
            {"dur": 5.0, "wm_factor": 0.85, "desc": "0.85 · sync (sub)"},
        ],
    },
    {
        "name": "step velocità · cross-sync  (gioco attorno al sync)",
        "stages": [
            {"dur": 4.0, "wm_factor": 1.00, "desc": "sync"},
            {"dur": 4.0, "wm_factor": 0.90, "desc": "sub 0.90"},
            {"dur": 4.0, "wm_factor": 1.00, "desc": "sync"},
            {"dur": 4.0, "wm_factor": 1.10, "desc": "super 1.10"},
            {"dur": 4.0, "wm_factor": 1.00, "desc": "sync"},
            {"dur": 4.0, "wm_factor": 0.85, "desc": "sub 0.85"},
            {"dur": 4.0, "wm_factor": 1.15, "desc": "super 1.15"},
            {"dur": 4.0, "wm_factor": 1.00, "desc": "sync"},
        ],
    },
    {
        "name": "step carico · generatore  (0 → 100% → 0)",
        "ensure": {"motor_mode": False},
        "stages": [
            {"dur": 3.0, "throttle": 0.0,  "desc": "no load"},
            {"dur": 5.0, "throttle": 0.25, "desc": "25% gen"},
            {"dur": 5.0, "throttle": 0.50, "desc": "50% gen"},
            {"dur": 5.0, "throttle": 0.75, "desc": "75% gen"},
            {"dur": 5.0, "throttle": 1.00, "desc": "100% gen"},
            {"dur": 7.0, "throttle": 0.0,  "desc": "back to no load"},
        ],
    },
    {
        "name": "sweep tan(φ_s*)  (capacitivo ↔ induttivo)",
        "stages": [
            {"dur":  5.0, "tan_phi": 0.0,  "desc": "PF unitario (start)"},
            {"dur": 15.0, "tan_phi_ramp": (0.0,  0.5), "desc": "rampa → induttivo"},
            {"dur":  5.0, "tan_phi": 0.5,  "desc": "induttivo (hold)"},
            {"dur": 15.0, "tan_phi_ramp": (0.5, -0.5), "desc": "rampa → capacitivo"},
            {"dur":  5.0, "tan_phi": -0.5, "desc": "capacitivo (hold)"},
            {"dur": 15.0, "tan_phi_ramp": (-0.5, 0.0), "desc": "ritorno a PF unitario"},
        ],
    },
]
from dfig_gamepad import GamepadController

# ============================================================
# Markup helpers — Pango → Qt rich-text translation
# ============================================================

# Approximate Pango "size" attribute → point sizes (Qt rich text)
_PANGO_SIZE = {
    "xx-small": 6, "x-small": 7, "small": 8, "medium": 10,
    "large": 12, "x-large": 14, "xx-large": 18,
}


def htm(text, color="#f1f5f9", size="small", bold=False, family="monospace"):
    """Build a single <span> with monospace styling (Pango-style equivalent)."""
    sz = _PANGO_SIZE.get(size, 9)
    w = "bold" if bold else "normal"
    return (f'<span style="color:{color};font-family:{family};'
            f'font-size:{sz}pt;font-weight:{w};">{text}</span>')


# ============================================================
# Drawing primitives (Cairo → QPainter port)
# ============================================================

def nice_ticks(lo, hi, n_target=6):
    """Round ticks (1·10ⁿ, 2·10ⁿ, 5·10ⁿ) between lo and hi, ~n_target ticks."""
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
    """Compact axis-label formatter, scale-aware."""
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


# Pre-built fonts (cheap to construct but pre-building keeps paint loop tight)
def _font(size, bold=False):
    f = QtGui.QFont("monospace", size)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    if bold:
        f.setWeight(QtGui.QFont.Weight.Bold)
    return f


def _color(r, g, b, a=1.0):
    return QtGui.QColor.fromRgbF(r, g, b, a)


def _rgb(hex_or_tuple):
    if isinstance(hex_or_tuple, str):
        return QtGui.QColor(hex_or_tuple)
    return _color(*hex_or_tuple)


def _set_pen(p, color, width=1.0, dash=None):
    pen = QtGui.QPen(color)
    pen.setWidthF(width)
    pen.setCosmetic(True)
    if dash is not None:
        pen.setDashPattern(dash)
    p.setPen(pen)


def draw_dq(p, w, h, snap, traces, t_win, title, axis_labels=("d", "q")):
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.fillRect(0, 0, w, h, _color(0.027, 0.047, 0.086))
    S = min(w, h)
    if snap is None or len(snap.t) < 2:
        p.setFont(_font(10, bold=True))
        _set_pen(p, _color(0.3, 0.3, 0.3))
        p.drawText(QtCore.QPointF(S / 2 - 30, S / 2), "premi RUN")
        return

    t_arr = snap.t
    tn = float(t_arr[-1]); ts = tn - t_win
    i0 = int(np.searchsorted(t_arr, ts, side="left"))
    n_visible = len(t_arr) - i0
    if n_visible < 2:
        return

    # Decimation: cap at ~S points per trajectory
    stride = max(1, n_visible // max(int(S), 1))

    # Vectorized range
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

    # Concentric circles (grid)
    _set_pen(p, _color(0.06, 0.09, 0.16), 0.5)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    for r in (0.25, 0.5, 0.75, 1.0):
        rr = r * (S / 2 - 14)
        p.drawEllipse(QtCore.QPointF(S / 2, S / 2), rr, rr)
    _set_pen(p, _color(0.14, 0.18, 0.26), 1.0)
    p.drawLine(QtCore.QLineF(S / 2, 6, S / 2, S - 6))
    p.drawLine(QtCore.QLineF(6, S / 2, S - 6, S / 2))

    # Numbered ticks on d/q axes
    p.setFont(_font(10))
    fm = QtGui.QFontMetricsF(p.font())
    tick_ticks = nice_ticks(-am, am, 5)
    for tv in tick_ticks:
        if abs(tv) < am * 0.05: continue   # skip the central "0"
        if abs(tv) > am: continue
        # d axis (horizontal)
        x = S / 2 + tv / am * (S / 2 - 14)
        _set_pen(p, _color(0.20, 0.26, 0.36), 1.0)
        p.drawLine(QtCore.QLineF(x, S / 2 - 3, x, S / 2 + 3))
        _set_pen(p, _color(0.45, 0.52, 0.62))
        label = fmt_axis(tv, tick_ticks[1] - tick_ticks[0] if len(tick_ticks) > 1 else 1)
        lw = fm.horizontalAdvance(label)
        p.drawText(QtCore.QPointF(x - lw / 2, S / 2 + 14), label)
        # q axis (vertical)
        y = S / 2 - tv / am * (S / 2 - 14)
        _set_pen(p, _color(0.20, 0.26, 0.36), 1.0)
        p.drawLine(QtCore.QLineF(S / 2 - 3, y, S / 2 + 3, y))
        _set_pen(p, _color(0.45, 0.52, 0.62))
        p.drawText(QtCore.QPointF(S / 2 + 5, y + 3), label)

    # d / q axis labels
    p.setFont(_font(12, bold=True))
    _set_pen(p, _color(0.55, 0.62, 0.72))
    p.drawText(QtCore.QPointF(S - 14, S / 2 - 6), axis_labels[0])
    p.drawText(QtCore.QPointF(S / 2 + 6, 16), axis_labels[1])

    for fd, fq, r, g, b, label in traces:
        dd_full = getattr(snap, fd); qq_full = getattr(snap, fq)
        dd = dd_full[i0::stride]; qq = qq_full[i0::stride]
        n_dec = len(dd)
        if n_dec < 2: continue
        # Trail with increasing alpha
        for i in range(1, n_dec):
            alpha = 0.05 + 0.95 * i / n_dec
            _set_pen(p, _color(r, g, b, alpha), 1.3)
            p.drawLine(QtCore.QLineF(mx(float(dd[i-1])), my(float(qq[i-1])),
                                     mx(float(dd[i])), my(float(qq[i]))))
        ld, lq = float(dd_full[-1]), float(qq_full[-1])
        p.setBrush(QtGui.QBrush(_color(r, g, b)))
        _set_pen(p, _color(r, g, b))
        p.drawEllipse(QtCore.QPointF(mx(ld), my(lq)), 5, 5)
        _set_pen(p, _color(r, g, b, 0.5), 1.5)
        p.drawLine(QtCore.QLineF(S / 2, S / 2, mx(ld), my(lq)))
        ang = math.atan2(-lq, ld); ax, ay, al = mx(ld), my(lq), 10
        for da in (-0.3, 0.3):
            p.drawLine(QtCore.QLineF(ax, ay,
                                     ax - al * math.cos(ang + da),
                                     ay + al * math.sin(ang + da)))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    p.setFont(_font(11))
    ly = S - 10
    for *_, r, g, b, label in reversed(traces):
        _set_pen(p, _color(r, g, b))
        p.drawText(QtCore.QPointF(8, ly), f"● {label}"); ly -= 16

    p.setFont(_font(13, bold=True))
    _set_pen(p, _color(0.62, 0.70, 0.80))
    fm = QtGui.QFontMetricsF(p.font())
    tw = fm.horizontalAdvance(title)
    p.drawText(QtCore.QPointF((S - tw) / 2, 16), title)


def draw_tplot(p, w, h, snap, signal_keys, t_win, title,
               plot_idx, y_persist, cursor_state, stacked=False):
    """Time-series plot with hysteretic Y axis, ticks, cross-plot cursor,
    and (optional) stacked/diverging area mode for power balances."""
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    # Plot area margins
    M_L, M_R, M_T, M_B = 60, 10, 28, 22
    plot_w = max(1, w - M_L - M_R)
    plot_h = max(1, h - M_T - M_B)

    # Background
    p.fillRect(0, 0, w, h, _color(0.027, 0.047, 0.086))

    # Title (top-left)
    p.setFont(_font(13, bold=True))
    _set_pen(p, _color(0.62, 0.70, 0.80))
    p.drawText(QtCore.QPointF(8, 18), title)

    # Empty state
    if not signal_keys:
        p.setFont(_font(13))
        _set_pen(p, _color(0.30, 0.35, 0.42))
        msg = "(plot vuoto — usa il composer SEGNALI)"
        fm = QtGui.QFontMetricsF(p.font())
        mw = fm.horizontalAdvance(msg)
        p.drawText(QtCore.QPointF((w - mw) / 2, h / 2), msg)
        return
    if snap is None or len(snap.t) < 2:
        p.setFont(_font(13))
        _set_pen(p, _color(0.30, 0.35, 0.42))
        fm = QtGui.QFontMetricsF(p.font())
        msg = "premi RUN"
        mw = fm.horizontalAdvance(msg)
        p.drawText(QtCore.QPointF((w - mw) / 2, h / 2), msg)
        return

    t_arr = snap.t
    tn = float(t_arr[-1]); ts = tn - t_win
    i0 = int(np.searchsorted(t_arr, ts, side="left"))
    n_visible = len(t_arr) - i0
    if n_visible < 2:
        return

    # Decimation: max ~plot_w points per stroke
    stride = max(1, n_visible // plot_w)

    # Visible-window data range (vectorized)
    if stacked:
        n_t = max(1, (len(t_arr) - i0 + stride - 1) // stride)
        cum_pos = np.zeros(n_t); cum_neg = np.zeros(n_t)
        for key in signal_keys:
            if key not in SIG_BY_KEY: continue
            a = getattr(snap, key, None)
            if a is None: continue
            a_sub = a[i0::stride]
            if a_sub.size != n_t: continue
            cum_pos = np.where(a_sub >= 0, cum_pos + a_sub, cum_pos)
            cum_neg = np.where(a_sub <  0, cum_neg + a_sub, cum_neg)
        mn_d = float(cum_neg.min()) if cum_neg.size else 0.0
        mx_d = float(cum_pos.max()) if cum_pos.size else 0.0
        if mn_d == 0.0 and mx_d == 0.0:
            mn_d = -1.0; mx_d = 1.0
    else:
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

    # SOFT ASYMMETRIC RESCALE: instant expansion (snap-out when new data
    # exceeds current bounds), smooth contraction (EWMA toward a tighter
    # range with time constant tau). Frame-rate-independent: relies on
    # wall-clock dt between paints, so the feel is consistent across
    # 60/120/144 Hz monitors and software rendering.
    _now = time.perf_counter()
    tau_contract = 0.5  # seconds for the bound to relax ~63% toward a tighter range
    persist = y_persist.get(plot_idx)
    if persist is None or len(persist) < 3:
        # First paint, or migration from the old (mn, mx) tuple.
        mn_p, mx_p = mn_d, mx_d
    else:
        mn_p, mx_p, last_t = persist
        dt = _now - last_t
        alpha = 1.0 - math.exp(-dt / tau_contract) if dt > 0 else 0.0
        if mn_d < mn_p:
            mn_p = mn_d                          # expand: snap to new lower bound
        else:
            mn_p = mn_p + alpha * (mn_d - mn_p)  # contract: ease toward tighter
        if mx_d > mx_p:
            mx_p = mx_d                          # expand: snap to new upper bound
        else:
            mx_p = mx_p + alpha * (mx_d - mx_p)  # contract: ease toward tighter
    y_persist[plot_idx] = (mn_p, mx_p, _now)

    span = mx_p - mn_p
    pad = span * 0.06 if span > 0 else 1.0
    mn = mn_p - pad; mx = mx_p + pad

    def xof(t): return M_L + (t - ts) / t_win * plot_w
    def yof(v): return M_T + plot_h - (v - mn) / (mx - mn) * plot_h

    # === GRID + TICKS ===
    y_ticks = nice_ticks(mn, mx, 5)
    x_ticks = nice_ticks(ts, tn, 6)
    y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0

    _set_pen(p, _color(0.10, 0.14, 0.20), 0.5)
    for v in y_ticks:
        y = yof(v)
        if M_T <= y <= M_T + plot_h:
            p.drawLine(QtCore.QLineF(M_L, y, M_L + plot_w, y))
    for tt in x_ticks:
        x = xof(tt)
        if M_L <= x <= M_L + plot_w:
            p.drawLine(QtCore.QLineF(x, M_T, x, M_T + plot_h))

    # Zero line (more visible)
    if mn < 0 < mx:
        zy = yof(0.0)
        _set_pen(p, _color(0.20, 0.26, 0.36), 1.0, dash=[3, 3])
        p.drawLine(QtCore.QLineF(M_L, zy, M_L + plot_w, zy))

    # Plot frame
    _set_pen(p, _color(0.18, 0.22, 0.30), 1.0)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRect(QtCore.QRectF(M_L, M_T, plot_w, plot_h))

    # Traces — vectorized coordinates, decimated by stride
    t_sub = t_arr[i0::stride]
    xs_arr = M_L + (t_sub - ts) / t_win * plot_w
    sig_legend = []
    if stacked:
        n_t = len(t_sub)
        cp = np.zeros(n_t); cn = np.zeros(n_t)
        for key in signal_keys:
            if key not in SIG_BY_KEY: continue
            meta = SIG_BY_KEY[key]
            a = getattr(snap, key, None)
            if a is None: continue
            a_sub = a[i0::stride]
            if a_sub.size != n_t: continue
            new_cp = np.where(a_sub >= 0, cp + a_sub, cp)
            new_cn = np.where(a_sub <  0, cn + a_sub, cn)
            upper = np.where(a_sub >= 0, new_cp, cn)
            lower = np.where(a_sub >= 0, cp,     new_cn)
            cp, cn = new_cp, new_cn
            ys_up = M_T + plot_h - (upper - mn) / (mx - mn) * plot_h
            ys_lo = M_T + plot_h - (lower - mn) / (mx - mn) * plot_h
            path = QtGui.QPainterPath()
            path.moveTo(float(xs_arr[0]), float(ys_lo[0]))
            for k in range(n_t):
                path.lineTo(float(xs_arr[k]), float(ys_up[k]))
            for k in range(n_t - 1, -1, -1):
                path.lineTo(float(xs_arr[k]), float(ys_lo[k]))
            path.closeSubpath()
            p.setBrush(QtGui.QBrush(_color(meta["r"], meta["g"], meta["b"], 0.55)))
            _set_pen(p, _color(meta["r"], meta["g"], meta["b"]), 0.8)
            p.drawPath(path)
            sig_legend.append((key, meta))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    else:
        for key in signal_keys:
            if key not in SIG_BY_KEY: continue
            meta = SIG_BY_KEY[key]
            a = getattr(snap, key, None)
            if a is None: continue
            a_sub = a[i0::stride]
            if a_sub.size < 2: continue
            ys_arr = M_T + plot_h - (a_sub - mn) / (mx - mn) * plot_h
            path = QtGui.QPainterPath()
            path.moveTo(float(xs_arr[0]), float(ys_arr[0]))
            for k in range(1, len(xs_arr)):
                path.lineTo(float(xs_arr[k]), float(ys_arr[k]))
            _set_pen(p, _color(meta["r"], meta["g"], meta["b"]), 1.8)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            sig_legend.append((key, meta))

    # === Y-axis tick labels (left) ===
    p.setFont(_font(11))
    fm = QtGui.QFontMetricsF(p.font())
    _set_pen(p, _color(0.55, 0.62, 0.72))
    for v in y_ticks:
        y = yof(v)
        if M_T - 4 <= y <= M_T + plot_h + 4:
            label = fmt_axis(v, y_step)
            lw = fm.horizontalAdvance(label)
            p.drawText(QtCore.QPointF(M_L - lw - 4, y + 4), label)
            _set_pen(p, _color(0.55, 0.62, 0.72), 1.0)
            p.drawLine(QtCore.QLineF(M_L - 3, y, M_L, y))

    # === X-axis tick labels (bottom) ===
    for tt in x_ticks:
        x = xof(tt)
        if M_L <= x <= M_L + plot_w:
            label = f"{tt:.1f}s"
            lw = fm.horizontalAdvance(label)
            p.drawText(QtCore.QPointF(x - lw / 2, h - 6), label)
            p.drawLine(QtCore.QLineF(x, M_T + plot_h, x, M_T + plot_h + 3))

    # Legend (top-right)
    p.setFont(_font(12))
    fm = QtGui.QFontMetricsF(p.font())
    lx = w - 8
    for key, meta in reversed(sig_legend):
        _set_pen(p, _color(meta["r"], meta["g"], meta["b"]))
        label = f"{meta['name']} [{meta['unit']}]"
        lw = fm.horizontalAdvance(label)
        lx -= lw + 22
        p.drawText(QtCore.QPointF(lx + 16, 18), label)
        _set_pen(p, _color(meta["r"], meta["g"], meta["b"]), 2.0)
        p.drawLine(QtCore.QLineF(lx, 14, lx + 14, 14))

    # === Cursor + cross-plot tooltip ===
    cx = cursor_state.get("x")
    if cx is not None and M_L <= cx <= M_L + plot_w:
        x_cur = cx
        ct = ts + (cx - M_L) / plot_w * t_win
        _set_pen(p, _color(0.92, 0.92, 0.95, 0.5), 1.0, dash=[3, 2])
        p.drawLine(QtCore.QLineF(x_cur, M_T, x_cur, M_T + plot_h))

        ts_arr = snap.t
        idx = int(np.searchsorted(ts_arr, ct, side="left"))
        if idx >= len(ts_arr): idx = len(ts_arr) - 1
        if idx > 0 and abs(ts_arr[idx - 1] - ct) < abs(ts_arr[idx] - ct):
            idx -= 1

        # Dots on traces at cursor x
        for key, meta in sig_legend:
            a = getattr(snap, key, None)
            if a is None or idx >= len(a): continue
            p.setBrush(QtGui.QBrush(_color(meta["r"], meta["g"], meta["b"])))
            _set_pen(p, _color(meta["r"], meta["g"], meta["b"]))
            p.drawEllipse(QtCore.QPointF(x_cur, yof(a[idx])), 4, 4)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        # Tooltip — fixed-width numbers (no jitter when digits change)
        lines = [(None, f"t = {ts_arr[idx]:>10.3f} s")]
        for key, meta in sig_legend:
            a = getattr(snap, key, None)
            if a is None or idx >= len(a): continue
            lines.append((meta, f"{meta['name']:<5s} = {a[idx]:>10.3f} {meta['unit']}"))

        p.setFont(_font(12))
        fm = QtGui.QFontMetricsF(p.font())
        max_w = 0
        for _, line in lines:
            lw = fm.horizontalAdvance(line)
            if lw > max_w: max_w = lw
        line_h = 16
        tip_w = max_w + 16
        tip_h = line_h * len(lines) + 10

        if x_cur > w * 0.6:
            tip_x = x_cur - tip_w - 8
        else:
            tip_x = x_cur + 8
        if tip_x < 2: tip_x = 2
        if tip_x + tip_w > w - 2: tip_x = w - tip_w - 2
        tip_y = M_T + 4
        if tip_y + tip_h > M_T + plot_h - 4:
            tip_y = M_T + plot_h - tip_h - 4

        p.fillRect(QtCore.QRectF(tip_x, tip_y, tip_w, tip_h),
                   _color(0.04, 0.06, 0.10, 0.93))
        _set_pen(p, _color(0.30, 0.35, 0.45), 0.5)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(QtCore.QRectF(tip_x, tip_y, tip_w, tip_h))

        for li, (meta, line) in enumerate(lines):
            if meta is None:
                _set_pen(p, _color(0.85, 0.88, 0.95))
            else:
                _set_pen(p, _color(meta["r"], meta["g"], meta["b"]))
            p.drawText(QtCore.QPointF(tip_x + 7, tip_y + line_h * (li + 1)), line)


def draw_saturation(p, w, h, state, params, title="SATURAZIONE"):
    """Saturation curve f(|ψ_s|) = L_m,eff / L_m0 plus the current operating
    point. The kernel formula is f = 1/(1+(σ−1)²) for σ = |ψ_s|/ψ_s,sat ≥ 1,
    f = 1 below the knee. In OFF mode the operating marker stays on Y=1
    (model is linear) but the curve is still drawn — visually shows where
    you would be if saturation were enabled."""
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.fillRect(0, 0, w, h, _color(0.027, 0.047, 0.086))

    # Plot area
    M_L, M_R, M_T, M_B = 56, 12, 28, 22
    plot_w = max(1, w - M_L - M_R)
    plot_h = max(1, h - M_T - M_B)

    # Title
    p.setFont(_font(13, bold=True))
    _set_pen(p, _color(0.62, 0.70, 0.80))
    p.drawText(QtCore.QPointF(8, 18), title)

    # Empty / unset state guard
    if state is None or params is None or len(params) < 10:
        p.setFont(_font(11))
        _set_pen(p, _color(0.30, 0.35, 0.42))
        msg = "premi RUN"
        fm = QtGui.QFontMetricsF(p.font())
        mw = fm.horizontalAdvance(msg)
        p.drawText(QtCore.QPointF((w - mw) / 2, h / 2), msg)
        return

    psi_s_sat = float(params[9])
    sat_en = float(params[8]) > 0.5
    if psi_s_sat <= 0:
        psi_s_sat = 1.0  # avoid div-by-zero, fallback

    # X axis: 0 to 2.5 × ψ_s,sat
    x_max = 2.5 * psi_s_sat

    psi_sd = float(state[0]); psi_sq = float(state[1])
    psi_s_mag = math.hypot(psi_sd, psi_sq)
    sigma = psi_s_mag / psi_s_sat
    in_saturation = (sigma > 1.0)
    if sat_en and in_saturation:
        excess = sigma - 1.0
        op_y = 1.0 / (1.0 + excess * excess)
    else:
        # OFF mode → linear, point always on Y=1.
        # ON mode + below threshold → still linear, also Y=1.
        op_y = 1.0
    op_x = psi_s_mag

    def xof(psi):
        return M_L + (psi / x_max) * plot_w
    def yof(f):
        return M_T + plot_h - f * plot_h

    # Grid (horizontal at Y = 0, .25, .5, .75, 1)
    _set_pen(p, _color(0.10, 0.14, 0.20), 0.5)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yof(f)
        p.drawLine(QtCore.QLineF(M_L, y, M_L + plot_w, y))
    # Vertical grid at X = 0.5, 1.0, 1.5, 2.0 × ψ_s,sat
    for x_ratio in (0.5, 1.0, 1.5, 2.0):
        x = xof(x_ratio * psi_s_sat)
        p.drawLine(QtCore.QLineF(x, M_T, x, M_T + plot_h))

    # Frame
    _set_pen(p, _color(0.18, 0.22, 0.30), 1.0)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRect(QtCore.QRectF(M_L, M_T, plot_w, plot_h))

    # Threshold line at ψ_s,sat (vertical, dashed orange)
    x_thr = xof(psi_s_sat)
    _set_pen(p, _color(0.97, 0.45, 0.09, 0.6), 1.2, dash=[5, 4])
    p.drawLine(QtCore.QLineF(x_thr, M_T, x_thr, M_T + plot_h))

    # Saturation curve — orange, smooth
    _set_pen(p, _color(0.97, 0.45, 0.09), 1.8)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    path = QtGui.QPainterPath()
    N = 120
    for i in range(N + 1):
        psi = (i / N) * x_max
        sigma = psi / psi_s_sat
        excess = sigma - 1.0
        f = 1.0 / (1.0 + excess * excess) if excess > 0 else 1.0
        xs = xof(psi); ys = yof(f)
        if i == 0:
            path.moveTo(xs, ys)
        else:
            path.lineTo(xs, ys)
    p.drawPath(path)

    # Operating point — fill always white, border colored by region:
    #   below threshold → gray (linear region, "safe")
    #   above threshold → orange (saturating territory)
    border = _color(0.97, 0.45, 0.09) if in_saturation else _color(0.55, 0.62, 0.72)
    p.setBrush(QtGui.QBrush(_color(0.95, 0.95, 1.0)))
    _set_pen(p, border, 2.0)
    op_x_clamped = min(op_x, x_max)
    p.drawEllipse(QtCore.QPointF(xof(op_x_clamped), yof(op_y)), 5, 5)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    # Y-axis tick labels (left of plot area, only 0/0.5/1)
    p.setFont(_font(11))
    fm = QtGui.QFontMetricsF(p.font())
    _set_pen(p, _color(0.55, 0.62, 0.72))
    for f_val in (0.0, 0.5, 1.0):
        y = yof(f_val)
        label = f"{f_val:.1f}"
        lw = fm.horizontalAdvance(label)
        p.drawText(QtCore.QPointF(M_L - lw - 4, y + 4), label)
    # Y axis name — small, rotated under the title (left of plot)
    p.setFont(_font(10))
    _set_pen(p, _color(0.45, 0.52, 0.62))
    p.drawText(QtCore.QPointF(8, M_T + 12), "L_m/L_m0")

    # X-axis tick labels (in Wb). Skip the 0.0 label (overlaps the y-axis
    # tick at "0.0"); skip ratios > x_max range.
    p.setFont(_font(11))
    fm = QtGui.QFontMetricsF(p.font())
    _set_pen(p, _color(0.55, 0.62, 0.72))
    for x_ratio in (0.5, 1.0, 1.5, 2.0):
        x_val = x_ratio * psi_s_sat
        if x_val > x_max:
            continue
        x = xof(x_val)
        if abs(x_ratio - 1.0) < 1e-6:
            label = f"ψ_sat={psi_s_sat:.2f}"
        else:
            label = f"{x_val:.2f}"
        lw = fm.horizontalAdvance(label)
        p.drawText(QtCore.QPointF(x - lw / 2, h - 6), label)
    # X axis name — bottom right
    p.setFont(_font(10))
    _set_pen(p, _color(0.45, 0.52, 0.62))
    p.drawText(QtCore.QPointF(M_L + plot_w - 56, M_T + plot_h + 14), "|ψ_s| [Wb]")

    # Top-right legend — compact, single line, with region indicator
    p.setFont(_font(11))
    fm = QtGui.QFontMetricsF(p.font())
    mode_lbl = "ON" if sat_en else "OFF"
    region_lbl = "saturazione" if in_saturation else "lineare"
    line = (f"{mode_lbl}  ·  σ={sigma:.2f} ({region_lbl})  ·  "
            f"L_m/L_m0={op_y:.3f}")
    lw = fm.horizontalAdvance(line)
    legend_color = (_color(0.97, 0.45, 0.09) if in_saturation
                    else _color(0.85, 0.88, 0.95))
    _set_pen(p, legend_color)
    p.drawText(QtCore.QPointF(w - lw - 8, 18), line)


# ============================================================
# Plot widgets — QOpenGLWidget + QPainter
# ============================================================

class PlotGL(QtOpenGLWidgets.QOpenGLWidget):
    """Generic GL-accelerated 2D plot. The actual drawing is delegated to
    `draw_func(painter, w, h)`, set externally."""

    def __init__(self, draw_func=None, parent=None):
        super().__init__(parent)
        self._draw_func = draw_func
        # Forward mouse moves even without a button held (for cross-plot cursor)
        self.setMouseTracking(True)

    def setDrawFunc(self, fn):
        self._draw_func = fn

    def paintGL(self):
        painter = QtGui.QPainter(self)
        try:
            if self._draw_func is not None:
                self._draw_func(painter, self.width(), self.height())
        finally:
            painter.end()


class MasterPlotGL(PlotGL):
    """Same as PlotGL but emits a per-frame signal at the start of paintGL
    and schedules its own next update — driving the vsync loop."""

    frame_tick = QtCore.Signal()

    def paintGL(self):
        # Fire BEFORE drawing so the main window can refresh the snapshot
        # and per-label state for the current frame.
        self.frame_tick.emit()
        super().paintGL()
        # Schedule next frame; SwapBuffers blocks at vsync (SwapInterval=1),
        # so the loop auto-paces to monitor refresh rate.
        QtCore.QTimer.singleShot(0, self.update)


# ============================================================
# Composite UI helpers
# ============================================================

class Slider(QtWidgets.QWidget):
    """Label + horizontal slider + live value readout. Takes float [lo, hi]
    with a step size; emits via callback.

    Two layouts:
    - default (compact=False): label/value on one row above the slider (2 rows)
    - compact=True: [label_fixed] [slider stretch] [value_fixed] (1 row)
    """

    def __init__(self, label, val, lo, hi, step, cb,
                 label_color="#94a3b8", compact=False,
                 snap_value=None, snap_tolerance=None):
        super().__init__()
        self._lo, self._hi, self._step = lo, hi, step
        self._cb = cb
        # Optional snap point: if the slider value falls within snap_tolerance
        # of snap_value, it's pulled exactly to snap_value. The point can be
        # updated at runtime via setSnap() (e.g. when synchronous speed
        # changes because f_s or n_p was edited).
        self._snap_value = snap_value
        self._snap_tolerance = snap_tolerance
        # Map float range onto integer slider range to keep precision
        self._n = max(1, int(round((hi - lo) / step)))

        self._sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sl.setRange(0, self._n)
        self._sl.setSingleStep(1)
        self._sl.setPageStep(max(1, self._n // 20))
        self._sl.setValue(self._to_int(val))
        self._sl.valueChanged.connect(self._on_change)

        self._nl = QtWidgets.QLabel()
        self._nl.setText(htm(label, color=label_color))
        self._nl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._vl = QtWidgets.QLabel()
        self._vl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._vl.setTextFormat(QtCore.Qt.TextFormat.RichText)

        if compact:
            h = QtWidgets.QHBoxLayout(self)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            self._nl.setFixedWidth(90)
            self._vl.setFixedWidth(60)
            h.addWidget(self._nl)
            h.addWidget(self._sl, 1)
            h.addWidget(self._vl)
        else:
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(0, 0, 0, 0); v.setSpacing(1)
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            row.addWidget(self._nl, 1)
            row.addWidget(self._vl, 0)
            v.addLayout(row)
            v.addWidget(self._sl)
        self._on_change(self._sl.value())

    def _to_int(self, v):
        return int(round((v - self._lo) / self._step))

    def _to_float(self, i):
        return self._lo + i * self._step

    def value(self):
        return self._to_float(self._sl.value())

    def setValue(self, v):
        self._sl.setValue(self._to_int(v))

    def setReadout(self, text):
        self._vl.setText(text)

    def setCallback(self, fn):
        self._cb = fn

    def setSnap(self, value, tolerance):
        """Update the snap point and tolerance at runtime. Pass None for
        either to disable snapping."""
        self._snap_value = value
        self._snap_tolerance = tolerance

    def setRange(self, lo, hi, step):
        """Resize the slider range at runtime (e.g. when machine size
        changes and the load fondoscala has to scale). Current value is
        clamped to the new [lo, hi] interval."""
        cur = self.value()
        self._lo, self._hi, self._step = lo, hi, step
        self._n = max(1, int(round((hi - lo) / step)))
        self._sl.blockSignals(True)
        self._sl.setRange(0, self._n)
        self._sl.setPageStep(max(1, self._n // 20))
        cur_clamped = max(lo, min(hi, cur))
        self._sl.setValue(self._to_int(cur_clamped))
        self._sl.blockSignals(False)
        # Force readout refresh + callback fire with the (possibly clamped) value
        self._on_change(self._sl.value())

    def _on_change(self, i):
        v = self._to_float(i)
        # Snap to a magnetic point if configured and within tolerance.
        if (self._snap_value is not None and self._snap_tolerance is not None
                and abs(v - self._snap_value) < self._snap_tolerance):
            v = self._snap_value
            target_i = self._to_int(v)
            if target_i != i:
                # blockSignals avoids re-entering _on_change from setValue
                self._sl.blockSignals(True)
                self._sl.setValue(target_i)
                self._sl.blockSignals(False)
        if abs(v) >= 1000: t = f"{v/1000:.1f}k"
        elif abs(v) >= 100: t = f"{v:.0f}"
        elif abs(v) >= 1: t = f"{v:.1f}"
        else: t = f"{v:.2f}"
        self._vl.setText(htm(t, color="#e2e8f0"))
        self._cb(v)


class ParamInput(QtWidgets.QWidget):
    """Label + numeric entry + ▼/▲ buttons. ``on_change(v_disp)`` gets
    the value in display units. With ``int_only=True`` values round to
    nearest integer and ▼/▲ apply ±1 instead of ±10%."""

    def __init__(self, label_text, initial_disp, unit, on_change,
                 int_only=False):
        super().__init__()
        self._on_change = on_change
        self._int_only = int_only
        self._initial = round(initial_disp) if int_only else initial_disp

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(2)

        lbl = QtWidgets.QLabel()
        lbl.setText(htm(label_text, color="#94a3b8"))
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setMinimumWidth(34)
        h.addWidget(lbl)

        self._entry = QtWidgets.QLineEdit(self._fmt(self._initial))
        self._entry.setMaxLength(12)
        self._entry.setMaximumWidth(80)
        self._entry.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._entry.returnPressed.connect(self._on_entry_activate)
        h.addWidget(self._entry)

        if unit:
            ul = QtWidgets.QLabel()
            ul.setText(htm(unit, color="#475569", size="x-small"))
            ul.setTextFormat(QtCore.Qt.TextFormat.RichText)
            ul.setMinimumWidth(40)
            h.addWidget(ul)

        btn_dn = QtWidgets.QPushButton("▼")
        btn_dn.setFixedSize(24, 22)
        btn_dn.clicked.connect(self._on_dn)
        btn_up = QtWidgets.QPushButton("▲")
        btn_up.setFixedSize(24, 22)
        btn_up.clicked.connect(self._on_up)
        h.addWidget(btn_dn)
        h.addWidget(btn_up)

    def _fmt(self, v):
        if self._int_only:
            return f"{int(round(v))}"
        if abs(v) >= 1000: return f"{v:.1f}"
        if abs(v) >= 100:  return f"{v:.2f}"
        if abs(v) >= 1:    return f"{v:.3f}"
        return f"{v:.4f}"

    def _apply(self, new_v):
        if self._int_only:
            new_v = round(new_v)
        # Reject negatives. b = 0 is allowed (wind-DFIG presets neglect friction).
        if new_v < 0: return
        self._entry.setText(self._fmt(new_v))
        self._on_change(new_v)

    def value(self):
        """Read the current display value as float (initial if entry text
        is malformed)."""
        try:
            return float(self._entry.text())
        except ValueError:
            return self._initial

    def setValue(self, v):
        """Programmatically push a new display value (e.g. when loading
        a preset). Fires the on_change callback like a manual edit would."""
        self._apply(v)

    def _on_entry_activate(self):
        try:
            v = float(self._entry.text())
        except ValueError:
            self._entry.setText(self._fmt(self._initial))
            return
        self._apply(v)

    def _on_dn(self):
        try: v = float(self._entry.text())
        except ValueError: return
        self._apply(v - 1 if self._int_only else v * 0.9)

    def _on_up(self):
        try: v = float(self._entry.text())
        except ValueError: return
        self._apply(v + 1 if self._int_only else v * 1.1)


def center_widget(widget, max_width):
    """Wrap a widget in an HBox with addStretch on both sides and a max-width
    clamp on the widget. Returns the container, ready for v.addWidget(...).
    Used to keep the PARAMETRI MACCHINA / SEGNALI sections from spreading
    edge-to-edge on wide displays."""
    container = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(container)
    h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
    widget.setMaximumWidth(max_width)
    h.addStretch(1)
    h.addWidget(widget)
    h.addStretch(1)
    return container


def panel(title, color, children):
    """STATORE / ROTORE / CARICO / SIMULAZIONE styled panel."""
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    v = QtWidgets.QVBoxLayout(f)
    v.setContentsMargins(6, 6, 6, 6); v.setSpacing(2)
    lbl = QtWidgets.QLabel()
    lbl.setText(htm(title, color=color, size="small", bold=True))
    lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
    v.addWidget(lbl)
    for c in children:
        if isinstance(c, QtWidgets.QLayout):
            v.addLayout(c)
        else:
            v.addWidget(c)
    return f


# ============================================================
# Main window
# ============================================================

INT_NAMES = {INT_EULER: "Eulero", INT_RK4: "RK4", INT_RK45: "RK45"}

SCALAR_KEYS = ["wm", "Ce", "Ps", "Qs", "Pr", "Qr", "PRs", "PRr", "slip",
               "Pmech", "Pem", "Pfric", "Ploss", "dKEdt"]
DQ_KEYS     = ["isd", "isq", "ird", "irq", "psd", "psq", "prd", "prq"]
# Magnitudes / derived signals — full citizens of the composer (value display
# + per-plot checkboxes). They aren't stored in the history buffer; the UI
# attaches them as numpy views on snap.* in _gui_tick before painting.
MOD_KEYS    = ["ism", "irm", "phis", "phir", "fslip"]


def fmt_value(v):
    """Numeric formatter for live signal value labels."""
    av = abs(v)
    if av >= 10000:  return f"{v:.0f}"
    if av >= 1000:   return f"{v:.1f}"
    if av >= 100:    return f"{v:.2f}"
    if av >= 10:     return f"{v:.2f}"
    if av >= 1:      return f"{v:.3f}"
    if av >= 0.01:   return f"{v:.3f}"
    return f"{v:.4f}"


class DfigWindow(QtWidgets.QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("DFIG — Simulatore Interattivo (PySide6 + OpenGL)")
        self.resize(1600, 1000)

        # Shared render state — read by every plot's draw_func
        self._twin = 10.0
        self._render_snap = None
        self._render_state = None
        self._render_stats = {}
        self._render_ctrl = {}
        self._render_params = None

        self._plot_count = 3
        # Default startup layout: ω_m alone, then P_s/P_r and Q_s/Q_r both
        # stacked (so the user immediately sees the active/reactive balance
        # between stator and rotor when the machine is generating).
        self._plot_signals = [["wm"], ["Ps", "Pr"], ["Qs", "Qr"]]
        self._plot_stacked = [False, True, True]
        self._plot_titles = [f"Plot {j+1}" for j in range(self._plot_count)]
        self._y_persist = {}
        self._cursor_state = {"x": None}

        # UI fps tracking
        self._ui_fps_last = 0.0
        self._ui_fps_ewma = 0.0
        self._gui_counter = 0

        # Load throttle state: C_load = sign × throttle × fondoscala.
        # Modello carico:
        #   C_load = throttle × factor × T_n
        # - factor (signed, ∈ [-2, +2]) è il moltiplicatore di T_nominal
        #   selezionato dallo slider "fondoscala [×T_n]". Il SEGNO codifica
        #   la modalità: positivo = motore (carico resistente, Cl>0),
        #   negativo = generatore (albero spinge, Cl<0). Default -1 (gen 100%).
        # - throttle ∈ [0, 1] è l'ingresso vivo (slider mouse o RT gamepad).
        # - T_n è la coppia nominale del preset macchina, aggiornata da
        #   _on_preset_changed().
        self._throttle = 0.0
        self._load_factor = -1.0
        self._Tn = 1.0   # placeholder, ribasato dal preset al primo load

        # Gamepad: optional, polled at 30 Hz from a QTimer. Initialized
        # before _build_ui so the footer label can show its status.
        self._gamepad = GamepadController()
        self._gp_rescan_counter = 0

        # Saturation modeling state. When OFF, the kernel runs linear and
        # the UI shows a visual warning on |ψ_s|, |ψ_r|, |i_s|, |i_r| if
        # |ψ_s| crosses the saturation threshold. When ON, L_m tapers in
        # the kernel and no UI warning is displayed.
        self._sat_enabled = False

        # Autopilot state: when active, a 10 Hz QTimer drives setpoints
        # programmatically through the existing sliders (visible motion,
        # callbacks fire as if the user moved them by hand).
        self._auto_active = False
        self._auto_t0 = 0.0
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(100)
        self._auto_timer.timeout.connect(self._autopilot_tick)

        # Stick engagement state — for "trim + perturbation" semantics:
        # when a stick exits the deadzone we capture the slider's current
        # value as the home; while engaged the stick drives the slider
        # absolutely; when the stick re-enters the deadzone we restore home.
        self._stick_engaged = {"lx": False, "ly": False, "rx": False, "rt": False}
        self._stick_home    = {"lx": 0.0,   "ly": 0.0,   "rx": 0.0,   "rt": 0.0}

        self._build_ui()

        # Wire up the gamepad tick. Even when no controller is attached we
        # keep the timer running so a hot-plug works (rescan every 30 ticks).
        self._gp_timer = QtCore.QTimer(self)
        self._gp_timer.timeout.connect(self._gamepad_tick)
        self._gp_timer.start(33)  # ~30 Hz

    # ---- UI construction ----
    def _build_ui(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)

        # ---- Header ----
        hdr = QtWidgets.QHBoxLayout()
        tbox = QtWidgets.QVBoxLayout(); tbox.setSpacing(0)
        self._t1 = QtWidgets.QLabel()
        self._t1.setText(htm("DFIG — Simulatore Interattivo",
                             color="#f1f5f9", size="large", bold=True))
        self._t1.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._time_lbl = QtWidgets.QLabel()
        self._time_lbl.setText(htm("compilazione JIT…", color="#475569"))
        self._time_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        tbox.addWidget(self._t1); tbox.addWidget(self._time_lbl)
        hdr.addLayout(tbox, 1)

        # Big RT-factor indicator
        rt_box = QtWidgets.QVBoxLayout(); rt_box.setSpacing(0)
        self._rt_main = QtWidgets.QLabel()
        self._rt_main.setText(htm("— ×", color="#475569", size="xx-large", bold=True))
        self._rt_main.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._rt_main.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._rt_sub = QtWidgets.QLabel()
        self._rt_sub.setText(htm("vs real-time", color="#475569", size="x-small"))
        self._rt_sub.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._rt_sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        rt_box.addWidget(self._rt_main); rt_box.addWidget(self._rt_sub)
        hdr.addLayout(rt_box)
        hdr.addSpacing(12)

        bbx = QtWidgets.QHBoxLayout(); bbx.setSpacing(4)
        self._run_btn = QtWidgets.QPushButton("▶ RUN")
        self._rst_btn = QtWidgets.QPushButton("RESET")
        self._run_btn.clicked.connect(self._on_run)
        self._rst_btn.clicked.connect(self._on_reset)
        bbx.addWidget(self._run_btn); bbx.addWidget(self._rst_btn)
        hdr.addLayout(bbx)
        v.addLayout(hdr)

        # ---- Controls (built into a floating QDockWidget) ----
        # The controls panel is detached by default so the user can keep it
        # off to the side and see plots and inputs at the same time. The
        # "🎛 controlli" button in the header toggles its visibility.
        # Sliders use the compact (single-row) layout; panels stack
        # vertically so the dock is a tall narrow strip with no wasted space.
        cbox = QtWidgets.QVBoxLayout(); cbox.setSpacing(6)

        # State for the synchronous-speed snap on the VC speed-reference
        # slider. f_s and n_p both feed into ω_sync = f_s / n_p; whenever
        # either changes we recompute and push the snap point to the slider.
        self._fs_value = 50.0
        self._np_value = 3.0
        self._snap_tol_giri = 0.3  # ±0.3 giri/s tolerance around sync

        # STATORE
        self._sl_vs = Slider("|V_s|", 690, 0, 1200, 10,
                             lambda x: self.engine.set_ctrl(Vs=x), compact=True)
        def on_fs_change(x):
            self._fs_value = x
            self.engine.set_ctrl(fs=x)
            self._update_sync_snap()
        self._sl_fs = Slider("f_s [Hz]", 50, 0, 200, 0.5,
                             on_fs_change, compact=True)
        cbox.addWidget(panel("STATORE", "#3b82f6", [self._sl_vs, self._sl_fs]))

        # ROTORE
        self._fslip_lbl = QtWidgets.QLabel()
        self._fslip_lbl.setText(htm("f_slip = 0.0 Hz", color="#78716c"))
        self._fslip_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)

        # Mode toggles: open-loop / DPC / VC PF=1
        mode_row = QtWidgets.QHBoxLayout(); mode_row.setSpacing(4)
        mode_lbl = QtWidgets.QLabel()
        mode_lbl.setText(htm("modo", color="#94a3b8"))
        mode_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mode_row.addWidget(mode_lbl, 1)
        self._rb_open = QtWidgets.QPushButton("open-loop"); self._rb_open.setCheckable(True)
        self._rb_dpc  = QtWidgets.QPushButton("DPC");        self._rb_dpc.setCheckable(True)
        self._rb_vc   = QtWidgets.QPushButton("VC PF=1");    self._rb_vc.setCheckable(True)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._rb_open, 0)
        self._mode_group.addButton(self._rb_dpc, 1)
        self._mode_group.addButton(self._rb_vc, 2)
        for b in (self._rb_open, self._rb_dpc, self._rb_vc):
            mode_row.addWidget(b)
        self._mode_group.idToggled.connect(self._on_mode_toggled)
        # Default mode is set further down (after the per-mode boxes are
        # created), so _on_mode_toggled can flip their visibility safely.
        mode_row_w = QtWidgets.QWidget()
        mode_row_w.setLayout(mode_row)

        # Open-loop sliders
        self._ol_vr = Slider("|V_r|", 0, 0, 200, 0.5,
                             lambda x: self.engine.set_ctrl(Vr=x), compact=True)
        self._ol_fr = Slider("f_r [Hz]", 0, -50, 50, 0.5,
                             lambda x: self.engine.set_ctrl(fr=x), compact=True)
        self._ol_box = QtWidgets.QWidget()
        ol_v = QtWidgets.QVBoxLayout(self._ol_box)
        ol_v.setContentsMargins(0, 0, 0, 0); ol_v.setSpacing(2)
        ol_v.addWidget(self._ol_vr); ol_v.addWidget(self._ol_fr); ol_v.addWidget(self._fslip_lbl)

        # DPC sliders (P_s*/Q_s* in kW/kVAR — internally converted to W/VAR)
        self._dpc_box = QtWidgets.QWidget()
        dpc_v = QtWidgets.QVBoxLayout(self._dpc_box)
        dpc_v.setContentsMargins(0, 0, 0, 0); dpc_v.setSpacing(2)
        # DPC setpoints are written by the user in generator convention
        # (positive = delivered to grid). The kernel runs the bang-bang loop
        # in motor convention internally, so we flip the sign on the way in.
        self._sl_dpc_psref = Slider("P_s* [kW]",  0,  -1000, 1000, 5,
                                    lambda x: self.engine.set_ctrl(Ps_ref=-x * 1e3),
                                    compact=True)
        self._sl_dpc_qsref = Slider("Q_s* [kVAR]", 0, -1000, 1000, 5,
                                    lambda x: self.engine.set_ctrl(Qs_ref=-x * 1e3),
                                    compact=True)
        dpc_v.addWidget(self._sl_dpc_psref)
        dpc_v.addWidget(self._sl_dpc_qsref)
        self._sl_dpc_vdc = Slider("V_dc rotore [V]", 100, 10, 500, 5,
                                  lambda x: self.engine.set_ctrl(Vdc_dpc=x),
                                  compact=True)
        self._sl_dpc_hp = Slider("banda h_P [kW]", 0.5, 0.05, 20, 0.05,
                                 lambda x: self.engine.set_ctrl(h_P=x * 1e3),
                                 compact=True)
        self._sl_dpc_hq = Slider("banda h_Q [kVAR]", 0.5, 0.05, 20, 0.05,
                                 lambda x: self.engine.set_ctrl(h_Q=x * 1e3),
                                 compact=True)
        dpc_v.addWidget(self._sl_dpc_vdc)
        dpc_v.addWidget(self._sl_dpc_hp)
        dpc_v.addWidget(self._sl_dpc_hq)
        self._dpc_box.setVisible(False)

        # VC sliders
        self._vc_box = QtWidgets.QWidget()
        vc_v = QtWidgets.QVBoxLayout(self._vc_box)
        vc_v.setContentsMargins(0, 0, 0, 0); vc_v.setSpacing(2)
        wm_sync_giri = self._fs_value / self._np_value
        self._sl_vc_wmref = Slider("ω̃_m [giri/s]", wm_sync_giri, 0, 30, 0.1,
                                   lambda x: self.engine.set_ctrl(wm_ref=x * 2*math.pi),
                                   compact=True,
                                   snap_value=wm_sync_giri,
                                   snap_tolerance=self._snap_tol_giri)
        vc_v.addWidget(self._sl_vc_wmref)

        self._vc_pf_lbl = QtWidgets.QLabel()
        self._vc_pf_lbl.setText(htm("cos(φ_s) = 1.000  (—)", color="#78716c", size="x-small"))
        self._vc_pf_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        def on_pf_change(v):
            self.engine.set_ctrl(tan_phi=v)
            cos_phi = 1.0 / math.sqrt(1.0 + v*v)
            sign = "ind" if v > 0.001 else ("cap" if v < -0.001 else "—")
            self._vc_pf_lbl.setText(htm(
                f"cos(φ_s) = {cos_phi:.3f}  ({sign})", color="#78716c", size="x-small"))
        self._sl_vc_tanphi = Slider("tan(φ_s*)  [Q_s/P_s]", 0.0, -1.0, 1.0, 0.05,
                                    on_pf_change, compact=True)
        vc_v.addWidget(self._sl_vc_tanphi)
        vc_v.addWidget(self._vc_pf_lbl)
        self._sl_vc_pw = Slider("P_w (vel.)", 1e4, 0, 5e4, 100,
                                lambda x: self.engine.set_ctrl(P_w=x), compact=True)
        self._sl_vc_iw = Slider("I_w (vel.)", 1e5, 0, 5e5, 1000,
                                lambda x: self.engine.set_ctrl(I_w=x), compact=True)
        self._sl_vc_pid = Slider("P_i_d", 0.1, 0, 5, 0.05,
                                 lambda x: self.engine.set_ctrl(P_id=x), compact=True)
        self._sl_vc_iid = Slider("I_i_d", 1.0, 0, 50, 0.1,
                                 lambda x: self.engine.set_ctrl(I_id=x), compact=True)
        self._sl_vc_piq = Slider("P_i_q", 1.0, 0, 50, 0.1,
                                 lambda x: self.engine.set_ctrl(P_iq=x), compact=True)
        self._sl_vc_iiq = Slider("I_i_q", 10.0, 0, 200, 1,
                                 lambda x: self.engine.set_ctrl(I_iq=x), compact=True)
        for w in (self._sl_vc_pw, self._sl_vc_iw,
                  self._sl_vc_pid, self._sl_vc_iid,
                  self._sl_vc_piq, self._sl_vc_iiq):
            vc_v.addWidget(w)
        self._vc_box.setVisible(False)

        cbox.addWidget(panel("ROTORE", "#ef4444",
                             [mode_row_w, self._ol_box, self._dpc_box, self._vc_box]))

        # Default mode = VC PF=1 (vector control with stator-flux orientation).
        # All sub-boxes exist now so _on_mode_toggled can drive setVisible.
        self._rb_vc.setChecked(True)

        # CARICO — modello: C_load = throttle × factor × T_n
        # - "fondoscala [×T_n]": signed multiplier in [-2, +2]; il segno
        #   codifica motore/generatore (no più bottone separato).
        # - "throttle [%]": ingresso vivo, mouse + RT gamepad (trim+perturbation).
        self._sl_load_fs = Slider("fondoscala [×T_n]", self._load_factor,
                                  -2.0, 2.0, 0.05, lambda x: None, compact=True)
        self._sl_throttle = Slider("throttle [%]", 0.0,
                                   0.0, 100.0, 1.0, lambda x: None, compact=True)

        self._load_value_lbl = QtWidgets.QLabel()
        self._load_value_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)

        # Now wire the real callbacks and paint initial state.
        self._sl_load_fs.setCallback(self._on_load_fs_change)
        self._sl_throttle.setCallback(self._on_throttle_slider)
        self._update_load()

        cbox.addWidget(panel("CARICO", "#22c55e", [
            self._sl_load_fs,
            self._sl_throttle,
            self._load_value_lbl,
        ]))

        # SIMULAZIONE
        sim_info = QtWidgets.QLabel()
        sim_info.setText(
            htm("RK45 Dormand-Prince · rtol=1e-9 · real-time lock", color="#94a3b8") + "<br>" +
            htm("CPU usata per accuratezza · target = wall × speed", color="#475569", size="x-small")
        )
        sim_info.setTextFormat(QtCore.Qt.TextFormat.RichText)

        # Construct slider with a no-op callback first, then bind the real one
        # (the real callback dereferences self._speed_slider, which only exists
        # after the constructor returns).
        self._speed_slider = Slider("speed (×wall)", 0.0, -2.0, 2.0, 0.05,
                                    lambda x: None, compact=True)
        def set_speed(x):
            sf = 10.0 ** x
            self.engine.set_ctrl(speed_factor=sf)
            if sf >= 100: t = f"{sf:.0f}×"
            elif sf >= 10: t = f"{sf:.1f}×"
            elif sf >= 1: t = f"{sf:.2f}×"
            else: t = f"{sf:.3f}×"
            self._speed_slider.setReadout(htm(t, color="#22c55e", bold=True))
        self._speed_slider.setCallback(set_speed)
        set_speed(0.0)  # trigger initial readout (1.00×)

        # Saturation toggle + threshold slider (compact in SIMULAZIONE panel).
        # The toggle flips engine.params[P_SAT_EN]; the slider sets the
        # |ψ_s| knee. Default threshold matches the active preset's nominal
        # flux × 1.2; preset changes call setRange() + setValue() to rebase.
        self._sat_btn = QtWidgets.QPushButton("saturazione magnetica:  OFF (warning)")
        self._sat_btn.setCheckable(True)
        self._sat_btn.setChecked(False)
        self._sat_btn.setMinimumHeight(24)
        self._sat_btn.toggled.connect(self._on_sat_toggle)

        psi_s_nom_default = 690.0 / (2.0 * math.pi * 50.0)
        self._sl_psi_sat = Slider(
            "ψ_s,sat [Wb]", 1.2 * psi_s_nom_default,
            0.5 * psi_s_nom_default, 2.5 * psi_s_nom_default, 0.01,
            lambda x: self.engine.set_param(9, x), compact=True)

        cbox.addWidget(panel("SIMULAZIONE", "#eab308",
                             [sim_info, self._speed_slider]))
        # AUTOPILOT, PARAMETRI MACCHINA, SEGNALI and the time-window slider are
        # appended to cbox below, after they're created — see further down. We
        # add the trailing stretch only at the very end (after all sections).

        # ---- Three-column main splitter ----
        # Left: stacked control panels (cbox built above), inside a vertical
        # scroll area so ROTORE/VC's tall content doesn't blow up the window.
        # Center: CORRENTI dq above, FLUSSI dq below (vertical splitter).
        # Right: 3 t-plots stacked at equal height (vertical splitter).
        cbox_w = QtWidgets.QWidget()
        cbox_w.setLayout(cbox)

        ctrl_scroll = QtWidgets.QScrollArea()
        ctrl_scroll.setWidget(cbox_w)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctrl_scroll.setMinimumWidth(400)
        ctrl_scroll.setMaximumWidth(520)

        # Center column — dq plots stacked
        traces_curr = [
            ("isd", "isq", 0.23, 0.51, 0.96, "ī_s (statore)"),
            ("ird", "irq", 0.94, 0.27, 0.27, "ī_r (rotore)"),
        ]
        traces_flux = [
            ("psd", "psq", 0.23, 0.51, 0.96, "φ̄_s (statore)"),
            ("prd", "prq", 0.94, 0.27, 0.27, "φ̄_r (rotore)"),
        ]
        # Master plot drives the vsync loop
        self._da_curr = MasterPlotGL()
        self._da_curr.setMinimumSize(220, 220)
        self._da_curr.setDrawFunc(self._make_dq_drawer(traces_curr, "CORRENTI (d,q) [A]"))
        self._da_curr.frame_tick.connect(self._gui_tick)
        self._da_flux = PlotGL()
        self._da_flux.setMinimumSize(220, 220)
        self._da_flux.setDrawFunc(self._make_dq_drawer(traces_flux, "FLUSSI (d,q) [mWb]"))

        # P-Q plane plot — same dq drawer reused with axis labels swapped
        # (P on x, Q on y) and gen-convention power values from snap.
        traces_pq = [
            ("Ps", "Qs", 0.23, 0.51, 0.96, "S̄_s (statore)"),
            ("Pr", "Qr", 0.94, 0.27, 0.27, "S̄_r (rotore)"),
        ]
        self._da_pq = PlotGL()
        self._da_pq.setMinimumSize(200, 180)
        self._da_pq.setDrawFunc(self._make_pq_drawer(
            traces_pq, "POTENZE (P,Q) [kW, kVAR]"))

        # Saturation curve plot — uses state + params (not snap), so it has
        # its own drawer factory that reads the latest snapshot from self.
        self._da_sat = PlotGL()
        self._da_sat.setMinimumSize(200, 180)
        self._da_sat.setDrawFunc(self._make_sat_drawer())

        # Center column laid out as a 2×2 grid via nested splitters:
        #   top    row: CORRENTI | POTENZE
        #   bottom row: FLUSSI   | SATURAZIONE
        center_top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        center_top.setChildrenCollapsible(False)
        center_top.addWidget(self._da_curr)
        center_top.addWidget(self._da_pq)
        center_top.setSizes([100, 100])

        center_bot = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        center_bot.setChildrenCollapsible(False)
        center_bot.addWidget(self._da_flux)
        center_bot.addWidget(self._da_sat)
        center_bot.setSizes([100, 100])

        center_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center_split.setChildrenCollapsible(False)
        center_split.addWidget(center_top)
        center_split.addWidget(center_bot)
        center_split.setSizes([100, 100])

        # Right column — N t-plots, equal height
        self._plot_drawing_areas = []
        right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_split.setChildrenCollapsible(False)
        for j in range(self._plot_count):
            da = PlotGL()
            da.setMinimumHeight(80)
            da.setDrawFunc(self._make_tplot_drawer(j))
            # Cross-plot cursor
            def make_motion(da_local):
                def mm(ev):
                    self._cursor_state["x"] = float(ev.position().x())
                    for d in self._plot_drawing_areas:
                        d.update()
                return mm
            def leave(ev):
                self._cursor_state["x"] = None
                for d in self._plot_drawing_areas:
                    d.update()
            da.mouseMoveEvent = make_motion(da)
            da.leaveEvent = leave
            self._plot_drawing_areas.append(da)
            right_split.addWidget(da)
        right_split.setSizes([100] * self._plot_count)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)
        main_split.addWidget(ctrl_scroll)
        main_split.addWidget(center_split)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setStretchFactor(2, 2)
        main_split.setSizes([360, 380, 800])
        v.addWidget(main_split, 1)  # take all available vertical space

        # ---- Time window slider ----
        tw_row = QtWidgets.QHBoxLayout(); tw_row.setSpacing(8)
        tw_lbl = QtWidgets.QLabel()
        tw_lbl.setText(htm("Finestra:", color="#94a3b8"))
        tw_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._tw_val = QtWidgets.QLabel()
        self._tw_val.setText(htm("10 s", color="#e2e8f0"))
        self._tw_val.setTextFormat(QtCore.Qt.TextFormat.RichText)
        def set_twin(x):
            self._twin = x
            self._tw_val.setText(htm(f"{x:.0f} s", color="#e2e8f0"))
        self._tw_slider = Slider("", 10, 1, 60, 1, set_twin)
        tw_row.addWidget(tw_lbl)
        tw_row.addWidget(self._tw_slider, 1)
        tw_row.addWidget(self._tw_val)
        # Note: tw_row is appended to cbox at the bottom (with the other
        # collapsible sections), not to v.

        # ---- Machine parameters grid (collapsible) ----
        # State: which preset's nominal bases (V_n, S_n, f_s) are used to
        # compute pu values for the sanity-check readout. The default preset
        # (1 MVA generic) matches the engine's startup PARAMS_DEFAULT.
        self._preset_si = preset_to_si(MACHINE_PRESETS[0])

        param_inner = QtWidgets.QWidget()
        pf_v = QtWidgets.QVBoxLayout(param_inner)
        pf_v.setContentsMargins(6, 6, 6, 6); pf_v.setSpacing(4)

        # --- preset dropdown ---
        preset_row = QtWidgets.QHBoxLayout()
        preset_lbl = QtWidgets.QLabel()
        preset_lbl.setText(htm("preset:", color="#94a3b8"))
        preset_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._preset_combo = QtWidgets.QComboBox()
        for p in MACHINE_PRESETS:
            self._preset_combo.addItem(p["name"])
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self._preset_combo, 1)
        pf_v.addLayout(preset_row)

        # --- 8-row parameter grid ---
        self._param_inputs = {}
        pgrid = QtWidgets.QGridLayout(); pgrid.setHorizontalSpacing(14); pgrid.setVerticalSpacing(2)
        for i, (key, label, default_si, scale, unit) in enumerate(PARAMS_DEF):
            display_v = default_si / scale
            int_only = (key == "NP")
            def make_cb(idx, sc, k):
                def cb(v_disp):
                    self.engine.set_param(idx, v_disp * sc)
                    if k == "NP":
                        self._np_value = v_disp
                        self._update_sync_snap()
                    self._refresh_pu_readout()
                return cb
            w = ParamInput(label, display_v, unit, make_cb(i, scale, key),
                           int_only=int_only)
            pgrid.addWidget(w, i, 0)
            self._param_inputs[key] = w
        pf_v.addLayout(pgrid)

        # --- pu-derived readout (sanity check) ---
        self._pu_lbl = QtWidgets.QLabel()
        self._pu_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._pu_lbl.setWordWrap(True)
        pf_v.addWidget(self._pu_lbl)

        # --- saturation sub-panel, nested inside PARAMETRI MACCHINA ---
        # The toggle and the ψ_s,sat slider both live here (they're machine
        # properties more than simulation properties).
        sat_subpanel = panel("SATURAZIONE MAGNETICA", "#f97316",
                             [self._sat_btn, self._sl_psi_sat])
        pf_v.addWidget(sat_subpanel)

        param_section = panel("PARAMETRI MACCHINA", "#06b6d4", [param_inner])
        # Initial pu readout (engine defaults match preset 0's bases).
        self._refresh_pu_readout()

        # ---- Signal composer (collapsible) ----
        comp_inner_w = QtWidgets.QWidget()
        comp_box = QtWidgets.QVBoxLayout(comp_inner_w)
        comp_box.setContentsMargins(6, 6, 6, 6); comp_box.setSpacing(2)

        self._comp_checks = {}
        self._value_labels = {}
        unified = self._build_unified_signal_grid([
            ("scalari", SCALAR_KEYS),
            ("dq", DQ_KEYS),
            ("moduli / derivati", MOD_KEYS),
        ])
        comp_box.addLayout(unified)

        # Stacked toggle row (one per plot)
        stack_row = QtWidgets.QHBoxLayout(); stack_row.setSpacing(8)
        sl_lbl = QtWidgets.QLabel()
        sl_lbl.setText(htm("visualizz. stacked Σ", color="#94a3b8", size="x-small"))
        sl_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        stack_row.addWidget(sl_lbl)
        self._stack_cbs = []
        for j in range(self._plot_count):
            cb = QtWidgets.QCheckBox(f"P{j+1}")
            cb.setChecked(self._plot_stacked[j])
            def make_handler(idx):
                def h(checked):
                    self._plot_stacked[idx] = bool(checked)
                    self._y_persist.pop(idx, None)
                    for d in self._plot_drawing_areas:
                        d.update()
                return h
            cb.toggled.connect(make_handler(j))
            stack_row.addWidget(cb)
            self._stack_cbs.append(cb)
        stack_row.addStretch(1)
        comp_box.addLayout(stack_row)

        composer = panel("SEGNALI", "#64748b", [comp_inner_w])

        # ---- Autopilot panel ----
        auto_inner = QtWidgets.QWidget()
        auto_v = QtWidgets.QVBoxLayout(auto_inner)
        auto_v.setContentsMargins(6, 6, 6, 6); auto_v.setSpacing(4)

        auto_row1 = QtWidgets.QHBoxLayout(); auto_row1.setSpacing(8)
        sl_lbl = QtWidgets.QLabel()
        sl_lbl.setText(htm("scenario:", color="#94a3b8"))
        sl_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._auto_scenario = QtWidgets.QComboBox()
        for s in AUTOPILOT_SCENARIOS:
            self._auto_scenario.addItem(s["name"])
        self._auto_loop = QtWidgets.QCheckBox("loop")
        self._auto_loop.setChecked(True)
        auto_row1.addWidget(sl_lbl)
        auto_row1.addWidget(self._auto_scenario, 1)
        auto_row1.addWidget(self._auto_loop)
        auto_v.addLayout(auto_row1)

        auto_row2 = QtWidgets.QHBoxLayout(); auto_row2.setSpacing(8)
        self._auto_btn = QtWidgets.QPushButton("▶ START")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setMinimumHeight(26)
        self._auto_btn.toggled.connect(self._on_autopilot_toggle)
        self._auto_status = QtWidgets.QLabel()
        self._auto_status.setText(htm("idle", color="#475569"))
        self._auto_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        auto_row2.addWidget(self._auto_btn)
        auto_row2.addWidget(self._auto_status, 1)
        auto_v.addLayout(auto_row2)

        self._auto_progress = QtWidgets.QProgressBar()
        self._auto_progress.setRange(0, 1000)
        self._auto_progress.setValue(0)
        self._auto_progress.setTextVisible(True)
        self._auto_progress.setFormat("%p ‰")
        self._auto_progress.setFixedHeight(14)
        auto_v.addWidget(self._auto_progress)

        self._auto_phase = QtWidgets.QLabel()
        self._auto_phase.setText(htm("—", color="#475569", size="x-small"))
        self._auto_phase.setTextFormat(QtCore.Qt.TextFormat.RichText)
        auto_v.addWidget(self._auto_phase)

        autopilot_section = panel("AUTOPILOT", "#a855f7", [auto_inner])

        # ---- Append all collapsible sections + time-window slider to the
        # left column (cbox), in order, then close with the stretch. They
        # were created out-of-order above because they cross-reference
        # widgets that needed to exist first.
        cbox.addWidget(autopilot_section)
        cbox.addWidget(param_section)
        cbox.addWidget(composer)
        cbox.addLayout(tw_row)
        cbox.addStretch(1)

        # ---- Footer ----
        ft_row = QtWidgets.QHBoxLayout()
        ft_row.setSpacing(12)
        ft = QtWidgets.QLabel()
        ft.setText(htm(
            "A_n=1MVA · V_n=690V · n_p=3 · R_s=4.8mΩ · R_r=2.4mΩ · "
            "L_s=6.8mH · L_r=7.1mH · M=6.8mH · J=50kg·m² · b=3.3 · "
            "JIT Numba · sim thread @ background",
            color="#334155", size="x-small"))
        ft.setTextFormat(QtCore.Qt.TextFormat.RichText)
        ft_row.addWidget(ft, 1)
        self._gp_lbl = QtWidgets.QLabel()
        self._gp_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._gp_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ft_row.addWidget(self._gp_lbl)
        v.addLayout(ft_row)
        self._refresh_gamepad_label()

        # Update plot titles to reflect the default _plot_signals (the
        # signal-grid checkboxes were checked before connect, so the
        # title-update callback didn't fire).
        self._update_plot_titles()

    def _build_unified_signal_grid(self, groups):
        """Single QGridLayout listing all signals as: ● name | value | u.m. |
        P1 P2 P3. ``groups`` is a list of (title, [keys]); each group is
        preceded by a section divider row spanning all columns. Used to fit
        the composer into the narrow left column instead of three side-by-
        side mini-grids."""
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(10); g.setVerticalSpacing(1)

        def hdr(text, align=QtCore.Qt.AlignmentFlag.AlignLeft):
            l = QtWidgets.QLabel()
            l.setText(htm(text, color="#475569", size="x-small"))
            l.setTextFormat(QtCore.Qt.TextFormat.RichText)
            l.setAlignment(align | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return l

        def section_hdr(text):
            l = QtWidgets.QLabel()
            l.setText(htm(f"── {text} ──", color="#64748b",
                          size="x-small", bold=True))
            l.setTextFormat(QtCore.Qt.TextFormat.RichText)
            l.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft |
                           QtCore.Qt.AlignmentFlag.AlignVCenter)
            return l

        n_cols = 3 + self._plot_count

        # Column headers — row 0
        g.addWidget(hdr("segnale"), 0, 0)
        g.addWidget(hdr("valore", QtCore.Qt.AlignmentFlag.AlignRight), 0, 1)
        g.addWidget(hdr("u.m."), 0, 2)
        for j in range(self._plot_count):
            g.addWidget(hdr(f"P{j+1}", QtCore.Qt.AlignmentFlag.AlignHCenter),
                        0, j + 3)

        row = 1
        for title, keys in groups:
            sec = section_hdr(title)
            g.addWidget(sec, row, 0, 1, n_cols)
            row += 1

            for key in keys:
                if key not in SIG_BY_KEY:
                    continue
                meta = SIG_BY_KEY[key]
                color_hex = (f"#{int(meta['r']*255):02x}"
                             f"{int(meta['g']*255):02x}"
                             f"{int(meta['b']*255):02x}")
                sl = QtWidgets.QLabel()
                sl.setText(htm(f"● {meta['name']}", color=color_hex))
                sl.setTextFormat(QtCore.Qt.TextFormat.RichText)
                g.addWidget(sl, row, 0)

                vl = QtWidgets.QLabel()
                vl.setText(htm("—", bold=True))
                vl.setTextFormat(QtCore.Qt.TextFormat.RichText)
                vl.setMinimumWidth(72)
                vl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                QtCore.Qt.AlignmentFlag.AlignVCenter)
                g.addWidget(vl, row, 1)
                self._value_labels[key] = vl

                ul = QtWidgets.QLabel()
                ul.setText(htm(meta["unit"], color="#475569", size="x-small"))
                ul.setTextFormat(QtCore.Qt.TextFormat.RichText)
                g.addWidget(ul, row, 2)

                for j in range(self._plot_count):
                    cb = QtWidgets.QCheckBox()
                    cb.setChecked(key in self._plot_signals[j])
                    def make_h(k=key, j=j):
                        def h(checked):
                            if checked:
                                if k not in self._plot_signals[j]:
                                    self._plot_signals[j].append(k)
                            else:
                                if k in self._plot_signals[j]:
                                    self._plot_signals[j].remove(k)
                            self._y_persist.pop(j, None)
                            self._update_plot_titles()
                            for d in self._plot_drawing_areas:
                                d.update()
                        return h
                    cb.toggled.connect(make_h())
                    g.addWidget(cb, row, j + 3,
                                QtCore.Qt.AlignmentFlag.AlignHCenter)
                    self._comp_checks[(key, j)] = cb
                row += 1

        g.setRowStretch(row, 1)
        return g

    # ---- draw_func factories ----
    def _make_dq_drawer(self, traces, title):
        def draw(painter, w, h):
            draw_dq(painter, w, h, self._render_snap, traces, self._twin, title)
        return draw

    def _make_pq_drawer(self, traces, title):
        """Reuse draw_dq with P/Q axis labels — same scia + leading-vector
        rendering, only the data fields and labels change."""
        def draw(painter, w, h):
            draw_dq(painter, w, h, self._render_snap, traces, self._twin,
                    title, axis_labels=("P", "Q"))
        return draw

    def _make_sat_drawer(self):
        def draw(painter, w, h):
            draw_saturation(painter, w, h,
                            self._render_state, self._render_params)
        return draw

    def _make_tplot_drawer(self, j):
        def draw(painter, w, h):
            draw_tplot(painter, w, h, self._render_snap,
                       self._plot_signals[j], self._twin, self._plot_titles[j],
                       j, self._y_persist, self._cursor_state,
                       stacked=self._plot_stacked[j])
        return draw

    def _update_plot_titles(self):
        for j in range(self._plot_count):
            names = [SIG_BY_KEY[k]["name"] for k in self._plot_signals[j] if k in SIG_BY_KEY]
            self._plot_titles[j] = ((f"Plot {j+1} · " + ", ".join(names))
                                    if names else f"Plot {j+1}")

    # ---- buttons ----
    def _on_run(self):
        running = self.engine.toggle_run()
        self._run_btn.setText("■ STOP" if running else "▶ RUN")

    def _on_reset(self):
        # 1. Stop autopilot if running, otherwise it would immediately
        #    overwrite the slider values we're about to restore.
        if self._auto_active:
            self._auto_btn.setChecked(False)

        # 2. Reset every control slider (preset-independent ones) to its
        #    "neutral" startup default. Slider.setValue propagates via the
        #    callback → engine.set_ctrl, so the engine sees the new values.
        # Open-loop
        self._ol_vr.setValue(0)
        self._ol_fr.setValue(0)
        # DPC
        self._sl_dpc_psref.setValue(0)
        self._sl_dpc_qsref.setValue(0)
        self._sl_dpc_vdc.setValue(100)
        self._sl_dpc_hp.setValue(0.5)
        self._sl_dpc_hq.setValue(0.5)
        # VC
        self._sl_vc_tanphi.setValue(0)
        self._sl_vc_pw.setValue(1e4)
        self._sl_vc_iw.setValue(1e5)
        self._sl_vc_pid.setValue(0.1)
        self._sl_vc_iid.setValue(1.0)
        self._sl_vc_piq.setValue(1.0)
        self._sl_vc_iiq.setValue(10.0)
        # Time-window + speed factor (view properties)
        self._tw_slider.setValue(10)
        self._speed_slider.setValue(0.0)  # log10(1) → 1.00× wall

        # 3. Macro state: control mode, load factor, throttle, saturation toggle.
        self._throttle = 0.0
        self._sl_throttle.setValue(0.0)
        # Default factor = -1.0 (generatore al 100% T_n). Lo slider scrive
        # _load_factor via _on_load_fs_change quando viene impostato.
        self._sl_load_fs.setValue(-1.0)
        if not self._rb_vc.isChecked():
            self._rb_vc.setChecked(True)  # back to VC mode
        if self._sat_btn.isChecked():
            self._sat_btn.setChecked(False)  # saturation OFF (warning mode)

        # 4. Re-apply the currently selected preset. This restores machine
        #    parameters, V_n / f_s nominals, ω̃_m_sync default and snap, ψ_s,sat
        #    default, and the C_load fondoscala range.
        self._on_preset_changed(self._preset_combo.currentIndex())

        # 5. Engine state reset (zeros integrators, history, anchors).
        self.engine.reset()

        # 6. Plot UI: clear the hysteretic Y bounds and the cross-plot cursor,
        #    then force a repaint so the new "fresh" state is visible.
        self._y_persist.clear()
        self._cursor_state["x"] = None
        for d in self._plot_drawing_areas:
            d.update()
        self._da_curr.update(); self._da_flux.update()
        self._da_pq.update(); self._da_sat.update()

    # ---- Autopilot ----
    def _on_autopilot_toggle(self, checked):
        if checked:
            scenario = AUTOPILOT_SCENARIOS[self._auto_scenario.currentIndex()]
            # Force VC mode (autopilot is wired against VC setpoints only)
            if not self._rb_vc.isChecked():
                self._rb_vc.setChecked(True)
            # Apply ensure constraints (e.g. force generator mode for load test).
            # motor_mode True/False mappa al SEGNO di _load_factor (positivo
            # motore, negativo generatore); preserva la magnitudine corrente.
            ensure = scenario.get("ensure", {})
            if "motor_mode" in ensure:
                want_motor = ensure["motor_mode"]
                cur_is_motor = self._load_factor > 0
                if want_motor != cur_is_motor:
                    self._sl_load_fs.setValue(-self._load_factor)
            self._auto_active = True
            self._auto_t0 = time.perf_counter()
            self._auto_btn.setText("■ STOP")
            self._auto_timer.start()
            self._auto_status.setText(htm("running …", color="#22c55e", bold=True))
        else:
            self._auto_active = False
            self._auto_timer.stop()
            self._auto_btn.setText("▶ START")
            self._auto_progress.setValue(0)
            self._auto_phase.setText(htm("—", color="#475569", size="x-small"))
            self._auto_status.setText(htm("idle", color="#475569"))

    def _autopilot_tick(self):
        if not self._auto_active:
            return
        scenario = AUTOPILOT_SCENARIOS[self._auto_scenario.currentIndex()]
        stages = scenario["stages"]
        total_dur = sum(s["dur"] for s in stages)
        if total_dur <= 0:
            return

        elapsed = time.perf_counter() - self._auto_t0
        if elapsed >= total_dur:
            if self._auto_loop.isChecked():
                self._auto_t0 = time.perf_counter()
                elapsed = 0.0
            else:
                # End of one-shot run — flip the toggle off, which restores idle UI
                self._auto_btn.setChecked(False)
                return

        # Locate the current stage
        t_acc = 0.0
        cur = stages[0]
        t_in_stage = elapsed
        for st in stages:
            if elapsed < t_acc + st["dur"]:
                cur = st
                t_in_stage = elapsed - t_acc
                break
            t_acc += st["dur"]

        # Apply the stage's setpoints. Each branch writes onto the corresponding
        # slider via setValue, which fires the slider's callback → engine.set_ctrl,
        # so the UI stays visually in sync with what the engine sees.
        if "wm_factor" in cur:
            sync_giri = self._fs_value / max(self._np_value, 1e-9)
            self._sl_vc_wmref.setValue(cur["wm_factor"] * sync_giri)
        if "tan_phi" in cur:
            self._sl_vc_tanphi.setValue(cur["tan_phi"])
        if "tan_phi_ramp" in cur:
            a, b = cur["tan_phi_ramp"]
            alpha = min(1.0, t_in_stage / cur["dur"]) if cur["dur"] > 0 else 1.0
            self._sl_vc_tanphi.setValue(a + (b - a) * alpha)
        if "throttle" in cur:
            self._throttle = cur["throttle"]
            self._update_load()

        # UI feedback
        permil = int(round(elapsed / total_dur * 1000))
        self._auto_progress.setValue(permil)
        self._auto_phase.setText(htm(
            f"fase: {cur.get('desc', '')}  ·  t = {elapsed:5.1f} / {total_dur:.1f} s",
            color="#22c55e", size="x-small"))

    # ---- Magnetic saturation toggle ----
    def _on_sat_toggle(self, checked):
        self._sat_enabled = bool(checked)
        # P_SAT_EN = 8 in dfig_engine.PARAMS layout
        self.engine.set_param(8, 1.0 if self._sat_enabled else 0.0)
        self._sat_btn.setText(
            f"saturazione magnetica:  {'ON  (modello)' if checked else 'OFF (warning)'}")

    # ---- Machine preset ----
    def _on_preset_changed(self, idx):
        """Load a machine preset: pushes SI parameter values to the grid,
        rebases V_s and f_s sliders to the preset's nominals, and rescales
        the C_load fondoscala slider so its max ≈ 1.5 × T_n."""
        if idx < 0 or idx >= len(MACHINE_PRESETS):
            return
        p = MACHINE_PRESETS[idx]
        self._preset_si = preset_to_si(p)
        si = self._preset_si

        # Map PARAMS_DEF keys → keys in the preset_to_si() output dict.
        # PARAMS_DEF uses "B" (upper) for the friction coefficient.
        si_map = {"Rs": "Rs", "Rr": "Rr", "Ls": "Ls", "Lr": "Lr",
                  "M": "M", "NP": "NP", "J": "J", "B": "B"}

        # Push values to each ParamInput. setValue fires the on_change callback,
        # which propagates to the engine via engine.set_param.
        for key, label, default_si, scale, unit in PARAMS_DEF:
            si_value = si[si_map[key]]
            self._param_inputs[key].setValue(si_value / scale)

        # Update V_s and f_s sliders to the preset's rated nominals.
        self._sl_vs.setValue(p["Vn"])
        self._sl_fs.setValue(p["fs"])

        # T_n del preset → ribasa la coppia nominale per il modello carico.
        # Lo slider "fondoscala" è dimensionless (×T_n), non va riranged.
        self._Tn = si["Tn"]
        self._update_load()

        # Saturation threshold default: 1.2 × ψ_s,nom = 1.2 · V_n / (2π·f_s).
        # Slider range stretched ±125% of the nominal flux.
        psi_nom = p["Vn"] / (2.0 * math.pi * p["fs"])
        sat_lo = 0.5 * psi_nom
        sat_hi = 2.5 * psi_nom
        sat_step = max(0.001, psi_nom / 200.0)
        self._sl_psi_sat.setRange(sat_lo, sat_hi, sat_step)
        self._sl_psi_sat.setValue(1.2 * psi_nom)

        self._update_sync_snap()
        self._refresh_pu_readout()

    def _refresh_pu_readout(self):
        """Read the current SI parameters from the widgets, convert to pu
        using the active preset's rated bases (V_n, S_n, f_s), and render a
        compact one-line readout. Each pu value is colored by sanity band:
        gray = within typical band, yellow = mild deviation, red = far off."""
        if not hasattr(self, "_pu_lbl") or not hasattr(self, "_param_inputs"):
            return
        si_now = {}
        for key, label, default_si, scale, unit in PARAMS_DEF:
            si_now[key] = self._param_inputs[key].value() * scale

        Vn = self._preset_si["Vn"]
        Sn = self._preset_si["Sn"]
        fs = self._preset_si["fs"]
        Zb = Vn * Vn / Sn
        Lb = Zb / (2.0 * math.pi * fs)

        Lls = (si_now["Ls"] - si_now["M"]) / Lb
        Llr = (si_now["Lr"] - si_now["M"]) / Lb
        pu = {
            "Rs":  si_now["Rs"] / Zb,
            "Rr":  si_now["Rr"] / Zb,
            "Lls": Lls,
            "Llr": Llr,
            "Lm":  si_now["M"] / Lb,
        }
        Tn = self._preset_si["Tn"]
        sigma = pu["Lls"] + pu["Llr"]

        def color_of(key, val):
            lo, hi = PU_BANDS[key]
            if lo <= val <= hi:
                return "#94a3b8"      # in band — neutral
            # Far outside: > 2× the band edge → red
            if val < 0.5 * lo or val > 2.0 * hi:
                return "#ef4444"
            return "#eab308"           # near edge → warning

        parts = []
        for label, key in (("R_s", "Rs"), ("R_r", "Rr"),
                           ("L_ls", "Lls"), ("L_lr", "Llr"), ("L_m", "Lm")):
            parts.append(htm(f"{label} = {pu[key]:.3f} pu",
                             color=color_of(key, pu[key]), size="x-small"))
        parts.append(htm(f"σ = {sigma:.3f} pu", color="#94a3b8", size="x-small"))
        parts.append(htm(f"T_n = {Tn:.0f} N·m", color="#06b6d4",
                         size="x-small", bold=True))
        sep = htm("  ·  ", color="#475569", size="x-small")
        self._pu_lbl.setText(sep.join(parts))

    # ---- Synchronous-speed snap update ----
    def _update_sync_snap(self):
        """Recompute the synchronous-speed snap point on the VC ω̃_m slider
        whenever f_s or n_p change. ω_sync_giri = f_s / n_p."""
        if not hasattr(self, "_sl_vc_wmref"):
            return  # called during _build_ui before the slider exists
        if self._np_value <= 0:
            return
        sync = self._fs_value / self._np_value
        self._sl_vc_wmref.setSnap(sync, self._snap_tol_giri)

    # ---- Load: C_load = throttle × factor × T_n ----
    def _on_load_fs_change(self, v):
        """Slider fondoscala (signed factor ×T_n): aggiorna factor e ricalcola
        C_load. Il segno codifica motore/generatore (non c'è più bottone)."""
        self._load_factor = v
        self._update_load()

    def _on_throttle_slider(self, v):
        """Mouse-driven throttle in [0, 100]. Map to 0..1 e applica."""
        self._throttle = v / 100.0
        self._update_load()

    def _update_load(self):
        """Push C_load = throttle × factor × T_n to the engine + readout."""
        cl = self._throttle * self._load_factor * self._Tn
        self.engine.set_ctrl(Cl=cl)
        pct = int(round(self._throttle * 100))
        # Colore: giallo se motore (factor>0), verde se generatore (<0).
        color = ("#eab308" if self._load_factor > 0
                 else "#22c55e" if self._load_factor < 0
                 else "#94a3b8")
        mode_lbl = ("MOTORE" if self._load_factor > 0
                    else "GENERATORE" if self._load_factor < 0
                    else "—")
        self._load_value_lbl.setText(htm(
            f"{mode_lbl}  ·  {self._load_factor:+.2f}×T_n  ·  "
            f"throttle {pct}%  →  {cl:+.0f} Nm",
            color=color, bold=True))

    def _on_mode_toggled(self, button_id, checked):
        if not checked:
            return  # exclusive group fires twice; we only care about the new active one
        if button_id == 2:
            self.engine.set_ctrl(mode="vc")
            self._ol_box.setVisible(False); self._dpc_box.setVisible(False); self._vc_box.setVisible(True)
        elif button_id == 1:
            self.engine.set_ctrl(mode="dpc")
            self._ol_box.setVisible(False); self._dpc_box.setVisible(True);  self._vc_box.setVisible(False)
        else:
            self.engine.set_ctrl(mode="open")
            self._ol_box.setVisible(True);  self._dpc_box.setVisible(False); self._vc_box.setVisible(False)
        # Mode change re-maps LStick to a different slider — drop any active
        # engagement so the next stick movement captures a fresh home from
        # the new target slider.
        self._stick_engaged["lx"] = False
        self._stick_engaged["ly"] = False
        self._refresh_gamepad_label()

    # ---- Gamepad ----
    def _refresh_gamepad_label(self):
        """Footer overlay: controller status + which sliders the LStick maps to."""
        # Defensive: _on_mode_toggled may fire during _build_ui (when the
        # default mode button is checked) before the footer label exists.
        if not hasattr(self, "_gp_lbl"):
            return
        if not self._gamepad.available:
            self._gp_lbl.setText(htm("🎮 nessun controller", color="#475569", size="x-small"))
            return
        cur_id = self._mode_group.checkedId()
        ctx = {0: "f_r / |V_r|",
               1: "P_s* / Q_s*",
               2: "ω_m_ref / tan(φ_s*)"}.get(cur_id, "")
        self._gp_lbl.setText(
            htm(f"🎮 {self._gamepad.name}  ", color="#22c55e", size="x-small") +
            htm(f"LStick→{ctx}", color="#94a3b8", size="x-small"))

    @staticmethod
    def _bipolar_to_range(v, lo, hi):
        """Map [-1, 1] → [lo, hi]."""
        return lo + (v + 1.0) * 0.5 * (hi - lo)

    def _drive_axis(self, name, stick_val, slider, scale_fn):
        """Drive a slider from an analog stick axis using trim+perturbation:
        - On the rising edge (deadzone → engaged) capture slider.value() as home.
        - While engaged, set slider = scale_fn(stick_val).
        - On the falling edge (engaged → deadzone) restore the captured home.
        scale_fn maps stick_val ∈ [-1, 1] to the absolute slider value.
        """
        if abs(stick_val) > 1e-6:
            if not self._stick_engaged[name]:
                self._stick_home[name] = slider.value()
                self._stick_engaged[name] = True
            slider.setValue(scale_fn(stick_val))
        else:
            if self._stick_engaged[name]:
                slider.setValue(self._stick_home[name])
                self._stick_engaged[name] = False

    def _refresh_gamepad_diag(self, rt, lt, n_axes, axes_raw):
        """Aggiorna la label gamepad con info live (RT/LT + assi raw) quando
        almeno un trigger o asse è attivo. Aiuta a diagnosticare mapping non
        standard (controller dove RT non è axis(5))."""
        if not hasattr(self, "_gp_lbl") or not self._gamepad.available:
            return
        cur_id = self._mode_group.checkedId()
        ctx = {0: "f_r / |V_r|",
               1: "P_s* / Q_s*",
               2: "ω_m_ref / tan(φ_s*)"}.get(cur_id, "")
        # Se qualche asse è "attivo" (oltre il rumore), mostra raw values.
        any_active = any(abs(v) > 0.05 for v in axes_raw) if axes_raw else False
        diag = ""
        if any_active or rt > 0.01 or lt > 0.01:
            ax = " ".join(f"a{i}={v:+.2f}" for i, v in enumerate(axes_raw))
            diag = (f"  RT={rt:.2f} LT={lt:.2f}  ({n_axes} axes: {ax})")
        self._gp_lbl.setText(
            htm(f"🎮 {self._gamepad.name}  ", color="#22c55e", size="x-small") +
            htm(f"LStick→{ctx}", color="#94a3b8", size="x-small") +
            htm(diag, color="#eab308", size="x-small"))

    def _gamepad_tick(self):
        # Hot-plug rescan every ~1 s while no controller is attached
        if not self._gamepad.available:
            self._gp_rescan_counter += 1
            if self._gp_rescan_counter >= 30:
                self._gp_rescan_counter = 0
                self._gamepad.rescan()
                if self._gamepad.available:
                    self._refresh_gamepad_label()
            return

        a = self._gamepad.poll()
        if not a:
            return

        # ---- Buttons (edge-detected) ----
        if a.get("A"):
            self._on_run()
        if a.get("B"):
            self._on_reset()
        if a.get("X"):
            cur_id = self._mode_group.checkedId()
            nxt_id = (cur_id + 1) % 3
            {0: self._rb_open, 1: self._rb_dpc, 2: self._rb_vc}[nxt_id].setChecked(True)
        if a.get("Y"):
            # Flip motore/generatore = inverte il segno del fattore di carico.
            self._sl_load_fs.setValue(-self._load_factor)

        # ---- Shoulders → time window ±1 s (auto-repeat naturale: il button
        # fa edge-detection, quindi ogni nuovo press è un singolo step) ----
        if a.get("LB"):
            self._tw_slider.setValue(self._tw_slider.value() - 1)
        if a.get("RB"):
            self._tw_slider.setValue(self._tw_slider.value() + 1)

        # ---- D-Pad (auto-repeat 5 Hz from the controller layer) ----
        if a.get("D_UP"):
            self._sl_fs.setValue(self._sl_fs.value() + 0.5)
        if a.get("D_DOWN"):
            self._sl_fs.setValue(self._sl_fs.value() - 0.5)
        if a.get("D_RIGHT"):
            self._sl_vs.setValue(self._sl_vs.value() + 10)
        if a.get("D_LEFT"):
            self._sl_vs.setValue(self._sl_vs.value() - 10)

        # ---- RT → slider throttle (trim+perturbation, come ω̃_m via L-stick).
        # Mentre RT è premuto lo slider va a rt*100; al rilascio torna al
        # valore "home" dov'era prima dell'engagement. Il valore di
        # self._throttle viene aggiornato dal callback dello slider stesso
        # (_on_throttle_slider), così mouse e gamepad condividono il path.
        rt_val = a.get("rt", 0.0)
        self._drive_axis("rt", rt_val, self._sl_throttle, lambda v: v * 100.0)
        # Diagnostica live: mostra RT/LT/assi raw nella label gamepad quando
        # qualcosa è attivo (utile per mapping non-standard).
        self._refresh_gamepad_diag(
            rt_val, a.get("lt", 0.0),
            a.get("_naxes", 0), a.get("_axes_raw", ()))

        # ---- LStick → context-aware (open / DPC / VC), trim+perturbation:
        # the slider value at the moment of stick engagement is captured as
        # "home" and restored when the stick returns to deadzone.
        lx = a.get("lx", 0.0); ly = a.get("ly", 0.0)
        cur_id = self._mode_group.checkedId()
        if cur_id == 0:
            self._drive_axis("lx", lx, self._ol_fr, lambda v: v * 50.0)
            # |V_r| ≥ 0; only positive half of the Y stick maps onto |V_r|,
            # negative half clamps the slider to 0 while engaged.
            self._drive_axis("ly", ly, self._ol_vr, lambda v: max(0.0, v) * 200.0)
        elif cur_id == 1:
            self._drive_axis("lx", lx, self._sl_dpc_psref, lambda v: v * 1000.0)
            self._drive_axis("ly", ly, self._sl_dpc_qsref, lambda v: v * 1000.0)
        else:
            self._drive_axis("lx", lx, self._sl_vc_wmref,
                             lambda v: self._bipolar_to_range(v, 0.0, 30.0))
            self._drive_axis("ly", ly, self._sl_vc_tanphi, lambda v: v * 1.0)

        # ---- RStick X → speed_factor (log10, -2 → +2 = 0.01× → 100×) ----
        rx = a.get("rx", 0.0)
        self._drive_axis("rx", rx, self._speed_slider, lambda v: v * 2.0)

    # ---- per-frame tick (vsync from MasterPlotGL.frame_tick) ----
    def _gui_tick(self):
        self._gui_counter += 1

        # FPS tracking (EWMA)
        now = time.perf_counter()
        if self._ui_fps_last > 0:
            ft = now - self._ui_fps_last
            if ft > 0:
                fps_inst = 1.0 / ft
                self._ui_fps_ewma = 0.9 * self._ui_fps_ewma + 0.1 * fps_inst
        self._ui_fps_last = now
        self.engine.stats["ui_fps"] = self._ui_fps_ewma

        # Snapshot + propagate to plots
        snap, st, stats, ctrl, prm = self.engine.snapshot_for_render()
        # Attach derived magnitudes as numpy views on snap so the t-plot and
        # composer treat them as first-class signals. Cheap: hypot is vectorised
        # and the arrays are at most MAX_PTS long.
        if len(snap.t):
            snap.ism  = np.hypot(snap.isd, snap.isq)
            snap.irm  = np.hypot(snap.ird, snap.irq)
            snap.phis = np.hypot(snap.psd, snap.psq)
            snap.phir = np.hypot(snap.prd, snap.prq)
            # f_slip = f_s - n_p · ω_m   (snap.wm is in giri/s mech already)
            snap.fslip = ctrl["fs"] - prm[5] * snap.wm
        else:
            empty = np.empty(0, dtype=np.float64)
            snap.ism = snap.irm = snap.phis = snap.phir = snap.fslip = empty
        self._render_snap = snap
        self._render_state = st
        self._render_stats = stats
        self._render_ctrl = ctrl
        self._render_params = prm

        # Repaint sibling plots (master will repaint itself)
        self._da_flux.update()
        self._da_pq.update()
        self._da_sat.update()
        for d in self._plot_drawing_areas:
            d.update()

        # Throttle text labels: ~20 Hz at 120 Hz vsync
        if self._gui_counter % 6 != 0:
            return
        self._refresh_labels(snap, st, stats, ctrl, prm)

    def _refresh_labels(self, snap, st, stats, ctrl, prm):
        # Runtime parameters
        Rs_p = prm[0]; Rr_p = prm[1]
        Ls_p = prm[2]; Lr_p = prm[3]; M_p = prm[4]
        NP_p = prm[5]; J_p = prm[6]; B_p = prm[7]
        D_p = Ls_p * Lr_p - M_p * M_p

        t_now = st[6]
        comp = "JIT pronto" if stats.get("compiled") else "compilazione JIT…"
        self._time_lbl.setText(htm(
            f"Park sincrono · t = {t_now:.2f} s · {comp}", color="#475569"))

        ws_now = 2 * math.pi * ctrl["fs"]
        wm_now = st[4]
        fslip = (ws_now - NP_p * wm_now) / (2 * math.pi) if ws_now > 0 else 0.0
        self._fslip_lbl.setText(htm(f"f_slip = {fslip:.2f} Hz", color="#78716c"))

        # Instantaneous values from current state
        psd, psq, prd, prq = st[0], st[1], st[2], st[3]
        isd = (Lr_p*psd - M_p*prd) / D_p
        isq = (Lr_p*psq - M_p*prq) / D_p
        ird = (Ls_p*prd - M_p*psd) / D_p
        irq = (Ls_p*prq - M_p*psq) / D_p
        Ce = NP_p * M_p * (isq*ird - isd*irq)
        # Compute P_s, Q_s in MOTOR convention first (they're used by the DPC
        # comparison below, which lives in motor convention internally), then
        # flip sign to generator convention before display.
        Ps_motor = ctrl["Vs"] * isd
        Qs_motor = -ctrl["Vs"] * isq
        mode_str = ctrl.get("mode", "open")
        if mode_str == "dpc":
            hP = ctrl.get("h_P", 500.0); hQ = ctrl.get("h_Q", 500.0)
            Vdc = ctrl.get("Vdc_dpc", 100.0)
            # ctrl["Ps_ref"] / ctrl["Qs_ref"] are stored in motor convention
            # (the UI flips the sign at write time), so compare directly.
            eP = ctrl.get("Ps_ref", 0.0) - Ps_motor
            eQ = ctrl.get("Qs_ref", 0.0) - Qs_motor
            sP = -1.0 if eP > hP else (+1.0 if eP < -hP else 0.0)
            sQ = +1.0 if eQ > hQ else (-1.0 if eQ < -hQ else 0.0)
            vrd = Vdc * sP; vrq = Vdc * sQ
        elif mode_str == "vc":
            ws_safe = ws_now if abs(ws_now) > 1e-3 else 1e-3
            alpha = (Ls_p*Lr_p - M_p*M_p) / Ls_p
            slip_w = ws_now - NP_p*wm_now
            vrd = -slip_w * alpha * irq + slip_w * M_p * ctrl["Vs"] / (Ls_p * ws_safe)
            vrq = +slip_w * alpha * ird
        else:
            a = 2*math.pi*ctrl["fr"]*t_now - ws_now*t_now + NP_p*st[5]
            vrd = ctrl["Vr"]*math.cos(a); vrq = ctrl["Vr"]*math.sin(a)
        # Generator convention for displayed quantities.
        Ps = -Ps_motor
        Qs = -Qs_motor
        Pr = -(vrd*ird + vrq*irq)
        Qr = -(vrq*ird - vrd*irq)
        PRs = Rs_p*(isd*isd + isq*isq)
        PRr = Rr_p*(ird*ird + irq*irq)
        slip = (ws_now - NP_p*wm_now)/ws_now*100 if ws_now > 0 else 0.0
        Cl_now = ctrl.get("Cl", 0.0)
        Pmech = -Cl_now * wm_now      # >0 when shaft drives (gen convention)
        Pem   = -Ce * wm_now          # >0 when machine generates (gen convention)
        Pfric = B_p * wm_now * wm_now
        Ploss = PRs + PRr + Pfric
        dKEdt = wm_now * (Ce - B_p*wm_now - Cl_now)

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
            "Pmech": Pmech / 1e3,
            "Pem":   Pem   / 1e3,
            "Pfric": Pfric / 1e3,
            "Ploss": Ploss / 1e3,
            "dKEdt": dKEdt / 1e3,
            # Derived magnitudes — first-class signals in the composer.
            "ism":   math.hypot(isd, isq),
            "irm":   math.hypot(ird, irq),
            "phis":  math.hypot(psd, psq) * 1e3,
            "phir":  math.hypot(prd, prq) * 1e3,
            "fslip": fslip,
        }
        # Saturation warning (only when the linear model is active): if
        # |ψ_s| crosses the threshold we recolor the four "magnitude" labels
        # so the user sees they're operating in a regime where saturation
        # *would* matter if turned on.
        psi_s_mag = math.hypot(psd, psq)
        psi_s_sat = float(prm[9]) if len(prm) > 9 else 0.0
        warn_color = None
        if (not self._sat_enabled) and psi_s_sat > 0.0:
            sigma = psi_s_mag / psi_s_sat
            if sigma > 1.5:
                warn_color = "#ef4444"
            elif sigma > 1.0:
                warn_color = "#eab308"
        warn_keys = ("phis", "phir", "ism", "irm")
        for k, vl in self._value_labels.items():
            v = sig_values.get(k, 0.0)
            if k in warn_keys and warn_color is not None:
                vl.setText(htm(fmt_value(v), color=warn_color, bold=True))
            else:
                vl.setText(htm(fmt_value(v), bold=True))

        # Header RT-factor indicator
        rt_factor = stats.get("rt_factor", 0.0)
        speed_now = ctrl.get("speed_factor", 1.0)
        sps = stats.get("steps_per_s", 0.0)
        dt_eff = stats.get("dt_eff_us", 0.0)
        if rt_factor <= 0.001 or not stats.get("running", False):
            rt_col_hdr = "#475569"; rt_text = "—"; sub_text = "in pausa"
        else:
            if rt_factor >= 0.97:
                rt_col_hdr = "#22c55e"
                sub_text = (f"target {speed_now:.2f}× wall · "
                            f"{fmt_value(sps)} step/s · dt={dt_eff:.1f}μs")
            elif rt_factor >= 0.5:
                rt_col_hdr = "#eab308"
                sub_text = f"in ritardo {(1.0/rt_factor):.2f}× sul target {speed_now:.2f}× wall"
            else:
                rt_col_hdr = "#ef4444"
                sub_text = (f"sim non regge target {speed_now:.2f}× wall "
                            f"(ritardo {(1.0/rt_factor):.1f}×)")
            rt_text = f"{rt_factor:.2f}×"
        self._rt_main.setText(htm(rt_text, color=rt_col_hdr, size="xx-large", bold=True))
        self._rt_sub.setText(htm(sub_text, color=rt_col_hdr, size="x-small"))

    def closeEvent(self, ev):
        self._gp_timer.stop()
        self._gamepad.shutdown()
        self.engine.shutdown()
        super().closeEvent(ev)


# ============================================================
# Entry point
# ============================================================

# Stylesheet — replicates the dark GTK theme with Qt selectors.
APP_STYLESHEET = """
QMainWindow, QScrollArea, QWidget#central { background-color: #060a12; }
QScrollArea > QWidget > QWidget { background-color: #060a12; }
QFrame { background-color: #0a1018; border: 1px solid #1a2030; border-radius: 4px; }
QLabel { color: #e2e8f0; background: transparent; }
QPushButton {
    background-color: #1a2030; color: #e2e8f0;
    border: 1px solid #2a3040; padding: 4px 10px; border-radius: 3px;
    font-family: monospace;
}
QPushButton:hover { background-color: #2a3040; }
QPushButton:checked { background-color: #1e40af; border-color: #3b82f6; color: #f1f5f9; }
QPushButton:pressed { background-color: #2a3040; }
QLineEdit {
    background-color: #0f1420; color: #e2e8f0;
    border: 1px solid #2a3040; padding: 2px 4px; border-radius: 2px;
    font-family: monospace;
}
QSlider::groove:horizontal { height: 4px; background: #1a2030; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #e2e8f0; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}
QCheckBox { color: #cbd5e1; spacing: 4px; background: transparent; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #2a3040; background: #0f1420; border-radius: 2px; }
QCheckBox::indicator:checked { background: #3b82f6; border-color: #3b82f6; }
QScrollBar:vertical { background: #0a1018; width: 10px; }
QScrollBar::handle:vertical { background: #2a3040; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; height: 0; }
"""


def run():
    # Vsync requires SwapInterval=1 on the default surface format,
    # set BEFORE QApplication is constructed.
    fmt = QtGui.QSurfaceFormat()
    fmt.setSwapInterval(1)
    fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)

    engine = SimEngine()
    engine.start()

    win = DfigWindow(engine)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
