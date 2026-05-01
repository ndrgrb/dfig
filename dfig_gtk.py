#!/usr/bin/env python3
"""
DFIG — Simulatore Interattivo (GTK4 + Cairo)
GUI thin layer on top of dfig_engine. Frame clock vsync via add_tick_callback.
"""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, Gdk
import cairo
import math
import time

import numpy as np

from dfig_engine import (
    P_RS, P_RR, P_LS, P_LR, P_M, P_NP, P_J, P_B, NPARAMS,
    PARAMS_DEFAULT, PARAMS_DEF,
    H_T, H_WM, H_CE, H_PS, H_QS, H_PR, H_SLIP,
    H_ISD, H_ISQ, H_IRD, H_IRQ,
    H_PSD, H_PSQ, H_PRD, H_PRQ,
    H_QR, H_PRS, H_PRR,
    H_PMECH, H_PEM, H_PFRIC, H_PLOSS, H_DKEDT,
    NH, HIST_FIELDS, SIGNALS_LIST, SIG_BY_KEY,
    MAX_PTS, LOG_INTERVAL, MIN_ITER_DT, MAX_ITER_DT,
    MAX_STEPS_PER_ITER, MAX_SAMPLES_PER_TICK, RESYNC_LAG,
    INT_EULER, INT_RK4, INT_RK45,
    NCTRL, C_MODE, C_PSREF, C_QSREF, C_HP, C_HQ, C_VDC,
    C_WMREF, C_PW, C_IW, C_PID, C_IID, C_PIQ, C_IIQ, C_TANPHI,
    MODE_OPEN, MODE_DPC, MODE_VC, CTRL_DEFAULT,
    NCS, CS_INT_ID, CS_INT_IQ, CS_INT_W, CS_T_LAST,
    SimEngine, warmup_jit,
)


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
               plot_idx, y_persist, cursor_state, stacked=False):
    """Plot tempo-serie con:
       - asse Y isteretico (si allarga, mai si stringe finché non si reset)
       - tick + grid sottile su entrambi gli assi
       - cursore verticale cross-plot + tooltip valori
       - tracce dinamiche dal catalogo SIG_BY_KEY
       - modo stacked: aree riempite cumulative con segno preservato
         (positivi sopra zero, negativi sotto). Per visualizzare bilanci
         di potenza dove la conservazione richiede ΣP = 0.
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
    if stacked:
        # In modo stacked: il range va dalla cum-neg (più bassa) alla cum-pos
        # (più alta) sommando ogni traccia con segno preservato.
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
    if stacked:
        # Stacked diverging: ogni traccia è una banda sopra/sotto il
        # cumulativo precedente in base al segno. La somma di tutte le
        # bande positive = -somma delle negative (conservazione).
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
            # Riempimento area
            cr.set_source_rgba(meta["r"], meta["g"], meta["b"], 0.55)
            cr.move_to(float(xs_arr[0]), float(ys_lo[0]))
            for k in range(n_t):
                cr.line_to(float(xs_arr[k]), float(ys_up[k]))
            for k in range(n_t - 1, -1, -1):
                cr.line_to(float(xs_arr[k]), float(ys_lo[k]))
            cr.close_path()
            cr.fill_preserve()
            # Bordo sottile
            cr.set_source_rgb(meta["r"], meta["g"], meta["b"]); cr.set_line_width(0.8)
            cr.stroke()
            sig_legend.append((key, meta))
        # Somma totale (linea bianca tratteggiata): mostra il residuo della
        # conservazione (in regime ≈ 0, in transitorio = dKE/dt netto).
        # Evidenzia visivamente quando il bilancio energetico chiude.
    else:
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

        # Toggle modo: open-loop ↔ DPC ↔ VC (vector control PF=1 + speed)
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        mode_lbl = Gtk.Label()
        mode_lbl.set_markup('<span font_family="monospace" foreground="#94a3b8" size="small">modo</span>')
        mode_lbl.set_halign(Gtk.Align.START); mode_lbl.set_hexpand(True)
        rb_open = Gtk.ToggleButton(label="open-loop")
        rb_dpc  = Gtk.ToggleButton(label="DPC")
        rb_vc   = Gtk.ToggleButton(label="VC PF=1")
        rb_dpc.set_group(rb_open); rb_vc.set_group(rb_open)
        rb_open.set_active(True)
        mode_box.append(mode_lbl); mode_box.append(rb_open)
        mode_box.append(rb_dpc); mode_box.append(rb_vc)

        # Sliders open-loop
        ol_vr = make_slider("|V_r|", 0, 0, 200, 0.5, lambda v: engine.set_ctrl(Vr=v))
        ol_fr = make_slider("f_r [Hz]", 0, -50, 50, 0.5, lambda v: engine.set_ctrl(fr=v))
        ol_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        ol_box.append(ol_vr); ol_box.append(ol_fr); ol_box.append(fslip_lbl)

        # Sliders DPC (P_s*/Q_s* in kW/kVAR — internamente convertiti in W/VAR)
        dpc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        dpc_psref = make_slider("P_s* [kW]",  0,  -1000, 1000, 5,
                                lambda v: engine.set_ctrl(Ps_ref=v * 1e3))
        dpc_qsref = make_slider("Q_s* [kVAR]", 0, -1000, 1000, 5,
                                lambda v: engine.set_ctrl(Qs_ref=v * 1e3))
        dpc_vdc   = make_slider("V_dc rotore [V]", 100, 10, 500, 5,
                                lambda v: engine.set_ctrl(Vdc_dpc=v))
        dpc_hp    = make_slider("banda h_P [kW]", 0.5, 0.05, 20, 0.05,
                                lambda v: engine.set_ctrl(h_P=v * 1e3))
        dpc_hq    = make_slider("banda h_Q [kVAR]", 0.5, 0.05, 20, 0.05,
                                lambda v: engine.set_ctrl(h_Q=v * 1e3))
        dpc_box.append(dpc_psref); dpc_box.append(dpc_qsref)
        dpc_box.append(dpc_vdc);   dpc_box.append(dpc_hp); dpc_box.append(dpc_hq)
        dpc_box.set_visible(False)

        # Sliders VC: riferimento velocità + target PF + guadagni PI
        vc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        wm_sync_giri = (2*math.pi*50.0/3.0) / (2*math.pi)  # ≈ 16.67 giri/s a 50 Hz, 3pp
        vc_wmref = make_slider("ω̃_m [giri/s]", wm_sync_giri, 0, 30, 0.1,
                               lambda v: engine.set_ctrl(wm_ref=v * 2*math.pi))

        # Slider tan(φ_s*): 0 = PF unitario, >0 = induttivo, <0 = capacitivo.
        # Label dinamico mostra cosφ corrispondente.
        vc_pf_lbl = Gtk.Label(); vc_pf_lbl.set_halign(Gtk.Align.START)
        def update_pf_lbl(tan_phi):
            cos_phi = 1.0 / math.sqrt(1.0 + tan_phi*tan_phi)
            sign = "ind" if tan_phi > 0.001 else ("cap" if tan_phi < -0.001 else "—")
            vc_pf_lbl.set_markup(
                f'<span font_family="monospace" foreground="#78716c" size="x-small">'
                f'cos(φ_s) = {cos_phi:.3f}  ({sign})</span>')
        update_pf_lbl(0.0)
        def on_pf_change(v):
            engine.set_ctrl(tan_phi=v)
            update_pf_lbl(v)
        vc_pf = make_slider("tan(φ_s*)  [Q_s/P_s]", 0.0, -1.0, 1.0, 0.05, on_pf_change)

        vc_pw = make_slider("P_w (vel.)", 1e4, 0, 5e4, 100,
                            lambda v: engine.set_ctrl(P_w=v))
        vc_iw = make_slider("I_w (vel.)", 1e5, 0, 5e5, 1000,
                            lambda v: engine.set_ctrl(I_w=v))
        vc_pid = make_slider("P_i_d", 0.1, 0, 5, 0.05,
                             lambda v: engine.set_ctrl(P_id=v))
        vc_iid = make_slider("I_i_d", 1.0, 0, 50, 0.1,
                             lambda v: engine.set_ctrl(I_id=v))
        vc_piq = make_slider("P_i_q", 1.0, 0, 50, 0.1,
                             lambda v: engine.set_ctrl(P_iq=v))
        vc_iiq = make_slider("I_i_q", 10.0, 0, 200, 1,
                             lambda v: engine.set_ctrl(I_iq=v))
        vc_box.append(vc_wmref)
        vc_box.append(vc_pf); vc_box.append(vc_pf_lbl)
        vc_box.append(vc_pw); vc_box.append(vc_iw)
        vc_box.append(vc_pid); vc_box.append(vc_iid)
        vc_box.append(vc_piq); vc_box.append(vc_iiq)
        vc_box.set_visible(False)

        def on_mode_toggled(_btn):
            if rb_vc.get_active():
                engine.set_ctrl(mode="vc")
                ol_box.set_visible(False); dpc_box.set_visible(False); vc_box.set_visible(True)
            elif rb_dpc.get_active():
                engine.set_ctrl(mode="dpc")
                ol_box.set_visible(False); dpc_box.set_visible(True);  vc_box.set_visible(False)
            else:
                engine.set_ctrl(mode="open")
                ol_box.set_visible(True);  dpc_box.set_visible(False); vc_box.set_visible(False)
        rb_open.connect("toggled", on_mode_toggled)
        rb_dpc.connect("toggled", on_mode_toggled)
        rb_vc.connect("toggled", on_mode_toggled)

        cbox.append(panel("ROTORE", "#ef4444", [mode_box, ol_box, dpc_box, vc_box]))

        # === CARICO ===
        # Due slider equivalenti che modificano lo stesso C_load:
        #   - "C_load [Nm]"  (coppia all'asse, range esteso ±50 kNm, step 10)
        #   - "P_mecc [kW]"  (potenza meccanica @ velocità di sincronismo)
        # Modifica programmatica di uno aggiorna anche l'altro (senza loop).
        sync_speed = 2 * math.pi * 50.0 / 3.0   # rad/s mech a fs=50 Hz, n_p=3
        cl_slider = make_slider("C_load [Nm]", 0, -50000, 50000, 10, lambda v: None)
        pm_slider = make_slider("P_mecc [kW] @ω_sync", 0, -5000, 5000, 1, lambda v: None)
        # Recupero gli oggetti Gtk.Scale interni per aggiornare il valore senza
        # riscatenare il callback (altrimenti loop infinito tra i due slider).
        cl_scale = cl_slider.get_last_child()
        pm_scale = pm_slider.get_last_child()
        cl_handler = [None]; pm_handler = [None]

        def on_cl(scale):
            v = scale.get_value()
            engine.set_ctrl(Cl=v)
            # aggiorno P_mecc senza scatenare il suo callback
            p_kw = -v * sync_speed / 1e3
            pm_scale.handler_block(pm_handler[0])
            pm_scale.set_value(p_kw)
            pm_scale.handler_unblock(pm_handler[0])

        def on_pm(scale):
            p_kw = scale.get_value()
            cl_nm = -p_kw * 1e3 / sync_speed
            engine.set_ctrl(Cl=cl_nm)
            cl_scale.handler_block(cl_handler[0])
            cl_scale.set_value(cl_nm)
            cl_scale.handler_unblock(cl_handler[0])

        cl_handler[0] = cl_scale.connect("value-changed", on_cl)
        pm_handler[0] = pm_scale.connect("value-changed", on_pm)

        cbox.append(panel("CARICO", "#22c55e", [cl_slider, pm_slider]))

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
            [],
            [],
            [],
        ]
        plot_stacked = [False] * plot_count
        plot_titles = [f"Plot {j+1}" for j in range(plot_count)]
        y_persist = {}
        cursor_state = {"x": None}
        plot_drawing_areas = []

        # Suddivisione del catalogo in due gruppi
        SCALAR_KEYS = ["wm", "Ce", "Ps", "Qs", "Pr", "Qr", "PRs", "PRr", "slip",
                       "Pmech", "Pem", "Pfric", "Ploss", "dKEdt"]
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
                # Larghezza fissa + testo right-anchored: niente reflow quando
                # il numero di cifre cambia (es. 9.99 → 10.0 → 12.3k).
                vl.set_width_chars(8); vl.set_max_width_chars(8); vl.set_xalign(1.0)
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

        # Riga di toggle "stacked" — uno per plot.  Se attivo, il plot disegna
        # le tracce come aree sovrapposte (positive sopra zero, negative sotto)
        # per visualizzare il bilancio di potenze.
        stack_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        stack_row.set_halign(Gtk.Align.CENTER)
        sl_lbl = Gtk.Label()
        sl_lbl.set_markup('<span font_family="monospace" foreground="#94a3b8" size="x-small">visualizz. stacked Σ</span>')
        stack_row.append(sl_lbl)
        for j in range(plot_count):
            cb_stk = Gtk.CheckButton(label=f"P{j+1}")
            def make_stk_handler(idx):
                def on_stk(btn):
                    plot_stacked[idx] = btn.get_active()
                    y_persist.pop(idx, None)
                    for d in plot_drawing_areas:
                        d.queue_draw()
                return on_stk
            cb_stk.connect("toggled", make_stk_handler(j))
            stack_row.append(cb_stk)

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
            vl.set_width_chars(8); vl.set_max_width_chars(8); vl.set_xalign(1.0)
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
        comp_box.append(stack_row)
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
                j, y_persist, cursor_state, stacked=plot_stacked[j])

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
            # Niente suffissi k/M: l'unità è già nella colonna u.m. Mostrare
            # "3.45k" insieme a "kW" si legge ambiguamente come "-3.45 kW"
            # quando in realtà è -3.45 MW. Mostra il numero per intero.
            av = abs(v)
            if av >= 10000:  return f"{v:.0f}"     # es: 12345
            if av >= 1000:   return f"{v:.1f}"     # es: 1234.5
            if av >= 100:    return f"{v:.2f}"     # es: 123.45
            if av >= 10:     return f"{v:.2f}"     # es:  12.34
            if av >= 1:      return f"{v:.3f}"     # es:   1.234
            if av >= 0.01:   return f"{v:.3f}"     # es:   0.012
            return f"{v:.4f}"                       # es:   0.0001

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
            mode_str = ctrl.get("mode", "open")
            if mode_str == "dpc":
                # Stessa logica del kernel: isteresi 3-livelli su (Ps, Qs)
                hP = ctrl.get("h_P", 500.0); hQ = ctrl.get("h_Q", 500.0)
                Vdc = ctrl.get("Vdc_dpc", 100.0)
                eP = ctrl.get("Ps_ref", 0.0) - Ps
                eQ = ctrl.get("Qs_ref", 0.0) - Qs
                sP = -1.0 if eP > hP else (+1.0 if eP < -hP else 0.0)
                sQ = +1.0 if eQ > hQ else (-1.0 if eQ < -hQ else 0.0)
                vrd = Vdc * sP; vrq = Vdc * sQ
            elif mode_str == "vc":
                # Stima visiva di v_r (solo reiettori): il PI integrato vive
                # nel kernel; qui basta un'indicazione qualitativa per Pr/Qr.
                ws_safe = ws_now if abs(ws_now) > 1e-3 else 1e-3
                alpha = (Ls_p*Lr_p - M_p*M_p) / Ls_p
                slip_w = ws_now - NP_p*wm_now
                vrd = -slip_w * alpha * irq + slip_w * M_p * ctrl["Vs"] / (Ls_p * ws_safe)
                vrq = +slip_w * alpha * ird
            else:
                a = 2*math.pi*ctrl["fr"]*t_now - ws_now*t_now + NP_p*st[5]
                vrd = ctrl["Vr"]*math.cos(a); vrq = ctrl["Vr"]*math.sin(a)
            Pr = vrd*ird + vrq*irq
            Qr = vrq*ird - vrd*irq
            PRs = Rs_p*(isd*isd + isq*isq)
            PRr = Rr_p*(ird*ird + irq*irq)
            slip = (ws_now - NP_p*wm_now)/ws_now*100 if ws_now > 0 else 0.0
            # Bilancio energetico (kW)
            Cl_now = ctrl.get("Cl", 0.0); J_p = prm[6]; B_p = prm[7]
            Pmech = -Cl_now * wm_now
            Pem   = Ce * wm_now
            Pfric = B_p * wm_now * wm_now
            Ploss = PRs + PRr + Pfric
            dKEdt = wm_now * (Ce - B_p*wm_now - Cl_now)

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
                "Pmech": Pmech / 1e3,
                "Pem":   Pem   / 1e3,
                "Pfric": Pfric / 1e3,
                "Ploss": Ploss / 1e3,
                "dKEdt": dKEdt / 1e3,
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
