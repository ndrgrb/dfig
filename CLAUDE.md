# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

An interactive, real-time **DFIG** (Doubly-Fed Induction Generator) simulator
for teaching electrical-machines / power-conversion courses. It integrates the
7-state Park (dq) model of the machine in a background thread and renders the
state live (dq-plane phasors, time plots, gauges, numerical diagnostics) in a
desktop GUI. The user can change machine parameters, grid/rotor excitation, load
torque, the integrator, and the rotor-side control law on the fly.

The project is **educational and pedagogical**. The UI, documentation, and
commit messages are written in **Italian**; code identifiers and most inline
comments are English (with some Italian comments). Preserve this convention when
editing — match the surrounding language of whatever you touch.

## Layout

```
dfig_engine.py    UI-agnostic simulation core (numba JIT kernels + SimEngine thread)
dfig_qt.py        PySide6 (Qt6) + OpenGL GUI — the application entry point
dfig_gamepad.py   Optional pygame gamepad reader (graceful no-op stub if absent)
README.md         Long-form Italian lab manual (physics derivations + study cases)
docs/saturation.md  Deep-dive on the optional magnetic-saturation model
```

There is **no build system, no test suite, no CI, no requirements.txt, no
packaging**. These are three runnable Python scripts. Don't invent a framework
the project doesn't have; if you add tooling, keep it minimal and justified.

## Running

```bash
python3 dfig_qt.py
```

Dependencies (system Python, no virtualenv assumed): `numpy`, `numba`,
`PySide6`, and optionally `pygame` (gamepad — the app runs fine without it).

> Note: `README.md` §Installazione is **stale** — it references `dfig.py` and a
> GTK4 stack. The real stack is **PySide6 + QOpenGLWidget + QPainter** and the
> entry point is `dfig_qt.py`. Trust the source, not that section of the README.

First launch is slow: numba AOT-compiles the JIT kernels (`warmup_jit()` runs on
the sim thread). Compiled kernels are cached (`cache=True`) so later launches are
fast. There is no headless mode — the engine thread runs, but the GUI needs a
display.

## Architecture

Two decoupled threads communicating through a `threading.Lock`:

```
SimEngine thread (dfig_engine)          GUI thread (dfig_qt)
  numba @njit nogil kernels               PySide6 event loop
  advance_rk45/rk4/euler                  MasterPlotGL.paintGL drives the tick:
  writes into a numpy ring buffer   ──►     snapshot_for_render() (one memcpy)
  real-time locked to wall clock            redraw all QOpenGLWidget plots
                                            refresh labels
```

- The sim thread integrates as fast as needed and **sleeps to lock sim-time to
  wall-time** (`speed_factor` multiplies wall). "RT factor" in the diagnostics
  should sit at 1.00 when locked; "headroom" is the spare CPU margin.
- `SimEngine.snapshot_for_render()` returns a zero-garbage snapshot
  (`SimpleNamespace` with numpy views per field) — the GUI never blocks the sim
  for long.
- The GUI is paced by vsync: `MasterPlotGL.paintGL` is the per-frame driver, and
  `QSurfaceFormat.setSwapInterval(1)` (set before `QApplication`) provides the
  clock. Other plots `update()` from that tick.

### `dfig_engine.py` — the model

- **State vector `s ∈ ℝ⁷`**: `[ψ_sd, ψ_sq, ψ_rd, ψ_rq, ω_m, θ_m, t]` (fluxes Wb,
  speed rad/s mech, angle rad, time s). Stator-voltage-oriented dq frame:
  `V_sd = |V_s|`, `V_sq = 0`.
- `deriv(...)` is the RHS (voltage equations + torque + mechanical balance);
  `observe(...)` derives all reported quantities (currents, P/Q, slip, power
  balance, dq and abc rotor voltages) from the state.
- Integrators: `advance_euler`, `advance_rk4`, `advance_rk45` (Dormand-Prince
  5(4), adaptive step, error-controlled — the default, `rtol=1e-9`).
- **Control modes** (selected via `ctrl["mode"]`): `"open"` (open-loop rotor
  voltage sinusoid at `f_r`), `"dpc"` (direct power control, bang-bang
  hysteresis), `"vc"` (vector control, stator-flux-oriented, with a speed PI
  outer loop and current PI inner loops). In DPC/VC the rotor voltage is sampled
  once per step (zero-order hold) so the RHS stays Lipschitz within a step and
  the adaptive `dt` doesn't collapse.
- **Machine presets** (`MACHINE_PRESETS`, in per-unit) resolve to SI via
  `preset_to_si()`; the GUI loads them and rescales sliders.
- **Optional magnetic saturation**: `_lm_eff_factor` tapers `L_m` as a function
  of `|ψ_s|` past a knee. Off by default (warning-only in UI). See
  `docs/saturation.md`.

### `dfig_qt.py` — the GUI

`DfigWindow(QMainWindow)` builds everything in `_build_ui`. Plots are
`QOpenGLWidget`s whose `paintGL` calls a free-function drawer (`draw_dq`,
`draw_tplot`, `draw_saturation`) using `QPainter`. Custom widgets: `Slider`,
`ParamInput`, `panel(...)`. Features include an **autopilot** (scripted VC
scenarios in `AUTOPILOT_SCENARIOS`) and **gamepad** drive (`_gamepad_tick`).
`run()` is the entry point.

## Cross-module contracts (read before editing)

These index/layout constants are a **hard ABI between the engine and the UI** —
the numba kernels index arrays positionally. Changing order or count requires
updating every consumer in lockstep:

- `PARAMS_*` indices + `NPARAMS` — the machine-parameter array `params[]`
  (the `P_RS, P_RR, ...` constants). UI reads/writes by these indices.
- `CTRL_*` indices + `NCTRL` and the `ctrl_arr` built in `SimEngine._loop` —
  the control array passed to kernels.
- `H_*` indices + `NH` + `HIST_FIELDS` — the history ring-buffer columns.
  `HIST_FIELDS` order **must** match the `out[...]` writes in `observe()` and the
  `snapshot_for_render()` field mapping.
- `SIGNALS_LIST` / `SIG_BY_KEY` — plottable-signal catalogue (key, label, unit,
  RGB). Derived magnitudes (`|i_s|`, `f_slip`, …) are computed in the UI's
  `_gui_tick`, not stored in history.

### Sign convention

Reported powers use the **generator convention**: positive `P_s, P_r, Q_s, Q_r`
= delivered *to* the grid; positive `P_mech` = shaft drives the machine. The
internal control kernels work in motor convention and the UI inverts setpoints
(`Ps_ref`, `Qs_ref`) before passing them in. Keep this consistent — a flipped
sign here silently makes the whole UI lie.

## Conventions

- **Commit messages**: Conventional-Commits style with an Italian subject, e.g.
  `feat(signals): ...`, `ui(plot): ...`, `fix: ...`, `docs: ...`. Scope tags seen
  in history: `signals`, `plot`, `slider`, `vc`, `ui`. Keep subjects short and in
  Italian to match the log.
- **Style**: plain stdlib + numpy + numba; no type hints, no formatter config.
  Match existing spacing and the mixed-language commenting already in each file.
- Numba kernels are `@njit(cache=True, fastmath=True, nogil=True)`. Inside them,
  stick to scalar/loop code over typed numpy arrays — no Python objects, no
  unsupported numpy calls — or compilation fails. Test by actually running the
  app (or calling `warmup_jit()`).
- The engine must stay **UI-toolkit-free** (`dfig_engine.py` imports no Qt). Keep
  rendering and widgets in `dfig_qt.py`.

## Verifying changes

There are no automated tests. To validate a change:

1. `python3 dfig_qt.py` and confirm it launches without a numba compile error.
2. Press RUN; check the dq planes settle and DIAGNOSTICA NUMERICA shows
   `RT factor ≈ 1.00`, low `step rejet.`, `‖err‖∞ < 1`.
3. Sanity-check the physics against README §10 study cases (e.g. default shorted
   rotor → small permanent slip, ~222 kVAR magnetizing reactive).

If you can't run a GUI here, at minimum import-check both modules and call
`dfig_engine.warmup_jit()` to exercise the kernels.

## Repository note

`.gitignore` currently lists `CLAUDE.md` (and `.claude/`, `__pycache__/`). This
file is therefore committed with `git add -f`. Once tracked, normal edits to it
are no longer ignored. If the project wants CLAUDE.md tracked long-term, consider
removing it from `.gitignore`.
