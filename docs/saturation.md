# Magnetic saturation — model and plot

This document explains the saturation model implemented in `dfig_engine.py`
and the diagnostic plot shown in the bottom-right of the central column of
the UI.

---

## 1. Why model saturation

In an ideal Park model the magnetizing inductance `L_m` is constant. This
gives:

- a perfectly linear flux↔current relationship: `ψ_m = L_m · i_m`,
- closed-form steady-state expressions (e.g. `i_m,nom ≈ V_s,n / (ω_s · L_m)`),
- numerically well-behaved ODEs (the matrix `[L_s, M; M, L_r]` is constant
  and time-invariant).

The price you pay is that the model is unphysical above ~1.0–1.2 × nominal
flux. Real iron cores saturate: as the flux density approaches the knee of
the B–H curve, the effective magnetizing inductance drops, the
magnetization current spikes, and the torque available for a given rotor
current decreases. Studies of overvoltage, fault ride-through, deep
under-/over-excitation, or any over-flux transient need at least a
first-order saturation model to be representative.

The simulator implements a smooth taper of `L_m` driven by stator flux
magnitude. It is opt-in via a UI toggle (`SATURAZIONE MAGNETICA`):

- **OFF** (default): the kernel runs the linear model. The UI shows a
  visual warning on `|i_s|`, `|i_r|`, `|φ_s|`, `|φ_r|` when the operating
  point would be above the saturation knee, but the model itself is not
  affected.
- **ON**: the kernel applies the taper. `L_m` becomes a function of the
  current flux state.

---

## 2. Choice of governing variable

The textbook "right" thing is to make `L_m` a function of the magnetizing
current `i_m = i_s + i_r` (in the dq frame), because magnetic saturation
is fundamentally a property of how much current the iron core has to carry
to set up its flux. That choice creates a chicken-and-egg situation,
though: solving for `i_s, i_r` from the state `ψ_s, ψ_r` requires
inverting the inductance matrix, whose entries depend on `L_m`, which
depends on `i_m`, which depends on `i_s, i_r`. You either need a
fixed-point iteration at every RHS evaluation or you accept a one-step
lag.

We side-step that by driving the taper from `|ψ_s|` instead of `|i_m|`.
The two are tightly coupled (`|ψ_s| ≈ |ψ_m|` for small leakage, and
`|ψ_m| = L_m · |i_m|`) but `ψ_s` is a **state variable** of the
integration: known at the start of each RHS evaluation, no recursion.
For machines with leakage in the typical band (`L_ls/L_m ≈ 0.03–0.05`)
the two are almost interchangeable and the simplification is invisible.
For pathological cases (heavy leakage, very low `L_m`) the model is less
accurate but still stable and monotonic.

---

## 3. Saturation curve

The taper is a smooth, monotonically-decreasing rational function:

```
σ = |ψ_s| / ψ_s,sat
                       ⎧ 1                          if σ ≤ 1
L_m,eff = L_m₀ · f(σ), ⎨
                       ⎩ 1 / (1 + (σ − 1)²)         if σ > 1
```

Key properties:

- `f(1) = 1`: continuous and `C¹` at the knee (no kink in the derivative).
- `f' (1) = 0`: the curve is tangent to the linear region at the threshold.
- `f(σ) → 0⁺` as `σ → ∞`: smooth approach to zero, no negative values.
- `f(2) = 0.5`: at twice the nominal flux, `L_m` is halved.
- `f(3) = 0.2`: at three times, it's reduced to one-fifth.

This is intentionally simpler than a curve-fitted B–H lookup or a
Brauer/Fröhlich model. It captures the right qualitative behavior (smooth
knee, asymptotic approach to zero, scale-invariant in `σ`) without
introducing per-machine fitting parameters that would have to ship with
each preset.

### Threshold default

`ψ_s,sat = 1.2 × ψ_s,nom = 1.2 × V_n / (2π · f_s)` for the active machine
preset. For the default 1 MVA preset (`V_n = 690 V, f_s = 50 Hz`):

```
ψ_s,nom = 690 / (2π · 50) ≈ 2.196 Wb
ψ_s,sat = 1.2 × 2.196     ≈ 2.636 Wb
```

The 1.2× factor is conventional: real machines are sized so that at
nominal voltage and frequency they sit comfortably below the knee
(typically 0.85–0.95 × ψ_s,sat). The slider in the UI lets you move the
threshold up to 2.5 × ψ_s,nom or down to 0.5 × ψ_s,nom; the range
re-centers automatically when you switch preset.

---

## 4. How it's applied in the kernel

The taper enters every kernel that needs the inductance matrix
(`deriv`, `observe`, `_compute_dpc`, `_compute_vc`). All of these now
recompute `M_eff, L_s, L_r, D` at the start of the function:

```python
f_sat = _lm_eff_factor(psd, psq, params[P_PSI_SAT], params[P_SAT_EN])
M_    = M0 * f_sat            # saturated mutual
Lls   = Ls0 - M0              # leakage stays constant
Llr   = Lr0 - M0
Ls    = Lls + M_              # saturated stator self-inductance
Lr    = Llr + M_              # saturated rotor self-inductance
D     = Ls * Lr - M_ * M_     # determinant for matrix inversion
```

`Ls0, Lr0, M0` are the user-edited parameters (or the preset values).
`Lls, Llr` are derived once from `L_s − M` and `L_r − M`; they are
**not** affected by saturation, only the mutual is. `Ls, Lr, D` are
recomputed every time `M_eff` changes.

This means:

- `i_s = (Lr · ψ_s − M_ · ψ_r) / D` uses the saturated `Lr` and `M_`.
- The torque `Ce = n_p · M_ · (i_sq · i_rd − i_sd · i_rq)` uses the
  saturated `M_`, so torque output drops in saturation.
- Stator and rotor electrical equations carry the saturated mutual
  through the cross-coupling terms (`ω_s · ψ_sq` etc. are unaffected,
  but `ω_s · ψ_sd` enters via the matrix inversion).

When `params[P_SAT_EN] < 0.5` (= toggle OFF), `_lm_eff_factor` short-
circuits to `1.0` and the kernel runs the linear model (no per-step
overhead — it's a single comparison plus an early return).

### What is NOT modeled

- **Hysteresis losses** — the curve has no memory; `f(σ)` is the same on
  rising and falling flux.
- **Eddy current losses** — no frequency-dependent core losses. If you
  push the simulator into transients with high `dψ/dt`, the iron-loss
  energy is not accounted for.
- **Cross-saturation** between d and q axes — the taper is isotropic
  (driven by `|ψ_s|` only), no anisotropic effects.
- **Skin effect** in the rotor windings.
- **Saturation of the leakage paths** — `L_ls, L_lr` are constants. In
  reality slot leakage saturates at high overload; we ignore that.

These omissions are conscious: the goal is a teaching tool that shows
the qualitative effect of `L_m` collapsing as the iron saturates, not a
finite-element reference model.

---

## 5. The plot — visual elements

The bottom-right tile in the central column is the saturation diagnostic:

```
┌── SATURAZIONE ────────  ON · σ=1.32 (saturazione) · L_m/L_m0=0.911 ──┐
│ L_m/L_m0                                                              │
│                                                                       │
│   1.0 ──────●────╮    ← linear region: f(σ) = 1                       │
│                  ╲╲                                                   │
│                   ╲                                                   │
│                    ╲   curva teorica f(σ)                             │
│   0.5               ╲╲                                                │
│                       ╲╲                                              │
│                          ──────_                                      │
│   0.0 ──────────┬──────────────────────→  |ψ_s| [Wb]                  │
│           ψ_sat=2.64                                                  │
│                  ↑ vertical dashed line at threshold                  │
└───────────────────────────────────────────────────────────────────────┘
```

### Axes

- **Horizontal**: `|ψ_s|` in Wb. Range `[0, 2.5 × ψ_s,sat]` — fixed;
  scales with the threshold so the entire knee + tail is always visible.
- **Vertical**: dimensionless ratio `L_m,eff / L_m₀` ∈ `[0, 1]`. Universal
  (independent of preset). Tick labels at 0.0, 0.5, 1.0.

### Static elements

- **The taper curve** (orange, solid): drawn over the full X range,
  always visible regardless of the toggle state. Shows where the
  operating point would land if saturation were active.
- **Threshold line** (orange, dashed vertical): drops at `|ψ_s| = ψ_s,sat`.
  Re-positions in real time when the user moves the threshold slider.
- **Grid**: gray horizontal at Y = 0, 0.25, 0.5, 0.75, 1.0; vertical at
  X = 0.5, 1.0, 1.5, 2.0 × ψ_s,sat.

### The operating-point marker (the key bit)

A circle of radius 5 px at `(|ψ_s|_now, op_y_now)`:

| Toggle  | Region            | Position of marker             |
|---------|-------------------|--------------------------------|
| OFF     | σ ≤ 1 (linear)    | on Y = 1, X = `|ψ_s|`          |
| OFF     | σ > 1 (over-flux) | **still** on Y = 1 (kernel is linear), X = `|ψ_s|` |
| ON      | σ ≤ 1             | on Y = 1, X = `|ψ_s|`          |
| ON      | σ > 1             | on the curve, Y = `f(σ)`       |

In OFF mode the marker stays on the horizontal line Y = 1 even when you
push the operating point past the threshold — the kernel does not
saturate, the curve is shown only as a "what-if". In ON mode the marker
slides down the curve as `|ψ_s|` increases past `ψ_s,sat`.

The marker fill is always white. Its **border color** indicates the
region:

- `gray` (`#94a3b8`) when `σ ≤ 1`: linear region, "safe".
- `orange` (`#f97316`) when `σ > 1`: above the knee, saturating
  territory.

This means at a glance you can see whether you are above or below the
threshold, regardless of whether the kernel is actually applying the
taper.

### The legend (top-right)

A single line that summarizes the current state:

```
ON  ·  σ=1.32 (saturazione)  ·  L_m/L_m0=0.911
```

Components, left to right:

- `ON` / `OFF` — current state of the kernel toggle.
- `σ = …` — normalized flux. Same as `|ψ_s| / ψ_s,sat`.
- `(saturazione)` / `(lineare)` — region label, derived from σ.
- `L_m/L_m0 = …` — value of the taper factor at the operating point.
  In OFF or sub-threshold this is always `1.000`.

The whole legend turns orange when `σ > 1`, so it's visible at a glance
even from across the room.

---

## 6. How to drive the operating point into saturation

For the default 1 MVA / 690 V preset, nominal regime gives `σ ≈ 0.83`.
You will not see the marker move along the inclined part of the curve
unless you push the machine. Three ways to do it:

### a) Lower the threshold

Drag the `ψ_s,sat [Wb]` slider down. From 2.64 Wb default to e.g. 1.8 Wb:

```
σ = 2.20 / 1.8 = 1.22 → f(σ) = 1 / (1 + 0.22²) = 0.954
```

You see the marker climb past the dashed line. With ON enabled, the
kernel starts applying the taper.

### b) Raise V_s

Move `|V_s|` from 690 V to e.g. 1100 V. At regime, stator flux follows
voltage:

```
ψ_s ≈ V_s / (2π · f_s) = 1100 / 314.16 ≈ 3.50 Wb
σ = 3.50 / 2.64 = 1.32 → f(σ) = 1 / (1 + 0.32²) = 0.907
```

Same effect, more physical (this is what happens during an over-voltage
event in a real grid).

### c) Lower f_s

`ψ_s,nom = V_n / (2π · f_s)`, so dropping `f_s` raises the nominal flux
and pushes you into saturation at fixed voltage. From 50 Hz to 35 Hz at
V_s = 690 V:

```
ψ_s ≈ 690 / (2π · 35) ≈ 3.14 Wb
σ = 3.14 / 2.64 = 1.19 → f(σ) = 0.967
```

This is also the V/f ratio violation — a known failure mode in motor
drives.

### Cross-check the effect on torque

When you cross into saturation with the kernel ON, watch:

- `|i_s|` and `|i_r|` (composer SEGNALI): both rise. The machine has to
  draw more current to maintain the same flux as `L_m` shrinks.
- `C_em` (composer SEGNALI): drops slightly for the same reference. The
  electromechanical torque is `n_p · M_eff · (i_sq · i_rd − i_sd · i_rq)`,
  so a smaller `M_eff` means less torque per ampere.
- `P_s` (composer SEGNALI, generator convention): drops for the same
  rotor-side excitation. The active power transfer through the air gap
  shrinks because the coupling weakens.

The effects are subtle near σ = 1.0 (a few percent) and become large
above σ = 1.5.

---

## 7. Default behavior on RESET

Pressing `RESET` puts the saturation toggle back to **OFF (warning)**
along with the rest of the macro state. The threshold `ψ_s,sat` is
reset to `1.2 × ψ_s,nom` for the **currently selected preset** (so it
correctly tracks any preset change you may have done before). The
slider range and step are also rescaled.

---

## 8. Files involved

| File              | Role                                           |
|-------------------|------------------------------------------------|
| `dfig_engine.py`  | Defines `_lm_eff_factor` (Numba kernel), `P_SAT_EN, P_PSI_SAT, SAT_EN_DEFAULT, PSI_S_SAT_DEFAULT`. Applies the taper in `deriv`, `observe`, `_compute_dpc`, `_compute_vc`. |
| `dfig_qt.py`      | `draw_saturation` (the plot). UI: toggle button + threshold slider in the `SATURAZIONE MAGNETICA` sub-panel of `PARAMETRI MACCHINA`. Warning colors on `|i_s|, |i_r|, |φ_s|, |φ_r|` when OFF and σ > 1. |

The model is **purely additive**: with `params[P_SAT_EN] = 0` the
kernel produces bit-identical output to the pre-saturation version, so
all earlier benchmarks and validation runs remain valid.
