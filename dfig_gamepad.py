"""Polled XInput-style gamepad reader (pygame backend, no display).

Used by the Qt UI: a QTimer ticks `poll()` ~30 Hz, the returned dict drives
sliders and buttons. Edge-detected button presses, deadzoned analog sticks,
auto-repeating D-Pad. Falls back to a no-op stub if pygame isn't installed
or no controller is plugged in.
"""

import os
import time

# Don't open an SDL window — we only want joystick input.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False


# Standard Xbox button mapping under SDL2/pygame on every platform we target.
_XBOX_BTN = {0: "A", 1: "B", 2: "X", 3: "Y",
             4: "LB", 5: "RB",
             6: "BACK", 7: "START",
             8: "L3", 9: "R3"}

DEADZONE = 0.12
DPAD_REPEAT_HZ = 5.0


class GamepadController:
    """Wrap a single (the first) connected gamepad.
    `available` is False if pygame is missing, no controller is plugged in,
    or pygame init failed. In that case `poll()` returns {}."""

    def __init__(self):
        self.available = False
        self.name = "—"
        self._js = None
        self._prev_buttons = {}
        self._dpad_last_fire = {}
        if not _PYGAME_OK:
            return
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._js = pygame.joystick.Joystick(0)
                self._js.init()
                self.name = self._js.get_name()
                self.available = True
        except Exception:
            self.available = False

    def rescan(self):
        """Re-attempt to attach to a controller (e.g. after hot-plug)."""
        if not _PYGAME_OK or self.available:
            return
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._js = pygame.joystick.Joystick(0)
                self._js.init()
                self.name = self._js.get_name()
                self.available = True
        except Exception:
            self.available = False

    @staticmethod
    def _dz(v):
        """Apply radial deadzone, then rescale [DZ, 1] → [0, 1]."""
        if abs(v) < DEADZONE:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - DEADZONE) / (1.0 - DEADZONE)

    def poll(self):
        """Return a dict of current actions.

        Sticks: 'lx', 'ly', 'rx', 'ry' in [-1, 1] (deadzoned, Y up = +1).
        Triggers: 'lt', 'rt' in [0, 1] (rest = 0).
        Buttons (edge-detected, True only on the press tick):
            'A', 'B', 'X', 'Y', 'LB', 'RB', 'BACK', 'START', 'L3', 'R3'.
        D-Pad (auto-repeat at DPAD_REPEAT_HZ when held):
            'D_UP', 'D_DOWN', 'D_LEFT', 'D_RIGHT'.
        """
        if not self.available:
            return {}
        try:
            pygame.event.pump()
        except Exception:
            self.available = False
            return {}

        out = {}
        n_axes = self._js.get_numaxes()

        # Sticks. Y axes are inverted so up = +1 (HID convention has down = +).
        if n_axes > 0: out["lx"] = self._dz(self._js.get_axis(0))
        if n_axes > 1: out["ly"] = self._dz(-self._js.get_axis(1))
        if n_axes > 3: out["rx"] = self._dz(self._js.get_axis(3))
        if n_axes > 4: out["ry"] = self._dz(-self._js.get_axis(4))

        # Triggers. SDL2 convention: trigger axes range -1..+1 with rest = -1.
        # Remap to 0..1.
        if n_axes > 2:
            out["lt"] = max(0.0, (self._js.get_axis(2) + 1.0) / 2.0)
        if n_axes > 5:
            out["rt"] = max(0.0, (self._js.get_axis(5) + 1.0) / 2.0)
        # Debug: tutti gli assi raw + numero. Permette di mappare trigger
        # diversi su controller non-standard (es. DualShock, generici HID).
        out["_naxes"] = n_axes
        out["_axes_raw"] = tuple(self._js.get_axis(i) for i in range(n_axes))

        # Buttons (edge detection: True only on the rising edge).
        for i, name in _XBOX_BTN.items():
            if i >= self._js.get_numbuttons():
                continue
            cur = bool(self._js.get_button(i))
            prev = self._prev_buttons.get(name, False)
            out[name] = cur and not prev
            self._prev_buttons[name] = cur

        # D-Pad (hat) with auto-repeat — emit on first press, then every
        # 1/DPAD_REPEAT_HZ seconds while held.
        if self._js.get_numhats() > 0:
            hx, hy = self._js.get_hat(0)
            now = time.perf_counter()
            period = 1.0 / DPAD_REPEAT_HZ
            for key, held in (("D_UP",    hy > 0),
                              ("D_DOWN",  hy < 0),
                              ("D_LEFT",  hx < 0),
                              ("D_RIGHT", hx > 0)):
                if held:
                    last = self._dpad_last_fire.get(key, 0.0)
                    if last == 0.0 or (now - last) >= period:
                        out[key] = True
                        self._dpad_last_fire[key] = now
                else:
                    self._dpad_last_fire[key] = 0.0

        return out

    def shutdown(self):
        if not _PYGAME_OK:
            return
        try:
            if self._js is not None:
                self._js.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
