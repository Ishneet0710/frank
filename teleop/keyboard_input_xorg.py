from __future__ import annotations

import os
from typing import Dict, Tuple, List

try:
    from Xlib import X, XK, display
except Exception as exc:  # pragma: no cover - optional dependency
    X = None  # type: ignore
    XK = None  # type: ignore
    display = None  # type: ignore


class KeyboardHandlerXorg:
    """Xorg-based keyboard handler to improve behavior on Linux/X11.

    Grabs specific keys globally via Xlib to ensure consistent behavior
    across window focus and keyboard layouts.
    """

    def __init__(self, key_states: Dict[str, bool]):
        if display is None:
            raise RuntimeError("python-xlib is not available")
        if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11":
            raise RuntimeError("Not running under Xorg/X11")

        self.key_states = key_states
        self.running = True

        self._disp = display.Display()
        self._root = self._disp.screen().root

        # Map X key names to our key_states entries and log labels
        # Key names are X11 KeySym strings
        self._keymap: Dict[str, Tuple[str, str]] = {
            "Up": ("up", "Arrow Up"),
            "Down": ("down", "Arrow Down"),
            "Left": ("left", "Arrow Left"),
            "Right": ("right", "Arrow Right"),
            "Prior": ("pgup", "PageUp"),
            "Next": ("pgdown", "PageDown"),
            "Home": ("home", "Home"),
            "End": ("cube", "End"),
            "bracketleft": ("grip_open", "Gripper Opening"),
            "bracketright": ("grip_close", "Gripper Closing"),
        }

        self._esc_keysym = XK.string_to_keysym("Escape")
        self._grabs: List[Tuple[int, int]] = []

        self._grab_keys()

    def _keysym_to_keycode(self, name: str) -> int:
        ks = XK.string_to_keysym(name)
        return self._disp.keysym_to_keycode(ks)

    def _grab_keys(self) -> None:
        # Common modifier masks to handle CapsLock/NumLock
        mods = [0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask]

        for name in list(self._keymap.keys()) + ["Escape"]:
            keycode = self._keysym_to_keycode(name)
            for m in mods:
                try:
                    self._root.grab_key(keycode, m, True, X.GrabModeAsync, X.GrabModeAsync)
                    self._grabs.append((keycode, m))
                except Exception:
                    pass
        self._disp.sync()

    def _ungrab_keys(self) -> None:
        for keycode, m in self._grabs:
            try:
                self._root.ungrab_key(keycode, m)
            except Exception:
                pass
        self._disp.sync()

    def start_listener(self) -> None:
        print("\n=== KEYBOARD LISTENER (Xorg) STARTED ===")
        print("Hold keys for continuous movement:")
        print("  Arrow Up/Down - Move Along X axis")
        print("  Arrow Left/Right - Move Along Y axis")
        print("  PageUp/PageDown - Move Up/Down (Z axis)")
        print("\nGripper Control:")
        print("  [ - Open Gripper (incremental)")
        print("] - Close Gripper (incremental)")
        print("\nUtility:")
        print("  Home - Reset to Home")
        print("  End - Move to Cube")
        print("  ESC - Exit")
        print("=====================================")

        try:
            while self.running:
                evt = self._disp.next_event()
                if evt.type == X.KeyPress:
                    if evt.detail == self._disp.keysym_to_keycode(self._esc_keysym):
                        print("ESC pressed - exiting...")
                        self.running = False
                        break
                    self._handle_key(evt.detail, True)
                elif evt.type == X.KeyRelease:
                    self._handle_key(evt.detail, False)
        finally:
            self._ungrab_keys()

    def _handle_key(self, keycode: int, is_press: bool) -> None:
        for name, (state_key, label) in self._keymap.items():
            if keycode == self._keysym_to_keycode(name):
                self.key_states[state_key] = is_press
                if is_press:
                    print(f"{label}: ON")
                else:
                    print(f"{label}: OFF")
                break


