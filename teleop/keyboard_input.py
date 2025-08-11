from __future__ import annotations

from pynput import keyboard


class KeyboardHandler:
    def __init__(self, key_states: dict[str, bool]):
        self.key_states = key_states
        self.running = True

    def on_press(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') and key.char else None

            if key == keyboard.Key.up:
                if not self.key_states['up']:
                    self.key_states['up'] = True
                    print("Arrow Up: ON")
            elif key == keyboard.Key.down:
                if not self.key_states['down']:
                    self.key_states['down'] = True
                    print("Arrow Down: ON")
            elif key == keyboard.Key.left:
                if not self.key_states['left']:
                    self.key_states['left'] = True
                    print("Arrow Left: ON")
            elif key == keyboard.Key.right:
                if not self.key_states['right']:
                    self.key_states['right'] = True
                    print("Arrow Right: ON")
            elif key == keyboard.Key.page_up:
                if not self.key_states['pgup']:
                    self.key_states['pgup'] = True
                    print("PageUp: ON")
            elif key == keyboard.Key.page_down:
                if not self.key_states['pgdown']:
                    self.key_states['pgdown'] = True
                    print("PageDown: ON")
            elif key_char in ['[', ']']:
                if key_char == '[' and not self.key_states['grip_open']:
                    self.key_states['grip_open'] = True
                    print("Gripper Opening: ON")
                elif key_char == ']' and not self.key_states['grip_close']:
                    self.key_states['grip_close'] = True
                    print("Gripper Closing: ON")
            elif key == keyboard.Key.home:
                self.key_states['home'] = True
            elif key == keyboard.Key.end:
                self.key_states['cube'] = True
            elif key == keyboard.Key.esc:
                print("ESC pressed - exiting...")
                self.running = False
                return False
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, 'char') and key.char else None

            if key == keyboard.Key.up:
                if self.key_states['up']:
                    self.key_states['up'] = False
                    print("Arrow Up: OFF")
            elif key == keyboard.Key.down:
                if self.key_states['down']:
                    self.key_states['down'] = False
                    print("Arrow Down: OFF")
            elif key == keyboard.Key.left:
                if self.key_states['left']:
                    self.key_states['left'] = False
                    print("Arrow Left: OFF")
            elif key == keyboard.Key.right:
                if self.key_states['right']:
                    self.key_states['right'] = False
                    print("Arrow Right: OFF")
            elif key == keyboard.Key.page_up:
                if self.key_states['pgup']:
                    self.key_states['pgup'] = False
                    print("PageUp: OFF")
            elif key == keyboard.Key.page_down:
                if self.key_states['pgdown']:
                    self.key_states['pgdown'] = False
                    print("PageDown: OFF")
            elif key_char in ['[', ']']:
                if key_char == '[' and self.key_states['grip_open']:
                    self.key_states['grip_open'] = False
                    print("Gripper Opening: OFF")
                elif key_char == ']' and self.key_states['grip_close']:
                    self.key_states['grip_close'] = False
                    print("Gripper Closing: OFF")
        except AttributeError:
            pass

    def start_listener(self):
        print("\n=== KEYBOARD LISTENER STARTED ===")
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

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


