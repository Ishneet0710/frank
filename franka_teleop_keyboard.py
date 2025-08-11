#!/usr/bin/env python3

import time
import threading

import mujoco
from mujoco.viewer import launch_passive

from teleop.robot import FrankaTeleop
from teleop.keyboard_input import KeyboardHandler
try:
    from teleop.keyboard_input_xorg import KeyboardHandlerXorg
except Exception:
    KeyboardHandlerXorg = None  # type: ignore


def print_controls() -> None:
    print("\n" + "=" * 60)
    print("FRANKA TELEOPERATION CONTROLS - CONTINUOUS MOVEMENT")
    print("=" * 60)
    print("Movement (hold for continuous movement):")
    print("  Arrow Up/Down - Move Along X axis")
    print("  Arrow Left/Right - Move Along Y axis")
    print("  PageUp/PageDown - Move Up/Down (Z axis)")
    print("\nGripper:")
    print("  [ - Open Gripper (incremental)")
    print("  ] - Close Gripper (incremental)")
    print("\nUtility:")
    print("  Home - Reset to Home Position")
    print("  End - Move to Cube Position")
    print("  ESC - Exit")
    print("=" * 60)


def main() -> None:
    teleop = FrankaTeleop()
    # Prefer Xorg handler on Linux/X11 for consistent key behavior
    if KeyboardHandlerXorg is not None:
        try:
            kbd_handler = KeyboardHandlerXorg(teleop.key_states)  # type: ignore
        except Exception:
            kbd_handler = KeyboardHandler(teleop.key_states)
    else:
        kbd_handler = KeyboardHandler(teleop.key_states)
    
    # Start keyboard listener thread
    kbd_thread = threading.Thread(target=kbd_handler.start_listener, daemon=True)
    kbd_thread.start()
    
    # Launch viewer
    with launch_passive(teleop.model, teleop.data) as viewer:
        print_controls()
        print("\nViewer is ready! Use keyboard for direct control.")
        print("Hold keys for continuous movement, ESC to exit.\n")
        
        last_update_time = time.time()
        
        while viewer.is_running() and kbd_handler.running:
            current_time = time.time()
            
            # Process input and control at ~50Hz
            if current_time - last_update_time > 0.01:
                dt = current_time - last_update_time
                moved = teleop.process_input(dt)
                if moved:
                    teleop.solve_ik_and_control()
                teleop.apply_gripper_ctrl()
                last_update_time = current_time
            
            mujoco.mj_step(teleop.model, teleop.data)
            viewer.sync()
            time.sleep(0.005)
        
        print("\nTeleoperation ended.")


if __name__ == "__main__":
    main()

