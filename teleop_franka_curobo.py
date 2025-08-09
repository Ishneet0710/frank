import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import os
import numpy as np

# MuJoCo
try:
    import mujoco
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "mujoco is required for XML sim: pip install mujoco (and set MUJOCO_GL=osmesa or egl on headless)."
    ) from e

# Keyboard/mouse input
try:
    from pynput import keyboard, mouse
except Exception as e:  # pragma: no cover
    raise RuntimeError("pynput is required: pip install pynput") from e


# cuRobo imports (keep narrow and version-tolerant)
try:
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "cuRobo is required. Install per docs: pip install curobo (and torch with CUDA/CPU)."
    ) from e


@dataclass
class TeleopConfig:
    # Path to cuRobo Franka YAML in this repo
    franka_yaml_path: str = "curobo_config/franka/franka.yml"

    # Control increments (meters per keypress tick)
    step_xy: float = 0.01
    step_z: float = 0.01

    # Control loop frequency
    control_hz: float = 30.0
    # Simulation frequency
    sim_hz: float = 500.0

    # IK solver settings
    num_seeds: int = 32
    position_threshold: float = 0.003
    rotation_threshold: float = 0.05

    # Initial EE pose if FK is unavailable (x, y, z, qw, qx, qy, qz)
    initial_pose: Tuple[float, float, float, float, float, float, float] = (
        0.5,
        0.0,
        0.4,
        1.0,
        0.0,
        0.0,
        0.0,
    )

    # MuJoCo XML environment
    xml_path: str = "assets/environments/pick_place.xml"


class MjFrankaSim:
    """MuJoCo simulator wrapper for Franka + pick_place environment."""

    def __init__(self, cfg: TeleopConfig) -> None:
        self.cfg = cfg
        xml = os.path.normpath(cfg.xml_path)
        if not os.path.exists(xml):
            raise FileNotFoundError(f"XML not found: {xml}")

        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        # Names
        self.joint_names = [f"robot_0/joint{i}" for i in range(1, 8)]
        self.finger_joint_names = ["robot_0/finger_joint1", "robot_0/finger_joint2"]
        self.actuator_names = [f"robot_0/actuator{i}" for i in range(1, 9)]
        self.ee_site_name = "robot_0/ee_site"

        # Indices
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.jnt_qposadr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.finger_joint_names
        ]
        self.finger_qposadr = [self.model.jnt_qposadr[jid] for jid in self.finger_joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)

        # Reset state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Cache ctrl ranges
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

    def set_joints_target(self, q_target: np.ndarray) -> None:
        # Position-servo actuators 1..7: ctrl = target position (clamped)
        for i in range(7):
            act_id = self.actuator_ids[i]
            lo, hi = self.ctrl_ranges[act_id]
            val = float(np.clip(q_target[i], lo, hi))
            self.data.ctrl[act_id] = val

    def set_gripper_open(self, is_open: bool) -> None:
        # Actuator8 (tendon) ctrl range [0, 255]; 255 ≈ max open, 0 ≈ closed
        act_id = self.actuator_ids[7]
        lo, hi = self.ctrl_ranges[act_id]
        self.data.ctrl[act_id] = hi if is_open else lo

    def set_qpos(self, q_arm: np.ndarray, finger_pos: float) -> None:
        # Initialize qpos to known values
        for i, adr in enumerate(self.jnt_qposadr):
            self.data.qpos[adr] = float(q_arm[i])
        # Fingers are slide joints with same value (0..0.04)
        f = float(np.clip(finger_pos, 0.0, 0.04))
        for adr in self.finger_qposadr:
            self.data.qpos[adr] = f
        mujoco.mj_forward(self.model, self.data)

    def get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self.jnt_qposadr], dtype=np.float32)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (pos[3], quat[4] as wxyz)
        pos = np.array(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        mat = np.array(self.data.site_xmat[self.ee_site_id], dtype=np.float64)  # shape (9,)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat.astype(np.float32)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)


class FrankaTeleop:
    def __init__(self, cfg: TeleopConfig) -> None:
        self.cfg = cfg

        # cuRobo device setup
        self.tensor_args = TensorDeviceType()

        # Load franka YAML
        franka_cfg = load_yaml(self.cfg.franka_yaml_path)

        kin = franka_cfg["robot_cfg"]["kinematics"]
        self.urdf_path: str = kin["urdf_path"]
        self.base_link: str = kin["base_link"]
        self.ee_link: str = kin["ee_link"]

        # Fallback to local URDF if YAML path isn't present in this workspace
        if not os.path.isabs(self.urdf_path):
            candidate = os.path.normpath(self.urdf_path)
            if not os.path.exists(candidate):
                local_urdf = os.path.normpath("assets/franka/urdf/franka.urdf")
                if os.path.exists(local_urdf):
                    print(f"Using local URDF fallback: {local_urdf}")
                    self.urdf_path = local_urdf

        # Optional retract and finger joints
        self.cspace_cfg = franka_cfg["robot_cfg"].get("cspace", {})
        self.retract_config = self.cspace_cfg.get(
            "retract_config", [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
        )

        # 7 arm joints only (fingers handled separately)
        if len(self.retract_config) >= 7:
            self.q_current = np.array(self.retract_config[:7], dtype=np.float32)
        else:
            self.q_current = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0], dtype=np.float32)

        # Gripper state
        self.gripper_open = True

        # Build cuRobo RobotConfig & IK
        self.robot_cfg = RobotConfig.from_basic(
            self.urdf_path, self.base_link, self.ee_link, self.tensor_args
        )

        self.ik_cfg = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=self.cfg.rotation_threshold,
            position_threshold=self.cfg.position_threshold,
            num_seeds=self.cfg.num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )

        self.ik_solver = IKSolver(self.ik_cfg)

        # MuJoCo sim
        self.sim = MjFrankaSim(cfg)

        # Initialize sim to retract pose and open gripper
        self.sim.set_qpos(self.q_current, finger_pos=0.04)

        # Seed EE target and IK seed from sim
        pos_w, quat_w = self.sim.get_ee_pose()
        # cuRobo expects (x,y,z, qw,qx,qy,qz)
        self.ee_target = np.array([pos_w[0], pos_w[1], pos_w[2], quat_w[0], quat_w[1], quat_w[2], quat_w[3]], dtype=np.float32)
        self.q_current = self.sim.get_arm_qpos()

        # Input state
        self._pressed = set()
        self._stop = False

    # -------- Input handling --------
    def _on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)
        self._pressed.add(k)

    def _on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)
        if k in self._pressed:
            self._pressed.remove(k)
        if key == keyboard.Key.esc:
            self._stop = True
            return False

    def _on_click(self, _x, _y, button, pressed):
        if not pressed:
            return
        if button == mouse.Button.left:
            self.gripper_open = False
        elif button == mouse.Button.right:
            self.gripper_open = True

    # -------- IK solve --------
    def _solve_ik(self, ee_pose: np.ndarray, q_seed: np.ndarray) -> Optional[np.ndarray]:
        # ee_pose: [x, y, z, qw, qx, qy, qz]
        pos = ee_pose[:3]
        quat = ee_pose[3:]

        pose = Pose.from_list(
            [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            self.tensor_args,
        ).unsqueeze(0)  # [B=1]

        seed = torch.tensor(q_seed, dtype=torch.float32, device=self.tensor_args.device).unsqueeze(0)

        result = self.ik_solver.solve(pose, seed)

        # Support common result shapes across versions
        if isinstance(result, dict):
            success = bool(result.get("success", False))
            q = result.get("q")
        else:
            success = bool(getattr(result, "success", False))
            q = getattr(result, "q", None)

        if not success or q is None:
            return None

        q_np = q.detach().cpu().numpy().reshape(-1)
        return q_np[:7]

    # -------- Control loop --------
    def _update_target_from_keys(self) -> None:
        dx = dy = dz = 0.0
        if "w" in self._pressed:
            dx += self.cfg.step_xy
        if "s" in self._pressed:
            dx -= self.cfg.step_xy
        if "d" in self._pressed:
            dy -= self.cfg.step_xy
        if "a" in self._pressed:
            dy += self.cfg.step_xy
        # Optional vertical motion with q/e
        if "q" in self._pressed:
            dz += self.cfg.step_z
        if "e" in self._pressed:
            dz -= self.cfg.step_z

        self.ee_target[0] += dx
        self.ee_target[1] += dy
        self.ee_target[2] += dz

    def _compose_full_joints(self, q_arm: np.ndarray) -> np.ndarray:
        # For logging only; MuJoCo is the source of truth
        finger = 0.04 if self.gripper_open else 0.0
        return np.concatenate([q_arm, np.array([finger, finger], dtype=np.float32)], axis=0)

    def run(self) -> None:
        print("Controls: WASD (xy), Q/E (z), LeftClick=Close, RightClick=Open, ESC=quit")

        kb_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        ms_listener = mouse.Listener(on_click=self._on_click)
        kb_listener.start()
        ms_listener.start()

        control_period = 1.0 / self.cfg.control_hz
        sim_period = 1.0 / self.cfg.sim_hz
        sim_steps_per_control = max(1, int(round(control_period / sim_period)))

        try:
            while not self._stop:
                t0 = time.time()

                # Update EE target from inputs
                self._update_target_from_keys()

                # Solve IK
                q_arm = self._solve_ik(self.ee_target, self.q_current)
                if q_arm is not None:
                    self.q_current = q_arm
                    # Send commands to MuJoCo
                    self.sim.set_joints_target(q_arm)
                    self.sim.set_gripper_open(self.gripper_open)
                else:
                    # Keep previous targets
                    pass

                # Step sim for the duration of one control period
                for _ in range(sim_steps_per_control):
                    self.sim.step(1)
                    time.sleep(sim_period)

                # Optional lightweight status
                q_full = self._compose_full_joints(self.q_current)
                print(
                    f"q: {np.array2string(q_full, precision=3, floatmode='fixed')}  | pose: "
                    f"[{self.ee_target[0]:.3f}, {self.ee_target[1]:.3f}, {self.ee_target[2]:.3f}]  "
                    f"grip: {'open' if self.gripper_open else 'close'}",
                    end="\r",
                    flush=True,
                )
        finally:
            kb_listener.stop()
            ms_listener.stop()
            print("\nTeleop stopped.")


def main():
    teleop = FrankaTeleop(TeleopConfig())
    teleop.run()


if __name__ == "__main__":
    main()


