from __future__ import annotations

import time
import threading
from typing import List

import numpy as np
import mujoco

from teleop.config import (
    INTEGRATION_DT,
    DELTA_STEP,
    GRIPPER_OPEN_M,
    GRIPPER_RATE_MPS,
)
from teleop.ik import compute_joint_velocity


class FrankaTeleop:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("assets/environments/pick_place.xml")
        self.data = mujoco.MjData(self.model)

        self.franka_joints: List[str] = [
            "robot_0/joint1",
            "robot_0/joint2",
            "robot_0/joint3",
            "robot_0/joint4",
            "robot_0/joint5",
            "robot_0/joint6",
            "robot_0/joint7",
        ]
        self.dof_ids: List[int] = [self.model.joint(name).qposadr[0] for name in self.franka_joints]

        self.actuator_names: List[str] = [
            "robot_0/actuator1",
            "robot_0/actuator2",
            "robot_0/actuator3",
            "robot_0/actuator4",
            "robot_0/actuator5",
            "robot_0/actuator6",
            "robot_0/actuator7",
        ]
        self.actuator_ids: List[int] = [self.model.actuator(name).id for name in self.actuator_names]

        # Gripper actuators (direct position control on each finger)
        self.gripper_actuator_ids: List[int] = []
        try:
            self.gripper_actuator_ids.append(self.model.actuator("robot_0/grip_pos1").id)
            self.gripper_actuator_ids.append(self.model.actuator("robot_0/grip_pos2").id)
        except Exception:
            # Fallback: try legacy tendon actuator
            try:
                self.gripper_actuator_ids = [self.model.actuator("robot_0/actuator8").id]
            except Exception:
                self.gripper_actuator_ids = [self.model.nu - 1]

        self.ee_site_name: str = "robot_0/ee_site"

        self.home_qpos = np.array([0, 0, 0, -1.57, 0, 1.57, 0.785])
        self.target_quat = np.array([0.0, 1.0, 0.0, 0.0])

        # Gripper state in meters
        self.gripper_state: float = GRIPPER_OPEN_M

        self.key_states = {
            'up': False, 'down': False, 'left': False, 'right': False,
            'pgup': False, 'pgdown': False,
            'grip_open': False, 'grip_close': False,
            'home': False, 'cube': False,
        }

        self._initialize_robot()

        site_id = self.model.site(self.ee_site_name).id
        self.target_pos = self.data.site_xpos[site_id].copy()
        self.current_target = self.target_pos.copy()
        # Cache body ids for contact checks
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_0/left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_0/right_finger")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object1")

    def _initialize_robot(self) -> None:
        self.data.qpos[self.dof_ids] = self.home_qpos
        if self.model.nq > 7:
            self.data.qpos[7:self.model.nq] = GRIPPER_OPEN_M

        # Fix cube pose during settle
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object1")
        cube_joint_start = self.model.body_jntadr[cube_body_id] if cube_body_id != -1 else -1

        # Set actuator ctrls to home
        for act_id, joint_val in zip(self.actuator_ids, self.home_qpos):
            self.data.ctrl[act_id] = joint_val

        for gripper_id in self.gripper_actuator_ids:
            self.data.ctrl[gripper_id] = np.clip(self.gripper_state, 0.0, GRIPPER_OPEN_M)

        for _ in range(100):
            if cube_body_id != -1 and cube_joint_start != -1:
                self.data.qpos[cube_joint_start:cube_joint_start+3] = [0.6, -0.02, 0.34]
                self.data.qpos[cube_joint_start+3:cube_joint_start+7] = [1, 0, 0, 0]
            mujoco.mj_step(self.model, self.data)

    def solve_ik_and_control(self) -> bool:
        dq = compute_joint_velocity(
            self.model, self.data, self.dof_ids, self.home_qpos,
            self.ee_site_name, self.current_target, self.target_quat
        )

        new_qpos = self.data.qpos.copy()
        dq_full = np.zeros(self.model.nv)
        dq_full[self.dof_ids] = dq
        mujoco.mj_integratePos(self.model, new_qpos, dq_full, INTEGRATION_DT)

        for act_id, joint_val in zip(self.actuator_ids, new_qpos[self.dof_ids]):
            self.data.ctrl[act_id] = joint_val

        return True

    def process_input(self, dt: float) -> bool:
        moved = False
        # Swapped: Up/Down move X, Left/Right move Y
        if self.key_states['up']:
            self.current_target[0] += DELTA_STEP; moved = True
        if self.key_states['down']:
            self.current_target[0] -= DELTA_STEP; moved = True
        if self.key_states['left']:
            self.current_target[1] -= DELTA_STEP; moved = True
        if self.key_states['right']:
            self.current_target[1] += DELTA_STEP; moved = True
        if self.key_states['pgup']:
            self.current_target[2] += DELTA_STEP; moved = True
        if self.key_states['pgdown']:
            self.current_target[2] -= DELTA_STEP; moved = True

        if self.key_states['home']:
            site_id = self.model.site(self.ee_site_name).id
            self.data.qpos[self.dof_ids] = self.home_qpos
            mujoco.mj_forward(self.model, self.data)
            self.current_target = self.data.site_xpos[site_id].copy()
            self.gripper_state = GRIPPER_OPEN_M
            print(f"Reset to home: {self.current_target}")
            self.key_states['home'] = False

        if self.key_states['cube']:
            self.current_target = np.array([0.6, -0.02, 0.4])
            print(f"Move to cube: {self.current_target}")
            self.key_states['cube'] = False

        # Fast gripper: jump to extremes while key held
        # Smooth rate-based gripper motion
        if self.key_states['grip_open']:
            self.gripper_state = min(GRIPPER_OPEN_M, self.gripper_state + GRIPPER_RATE_MPS * dt)
            moved = True
            print(f"Gripper Opening: {self.gripper_state:.3f}m")

        if self.key_states['grip_close']:
            # Stop closing once both fingers contact the object to avoid crossing
            if not self._both_fingers_contact_object():
                self.gripper_state = max(0.0, self.gripper_state - GRIPPER_RATE_MPS * dt)
                moved = True
                print(f"Gripper Closing: {self.gripper_state:.3f}m")

        return moved

    def apply_gripper_ctrl(self) -> None:
        # Directly command finger positions (fast)
        ctrl_val = np.clip(self.gripper_state, 0.0, GRIPPER_OPEN_M)
        for gripper_id in self.gripper_actuator_ids:
            self.data.ctrl[gripper_id] = ctrl_val

    def _both_fingers_contact_object(self) -> bool:
        if self.object_body_id < 0:
            return False
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = con.geom1
            g2 = con.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if (b1 == self.left_finger_body_id and b2 == self.object_body_id) or (b2 == self.left_finger_body_id and b1 == self.object_body_id):
                left_contact = True
            if (b1 == self.right_finger_body_id and b2 == self.object_body_id) or (b2 == self.right_finger_body_id and b1 == self.object_body_id):
                right_contact = True
            if left_contact and right_contact:
                break
        return left_contact and right_contact


