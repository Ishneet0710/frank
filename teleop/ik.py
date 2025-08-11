from __future__ import annotations

import numpy as np
import mujoco

from teleop.config import INTEGRATION_DT, DAMPING, KPOS, KORI, NULLSPACE_GAIN


def compute_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dof_ids: list[int],
    home_qpos: np.ndarray,
    site_name: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """Compute joint velocities via damped-least-squares IK with nullspace term."""
    site_id = model.site(site_name).id
    current_pos = data.site_xpos[site_id]
    mat_3x3 = data.site_xmat[site_id].reshape((3, 3))
    cur_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cur_quat, mat_3x3.ravel())

    # Position twist
    pos_err = target_pos - current_pos
    twist = np.zeros(6)
    twist[:3] = KPOS * pos_err / INTEGRATION_DT

    # Orientation twist
    conj_cur = np.array([cur_quat[0], -cur_quat[1], -cur_quat[2], -cur_quat[3]])
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, conj_cur)
    ang_vel = np.zeros(3)
    mujoco.mju_quat2Vel(ang_vel, err_quat, 1.0)
    twist[3:] = KORI * ang_vel / INTEGRATION_DT

    # Jacobian at site
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    full_jac = np.vstack((jacp, jacr))
    J = full_jac[:, dof_ids]

    # Damped least squares
    damp_eye = DAMPING * np.eye(6)
    inv_term = np.linalg.inv(J @ J.T + damp_eye)
    dq_raw = J.T @ (inv_term @ twist)

    # Nullspace attractor to home
    q_now = data.qpos[dof_ids]
    diff_home = home_qpos - q_now
    J_pinv = np.linalg.pinv(J)
    I = np.eye(len(dof_ids))
    null_term = (I - J_pinv @ J) @ (NULLSPACE_GAIN * diff_home)

    return dq_raw + null_term


