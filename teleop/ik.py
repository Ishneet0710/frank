from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from teleop.config import JOINT_RATE_LIMIT_RAD_PER_S

import torch  
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # type: ignore
from curobo.types.base import TensorDeviceType  # type: ignore
from curobo.types.math import Pose  # type: ignore
from curobo.util_file import get_robot_configs_path, join_path  # type: ignore

# Prefer GPU and enable TF32 for speed
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def lowpass_filter_joint_command(prev_q: np.ndarray, q_cmd: np.ndarray, dt: float) -> np.ndarray:
    """Rate limit joint command to reduce jerk."""
    max_step = JOINT_RATE_LIMIT_RAD_PER_S * dt
    dq = q_cmd - prev_q
    step = np.clip(dq, -max_step, max_step)
    return prev_q + step


class CuroboIK:
    """Wrapper around cuRobo's IKSolver."""

    def __init__(self) -> None:
        self.available: bool = False
        self._solver = None
        self._tensor_args = None
        self._Pose = None
        device = torch.device("cuda") 
        self._tensor_args = TensorDeviceType(device=device)
        robot_cfg_path = join_path(get_robot_configs_path(), "franka.yml")
        cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg_path,
            tensor_args=self._tensor_args,
            ee_link_name=None,
            use_cuda_graph=True,
            self_collision_check=False,
            self_collision_opt=False,
            regularization=True,
            num_seeds=24,
            position_threshold=0.01,
            rotation_threshold=0.05,
            grad_iters=60,
            use_fixed_samples=True,
        )
        self._solver = IKSolver(cfg)
        # Stash types for later
        self._Pose = Pose
        self.available = True


    def solve(self, target_pos: np.ndarray, target_quat_wxyz: np.ndarray,
              seed_q: Optional[np.ndarray] = None,
              dt: float = 0.02) -> Tuple[Optional[np.ndarray], bool]:
        """Solve IK for a single pose. Returns (q_solution, success)."""
        if not self.available or self._solver is None:
            return None, False
        try:
            import torch  # type: ignore
        except Exception:
            return None, False

        # Compose goal pose
        px, py, pz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
        qw, qx, qy, qz = [float(x) for x in target_quat_wxyz]
        pose_list = [px, py, pz, qw, qx, qy, qz]
        goal_pose = self._Pose.from_list(pose_list, tensor_args=self._tensor_args, q_xyzw=False)

        # Seed shape: (n, batch, dof). Use current q if provided
        seed_config = None
        if seed_q is not None:
            seed7 = np.asarray(seed_q, dtype=np.float32).reshape(1, 1, -1)
            seed_config = torch.as_tensor(seed7, device=self._tensor_args.device, dtype=self._tensor_args.dtype)

        # Use modest seeds and iterations for servoing smoothness
        result = self._solver.solve_single(
            goal_pose,
            seed_config=seed_config,
            return_seeds=1,
            num_seeds=24,
            newton_iters=40,
        )
        success = bool(result.success.view(-1)[0].item())
        if not success:
            return None, False
        # result.solution: shape (batch=1, return_seeds=1, dof)
        q_sol = result.solution.view(-1).detach().cpu().numpy()
        # Apply rate limit to reduce jerk relative to seed
        if seed_q is not None:
            q_sol = lowpass_filter_joint_command(seed_q, q_sol, dt)
        return q_sol, True

    def prewarm(self, target_pos: np.ndarray, target_quat_wxyz: np.ndarray,
                seed_q: Optional[np.ndarray] = None) -> None:
        if not self.available:
            return
        try:
            _ = self.solve(target_pos, target_quat_wxyz, seed_q=seed_q, dt=0.02)
        except Exception:
            pass


