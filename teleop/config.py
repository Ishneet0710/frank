import numpy as np

# IK and control parameters
INTEGRATION_DT: float = 0.1
DAMPING: float = 1e-3
KPOS: float = 0.9
KORI: float = 0.9
NULLSPACE_GAIN: float = 0.5

# Teleop step size
DELTA_STEP: float = 0.005  # meters per control tick

# Gripper parameters (interpreted as opening in meters)
GRIPPER_OPEN_M: float = 0.04
# Larger gripper increment per tick for faster open/close
GRIPPER_STEP_M: float = 0.01


