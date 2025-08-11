import numpy as np

# Control loop dt (s)
INTEGRATION_DT: float = 0.1

# Teleop step size
DELTA_STEP: float = 0.005  # meters per control tick

# Gripper parameters (interpreted as opening in meters)
GRIPPER_OPEN_M: float = 0.04
# Smooth gripper rate (meters/second) when holding open/close
GRIPPER_RATE_MPS: float = 0.05

# Joint command smoothing for IK solutions
JOINT_RATE_LIMIT_RAD_PER_S: float = 2.0


