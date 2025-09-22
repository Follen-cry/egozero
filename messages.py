from dataclasses import dataclass
import numpy as np


@dataclass
class AlohaState:
    """Franka-compatible state schema for Aloha.

    Fields mirror frankateach.messages.FrankaState for API compatibility.
    """

    pos: np.ndarray
    quat: np.ndarray
    gripper: np.ndarray
    timestamp: float
    start_teleop: bool = False


@dataclass
class AlohaAction:
    """Franka-compatible action schema for Aloha.

    Fields mirror frankateach.messages.FrankaAction for API compatibility.
    """

    pos: np.ndarray
    quat: np.ndarray
    gripper: np.ndarray
    reset: bool
    timestamp: float
