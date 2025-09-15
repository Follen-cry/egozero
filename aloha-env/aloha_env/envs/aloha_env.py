import os
import sys
import time
from pathlib import Path

import cv2
import gym
import numpy as np
from aloha_ego.aloha.real_env import make_real_env
from aloha.aloha_messages import AlohaState
from scipy.spatial.transform import Rotation as R

# Allow overriding the internet host for camera 6 via env var
INTERNET_HOST = os.environ.get("ALOHA_INTERNET_HOST", "10.19.143.251")


def _load_camera_calibs():
    """Load camera calibrations for Aloha.

    Uses env var `ALOHA_ENV_CALIB_PATH` if provided; otherwise fall back to
    simple placeholders so the module can import in dev/offline mode.
    """
    calib_path = os.environ.get("ALOHA_ENV_CALIB_PATH")
    if calib_path and os.path.exists(calib_path):
        try:
            return np.load(calib_path, allow_pickle=True)[()]
        except Exception as e:
            print(f"Warning: failed to load Aloha calibration from {calib_path}: {e}")
    else:
        if calib_path:
            print(
                f"Warning: Aloha calibration file not found at {calib_path}. Using placeholders."
            )
        else:
            print(
                "Warning: ALOHA_ENV_CALIB_PATH not set. Using default calibration placeholders."
            )

    # Defaults to keep import working; replace with real intrinsics/extrinsics
    return {
        "cam_3": {
            "int": np.eye(3, dtype=np.float32),
            "dist_coeff": np.zeros((1, 5), dtype=np.float32),
            "ext": np.eye(4, dtype=np.float32),
        },
        "cam_4": {
            "int": np.eye(3, dtype=np.float32),
            "dist_coeff": np.zeros((1, 5), dtype=np.float32),
            "ext": np.eye(4, dtype=np.float32),
        },
        # If you use ARIA/iPhone for cam 6, override via ALOHA_ENV_CALIB_PATH
        "cam_6": {
            "int": np.array(
                [
                    [706.01969952, 0.0, 360.86504065],
                    [0.0, 706.15628068, 490.34852859],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "dist_coeff": np.array(
                [[2.97673215e-01, -1.69844695e00, 1.65368204e-03, 3.61532041e-05, 3.03597517e00]],
                dtype=np.float32,
            ),
            "ext": np.eye(4, dtype=np.float32),
        },
    }


_calibs = _load_camera_calibs()

K = {
    3: _calibs.get("cam_3", {}).get("int", np.eye(3, dtype=np.float32)),
    4: _calibs.get("cam_4", {}).get("int", np.eye(3, dtype=np.float32)),
    6: _calibs.get("cam_6", {}).get("int", np.eye(3, dtype=np.float32)),
}
D = {
    3: _calibs.get("cam_3", {}).get("dist_coeff", np.zeros((1, 5), dtype=np.float32)),
    4: _calibs.get("cam_4", {}).get("dist_coeff", np.zeros((1, 5), dtype=np.float32)),
    6: _calibs.get("cam_6", {}).get("dist_coeff", np.zeros((1, 5), dtype=np.float32)),
}
T_robot_to_camera = {
    3: _calibs.get("cam_3", {}).get("ext", np.eye(4, dtype=np.float32)),
    4: _calibs.get("cam_4", {}).get("ext", np.eye(4, dtype=np.float32)),
    6: _calibs.get("cam_6", {}).get("ext", np.eye(4, dtype=np.float32)),
}


class AlohaEnv(gym.Env):
    """Aloha robot environment using Franka-compatible messages.

    Sends AlohaAction with (pos, quat, gripper, reset, timestamp) and receives
    AlohaState with (pos, quat, gripper, timestamp). The server transforms
    Franka-like actions into Aloha RealEnv API calls.
    """

    def __init__(
        self,
        width=640,
        height=480,
        use_robot=True,
        use_gt_depth=False,
        crop_h=None,
        crop_w=None,
        cam_ids=[3, 4, 6],
    ):
        super(AlohaEnv, self).__init__()
        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

        # Franka-compatible observation/action layout
        self.feature_dim = 8
        self.action_dim = 7
        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth
        self.n_channels = 3
        self.reward = 0

        self.robot_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )

        if self.use_robot:
            # Which numeric camera IDs to expose in obs keys, and how they map to
            # RealEnv camera names. By default, mimic FrankaEnv keys:
            # - pixels6: overhead/high view (cam_high)
            # - pixels3: low/front view (cam_low)
            # - pixels4: left wrist view (cam_left_wrist)
            # Adjust by passing a different cam_ids list if needed.
            self.cam_ids = cam_ids
            self._cam_id_to_name = {
                6: "cam_high",
                3: "cam_low",
                4: "cam_left_wrist",
                # Optionally: 5x series could be added if needed in future
            }

            # Initialize RealEnv via factory (no ROS node init here)
            # NOTE: RealEnv signature is make_real_env(init_node, setup_robots=True)
            self._real_env = make_real_env(init_node=False)
            self._left_pose6_cache = None

            # Initialize RealEnv via factory; this starts ROS nodes inside
            self._real_env = make_real_env()
            self._left_pose6_cache = None

    def get_state(self):
        """Return AlohaState consistent with Franka-compatible schema.

        - pos, quat: RIGHT end-effector pose if available; otherwise sensible
          defaults (zeros + identity quaternion).
        - gripper: RIGHT gripper mapped from normalized [0,1] to [-1,1].
        """
        if not self.use_robot:
            return AlohaState(
                pos=np.zeros(3, dtype=np.float32),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                gripper=-1.0,
                timestamp=time.time(),
            )

        # Default pose values
        pos = np.zeros(3, dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Try to read RIGHT EE pose from aloha_ego follower bot
        pose = self._real_env.follower_bot_right.arm.get_ee_pose()
        # Convert 4x4 homogeneous pose -> position (3,) and quaternion (x,y,z,w)
        try:
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                pos = pose[:3, 3].astype(np.float32)
                quat = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
        except Exception:
            print("Error converting follower bot RIGHT pose to pos+quat:", pose)
            pass

        # Read qpos to get RIGHT gripper normalized value and map to [-1,1], only to get gripper flag
        try:
            qpos = np.asarray(self._real_env.get_qpos(), dtype=np.float32)
            right_g_norm = float(qpos[13]) if qpos.shape[0] >= 14 else 0.0
        except Exception:
            right_g_norm = 0.0
            print("Error reading qpos for gripper state")
        gripper = -2.0 * right_g_norm + 1.0

        robot_state = AlohaState(pos=pos, quat=quat, gripper=gripper, timestamp=time.time())
        self.robot_state = robot_state
        return robot_state

    def step(self, abs_action):
        if not self.use_robot:
            obs = {
                "features": np.zeros(self.feature_dim, dtype=np.float32),
                "pixels": np.zeros(
                    (self.height, self.width, self.n_channels), dtype=np.uint8
                ),
            }
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width), dtype=np.float32)
            self.curr_images = {6: obs["pixels"]}
            return obs, 0.0, False, {}


        ee_cmd = self._franka_action_to_ee14(abs_action)
        self._real_env.step_ee(ee_cmd, get_obs=False)

        robot_state = self.get_state()

        # Pull images directly from RealEnv (ROS image topics via CvBridge)
        image_dict = self._safe_get_images()

        # Build numeric-keyed image list consistent with FrankaEnv
        image_list = {}
        for cam_idx in self.cam_ids:
            cam_name = self._cam_id_to_name.get(cam_idx)
            if cam_name is None:
                continue
            img = image_dict.get(cam_name)
            if img is None:
                continue
            if self.crop_h is not None and self.crop_w is not None:
                h, w = img.shape[:2]
                img = img[
                    int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                    int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                ]
            image_list[cam_idx] = img

        # Depth not provided by RealEnv; optionally keep API but warn once
        depth_list = None
        if self.use_gt_depth and not hasattr(self, "_depth_warned"):
            print("Warning: use_gt_depth requested but RealEnv does not provide depth. Skipping.")
            self._depth_warned = True

        self.curr_images = image_list

        obs = {
            "features": np.concatenate(
                (robot_state.pos, robot_state.quat, [robot_state.gripper])
            )
        }

        for cam_idx, image in image_list.items():
            obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
        if self.use_gt_depth and depth_list is not None:
            for cam_idx, depth in depth_list.items():
                obs[f"depth{cam_idx}"] = cv2.resize(depth, (self.width, self.height))

        return obs, self.reward, False, None

    def reset(self):
        if self.use_robot:
            print("resetting AlohaEnv")
            # Reset robots via RealEnv
            self._real_env.reset(fake=False)
            robot_state: AlohaState = self.get_state()
            self.robot_state = robot_state
            print("reset done: ", robot_state)

            # Pull images via RealEnv
            image_dict = self._safe_get_images()

            image_list = {}
            for cam_idx in self.cam_ids:
                cam_name = self._cam_id_to_name.get(cam_idx)
                if cam_name is None:
                    continue
                img = image_dict.get(cam_name)
                if img is None:
                    continue
                if self.crop_h is not None and self.crop_w is not None:
                    h, w = img.shape[:2]
                    img = img[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]
                image_list[cam_idx] = img

            # Depth not provided by RealEnv; optionally keep API but warn once
            depth_list = None
            if self.use_gt_depth and not hasattr(self, "_depth_warned"):
                print("Warning: use_gt_depth requested but RealEnv does not provide depth. Skipping.")
                self._depth_warned = True

            self.curr_images = image_list

            obs = {
                "features": np.concatenate(
                    (robot_state.pos, robot_state.quat, [robot_state.gripper])
                )
            }
            for cam_idx, image in image_list.items():
                obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
            if self.use_gt_depth and depth_list is not None:
                for cam_idx, depth in depth_list.items():
                    obs[f"depth{cam_idx}"] = cv2.resize(depth, (self.width, self.height))

            return obs

        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width))
            return obs

    def _safe_get_images(self):
        """Return latest images from RealEnv, handling transient None frames.

        RealEnv.ImageRecorder may briefly return None before the first frames
        arrive. This helper normalizes to an empty dict for missing frames.
        """
        images = {}
        try:
            raw = self._real_env.get_images()
        except Exception as e:
            print(f"Warning: failed to fetch images from RealEnv: {e}")
            raw = {}

        if isinstance(raw, dict):
            for k, v in raw.items():
                if v is not None:
                    images[k] = v
        return images


    def render(self, mode="rgb_array", cam_idx=None, width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            if cam_idx is not None:
                return self.curr_images[cam_idx]

            image_list = []
            for _, im in self.curr_images.items():
                h, w = im.shape[:2]
                aspect_ratio = w / h
                new_height = height
                new_width = int(aspect_ratio * new_height)
                image_list.append(cv2.resize(im, (new_width, new_height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError

    def _franka_action_to_ee14(self, abs_action: np.ndarray) -> np.ndarray:
        """Map a Franka-style 8D action (pos, quat, gripper) to 14D ee pose cmd.

        Layout: [left_xyzrpy(6), left_g(1), right_xyzrpy(6), right_g(1)]
        - Uses provided action for RIGHT arm pose and gripper.
        - Keeps LEFT arm at its current ee pose if available; otherwise uses the
          right pose as a neutral placeholder to minimize spurious motion.
        """
        pos = np.asarray(abs_action[:3], dtype=np.float32)
        quat = np.asarray(abs_action[3:7], dtype=np.float32)
        # Convert quat [x,y,z,w] to rpy (rad)
        try:
            rpy = R.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)
        except Exception:
            print("Setting rpy to zeros due to invalid quat:", quat)
            rpy = np.zeros(3, dtype=np.float32)

        # Gripper mapping Franka [-1,1] -> normalized [1,0]
        g = float(abs_action[-1])
        right_g = float(np.clip(0.5 * (-g + 1.0), 0.0, 1.0))

        # Build right arm 7D
        right_pose6 = np.concatenate([pos, rpy]).astype(np.float32)

        # Get current LEFT ee pose if available to hold position (follower bot)
        left_pose6 = None
        try:
            arm = getattr(self._real_env, "follower_bot_left").arm
            get_comp = getattr(arm, "get_ee_pose_components", None)
            if callable(get_comp):
                comp = get_comp()
                if isinstance(comp, (list, tuple, np.ndarray)) and len(comp) >= 6:
                    left_pose6 = np.asarray(comp[:6], dtype=np.float32)
                elif all(
                    hasattr(comp, k) for k in ("x", "y", "z", "roll", "pitch", "yaw")
                ):
                    left_pose6 = np.array(
                        [comp.x, comp.y, comp.z, comp.roll, comp.pitch, comp.yaw],
                        dtype=np.float32,
                    )
            else:
                get_pose = getattr(arm, "get_ee_pose", None)
                if callable(get_pose):
                    pose = get_pose()
                    if hasattr(pose, "position") and hasattr(pose, "orientation"):
                        pos_l = np.array(
                            [pose.position.x, pose.position.y, pose.position.z],
                            dtype=np.float32,
                        )
                        quat_l = np.array(
                            [
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w,
                            ],
                            dtype=np.float32,
                        )
                        rpy_l = (
                            R.from_quat(quat_l)
                            .as_euler("xyz", degrees=False)
                            .astype(np.float32)
                        )
                        left_pose6 = np.concatenate([pos_l, rpy_l])
        except Exception:
            left_pose6 = None
        if left_pose6 is not None:
            self._left_pose6_cache = left_pose6.copy()
        else:
            # Use cache if available; otherwise, avoid moving LEFT by falling back
            left_pose6 = getattr(self, "_left_pose6_cache", None)
            if left_pose6 is None:
                raise RuntimeError("Left EE pose unavailable; cannot call step_ee safely")

        # Left gripper unchanged (use current normalized value if available)
        try:
            qpos = np.asarray(self._real_env.get_qpos(), dtype=np.float32)
            left_g = float(np.clip(qpos[6], 0.0, 1.0)) if qpos.shape[0] >= 7 else right_g
        except Exception:
            left_g = right_g

        return np.concatenate([left_pose6, [left_g], right_pose6, [right_g]]).astype(
            np.float32
        )
