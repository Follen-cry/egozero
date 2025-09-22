#!/usr/bin/env python3

import json
import os
import sys
import time
import warnings
from collections import defaultdict

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch

from logger import Logger
from PIL import Image
from replay_buffer import make_expert_replay_loader
from scipy.spatial.transform import Rotation
from video import VideoRecorder

import utils
from robot_utils.franka.utils import matrix_to_rotation_6d

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from io_utils import save_video
from transform_utils import average_poses
from vis_utils import add_border, detect_aruco, draw_axis, plot_points

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

#Hard Coded camera intrinsics for the 6 cameras of d455
K = np.array(
        [
            [386.12152099609375, 0.0, 324.1443176269531],
            [0.0, 385.7033386230469, 243.736328125],
            [0.0, 0.0, 1.0],
        ]
    )

D = np.array(
        [
            [
                -0.05693485587835312,
                0.0698220282793045,
                -0.0007267515175044537,
                0.0006372604402713478,
                -0.023148125037550926,
            ]
        ]
)



# Load transform from YAML and compute T_ego_to_robot (4x4 matrix)
import yaml

def _quat_trans_to_matrix(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> np.ndarray:
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def load_T_ego_to_robot_from_yaml(filepath: str) -> np.ndarray:
    """Load a transform from YAML and return T_ego_to_robot (4x4).

    Expected YAML formats (pick first matching):
    - Dict with keys: x, y, z, qx, qy, qz, qw
    - Dict with nested translation {x,y,z} and quaternion {x,y,z,w} or {qx,qy,qz,qw}
    - List of such dicts (use the first one that matches)
    """
    with open(os.path.expanduser(os.path.expandvars(filepath)), "r") as f:
        data = yaml.safe_load(f)

    def try_build(d):
        if not isinstance(d, dict):
            return None
        # flat keys
        flat_keys = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if all(k in d for k in flat_keys):
            return _quat_trans_to_matrix(d["qx"], d["qy"], d["qz"], d["qw"], d["x"], d["y"], d["z"])
        # nested forms
        t = d.get("translation") or d.get("t")
        q = d.get("quaternion") or d.get("q") or d.get("rotation")
        if isinstance(t, dict) and isinstance(q, dict):
            # Support both {x,y,z,w} and {qx,qy,qz,qw}
            if {"x", "y", "z", "w"}.issubset(q.keys()):
                return _quat_trans_to_matrix(q["x"], q["y"], q["z"], q["w"], t["x"], t["y"], t["z"])
            if {"qx", "qy", "qz", "qw"}.issubset(q.keys()):
                return _quat_trans_to_matrix(q["qx"], q["qy"], q["qz"], q["qw"], t["x"], t["y"], t["z"])
        return None

    # If dict at top level
    T = try_build(data)
    if T is not None:
        return T
    # If list, scan for first match
    if isinstance(data, list):
        for item in data:
            T = try_build(item)
            if T is not None:
                return T
    raise ValueError(f"Unsupported YAML format for transform: {filepath}")


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        print("Initalizing workspace")
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.cfg.root_dir = os.path.abspath(os.path.expanduser(self.cfg.root_dir))
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            if self.cfg.suite.use_object_points:
                import yaml

                cfg_path = f"{cfg.root_dir}/point_policy/cfgs/suite/points_cfg.yaml"
                with open(cfg_path) as stream:
                    try:
                        points_cfg = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                    root_dir, dift_path, cotracker_checkpoint = (
                        points_cfg["root_dir"],
                        points_cfg["dift_path"],
                        points_cfg["cotracker_checkpoint"],
                    )
                    points_cfg["dift_path"] = f"{root_dir}/{dift_path}"
                    points_cfg["cotracker_checkpoint"] = os.path.abspath(
                        os.path.expanduser(cotracker_checkpoint)
                    )
                self.cfg.suite.task_make_fn.points_cfg = points_cfg
        except:
            print("fail using object points")
            pass

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        # get average first position
        first_pos_average = []
        for action in self.expert_replay_loader.dataset.actions:
            first_pos = action[0, :3]
            first_pos_average.append(first_pos)
        self.first_pos_ego = np.mean(first_pos_average, axis=0)
        self.first_pos_robot = self.ego_to_robot(self.first_pos_ego)
        print(self.first_pos_ego)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def T_ego_to_robot(self):
        T = load_T_ego_to_robot_from_yaml(self.cfg.suite.T_ego_to_robot_yaml)
        return T

    @property
    def robot_wrist_to_eeff(self):
        return np.array([-0.02, 0.0, 0.02])  # put book absolute
        # return np.array([-0.02, -0.035, 0.035]) # sort fruit absolute
        # return np.array([-0.02, -0.035, 0.035]) # fold towel absolute
        # return np.array([-0.02, -0.03, 0.045]) #sweep board absolute

    def ego_to_robot(self, pos_in_ego: np.ndarray):
        pose_in_ego = np.eye(4)
        pose_in_ego[:3, 3] = pos_in_ego
        pose_in_robot = self.T_ego_to_robot @ pose_in_ego
        pos_in_robot = pose_in_robot[:3, 3] + self.robot_wrist_to_eeff
        return pos_in_robot

    def robot_to_ego(self, pos_in_robot: np.ndarray):
        pose_in_robot = np.eye(4)
        pose_in_robot[:3, 3] = pos_in_robot - self.robot_wrist_to_eeff
        pose_in_ego = np.linalg.inv(self.T_ego_to_robot) @ pose_in_robot
        pos_in_ego = pose_in_ego[:3, 3]
        return pos_in_ego

    def unproject(self, time_step, visualize=True):
        distorted_coords, depths = time_step.observation["point_tracks_pixels6"]
        rgb = time_step.observation["pixels6"]
        undistorted_coords = np.stack(
            [
                cv2.undistortPoints(
                    distorted_coord.reshape(1, 1, 2),
                    K[6],
                    D[6],
                    P=K[6],
                ).reshape(2)
                for distorted_coord in distorted_coords
            ],
            axis=0,
        )
        fx, fy = K[6][0, 0], K[6][1, 1]
        cx, cy = K[6][0, 2], K[6][1, 2]
        pixel_x, pixel_y = undistorted_coords.T
        x = (pixel_x - cx) * depths
        y = (pixel_y - cy) * depths
        x /= fx
        y /= fy
        object_keypoints = np.stack([x, y, depths], axis=-1)
        print(f"detected depths: {depths}")

        if visualize:
            rgb_vis = np.copy(rgb)
            for pt in distorted_coords:
                cv2.circle(
                    rgb_vis,
                    tuple(pt.astype(int)),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
            for pt in undistorted_coords:
                cv2.circle(
                    rgb_vis,
                    tuple(pt.astype(int)),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                )
            cv2.imwrite("projected_points_on_rgb.png", rgb_vis)

        return object_keypoints

    def plot_points_with_depth(self, frame, points, **kwargs):
        return plot_points(
            frame,
            np.eye(4),
            K[6],
            D[6],
            points=points,
            labels=[f"{z:.4f}" for z in points[:, 2]],
            radius=5,
            thickness=-1,
            **kwargs,
        )

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # 1. object keypoints
                # --------------------------------

                # unproject xy with depth
                object_keypoints = self.unproject(time_step)
                with open(
                    os.path.join(self.video_recorder.save_dir, "object_keypoints.json"),
                    "w",
                ) as f:
                    json.dump({"t": object_keypoints.tolist()}, f)
                a_rot = matrix_to_rotation_6d(
                    Rotation.from_quat(
                        time_step.observation["features"][3:7]
                    ).as_matrix()
                )
                a_pos_prev = np.array([0.4579441, 0.0321529, 0.56579893])
                a_grip_prev = -1

                # 2. eeff keypoints
                # --------------------------------

                # set the start position of the robot/policy
                n = 5
                self.first_pos_robot = a_pos_prev + np.array([0.25, 0, 0])
                for i in range(1, n + 1):
                    a_pos = a_pos_prev + i / n * (self.first_pos_robot - a_pos_prev)
                    action = np.concatenate([a_pos, a_rot, [a_grip_prev]])
                    time_step = self.env[env_idx].step(action)

                time.sleep(1)
                aloha_state = self.env[env_idx].get_state()
                a_pos_prev = np.array(aloha_state.pos)
                a_grip_prev = np.array(aloha_state.gripper)

                if not self.cfg.suite.history:

                    # 2. open loop
                    # --------------------------------

                    try:

                        frames = []
                        time_step.observation["point_tracks_pixels6"] = (
                            object_keypoints.reshape(1, -1)
                        )
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action = self.agent.act(
                                time_step.observation,
                                None,
                                step,
                                self.global_step,
                                eval_mode=True,
                            )
                            action = action.reshape(
                                self.cfg.num_queries, self.agent._act_dim
                            )

                            for a in action:
                                frame = cv2.cvtColor(
                                    np.copy(time_step.observation["pixels6"]),
                                    cv2.COLOR_BGR2RGB,
                                )
                                frame = self.plot_points_with_depth(
                                    frame,
                                    object_keypoints,
                                    color=(0, 153, 85),
                                )
                                frame = self.plot_points_with_depth(
                                    frame,
                                    a[:3].reshape(1, -1),
                                    color=(0, 135, 255),
                                )
                                frame = add_border(
                                    frame,
                                    text=f"{a[-1]:.4f}",
                                    color=(0, 255, 0) if a[-1] > 0 else (255, 0, 0),
                                )
                                frames.append(frame)

                                a_pos = self.ego_to_robot(np.copy(a[:3]))
                                a_grip = np.array((a[-1] > -0.2) * 2 - 1)
                                delta = a_pos - a_pos_prev
                                print(delta, a[-1])
                                n = 5
                                for i in range(1, n + 1):
                                    a_pos = a_pos_prev + i / n * delta
                                    a = np.concatenate(
                                        [
                                            a_pos,
                                            a_rot,
                                            [a_grip],
                                        ]
                                    )
                                    time_step = self.env[env_idx].step(a)
                                    self.video_recorder.record(self.env[env_idx])
                                    total_reward += time_step.reward
                                a_pos_prev = a_pos
                                a_grip_prev = a_grip

                    except:
                        pass

                else:

                    # 2. closed loop
                    # --------------------------------

                    # inference the model
                    try:
                        has_grasped = False
                        frames = []
                        while True:
                            # update the state here
                            robot_eeff = self.robot_to_ego(a_pos_prev)
                            action_keypoints = np.stack(
                                [
                                    robot_eeff,
                                    [a_grip_prev] * 3,
                                ],
                                axis=0,
                            )
                            time_step.observation["point_tracks_pixels6"] = (
                                np.concatenate(
                                    [
                                        object_keypoints,
                                        action_keypoints,
                                    ],
                                    axis=-2,
                                )
                            )

                            with torch.no_grad(), utils.eval_mode(self.agent):
                                action = self.agent.act(
                                    time_step.observation,
                                    None,
                                    step,
                                    self.global_step,
                                    eval_mode=True,
                                )

                            action = action.reshape(self.agent._act_dim)
                            delta_x=np.abs(action[0]-a_pos_prev[0])
                            delta_y=np.abs(action[1]-a_pos_prev[1])
                            delta_z=np.abs(action[2]-a_pos_prev[2])
                            
                            import ipdb; ipdb.set_trace()

                            if self.cfg.suite.action_type == "delta":
                                action[:3] = a_pos_prev + action[:3]

                            # annotated the camera view for sanity checking
                            frame = cv2.cvtColor(
                                np.copy(time_step.observation["pixels6"]),
                                cv2.COLOR_BGR2RGB,
                            )
                            frame = self.plot_points_with_depth(
                                frame,
                                object_keypoints,
                                color=(0, 153, 85),
                            )
                            frame = self.plot_points_with_depth(
                                frame,
                                action[:3].reshape(1, -1),
                                color=(0, 135, 255),
                            )
                            frame = add_border(
                                frame,
                                text=f"{action[-1]:.4f}",
                                color=(0, 255, 0) if action[-1] > 0 else (255, 0, 0),
                            )
                            frames.append(frame)

                            # ego->robot frame
                            a_pos = self.ego_to_robot(np.copy(action[:3]))
                            a_grip = np.array((action[-1] > 0) * 2 - 1)
                            print(step, a_pos - a_pos_prev, action[-1])
                            action = np.concatenate([a_pos, a_rot, a_grip.reshape(1)])
                            if not has_grasped and action[-1] == 1:
                                grasp_action = np.concatenate(
                                    [a_pos_prev, a_rot, np.array([1])]
                                )
                                for i in range(5):
                                    time_step = self.env[env_idx].step(grasp_action)
                                    time.sleep(0.1)
                                has_grasped = True
                            else:
                                time_step = self.env[env_idx].step(action)
                            self.video_recorder.record(self.env[env_idx])

                            # make sure it completes the grasp/ungrasp completely before moving
                            if not has_grasped and a_grip > 0:
                                time_step = self.env[env_idx].step(action)
                                time.sleep(1)
                                has_grasped = True

                            # check if the robot actually got there
                            time.sleep(0.05)
                            aloha_state_state = self.env[env_idx].get_state()
                            a_pos_prev = np.array(aloha_state_state.pos)
                            a_grip_prev = np.array(aloha_state_state.gripper)
                            print(np.linalg.norm(a_pos_prev - a_pos))

                            # just in case it didn't, assume that it did and use models' own predictions
                            # a_pos_prev = a_pos
                            # a_grip_prev = a_grip
                            total_reward += time_step.reward
                            step += 1

                    except KeyboardInterrupt:
                        pass

                episode += 1
                success.append(time_step.observation["goal_achieved"])

            save_video(
                str(
                    self.video_recorder.save_dir
                    / f"{self.global_frame}_anno{env_idx}.mp4"
                ),
                frames,
                fps=20,
            )
            self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    print("11111")
    workspace = Workspace(cfg)

    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    workspace.eval()


if __name__ == "__main__":
    main()
