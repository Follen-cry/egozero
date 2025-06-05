#!/usr/bin/env python3

import os
import sys
import time
import warnings

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from franka_env.envs.franka_env import (
    INTERNET_HOST,
    D,
    K,
    T_aruco_to_camera,
    T_robot_to_camera,
)
from frankateach.constants import CAM_PORT
from frankateach.network import ZMQCameraSubscriber
from logger import Logger
from PIL import Image
from replay_buffer import make_expert_replay_loader
from scipy.spatial.transform import Rotation
from video import VideoRecorder

import utils
from robot_utils.franka.utils import matrix_to_rotation_6d

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from transform_utils import average_poses
from vis_utils import detect_aruco

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


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


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # calibrate aruco-to-aria
        aria_cam_idx = 6
        aruco_calib_duration = 10  # seconds
        aria_subscriber = ZMQCameraSubscriber(
            host=INTERNET_HOST,  # Internet IP
            port=CAM_PORT + aria_cam_idx,
            topic_type="RGB",
        )
        T_aruco_to_aria = []
        start = time.time()
        print(f"calibrating Aria with Aruco tag for {aruco_calib_duration} seconds ...")
        while time.time() - start < aruco_calib_duration:
            rgb, _ = aria_subscriber.recv_rgb_image()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            aruco_pose = detect_aruco(rgb, K[aria_cam_idx], D[aria_cam_idx])
            if aruco_pose is not None:
                T_aruco_to_aria.append(aruco_pose)
            time.sleep(0.2)

        if len(T_aruco_to_aria) == 0:
            raise RuntimeError("No Aruco was detected in Aria frame!")
        self.T_aruco_to_aria = average_poses(np.stack(T_aruco_to_aria))
        Image.fromarray(rgb).save("first_frame.png")
        print(f"\033[92mcalibrated {self.T_aruco_to_aria=}\033[0m")

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = 400
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
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

        cam_ids = [3, 4]  # NOTE: this is hard-coded
        self.T_aruco_to_camera = None
        self.T_robot_to_camera = None
        for cam_idx in cam_ids:
            self.T_aruco_to_camera = T_aruco_to_camera.get(cam_idx, None)
            self.T_robot_to_camera = T_robot_to_camera.get(cam_idx, None)
        if self.T_aruco_to_camera is None or self.T_robot_to_camera is None:
            raise RuntimeError(
                f"T_aruco_to_camera and/or T_robot_to_camera is missing for cameras {cam_ids}"
            )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

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
                breakpoint()

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation,
                        self.stats,
                        step,
                        self.global_step,
                        eval_mode=True,
                    )
                    a_rot = matrix_to_rotation_6d(
                        Rotation.from_quat(
                            time_step.observation["features"][3:7]
                        ).as_matrix()
                    )
                    action = action.reshape(self.cfg.num_queries, self.agent._act_dim)
                    a_pos_prev = time_step.observation["features"][0:3]
                    has_moved = False
                    for a in action:
                        # aria frame
                        pos_in_aruco = np.eye(4)
                        pos_in_aruco[:3, 3] = np.copy(a[:3])
                        # camera frame
                        pos_in_camera = self.T_aruco_to_camera @ pos_in_aruco
                        # robot frame
                        pos_in_robot = (
                            np.linalg.inv(self.T_robot_to_camera) @ pos_in_camera
                        )

                        a_pos = pos_in_robot[:3, 3]
                        a_grip = np.array([(a[-1] > 0) * 2 - 1])
                        print(a_pos - a_pos_prev)
                        if not has_moved:
                            breakpoint()
                            has_moved = True
                        a_pos_prev = a_pos
                        a = np.concatenate([a_pos, a_rot, a_grip])

                        time_step = self.env[env_idx].step(a)
                        self.video_recorder.record(self.env[env_idx])
                        total_reward += time_step.reward
                        step += 1
                        time.sleep(0.5)

                episode += 1
                success.append(time_step.observation["goal_achieved"])
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
    from eval import WorkspaceIL as W

    workspace = W(cfg)

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
