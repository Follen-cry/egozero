"""Implements an IterableDataset for Aria data"""

import glob
import json
import os
import random
from abc import abstractmethod
from typing import Iterable, List, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from scipy.stats import median_abs_deviation, truncnorm
from torch.utils.data import IterableDataset


def break_long_segments(trajectory: np.ndarray, labels: np.ndarray, max_eps: float):
    """
    Splits segments in a 3D trajectory that are longer than max_eps into equally spaced segments.

    Args:
        trajectory (np.ndarray): (N, 3) array of 3D points.
        labels (np.ndarray): (N,) array of bool labels corresponding to each point.
        max_eps (float): Maximum allowable segment length.

    Returns:
        new_traj (np.ndarray): Resampled trajectory with all segment lengths <= max_eps.
        new_labels (np.ndarray): Resampled boolean labels for the new trajectory.
    """
    new_traj = []
    new_labels = []

    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        l1 = labels[i]

        seg = p2 - p1
        dist = np.linalg.norm(seg)
        n_steps = max(1, int(np.ceil(dist / max_eps)))

        for j in range(n_steps):
            alpha = j / n_steps
            point = (1 - alpha) * p1 + alpha * p2
            new_traj.append(point)
            new_labels.append(l1)

    # Include the last point and its label
    new_traj.append(trajectory[-1])
    new_labels.append(labels[-1])

    return np.array(new_traj), np.array(new_labels, dtype=bool)


def _remove_stationary_points(points, labels, min_eps=0.01):
    points = np.asarray(points)
    labels = np.asarray(labels).squeeze()

    # merge small segments (less than `eps` distance)
    new_points = [points[0]]
    new_labels = [labels[0]]
    last_point = points[0]
    for i in range(1, len(points)):
        # if the current segment is too small, merge it with the previous one
        if np.linalg.norm(points[i] - last_point) > min_eps:
            new_points.append(points[i])
            new_labels.append(labels[i])
            last_point = points[i]

    new_points = np.array(new_points)
    new_labels = np.array(new_labels)
    return new_points, new_labels


def remove_stationary_points(points, labels, min_eps=0.01, iters=5):
    for _ in range(iters):
        points, labels = _remove_stationary_points(points, labels, min_eps=min_eps)

    return points, labels


def keep_longest_true_segment(arr):
    arr = np.asarray(arr, dtype=bool)
    padded = np.r_[False, arr, False]
    diff = np.diff(padded.astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

    if len(starts) == 0:
        return np.zeros_like(arr, dtype=bool)

    lengths = ends - starts
    i = np.argmax(lengths)
    out = np.zeros_like(arr, dtype=bool)
    out[starts[i] : ends[i]] = True
    return out


def load_eeff_in_aruco_frame(demonstration_dir: str):
    """Load eeff in aruco frame"""
    t_index_to_aria = np.load(os.path.join(demonstration_dir, "index.npy"))
    T_index_to_aria = np.stack([np.eye(4) for _ in range(len(t_index_to_aria))])
    T_index_to_aria[:, :3, 3] = t_index_to_aria

    t_thumb_to_aria = np.load(os.path.join(demonstration_dir, "thumb.npy"))
    T_thumb_to_aria = np.stack([np.eye(4) for _ in range(len(t_thumb_to_aria))])
    T_thumb_to_aria[:, :3, 3] = t_thumb_to_aria

    T_aria_to_g = np.load(os.path.join(demonstration_dir, "pose.npy"))
    T_aruco_to_w = np.load(os.path.join(demonstration_dir, "first_frame_a2w.npy"))
    T_g_to_w = np.load(os.path.join(demonstration_dir, "first_frame_g2w.npy"))

    T_aria_to_aruco = np.einsum(
        "ij,jk,nkl->nil",
        np.linalg.inv(T_aruco_to_w),
        T_g_to_w,
        T_aria_to_g,
    )
    t_index_to_aruco = np.einsum("nij,njk->nik", T_aria_to_aruco, T_index_to_aria)[
        :, :3, 3
    ]
    t_thumb_to_aruco = np.einsum("nij,njk->nik", T_aria_to_aruco, T_thumb_to_aria)[
        :, :3, 3
    ]
    t_eeff_to_aruco = (t_index_to_aruco + t_thumb_to_aruco) / 2.0

    return t_eeff_to_aruco, t_index_to_aruco, t_thumb_to_aruco


def load_eeff_in_first_frame(demonstration_dir: str):
    """Load eeff in aria first frame (world frame)"""
    # eeff in aria frame
    t_index_to_aria = np.load(os.path.join(demonstration_dir, "index.npy"))
    t_thumb_to_aria = np.load(os.path.join(demonstration_dir, "thumb.npy"))
    t_eeff_to_aria = (t_index_to_aria + t_thumb_to_aria) / 2.0
    T_eeff_to_aria = np.stack([np.eye(4) for _ in range(len(t_eeff_to_aria))])
    T_eeff_to_aria[:, :3, 3] = t_eeff_to_aria

    T_aria_to_g = np.load(os.path.join(demonstration_dir, "pose.npy"))
    T_g_to_w = np.load(os.path.join(demonstration_dir, "first_frame_g2w.npy"))
    T_eeff_to_w = np.einsum("ij,njk,nkl->nil", T_g_to_w, T_aria_to_g, T_eeff_to_aria)
    t_eeff_to_w = T_eeff_to_w[:, :3, 3]

    return t_eeff_to_w


def plot_scalar_boxplots(data1, data2, save_path, labels=("Set 1", "Set 2")):
    plt.clf()
    plt.cla()

    data1 = np.array(data1)
    data2 = np.array(data2)
    B = data1.shape[0]

    fig, axes = plt.subplots(2, B, figsize=(6 * B, 8), sharey="row")

    # Ensure axes is always 2D
    if B == 1:
        axes = np.expand_dims(axes, axis=1)

    for b in range(B):
        for row, data, label in zip([0, 1], [data1[b], data2[b]], labels):
            ax = axes[row, b]
            ax.boxplot(data, vert=True, widths=0.3)
            ax.set_title(f"{b} - {label}")
            ax.set_xticks([])
            if b == 0:
                ax.set_ylabel("Value")

            stats = {
                "min": np.min(data),
                "max": np.max(data),
                "mean": np.mean(data),
                "median": np.median(data),
                "std": np.std(data),
            }

            for stat_name, stat_val in stats.items():
                x_offset = 1.2
                if stat_name == "mean":
                    x_offset = 1.4
                if stat_name == "median":
                    x_offset = 1.3
                ax.annotate(
                    f"{stat_name}:\n{stat_val:.5f}",
                    xy=(1, stat_val),
                    xytext=(x_offset, stat_val),
                    textcoords="data",
                    va="center",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.5),
                )

    plt.tight_layout()
    plt.savefig(save_path)


def plot_consecutive_distances_with_bools(
    points_and_bools: np.ndarray,
    save_path: str,
    anchors: List[np.ndarray] = None,
    reference_points: np.ndarray = None,
):
    """
    Plot distances between consecutive 3D points and the norm of each point over time.
    In the lower subplot, highlight continuous segments where the point norms are within
    `eps` of the anchor's norm.

    Args:
        points_and_bools (np.ndarray): Array of shape (n, 4) with x, y, z, and boolean.
        save_path (str): File path to save the plot.
        title (str): Title for the plot.
        anchor (np.ndarray): 3D anchor point.
        eps (float): Distance threshold in norm space.
    """
    points = points_and_bools[:, :3]
    if points_and_bools.shape[1] == 4:
        bools = points_and_bools[:, 3] > 0
    else:
        bools = np.ones(len(points)).astype(bool)

    # Compute distances between consecutive points
    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    bool_steps = bools[:-1]

    # Plot
    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top plot: step distances
    ax1.plot(
        range(len(distances)),
        distances,
        color="gray",
        alpha=0.6,
        label="Consecutive distance",
    )
    for i, (d, f) in enumerate(zip(distances, bool_steps)):
        ax1.plot(i, d, marker=".", color="darkorange" if f else "royalblue")
    ax1.axhline(
        np.mean(distances),
        color="red",
        linestyle="--",
        label="Mean consecutive distance",
    )
    ax1.set_ylabel("Distance between consecutive frames")
    ax1.grid(True, linestyle="--", alpha=0.5)
    orange_dot = mlines.Line2D(
        [], [], color="darkorange", marker=".", linestyle="None", label="Closed"
    )
    blue_dot = mlines.Line2D(
        [], [], color="royalblue", marker=".", linestyle="None", label="Open"
    )
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([orange_dot, blue_dot])
    ax1.legend(handles=handles)

    # Bottom plot: point norms
    for i in range(len(points)):
        d = np.linalg.norm(points[i] - points[anchors[0]])
        if max(anchors) >= i >= min(anchors):
            color = "green"
        else:
            color = "black"
        ax2.plot(
            i,
            d,
            marker=".",
            color=color,
            label="Grasp" if color == "green" and i == min(anchors) else "",
        )
    if reference_points is not None:
        for i, p in enumerate(reference_points):
            d = np.linalg.norm(points[anchors[0]] - p)
            ax2.axhline(
                d,
                color="red",
                linestyle="--",
                label="Distance from grasp to objects" if i == 0 else "",
            )
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Distance to grasp frame")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class Random3DAugmentation:

    def __init__(
        self,
        std: float = None,  # if none, uses a uniform distribution
        mean: List[float] = [0, 0, 0],
        lower: List[float] = [-1, -1, -1],
        upper: List[float] = [1, 1, 1],
    ):
        self.std = np.abs(std) if std is not None else None
        self.mean = np.array(mean)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        if not self.std:
            assert (self.lower < self.upper).all()
        else:
            assert ((self.lower < self.mean) * (self.mean < self.upper)).all()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  std={self.std},\n"
            f"  mean={self.mean.tolist()},\n"
            f"  lower={self.lower.tolist()},\n"
            f"  upper={self.upper.tolist()}\n"
            f")"
        )

    def _sample(self, size):
        # gaussian
        if self.std > 0:
            a, b = (self.lower - self.mean) / self.std, (
                self.upper - self.mean
            ) / self.std
            return truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=size)

        # uniform
        return np.random.uniform(low=self.lower, high=self.upper, size=size)

    @abstractmethod
    def __call__(self, states_and_actions):
        raise NotImplementedError


class RandomTranslation(Random3DAugmentation):
    """Applies a fixed 3D translation to all input keypoints"""

    def __call__(self, states_and_actions):
        states, actions = states_and_actions

        # states/actions: shape (b, 3n)
        d = self._sample(size=(3,)).reshape(1, 1, 3)

        def _translate(points):
            b = points.shape[0]
            points = points.reshape(b, -1, 3)  # (b, n, 3)
            mask = points[:, :, 2] == 0
            points = points + d  # (b, n, 3)
            points[mask] = 0  # zero out points that should be zero'd out
            points = points.reshape(b, -1)  # (b, 3n)
            return points

        states = _translate(states)
        actions = _translate(actions)
        return states, actions


class RandomRotation(Random3DAugmentation):
    """Applies a fixed 3D rotation to all input keypoints wrt states centroid"""

    def __call__(self, states_and_actions):
        states, actions = states_and_actions

        # states/actions: shape (b, 3n)
        d = self._sample(size=(3,))  # sample a random xyz from N(0, d)
        r = Rotation.from_euler(
            "xyz", d, degrees=True
        ).as_matrix()  # 3x3 rotation matrix

        def _rotate(points):
            b, n = points.shape[0], points.shape[1] // 3
            points = points.reshape(b, n, 3).reshape(b * n, 3)  # (b*n, 3)
            mask = points[:, 2] == 0
            points = points @ r.T
            points[mask] = 0  # zero out points that should be zero'd out
            points = points.reshape(b, n, 3).reshape(b, 3 * n)
            return points

        states = _rotate(states)
        actions = _rotate(actions)
        return states, actions


def get_relative_action(actions, action_after_steps):
    """
    Vectorized computation of relative 3D positions for a series of `n` points.
    `actions` is a (T, 3*n + 1) array:
        - First 3*n columns are 3D positions for n points.
        - Last column is the gripper state.
    """
    T, D = actions.shape
    n = (D - 1) // 3  # number of 3D points

    # Compute next indices with clipping at the last frame
    indices = np.arange(T)
    next_indices = np.clip(indices + action_after_steps, 0, T - 1)

    # Gather current and next positions
    pos_prev = actions[:, : 3 * n]  # (T, 3n)
    pos_next = actions[next_indices, : 3 * n]  # (T, 3n)
    pos_rel = pos_next - pos_prev  # (T, 3n)

    # Gripper from the "next" timestep
    gripper = actions[next_indices, -1:]  # (T, 1)

    # Concatenate relative positions and gripper value
    relative_actions = np.concatenate([pos_rel, gripper], axis=1)

    return relative_actions.astype(np.float32)


class BCDataset(IterableDataset):
    def __init__(
        self,
        path: Union[Iterable[str], str],
        pixel_keys: Iterable[str],
        history: bool,
        history_len: int,
        num_queries: int = 150,  # tune per task
        temporal_agg: bool = True,
        action_type: str = "absolute",
        subsample: int = 3,
        num_demos_per_task: int = 10000,
        # to control random 3d augmentations
        # - std=None to disable transforms
        # - std=0 to sample uniformly between [lower, upper]
        # - std>0 to sample from truncated normal N(mean, std) between [lower, upper]
        random_translation_std: float = 0,
        random_translation_mean: Union[float, List[float]] = [0.0, 0.0, 0.0],
        random_translation_lower: Union[float, List[float]] = [-0.5, -0.5, -0.5],
        random_translation_upper: Union[float, List[float]] = [0.5, 0.5, 0.5],
        random_rotation_std: float = 15,
        random_rotation_mean: Union[float, List[float]] = [0, 0, 0],
        random_rotation_lower: Union[float, List[float]] = [-30, -30, -30],
        random_rotation_upper: Union[float, List[float]] = [30, 30, 30],
        point_aug_prob: float = 0.1,
        history_aug_prob: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        assert num_demos_per_task > 0
        self._pixel_keys = pixel_keys
        self._history = history
        self._history_len = history_len if history else 1
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries
        self._action_type = action_type
        self._subsample = subsample
        self._point_aug_prob = point_aug_prob
        self._history_aug_prob = history_aug_prob
        assert action_type in ["absolute", "delta"]

        if isinstance(path, str):
            path = [path]
        preprocessed_data_dirs = sorted(path)

        num_actions = (
            (history_len + num_queries - 1)
            if temporal_agg
            else history_len if history else num_queries
        )
        min_demo_length = num_actions * subsample + 1

        # for open loop, we want the dataset to be in pairs of (image, trajectory)
        states, actions, anchors, grasp2dists, grasp2depths, dirs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i, preprocessed_data_dir in enumerate(preprocessed_data_dirs):
            preprocessed_data_dir = os.path.abspath(
                os.path.expanduser(preprocessed_data_dir)
            )
            demonstration_dirs = sorted(
                glob.glob(os.path.join(preprocessed_data_dir, "demonstration_*"))
            )
            demonstration_dirs = demonstration_dirs[:-1]
            for demonstration_dir in demonstration_dirs:
                if demonstration_dir.endswith("_00000"):
                    continue

                # load rgb image
                # image = Image.open(os.path.join(demonstration_dir, "first_frame.png"))

                # load the object points in first frame
                triangulation_json = os.path.join(
                    demonstration_dir, "triangulation.json"
                )
                if not os.path.exists(triangulation_json):
                    continue
                with open(triangulation_json, "r") as f:
                    state = np.array(json.load(f)["t*"])

                # load eeff trajectory in first frame
                t_eeff_to_w = load_eeff_in_first_frame(demonstration_dir)
                grasp = np.load(os.path.join(demonstration_dir, "grasp.npy")).astype(
                    bool
                )

                # remove spurious grasps (this can hurt gripper loss -> overall convergence)
                grasp = keep_longest_true_segment(grasp)
                t_eeff_to_w, grasp = break_long_segments(
                    t_eeff_to_w, grasp, max_eps=0.04
                )
                # resample so that the trajectory is uniform in space
                first_grasp_idx = np.nonzero(grasp.astype(int))[0][0].item()
                last_grasp_idx = np.nonzero(grasp.astype(int))[0][-1].item()
                t_eeff_to_w, grasp = remove_stationary_points(
                    t_eeff_to_w,
                    grasp,
                    min_eps=0.01,
                )
                first_grasp_idx = np.nonzero(grasp.astype(int))[0][0].item()
                last_grasp_idx = np.nonzero(grasp.astype(int))[0][-1].item()

                # filter out bad examples
                dists = np.linalg.norm(
                    t_eeff_to_w[first_grasp_idx, None] - state, axis=-1
                )

                # process action sequences for open/closed loop
                action = np.concatenate(
                    [t_eeff_to_w, 2 * grasp.reshape(-1, 1) - 1], axis=-1
                )

                # Compute distance of all frames *before* the first grasp frame to the first grasp position
                first_grasp_pos = t_eeff_to_w[first_grasp_idx]
                pre_positions = t_eeff_to_w[:first_grasp_idx]
                pre_grasp_dists = np.linalg.norm(
                    pre_positions - first_grasp_pos, axis=1
                )
                candidates = np.where(pre_grasp_dists >= 0.30)[0]

                if len(candidates) > 0:
                    start = candidates[-1]  # last frame satisfying the condition
                else:
                    start = max(
                        0, first_grasp_idx - 15
                    )  # fallback: 15 frames if no such point found

                n_post = min(15, len(action) - last_grasp_idx)
                end = last_grasp_idx + n_post

                if history:
                    action = action[start:end]
                    anchor = [first_grasp_idx - start, last_grasp_idx - start]
                else:
                    total = end - start
                    stride = total // num_queries
                    indices = np.linspace(start, end - 1, num=num_queries, dtype=int)
                    action = action[indices]
                    anchor = [
                        int(np.ceil((first_grasp_idx - start) / stride)),
                        int(np.ceil((last_grasp_idx - start) / stride)),
                    ]

                # throw out demonstrations that are too short
                if history and len(action) < min_demo_length:
                    continue

                states.append(state)
                actions.append(action)
                anchors.append(anchor)
                grasp2dists.append(dists)
                grasp2depths.append(
                    abs(t_eeff_to_w[first_grasp_idx, None, 2] - state[:, 2])
                )
                dirs.append(demonstration_dir)

        if len(actions) == 0:
            raise ValueError(
                "No Aria demonstrations were loaded. Check that "
                f"the dataset exists at {preprocessed_data_dirs} and contains valid preprocess/"
                "demonstration_* directories."
            )

        grasp2depths = np.array(grasp2depths)
        grasp2dists = np.array(grasp2dists)  # shape (num_examples, num_points)
        medians = np.median(grasp2dists, axis=0)
        mads = median_abs_deviation(grasp2dists, axis=0, scale="normal")
        thresholds = medians + 1 * mads  # shape (num_points,)
        outlier_mask = grasp2dists > thresholds  # shape (num_examples, num_points)
        num_outliers_per_demo = outlier_mask.sum(axis=1)
        outlier_demos = num_outliers_per_demo >= 2
        print(f"median distance to grasp per point: {medians}")

        for i in range(len(actions)):
            if outlier_demos[i]:
                print(f"skipping {dirs[i]}")
                continue

            if i % 10 > 0:
                continue

            plot_consecutive_distances_with_bools(
                actions[i],
                f"{i}_{os.path.basename(dirs[i])}.png",
                anchors=anchors[i],
                reference_points=states[i],
            )

        self.states = np.stack(states)[~outlier_demos]
        self.actions = [
            action
            for action, is_outlier in zip(actions, outlier_demos)
            if not is_outlier
        ]
        print(
            f"### found {outlier_demos.sum()} outliers, remaining {len(self.states)} training examples ###"
        )
        self.states = self.states[:num_demos_per_task]
        self.actions = self.actions[:num_demos_per_task]

        plot_scalar_boxplots(
            grasp2dists[~outlier_demos].T,
            grasp2depths[~outlier_demos].T,
            "triangulate_depth.png",
            labels=[
                "L2 distance",
                "Depth distance",
            ],
        )

        self.stats = {
            "past_tracks": {
                "min": 0,
                "max": 1,
            },
            "actions": {
                "min": 0,
                "max": 1,
            },
        }

        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
        }
        self._max_episode_len = 1000
        self._max_state_dim = self.states.shape[-2] * self.states.shape[-1]
        self.envs_till_idx = len(self.actions)

        # random 3d transformations
        self.random_transforms = []
        if random_rotation_std is not None:
            self.random_transforms.append(
                RandomRotation(
                    random_rotation_std,
                    random_rotation_mean,
                    random_rotation_lower,
                    random_rotation_upper,
                )
            )
        if random_translation_std is not None:
            self.random_transforms.append(
                RandomTranslation(
                    random_translation_std,
                    random_translation_mean,
                    random_translation_lower,
                    random_translation_upper,
                )
            )
        self.random_transforms = transforms.Compose(self.random_transforms)

    def _sample(self):
        idx = random.sample(range(len(self)), k=1)[0]
        states = np.copy(self.states[idx])
        actions = np.copy(self.actions[idx])

        # random rigid 3d transformations
        states, actions[:, :3] = self.random_transforms([states, actions[:, :3]])

        # subsample actions
        subsample_idx = np.random.randint(0, self._subsample)
        actions = actions[subsample_idx :: self._subsample]

        if self._history:
            # repeat actions history_len-1 times at the start to mimic inference time behavior
            actions = np.concatenate(
                [np.repeat(actions[[0]], self._history_len - 1, axis=0), actions],
                axis=0,
            )
            proprio = np.copy(actions)

            if self._action_type == "delta":
                actions = get_relative_action(
                    actions, 1
                )  # actions should already be subsampled at this point

            sample_idx = np.random.randint(0, len(actions) - self._history_len)

            # prepare action chunking + temporal aggregation
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[: min(len(actions), sample_idx + num_actions) - sample_idx] = (
                    actions[sample_idx : sample_idx + num_actions]
                )
                if len(actions) < sample_idx + num_actions:
                    act[len(actions) - sample_idx :] = (
                        actions[-1] if self._action_type == "absolute" else 0
                    )
                action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                action = action[:, 0]
            else:
                action = actions[sample_idx + 1 : sample_idx + self._history_len + 1]

        else:
            proprio = np.copy(actions)
            action = actions[None]

        # add noise to some points randomly
        selection_mask = np.random.rand(states.shape[0]) < self._point_aug_prob
        noise = np.random.randn(np.sum(selection_mask), 3) * 0.01
        states[selection_mask] += noise

        states = np.stack(
            [states for _ in range(self._history_len)], axis=0
        )  # (history_len, num_points, dim_points)

        if self._history:
            gripper = proprio[sample_idx : sample_idx + self._history_len, None, -1:]
            gripper = np.concatenate([gripper, gripper, gripper], axis=-1)
            proprio = proprio[
                sample_idx : sample_idx + self._history_len, None, :3
            ]  # (history_len, 1, dim_points)

            states = np.concatenate(
                [states, proprio, gripper], axis=1
            )  # (history_len, num_points+2, dim_points)

            # randomly repeat history to mimic start of trajectory (inference behavior)
            # NOTE: this hurts training convergence of gripper significantly, may need to restrict when this happens
            # if random.random() < self._history_aug_prob:
            #     # sample from zipfian distribution (so probability of repeating n frames is proportional to 1/n)
            #     ns = np.arange(1, self._history_len)
            #     weights = 1.0 / ns**2
            #     pmf = weights / weights.sum()
            #     n = np.random.choice(ns, p=pmf)
            #     states[:n] = states[n]  # replace the first n frames with the nth frame

        return {
            "past_tracks": torch.tensor(states),
            "actions": torch.tensor(action),
        }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return len(self.states)


if __name__ == "__main__":
    dataset = BCDataset(
        "/data/projectaria/mps/mps_pick-and-place-block-9-demos_vrs/preprocess"
    )
