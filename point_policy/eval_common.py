"""Shared utilities to expose online evaluation and array serialization.

This module provides helpers that can be reused by the evaluation server
and client to pack numpy arrays for transport and to run the object point
grounding pipeline outside of the dm_env wrappers.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from point_policy.point_utils.points_class import PointsClass
from point_policy.robot_utils.franka.utils import pixel2d_to_3d, pixelkey2camera


class ArrayCodec:
    """Encode/decode numpy arrays into JSON-serializable dictionaries."""

    @staticmethod
    def encode(array: np.ndarray) -> Dict[str, Any]:
        if not isinstance(array, np.ndarray):  # defensive programming for callers
            raise TypeError("ArrayCodec.encode expects a numpy array")
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }

    @staticmethod
    def decode(payload: Mapping[str, Any]) -> np.ndarray:
        required = {"shape", "dtype", "data"}
        if not required.issubset(payload.keys()):
            missing = ", ".join(sorted(required - set(payload.keys())))
            raise ValueError(f"Array payload missing required fields: {missing}")

        raw = base64.b64decode(payload["data"].encode("ascii"))
        array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
        return array.reshape(payload["shape"])


@dataclass
class GroundingResult:
    """Container for grounded object points."""

    points_3d: Dict[str, np.ndarray]
    points_2d: Dict[str, np.ndarray]


class PointGrounder:
    """Runs the object point grounding pipeline online using ``PointsClass``."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        points_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if points_cfg is None:
            points_cfg = self._load_points_cfg(cfg)
        else:
            # shallow copy to avoid mutating caller owned dicts
            points_cfg = dict(points_cfg)

        self.pixel_keys = list(cfg.suite.pixel_keys)
        self.point_dim = int(cfg.suite.point_dim)
        self.use_gt_depth = bool(cfg.suite.gt_depth)
        self.object_labels = list(points_cfg.get("object_labels", []))

        normalized_cfg = self._normalize_points_cfg(points_cfg)
        self.points_class = PointsClass(**normalized_cfg)

        calib_path = os.path.expanduser(cfg.suite.task_make_fn.calib_path)
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        self.calibration_data = np.load(calib_path, allow_pickle=True).item()

        self._initialized = False

    def reset(self) -> None:
        self.points_class.reset_episode()
        self._initialized = False

    def infer(
        self,
        frames_bgr: Mapping[str, np.ndarray],
        depths: Optional[Mapping[str, np.ndarray]] = None,
    ) -> GroundingResult:
        if not frames_bgr:
            raise ValueError("frames_bgr must contain at least one camera frame")

        for key in self.pixel_keys:
            if key not in frames_bgr:
                raise KeyError(f"Missing frame for pixel key '{key}'")

        for pixel_key, frame in frames_bgr.items():
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Frame for {pixel_key} must be HxWx3 BGR image, got {frame.shape}"
                )
            # PointsClass expects RGB input
            self.points_class.add_to_image_list(frame[:, :, ::-1], pixel_key)

        if not self._initialized:
            for pixel_key in self.pixel_keys:
                for object_label in self.object_labels:
                    self.points_class.find_semantic_similar_points(
                        pixel_key, object_label
                    )
            for pixel_key in self.pixel_keys:
                self.points_class.track_points(pixel_key, is_first_step=True)
            self._initialized = True

        for pixel_key in self.pixel_keys:
            self.points_class.track_points(pixel_key)

        points2d: Dict[str, np.ndarray] = {}
        for pixel_key in self.pixel_keys:
            pts = self.points_class.get_points_on_image(pixel_key)
            points2d[pixel_key] = pts.detach().cpu().numpy()[0]

        points3d: Dict[str, np.ndarray] = {}
        if self.point_dim == 3:
            if depths is None:
                raise ValueError("Depth maps are required for 3D point grounding")
            for pixel_key in self.pixel_keys:
                if pixel_key not in depths:
                    raise KeyError(f"Missing depth map for pixel key '{pixel_key}'")
                depth = np.asarray(depths[pixel_key])
                if depth.ndim != 2:
                    raise ValueError(
                        f"Depth map for {pixel_key} must be 2D array, got {depth.shape}"
                    )
                camera_name = pixelkey2camera[pixel_key]
                intr = self.calibration_data[camera_name]["int"]
                extr = self.calibration_data[camera_name]["ext"]
                pts = points2d[pixel_key]
                sampled_depths = []
                for x, y in pts:
                    xi = int(np.clip(round(x), 0, depth.shape[1] - 1))
                    yi = int(np.clip(round(y), 0, depth.shape[0] - 1))
                    raw_depth = depth[yi, xi]
                    sampled_depths.append(
                        float(raw_depth) / 1000.0
                        if depth.dtype != np.float32
                        else float(raw_depth)
                    )
                points3d[pixel_key] = pixel2d_to_3d(
                    pts,
                    np.asarray(sampled_depths),
                    intr,
                    extr,
                )
        else:
            points3d = {key: points2d[key] for key in self.pixel_keys}

        return GroundingResult(points_3d=points3d, points_2d=points2d)

    @staticmethod
    def _normalize_points_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = dict(config)
        for key in ["root_dir", "dift_path", "cotracker_checkpoint"]:
            if key in normalized:
                normalized[key] = os.path.abspath(os.path.expanduser(normalized[key]))
        return normalized

    @staticmethod
    def _load_points_cfg(cfg: DictConfig) -> MutableMapping[str, Any]:
        cfg_path = os.path.join(cfg.root_dir, "point_policy/cfgs/suite/points_cfg.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"points_cfg.yaml not found at {cfg_path}")
        loaded = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(loaded, resolve=True)

